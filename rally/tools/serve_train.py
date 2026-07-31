"""Train a serve classifier with held-out-*match* validation.

The input is the versioned multi-match JSON produced by::

    python -m rally.tools.serve_dataset from-web \
      --job match-a a.mp4 a-labels.json \
      --job match-b b.mp4 b-labels.json \
      --job match-c c.mp4 c-labels.json --out serve-training.json

Samples from one match are never split between training and validation. Artifacts are
saved with a conservative live-eligibility gate; the live pipeline revalidates that gate
before loading an explicitly configured artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .serve_learning import (
    ARTIFACT_SCHEMA,
    DATASET_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    dataset_fingerprint,
    held_out_match_splits,
    live_gate,
)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    if y.shape != prediction.shape or y.size == 0:
        raise ValueError("metrics need equally sized, non-empty arrays")
    tp = int(np.sum((prediction == 1) & (y == 1)))
    fp = int(np.sum((prediction == 1) & (y == 0)))
    fn = int(np.sum((prediction == 0) & (y == 1)))
    tn = int(np.sum((prediction == 0) & (y == 0)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float((tp + tn) / y.size),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def load_dataset(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        dataset = json.load(handle)
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {dataset.get('schema_version')!r}")
    if dataset.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(f"unsupported feature schema: {dataset.get('feature_schema')!r}")
    if dataset.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("dataset feature order does not match this trainer")
    if not isinstance(dataset.get("samples"), list) or not dataset["samples"]:
        raise ValueError("dataset has no samples")
    return dataset


def _arrays(dataset: dict[str, Any]):
    samples = dataset["samples"]
    X = np.asarray([
        [float(sample["features"][name]) for name in FEATURE_NAMES]
        for sample in samples
    ], dtype=float)
    y = np.asarray([int(sample["label"]) for sample in samples], dtype=int)
    groups = np.asarray([str(sample["match_id"]) for sample in samples], dtype=object)
    rules = np.asarray([int(sample.get("rule_prediction", 0)) for sample in samples], dtype=int)
    rule_available = np.asarray(
        [bool(sample.get("rule_available", False)) for sample in samples], dtype=bool)
    if not np.isfinite(X).all():
        raise ValueError("dataset contains non-finite features")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("serve labels must be binary")
    if np.unique(y).size != 2:
        raise ValueError("training needs both serve and non-serve labels")
    return X, y, groups, rules, rule_available


def _default_estimator():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional training extra
        raise RuntimeError(
            "serve training needs scikit-learn; install the 'training' optional extra"
        ) from exc
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )


EstimatorFactory = Callable[[], Any]


def evaluate_dataset(dataset: dict[str, Any], *, model_out: str | None = None,
                     estimator_factory: EstimatorFactory = _default_estimator
                     ) -> dict[str, Any]:
    X, y, groups, rule_prediction, rule_available = _arrays(dataset)
    splits = held_out_match_splits(groups)
    predictions = np.full(y.shape, -1, dtype=int)
    fold_records: list[dict[str, Any]] = []
    valid_folds = 0
    for train, test in splits:
        held_out = str(groups[test[0]])
        record = {
            "held_out_match": held_out,
            "train_samples": int(train.size),
            "test_samples": int(test.size),
        }
        if np.unique(y[train]).size != 2:
            record.update(status="invalid", reason="training side has only one class")
            fold_records.append(record)
            continue
        estimator = estimator_factory()
        estimator.fit(X[train], y[train])
        predictions[test] = np.asarray(estimator.predict(X[test]), dtype=int)
        record["status"] = "used"
        fold_records.append(record)
        valid_folds += 1

    predicted = predictions >= 0
    if not np.any(predicted):
        raise ValueError("no held-out match fold had both classes on its training side")
    model_metrics = _metrics(y[predicted], predictions[predicted])
    rule_metrics = _metrics(y, rule_prediction)
    unique_matches = list(dict.fromkeys(str(group) for group in groups))
    positives = int(y.sum())
    negatives = int(y.size - positives)
    gate = live_gate(
        model_metrics, rule_metrics,
        matches=len(unique_matches), samples=int(y.size),
        positives=positives, negatives=negatives,
        valid_folds=valid_folds,
        rule_coverage=float(rule_available.mean()),
    )
    stable_context_alignment = bool(dataset.get("stable_context_alignment", False))
    if not stable_context_alignment:
        gate = {
            **gate,
            "allowed": False,
            "status": "blocked",
            "reasons": [
                *gate.get("reasons", []),
                "all samples need stable observation/group ids for live deployment",
            ],
        }

    full_model = estimator_factory()
    full_model.fit(X, y)
    importances = getattr(full_model, "feature_importances_", None)
    importance_records = [] if importances is None else [
        {"feature": name, "importance": round(float(importance), 6)}
        for name, importance in sorted(
            zip(FEATURE_NAMES, importances), key=lambda item: -float(item[1]))
    ]
    artifact = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "estimator": {
            "module": type(full_model).__module__,
            "class": type(full_model).__name__,
        },
        "training": {
            "dataset_schema": dataset["schema_version"],
            "dataset_sha256": dataset_fingerprint(dataset),
            "matches": len(unique_matches),
            "samples": int(y.size),
            "positive": positives,
            "negative": negatives,
            "stable_context_alignment": stable_context_alignment,
        },
        "evaluation": {
            "scheme": "leave-one-match-out",
            "group_key": "match_id",
            "folds": fold_records,
            "model": model_metrics,
            "rules": rule_metrics,
            "rule_coverage": float(rule_available.mean()),
            "feature_importance": importance_records,
        },
        "live_gate": gate,
        "live_loading": {
            "enabled": bool(gate["allowed"]),
            "eligible": bool(gate["allowed"]),
            "note": ("live loader must independently revalidate this gate before use"),
        },
        "model": full_model,
    }
    if model_out:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - part of sklearn installations
            raise RuntimeError("saving a serve artifact needs joblib") from exc
        destination = Path(model_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination)
    return artifact


def _print_report(artifact: dict[str, Any]) -> None:
    training = artifact["training"]
    evaluation = artifact["evaluation"]
    model = evaluation["model"]
    rules = evaluation["rules"]
    print(
        f"validation: leave-one-match-out across {training['matches']} matches; "
        f"{training['samples']} samples "
        f"({training['positive']} serve, {training['negative']} non-serve)"
    )
    print(
        f"model balanced accuracy={model['balanced_accuracy']:.1%}, "
        f"precision={model['precision']:.1%}, recall={model['recall']:.1%}"
    )
    print(
        f"rules balanced accuracy={rules['balanced_accuracy']:.1%}, "
        f"precision={rules['precision']:.1%}, recall={rules['recall']:.1%}; "
        f"coverage={evaluation['rule_coverage']:.1%}"
    )
    gate = artifact["live_gate"]
    print(f"live gate: {gate['status']}")
    for reason in gate["reasons"]:
        print(f"  - {reason}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rally.tools.serve_train")
    parser.add_argument("dataset", help="JSON from 'serve_dataset from-web'")
    parser.add_argument("--model-out", default=None,
                        help="write a guarded offline joblib artifact")
    args = parser.parse_args(argv)
    artifact = evaluate_dataset(load_dataset(args.dataset), model_out=args.model_out)
    _print_report(artifact)
    if args.model_out:
        print(f"saved guarded offline artifact -> {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
