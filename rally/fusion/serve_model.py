"""Guarded live adapter for the optional human-label-trained serve classifier.

The estimator is not trusted merely because a joblib file exists. Its artifact schema,
feature order, grouped held-out evaluation, and improvement over the rule baseline are
revalidated before any prediction can replace a rule decision.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..domain.observations import ServeSetupObservation
from ..learning.serve_schema import FEATURE_NAMES, assert_live_eligible
from ..learning.serve_features import audio_features, observation_features


def _observation_features(
    observation: ServeSetupObservation, onsets: np.ndarray, cfg,
) -> np.ndarray:
    values = observation_features(observation)
    values.update(audio_features(onsets, observation.first_strike, cfg.point_gap_s))
    vector = np.asarray([values[name] for name in FEATURE_NAMES], dtype=float)
    if not np.isfinite(vector).all():
        raise ValueError("serve classifier features contain non-finite values")
    return vector


def apply_eligible_serve_model(
    observations: Sequence[ServeSetupObservation], onsets: np.ndarray, cfg,
) -> tuple[list[ServeSetupObservation], dict[str, Any]]:
    """Return classifier-enriched observations, or unchanged observations when disabled."""
    path = getattr(cfg, "serve_model", None)
    if not path:
        return list(observations), {"status": "disabled"}
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"serve classifier not found: {source}")
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - joblib ships with sklearn
        raise RuntimeError("loading a serve classifier requires joblib") from exc
    artifact = joblib.load(source)
    if not isinstance(artifact, dict):
        raise ValueError("serve classifier artifact must be a mapping")
    assert_live_eligible(artifact)
    estimator = artifact.get("model")
    if estimator is None or not callable(getattr(estimator, "predict", None)):
        raise ValueError("serve classifier artifact has no usable estimator")
    matrix = np.stack([
        _observation_features(observation, onsets, cfg)
        for observation in observations
    ]) if observations else np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    prediction = np.asarray(estimator.predict(matrix), dtype=int) if len(matrix) else np.zeros(0)
    if prediction.shape != (len(observations),) or set(np.unique(prediction)) - {0, 1}:
        raise ValueError("serve classifier returned invalid predictions")
    scores: list[float | None] = [None] * len(observations)
    if callable(getattr(estimator, "predict_proba", None)) and len(matrix):
        probability = np.asarray(estimator.predict_proba(matrix), dtype=float)
        classes = list(getattr(estimator, "classes_", []))
        if probability.ndim == 2 and 1 in classes:
            column = classes.index(1)
            scores = [float(value) for value in probability[:, column]]
    enriched = [
        replace(
            observation,
            learned_serve_checked=True,
            learned_serve_evidence=bool(prediction[index]),
            learned_serve_score=scores[index],
        )
        for index, observation in enumerate(observations)
    ]
    return enriched, {
        "status": "used",
        "path": str(source),
        "dataset_sha256": (artifact.get("training") or {}).get("dataset_sha256"),
        "predicted_serves": int(np.sum(prediction)),
        "candidates": len(observations),
        "gate": "revalidated",
    }
