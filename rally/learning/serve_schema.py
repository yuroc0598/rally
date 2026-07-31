"""Stable serve-classifier schema and conservative live-deployment gate."""

from __future__ import annotations

from typing import Any

LABEL_SCHEMA = "rally.web_labels.v2"
CONTEXT_SCHEMA = "rally.serve_rule_context.v1"
DATASET_SCHEMA = "rally.serve_training_dataset.v1"
FEATURE_SCHEMA = "rally.serve_features.v2"
ARTIFACT_SCHEMA = "rally.serve_classifier.v1"

# Missing optional channels have explicit availability features. Numeric values therefore
# stay finite without conflating "not observed" with strong negative evidence.
FEATURE_NAMES = (
    "audio_available",
    "audio_gap_before_s",
    "audio_cluster_strikes",
    "audio_cluster_duration_s",
    "audio_interval_mean_s",
    "audio_interval_std_s",
    "pose_available",
    "pose_frame_fraction",
    "pose_ready_fraction",
    "pose_overhead_frames",
    "pose_overhead_max_ratio",
    "pose_side_confidence",
    "position_available",
    "position_score",
    "position_stable_fraction",
    "position_player_tracks",
    "position_server_span",
    "position_end_near",
    "position_end_far",
    "court_filtered",
    "ball_available",
    "ball_coverage",
    "ball_vertical_span",
    "ball_outgoing_span",
    "ball_ordered",
)


def live_gate(model: dict[str, float], rules: dict[str, float], *,
              matches: int, samples: int, positives: int, negatives: int,
              valid_folds: int, rule_coverage: float,
              min_matches: int = 3, min_samples: int = 30,
              min_balanced_accuracy: float = 0.70,
              min_rule_margin: float = 0.03) -> dict[str, Any]:
    """Require grouped evidence that the classifier materially improves on rules."""
    reasons: list[str] = []
    if matches < min_matches:
        reasons.append(f"needs at least {min_matches} held-out matches")
    if samples < min_samples:
        reasons.append(f"needs at least {min_samples} labeled samples")
    if min(positives, negatives) < 10:
        reasons.append("needs at least 10 samples from each class")
    if valid_folds != matches:
        reasons.append("not every match produced a valid held-out fold")
    if rule_coverage < 0.80:
        reasons.append("rule baseline coverage is below 80%")
    model_bal = float(model.get("balanced_accuracy", 0.0))
    rule_bal = float(rules.get("balanced_accuracy", 0.0))
    if model_bal < min_balanced_accuracy:
        reasons.append(f"model balanced accuracy is below {min_balanced_accuracy:.0%}")
    if model_bal < rule_bal + min_rule_margin:
        reasons.append(
            f"model does not beat rules by {min_rule_margin:.0%} balanced accuracy")
    if float(model.get("precision", 0.0)) + 1e-12 < float(rules.get("precision", 0.0)):
        reasons.append("model precision is below the rule baseline")
    return {
        "allowed": not reasons,
        "status": "passed" if not reasons else "blocked",
        "reasons": reasons,
        "thresholds": {
            "min_matches": min_matches,
            "min_samples": min_samples,
            "min_class_samples": 10,
            "min_rule_coverage": 0.80,
            "min_balanced_accuracy": min_balanced_accuracy,
            "min_rule_margin": min_rule_margin,
            "precision_not_below_rules": True,
        },
    }


def assert_live_eligible(artifact: dict[str, Any]) -> None:
    """Recompute the gate; never trust a serialized ``allowed`` flag by itself."""
    if artifact.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("unrecognized serve model artifact schema")
    if artifact.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("serve model feature schema is incompatible")
    if artifact.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("serve model feature order is incompatible")
    training = artifact.get("training") or {}
    if training.get("stable_context_alignment") is not True:
        raise ValueError("serve model training used heuristic label/context alignment")
    evaluation = artifact.get("evaluation") or {}
    folds = evaluation.get("folds") or []
    try:
        recomputed = live_gate(
            evaluation["model"], evaluation["rules"],
            matches=int(training["matches"]), samples=int(training["samples"]),
            positives=int(training["positive"]), negatives=int(training["negative"]),
            valid_folds=sum(fold.get("status") == "used" for fold in folds),
            rule_coverage=float(evaluation["rule_coverage"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("serve model artifact lacks complete evaluation metadata") from exc
    gate = artifact.get("live_gate") or {}
    if gate.get("thresholds") != recomputed["thresholds"]:
        raise ValueError("serve model artifact was evaluated under incompatible gate thresholds")
    if (not recomputed["allowed"] or gate.get("allowed") is not True
            or gate.get("status") != "passed"):
        raise ValueError("serve model artifact has not passed the live evaluation gate")
