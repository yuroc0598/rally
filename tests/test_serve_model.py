import numpy as np
import pytest
import sys
from types import SimpleNamespace

from rally.config import RallyConfig
from rally.fusion.serve_model import apply_eligible_serve_model
from rally.signals.player import ServeSetupObservation
from rally.tools import serve_learning


class _AudioEstimator:
    classes_ = np.array([0, 1])

    def predict(self, values):
        return (values[:, 0] > 0.5).astype(int)

    def predict_proba(self, values):
        positive = (values[:, 0] > 0.5).astype(float) * 0.8 + 0.1
        return np.column_stack([1.0 - positive, positive])


def _artifact(model=None):
    gate = serve_learning.live_gate(
        {"balanced_accuracy": 0.82, "precision": 0.91},
        {"balanced_accuracy": 0.72, "precision": 0.86},
        matches=4, samples=50, positives=25, negatives=25,
        valid_folds=4, rule_coverage=1.0,
    )
    return {
        "artifact_schema": serve_learning.ARTIFACT_SCHEMA,
        "feature_schema": serve_learning.FEATURE_SCHEMA,
        "feature_names": list(serve_learning.FEATURE_NAMES),
        "training": {
            "matches": 4, "samples": 50, "positive": 25, "negative": 25,
            "dataset_sha256": "test-dataset",
            "stable_context_alignment": True,
        },
        "evaluation": {
            "model": {"balanced_accuracy": 0.82, "precision": 0.91},
            "rules": {"balanced_accuracy": 0.72, "precision": 0.86},
            "rule_coverage": 1.0,
            "folds": [{"status": "used"} for _ in range(4)],
        },
        "live_gate": gate,
        "model": model or _AudioEstimator(),
    }


def _observation():
    return ServeSetupObservation(
        point=(10.0, 14.0), first_strike=10.2, side="left", side_confidence=0.9,
        near_x=0.3, near_x_std=0.01, sampled_frames=10, pose_frames=8,
        ready_frames=4, serve_motion=False, setup_evidence=False, observable=True,
        target_court_filtered=True,
    )


def test_eligible_classifier_is_revalidated_and_applied(monkeypatch, tmp_path):
    model_path = tmp_path / "serve.joblib"
    model_path.touch()
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda _path: _artifact()))
    cfg = RallyConfig(serve_model=str(model_path))

    observations, stage = apply_eligible_serve_model(
        [_observation()], np.array([10.2, 10.9]), cfg)

    assert stage["status"] == "used"
    assert observations[0].learned_serve_checked is True
    assert observations[0].learned_serve_evidence is True
    assert observations[0].learned_serve_score == pytest.approx(0.9)


def test_classifier_with_failed_held_out_gate_cannot_load(monkeypatch, tmp_path):
    model_path = tmp_path / "serve.joblib"
    model_path.touch()
    artifact = _artifact()
    artifact["evaluation"]["model"]["balanced_accuracy"] = 0.70
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda _path: artifact))

    with pytest.raises(ValueError, match="not passed"):
        apply_eligible_serve_model(
            [_observation()], np.array([10.2]), RallyConfig(serve_model=str(model_path)))
