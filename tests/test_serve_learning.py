import json

import numpy as np
import pytest

from rally.tools import serve_dataset, serve_learning, serve_train


def _export(path, *, first_label="yes", second_label="no"):
    data = {
        "schema_version": serve_learning.LABEL_SCHEMA,
        "job_id": path.stem,
        "filename": f"{path.stem}.mp4",
        "tasks": [
            {"id": "serve_0000", "kind": "serve_motion", "time_s": 10.0},
            {"id": "serve_0001", "kind": "serve_motion", "time_s": 30.0},
            {"id": "player_0000", "kind": "player_identity", "time_s": 10.0},
        ],
        "labels": {
            "serve_0000": {"kind": "serve_motion", "values": {
                "is_serve": first_label, "end": "near"}},
            "serve_0001": {"kind": "serve_motion", "values": {
                "is_serve": second_label}},
            "player_0000": {"kind": "player_identity", "values": {"player": "P1"}},
        },
        "feature_context": {
            "schema_version": serve_learning.CONTEXT_SCHEMA,
            "strike_times": [11.0, 31.0],
            "segments": [],
            "match_state": {
                "logical_groups": [
                    {"output": [10.0, 20.0], "serve_member_index": 0},
                    {"output": [30.0, 40.0], "serve_member_index": 1},
                ],
                "observations": [
                    {
                        "point": [10.0, 20.0], "first_strike": 11.0,
                        "observable": True, "sampled_frames": 10, "pose_frames": 8,
                        "ready_frames": 5, "overhead_frames": 2,
                        "overhead_max_ratio": 0.8, "side_confidence": 0.9,
                        "position_checked": True, "position_score": 0.8,
                        "position_stable_fraction": 0.75, "position_player_tracks": 4,
                        "position_server_span": 0.05, "position_server_end": "near",
                        "target_court_filtered": True, "ball_checked": True,
                        "ball_coverage": 0.7, "ball_vertical_span": 0.2,
                        "ball_outgoing_span": 0.15, "ball_ordered_evidence": True,
                        "serve_evidence_sources": ["tracknet_ball_motion"],
                    },
                    {
                        "point": [30.0, 40.0], "first_strike": 31.0,
                        "observable": True, "sampled_frames": 8, "pose_frames": 4,
                        "ready_frames": 1, "position_checked": True,
                        "ball_checked": True, "serve_evidence_sources": [],
                    },
                ],
            },
        },
    }
    path.write_text(json.dumps(data))
    return data


def _fake_audio(_video, rows, _context):
    return {
        row["task_id"]: {
            "audio_available": 1.0,
            "audio_gap_before_s": 4.0,
            "audio_cluster_strikes": 3.0,
        }
        for row in rows
    }


def test_web_export_adapter_joins_human_labels_and_rich_context(tmp_path):
    labels = tmp_path / "match-a.json"
    _export(labels)
    export = serve_learning.load_web_label_export(labels)
    rows = serve_learning.labelled_serve_tasks(export)
    context = serve_learning.context_features(export, rows)

    assert [(row["task_id"], row["label"]) for row in rows] == [
        ("serve_0000", 1), ("serve_0001", 0)]
    first = context["serve_0000"]
    assert first["contact_time_s"] == 11.0
    assert first["rule_prediction"] == 1
    assert first["features"]["pose_frame_fraction"] == 0.8
    assert first["features"]["position_end_near"] == 1.0
    assert first["features"]["ball_ordered"] == 1.0


def test_stable_task_context_ids_override_ambiguous_times_and_are_validated(tmp_path):
    labels = tmp_path / "match-stable.json"
    data = _export(labels)
    data["tasks"][0].update({
        "source_segment_index": 0,
        "logical_group": 7,
        "match_state_observation_index": 1,
    })
    data["feature_context"]["match_state"]["logical_groups"][1].update({
        "group_index": 7, "member_indices": [1],
    })
    labels.write_text(json.dumps(data))
    export = serve_learning.load_web_label_export(labels)
    rows = serve_learning.labelled_serve_tasks(export)

    context = serve_learning.context_features(export, rows)
    assert context["serve_0000"]["contact_time_s"] == 31.0

    rows[0]["match_state_observation_index"] = 0
    with pytest.raises(ValueError, match="mismatched stable context"):
        serve_learning.context_features(export, rows)


def test_adapter_accepts_raw_labels_json_with_sibling_tasks(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "tasks.json").write_text(json.dumps([
        {"id": "serve_0000", "kind": "serve_motion", "time_s": 4.5},
    ]))
    (labels_dir / "labels.json").write_text(json.dumps({
        "serve_0000": {"kind": "serve_motion", "values": {"is_serve": "no"}},
    }))

    export = serve_learning.load_web_label_export(labels_dir / "labels.json")

    assert serve_learning.labelled_serve_tasks(export)[0]["label"] == 0


def test_build_dataset_preserves_match_groups_and_finite_schema(tmp_path):
    specs = []
    for match_id in ("a", "b", "c"):
        labels = tmp_path / f"{match_id}.json"
        _export(labels)
        specs.append(serve_learning.WebJobSpec(match_id, f"{match_id}.mp4", str(labels)))

    dataset = serve_learning.build_training_dataset(specs, audio_extractor=_fake_audio)

    assert dataset["schema_version"] == serve_learning.DATASET_SCHEMA
    assert dataset["feature_schema"] == serve_learning.FEATURE_SCHEMA
    assert {sample["match_id"] for sample in dataset["samples"]} == {"a", "b", "c"}
    assert len(dataset["samples"]) == 6
    assert all(list(sample["features"]) == list(serve_learning.FEATURE_NAMES)
               for sample in dataset["samples"])
    assert all(np.isfinite(list(sample["features"].values())).all()
               for sample in dataset["samples"])


def test_held_out_splits_never_mix_match_samples():
    groups = np.array(["a", "a", "b", "b", "c"])
    splits = serve_learning.held_out_match_splits(groups)
    assert len(splits) == 3
    for train, test in splits:
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(set(groups[test])) == 1


class _ThresholdEstimator:
    def fit(self, X, _y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.feature_importances_[0] = 1.0
        return self

    def predict(self, X):
        return (X[:, 0] >= 0.5).astype(int)


def test_training_evaluation_is_grouped_and_artifact_stays_blocked_when_small():
    samples = []
    for match_id in ("a", "b", "c"):
        for label in (0, 0, 1, 1):
            features = {name: 0.0 for name in serve_learning.FEATURE_NAMES}
            features["audio_available"] = float(label)
            samples.append({
                "match_id": match_id, "task_id": f"{match_id}-{len(samples)}",
                "label": label, "features": features,
                "rule_prediction": 0, "rule_available": True,
            })
    dataset = {
        "schema_version": serve_learning.DATASET_SCHEMA,
        "feature_schema": serve_learning.FEATURE_SCHEMA,
        "feature_names": list(serve_learning.FEATURE_NAMES),
        "samples": samples,
    }

    artifact = serve_train.evaluate_dataset(
        dataset, estimator_factory=_ThresholdEstimator)

    assert artifact["evaluation"]["scheme"] == "leave-one-match-out"
    assert {fold["held_out_match"] for fold in artifact["evaluation"]["folds"]} == {
        "a", "b", "c"}
    assert artifact["evaluation"]["model"]["balanced_accuracy"] == 1.0
    assert artifact["live_gate"]["allowed"] is False
    assert artifact["live_loading"]["enabled"] is False
    with pytest.raises(ValueError, match="heuristic|not passed"):
        serve_learning.assert_live_eligible(artifact)


def test_live_gate_requires_and_records_a_real_rules_win():
    passed = serve_learning.live_gate(
        {"balanced_accuracy": 0.80, "precision": 0.90},
        {"balanced_accuracy": 0.72, "precision": 0.85},
        matches=4, samples=50, positives=25, negatives=25,
        valid_folds=4, rule_coverage=1.0,
    )
    blocked = serve_learning.live_gate(
        {"balanced_accuracy": 0.73, "precision": 0.90},
        {"balanced_accuracy": 0.72, "precision": 0.85},
        matches=4, samples=50, positives=25, negatives=25,
        valid_folds=4, rule_coverage=1.0,
    )
    assert passed["allowed"] is True
    assert blocked["allowed"] is False
    artifact = {
        "artifact_schema": serve_learning.ARTIFACT_SCHEMA,
        "feature_schema": serve_learning.FEATURE_SCHEMA,
        "feature_names": list(serve_learning.FEATURE_NAMES),
        "training": {
            "matches": 4, "samples": 50, "positive": 25, "negative": 25,
            "stable_context_alignment": True,
        },
        "evaluation": {
            "model": {"balanced_accuracy": 0.80, "precision": 0.90},
            "rules": {"balanced_accuracy": 0.72, "precision": 0.85},
            "rule_coverage": 1.0,
            "folds": [{"status": "used"} for _ in range(4)],
        },
        "live_gate": passed,
    }
    serve_learning.assert_live_eligible(artifact)

    artifact["evaluation"]["model"]["balanced_accuracy"] = 0.72
    with pytest.raises(ValueError, match="not passed"):
        serve_learning.assert_live_eligible(artifact)


def test_from_web_cli_adapter_writes_versioned_dataset(monkeypatch, tmp_path):
    dataset = {
        "schema_version": serve_learning.DATASET_SCHEMA,
        "matches": [{"match_id": "a"}],
        "samples": [{"label": 1}],
    }
    monkeypatch.setattr(serve_learning, "build_training_dataset", lambda _specs: dataset)
    output = tmp_path / "dataset.json"

    assert serve_dataset.main([
        "from-web", "--job", "a", "a.mp4", "a-labels.json", "--out", str(output),
    ]) == 0
    assert json.loads(output.read_text()) == dataset
