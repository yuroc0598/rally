import numpy as np
import pytest

from rally import pipeline
from rally.cli import _config_from_args, build_parser
from rally.config import (
    DEFAULT_RTMPOSE_MODEL,
    DEFAULT_YOLO_DETECTION_MODEL,
    RallyConfig,
)
from rally.signals import player, visual


def test_yolo12_config_defaults_and_environment_overrides(monkeypatch):
    monkeypatch.delenv("RALLY_YOLO_DETECTION_MODEL", raising=False)
    monkeypatch.delenv("RALLY_PLAYER_POSE_BACKEND", raising=False)
    monkeypatch.delenv("RALLY_PLAYER_POSE_MODEL", raising=False)
    monkeypatch.delenv("RALLY_YOLO_POSE_MODEL", raising=False)
    defaults = RallyConfig()
    assert defaults.player_detection_model == "yolo12n.pt"
    assert defaults.player_pose_backend == "rtmlib"
    assert defaults.player_pose_model == DEFAULT_RTMPOSE_MODEL
    assert defaults.player_detection_model == DEFAULT_YOLO_DETECTION_MODEL

    monkeypatch.setenv("RALLY_YOLO_DETECTION_MODEL", "models/custom-detect.pt")
    monkeypatch.setenv("RALLY_PLAYER_POSE_MODEL", "models/custom-pose.onnx")
    configured = RallyConfig()
    assert configured.player_detection_model == "models/custom-detect.pt"
    assert configured.player_pose_backend == "rtmlib"
    assert configured.player_pose_model == "models/custom-pose.onnx"

    monkeypatch.delenv("RALLY_PLAYER_POSE_MODEL")
    monkeypatch.setenv("RALLY_YOLO_POSE_MODEL", "models/legacy-pose.pt")
    legacy = RallyConfig()
    assert legacy.player_pose_backend == "yolo"
    assert legacy.player_pose_model == "models/legacy-pose.pt"

    explicit = RallyConfig(
        player_detection_model="explicit-detect.pt",
        player_pose_backend="yolo",
        player_pose_model="explicit-pose.pt",
    )
    assert explicit.player_detection_model == "explicit-detect.pt"
    assert explicit.player_pose_model == "explicit-pose.pt"


def test_yolo_model_config_rejects_empty_explicit_values():
    with pytest.raises(ValueError, match="player_detection_model"):
        RallyConfig(player_detection_model=" ")
    with pytest.raises(ValueError, match="player_pose_model"):
        RallyConfig(player_pose_model="")
    with pytest.raises(ValueError, match="player_pose_backend"):
        RallyConfig(player_pose_backend="unknown")


def test_cli_propagates_detection_and_pose_model_overrides():
    args = build_parser().parse_args([
        "input.mp4",
        "--player-detection-model", "detect.pt",
        "--player-pose-backend", "yolo",
        "--player-pose-model", "pose.pt",
    ])
    cfg = _config_from_args(args)
    assert cfg.player_detection_model == "detect.pt"
    assert cfg.player_pose_backend == "yolo"
    assert cfg.player_pose_model == "pose.pt"


def test_live_visual_and_pose_channels_receive_configured_models(monkeypatch):
    captured = {}

    class Detector:
        available = True
        device = "cpu"

        def __init__(self, model):
            captured["detection"] = model

    monkeypatch.setattr(player, "PlayerDetector", Detector)
    monkeypatch.setattr(visual, "opencv_available", lambda: True)
    monkeypatch.setattr(visual, "analyze_visual", lambda *args, **kwargs: {
        "motion": np.zeros(1), "camera_moving": np.zeros(1, dtype=bool),
        "geometry": np.zeros(1), "near_track": None, "player_samples": [],
        "frame_size": (100, 100), "target_court_filtered": False,
    })

    def fake_pose(_path, timeline, **kwargs):
        captured["pose"] = kwargs["model_name"]
        captured["pose_backend"] = kwargs["pose_backend"]
        return np.zeros(len(timeline)), np.zeros(len(timeline))

    monkeypatch.setattr(player, "pose_activity_track", fake_pose)
    cfg = RallyConfig(
        court_auto=False, player_pose=True,
        play_mode="casual",
        player_detection_model="detect-live.pt",
        player_pose_backend="yolo",
        player_pose_model="pose-live.pt",
    )
    ch = pipeline._Channels()
    pipeline._visual_channels(
        "unused.mp4", cfg, np.array([0.0]), True, ch, lambda _m: None)
    pipeline._pose_channel(
        "unused.mp4", np.array([0.0]), cfg, ch, lambda _m: None)

    assert captured == {
        "detection": "detect-live.pt", "pose": "pose-live.pt",
        "pose_backend": "yolo"}
    assert ch.stages["visual"]["detection_model"] == "detect-live.pt"
    assert ch.stages["pose"]["model"] == "pose-live.pt"


def test_serve_pose_loader_uses_configured_pose_model(monkeypatch):
    captured = []

    class Model:
        def __init__(self, name):
            captured.append(name)

        def to(self, _device):
            return self

    class Capture:
        def isOpened(self):
            return True

        def get(self, _property):
            return 30.0

        def release(self):
            pass

    monkeypatch.setattr("ultralytics.YOLO", Model)
    monkeypatch.setattr("cv2.VideoCapture", lambda _path: Capture())
    monkeypatch.setattr(player, "resolve_yolo_device", lambda: "cpu")
    cfg = RallyConfig(player_pose_backend="yolo", player_pose_model="serve-pose.pt")

    assert player.observe_serve_setups(
        "unused.mp4", [], np.zeros(0), cfg) == []
    assert captured == ["serve-pose.pt"]
