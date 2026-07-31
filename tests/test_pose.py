from types import SimpleNamespace

import numpy as np
import pytest

from rally.config import DEFAULT_RTMPOSE_MODEL, RallyConfig
from rally.signals.player import _body_pose_features
from rally.signals.pose import (
    CroppedRTMPose,
    discover_rtmpose_weights,
    resolve_rtmpose_device,
)


def test_rtmpose_url_prefers_matching_local_onnx(tmp_path):
    local = tmp_path / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"
    local.write_bytes(b"onnx")

    assert discover_rtmpose_weights(
        DEFAULT_RTMPOSE_MODEL, models_dir=str(tmp_path)) == str(local)


def test_rtmpose_device_uses_installed_execution_providers():
    assert resolve_rtmpose_device("auto", "onnxruntime") == "cpu"
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        resolve_rtmpose_device("cuda", "onnxruntime")


def test_cropped_rtmpose_filters_neighbor_court_before_pose():
    class Boxes:
        def __init__(self):
            self.xyxy = np.array([
                [10.0, 10.0, 30.0, 80.0],
                [180.0, 10.0, 220.0, 80.0],
            ])

        def __len__(self):
            return 2

    class Detector:
        def predict(self, frames, **_kwargs):
            return [SimpleNamespace(boxes=Boxes()) for _frame in frames]

    class Estimator:
        def __init__(self):
            self.boxes = None

        def __call__(self, _frame, *, bboxes):
            self.boxes = bboxes
            count = len(bboxes)
            return np.zeros((count, 17, 2)), np.ones((count, 17))

    class Court:
        def to_court(self, feet):
            feet = np.asarray(feet)
            # The right-hand detection maps beyond the target-court sideline.
            return np.column_stack((feet[:, 0] / 10.0, np.full(len(feet), 5.0)))

    estimator = Estimator()
    backend = CroppedRTMPose(
        detector=Detector(), estimator=estimator,
        detection_device="cpu", pose_device="cpu")
    result = backend.predict(
        [np.zeros((100, 240, 3), np.uint8)], court=Court())[0]

    assert result.boxes.shape == (1, 4)
    assert result.keypoints.shape == (1, 17, 2)
    assert estimator.boxes == [[10.0, 10.0, 30.0, 80.0]]


def test_pose_features_recognize_overhead_wrist_and_ready_stance():
    pose = np.zeros((17, 2), dtype=float)
    confidence = np.ones(17, dtype=float)
    pose[5], pose[6] = (40, 40), (60, 40)
    pose[11], pose[12] = (42, 80), (58, 80)
    pose[9], pose[10] = (40, 15), (60, 20)
    pose[13], pose[14] = (38, 105), (62, 105)
    pose[15], pose[16] = (25, 120), (75, 120)

    usable, ready, overhead_ratio = _body_pose_features(
        pose, confidence, RallyConfig())

    assert usable is True
    assert ready is True
    assert overhead_ratio > 0.35
