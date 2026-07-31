import numpy as np
import pytest

from rally.config import RallyConfig
from rally.signals.court import Court
from rally.signals.visual import (
    analyze_visual,
    masked_frame_diff_energy,
    target_court_mask,
)

cv2 = pytest.importorskip("cv2")


def _court():
    return Court.from_image_corners((20, 90), (80, 90), (65, 20), (35, 20))


def test_motion_mask_excludes_neighboring_court_activity():
    mask = target_court_mask(
        _court(), (100, 100), (100, 100),
        sideline_margin_m=0.0, baseline_margin_m=0.0)
    assert mask is not None
    previous = np.zeros((100, 100), np.uint8)
    outside = previous.copy()
    outside[40:60, :10] = 255
    assert masked_frame_diff_energy(previous, outside, mask) == 0.0

    inside = previous.copy()
    inside[50:60, 45:55] = 255
    assert masked_frame_diff_energy(previous, inside, mask) > 0.0


def test_requested_but_missing_court_makes_visual_evidence_abstain(tmp_path):
    path = str(tmp_path / "tiny.avi")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100))
    assert writer.isOpened()
    for i in range(4):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:, :20] = i * 60  # strong off-court/whole-frame motion
        writer.write(frame)
    writer.release()

    class Detector:
        available = True

        def detect_persons(self, _frame):
            return [(0.05, 0.3, 0.03), (0.95, 0.7, 0.03)]

    result = analyze_visual(
        path, RallyConfig(analysis_fps=10.0, player_fps=10.0, court_auto=True),
        np.arange(0.0, 0.4, 0.1), Detector(), court=None)
    assert result["motion"] is None
    assert result["geometry"] is None
    assert result["near_track"] is None
    assert result["player_samples"] == []


def test_player_inference_is_batched_without_changing_sample_order(tmp_path):
    path = str(tmp_path / "batch.avi")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100))
    assert writer.isOpened()
    for _index in range(20):
        writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()

    class Detector:
        available = True
        batch_calls = 0

        def detect_persons_batch(self, frames):
            self.batch_calls += 1
            return [[(0.4, 0.8, 0.05), (0.6, 0.3, 0.02)] for _frame in frames]

    detector = Detector()
    timeline = np.arange(0.0, 2.0, 0.1)
    result = analyze_visual(
        path, RallyConfig(
            analysis_fps=10.0, player_fps=10.0,
            court_auto=False, play_mode="casual",
        ),
        timeline, detector, court=None)

    assert detector.batch_calls == 2
    assert [sample[0] for sample in result["player_samples"]] == pytest.approx(timeline)
    assert result["geometry"] is not None


def test_decode_shortfall_does_not_forward_fill_player_geometry(tmp_path):
    path = str(tmp_path / "short.avi")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (100, 100))
    assert writer.isOpened()
    for _index in range(5):
        writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()

    class Detector:
        available = True

        def detect_persons_batch(self, frames):
            return [[(0.4, 0.8, 0.05), (0.6, 0.3, 0.02)] for _frame in frames]

    result = analyze_visual(
        path, RallyConfig(
            analysis_fps=10.0, player_fps=10.0,
            court_auto=False, play_mode="casual",
        ),
        np.arange(0.0, 1.0, 0.1), Detector(), court=None)

    assert np.any(result["geometry"][:5] > 0)
    assert np.all(result["geometry"][5:] == 0)
