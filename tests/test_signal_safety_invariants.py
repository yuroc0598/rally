"""Safety invariants for signal code; these are not claims of real-video accuracy."""

import numpy as np
import pytest

from rally.fusion.ball_verify import _in_play_mask, _live_components, _net_crossings
from rally.signals.ball import BallTrack, _decode_heatmap, ball_in_play_channel
from rally.signals.court import SERVICE_Y
from rally.signals.court_detect import valid_court_quad
from rally.signals.motion import camera_shift_px
from rally.signals.trajectory import SmoothTrack, bounces_from_velocity


def _smooth(t, x, y, vx, vy, confidence=None, measured=None):
    n = len(t)
    return SmoothTrack(
        np.asarray(t, float), np.asarray(x, float), np.asarray(y, float),
        np.asarray(vx, float), np.asarray(vy, float),
        np.ones(n) if confidence is None else np.asarray(confidence, float),
        np.ones(n, bool) if measured is None else np.asarray(measured, bool),
    )


def test_heatmap_speed_gate_rejects_far_candidate(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "HoughCircles", lambda *a, **k: np.array([[[100, 100, 3]]]))
    result = _decode_heatmap(np.zeros((360, 640), np.uint8), prev=(0, 0),
                             speed_limit_px=10)
    assert result is None


def test_ball_activity_matches_direct_window_definition():
    t = np.arange(0.0, 2.0, 0.1)
    track = BallTrack(t, 10 * t, np.zeros_like(t))
    timeline = np.array([0.0, 0.5, 1.0, 2.0])
    got = ball_in_play_channel(track, timeline, window_s=0.4, min_speed_px=3.0)
    speed = np.r_[0.0, np.full(t.size - 1, 10.0)]
    expected = []
    for at in timeline:
        window = (t >= at - 0.2) & (t <= at + 0.2)
        expected.append(np.mean(speed[window] > 3.0) if window.any() else 0.0)
    assert got == pytest.approx(expected)


def test_flat_frames_do_not_produce_phase_correlation_pan():
    flat = np.full((80, 120), 127, np.uint8)
    assert camera_shift_px(flat, flat) == 0.0


def test_image_y_reversal_without_2d_turn_is_not_a_bounce():
    t = np.arange(0.0, 0.5, 0.02)
    vy = np.where(t < 0.25, 50.0, -50.0)
    st = _smooth(t, 1000 * t, np.zeros_like(t), np.full(t.size, 1000.0), vy)
    assert bounces_from_velocity(st, min_descent_px_s=40,
                                 min_turn_angle_deg=20) == []


def test_live_components_do_not_bridge_long_time_holes():
    t = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    assert _live_components(np.ones(t.size, bool), t, 0.2) == [(0, 3), (3, 6)]


def test_short_confident_prediction_gap_can_bridge_but_is_visible_to_coverage_guard():
    t = np.arange(0.0, 0.2, 0.04)
    measured = np.array([True, False, False, False, True])
    st = _smooth(t, t, t, np.full(t.size, 100.0), np.zeros(t.size),
                 measured=measured)
    live = _in_play_mask(st, min_speed_px_s=25.0, min_conf=0.3, max_fill_gap_s=0.2)
    assert live.all()
    assert measured[live].mean() < 0.55


class _IdentityCourt:
    def to_court(self, points):
        return np.asarray(points, float).reshape(-1, 2)


def test_net_crossing_is_not_counted_across_tracking_gap():
    t = np.array([0.0, 2.0])
    st = _smooth(t, [1, 1], [5, 18], [100, 100], [0, 0])
    assert _net_crossings(st, _IdentityCourt(), np.ones(2, bool), max_gap_s=0.35) == 0


def test_court_quad_rejects_small_and_off_frame_geometry():
    shape = (1000, 1200, 3)
    assert valid_court_quad([(200, 900), (1000, 900), (850, 250), (350, 250)], shape)
    assert not valid_court_quad([(20, 30), (60, 30), (55, 10), (25, 10)], shape)
    assert not valid_court_quad([(-500, 900), (1000, 900), (850, 250), (350, 250)], shape)


def test_service_line_uses_18_foot_measurement():
    assert SERVICE_Y == pytest.approx(18 * 0.3048)
