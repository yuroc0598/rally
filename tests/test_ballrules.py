import numpy as np
import pytest

from rally.signals.ballrules import (
    ball_speed_kmh,
    detect_bounces,
    is_in,
    point_end_events,
    refine_end_from_events,
    side_of,
)
from rally.signals.court import COURT_L, DOUBLES_W, NET_Y


class FakeCourt:
    """Identity homography: treat image px as court metres (for speed/mapping tests)."""
    def to_court(self, pts):
        return np.asarray(pts, float).reshape(-1, 2)


def test_detect_bounces_finds_ground_contacts():
    t = np.arange(0, 3.0, 0.05)
    # image-y oscillates 20 (apex, screen-high) .. 100 (ground, screen-low); maxima = bounces
    y = 60 - 40 * np.cos(2 * np.pi * t / 1.0)
    idx = detect_bounces(t, y, prominence_px=10, min_sep_s=0.3)
    bt = sorted(t[i] for i in idx)
    assert len(bt) == 3
    for got, exp in zip(bt, [0.5, 1.5, 2.5]):
        assert got == pytest.approx(exp, abs=0.1)


def test_detect_bounces_interpolates_nan_gaps():
    t = np.arange(0, 2.0, 0.05)
    y = 60 - 40 * np.cos(2 * np.pi * t / 1.0)
    y[10:13] = np.nan  # detection gap
    assert len(detect_bounces(t, y, prominence_px=10)) == 2


def test_is_in_and_out():
    assert is_in(DOUBLES_W / 2, COURT_L / 2)                 # centre -> in
    assert is_in(2.0, 1.0)                                    # inside near court
    assert not is_in(-1.0, 5.0)                               # wide of the sideline
    assert not is_in(DOUBLES_W / 2, COURT_L + 2.0)           # long past the baseline


def test_side_of():
    assert side_of(NET_Y - 3) == "near"
    assert side_of(NET_Y + 3) == "far"


def test_point_end_double_bounce_same_side():
    # two bounces on the near side 1 s apart -> double bounce (point ends at the 2nd)
    bounces = [(10.0, 5.0, 4.0), (11.0, 6.0, 3.0)]
    ev = point_end_events(bounces, double_bounce_window_s=2.5)
    assert (11.0, "double_bounce") in ev


def test_point_end_opposite_sides_is_not_double_bounce():
    bounces = [(10.0, 5.0, 4.0), (11.0, 5.0, NET_Y + 5)]  # near then far = normal rally
    ev = [e for e in point_end_events(bounces) if e[1] == "double_bounce"]
    assert ev == []


def test_point_end_out_of_bounds():
    bounces = [(10.0, DOUBLES_W / 2, COURT_L + 3.0)]  # lands well past the baseline
    ev = point_end_events(bounces)
    assert (10.0, "out") in ev


def test_ball_speed_kmh():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    x = np.array([0.0, 1.0, 2.0, 3.0])   # 1 m per 0.1 s = 10 m/s = 36 km/h
    y = np.zeros(4)
    sp = ball_speed_kmh(t, x, y, FakeCourt(), smooth=1)
    assert sp[1] == pytest.approx(36.0, abs=0.5)


def test_refine_end_trims_to_point_ending_event():
    # rally 10..20; a double-bounce at 13.0 ends it -> end trimmed to ~13.8
    new_e, reason = refine_end_from_events(10.0, 20.0, [(13.0, "double_bounce")],
                                           min_rally_s=1.5, tail_s=0.8, max_extend_s=3.0)
    assert reason == "double_bounce"
    assert new_e == pytest.approx(13.8, abs=0.01)


def test_refine_end_ignores_serve_bounce_and_keeps_end_when_none():
    # an event too early (within min_rally, e.g. the serve bounce) is ignored
    ne, r = refine_end_from_events(10.0, 20.0, [(10.5, "double_bounce")], min_rally_s=1.5)
    assert r is None and ne == 20.0
    # no events -> unchanged
    assert refine_end_from_events(10.0, 20.0, []) == (20.0, None)


def test_ball_speed_rejects_implausible_jumps():
    # a detection jump of 100 m in 0.1 s = 3600 km/h -> dropped as noise
    t = np.array([0.0, 0.1, 0.2])
    x = np.array([0.0, 100.0, 100.5])
    y = np.zeros(3)
    sp = ball_speed_kmh(t, x, y, FakeCourt(), smooth=1, max_kmh=250.0)
    assert not np.isfinite(sp[1])            # jump rejected
    assert sp[2] == pytest.approx(18.0, abs=1.0)  # 0.5 m / 0.1 s = 5 m/s = 18 km/h
