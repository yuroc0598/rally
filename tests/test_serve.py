import numpy as np
import pytest

from rally.signals.player import (
    clean_track,
    court_speed,
    find_serve_start,
    refine_starts_with_serve,
)


def test_clean_track_rejects_teleport():
    t = np.arange(0, 1.0, 0.1)
    cx = np.full(t.size, 3.0)
    cy = np.full(t.size, 5.0)
    cx[5] = 30.0  # impossible jump (27 m in 0.1 s) -> must be rejected
    gx, gy = clean_track(t, cx, cy, speed_limit_mps=8.0, smooth_win=1)
    assert abs(gx[5] - 3.0) < 1.0  # interpolated back near the real position


def test_clean_track_interpolates_gaps():
    t = np.arange(0, 1.0, 0.1)
    # plausible motion (~2 m/s) with a detection gap at indices 1,2
    cx = np.array([0.0, np.nan, np.nan, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], float)
    cy = np.zeros(t.size)
    gx, _ = clean_track(t, cx, cy, speed_limit_mps=8.0, smooth_win=1)
    assert np.isfinite(gx).all()
    assert 0 < gx[1] < 0.6  # interpolated between 0 and 0.6


def test_find_serve_start_anchors_to_setup():
    # 0..5 s: near player set at baseline (y=0.3) and still; rally strike at 5.0
    t = np.arange(0, 6.0, 0.2)
    cy = np.where(t < 3.0, 4.0, 0.3)          # walks in, then set at baseline from t=3
    speed = np.where(t < 3.0, 1.5, 0.1)        # moving, then still
    s = find_serve_start(5.0, t, cy, speed, lookback_s=6.0, baseline_y=1.5,
                         still_speed=0.6, preroll_s=0.8, max_lead_s=5.0)
    assert s == pytest.approx(3.0 - 0.8, abs=0.25)  # start of set period minus preroll


def test_find_serve_start_caps_lead_to_drop_loiter():
    # player is set at baseline & still for a long time (6 s) before the rally at t=10
    t = np.arange(0, 11.0, 0.2)
    cy = np.full(t.size, 0.3)   # at baseline throughout
    speed = np.full(t.size, 0.1)  # still throughout
    s = find_serve_start(10.0, t, cy, speed, lookback_s=8.0, baseline_y=1.5,
                         still_speed=0.6, preroll_s=0.8, max_lead_s=2.5)
    assert s == pytest.approx(10.0 - 2.5, abs=0.05)  # capped, not 8 s of loiter


def test_find_serve_start_none_when_never_set():
    t = np.arange(0, 6.0, 0.2)
    cy = np.full(t.size, 5.0)      # never at baseline
    speed = np.full(t.size, 2.0)   # never still
    assert find_serve_start(5.0, t, cy, speed) is None


def test_refine_starts_extends_and_no_overlap():
    pts = [(4.5, 10.0), (20.0, 26.0)]
    onsets = np.array([5.0, 6.0, 7.0, 21.0, 22.0])
    t = np.arange(0, 27.0, 0.2)
    cy = np.full(t.size, 0.3)      # always at baseline
    speed = np.full(t.size, 0.1)   # always still -> set-up found for both
    out = refine_starts_with_serve(pts, onsets, t, cy, speed,
                                   lookback_s=6.0, baseline_y=1.5,
                                   still_speed=0.6, preroll_s=0.8)
    assert out[0][0] <= 4.5              # first start moved earlier (or equal)
    assert out[1][0] >= out[0][1] - 1e-6  # no overlap with previous point
