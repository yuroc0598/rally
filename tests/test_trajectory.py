import numpy as np
import pytest

from rally.signals.ball import BallTrack
from rally.signals.trajectory import bounces_from_velocity, smooth_track


def _bouncing_track(fps=30.0, dur=3.0, period=1.0, amp=40.0, base=60.0):
    """Synthetic image-space ball: cosine wave whose maxima (screen-lowest) are bounces."""
    t = np.arange(0.0, dur, 1.0 / fps)
    x = 100.0 + 50.0 * t                       # moving steadily across the frame
    y = base - amp * np.cos(2 * np.pi * t / period)
    return BallTrack(t, x, y)


def test_smooth_track_fills_gaps_and_flags_confidence():
    tr = _bouncing_track()
    x = tr.x.copy(); y = tr.y.copy()
    x[40:55] = np.nan; y[40:55] = np.nan       # a 0.5 s detection dropout
    st = smooth_track(BallTrack(tr.t, x, y), max_gap_s=0.3)

    assert np.isfinite(st.x).all() and np.isfinite(st.y).all()   # gaps filled
    # positions outside the gap stay close to ground truth
    good = np.r_[np.arange(0, 40), np.arange(55, tr.t.size)]
    assert np.nanmedian(np.abs(st.y[good] - tr.y[good])) < 6.0
    # confidence collapses inside the long dropout, stays high where measured
    assert st.confidence[47] < 0.2
    assert st.confidence[10] > 0.7
    assert st.measured[10] and not st.measured[47]


def test_smooth_track_gates_single_frame_outlier():
    tr = _bouncing_track()
    y = tr.y.copy()
    y[60] += 300.0                             # a one-frame lock onto a distractor
    st = smooth_track(BallTrack(tr.x if False else tr.t, tr.x, y))
    # smoother should not follow the 300 px spike
    assert abs(st.y[60] - tr.y[60]) < 60.0
    assert not st.measured[60]                 # gated out


def test_bounces_from_velocity_finds_ground_contacts():
    tr = _bouncing_track(dur=3.0, period=1.0)
    st = smooth_track(tr)
    idx = bounces_from_velocity(st, min_descent_px_s=20.0)
    bt = sorted(st.t[i] for i in idx)
    assert len(bt) == 3                        # bounces at ~0.5, 1.5, 2.5
    for got, exp in zip(bt, [0.5, 1.5, 2.5]):
        assert got == pytest.approx(exp, abs=0.08)


def test_bounces_from_velocity_ignores_flat_track():
    t = np.arange(0, 2.0, 1 / 30.0)
    st = smooth_track(BallTrack(t, 100 + 0 * t, 200 + 0 * t))
    assert bounces_from_velocity(st, min_descent_px_s=20.0) == []
