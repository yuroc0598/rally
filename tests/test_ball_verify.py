import numpy as np

from rally.fusion.ball_verify import (
    _dedupe_non_overlapping,
    _net_crossings,
    rally_verdict,
)
from rally.signals.ball import BallTrack
from rally.signals.trajectory import smooth_track


class FakeCourt:
    """Identity homography: image px == court metres (net at y = NET_Y ~ 11.885)."""
    def to_court(self, pts):
        return np.asarray(pts, float).reshape(-1, 2)


def _rally_track(dur=4.0, fps=30.0, period=0.8, amp=9.0, net=11.885):
    """A live ball: moves across the frame, bounces, and crosses the net repeatedly."""
    t = np.arange(0.0, dur, 1.0 / fps)
    x = 5.0 + 2.0 * t
    y = net + amp * np.cos(2 * np.pi * t / period)
    return smooth_track(BallTrack(t, x, y))


def test_verdict_accepts_a_live_rally():
    st = _rally_track()
    v = rally_verdict(st, FakeCourt(), 0.0, 4.0)
    assert v.is_rally
    assert v.n_net_crossings >= 2
    assert v.n_bounces >= 2
    assert v.start <= 1.0 and v.end <= 4.0 + 3.0


def test_verdict_rejects_a_dead_ball():
    # ball essentially stationary -> never "in play"
    t = np.arange(0.0, 4.0, 1 / 30.0)
    st = smooth_track(BallTrack(t, 100 + 0 * t, 200 + 0 * t))
    v = rally_verdict(st, FakeCourt(), 0.0, 4.0)
    assert not v.is_rally
    assert v.in_play_frac < 0.1


def test_verdict_rejects_short_structureless_blip():
    # a brief moving blob on one side, no net crossing, too short
    t = np.arange(0.0, 0.8, 1 / 30.0)
    x = 5.0 + 30.0 * t
    y = 5.0 + 0.0 * t                      # stays on the near side, no crossing/bounce
    st = smooth_track(BallTrack(t, x, y))
    v = rally_verdict(st, FakeCourt(), 0.0, 0.8, min_in_play_span_s=1.5)
    assert not v.is_rally


def test_verdict_snaps_start_to_serve():
    # dead for the first ~1.5 s, then a live rally -> start should jump forward to the ball
    st_live = _rally_track(dur=3.0)
    t = np.r_[np.arange(0.0, 1.5, 1 / 30.0), st_live.t + 1.5]
    x = np.r_[np.full(int(1.5 * 30), 5.0), st_live.x]
    y = np.r_[np.full(int(1.5 * 30), 5.0), st_live.y]
    st = smooth_track(BallTrack(t, x, y))
    v = rally_verdict(st, FakeCourt(), 0.0, 4.5, toss_preroll_s=1.0)
    assert v.is_rally
    assert v.start > 0.3        # not anchored at the dead window's start


def test_net_crossings_counts_side_changes():
    t = np.arange(0.0, 3.0, 1 / 30.0)
    y = 11.885 + 8.0 * np.cos(2 * np.pi * t / 1.0)   # crosses net twice per period
    st = smooth_track(BallTrack(t, 100 + 5 * t, y))
    in_play = np.ones(t.size, bool)
    assert _net_crossings(st, FakeCourt(), in_play) >= 4


def test_net_crossings_zero_without_court():
    st = _rally_track()
    assert _net_crossings(st, None, np.ones(st.t.size, bool)) == 0


def test_dedupe_non_overlapping():
    segs = [(0.0, 5.0), (4.0, 8.0), (10.0, 12.0)]
    out = _dedupe_non_overlapping(segs)
    assert out == [(0.0, 4.0), (4.0, 8.0), (10.0, 12.0)]   # previous end trimmed to next start


def test_verify_segments_keeps_rally_rejects_dead(monkeypatch):
    """Orchestrator keeps live-ball candidates and drops dead ones (tracker stubbed)."""
    import rally.signals.ball as ball_mod
    from rally.fusion import ball_verify

    def fake_track(video, model=None, start_s=0.0, end_s=None, **kw):
        # candidate near t=10 is a real rally; candidate near t=100 is a dead ball
        dur = (end_s or 0.0) - start_s
        t = np.arange(0.0, max(dur, 0.1), 1 / 30.0) + start_s
        if start_s < 50:
            base = _rally_track(dur=max(dur, 4.0))
            n = min(t.size, base.t.size)
            return BallTrack(t[:n], base.x[:n], base.y[:n])
        return BallTrack(t, 100 + 0 * t, 200 + 0 * t)     # stationary -> not a rally

    monkeypatch.setattr(ball_mod, "track_tracknet", fake_track)

    kept = ball_verify.verify_segments(
        "dummy.mp4", [(10.0, 14.0), (100.0, 104.0)],
        court=FakeCourt(), model=object())
    assert len(kept) == 1
    assert 8.0 <= kept[0][0] <= 12.0        # bounded near the live candidate's serve
