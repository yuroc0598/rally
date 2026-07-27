import numpy as np
import pytest

from rally.fusion.points import (
    drop_isolated_points,
    effective_strikes,
    is_coherent_rally,
    points_from_strikes,
    points_from_strikes_movement,
    serve_anchor,
    snap_serve_starts,
)


def _flat_track(n, x, y):
    return np.full(n, x), np.full(n, y)


def test_snap_serve_extends_start_to_first_strike():
    # decoded start at 60, but the serve strike is at 57.0
    segs = [(60.0, 68.0)]
    onsets = np.array([57.0, 58.5, 60.2, 61.0, 62.5])
    out = snap_serve_starts(segs, onsets, lookback_s=6.0, preroll_s=1.5)
    assert out[0][0] == pytest.approx(55.5)  # 57.0 - 1.5 preroll
    assert out[0][1] == 68.0                  # end unchanged


def test_snap_serve_never_moves_later_or_negative():
    segs = [(2.0, 9.0)]
    onsets = np.array([1.0])  # would give -0.5; clamp to 0
    out = snap_serve_starts(segs, onsets, lookback_s=6.0, preroll_s=1.5)
    assert out[0][0] == 0.0
    # no onset near start -> unchanged
    assert snap_serve_starts([(50.0, 55.0)], np.array([10.0]), 6.0, 1.5)[0][0] == 50.0


def test_points_from_strikes_bounds_tightly_to_play():
    # region padded out to [10, 30], but strikes (play) only span 15..22
    regions = [(10.0, 30.0)]
    onsets = np.array([15, 16, 17, 18, 19, 20, 21, 22])
    out = points_from_strikes(regions, onsets, gap_s=2.5, min_strikes=2,
                              toss_preroll_s=1.0, landing_tail_s=1.2, total_s=40.0)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(14.0)   # 15 - 1.0 toss preroll
    assert out[0][1] == pytest.approx(23.2)   # 22 + 1.2 landing tail
    # the dead time 23.2..30 (walking) is excluded
    assert out[0][1] < 30.0


def test_points_from_strikes_splits_two_points_and_excludes_gap_walking():
    regions = [(10.0, 40.0)]
    onsets = np.array([15, 16, 17,          # point A -> ends 17
                       25, 26, 27])         # point B (gap 17->25 = 8s of walking)
    out = points_from_strikes(regions, onsets, gap_s=2.5, min_strikes=2,
                              toss_preroll_s=1.0, landing_tail_s=1.0, total_s=50.0)
    assert len(out) == 2
    assert out[0] == pytest.approx((14.0, 18.0))   # A: 15-1 .. 17+1
    assert out[1] == pytest.approx((24.0, 28.0))   # B: 25-1 .. 27+1
    assert out[0][1] < out[1][0]                    # the 18..24 walking is gone


def test_movement_merge_keeps_lull_as_one_point():
    # two strike clusters split by a 3s gap, but the near player barely moved -> one point
    regions = [(10.0, 40.0)]
    onsets = np.array([15, 16, 17,  20, 21, 22])   # gap 17->20 = 3s
    timeline = np.arange(10.0, 40.0, 0.5)
    px, py = _flat_track(timeline.size, 0.5, 0.8)   # stationary near player
    out = points_from_strikes_movement(
        regions, onsets, timeline, px, py,
        gap_s=2.5, merge_max_gap_s=4.0, move_thresh=0.15, min_strikes=2,
        toss_preroll_s=1.0, landing_tail_s=1.0, total_s=50.0)
    assert len(out) == 1                     # lull merged, not split
    assert out[0] == pytest.approx((14.0, 23.0))


def test_movement_merge_splits_when_player_resets():
    regions = [(10.0, 40.0)]
    onsets = np.array([15, 16, 17,  20, 21, 22])   # same 3s gap
    timeline = np.arange(10.0, 40.0, 0.5)
    px = np.full(timeline.size, 0.5)
    py = np.full(timeline.size, 0.8)
    # near player walks far right during the gap (17..20) -> a real reset
    moving = (timeline >= 17.5) & (timeline <= 20.5)
    px[moving] = 0.9
    out = points_from_strikes_movement(
        regions, onsets, timeline, px, py,
        gap_s=2.5, merge_max_gap_s=4.0, move_thresh=0.15, min_strikes=2,
        toss_preroll_s=1.0, landing_tail_s=1.0, total_s=50.0)
    assert len(out) == 2                     # big movement -> two points


def test_movement_merge_long_gap_always_splits():
    # a long gap is a real boundary even if movement is unknown/small
    regions = [(10.0, 60.0)]
    onsets = np.array([15, 16, 17,  40, 41, 42])   # 23s gap >> merge_max_gap
    timeline = np.arange(10.0, 60.0, 0.5)
    px, py = _flat_track(timeline.size, 0.5, 0.8)
    out = points_from_strikes_movement(
        regions, onsets, timeline, px, py,
        gap_s=2.5, merge_max_gap_s=4.0, move_thresh=0.15, min_strikes=2,
        toss_preroll_s=1.0, landing_tail_s=1.0, total_s=70.0)
    assert len(out) == 2


def test_serve_anchor_recovers_split_off_serve():
    # serve at 1.05 (preceded by silence), then rally starting 3.97 (gap 2.9 > point_gap)
    onsets = np.array([1.05, 3.97, 4.42, 5.71, 6.09])
    s = serve_anchor(3.97, onsets, serve_window_s=4.0, point_gap_s=2.5)
    assert s == pytest.approx(1.05)


def test_serve_anchor_none_when_no_isolated_serve():
    # the strike before 'first' is part of continuous prior activity (small gaps)
    onsets = np.array([1.0, 1.5, 2.0, 2.5, 3.97, 4.4])
    assert serve_anchor(3.97, onsets, serve_window_s=4.0, point_gap_s=2.5) is None
    # nothing within the window at all
    assert serve_anchor(100.0, np.array([1.0, 2.0]), serve_window_s=4.0, point_gap_s=2.5) is None


def test_serve_attach_in_points_from_strikes_extends_start_to_serve():
    regions = [(0.0, 12.0)]
    onsets = np.array([1.05, 3.97, 4.42, 5.71, 6.09, 7.76])  # lone serve + rally
    out = points_from_strikes(regions, onsets, gap_s=2.5, min_strikes=2,
                              toss_preroll_s=1.0, landing_tail_s=1.0, total_s=20.0,
                              echo_s=0.35, min_dur_s=1.0, serve_window_s=4.0)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(0.05)  # 1.05 serve - 1.0 preroll (not 3.97-1.0)


def test_effective_strikes_folds_echoes():
    # a strike + its bounce/echo 0.2s later count as one event
    assert effective_strikes([10.0, 10.2, 11.5, 11.7], echo_s=0.35) == 2
    assert effective_strikes([10.0, 11.0, 12.0], echo_s=0.35) == 3


def test_effective_strikes_collapses_relative_to_last_counted_hit():
    # chain of transients, each <echo of the previous but drifting: anchored to the last
    # COUNTED hit -> [0, 0.4] are the two distinct events, not one.
    assert effective_strikes([0.0, 0.2, 0.4, 0.6], echo_s=0.35) == 2
    # a genuine tight ring (all within echo of the anchor) stays one event
    assert effective_strikes([0.0, 0.1, 0.2, 0.3], echo_s=0.35) == 1


def test_coherence_rejects_echo_doublet_but_keeps_real_short_point():
    # 72.5/72.8 echo doublet -> 1 effective strike, span 0.3s -> not a rally
    assert not is_coherent_rally([72.5, 72.8], min_strikes=2, min_dur_s=1.0, echo_s=0.35)
    # real serve+return ~1.3s apart -> coherent
    assert is_coherent_rally([100.0, 101.3], min_strikes=2, min_dur_s=1.0, echo_s=0.35)
    # sustained rally -> coherent
    assert is_coherent_rally([1, 2, 3, 4, 5], min_strikes=2, min_dur_s=1.0, echo_s=0.35)


def test_coherence_filter_in_points_from_strikes():
    regions = [(60.0, 80.0)]
    # a real rally (many strikes) + an echo doublet fragment
    onsets = np.array([61, 62, 63, 64, 65,  72.5, 72.8])
    out = points_from_strikes(regions, onsets, gap_s=2.5, min_strikes=2,
                              toss_preroll_s=1.0, landing_tail_s=1.0, total_s=100.0,
                              echo_s=0.35, min_dur_s=1.0)
    assert len(out) == 1                     # echo doublet rejected
    assert out[0][0] == pytest.approx(60.0)  # 61 - 1.0


def test_drop_isolated_points():
    # three clustered points + one lone point 5 min later
    pts = [(100.0, 108.0), (130.0, 137.0), (160.0, 166.0), (460.0, 466.0)]
    out = drop_isolated_points(pts, isolation_gap_s=120.0)
    assert (460.0, 466.0) not in out
    assert len(out) == 3


def test_drop_isolated_keeps_all_when_dense():
    pts = [(10.0, 16.0), (40.0, 46.0), (80.0, 86.0)]
    assert drop_isolated_points(pts, isolation_gap_s=120.0) == pts


def test_points_from_strikes_drops_stray_single_hits():
    regions = [(10.0, 40.0)]
    onsets = np.array([15.0,                 # lone stray sound
                       25, 26, 27, 28])      # real point
    out = points_from_strikes(regions, onsets, gap_s=2.5, min_strikes=2,
                              toss_preroll_s=1.0, landing_tail_s=1.0, total_s=50.0)
    assert len(out) == 1
    assert out[0][0] == pytest.approx(24.0)


def test_snap_serve_no_overlap_with_previous_segment():
    segs = [(10.0, 20.0), (22.0, 30.0)]
    onsets = np.array([10.0, 18.0, 21.0, 24.0])  # 21.0 - preroll would back into seg 0
    out = snap_serve_starts(segs, onsets, lookback_s=6.0, preroll_s=2.0)
    assert out[1][0] >= out[0][1]  # second start does not overlap first end
