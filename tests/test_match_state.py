from dataclasses import replace

import numpy as np

from rally.config import RallyConfig
from rally.fusion.match_state import validate_match_sequence
from rally.signals.ball import BallTrack
from rally.signals.player import ServeSetupObservation, classify_position_setup
from rally.signals.serve import classify_ball_serve


def _ob(point, side, *, serve=True, ball_checked=True, setup=True):
    return ServeSetupObservation(
        point=point, first_strike=point[0] + 1.0, side=side,
        side_confidence=1.0, near_x=0.2 if side == "left" else 0.8,
        near_x_std=0.01, sampled_frames=5, pose_frames=5,
        ready_frames=3 if setup else 0, serve_motion=False,
        setup_evidence=setup, observable=True, ball_checked=ball_checked,
        ball_serve_evidence=serve,
        ball_coverage=0.5 if serve else 0.0,
        ball_vertical_span=0.1 if serve else 0.0,
    )


def test_auto_match_drops_candidate_without_serve_between_confirmed_serves():
    points = [(0, 3), (10, 13), (20, 23), (30, 33)]
    observations = [
        _ob(points[0], "right"),
        _ob(points[1], "left", serve=False, setup=True),  # receiver-ready is insufficient
        _ob(points[2], "left"),
        _ob(points[3], "right"),
    ]
    onsets = np.array([1, 2, 11, 12.2, 21, 22, 22.8, 31, 32])

    kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="auto", match_point_start_preroll_s=1.0))

    assert kept == [points[0], points[2], points[3]]
    assert stage["dropped"][0]["reason_code"] == "missing_confirmed_serve"


def test_match_state_does_not_treat_side_repetition_alone_as_invalid():
    points = [(0, 3), (10, 13), (20, 23), (30, 33)]
    observations = [
        _ob(points[0], "right"), _ob(points[1], "left"),
        _ob(points[2], "left"), _ob(points[3], "right"),
    ]
    onsets = np.array([1, 2, 11, 12, 21, 22, 31, 32])

    kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0))

    assert kept == points
    assert stage["dropped"] == []


def test_auto_mode_leaves_unstructured_warmup_without_three_serve_anchors():
    points = [(0, 3), (10, 13), (20, 23)]
    observations = [
        _ob(points[0], "left", serve=False),
        _ob(points[1], "left"),
        _ob(points[2], "right"),
    ]
    onsets = np.array([1, 2, 11, 12, 21, 22])

    kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="auto", match_point_start_preroll_s=1.0))

    assert kept == points
    assert stage["status"] == "abstained"


def _track(*, vertical_span=0.1, x_start=0.5, x_end=0.55):
    t = np.arange(0.2, 1.21, 1 / 60)
    x = np.linspace(x_start, x_end, t.size) * 1920
    y = np.linspace(0.58, 0.58 - vertical_span, t.size) * 1080
    return BallTrack(t, x, y)


def test_ball_serve_requires_sustained_vertical_in_court_motion():
    cfg = RallyConfig()
    positive = classify_ball_serve((0, 2), np.array([1.0]), _track(), 1920, 1080, cfg)
    horizontal_feed = classify_ball_serve(
        (0, 2), np.array([1.0]), _track(vertical_span=0.015), 1920, 1080, cfg)
    outside_court = classify_ball_serve(
        (0, 2), np.array([1.0]), _track(x_start=0.91, x_end=0.95), 1920, 1080, cfg)

    assert positive.confirmed is True
    assert horizontal_feed.confirmed is False
    assert outside_court.confirmed is False


def _player_samples(*, moving=False, baseline=True):
    samples = []
    for index, time in enumerate((0.0, 0.4, 0.8, 1.1)):
        far_y = 0.46 if baseline else 0.63
        near_x = 0.70 + (0.05 * index if moving else 0.0)
        samples.append((time, [(0.30, far_y, 0.002), (near_x, 0.82, 0.02)]))
    return samples


def test_position_setup_requires_baseline_player_and_static_formation():
    cfg = RallyConfig()
    static = classify_position_setup(
        (0, 3), 1.0, _player_samples(), cfg)
    moving = classify_position_setup(
        (0, 3), 1.0, _player_samples(moving=True), cfg)
    no_baseline = classify_position_setup(
        (0, 3), 1.0,
        [(time, [(0.30, 0.63, 0.002), (0.70, 0.64, 0.02)])
         for time in (0.0, 0.4, 0.8, 1.1)],
        cfg,
    )

    assert static.setup_evidence is True
    assert static.player_tracks == 2
    assert static.stable_tracks == 2
    assert moving.setup_evidence is False
    assert no_baseline.setup_evidence is False


def test_pose_only_confirms_serve_when_position_setup_is_present():
    point = (0, 3)
    base = dict(
        point=point, first_strike=1.0, side="left", side_confidence=1.0,
        near_x=0.2, near_x_std=0.01, sampled_frames=5, pose_frames=5,
        ready_frames=4, serve_motion=True, setup_evidence=True, observable=True,
        overhead_strikes=(1.0,),
        ball_checked=True, ball_serve_evidence=False,
    )

    assert ServeSetupObservation(**base).confirmed_serve is False
    assert ServeSetupObservation(
        **base, position_checked=True, position_setup_evidence=True,
        position_setup_strikes=(1.0,),
    ).confirmed_serve is True
    assert ServeSetupObservation(
        **base, position_checked=True, position_setup_evidence=True,
        position_setup_strikes=(2.0,),
    ).confirmed_serve is False


def test_match_decoder_merges_fault_and_retry_into_one_logical_point():
    points = [(35.0, 42.7), (43.9, 47.5)]
    observations = []
    for point, strike in zip(points, (38.2, 44.9)):
        observations.append(ServeSetupObservation(
            point=point, first_strike=strike, side="left", side_confidence=1.0,
            near_x=0.2, near_x_std=0.01, sampled_frames=5, pose_frames=5,
            ready_frames=4, serve_motion=True, setup_evidence=True, observable=True,
            overhead_frames=2, overhead_strikes=(strike,),
            position_checked=True, position_setup_evidence=True,
            position_best_strike=strike, position_setup_strikes=(strike,),
            position_score=0.8, ball_checked=True, ball_serve_evidence=True,
            ball_best_strike=strike, ball_coverage=0.6, ball_vertical_span=0.1,
            ball_outgoing_span=0.04, ball_ordered_evidence=True,
        ))

    kept, stage = validate_match_sequence(
        points, np.array([38.2, 40.0, 44.9, 46.5]), observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(40.9, 47.5)]
    assert stage["logical_groups"][0]["member_indices"] == [0, 1]
    assert stage["logical_groups"][0]["serve_member_index"] == 1


def test_match_decoder_anchors_compact_serve_before_long_exchange():
    points = [(41.044, 43.376), (48.520, 50.520), (52.357, 69.165)]
    observations = [
        _ob(points[0], "right", serve=False, setup=False),
        _ob(points[1], "right", serve=True, setup=False),
        _ob(points[2], "right", serve=True, setup=False),
    ]

    kept, stage = validate_match_sequence(
        points, np.array([42.044, 42.933, 49.520, 53.357, 55.313, 68.165]),
        observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(45.52, 69.165)]
    assert stage["logical_groups"][0]["serve_member_index"] == 1


def test_match_decoder_does_not_anchor_pickup_feed_before_later_real_serve():
    points = [
        (80.801, 84.667), (87.313, 89.313),
        (94.443, 97.025), (107.089, 111.160),
    ]
    early_feed = replace(
        _ob(points[0], "left", serve=True, setup=False),
        position_checked=True, position_setup_evidence=True,
        position_best_strike=81.801, position_setup_strikes=(81.801,),
        position_score=0.95, ball_best_strike=83.625,
        ball_coverage=0.194, ball_vertical_span=0.144,
        ball_outgoing_span=0.018, ball_ordered_evidence=True,
    )
    real_serve = replace(
        _ob(points[3], "left", serve=True, setup=False),
        ball_best_strike=108.089, ball_coverage=0.472,
        ball_vertical_span=0.169, ball_outgoing_span=0.056,
        ball_ordered_evidence=True,
    )
    observations = [
        early_feed,
        _ob(points[1], "left", serve=False, setup=True),
        _ob(points[2], "left", serve=False, setup=False),
        real_serve,
    ]

    kept, stage = validate_match_sequence(
        points,
        np.array([81.801, 83.625, 88.313, 95.443, 96.025, 108.089, 109.069]),
        observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(104.089, 111.16)]
    assert stage["logical_groups"][0]["serve_member_index"] == 3


def test_match_decoder_rejects_weak_ball_only_single_impact():
    point = (0.0, 2.6)
    weak = replace(
        _ob(point, "right", serve=True, setup=False),
        ball_coverage=0.213, ball_vertical_span=0.036,
        ball_outgoing_span=0.095, ball_ordered_evidence=True,
    )

    kept, stage = validate_match_sequence(
        [point], np.array([1.0]), [weak], RallyConfig(play_mode="match"))

    assert kept == []
    assert stage["dropped"][0]["reason_code"] == "missing_confirmed_serve"

    strong = replace(weak, ball_coverage=0.398, ball_vertical_span=0.125)
    kept, _stage = validate_match_sequence(
        [point], np.array([1.0]), [strong],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )
    assert kept == [point]


def test_match_decoder_reapplies_minimum_duration_after_overlap_clamp():
    points = [(0.0, 3.0), (2.4, 3.4)]
    observations = [_ob(points[0], "left"), _ob(points[1], "right")]

    kept, stage = validate_match_sequence(
        points, np.array([1.0, 2.0, 3.4]), observations,
        RallyConfig(
            play_mode="match", min_rally_s=2.0,
            match_point_start_preroll_s=1.0,
        ),
    )

    assert kept == [points[0]]
    assert stage["logical_groups"][1]["reason_code"] == (
        "clamped_below_minimum_duration")
