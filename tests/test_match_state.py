from dataclasses import replace

import numpy as np

from rally.config import RallyConfig
from rally.fusion.match_state import validate_match_sequence
from rally.signals.ball import BallTrack
from rally.signals.court import Court
from rally.signals.player import ServeSetupObservation, classify_position_setup
from rally.signals.serve import _cached_track_window, classify_ball_serve


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
    assert stage["logical_groups"][1]["member_indices"] == [1]


def test_match_state_merges_close_same_side_service_attempts():
    points = [(0, 3), (10, 13), (20, 23), (30, 33)]
    observations = [
        _ob(points[0], "right"), _ob(points[1], "left"),
        _ob(points[2], "left"), _ob(points[3], "right"),
    ]
    onsets = np.array([1, 2, 11, 12, 21, 22, 31, 32])

    kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0))

    # Ball-only attempts use the later confirmed retry as the conservative point start;
    # both hypotheses still form one logical point rather than two outputs.
    assert kept == [points[0], (20.0, 23.0), points[3]]
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


def test_auto_match_drops_single_background_hit_before_target_match_but_keeps_warmup_rally():
    points = [(0, 3), (25, 28), (35, 38), (45, 48)]
    leading_noise = replace(
        _ob(points[0], None, serve=False, setup=False),
        target_court_filtered=True,
    )
    observations = [leading_noise, *[
        _ob(point, side) for point, side in zip(points[1:], ("right", "left", "right"))
    ]]

    kept, stage = validate_match_sequence(
        points, np.array([1.0, 26.0, 36.0, 46.0]), observations,
        RallyConfig(play_mode="auto", match_point_start_preroll_s=1.0),
    )

    assert kept == points[1:]
    assert stage["dropped"][0]["reason_code"] == (
        "weak_boundary_noise_outside_match_phase")

    warmup_kept, _stage = validate_match_sequence(
        points, np.array([1.0, 2.0, 26.0, 36.0, 46.0]), observations,
        RallyConfig(play_mode="auto", match_point_start_preroll_s=1.0),
    )
    assert warmup_kept[0] == points[0]


def test_auto_match_extends_through_contiguous_edge_candidates():
    points = [(0, 2), (12, 15), (22, 25), (32, 35), (44, 47), (58, 61)]
    observations = [
        _ob(points[0], None, serve=False, setup=False),
        _ob(points[1], "right"),
        _ob(points[2], "left"),
        _ob(points[3], "right"),
        _ob(points[4], None, serve=False, setup=False),
        _ob(points[5], "left"),
    ]
    onsets = np.array([1, 13, 14, 23, 24, 33, 34, 45, 46, 59, 60])

    kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="auto", match_point_start_preroll_s=1.0),
    )

    assert kept == points[1:4] + points[5:]
    assert stage["match_phases"] == [{"start_index": 0, "end_index": 5}]
    assert {item["member_indices"][0] for item in stage["dropped"]} == {0, 4}


def test_auto_match_bridges_long_anchor_gap_when_candidates_remain_continuous():
    points = [(0, 3), (20, 23), (40, 43), (70, 73), (100, 103),
              (130, 133), (150, 153), (170, 173)]
    observations = [
        _ob(point, "left" if index % 2 else "right",
            serve=index not in {3, 4})
        for index, point in enumerate(points)
    ]
    onsets = np.array([point[0] + 1.0 for point in points])

    _kept, stage = validate_match_sequence(
        points, onsets, observations,
        RallyConfig(play_mode="auto", match_phase_max_gap_s=50.0),
    )

    assert stage["match_phases"] == [{"start_index": 0, "end_index": 7}]
    assert {item["member_indices"][0] for item in stage["dropped"]} == {3, 4}


def test_auto_match_extends_to_supported_far_side_reaction_after_game_break():
    points = [(0, 3), (10, 13), (20, 23), (45, 49)]
    final = replace(
        _ob(points[-1], None, serve=False, setup=False),
        target_court_filtered=True,
        position_checked=True,
        position_best_strike=47.0,
        position_server_end="far",
        position_server_span=0.02,
        position_stable_fraction=2 / 3,
        position_score=0.30,
        receiver_reaction_evidence=True,
        receiver_reaction_time=44.0,
    )
    observations = [
        _ob(points[0], "right"),
        _ob(points[1], "left"),
        _ob(points[2], "right"),
        final,
    ]
    onsets = np.array([1, 11, 21, 46.0, 46.5, 47.0, 47.5, 48.0])

    kept, stage = validate_match_sequence(
        points, onsets, observations, RallyConfig(play_mode="auto"))

    assert len(kept) == 4
    assert stage["match_phases"] == [{"start_index": 0, "end_index": 3}]
    # The independently observed stable-to-active transition, not a later formation score
    # peak, is the inferred contact for a quiet far-side serve.
    assert stage["logical_groups"][-1]["serve_contact"] == 44.0


def test_confirmed_serve_absorbs_delayed_multi_impact_exchange_fragment():
    points = [(0.0, 4.0), (7.6, 12.0)]
    trailing_exchange = replace(
        _ob(points[1], None, serve=False, setup=False),
        target_court_filtered=True,
    )

    kept, stage = validate_match_sequence(
        points,
        np.array([1.0, 2.0, 8.0, 8.6, 9.2, 9.8]),
        [_ob(points[0], "right"), trailing_exchange],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [(0.0, 12.0)]
    assert stage["logical_groups"][0]["member_indices"] == [0, 1]


def test_fault_and_retry_with_pose_identity_switch_remain_one_point():
    points = [(0.0, 4.0), (12.5, 16.0)]
    observations = [
        replace(
            _ob(points[0], "right", serve=False, setup=False),
            first_strike=1.0, serve_motion=True, overhead_frames=3,
            overhead_strikes=(1.0,), target_court_filtered=True,
        ),
        replace(
            _ob(points[1], "left", serve=False, setup=False),
            first_strike=16.0, serve_motion=True, overhead_frames=3,
            overhead_strikes=(16.0,), target_court_filtered=True,
            side_confidence=0.2,
        ),
    ]

    kept, stage = validate_match_sequence(
        points, np.array([1.0, 16.0]), observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [(0.0, 16.0)]
    assert stage["logical_groups"][0]["member_indices"] == [0, 1]


def test_new_overhead_after_ordered_exchange_starts_next_point():
    points = [(0.0, 4.0), (9.5, 13.0)]
    first = replace(
        _ob(points[0], "right"), ball_ordered_evidence=True,
        target_court_filtered=True)
    second = replace(
        _ob(points[1], "right", serve=False, setup=False),
        first_strike=10.5, serve_motion=True, overhead_frames=3,
        overhead_strikes=(10.5,), target_court_filtered=True,
    )

    kept, _stage = validate_match_sequence(
        points, np.array([1.0, 2.0, 10.5]), [first, second],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [(0.0, 4.0), (9.5, 13.0)]


def test_weak_reaction_or_pass_inside_reset_gap_is_not_a_new_point():
    points = [(0.0, 5.0), (14.0, 19.0)]
    first = replace(
        _ob(points[0], "right"), target_court_filtered=True,
        ball_coverage=0.5)
    reset_pass = replace(
        _ob(points[1], None, serve=True, setup=False),
        target_court_filtered=True,
        receiver_reaction_evidence=True, receiver_reaction_time=15.0,
        ball_coverage=0.2,
    )

    kept, stage = validate_match_sequence(
        points, np.array([1.0, 2.0, 15.0]), [first, reset_pass],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [points[0]]
    assert stage["logical_groups"][1]["decision"] == "drop"


def test_lookback_reaction_cannot_inflate_subsecond_fragment_into_point():
    point = (20.0, 20.3)
    reaction = replace(
        _ob(point, None, serve=False, setup=False),
        target_court_filtered=True,
        receiver_reaction_evidence=True, receiver_reaction_time=18.0,
    )

    kept, stage = validate_match_sequence(
        [point], np.array([18.0]), [reaction],
        RallyConfig(play_mode="match"),
    )

    assert kept == []
    assert stage["dropped"][0]["reason_code"] == "missing_confirmed_serve"


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


def test_ball_serve_uses_calibrated_target_court_not_wide_image_rectangle():
    cfg = RallyConfig()
    court = Court.from_image_corners(
        (400, 900), (1400, 900), (1300, 300), (500, 300))
    # This track lies inside the legacy 12%-88% image rectangle but entirely to the
    # right of the selected court, as happens when an adjacent court is in frame.
    neighboring_court_track = _track(x_start=0.79, x_end=0.82)

    image_only = classify_ball_serve(
        (0, 2), np.array([1.0]), neighboring_court_track, 1920, 1080, cfg)
    calibrated = classify_ball_serve(
        (0, 2), np.array([1.0]), neighboring_court_track, 1920, 1080, cfg,
        court=court)

    assert image_only.confirmed is True
    assert calibrated.confirmed is False


def test_serve_validation_reuses_and_combines_cached_track_windows():
    first = BallTrack(
        np.arange(0.0, 1.1, 0.1), np.arange(11.0), np.arange(11.0))
    second = BallTrack(
        np.arange(1.0, 2.1, 0.1), np.arange(11.0), np.arange(11.0))

    cached = _cached_track_window(
        [((0.0, 1.1), first), ((1.0, 2.1), second)], 0.5, 1.5)

    assert cached is not None
    assert cached.t[0] <= 0.5 and cached.t[-1] >= 1.5
    assert np.all(np.diff(cached.t) > 0)


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

    plain = ServeSetupObservation(**base)
    aligned = ServeSetupObservation(
        **base, position_checked=True, position_setup_evidence=True,
        position_setup_strikes=(1.0,),
    )
    misaligned = ServeSetupObservation(
        **base, position_checked=True, position_setup_evidence=True,
        position_setup_strikes=(2.0,),
    )
    cfg = RallyConfig(play_mode="match", match_point_start_preroll_s=1.0)
    assert validate_match_sequence([point], np.array([1.0]), [plain], cfg)[0] == []
    assert validate_match_sequence([point], np.array([1.0]), [aligned], cfg)[0] == [point]
    assert validate_match_sequence([point], np.array([1.0]), [misaligned], cfg)[0] == []


def test_repeated_overhead_is_independent_only_on_target_court():
    point = (0.0, 3.0)
    base = dict(
        point=point, first_strike=1.0, side="left", side_confidence=1.0,
        near_x=0.2, near_x_std=0.01, sampled_frames=5, pose_frames=5,
        ready_frames=0, serve_motion=True, setup_evidence=True, observable=True,
        overhead_frames=2, overhead_strikes=(1.0,),
        ball_checked=True, ball_serve_evidence=False,
    )

    uncalibrated = ServeSetupObservation(**base)
    one_frame = replace(
        uncalibrated, target_court_filtered=True, overhead_frames=1)
    calibrated = replace(uncalibrated, target_court_filtered=True)
    cfg = RallyConfig(play_mode="match", match_point_start_preroll_s=1.0)
    assert validate_match_sequence([point], np.array([1.0]), [uncalibrated], cfg)[0] == []
    assert validate_match_sequence([point], np.array([1.0]), [one_frame], cfg)[0] == []
    assert validate_match_sequence([point], np.array([1.0]), [calibrated], cfg)[0] == [point]


def test_target_court_receiver_reaction_requires_stationary_setup():
    point = (10.0, 14.0)
    base = replace(
        _ob(point, "right", serve=False, setup=False),
        first_strike=11.0,
        receiver_reaction_evidence=True,
        receiver_reaction_time=11.0,
    )
    kept, _stage = validate_match_sequence(
        [point], np.array([11.0]),
        [replace(base, target_court_filtered=True)],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )
    assert kept == []

    corroborated = replace(
        base, target_court_filtered=True, position_checked=True,
        position_setup_evidence=True, position_best_strike=11.0,
        position_setup_strikes=(11.0,),
    )
    kept, stage = validate_match_sequence(
        [point], np.array([11.0]), [corroborated],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )
    assert kept == [point]
    assert "target_court_receiver_reaction" in (
        stage["observations"][0]["serve_evidence_sources"])

    kept, _stage = validate_match_sequence(
        [point], np.array([11.0]), [base],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )
    assert kept == []


def test_target_court_overhead_source_keeps_match_candidate_without_ball():
    point = (20.0, 24.0)
    observation = replace(
        _ob(point, "left", serve=False, setup=False),
        first_strike=21.0, serve_motion=True, overhead_frames=2,
        overhead_strikes=(21.0,), target_court_filtered=True,
    )

    kept, stage = validate_match_sequence(
        [point], np.array([21.0]), [observation],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [point]
    assert "robust_target_court_overhead_pose" in (
        stage["observations"][0]["serve_evidence_sources"])


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
    kept_group = next(
        group for group in stage["logical_groups"] if group["decision"] == "keep")
    assert kept_group["serve_member_index"] == 1


def test_match_decoder_keeps_earliest_independent_serve_in_fragment_chain():
    points = [(78.0, 82.0), (85.0, 90.0)]
    early_serve = replace(
        _ob(points[0], "right", serve=True, setup=False),
        first_strike=79.0, ball_best_strike=79.0,
        ball_coverage=0.4, ball_vertical_span=0.12,
        ball_outgoing_span=0.03, ball_ordered_evidence=True,
    )
    stronger_return = replace(
        _ob(points[1], "right", serve=True, setup=False),
        first_strike=86.0, serve_motion=True, overhead_strikes=(86.0,),
        ball_best_strike=86.0, ball_coverage=0.45, ball_vertical_span=0.12,
        ball_outgoing_span=0.08, ball_ordered_evidence=True,
    )

    kept, stage = validate_match_sequence(
        points, np.array([79.0, 80.0, 86.0, 88.0]),
        [early_serve, stronger_return],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [(78.0, 90.0)]
    assert stage["logical_groups"][0]["serve_member_index"] == 0


def test_match_decoder_prefers_much_stronger_pose_ball_serve_after_bounces():
    points = [(112.0, 117.0), (118.0, 125.0)]
    preparation_bounces = replace(
        _ob(points[0], "left"), first_strike=113.0,
        ball_coverage=0.5, ball_vertical_span=0.1,
        ball_outgoing_span=0.03, ball_ordered_evidence=True,
    )
    actual_serve = replace(
        _ob(points[1], "left"), first_strike=119.0,
        serve_motion=True, overhead_strikes=(119.0,),
        ball_coverage=0.85, ball_vertical_span=0.32,
        ball_outgoing_span=0.08, ball_ordered_evidence=True,
    )

    kept, stage = validate_match_sequence(
        points, np.array([113.0, 115.0, 116.0, 119.0, 121.0]),
        [preparation_bounces, actual_serve],
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(115.0, 125.0)]
    assert stage["logical_groups"][0]["serve_member_index"] == 1


def test_match_decoder_combines_stationary_setup_with_near_threshold_ball_motion():
    point = (240.0, 250.0)
    observation = replace(
        _ob(point, "right", serve=False, setup=False),
        first_strike=245.0,
        position_checked=True, position_setup_evidence=True,
        position_best_strike=245.0, position_setup_strikes=(245.0,),
        position_score=0.85,
        ball_coverage=0.15, ball_vertical_span=0.1,
        ball_outgoing_span=0.06, ball_ordered_evidence=False,
    )

    kept, _stage = validate_match_sequence(
        [point], np.array([245.0, 247.0, 248.0]), [observation],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [(244.0, 250.0)]


def test_match_decoder_retains_one_second_after_grouped_terminal_contact():
    points = [(106.0, 112.0), (113.0, 117.0)]
    observations = [
        replace(_ob(points[0], "left"), first_strike=110.0),
        replace(_ob(points[1], "left", serve=False), first_strike=116.0),
    ]

    kept, _stage = validate_match_sequence(
        points, np.array([110.0, 116.0]), observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(106.0, 117.0)]


def test_match_decoder_does_not_extend_for_unowned_single_audio_tail():
    points = [(106.0, 112.0), (113.0, 117.0)]
    observations = [
        replace(_ob(points[0], "left"), first_strike=110.0),
        replace(_ob(points[1], None, serve=False, setup=False), first_strike=116.0),
    ]

    kept, _stage = validate_match_sequence(
        points, np.array([110.0, 116.0]), observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
    )

    assert kept == [(106.0, 112.0)]


def test_confirmed_group_uses_accepted_trajectory_fragment_as_terminal_tail():
    points = [(106.0, 112.0), (113.0, 117.0)]
    observations = [
        replace(_ob(points[0], "left"), first_strike=110.0),
        replace(_ob(points[1], None, serve=False, setup=False), first_strike=116.0),
    ]

    kept, _stage = validate_match_sequence(
        points, np.array([110.0, 116.0]), observations,
        RallyConfig(play_mode="match", match_point_start_preroll_s=4.0),
        protected_indices={1},
    )

    assert kept == [(106.0, 117.0)]


def test_high_coverage_ball_endpoint_is_not_extended_by_outside_audio():
    point = (0.0, 4.0)
    observation = replace(
        _ob(point, "left"),
        ball_ordered_evidence=True,
        ball_coverage=0.8,
        ball_measured_samples=20,
    )

    kept, _stage = validate_match_sequence(
        [point], np.array([1.0, 2.0, 4.5, 5.0]), [observation],
        RallyConfig(play_mode="match", match_point_start_preroll_s=1.0),
    )

    assert kept == [point]


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
