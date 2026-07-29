"""Deterministic proposal-budget invariants (not real-video accuracy evidence)."""

from types import SimpleNamespace

import numpy as np

from rally.config import RallyConfig
from rally.pipeline import (
    _Channels,
    _bounded_arbiter_regions,
    _ball_arbiter,
    _apply_ball_end_hints,
    _coherent_audio_fallback,
    _derive_points,
    _indeterminate_audio_fallback,
    _merge_point_sources,
    _player_activity_proposal,
    _recover_player_hint_points,
    trim,
)


def test_stable_receiver_transition_opens_serve_hint():
    cfg = RallyConfig(analysis_fps=5.0, player_fps=2.0)
    px = np.full(40, np.nan)
    py = np.full(40, np.nan)
    sample_indices = np.arange(0, 30, 2)
    px[sample_indices] = 0.44
    py[sample_indices] = 0.69
    moving = sample_indices >= 16
    px[sample_indices[moving]] += np.arange(np.sum(moving)) * 0.015
    ch = _Channels(near_track=(px, py))

    activity = _player_activity_proposal(ch, cfg)

    assert activity[16:28].max() == 1.0
    assert ch.player_serve_hints.size == 1
    assert 2.0 <= ch.player_serve_hints[0] <= 3.0

    # Sustained motion without a preceding stable formation is ordinary movement, not a
    # reason to spend TrackNet work or manufacture a serve hint.
    px[sample_indices] = np.linspace(0.30, 0.70, sample_indices.size)
    moving_only = _Channels(near_track=(px, py))
    assert not np.any(_player_activity_proposal(moving_only, cfg))
    assert moving_only.player_serve_hints.size == 0


def test_fragmented_ball_structure_recovers_only_with_player_serve_hint():
    cfg = RallyConfig()
    verdict = SimpleNamespace(
        state="indeterminate", reason_code="fragmented_live_track",
        selected_component=(4.0, 7.0), in_play_span_s=2.0,
        measured_coverage=0.60, n_bounces=2, n_net_crossings=0,
    )
    report = SimpleNamespace(candidates=[SimpleNamespace(
        candidate=(2.0, 8.0), verdict=verdict)])
    ch = _Channels(player_serve_hints=np.array([3.0]))

    assert _recover_player_hint_points(report, ch, cfg) == [(2.0, 8.0)]

    ch.player_serve_hints = np.zeros(0)
    assert _recover_player_hint_points(report, ch, cfg) == []

    # A short fragmented tail with one measured bounce may reach match validation, where
    # the independently tracked serve event remains mandatory.
    verdict.in_play_span_s = 1.05
    verdict.n_bounces = 1
    ch.player_serve_hints = np.array([3.0])
    assert _recover_player_hint_points(report, ch, cfg) == [(2.0, 8.0)]


def test_short_video_keeps_complete_proposal_set():
    duration = 90.0
    cfg = RallyConfig(
        accuracy_mode=False,
        analysis_fps=1.0,
        arbiter_max_candidate_s=30.0,
        arbiter_min_total_s=120.0,
        arbiter_max_total_fraction=0.2,
    )
    ch = _Channels()
    selected, omitted = _bounded_arbiter_regions(
        [(0.0, duration)], np.ones(int(duration)), duration, cfg, ch
    )
    assert selected == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]
    assert omitted == []
    assert ch.stages["arbiter_proposal"]["tracked_seconds"] == duration


def test_audio_strength_wins_and_equal_evidence_spreads_over_time():
    duration = 300.0
    cfg = RallyConfig(
        accuracy_mode=False,
        analysis_fps=1.0,
        arbiter_max_candidate_s=20.0,
        arbiter_min_total_s=50.0,
        arbiter_max_total_fraction=0.0,
        arbiter_max_total_s=50.0,
        arbiter_pre_pad_s=0.0,
        arbiter_post_pad_s=0.0,
        ball_max_extend_s=0.0,
    )
    rate = np.zeros(int(duration))
    regularity = np.zeros(int(duration))
    rate[280:300] = 1.0
    regularity[280:300] = 1.0
    ch = _Channels(
        audio_rate=rate,
        audio_reg=regularity,
        onsets=np.array([282.0, 284.0, 286.0]),
    )
    regions = [(0.0, 20.0), (140.0, 160.0), (280.0, 300.0)]
    selected, omitted = _bounded_arbiter_regions(
        regions, np.full(int(duration), 0.65), duration, cfg, ch
    )
    assert (280.0, 300.0) in selected  # strongest coherent audio, despite being last
    assert (0.0, 20.0) in selected     # farthest tied region, not next-in-input-order
    assert omitted == [(140.0, 160.0)]


def test_throughput_mode_omitted_regions_retain_only_coherent_audio_points():
    cfg = RallyConfig(accuracy_mode=False)
    ch = _Channels(onsets=np.array([105.0, 106.2, 107.4, 205.0]))
    fallback = _coherent_audio_fallback(
        [(100.0, 120.0), (200.0, 210.0)], ch, 300.0, cfg
    )
    assert len(fallback) == 1
    assert fallback[0][0] <= 105.0 < fallback[0][1]
    assert not any(start <= 205.0 <= end for start, end in fallback)

    combined = _merge_point_sources([(104.5, 106.0)], fallback)
    # Fallback inputs have already survived whole-point ownership adjudication. A partial
    # accepted interval may not erase or truncate one merely because their padding overlaps.
    assert combined == fallback

    # A computational chunk edge is not a point boundary: two hits on opposite sides
    # must remain one coherent fallback point.
    boundary_ch = _Channels(onsets=np.array([44.0, 46.0]))
    across_boundary = _coherent_audio_fallback(
        [(0.0, 45.0), (45.0, 90.0)], boundary_ch, 90.0, cfg
    )
    assert len(across_boundary) == 1


def test_audio_fallback_does_not_attach_unverified_isolated_prepoint_sound():
    # The isolated 38.649 transient is separated from the coherent exchange by more
    # than point_gap_s. Without court/trajectory proof it must not pull walking/reset
    # footage into the point as a guessed serve.
    cfg = RallyConfig(accuracy_mode=False)
    assert cfg.serve_attach is False
    ch = _Channels(onsets=np.array([38.649, 42.044, 42.933, 45.075]))
    assert _coherent_audio_fallback([(37.0, 47.0)], ch, 60.0, cfg) == [
        (41.044, 46.075)
    ]


def test_accuracy_mode_keeps_all_proposals_and_short_audio_hypotheses():
    duration = 300.0
    cfg = RallyConfig(
        accuracy_mode=True, analysis_fps=1.0, arbiter_max_candidate_s=20.0,
        arbiter_min_total_s=20.0, arbiter_max_total_fraction=0.01,
        arbiter_max_total_s=20.0, arbiter_pre_pad_s=0.0,
        arbiter_post_pad_s=0.0, ball_max_extend_s=0.0,
    )
    ch = _Channels(onsets=np.array([33.006, 33.422, 205.0]))
    selected, omitted = _bounded_arbiter_regions(
        [(20.0, 40.0), (190.0, 210.0)], np.ones(int(duration)),
        duration, cfg, ch,
    )
    assert selected == [(20.0, 40.0), (190.0, 210.0)]
    assert omitted == []
    assert _coherent_audio_fallback(selected, ch, duration, cfg) == [
        (32.006, 34.422), (204.0, 206.0),
    ]


def test_ball_end_hint_trims_pickup_tail_but_not_a_fragmented_rally():
    cfg = RallyConfig(ball_end_hint_max_uncalibrated_trim_s=5.0)
    ch = _Channels(
        onsets=np.array([11.0, 12.0, 21.0, 30.0, 38.0]),
        ball_end_hints=[((10.0, 16.0), 13.0), ((20.0, 40.0), 25.0)],
    )

    assert _apply_ball_end_hints([(10.0, 16.0), (20.0, 40.0)], ch, cfg) == [
        (10.0, 13.0),
        (20.0, 40.0),
    ]
    assert ch.stages["ball_end_hints"]["changed"] == 1

    calibrated = _Channels(
        court=object(), onsets=np.array([21.0, 30.0, 38.0]),
        ball_end_hints=[((20.0, 40.0), 25.0)],
    )
    assert _apply_ball_end_hints([(20.0, 40.0)], calibrated, cfg) == [(20.0, 25.0)]


def test_budget_boundary_does_not_split_a_whole_coherent_point(monkeypatch):
    cfg = RallyConfig(analysis_fps=1.0)
    ch = _Channels(
        audio_rate=np.ones(10), audio_reg=np.ones(10),
        onsets=np.array([4.5, 5.5]), used=["audio"],
    )
    monkeypatch.setattr("rally.pipeline.segments_from_prob", lambda *a, **k: [(0.0, 10.0)])

    def split_budget(regions, evidence, duration, cfg, channels):
        channels.stages["arbiter_proposal"] = {
            "status": "capped", "selected_regions": 1, "omitted_regions": 1}
        return [(0.0, 5.0)], [(5.0, 10.0)]

    monkeypatch.setattr("rally.pipeline._bounded_arbiter_regions", split_budget)
    _derive_points(ch, 10.0, cfg, lambda _m: None, for_ball_arbiter=True)
    assert ch.arbiter_selected_audio_fallback == []
    assert ch.arbiter_audio_fallback == [(3.5, 6.5)]


def test_selected_audio_fallback_uses_abstentions_but_not_explicit_rejects():
    ch = _Channels(
        onsets=np.array([11.0, 12.0, 31.0, 32.0, 51.0, 53.0, 71.0, 72.0]),
        arbiter_selected_audio_fallback=[
            (10.0, 14.0), (30.0, 34.0), (50.0, 54.0), (70.0, 74.0)],
        arbiter_accepted_regions=[(69.0, 75.0)],
        arbiter_indeterminate_regions=[(9.0, 15.0)],
        arbiter_rejected_regions=[(29.0, 35.0), (49.0, 52.0)],
    )
    fallback, suppressed, superseded = _indeterminate_audio_fallback(ch)
    assert fallback == [(10.0, 14.0), (50.0, 54.0)]
    assert suppressed == 1       # only the point wholly covered by a reject
    assert superseded == 1       # accepted ball bounds replace its audio bounds


def test_partial_accept_cannot_truncate_a_boundary_split_whole_point():
    point = (43.0, 51.2)
    ch = _Channels(
        onsets=np.array([44.0, 46.0, 48.0, 50.0]),
        arbiter_selected_audio_fallback=[point],
        arbiter_accepted_regions=[(0.0, 45.0)],
        arbiter_indeterminate_regions=[(45.0, 90.0)],
    )
    fallback, suppressed, superseded = _indeterminate_audio_fallback(ch)
    assert fallback == [point]
    assert suppressed == 0
    assert superseded == 0

    # The accepted trajectory covers only the first fragment. Final merging must retain
    # the complete coherent point, not replace it with the truncated TrackNet interval.
    assert _merge_point_sources([(43.5, 47.0)], fallback) == [point]


def test_ball_arbiter_consumes_tri_state_verification_report(monkeypatch, tmp_path):
    from rally.fusion import ball_verify
    from rally.signals import ball

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"identity-for-provenance")
    candidates = [(10.0, 14.0), (30.0, 34.0), (50.0, 54.0)]
    entries = [
        SimpleNamespace(candidate=candidates[0], verdict=SimpleNamespace(
            state="accept", evidence_core=None, selected_component=None)),
        SimpleNamespace(candidate=candidates[1], verdict=SimpleNamespace(
            state="indeterminate", evidence_core=None, selected_component=None)),
        SimpleNamespace(candidate=candidates[2], verdict=SimpleNamespace(
            state="reject", evidence_core=(51.0, 53.0),
            selected_component=(50.5, 53.5))),
    ]
    report = SimpleNamespace(
        segments=[(10.5, 13.5)], candidates=entries,
        as_dict=lambda: {"counts": {"accept": 1, "indeterminate": 1, "reject": 1}},
    )
    monkeypatch.setattr(ball_verify, "verify_segments_detailed", lambda *a, **k: report)
    monkeypatch.setattr(ball, "resolve_device", lambda: "cpu")
    monkeypatch.setattr("rally.pipeline._resolve_court", lambda *a, **k: None)

    ch = _Channels(onsets=np.zeros(0))
    got = _ball_arbiter(
        "unused.mp4", candidates, ch, RallyConfig(), lambda _m: None,
        weights=str(weights),
    )
    assert got == [(10.5, 13.5)]
    assert ch.arbiter_accepted_regions == [(10.0, 14.0)]
    assert ch.arbiter_indeterminate_regions == [(30.0, 34.0)]
    assert ch.arbiter_rejected_regions == [(51.0, 53.0)]
    assert ch.stages["ball_arbiter"]["verification"]["counts"]["reject"] == 1


def test_degraded_arbiter_failure_never_publishes_broad_proposals(monkeypatch, tmp_path):
    from rally.fusion import ball_verify
    from rally.signals import ball

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"identity-for-provenance")
    candidates = [(0.0, 45.0), (45.0, 90.0)]
    point = (43.0, 51.2)

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError("synthetic tracker failure")

    monkeypatch.setattr(ball_verify, "verify_segments_detailed", fail_verification)
    monkeypatch.setattr(ball, "resolve_device", lambda: "cpu")
    monkeypatch.setattr("rally.pipeline._resolve_court", lambda *a, **k: None)
    ch = _Channels(
        onsets=np.array([44.0, 46.0, 48.0, 50.0]),
        arbiter_selected_audio_fallback=[point],
    )

    primary = _ball_arbiter(
        "unused.mp4", candidates, ch, RallyConfig(allow_degraded=True),
        lambda _m: None, weights=str(weights),
    )
    assert primary == []
    assert ch.arbiter_indeterminate_regions == candidates
    fallback, suppressed, superseded = _indeterminate_audio_fallback(ch)
    assert fallback == [point]
    assert _merge_point_sources(primary, fallback) == [point]
    assert (ch.stages["ball_arbiter"]["fallback_policy"]
            == "coherent_audio_only")


def test_ball_mode_skips_preverification_serve_anchoring(monkeypatch, tmp_path):
    from rally.signals import ball

    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "rally.pipeline.probe",
        lambda _p: SimpleNamespace(duration_s=30.0, fps=30.0, width=640,
                                   height=360, has_audio=True),
    )

    def audio_stage(_path, _info, _timeline, _cfg, ch, _progress, _cancel_check):
        ch.used.append("audio")
        ch.onsets = np.array([11.0, 12.0])

    monkeypatch.setattr("rally.pipeline._audio_channel", audio_stage)
    monkeypatch.setattr("rally.pipeline._visual_channels", lambda *a, **k: None)
    monkeypatch.setattr(ball, "discover_ball_weights", lambda: "weights.pt")
    monkeypatch.setattr("rally.pipeline._derive_points", lambda *a, **k: [(10.0, 14.0)])
    monkeypatch.setattr(
        "rally.pipeline._anchor_serves",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not anchor before budgeted ball tracking")),
    )
    monkeypatch.setattr("rally.pipeline._ball_arbiter", lambda *a, **k: [])
    monkeypatch.setattr("rally.pipeline._write_output", lambda *a, **k: None)

    result = trim(str(source), cfg=RallyConfig(ball_arbiter=True))
    assert result.stages["serve_anchor"]["status"] == "skipped"
