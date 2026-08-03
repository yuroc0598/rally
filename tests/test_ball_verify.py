import numpy as np

from rally.fusion.ball_verify import (
    _audio_aligned_trajectory_end,
    _dedupe_non_overlapping,
    _group_tracking_windows,
    _net_crossings,
    rally_verdict,
    verify_segments_detailed,
)
from rally.signals.ball import (
    BallTrack,
    _target_court_heatmap_candidates,
    get_cached_ball_model,
    resolve_ball_batch_size,
)
from rally.signals.ball import ball_in_play_channel
from rally.signals.court import DOUBLES_W, NET_Y
from rally.signals.trajectory import SmoothTrack, smooth_track


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


def test_match_verdict_abstains_when_target_court_is_unavailable():
    v = rally_verdict(_rally_track(), None, 0.0, 4.0, require_court=True)

    assert v.state == "indeterminate"
    assert v.reason_code == "target_court_geometry_required"


def test_tracking_window_grouping_caps_transitive_overlap():
    candidates = [(0.0, 10.0), (9.0, 20.0), (19.0, 30.0)]
    groups = _group_tracking_windows(
        candidates, pre_pad_s=2.0, post_pad_s=2.0,
        max_extend_s=3.0, max_group_s=20.0)

    assert len(groups) > 1
    assert all(end - start <= 20.0 for start, end, _members in groups)
    assert [member for _start, _end, members in groups for member in members] == candidates


def test_union_tracking_plan_decodes_connected_padding_once(monkeypatch):
    monkeypatch.setenv("RALLY_TRACKNET_WINDOW_PLAN", "union")
    candidates = [(0.0, 10.0), (9.0, 20.0), (19.0, 30.0)]

    groups = _group_tracking_windows(
        candidates, pre_pad_s=2.0, post_pad_s=2.0,
        max_extend_s=3.0, max_group_s=20.0)

    assert groups == [(0.0, 33.0, candidates)]


def test_cuda_batch_size_scales_with_free_memory(monkeypatch):
    monkeypatch.delenv("RALLY_BALL_BATCH_SIZE", raising=False)

    class FakeCuda:
        @staticmethod
        def mem_get_info(_device):
            return 48 * 1024 ** 3, 80 * 1024 ** 3

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})
    device = type("Device", (), {"type": "cuda"})()

    assert resolve_ball_batch_size(device, 0, torch_module=fake_torch) == 16
    assert resolve_ball_batch_size(device, 7, torch_module=fake_torch) == 7


def test_ball_model_cache_keys_checkpoint_device_and_precision(monkeypatch, tmp_path):
    from rally.signals import ball

    checkpoint = tmp_path / "tracknet.pt"
    checkpoint.write_bytes(b"weights")
    loaded = []

    def fake_load(path, device=None):
        model = object()
        loaded.append((path, str(device), model))
        return model

    ball._BALL_MODEL_CACHE.clear()
    monkeypatch.setattr(ball, "load_ball_model", fake_load)

    first = get_cached_ball_model(str(checkpoint), device="cpu", half_precision=False)
    second = get_cached_ball_model(str(checkpoint), device="cpu", half_precision=False)

    assert first is second
    assert len(loaded) == 1


def test_tracknet_candidates_are_filtered_before_neighbor_court_association():
    candidates = [(5.0, 5.0), (50.0, 5.0), (5.0, 50.0)]

    kept = _target_court_heatmap_candidates(
        candidates, FakeCourt(), 1.0, 1.0,
        sideline_margin_m=1.0, baseline_margin_m=3.0,
    )

    assert kept == [(5.0, 5.0)]


def test_trajectory_end_hint_ignores_audio_after_outgoing_motion_ended():
    t = np.arange(0.0, 6.0, 0.1)
    # The first contact is inside live motion; the second shortly precedes outgoing
    # motion.  A later post-point transient occurs only after that component ended.
    components = [(5, 16), (30, 41)]
    hint = _audio_aligned_trajectory_end(
        components, t, np.array([1.0, 2.7, 4.4]), 0.0, 5.0,
        max_contact_to_flight_s=0.8, tail_s=1.0,
    )
    assert hint == 5.0  # second component ends at 4.0; 4.4 may not borrow it

    reset_hint = _audio_aligned_trajectory_end(
        [(5, 16), (40, 51)], t, np.array([1.0, 3.3]), 0.0, 5.0,
        max_contact_to_flight_s=0.8, tail_s=1.0,
    )
    assert reset_hint == 2.5  # both audio and ball went quiet before the new event


def test_stationary_measurement_is_indeterminate_without_ball_identity():
    # A stationary accepted heatmap candidate may be a line/shoe distractor while the
    # actual ball was missed. It is not calibrated negative evidence.
    t = np.arange(0.0, 4.0, 1 / 30.0)
    st = smooth_track(BallTrack(t, 100 + 0 * t, 200 + 0 * t))
    v = rally_verdict(st, FakeCourt(), 0.0, 4.0)
    assert not v.is_rally
    assert v.state == "indeterminate"
    assert v.reason_code == "no_moving_ball_identity_unproven"
    assert v.candidate_coverage > 0.9
    assert v.in_play_frac < 0.1


def test_verdict_abstains_on_short_structureless_blip():
    # a brief moving blob on one side, no net crossing, too short
    t = np.arange(0.0, 0.8, 1 / 30.0)
    x = 5.0 + 30.0 * t
    y = 5.0 + 0.0 * t                      # stays on the near side, no crossing/bounce
    st = smooth_track(BallTrack(t, x, y))
    v = rally_verdict(st, FakeCourt(), 0.0, 0.8, min_in_play_span_s=1.5)
    assert not v.is_rally
    assert v.state == "indeterminate"
    assert v.reason_code == "component_too_short"


def test_verdict_abstains_on_low_occupancy_structureless_motion():
    # A locally well-measured moving distractor cannot contradict a broad candidate when
    # it occupies only a small fraction of the candidate's evidence core.
    from rally.signals.trajectory import SmoothTrack

    t = np.arange(0.0, 20.0, 1 / 30.0)
    measured = (t >= 8.0) & (t <= 10.0)
    st = SmoothTrack(
        t=t,
        x=5.0 + 30.0 * t,
        y=np.full(t.size, 5.0),
        vx=np.full(t.size, 30.0),
        vy=np.zeros(t.size),
        confidence=np.ones(t.size),
        measured=measured,
    )
    v = rally_verdict(st, FakeCourt(), 0.0, 20.0)
    assert v.state == "indeterminate"
    assert v.reason_code == "live_component_diluted_by_proposal"


def test_verdict_still_rejects_sustained_structureless_motion():
    t = np.arange(0.0, 4.0, 1 / 30.0)
    st = smooth_track(BallTrack(t, 5.0 + 30.0 * t, np.full(t.size, 5.0)))
    v = rally_verdict(st, FakeCourt(), 0.0, 4.0)
    assert v.state == "reject"
    assert v.reason_code == "reliable_no_rally_structure"


def test_missing_court_structure_is_indeterminate_not_rejected():
    t = np.arange(0.0, 4.0, 1 / 30.0)
    st = smooth_track(BallTrack(t, 5.0 + 30.0 * t, 5.0 + 0.0 * t))
    v = rally_verdict(st, None, 0.0, 4.0)
    assert v.state == "indeterminate"
    assert v.reason_code == "court_unavailable_no_reliable_structure"


def test_fragmented_low_coverage_track_is_indeterminate():
    t = np.arange(0.0, 4.0, 1 / 30.0)
    x = 5.0 + 30.0 * t
    y = 5.0 + 0.0 * t
    missing = np.arange(t.size) % 10 != 0
    x[missing] = np.nan
    y[missing] = np.nan
    v = rally_verdict(smooth_track(BallTrack(t, x, y)), FakeCourt(), 0.0, 4.0)
    assert v.state == "indeterminate"
    assert v.reason_code in {
        "fragmented_live_track", "live_component_diluted_by_proposal",
    }
    assert v.candidate_coverage < 0.2


def test_strike_local_core_recovers_rally_inside_broad_proposal():
    t = np.arange(0.0, 20.0, 1 / 30.0)
    x = np.full(t.size, np.nan)
    y = np.full(t.size, np.nan)
    live = (t >= 8.0) & (t < 12.0)
    local_t = t[live] - 8.0
    x[live] = 5.0 + 2.0 * local_t
    y[live] = 11.885 + 9.0 * np.cos(2 * np.pi * local_t / 0.8)
    st = smooth_track(BallTrack(t, x, y))

    diluted = rally_verdict(st, FakeCourt(), 0.0, 20.0)
    assert diluted.state == "indeterminate"
    assert diluted.reason_code == "live_component_diluted_by_proposal"

    strikes = np.array([8.1, 8.9, 9.7, 10.5, 11.3])
    local = rally_verdict(st, FakeCourt(), 0.0, 20.0, serve_times=strikes)
    assert local.state == "accept"
    assert local.evidence_core[0] > 7.0
    assert local.evidence_core[1] < 13.0


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
    st = smooth_track(BallTrack(t, 5.0 + 0.2 * t, y))
    in_play = np.ones(t.size, bool)
    assert _net_crossings(st, FakeCourt(), in_play) >= 4


def test_neighboring_court_motion_cannot_supply_rally_structure():
    t = np.arange(0.0, 4.0, 1 / 30.0)
    y = NET_Y + 8.0 * np.cos(2 * np.pi * t / 0.8)
    # Identity court mapping puts this otherwise rally-shaped trajectory well right of
    # the target doubles court. Its apparent crossings and bounces must not validate it.
    st = smooth_track(BallTrack(t, np.full(t.size, DOUBLES_W + 5.0), y))
    in_play = np.ones(t.size, bool)
    assert _net_crossings(st, FakeCourt(), in_play) == 0

    verdict = rally_verdict(st, FakeCourt(), 0.0, 4.0)
    assert verdict.state == "reject"
    assert verdict.reason_code == "reliable_no_rally_structure"
    assert verdict.n_net_crossings == 0
    assert verdict.n_bounces == 0


def test_whole_video_ball_channel_filters_neighboring_court_motion():
    t = np.arange(0.0, 3.0, 0.1)
    track = BallTrack(
        t=t, x=np.full(t.size, DOUBLES_W + 5.0), y=NET_Y + np.sin(t * 8.0))
    timeline = np.arange(0.0, 3.0, 0.2)

    assert np.max(ball_in_play_channel(track, timeline)) > 0.0
    assert np.max(ball_in_play_channel(track, timeline, court=FakeCourt())) == 0.0


def test_fragmented_audio_aligned_neighboring_point_is_explicitly_rejected():
    t = np.arange(0.0, 4.0, 0.1)
    measured = np.zeros(t.size, bool)
    measured[5:10] = True
    measured[18:23] = True
    measured[30:35] = True
    track = SmoothTrack(
        t=t, x=np.full(t.size, DOUBLES_W + 8.0),
        y=NET_Y + np.sin(t * 8.0), vx=np.full(t.size, 40.0),
        vy=np.full(t.size, 40.0), confidence=np.ones(t.size), measured=measured,
    )
    verdict = rally_verdict(
        track, FakeCourt(), 0.0, 4.0,
        serve_times=np.array([0.6, 1.9, 3.1]),
    )
    assert verdict.state == "reject"
    assert verdict.reason_code == "audio_aligned_activity_outside_target_court"
    assert verdict.n_live_components == 3


def test_neighbor_track_abstains_when_target_court_has_independent_serve_evidence():
    t = np.arange(0.0, 4.0, 0.1)
    measured = np.zeros(t.size, bool)
    measured[5:10] = True
    measured[18:23] = True
    measured[30:35] = True
    track = SmoothTrack(
        t=t, x=np.full(t.size, DOUBLES_W + 8.0),
        y=NET_Y + np.sin(t * 8.0), vx=np.full(t.size, 40.0),
        vy=np.full(t.size, 40.0), confidence=np.ones(t.size), measured=measured,
    )
    verdict = rally_verdict(
        track, FakeCourt(), 0.0, 4.0,
        serve_times=np.array([0.6, 1.9, 3.1]),
        audio_strike_times=np.array([0.6, 1.9, 3.1]),
        target_serve_times=np.array([0.7]),
    )
    assert verdict.state == "indeterminate"
    assert verdict.reason_code == "fragmented_live_track"


def test_one_noisy_target_motion_sample_does_not_hide_neighbor_contradiction():
    t = np.arange(0.0, 4.0, 0.1)
    measured = np.zeros(t.size, bool)
    measured[5:10] = True
    measured[18:23] = True
    measured[30:35] = True
    track = SmoothTrack(
        t=t, x=np.full(t.size, DOUBLES_W + 8.0),
        y=NET_Y + np.sin(t * 8.0), vx=np.full(t.size, 40.0),
        vy=np.full(t.size, 40.0), confidence=np.ones(t.size), measured=measured,
    )
    verdict = rally_verdict(
        track, FakeCourt(), 0.0, 4.0,
        serve_times=np.array([0.6, 1.9, 3.1]),
        target_motion_times=np.array([2.0]),
    )
    assert verdict.reason_code == "audio_aligned_activity_outside_target_court"


def test_net_crossings_zero_without_court():
    st = _rally_track()
    assert _net_crossings(st, None, np.ones(st.t.size, bool)) == 0


def test_dedupe_non_overlapping():
    segs = [(0.0, 5.0), (4.0, 8.0), (10.0, 12.0)]
    out = _dedupe_non_overlapping(segs)
    assert out == [(0.0, 4.0), (4.0, 8.0), (10.0, 12.0)]   # previous end trimmed to next start


def test_verify_segments_keeps_rally_rejects_dead(monkeypatch):
    """Orchestrator keeps accepted and abstained candidates in the legacy API."""
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
    assert len(kept) == 2
    assert 8.0 <= kept[0][0] <= 12.0        # bounded near the live candidate's serve
    assert kept[1] == (100.0, 104.0)        # stationary identity is unproven -> fallback

    report = verify_segments_detailed(
        "dummy.mp4", [(10.0, 14.0)], court=FakeCourt(), model=object())
    assert report.candidates[0].peak_ball_speed_kmh is not None
    speed = report.as_dict()["candidates"][0]
    assert speed["peak_ball_speed_kmh"] > 0
    assert speed["ball_speed_estimate"]["uncertain"] is True
    assert speed["ball_speed_estimate"]["uncertainty_kmh"] > 0
    assert speed["ball_speed_estimate"]["sample_count"] >= 3
    assert "not full 3-D velocity" in speed["ball_speed_estimate"]["limitations"][0]


def test_detailed_report_separates_accepts_from_indeterminate_fallback(monkeypatch):
    """The report exposes accepted bounds; the legacy API keeps abstained proposals."""
    import rally.signals.ball as ball_mod
    from rally.fusion import ball_verify

    def sparse_track(video, model=None, start_s=0.0, end_s=None, **kw):
        t = np.arange(start_s, end_s, 1 / 30.0)
        x = 5.0 + 30.0 * (t - start_s)
        y = 5.0 + 0.0 * t
        missing = np.arange(t.size) % 10 != 0
        x[missing] = np.nan
        y[missing] = np.nan
        return BallTrack(t, x, y)

    monkeypatch.setattr(ball_mod, "track_tracknet", sparse_track)
    candidate = (10.0, 14.0)
    report = verify_segments_detailed(
        "dummy.mp4", [candidate], court=FakeCourt(), model=object())
    assert report.segments == []
    assert report.candidates[0].verdict.state == "indeterminate"
    assert report.as_dict()["counts"] == {
        "accept": 0, "reject": 0, "indeterminate": 1,
    }

    diagnostics = []
    kept = ball_verify.verify_segments(
        "dummy.mp4", [candidate], court=FakeCourt(), model=object(),
        diagnostics_out=diagnostics)
    assert kept == [candidate]
    assert diagnostics[0]["state"] == "indeterminate"
    assert diagnostics[0]["reason_code"] in {
        "fragmented_live_track", "live_component_diluted_by_proposal",
    }


def test_detailed_report_aggregates_tracker_performance(monkeypatch):
    import rally.signals.ball as ball_mod

    def measured_track(_video, start_s=0.0, end_s=None, metrics=None, **_kwargs):
        if metrics is not None:
            metrics.update({
                "wall_seconds": 2.5,
                "decoded_frames": 120,
                "sampled_frames": 60,
                "inferred_frames": 58,
                "pipeline_enabled": 1,
                "batch_histogram": {"8": 7, "2": 1},
            })
        t = np.arange(start_s, end_s, 1 / 30.0)
        return BallTrack(t, np.full(t.size, np.nan), np.full(t.size, np.nan))

    monkeypatch.setattr(ball_mod, "track_tracknet", measured_track)
    report = verify_segments_detailed(
        "dummy.mp4", [(10.0, 14.0)], court=FakeCourt(), model=object())
    performance = report.as_dict()["performance"]

    assert performance["execution_modes"] == ["pipelined"]
    assert performance["wall_seconds"] == 2.5
    assert performance["decoded_frames"] == 120
    assert performance["sampled_frames"] == 60
    assert performance["inferred_frames"] == 58
    assert performance["batch_histogram"] == {"2": 1, "8": 7}
    assert performance["physical_window_seconds"] >= performance["union_seconds"]
