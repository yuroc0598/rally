"""Ball-as-arbiter: let the ball trajectory *decide* which candidate windows are real
rallies and where each one starts and ends.

This is the SwingVision-style inversion of the pipeline. The cheap channels (audio /
motion / geometry) are recall-oriented — they over-propose candidate windows, including
warm-up cooperative hitting, adjacent-court noise, and crowd transients. The ball tracker
is expensive but *discriminative*: during a real point the ball is continuously in play,
crosses the net, and bounces; between/around points it is dead. So we run the tracker only
inside each candidate (CPU-affordable), reconstruct the trajectory
(:mod:`rally.signals.trajectory`), and apply a verdict:

* **is it a rally?** — the ball must be in play for a meaningful span AND show rally
  structure (a net crossing and/or bounces). This rejects many false positives that pure
  audio cannot. Warm-up and match play remain only partially distinguishable; optional
  strike-plus-baseline start evidence is a precision gate, not a learned serve classifier.
* **where does it start?** — snap to the serve: the first sustained in-play instant, minus
  a short toss pre-roll.
* **where does it end?** — the first point-ending event (double bounce on one side / ball
  out), via :mod:`rally.signals.ballrules`, else the last live-ball instant plus a tail.

The verdict logic (:func:`rally_verdict`) is pure — it takes a reconstructed
:class:`~rally.signals.trajectory.SmoothTrack` (+ optional court homography) and returns a
:class:`RallyVerdict` — so it is unit-testable on synthetic trajectories with no video,
model, or weights. :func:`verify_segments` is the orchestrator that does the tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np

from ..signals.ballrules import ball_speed_kmh, is_in, point_end_events, refine_end_from_events
from ..signals.court import COURT_L, DOUBLES_W, NET_Y
from ..signals.trajectory import SmoothTrack, bounces_from_velocity, smooth_track
from .intervals import trim_previous_on_overlap

Segment = Tuple[float, float]
VerdictState = Literal["accept", "reject", "indeterminate"]

BALL_SPEED_LIMITATIONS = (
    "single-camera homography measures court-plane displacement, not full 3-D velocity",
    "ball height is not recovered, so airborne speed is generally underestimated",
    "court calibration and ball-tracking error can inflate or suppress the estimate",
)


def _ground_plane_speed_estimate(
    values: np.ndarray, measured_fraction: float,
) -> Optional[dict[str, Any]]:
    """Summarise a robust peak with an honest heuristic error scale.

    This is deliberately not labelled a confidence interval: the dominant missing-height
    error is systematic and cannot be inferred from a single camera. The 30% geometry
    floor is inflated for sparse trajectory coverage and small samples.
    """
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    value = float(np.percentile(values, 95))
    coverage = float(np.clip(measured_fraction, 0.0, 1.0))
    relative_error = (
        0.30
        + 0.20 * (1.0 - coverage)
        + 0.15 * min(1.0, 8.0 / float(values.size))
    )
    return {
        "value_kmh": round(value, 1),
        "uncertainty_kmh": round(value * relative_error, 1),
        "uncertain": True,
        "method": "single_camera_ground_plane_p95",
        "sample_count": int(values.size),
        "measured_fraction": round(coverage, 4),
        "uncertainty_basis": (
            "heuristic error scale from a 30% single-camera geometry floor, "
            "inflated for sparse trajectory evidence; not a confidence interval"
        ),
        "limitations": list(BALL_SPEED_LIMITATIONS),
    }


@dataclass
class RallyVerdict:
    is_rally: bool
    start: float
    end: float
    n_bounces: int = 0
    n_net_crossings: int = 0
    in_play_frac: float = 0.0
    in_play_span_s: float = 0.0
    reason: str = ""
    measured_coverage: float = 0.0
    state: Optional[VerdictState] = None
    reason_code: str = ""
    candidate_coverage: float = 0.0
    n_live_components: int = 0
    selected_component: Optional[Segment] = None
    evidence_core: Optional[Segment] = None
    strike_aligned: bool = False
    court_available: bool = False
    trajectory_end_hint: Optional[float] = None

    def __post_init__(self) -> None:
        # Keep older callers that construct/read ``is_rally`` working while making the
        # third state explicit to diagnostic consumers.
        if self.state is None:
            self.state = "accept" if self.is_rally else "reject"
        self.is_rally = self.state == "accept"

    @property
    def is_indeterminate(self) -> bool:
        return self.state == "indeterminate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "n_bounces": self.n_bounces,
            "n_net_crossings": self.n_net_crossings,
            "in_play_frac": round(float(self.in_play_frac), 4),
            "in_play_span_s": round(float(self.in_play_span_s), 3),
            "measured_coverage": round(float(self.measured_coverage), 4),
            "candidate_coverage": round(float(self.candidate_coverage), 4),
            "n_live_components": self.n_live_components,
            "selected_component": (list(self.selected_component)
                                   if self.selected_component is not None else None),
            "evidence_core": (list(self.evidence_core)
                              if self.evidence_core is not None else None),
            "strike_aligned": self.strike_aligned,
            "court_available": self.court_available,
            "trajectory_end_hint": (round(float(self.trajectory_end_hint), 3)
                                    if self.trajectory_end_hint is not None else None),
        }


@dataclass
class CandidateVerification:
    candidate: Segment
    tracked_window: Segment
    verdict: RallyVerdict
    output: Optional[Segment] = None
    peak_ball_speed_kmh: Optional[float] = None
    ball_speed_estimate: Optional[dict[str, Any]] = None

    def as_dict(self) -> dict[str, Any]:
        data = self.verdict.as_dict()
        data.update({
            "candidate": [round(float(v), 3) for v in self.candidate],
            "tracked_window": [round(float(v), 3) for v in self.tracked_window],
            "output": ([round(float(v), 3) for v in self.output]
                       if self.output is not None else None),
            "peak_ball_speed_kmh": (round(float(self.peak_ball_speed_kmh), 1)
                                    if self.peak_ball_speed_kmh is not None else None),
            "ball_speed_estimate": self.ball_speed_estimate,
        })
        return data


@dataclass
class VerificationReport:
    segments: List[Segment]
    candidates: List[CandidateVerification]
    # Ephemeral raw tracks are intentionally excluded from ``as_dict``. They are reused by
    # serve validation in the same pipeline run, avoiding a second TrackNet decode.
    track_cache: List[Tuple[Segment, Any]] = field(default_factory=list, repr=False)
    inference_batch_size: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        counts = {state: sum(c.verdict.state == state for c in self.candidates)
                  for state in ("accept", "reject", "indeterminate")}
        return {
            "segments": [[round(float(s), 3), round(float(e), 3)]
                         for s, e in self.segments],
            "counts": counts,
            "inference_batch_size": self.inference_batch_size,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _group_tracking_windows(
    segments: List[Segment], pre_pad_s: float, post_pad_s: float,
    max_extend_s: float, max_group_s: float,
) -> List[Tuple[float, float, List[Segment]]]:
    """Merge padding overlap without allowing one transitive near-full-video group."""
    grouped: List[Tuple[float, float, List[Segment]]] = []
    tail = max(post_pad_s, max_extend_s)
    for s, e in sorted(segments):
        ts, te = max(0.0, s - pre_pad_s), e + tail
        if (grouped and ts <= grouped[-1][1]
                and max(grouped[-1][1], te) - grouped[-1][0] <= max_group_s):
            gs, ge, members = grouped[-1]
            grouped[-1] = (gs, max(ge, te), [*members, (s, e)])
        else:
            grouped.append((ts, te, [(s, e)]))
    return grouped


def _verdict(state: VerdictState, start: float, end: float, *,
             reason_code: str, reason: str, **metrics: Any) -> RallyVerdict:
    return RallyVerdict(state == "accept", start, end, state=state,
                        reason_code=reason_code, reason=reason, **metrics)


def _in_play_mask(track: SmoothTrack, min_speed_px_s: float, min_conf: float,
                  max_fill_gap_s: float = 0.20) -> np.ndarray:
    """Measured live-ball samples plus short, confident gaps between them.

    Smoothed predictions are not independently accepted as ball evidence.  They may only
    bridge two measured moving samples when every bridged sample stays confident and the
    entire hole is short.  The verdict applies a second, component-level measured-coverage
    guard before accepting the result.
    """
    speed = np.hypot(track.vx, track.vy)
    confident = track.confidence >= min_conf
    live = track.measured & confident & (speed >= min_speed_px_s)
    t = np.asarray(track.t, float)
    i = 0
    while i < live.size:
        if live[i]:
            i += 1
            continue
        j = i
        while j < live.size and not live[j]:
            j += 1
        if (i > 0 and j < live.size and live[i - 1] and live[j]
                and t[j] - t[i - 1] <= max_fill_gap_s
                and np.all(confident[i:j])):
            live[i:j] = True
        i = j
    return live


def _live_components(mask: np.ndarray, t: np.ndarray,
                     max_sample_gap_s: float = 0.20) -> List[Tuple[int, int]]:
    """Return ``[start, stop)`` runs, splitting even adjacent samples across time gaps."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero((np.diff(idx) > 1)
                            | (np.diff(np.asarray(t, float)[idx]) > max_sample_gap_s))
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, idx.size]
    return [(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(starts, stops)]


def _join_live_components(
    components: List[Tuple[int, int]],
    t: np.ndarray,
    max_gap_s: float,
) -> List[Tuple[int, int]]:
    """Join measured live fragments separated by a plausible short occlusion.

    The joined interval still uses the original sparse live mask, so this does not invent
    ball samples.  It only lets the verdict evaluate net/bounce/temporal structure across a
    dropout instead of treating frame-contiguous detection as a prerequisite for a point.
    """
    if not components:
        return []
    t = np.asarray(t, float)
    joined: List[Tuple[int, int]] = [components[0]]
    for start, stop in components[1:]:
        prior_start, prior_stop = joined[-1]
        gap = float(t[start] - t[prior_stop - 1])
        if gap <= max_gap_s:
            joined[-1] = (prior_start, stop)
        else:
            joined.append((start, stop))
    return joined


def _audio_aligned_trajectory_end(
    components: List[Tuple[int, int]],
    t: np.ndarray,
    strikes: Optional[np.ndarray],
    win_start: float,
    win_end: float,
    *,
    max_contact_to_flight_s: float = 0.8,
    max_audio_bridge_s: float = 2.0,
    max_ball_bridge_s: float = 2.0,
    tail_s: float = 1.0,
) -> Optional[float]:
    """End hint from the last impact followed by measured outgoing ball motion.

    Post-point bounces, pickups, and speech can remain in the audio cluster.  A plausible
    racket contact is inside a measured live component or shortly precedes its start; an
    impact occurring only *after* a component ended is not allowed to borrow that motion.
    """
    if strikes is None or not components:
        return None
    t = np.asarray(t, float)
    ordered = np.sort(np.asarray(strikes, float))
    ordered = ordered[(ordered >= win_start - 1e-9) & (ordered <= win_end + 1e-9)]
    aligned: list[tuple[float, float, float]] = []
    for strike in ordered:
        matches: list[tuple[float, float, float]] = []
        for start, stop in components:
            live_start = float(t[start])
            live_end = float(t[stop - 1])
            if live_start - max_contact_to_flight_s <= strike <= live_end:
                distance = max(0.0, live_start - float(strike))
                matches.append((distance, live_start, live_end))
        if matches:
            _distance, live_start, live_end = min(matches)
            aligned.append((float(strike), live_start, live_end))
    if not aligned:
        return None

    # Follow the first coherent contact/flight chain.  A long ball dropout is allowed when
    # intervening racket contacts bridge it (a TrackNet miss); a later pickup/bounce begins
    # a new chain when both audio and ball evidence have gone quiet.
    last_strike, _last_start, last_end = aligned[0]
    for strike, live_start, live_end in aligned[1:]:
        between = ordered[(ordered >= last_strike - 1e-9)
                          & (ordered <= strike + 1e-9)]
        max_audio_gap = float(np.max(np.diff(between))) if between.size >= 2 else float("inf")
        ball_gap = max(0.0, live_start - last_end)
        if max_audio_gap > max_audio_bridge_s and ball_gap > max_ball_bridge_s:
            break
        last_strike = strike
        last_end = max(last_end, live_end)
    return float(last_end + tail_s)


def _net_crossings(track: SmoothTrack, court, in_play: np.ndarray,
                   dead_band_m: float = 0.5,
                   max_gap_s: float = 0.35,
                   court_margin_m: float = 0.35) -> int:
    """Count times the ball crosses the net, in court metres.

    The ball's court-y is compared to the net line with a dead-band so jitter around the
    net doesn't inflate the count; only a clean side-to-side transition among in-play
    samples is a crossing. Returns 0 without a court (can't map to metres).
    """
    if court is None:
        return 0
    idx = np.where(in_play)[0]
    if idx.size < 2:
        return 0
    pts = np.stack([track.x[idx], track.y[idx]], axis=1)
    court_pts = court.to_court(pts)
    side = 0            # -1 near, +1 far, 0 unknown (inside dead-band)
    side_time = None
    crossings = 0
    for sample_i, (court_x, court_y) in zip(idx, court_pts):
        sample_t = float(track.t[sample_i])
        # A neighboring-court trajectory can cross target net-y after homography
        # projection. It is rally structure only while inside the target doubles court.
        if not is_in(float(court_x), float(court_y), court_margin_m, singles=False):
            side = 0
            side_time = None
            continue
        s = (-1 if court_y < NET_Y - dead_band_m
             else (1 if court_y > NET_Y + dead_band_m else 0))
        if s == 0:
            continue
        if side_time is not None and sample_t - side_time > max_gap_s:
            side = 0
        if side != 0 and s != side:
            crossings += 1
        side = s
        side_time = sample_t
    return crossings


def _strike_local_cores(serve_times: Optional[np.ndarray], win_start: float,
                        win_end: float, *, pre_s: float, post_s: float,
                        cluster_gap_s: float = 2.5) -> List[Segment]:
    """Evidence-local cores from nearby impact clusters inside a broad proposal."""
    if serve_times is None:
        return []
    strikes = np.asarray(serve_times, float)
    strikes = np.unique(np.sort(strikes[np.isfinite(strikes)]))
    strikes = strikes[(strikes >= win_start - pre_s) & (strikes <= win_end + post_s)]
    if not strikes.size:
        return []
    split = np.flatnonzero(np.diff(strikes) > cluster_gap_s) + 1
    cores: List[Segment] = []
    for cluster in np.split(strikes, split):
        start = max(win_start, float(cluster[0]) - pre_s)
        end = min(win_end, float(cluster[-1]) + post_s)
        if end > start:
            cores.append((start, end))
    return cores


def _off_target_audio_contradiction(
    track: SmoothTrack, court, live: np.ndarray, core: np.ndarray,
    strikes: Optional[np.ndarray], *, court_margin_m: float,
    target_serve_times: Optional[np.ndarray] = None,
    target_motion_times: Optional[np.ndarray] = None,
    min_live_fraction: float = 0.20, min_aligned_strikes: int = 2,
    max_strike_distance_s: float = 0.80,
) -> bool:
    """Whether coherent measured activity belongs clearly to another court.

    Merely missing the target ball is never negative evidence. This stronger condition
    requires substantial moving-ball coverage, at least two audio impacts aligned to that
    motion, and at least 90% of the motion to be several metres outside the target court.
    It identifies the common multi-court failure where TrackNet and audio both follow the
    neighboring point, rather than treating an ordinary out ball as a contradiction.
    """
    if court is None or strikes is None:
        return False
    core_times = np.asarray(track.t, float)[np.asarray(core, bool)]
    if not core_times.size:
        return False
    core_start, core_end = float(core_times[0]), float(core_times[-1])

    # TrackNet emits one trajectory, not one trajectory per court. In multi-court footage
    # it can lock onto a well-lit neighboring ball while missing the target ball. That is
    # a contradiction only when the calibrated target court is independently quiet.
    # Player-derived serve/reaction hints are sparse events; target-masked frame motion is
    # continuous support and therefore needs at least two samples before it can veto.
    if target_serve_times is not None:
        target_serves = np.asarray(target_serve_times, float)
        target_serves = target_serves[np.isfinite(target_serves)]
        if np.any((target_serves >= core_start - max_strike_distance_s)
                  & (target_serves <= core_end + max_strike_distance_s)):
            return False
    if target_motion_times is not None:
        target_motion = np.asarray(target_motion_times, float)
        target_motion = target_motion[np.isfinite(target_motion)]
        supported = ((target_motion >= core_start)
                     & (target_motion <= core_end))
        if int(np.sum(supported)) >= 2:
            return False
    live_idx = np.flatnonzero(np.asarray(live, bool) & np.asarray(core, bool))
    core_count = int(np.sum(core))
    if live_idx.size < 2 or live_idx.size / max(core_count, 1) < min_live_fraction:
        return False
    court_pts = court.to_court(np.stack([
        np.asarray(track.x)[live_idx], np.asarray(track.y)[live_idx],
    ], axis=1))
    # Two metres is deliberately much wider than line-call tolerance: a normal out ball
    # must not turn an uncertain target-court track into an explicit rejection.
    clear_margin = max(2.0, float(court_margin_m))
    clearly_outside = np.array([
        not is_in(float(x), float(y), clear_margin, singles=False)
        for x, y in court_pts
    ], dtype=bool)
    if float(np.mean(clearly_outside)) < 0.90:
        return False
    outside_times = np.asarray(track.t, float)[live_idx[clearly_outside]]
    candidate_strikes = np.asarray(strikes, float)
    candidate_strikes = candidate_strikes[np.isfinite(candidate_strikes)]
    aligned = sum(
        bool(np.any(np.abs(outside_times - strike) <= max_strike_distance_s))
        for strike in candidate_strikes
    )
    return aligned >= min_aligned_strikes


def _component_verdict(
    track: SmoothTrack, court, win_start: float, win_end: float,
    component: Tuple[int, int], evidence_core: Segment, all_live: np.ndarray, *,
    candidate_coverage: float, n_live_components: int,
    min_conf: float, min_in_play_frac: float, min_in_play_span_s: float,
    min_measured_coverage: float, min_bounces: int, min_rally_s: float,
    toss_preroll_s: float, tail_s: float, max_extend_s: float,
    bounce_min_descent_px_s: float, double_bounce_window_s: float, margin_m: float,
    serve_times: Optional[np.ndarray], require_serve_evidence: bool,
    serve_lag_s: float, serve_baseline_margin_m: float,
) -> RallyVerdict:
    t = np.asarray(track.t, float)
    a, b = component
    in_play = np.zeros_like(all_live)
    in_play[a:b] = all_live[a:b]
    live_idx = np.flatnonzero(in_play)
    core = (t >= evidence_core[0]) & (t <= evidence_core[1])
    core_count = max(1, int(core.sum()))
    in_play_frac = float(np.sum(in_play & core) / core_count)
    selected = (float(t[live_idx[0]]), float(t[live_idx[-1]]))
    span = selected[1] - selected[0]
    measured_coverage = float(np.mean(track.measured[live_idx]))
    common = dict(in_play_frac=in_play_frac, in_play_span_s=span,
                  measured_coverage=measured_coverage,
                  candidate_coverage=candidate_coverage,
                  n_live_components=n_live_components, selected_component=selected,
                  evidence_core=evidence_core, court_available=court is not None)

    if live_idx.size < 2 or measured_coverage < min_measured_coverage:
        return _verdict(
            "indeterminate", win_start, win_end,
            reason_code="insufficient_component_coverage",
            reason="live component has too little measured TrackNet support", **common)

    bounce_idx = bounces_from_velocity(track, min_descent_px_s=bounce_min_descent_px_s,
                                       min_conf=min_conf)
    bounce_idx = [i for i in bounce_idx if in_play[i]]
    structure_bounce_idx = bounce_idx
    if court is not None and bounce_idx:
        bounce_court = court.to_court(np.stack([
            np.asarray(track.x)[bounce_idx], np.asarray(track.y)[bounce_idx],
        ], axis=1))
        structure_bounce_idx = [
            i for i, (court_x, court_y) in zip(bounce_idx, bounce_court)
            if is_in(float(court_x), float(court_y), margin_m, singles=False)
        ]
    n_bounces = len(structure_bounce_idx)
    n_cross = _net_crossings(track, court, in_play, court_margin_m=margin_m)

    strikes = None if serve_times is None else np.asarray(serve_times, float)
    serve_time = None
    if strikes is not None:
        plausible = strikes[(strikes >= selected[0] - toss_preroll_s)
                            & (strikes <= selected[0] + serve_lag_s)]
        if plausible.size:
            serve_time = float(plausible[np.argmin(np.abs(plausible - selected[0]))])
    common.update(n_bounces=n_bounces, n_net_crossings=n_cross,
                  strike_aligned=serve_time is not None)

    baseline_start = False
    if court is not None:
        start_x, start_y = court.to_court(
            [[float(track.x[live_idx[0]]), float(track.y[live_idx[0]])]])[0]
        baseline_start = bool(
            -margin_m <= start_x <= DOUBLES_W + margin_m
            and (start_y <= serve_baseline_margin_m
                 or start_y >= COURT_L - serve_baseline_margin_m))

    short_serve = (serve_time is not None and baseline_start and n_cross >= 1
                   and span >= min(0.5, min_in_play_span_s)
                   and span < min_in_play_span_s)
    has_structure = n_cross >= 1 or n_bounces >= min_bounces
    occupancy_ok = in_play_frac >= min_in_play_frac or short_serve

    if span < min_in_play_span_s and not short_serve:
        return _verdict(
            "indeterminate", win_start, win_end,
            reason_code="component_too_short",
            reason="moving-ball evidence is too short for a decisive point", **common)
    if not occupancy_ok:
        return _verdict(
            "indeterminate", win_start, win_end,
            reason_code="live_component_diluted_by_proposal",
            reason="moving-ball evidence occupies too little of its evidence core", **common)
    if not has_structure:
        if court is None:
            return _verdict(
                "indeterminate", win_start, win_end,
                reason_code="court_unavailable_no_reliable_structure",
                reason="court is unavailable and bounce evidence is insufficient", **common)
        if candidate_coverage < min_measured_coverage:
            return _verdict(
                "indeterminate", win_start, win_end,
                reason_code="structure_missing_with_incomplete_track",
                reason=("net/bounce structure is missing, but overall TrackNet coverage "
                        "is too low for that absence to reject a real short point"),
                **common)
        return _verdict(
            "reject", win_start, win_end,
            reason_code="reliable_no_rally_structure",
            reason="well-measured sustained motion has no net crossing or bounce structure",
            **common)
    if require_serve_evidence and (serve_time is None or not baseline_start):
        state: VerdictState = "indeterminate" if court is None or strikes is None else "reject"
        return _verdict(
            state, win_start, win_end,
            reason_code="required_serve_evidence_missing",
            reason="required strike-plus-baseline evidence is missing", **common)

    start_anchor = serve_time if serve_time is not None else selected[0]
    start = max(float(t[0]), start_anchor - toss_preroll_s)
    end = min(win_end + max_extend_s, selected[1] + tail_s)
    if court is not None and bounce_idx:
        bounces = []
        for i in bounce_idx:
            cx, cy = court.to_court([[float(track.x[i]), float(track.y[i])]])[0]
            bounces.append((float(t[i]), float(cx), float(cy)))
        events = point_end_events(bounces, double_bounce_window_s=double_bounce_window_s,
                                  margin_m=margin_m)
        new_end, end_reason = refine_end_from_events(
            start, win_end, events, min_rally_s=min_rally_s,
            tail_s=tail_s, max_extend_s=max_extend_s)
        if end_reason is not None:
            end = new_end
    end = min(max(end, start + min_rally_s), win_end + max_extend_s)
    if end - start < min_rally_s:
        return _verdict(
            "reject", win_start, win_end,
            reason_code="validated_bounds_too_short",
            reason="validated point bounds are shorter than the minimum rally", **common)
    return _verdict("accept", start, end, reason_code="accepted_rally_structure",
                    reason="credible measured live-ball rally structure", **common)


def rally_verdict(
    track: SmoothTrack,
    court,
    win_start: float,
    win_end: float,
    *,
    min_speed_px_s: float = 25.0,
    min_conf: float = 0.3,
    min_in_play_frac: float = 0.35,
    min_in_play_span_s: float = 1.5,
    min_measured_coverage: float = 0.55,
    max_live_gap_s: float = 0.20,
    max_fragment_join_gap_s: float = 0.80,
    min_bounces: int = 2,
    min_rally_s: float = 1.5,
    toss_preroll_s: float = 1.0,
    tail_s: float = 0.8,
    max_extend_s: float = 3.0,
    bounce_min_descent_px_s: float = 40.0,
    double_bounce_window_s: float = 2.5,
    margin_m: float = 0.35,
    serve_times: Optional[np.ndarray] = None,
    audio_strike_times: Optional[np.ndarray] = None,
    target_serve_times: Optional[np.ndarray] = None,
    target_motion_times: Optional[np.ndarray] = None,
    require_serve_evidence: bool = False,
    require_court: bool = False,
    strike_cluster_gap_s: float = 2.5,
    serve_lag_s: float = 1.5,
    serve_baseline_margin_m: float = 3.0,
) -> RallyVerdict:
    """Return an accept/reject/indeterminate verdict with persistent evidence reasons.

    Reliable measured contradictions reject. Missing court structure, fragmented tracks,
    and weak TrackNet coverage abstain instead, allowing the caller to preserve the broad
    proposal. Nearby strike clusters define local evidence cores so a credible point is
    not diluted merely because a recall-oriented proposal is broad.
    """
    t = np.asarray(track.t, float)
    court_available = court is not None
    if require_court and not court_available:
        return _verdict(
            "indeterminate", win_start, win_end,
            reason_code="target_court_geometry_required",
            reason="target-court geometry is required for match/auto verification",
            court_available=False,
        )
    if t.size < 3:
        return _verdict("indeterminate", win_start, win_end,
                        reason_code="track_too_short",
                        reason="TrackNet returned too few samples",
                        court_available=court_available)

    core = (t >= win_start) & (t <= win_end)
    core_count = int(core.sum())
    if core_count == 0:
        return _verdict("indeterminate", win_start, win_end,
                        reason_code="candidate_outside_track",
                        reason="tracked timestamps do not cover the candidate",
                        court_available=court_available)
    candidate_coverage = float(np.mean(
        np.asarray(track.measured, bool)[core]
        & (np.asarray(track.confidence, float)[core] >= min_conf)))
    all_live = _in_play_mask(track, min_speed_px_s, min_conf, max_live_gap_s)
    raw_components = [
        (a, b) for a, b in _live_components(all_live, t, max_live_gap_s)
        if np.any(core[a:b] & all_live[a:b])
    ]
    if not raw_components:
        # ``measured`` means a heatmap/Hough candidate passed the trajectory gate; it is
        # not calibrated proof that the candidate is the tennis ball. A stationary line,
        # shoe, or other distractor can therefore produce excellent nominal coverage while
        # the real moving ball is missed. Without ball-identity/negative evidence this is
        # an abstention, never a contradiction that may suppress another channel.
        return _verdict(
            "indeterminate", win_start, win_end,
            reason_code=("no_moving_ball_identity_unproven"
                         if candidate_coverage >= min_measured_coverage
                         else "insufficient_track_coverage"),
            reason=("stationary measurements are not calibrated proof that the ball was dead"
                    if candidate_coverage >= min_measured_coverage else
                    "TrackNet coverage is too low to decide whether a ball was live"),
            candidate_coverage=candidate_coverage, court_available=court_available)

    contradiction_strikes = (
        serve_times if audio_strike_times is None else audio_strike_times)
    if _off_target_audio_contradiction(
            track, court, all_live, core, contradiction_strikes,
            court_margin_m=margin_m,
            target_serve_times=target_serve_times,
            target_motion_times=target_motion_times):
        live_idx = np.flatnonzero(all_live & core)
        measured_coverage = float(np.mean(np.asarray(track.measured, bool)[live_idx]))
        span = float(t[live_idx[-1]] - t[live_idx[0]])
        return _verdict(
            "reject", win_start, win_end,
            reason_code="audio_aligned_activity_outside_target_court",
            reason=("measured moving-ball activity aligned to multiple impacts belongs "
                    "clearly outside the calibrated target court"),
            in_play_frac=float(live_idx.size / core_count), in_play_span_s=span,
            measured_coverage=measured_coverage,
            candidate_coverage=candidate_coverage,
            n_live_components=len(raw_components),
            selected_component=(float(t[live_idx[0]]), float(t[live_idx[-1]])),
            evidence_core=(win_start, win_end), strike_aligned=True,
            court_available=True)

    trajectory_end_hint = _audio_aligned_trajectory_end(
        raw_components, t, serve_times, win_start, win_end, tail_s=tail_s)

    components = _join_live_components(
        raw_components, t, max_gap_s=max_fragment_join_gap_s)

    strike_cores = _strike_local_cores(
        serve_times, win_start, win_end, pre_s=toss_preroll_s,
        post_s=max(tail_s, 0.5), cluster_gap_s=strike_cluster_gap_s)
    evaluated: List[RallyVerdict] = []
    for component in components:
        a, b = component
        local = [candidate_core for candidate_core in strike_cores
                 if np.any((t[a:b] >= candidate_core[0])
                           & (t[a:b] <= candidate_core[1]) & all_live[a:b])]
        evidence_core = max(
            local,
            key=lambda candidate_core: int(np.sum(
                (t[a:b] >= candidate_core[0]) & (t[a:b] <= candidate_core[1])
                & all_live[a:b])),
            default=(win_start, win_end),
        )
        evaluated.append(_component_verdict(
            track, court, win_start, win_end, component, evidence_core, all_live,
            candidate_coverage=candidate_coverage,
            n_live_components=len(raw_components),
            min_conf=min_conf, min_in_play_frac=min_in_play_frac,
            min_in_play_span_s=min_in_play_span_s,
            min_measured_coverage=min_measured_coverage, min_bounces=min_bounces,
            min_rally_s=min_rally_s, toss_preroll_s=toss_preroll_s, tail_s=tail_s,
            max_extend_s=max_extend_s,
            bounce_min_descent_px_s=bounce_min_descent_px_s,
            double_bounce_window_s=double_bounce_window_s, margin_m=margin_m,
            serve_times=serve_times, require_serve_evidence=require_serve_evidence,
            serve_lag_s=serve_lag_s,
            serve_baseline_margin_m=serve_baseline_margin_m,
        ))

    for verdict in evaluated:
        verdict.trajectory_end_hint = trajectory_end_hint

    accepted = [verdict for verdict in evaluated if verdict.state == "accept"]
    if accepted:
        return max(accepted, key=lambda verdict: (
            verdict.n_net_crossings + verdict.n_bounces,
            verdict.in_play_span_s, verdict.measured_coverage))
    best = max(evaluated, key=lambda verdict: (
        verdict.measured_coverage, verdict.in_play_span_s,
        verdict.n_net_crossings + verdict.n_bounces))
    if len(components) > 1:
        best.state = "indeterminate"
        best.is_rally = False
        best.reason_code = "fragmented_live_track"
        best.reason = "live-ball evidence is split across disconnected track fragments"
    return best


def _dedupe_non_overlapping(segments: List[Segment]) -> List[Segment]:
    """Sort and remove overlaps, preserving each segment's (serve) start where possible."""
    return trim_previous_on_overlap(segments)


def verify_segments_detailed(
    video: str,
    segments: List[Segment],
    *,
    court=None,
    weights_path: Optional[str] = None,
    model=None,
    pre_pad_s: float = 2.0,
    post_pad_s: float = 2.0,
    max_extend_s: float = 3.0,
    max_tracking_group_s: float = 60.0,
    smooth_max_gap_s: float = 0.5,
    inference_batch_size: Optional[int] = None,
    tracking_fps: Optional[float] = 30.0,
    half_precision: bool = True,
    verdict_kwargs: Optional[dict] = None,
    serve_times: Optional[np.ndarray] = None,
    audio_strike_times: Optional[np.ndarray] = None,
    target_serve_times: Optional[np.ndarray] = None,
    target_motion_times: Optional[np.ndarray] = None,
    require_serve_evidence: bool = False,
    progress: Callable[[str], None] = lambda _m: None,
    cancel_check: Callable[[], None] = lambda: None,
) -> VerificationReport:
    """Track candidates and return segments plus durable per-candidate diagnostics.

    Needs a TrackNet ``weights_path`` (or a preloaded ``model``). ``court`` is optional but
    strongly recommended — without it the verdict can't use net-crossing / in-out geometry
    and leans on bounce structure. ``report.segments`` contains accepted ball-bounded
    segments only; callers recover their own coherent fallback for indeterminate candidate
    intervals from ``report.candidates``.
    """
    from ..signals.ball import load_ball_model, track_tracknet

    if model is None:
        if not weights_path:
            raise ValueError("verify_segments needs a TrackNet weights_path or model")
        model = load_ball_model(weights_path)

    verdict_kwargs = dict(verdict_kwargs or {})
    verdict_kwargs.setdefault("max_extend_s", max_extend_s)
    verdict_kwargs.setdefault("require_serve_evidence", require_serve_evidence)

    grouped = _group_tracking_windows(
        segments, pre_pad_s, post_pad_s, max_extend_s, max_tracking_group_s)

    kept: List[Segment] = []
    diagnostics: List[CandidateVerification] = []
    track_cache: List[Tuple[Segment, Any]] = []
    total_tracking_s = sum(end - start for start, end, _members in grouped)
    completed_tracking_s = 0.0
    last_reported_percent = -1
    actual_batch_size: Optional[int] = None
    for track_start, track_end, members in grouped:
        cancel_check()
        group_s = track_end - track_start

        def tracking_progress(done_frames: int, total_frames: int, batch: int) -> None:
            nonlocal last_reported_percent, actual_batch_size
            actual_batch_size = batch
            fraction = done_frames / max(1, total_frames)
            done_s = completed_tracking_s + group_s * min(1.0, fraction)
            percent = int(round(100.0 * done_s / max(total_tracking_s, 1e-9)))
            if percent == last_reported_percent and done_frames < total_frames:
                return
            last_reported_percent = percent
            progress(
                f"ball tracking progress {percent}% "
                f"({done_s:.1f}/{total_tracking_s:.1f}s, batch {batch})")

        track = track_tracknet(
            video, model=model, start_s=track_start, end_s=track_end,
            batch_size=inference_batch_size,
            tracking_fps=tracking_fps,
            half_precision=half_precision,
            court=court,
            progress_callback=tracking_progress,
            cancel_check=cancel_check,
        )
        track_cache.append(((track_start, track_end), track))
        completed_tracking_s += group_s
        st = smooth_track(track, max_gap_s=smooth_max_gap_s)
        for s, e in members:
            candidate_kwargs = dict(verdict_kwargs)
            if serve_times is not None:
                candidate_serve_times = np.asarray(serve_times, float)
                candidate_serve_times = candidate_serve_times[
                    (candidate_serve_times >= s - pre_pad_s)
                    & (candidate_serve_times <= e + max(post_pad_s, max_extend_s))]
                candidate_kwargs["serve_times"] = candidate_serve_times
            for name, values in (
                ("audio_strike_times", audio_strike_times),
                ("target_serve_times", target_serve_times),
                ("target_motion_times", target_motion_times),
            ):
                if values is None:
                    continue
                candidate_values = np.asarray(values, float)
                candidate_values = candidate_values[
                    (candidate_values >= s - pre_pad_s)
                    & (candidate_values <= e + max(post_pad_s, max_extend_s))]
                candidate_kwargs[name] = candidate_values
            v = rally_verdict(st, court, s, e, **candidate_kwargs)
            output: Optional[Segment] = None
            peak_ball_speed_kmh: Optional[float] = None
            speed_estimate: Optional[dict[str, Any]] = None
            if v.state == "accept":
                output = (v.start, v.end)
                if court is not None:
                    speeds = ball_speed_kmh(st.t, st.x, st.y, court)
                    reliable = (
                        (st.t >= v.start) & (st.t <= v.end) & st.measured
                        & np.isfinite(speeds)
                    )
                    values = speeds[reliable]
                    if values.size >= 3:
                        interval_count = int(np.sum(
                            (st.t >= v.start) & (st.t <= v.end)))
                        speed_estimate = _ground_plane_speed_estimate(
                            values, values.size / max(interval_count, 1))
                        if speed_estimate is not None:
                            peak_ball_speed_kmh = float(speed_estimate["value_kmh"])
            if output is not None:
                kept.append(output)
            diagnostics.append(CandidateVerification(
                candidate=(s, e), tracked_window=(track_start, track_end),
                verdict=v, output=output,
                peak_ball_speed_kmh=peak_ball_speed_kmh,
                ball_speed_estimate=speed_estimate))
            progress(f"    candidate {s:.2f}-{e:.2f}s: {v.state} "
                     f"[{v.reason_code}] {v.reason}")
    result = VerificationReport(
        _dedupe_non_overlapping(kept), diagnostics, track_cache=track_cache,
        inference_batch_size=actual_batch_size)
    counts = result.as_dict()["counts"]
    progress("  ball arbiter: "
             f"{counts['accept']} accepted, {counts['reject']} rejected, "
             f"{counts['indeterminate']} indeterminate/{len(segments)} candidates")
    return result


def verify_segments(
    video: str,
    segments: List[Segment],
    *,
    court=None,
    weights_path: Optional[str] = None,
    model=None,
    pre_pad_s: float = 2.0,
    post_pad_s: float = 2.0,
    max_extend_s: float = 3.0,
    smooth_max_gap_s: float = 0.5,
    inference_batch_size: Optional[int] = None,
    verdict_kwargs: Optional[dict] = None,
    serve_times: Optional[np.ndarray] = None,
    require_serve_evidence: bool = False,
    keep_indeterminate: bool = True,
    diagnostics_out: Optional[List[dict[str, Any]]] = None,
    progress: Callable[[str], None] = lambda _m: None,
    cancel_check: Callable[[], None] = lambda: None,
) -> List[Segment]:
    """Backward-compatible segment API with optional diagnostic persistence.

    The return type remains ``List[Segment]``. Pass ``diagnostics_out`` to retain the
    serialisable reason/metric records, or call :func:`verify_segments_detailed`.
    """
    report = verify_segments_detailed(
        video, segments, court=court, weights_path=weights_path, model=model,
        pre_pad_s=pre_pad_s, post_pad_s=post_pad_s, max_extend_s=max_extend_s,
        smooth_max_gap_s=smooth_max_gap_s,
        inference_batch_size=inference_batch_size,
        verdict_kwargs=verdict_kwargs,
        serve_times=serve_times, require_serve_evidence=require_serve_evidence,
        progress=progress, cancel_check=cancel_check)
    if diagnostics_out is not None:
        diagnostics_out.extend(candidate.as_dict() for candidate in report.candidates)
    kept = list(report.segments)
    if keep_indeterminate:
        kept.extend(candidate.candidate for candidate in report.candidates
                    if candidate.verdict.state == "indeterminate")
    return _dedupe_non_overlapping(kept)
