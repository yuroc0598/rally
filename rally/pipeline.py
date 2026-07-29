"""End-to-end orchestration: video in -> rally-only video (+ JSON sidecar) out.

``trim`` is a thin orchestrator; each analysis/refinement stage is its own
function below so the flow reads top-to-bottom:

    probe -> [audio | visual | ball | pose] channels -> co-decide (fuse)
          -> derive points -> anchor serves
          -> ball arbiter: validate + bound each candidate   (default; ball_arbiter)
             OR legacy trim-ball-ends                          (when ball_arbiter off)
          -> validate match serve/setup sequence -> drop non-play -> write sidecar + cut
"""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import RallyConfig
from .io.ffmpeg import (
    add_real_context,
    cut_segments,
    find_font,
    iter_audio_mono,
    probe,
    render_labeled,
)
from .signals.audio import detect_strikes_stream, strike_rhythm_features
from .fusion.score import audio_score, motion_score, rally_probability
from .fusion.decode import segments_from_prob
from .fusion.points import (
    drop_isolated_points,
    points_from_strikes,
    points_from_strikes_movement,
    snap_serve_starts,
    total_kept_seconds,
)

Segment = Tuple[float, float]
Progress = Callable[[str], None]
CancelCheck = Callable[[], None]


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class RallyResult:
    input_path: str
    output_path: Optional[str]
    segments: List[Segment]
    total_seconds: float
    kept_seconds: float
    compression_ratio: float
    channels_used: List[str] = field(default_factory=list)
    n_strikes: int = 0
    strike_times: List[float] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def sidecar(self) -> dict:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "total_seconds": round(self.total_seconds, 3),
            "kept_seconds": round(self.kept_seconds, 3),
            "compression_ratio": round(self.compression_ratio, 4),
            "n_rallies": len(self.segments),
            "n_strikes": self.n_strikes,
            "strike_times": [round(float(t), 3) for t in self.strike_times],
            "channels_used": self.channels_used,
            "stages": self.stages,
            "config": self.config,
            "segments": [
                {"index": i, "start": round(s, 3), "end": round(e, 3),
                 "duration": round(e - s, 3)}
                for i, (s, e) in enumerate(self.segments)
            ],
        }


@dataclass
class _Channels:
    """Per-frame source signals gathered before fusion (None = unavailable)."""
    audio_rate: Optional[np.ndarray] = None
    audio_reg: Optional[np.ndarray] = None
    geometry: Optional[np.ndarray] = None
    motion: Optional[np.ndarray] = None
    camera_moving: Optional[np.ndarray] = None
    near_track: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ball: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    pose_conf: Optional[np.ndarray] = None
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    n_strikes: int = 0
    used: List[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    detector: Optional[object] = None   # shared YOLO PlayerDetector (loaded once)
    player_samples: list = field(default_factory=list)
    court: Optional[object] = None
    frame_size: Optional[Tuple[int, int]] = None
    arbiter_audio_fallback: List[Segment] = field(default_factory=list)
    arbiter_selected_audio_fallback: List[Segment] = field(default_factory=list)
    arbiter_accepted_regions: List[Segment] = field(default_factory=list)
    arbiter_indeterminate_regions: List[Segment] = field(default_factory=list)
    arbiter_rejected_regions: List[Segment] = field(default_factory=list)
    ball_end_hints: list[tuple[Segment, float]] = field(default_factory=list)
    player_serve_hints: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))


# --------------------------------------------------------------------------- #
# analysis stages: each fills part of _Channels                               #
# --------------------------------------------------------------------------- #
def _audio_channel(input_path, info, timeline, cfg, ch, progress,
                   cancel_check: CancelCheck = lambda: None) -> None:
    """Ball-strike detection + strike-rhythm features (the primary rally cue)."""
    if not info.has_audio:
        ch.stages["audio"] = {"status": "unavailable", "reason": "no audio stream"}
        return
    try:
        progress("decoding audio and detecting ball strikes")
        def cancellable_chunks():
            for chunk in iter_audio_mono(input_path, cfg.audio_sr, chunk_s=60.0):
                cancel_check()
                yield chunk

        chunks = cancellable_chunks()
        ch.onsets = detect_strikes_stream(chunks, cfg.audio_sr, cfg)
        ch.n_strikes = int(ch.onsets.size)
        progress(f"  {ch.n_strikes} strikes detected")
        # Only vote when there are strikes. With zero strikes the rhythm features are all
        # zero; adding them as a confident-0 channel would drag the fused probability down
        # (a confident "no rally" everywhere) instead of abstaining. Let the other channels
        # decide when audio has nothing to say.
        if ch.n_strikes > 0:
            ch.audio_rate, ch.audio_reg = strike_rhythm_features(ch.onsets, timeline, cfg)
            ch.used.append("audio")
            ch.stages["audio"] = {"status": "used", "strikes": ch.n_strikes,
                                  "streaming": True}
        else:
            progress("  no strikes -> audio abstains from the fusion")
            ch.stages["audio"] = {"status": "abstained", "strikes": 0,
                                  "streaming": True}
    except Exception as exc:  # pragma: no cover
        ch.stages["audio"] = {"status": "failed", "reason": str(exc)}
        progress(f"  audio channel failed: {exc}")


def _visual_channels(input_path, cfg, timeline, detect_players, ch, progress,
                     cancel_check: CancelCheck = lambda: None) -> None:
    """Frame-diff motion, camera-motion flag, player-geometry + near-player track."""
    from .signals.visual import analyze_visual, opencv_available

    if not opencv_available():  # pragma: no cover
        ch.stages["visual"] = {"status": "unavailable", "reason": "OpenCV unavailable"}
        progress("OpenCV not installed -> motion/geometry channels disabled")
        return
    try:
        progress("sampling frames (motion / camera / players)")
        detector = None
        if detect_players:
            from .signals.player import PlayerDetector

            detector = PlayerDetector()
            ch.detector = detector  # reused by the serve-anchoring stage (no second load)
            if not detector.available:
                progress("  YOLO unavailable -> geometry channel disabled")
        vis = analyze_visual(
            input_path, cfg, timeline, detector, progress=progress,
            cancel_check=cancel_check)
        ch.motion = vis["motion"]
        ch.camera_moving = vis["camera_moving"]
        ch.geometry = vis["geometry"]
        ch.near_track = vis.get("near_track")
        ch.player_samples = vis.get("player_samples") or []
        ch.used.append("motion")
        if ch.geometry is not None:
            ch.used.append("geometry")
        ch.stages["visual"] = {
            "status": "used", "players": ch.geometry is not None,
            "device": (getattr(detector, "device", None) if detector is not None else None),
        }
    except Exception as exc:  # pragma: no cover
        ch.stages["visual"] = {"status": "failed", "reason": str(exc)}
        progress(f"  visual channel failed: {exc}")


def _ball_channel(input_path, timeline, cfg, ch, progress,
                  cancel_check: CancelCheck = lambda: None) -> None:
    """Opt-in: track the ball over the whole video as a co-deciding vote."""
    if not (cfg.ball_channel and cfg.ball_weights):
        if cfg.ball_channel:
            ch.stages["ball_channel"] = {
                "status": "unavailable", "reason": "explicit ball weights are required"}
        return
    try:
        from .signals.ball import ball_in_play_channel, track_tracknet
        progress("ball-in-play channel: tracking ball over the whole video (slow on CPU)")
        btrack = track_tracknet(
            input_path, cfg.ball_weights,
            batch_size=cfg.ball_inference_batch_size, cancel_check=cancel_check)
        ch.ball = ball_in_play_channel(btrack, timeline)
        ch.used.append("ball")
        ch.stages["ball_channel"] = {"status": "used", "weights": cfg.ball_weights}
    except Exception as exc:  # pragma: no cover
        ch.stages["ball_channel"] = {"status": "failed", "reason": str(exc)}
        progress(f"  ball channel failed: {exc}")


def _pose_channel(input_path, timeline, cfg, ch, progress,
                  cancel_check: CancelCheck = lambda: None) -> None:
    """Opt-in: near-player pose activity as a confidence-weighted vote."""
    if not cfg.player_pose:
        return
    try:
        from .signals.player import pose_activity_track
        progress("player pose channel: tracking pose over the video (slow)")
        ch.pose, ch.pose_conf = pose_activity_track(
            input_path, timeline, fps_a=cfg.player_pose_fps,
            cancel_check=cancel_check)
        ch.used.append("pose")
        ch.stages["pose"] = {"status": "used"}
    except Exception as exc:  # pragma: no cover
        ch.stages["pose"] = {"status": "failed", "reason": str(exc)}
        progress(f"  pose channel failed: {exc}")


# --------------------------------------------------------------------------- #
# decision + refinement stages: probability -> points -> refined points       #
# --------------------------------------------------------------------------- #
def _arbiter_candidates(regions: List[Segment], onsets: np.ndarray,
                        split_gap_s: float) -> List[Segment]:
    """Partition coarse regions without discarding any time or single-hit candidates.

    Audio gaps provide useful workload boundaries, but proposal generation must remain
    high-recall: every part of every coarse region is retained for the ball arbiter.
    """
    onsets = np.sort(np.asarray(onsets, dtype=float))
    out: List[Segment] = []
    for rs, re in regions:
        w = onsets[(onsets >= rs) & (onsets <= re)]
        if w.size < 2:
            out.append((rs, re))
            continue
        cuts = np.where(np.diff(w) > split_gap_s)[0]
        boundaries = [0.5 * (float(w[i]) + float(w[i + 1])) for i in cuts]
        edges = [rs, *boundaries, re]
        out.extend((float(a), float(b)) for a, b in zip(edges, edges[1:]) if b > a)
    return out


def _bounded_arbiter_regions(regions: List[Segment], evidence: np.ndarray,
                             duration: float, cfg, ch) -> Tuple[List[Segment], List[Segment]]:
    """Rank and cap expensive trajectory work without creating giant candidates.

    The cap is a safety valve for adversarial/noisy footage, not an accuracy claim: stage
    provenance records how much proposal time was omitted so callers can see degradation.
    """
    max_len = float(cfg.arbiter_max_candidate_s)
    chunks: List[Segment] = []
    for start, end in regions:
        cursor = float(start)
        while cursor < end:
            stop = min(float(end), cursor + max_len)
            if stop > cursor:
                chunks.append((cursor, stop))
            cursor = stop
    proposed_s = sum(e - s for s, e in chunks)
    if cfg.accuracy_mode:
        # Accuracy runs must not make a correctness decision from whether an otherwise
        # valid proposal happened to win a workload ranking.  The expensive model may be
        # slower, but every proposed region receives the same trajectory opportunity.
        budget = float(duration)
    else:
        budget = min(float(duration), float(cfg.arbiter_max_total_s),
                     max(float(cfg.arbiter_min_total_s),
                         float(duration) * float(cfg.arbiter_max_total_fraction)))

    def tracked_seconds(items: List[Segment]) -> float:
        padded = sorted((max(0.0, s - cfg.arbiter_pre_pad_s),
                         min(duration, e + max(cfg.arbiter_post_pad_s,
                                               cfg.ball_max_extend_s)))
                        for s, e in items)
        merged: List[List[float]] = []
        for start, end in padded:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged)

    proposed_tracked_s = tracked_seconds(chunks)
    if proposed_tracked_s <= budget:
        ch.stages["arbiter_proposal"] = {
            "status": "used", "proposed_seconds": round(proposed_s, 3),
            "selected_seconds": round(proposed_s, 3),
            "tracked_seconds": round(proposed_tracked_s, 3), "capped": False,
            "accuracy_mode": bool(cfg.accuracy_mode),
            "selected_regions": len(chunks), "omitted_regions": 0,
            "ranking": "audio_coherence_strikes_then_temporal_diversity"}
        return chunks, []

    fps = float(cfg.analysis_fps)

    audio = None
    if ch.audio_rate is not None and ch.audio_reg is not None:
        audio = audio_score(ch.audio_rate, ch.audio_reg)
    onsets = np.asarray(ch.onsets, dtype=float)

    def peak(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        # A point is short relative to a 45 s chunk. Mean-only ranking dilutes a real
        # cadence into surrounding dead time, so combine its peak with a smaller mean.
        return 0.8 * float(np.max(values)) + 0.2 * float(np.mean(values))

    def quality(seg: Segment) -> float:
        lo = max(0, int(np.floor(seg[0] * fps)))
        hi = min(evidence.size, max(lo + 1, int(np.ceil(seg[1] * fps))))
        generic = peak(evidence[lo:hi]) if hi > lo else 0.0
        coherence = peak(audio[lo:hi]) if audio is not None and hi > lo else 0.0
        n_strikes = int(np.sum((onsets >= seg[0]) & (onsets <= seg[1])))
        strike_support = min(1.0, n_strikes / max(1, cfg.min_rally_strikes + 1))
        # Audio cadence and multiple accepted transients are discriminative. Generic
        # proposal evidence only breaks otherwise weak cases; it must not dominate.
        return 0.70 * coherence + 0.20 * strike_support + 0.10 * generic

    def diversity(seg: Segment, selected: List[Segment]) -> float:
        midpoint = 0.5 * (seg[0] + seg[1])
        if not selected:
            # With tied evidence, begin near the centre rather than depending on input
            # ordering. Subsequent farthest-point choices spread across the timeline.
            return max(0.0, 1.0 - abs(midpoint - duration / 2) / max(duration / 2, 1e-9))
        nearest = min(abs(midpoint - 0.5 * (s + e)) for s, e in selected)
        return min(1.0, 2.0 * nearest / max(duration, 1e-9))

    selected: List[Segment] = []
    remaining = list(chunks)
    while remaining:
        feasible = [seg for seg in remaining
                    if tracked_seconds([*selected, seg]) <= budget]
        if not feasible:
            break
        chosen = max(
            feasible,
            key=lambda seg: (
                quality(seg) + cfg.arbiter_diversity_weight * diversity(seg, selected),
                quality(seg),
                -abs(0.5 * (seg[0] + seg[1]) - duration / 2),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    selected.sort()
    omitted = list(remaining)
    omitted.sort()
    selected_s = sum(e - s for s, e in selected)
    selected_tracked_s = tracked_seconds(selected)
    ch.stages["arbiter_proposal"] = {
        "status": "capped", "proposed_seconds": round(proposed_s, 3),
        "selected_seconds": round(selected_s, 3),
        "tracked_seconds": round(selected_tracked_s, 3),
        "omitted_seconds": round(max(0.0, proposed_s - selected_s), 3),
        "budget_seconds": round(budget, 3),
        "selected_regions": len(selected), "omitted_regions": len(omitted),
        "ranking": "audio_coherence_strikes_then_temporal_diversity"}
    return selected, omitted


def _coherent_audio_fallback(regions: List[Segment], ch, duration: float,
                             cfg) -> List[Segment]:
    """Build audio hypotheses for TrackNet-omitted or indeterminate regions.

    Accuracy mode deliberately retains short and single-contact hypotheses.  They remain
    unevaluated evidence for the later serve/match decoder, not accepted rallies.  The
    throughput mode keeps the older conservative coherence gate.
    """
    if not regions or ch.onsets.size == 0:
        return []
    # Computational chunk boundaries are not point boundaries. Rejoin adjacent selected
    # or omitted chunks before audio clustering so a rally straddling 45 s is not split.
    contiguous: List[Segment] = []
    for start, end in sorted(regions):
        if contiguous and start <= contiguous[-1][1] + 1e-9:
            contiguous[-1] = (contiguous[-1][0], max(contiguous[-1][1], end))
        else:
            contiguous.append((start, end))
    serve_window = cfg.serve_attach_window_s if cfg.serve_attach else 0.0
    min_strikes = 1 if cfg.accuracy_mode else cfg.min_rally_strikes
    min_duration = 0.0 if cfg.accuracy_mode else cfg.min_rally_dur_s
    return points_from_strikes(
        contiguous, ch.onsets, cfg.point_gap_s, min_strikes,
        cfg.toss_preroll_s, cfg.landing_tail_s, duration,
        echo_s=cfg.echo_collapse_s, min_dur_s=min_duration,
        serve_window_s=serve_window,
    )


def _merge_point_sources(primary: List[Segment], fallback: List[Segment]) -> List[Segment]:
    """Combine already-adjudicated sources without truncating whole fallback points.

    Every interval in ``fallback`` reached this function because no single ball-verifier
    verdict owned all of that point's strikes (or because proposal budgeting omitted it).
    A partially accepted trajectory must therefore defer to the whole fallback point,
    rather than deleting it merely because the two intervals overlap. Fully owned audio
    points are removed earlier by :func:`_indeterminate_audio_fallback`, so they never
    reach this merge.
    """
    precise = [(s, e) for s, e in primary if e > s]
    owned_fallback = [
        (s, e) for s, e in fallback
        if e > s
    ]
    retained_precise = [
        (s, e) for s, e in precise
        if not any(_overlaps((s, e), point) for point in owned_fallback)
    ]
    return sorted([*retained_precise, *owned_fallback])


def _player_activity_proposal(ch, cfg) -> np.ndarray:
    """Recall-only player activity and stable-to-active serve-time hints.

    The near-player track is sampled sparsely. Fill only intervals whose measured
    displacement rate clears the activity threshold, discard short jitter bursts, then
    require a stable near-court formation immediately before an activity run. The inferred
    contact time precedes receiver movement by a small reaction lag. These hints only open
    TrackNet work; later match validation must independently confirm the serve.
    """
    if ch.near_track is None:
        return np.zeros(0, dtype=float)
    px, py = (np.asarray(values, dtype=float) for values in ch.near_track)
    n = min(px.size, py.size)
    activity = np.zeros(n, dtype=float)
    valid = np.flatnonzero(np.isfinite(px[:n]) & np.isfinite(py[:n]))
    if valid.size < 3:
        return activity

    fps = float(cfg.analysis_fps)
    for prior, current in zip(valid, valid[1:]):
        dt = (current - prior) / fps
        if dt <= 0.0 or dt > max(1.0, 3.0 / max(float(cfg.player_fps), 1e-6)):
            continue
        speed = float(np.hypot(px[current] - px[prior], py[current] - py[prior]) / dt)
        if speed >= cfg.match_player_activity_speed:
            activity[prior:current + 1] = 1.0

    min_frames = max(1, int(round(cfg.match_player_activity_min_s * fps)))
    active = activity > 0.0
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    activity[:] = 0.0
    hints: list[float] = []
    stable_frames = max(1, int(round(cfg.match_player_stable_s * fps)))
    for start, stop in runs:
        if stop - start < min_frames:
            continue
        stable_start = max(0, int(start) - stable_frames)
        stable_idx = valid[(valid >= stable_start) & (valid <= start)]
        if stable_idx.size < max(3, int(round(cfg.match_player_stable_s * cfg.player_fps))):
            continue
        stable_span = max(
            float(np.ptp(px[stable_idx])), float(np.ptp(py[stable_idx])))
        near_y = float(np.median(py[stable_idx]))
        if (stable_span <= cfg.match_player_stable_span
                and near_y >= cfg.match_player_hint_near_y_min):
            # Only the early reaction is needed to bridge the serve into a candidate.
            # Capping it prevents subsequent pickup/walking from joining many otherwise
            # independent TrackNet windows into one near-full-video inference group.
            proposal_stop = min(
                int(stop), int(start) + max(
                    1, int(round(cfg.match_fragment_merge_gap_s * fps))))
            activity[start:proposal_stop] = 1.0
            hints.append(max(
                0.0, float(start) / fps - cfg.match_receiver_reaction_lag_s))

    ch.player_serve_hints = np.asarray(hints, dtype=float)
    ch.stages["player_activity_proposal"] = {
        "status": "used" if hints else "no_hints",
        "active_fraction": round(float(np.mean(activity > 0.0)), 4),
        "serve_hints": [round(float(value), 3) for value in hints],
        "rule": "stable_formation_then_receiver_activity",
    }
    return activity


def _overlaps(a: Segment, b: Segment) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _point_strikes(point: Segment, onsets: np.ndarray) -> np.ndarray:
    onsets = np.asarray(onsets, dtype=float)
    return onsets[(onsets >= point[0] - 1e-9) & (onsets <= point[1] + 1e-9)]


def _point_fully_evaluated(point: Segment, region: Segment, onsets: np.ndarray) -> bool:
    """Whether one candidate core contains every accepted strike belonging to a point."""
    strikes = _point_strikes(point, onsets)
    if strikes.size:
        return bool(np.all((strikes >= region[0] - 1e-9)
                           & (strikes <= region[1] + 1e-9)))
    return point[0] >= region[0] - 1e-9 and point[1] <= region[1] + 1e-9


def _point_covered_by_regions(point: Segment, regions: List[Segment],
                              onsets: np.ndarray) -> bool:
    """Whether selected region cores collectively cover all strikes in ``point``."""
    strikes = _point_strikes(point, onsets)
    if not strikes.size:
        return any(point[0] >= s - 1e-9 and point[1] <= e + 1e-9 for s, e in regions)
    return all(any(s - 1e-9 <= strike <= e + 1e-9 for s, e in regions)
               for strike in strikes)


def _apply_ball_end_hints(segments: List[Segment], ch, cfg) -> List[Segment]:
    """Refine fallback ends using impacts followed by measured outgoing ball motion."""
    if not ch.ball_end_hints:
        return segments
    out: List[Segment] = []
    changed = 0
    for point in segments:
        candidates = [
            (region, hint) for region, hint in ch.ball_end_hints
            if _point_fully_evaluated(point, region, ch.onsets)
        ]
        if not candidates:
            out.append(point)
            continue
        # Computational proposal boundaries can overlap.  Prefer the hint requiring the
        # smallest correction to this audio hypothesis, then keep it inside the source.
        _region, hint = min(candidates, key=lambda item: abs(item[1] - point[1]))
        # A trajectory component can disappear at an ordinary mid-rally occlusion.  Such
        # a local endpoint is useful for removing a few seconds of chained pickup noise,
        # but is not strong enough to delete a large suffix of an audio-supported point.
        if (ch.court is None
                and float(hint) < point[1] - cfg.ball_end_hint_max_uncalibrated_trim_s):
            out.append(point)
            continue
        end = max(point[0] + min(cfg.min_rally_s, 0.25), float(hint))
        out.append((point[0], end))
        changed += int(abs(end - point[1]) > 1e-6)
    ch.stages["ball_end_hints"] = {
        "status": "used", "available": len(ch.ball_end_hints), "changed": changed,
    }
    return out


def _indeterminate_audio_fallback(ch) -> Tuple[List[Segment], int, int]:
    """Resolve selected audio points by whole-point tri-state ownership."""
    kept: List[Segment] = []
    suppressed = 0
    superseded = 0
    for point in ch.arbiter_selected_audio_fallback:
        if any(_point_fully_evaluated(point, region, ch.onsets)
               for region in ch.arbiter_accepted_regions):
            superseded += 1
            continue
        if any(_point_fully_evaluated(point, region, ch.onsets)
               for region in ch.arbiter_rejected_regions):
            suppressed += 1
            continue
        if (any(_point_fully_evaluated(point, region, ch.onsets)
                for region in ch.arbiter_indeterminate_regions)
                # No single verdict evaluated the whole point (for example a computational
                # boundary split it): retain conservative audio instead of silently losing it.
                or not any(_point_fully_evaluated(point, region, ch.onsets)
                           for region in [*ch.arbiter_accepted_regions,
                                          *ch.arbiter_rejected_regions,
                                          *ch.arbiter_indeterminate_regions])):
            kept.append(point)
    return kept, suppressed, superseded


def _derive_points(ch, duration, cfg, progress, *, for_ball_arbiter: bool = False) -> List[Segment]:
    """Fuse channels into a rally probability, then cut it into individual points."""
    # player presence (geometry) is a supporting vote only — two players are on court
    # between points too, so audio/ball carry the "is it a rally" decision.
    progress(f"co-deciding rally probability from channels: {', '.join(ch.used)}")
    prob = rally_probability(
        cfg, n=timeline_size(duration, cfg),
        audio_rate=ch.audio_rate, audio_regularity=ch.audio_reg,
        geometry=ch.geometry, motion=ch.motion, camera_moving=ch.camera_moving,
        ball=ch.ball, pose=ch.pose, pose_conf=ch.pose_conf,
    )
    decision_prob = prob
    if for_ball_arbiter:
        # Proposal mode is intentionally more permissive than final classification. Weak
        # visual cues may open a candidate but cannot survive without trajectory evidence.
        proposal = prob.copy()
        if ch.motion is not None:
            motion_evidence = np.nan_to_num(
                motion_score(ch.motion, ch.camera_moving, cfg),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            active_fraction = float(np.mean(motion_evidence >= 0.5)) if motion_evidence.size else 0.0
            if active_fraction <= cfg.arbiter_motion_max_active_frac:
                proposal = np.maximum(proposal, 0.65 * motion_evidence)
                ch.stages["motion_proposal"] = {
                    "status": "used", "active_fraction": round(active_fraction, 4)}
            else:
                ch.stages["motion_proposal"] = {
                    "status": "suppressed", "active_fraction": round(active_fraction, 4),
                    "reason": "motion saturated; refusing unbounded ball-tracking work"}
        if ch.pose is not None:
            pose_evidence = np.nan_to_num(
                ch.pose, nan=0.0, posinf=0.0, neginf=0.0)
            proposal = np.maximum(proposal, 0.60 * np.clip(pose_evidence, 0.0, 1.0))
        if ch.geometry is not None:
            geometry_evidence = np.nan_to_num(
                ch.geometry, nan=0.0, posinf=0.0, neginf=0.0)
            proposal = np.maximum(
                proposal, 0.40 * np.clip(geometry_evidence, 0.0, 1.0))
        if cfg.play_mode != "casual" and ch.near_track is not None:
            player_activity = _player_activity_proposal(ch, cfg)
            if player_activity.size == proposal.size:
                proposal = np.maximum(proposal, 0.65 * player_activity)
                # The reaction begins shortly after contact. Open a compact bridge from
                # the inferred contact into the measured activity run so smoothing cannot
                # leave the serve itself outside the TrackNet candidate.
                hint_pre = max(1, int(round(0.75 * cfg.analysis_fps)))
                hint_post = max(1, int(round(0.75 * cfg.analysis_fps)))
                for hint in ch.player_serve_hints:
                    center = int(round(float(hint) * cfg.analysis_fps))
                    lo = max(0, center - hint_pre)
                    hi = min(proposal.size, center + hint_post + 1)
                    proposal[lo:hi] = np.maximum(proposal[lo:hi], 0.65)
        # Every accepted transient gets a small proposal window. This is deliberately
        # independent of cadence/coherence so one-hit points can reach the arbiter.
        pre_frames = max(1, int(round(0.75 * cfg.analysis_fps)))
        post_frames = max(1, int(round(1.25 * cfg.analysis_fps)))
        for onset in ch.onsets:
            center = int(round(float(onset) * cfg.analysis_fps))
            lo, hi = max(0, center - pre_frames), min(proposal.size, center + post_frames + 1)
            proposal[lo:hi] = np.maximum(proposal[lo:hi], 0.65)
        decision_prob = proposal
    regions = segments_from_prob(decision_prob, cfg.analysis_fps, cfg, total_s=duration)
    onsets = ch.onsets
    serve_window = cfg.serve_attach_window_s if cfg.serve_attach else 0.0

    if for_ball_arbiter:
        # Do not apply audio coherence here. Missed impacts and legitimate one-hit points
        # (aces/double faults) must reach the expensive trajectory decision.
        # Audio coherence is a point-level property. Compute it before computational
        # chunking so a rally with strikes on both sides of a budget edge remains whole.
        coherent_points = _coherent_audio_fallback(regions, ch, duration, cfg)
        selected, omitted = _bounded_arbiter_regions(
            regions, decision_prob, duration, cfg, ch)
        ch.arbiter_selected_audio_fallback = [
            point for point in coherent_points
            if _point_covered_by_regions(point, selected, ch.onsets)
        ]
        ch.arbiter_audio_fallback = [
            point for point in coherent_points
            if not _point_covered_by_regions(point, selected, ch.onsets)
        ]
        candidates = _arbiter_candidates(selected, onsets, cfg.point_gap_s)
        proposal_stage = ch.stages["arbiter_proposal"]
        proposal_stage.update({
            "selected_candidates": len(candidates),
            "audio_fallback_points": len(ch.arbiter_audio_fallback),
            "audio_fallback_seconds": round(
                total_kept_seconds(ch.arbiter_audio_fallback), 3),
            "selected_audio_points_pending_verdict": len(
                ch.arbiter_selected_audio_fallback),
        })
        player_stage = ch.stages.get("player_activity_proposal")
        if player_stage is not None:
            progress(
                "  player serve proposal: "
                f"{len(ch.player_serve_hints)} stable-to-active hint(s); "
                f"TrackNet workload {proposal_stage.get('tracked_seconds', 0):.1f}s"
            )
        ch.stages["arbiter_audio_fallback"] = {
            "status": ("ready" if (ch.arbiter_audio_fallback
                                     or ch.arbiter_selected_audio_fallback) else "none"),
            "source": "whole_coherent_audio_points_by_tracknet_ownership",
            "omitted_points_ready": len(ch.arbiter_audio_fallback),
            "omitted_seconds_ready": round(
                total_kept_seconds(ch.arbiter_audio_fallback), 3),
            "selected_points_pending_verdict": len(
                ch.arbiter_selected_audio_fallback),
        }
        return candidates

    if cfg.point_split and onsets.size:
        if cfg.movement_merge and ch.near_track is not None:
            # strike-bounded points, with short low-movement lulls merged (not new points)
            px, py = ch.near_track
            return points_from_strikes_movement(
                regions, onsets, timeline_array(duration, cfg), px, py,
                gap_s=cfg.point_gap_s, merge_max_gap_s=cfg.merge_max_gap_s,
                move_thresh=cfg.move_thresh, min_strikes=cfg.min_rally_strikes,
                toss_preroll_s=cfg.toss_preroll_s, landing_tail_s=cfg.landing_tail_s,
                total_s=duration, echo_s=cfg.echo_collapse_s, min_dur_s=cfg.min_rally_dur_s,
                serve_window_s=serve_window)
        if cfg.movement_merge:
            progress("  no player track -> movement merge disabled (gap-only splitting)")
        # tightly bound each point to its ball strikes (excludes inter-point walking)
        return points_from_strikes(
            regions, onsets, cfg.point_gap_s, cfg.min_rally_strikes,
            cfg.toss_preroll_s, cfg.landing_tail_s, duration,
            echo_s=cfg.echo_collapse_s, min_dur_s=cfg.min_rally_dur_s,
            serve_window_s=serve_window)
    if cfg.snap_serve and onsets.size:
        return snap_serve_starts(regions, onsets, cfg.serve_lookback_s, cfg.serve_preroll_s)
    return regions


def _anchor_serves(input_path, segments, ch, cfg, progress) -> List[Segment]:
    """Opt-in (calibration + YOLO): move each point start back to the serve set-up."""
    if not (cfg.court_corners is not None and ch.onsets.size and segments):
        return segments
    try:
        from .signals.court import Court
        from .signals.player import PlayerTracker, refine_starts_with_serve
        progress("court serve detection: tracking near player in court metres")
        court = Court.calibrate(*cfg.court_corners)
        pt = PlayerTracker(ch.detector).court_track(input_path, court, fps_a=cfg.serve_track_fps)
        before = list(segments)
        segments = refine_starts_with_serve(
            segments, ch.onsets, pt.t, pt.cy, pt.speed,
            lookback_s=cfg.serve_setup_lookback_s, baseline_y=cfg.serve_baseline_y_m,
            still_speed=cfg.serve_still_speed_mps, preroll_s=cfg.serve_setup_preroll_s,
            max_lead_s=cfg.serve_max_lead_s)
        moved = sum(1 for (a, _), (b, _) in zip(before, segments) if b < a - 0.3)
        ch.stages["serve_anchor"] = {"status": "used", "moved": moved,
                                     "candidates": len(segments)}
        progress(f"  serve set-up moved {moved}/{len(segments)} point starts to the serve")
    except Exception as exc:  # pragma: no cover
        ch.stages["serve_anchor"] = {"status": "failed", "reason": str(exc)}
        progress(f"  court serve detection failed: {exc}")
        if not cfg.allow_degraded:
            raise RuntimeError(f"court serve detection failed: {exc}") from exc
    return segments


def _resolve_court(input_path, cfg, progress):
    """Court homography from (in priority order) manual corners, then auto-detection.

    Returns a :class:`~rally.signals.court.Court` or ``None`` (the arbiter degrades to
    in-play + bounce-count structure without a court)."""
    # Explicit manual calibration wins — it's the user's stated intent and the most
    # reliable; only auto-detect when no corners were given.
    if cfg.court_corners is not None:
        from .signals.court import Court
        return Court.calibrate(*cfg.court_corners)
    if cfg.court_auto:
        try:
            from .signals.court_detect import detect_court
            court = detect_court(input_path, cfg, progress=progress)
            if court is not None:
                progress("  court auto-detected")
                return court
            progress("  court auto-detection found no court -> in-play + bounce only")
        except Exception as exc:  # pragma: no cover
            progress(f"  court auto-detection error: {exc}")
    return None


def _recover_player_hint_points(report, ch, cfg) -> List[Segment]:
    """Recover audio-missed points from independent player and ball structure.

    Fragmentation is missing evidence, not proof of a dead ball. A recovery is allowed
    only when a stable-to-active player hint precedes a well-measured live-ball component
    that contains normal rally structure. The later match stage re-tracks the serve window
    and requires the stationary-baseline/serve event before publication.
    """
    recovered: List[Segment] = []
    records: list[dict] = []
    for candidate in report.candidates:
        verdict = candidate.verdict
        selected = getattr(verdict, "selected_component", None)
        if (verdict.state != "indeterminate"
                or getattr(verdict, "reason_code", "") != "fragmented_live_track"
                or selected is None
                # A fragmented component is allowed to be shorter than a direct accept:
                # the subsequent dedicated serve check is the second independent gate.
                or getattr(verdict, "in_play_span_s", 0.0)
                < min(1.0, cfg.arbiter_min_in_play_span_s)
                or getattr(verdict, "measured_coverage", 0.0)
                < cfg.arbiter_player_recovery_min_coverage
                or (getattr(verdict, "n_bounces", 0)
                    < min(1, cfg.arbiter_min_bounces)
                    and getattr(verdict, "n_net_crossings", 0) < 1)):
            continue
        hints = [
            float(hint) for hint in ch.player_serve_hints
            if candidate.candidate[0] - cfg.arbiter_pre_pad_s <= hint
            <= candidate.candidate[1] + 1e-9
            and hint <= selected[0] + cfg.match_receiver_reaction_lag_s
            and selected[0] - hint <= cfg.match_fragment_merge_gap_s
        ]
        if not hints:
            continue
        contact = max(hints)
        start = max(0.0, contact - cfg.toss_preroll_s)
        end = float(selected[1]) + cfg.ball_tail_s
        if end - start < cfg.min_rally_s:
            continue
        point = (start, end)
        recovered.append(point)
        records.append({
            "candidate": [round(float(v), 3) for v in candidate.candidate],
            "serve_hint": round(contact, 3),
            "ball_component": [round(float(v), 3) for v in selected],
            "output": [round(start, 3), round(end, 3)],
            "reason": "stationary setup plus fragmented rally structure",
        })
    if records:
        ch.stages.setdefault("player_activity_proposal", {
            "status": "used", "serve_hints": [
                round(float(value), 3) for value in ch.player_serve_hints],
        })["recoveries"] = records
    return recovered


def _ball_arbiter(input_path, segments, ch, cfg, progress, *, weights=None,
                  cancel_check: CancelCheck = lambda: None) -> List[Segment]:
    """Ball-primary decision: track the ball inside each candidate window and keep only
    the real rallies, bounded to their serve start / point-end (see fusion.ball_verify)."""
    if not cfg.ball_arbiter:
        return segments
    if not segments:
        ch.stages["ball_arbiter"] = {"status": "skipped", "reason": "no candidates"}
        return segments
    from .signals.ball import discover_ball_weights
    weights = weights or cfg.ball_weights or discover_ball_weights()
    if not weights:
        progress("  ball arbiter: no TrackNet weights (set --ball-weights or drop one in "
                 "models/) -> skipping validation")
        ch.stages["ball_arbiter"] = {"status": "unavailable", "reason": "no weights"}
        return segments
    try:
        from .fusion.ball_verify import verify_segments_detailed
        from .signals.ball import resolve_device
        court = _resolve_court(input_path, cfg, progress)
        ch.court = court
        if court is None:
            progress("  ball arbiter: no court -> in-play + bounce structure only")
        dev = resolve_device()
        slow = " — slow" if str(dev) == "cpu" else ""
        progress(f"ball arbiter: validating candidates by ball trajectory on {dev}{slow}")
        verdict_kwargs = dict(
            min_speed_px_s=cfg.arbiter_min_speed_px_s, min_conf=cfg.arbiter_min_conf,
            min_in_play_frac=cfg.arbiter_min_in_play_frac,
            min_in_play_span_s=cfg.arbiter_min_in_play_span_s,
            max_fragment_join_gap_s=cfg.arbiter_fragment_join_gap_s,
            min_bounces=cfg.arbiter_min_bounces, min_rally_s=cfg.min_rally_s,
            toss_preroll_s=cfg.toss_preroll_s, tail_s=cfg.ball_tail_s,
        )
        serve_times = np.unique(np.r_[ch.onsets, ch.player_serve_hints])
        report = verify_segments_detailed(
            input_path, segments, court=court, weights_path=weights,
            pre_pad_s=cfg.arbiter_pre_pad_s, post_pad_s=cfg.arbiter_post_pad_s,
            max_extend_s=cfg.ball_max_extend_s, verdict_kwargs=verdict_kwargs,
            inference_batch_size=cfg.ball_inference_batch_size,
            serve_times=serve_times,
            require_serve_evidence=cfg.arbiter_require_serve_evidence,
            progress=progress, cancel_check=cancel_check)
        player_recoveries = _recover_player_hint_points(report, ch, cfg)
        segments = sorted([*report.segments, *player_recoveries])
        ch.arbiter_accepted_regions = [
            tuple(candidate.candidate) for candidate in report.candidates
            if candidate.verdict.state == "accept"
        ]
        ch.arbiter_indeterminate_regions = [
            tuple(candidate.candidate) for candidate in report.candidates
            if candidate.verdict.state == "indeterminate"
        ]
        ch.arbiter_rejected_regions = [
            # Suppress fallback only where the verdict actually evaluated contradictory
            # evidence. A broad recall-oriented proposal can contain several audio points;
            # rejecting one local core does not justify deleting all of them.
            tuple(candidate.verdict.evidence_core
                  or candidate.verdict.selected_component
                  or candidate.candidate)
            for candidate in report.candidates
            if candidate.verdict.state == "reject"
        ]
        ch.ball_end_hints = [
            (tuple(candidate.candidate), float(
                getattr(candidate.verdict, "trajectory_end_hint")))
            for candidate in report.candidates
            if getattr(candidate.verdict, "trajectory_end_hint", None) is not None
        ]
        ch.stages["ball_arbiter"] = {
            "status": "used", "weights": str(weights), "court": court is not None,
            "weights_sha256": _file_sha256(str(weights)),
            "device": str(dev),
            "inference_batch_size": cfg.ball_inference_batch_size,
            "serve_evidence_required": cfg.arbiter_require_serve_evidence}
        ch.stages["ball_arbiter"]["verification"] = report.as_dict()
    except Exception as exc:  # pragma: no cover
        ch.stages["ball_arbiter"] = {"status": "failed", "reason": str(exc),
                                     "weights": str(weights),
                                     "fallback_policy": "coherent_audio_only"}
        if cfg.allow_degraded:
            progress(f"  ball arbiter failed; using coherent audio fallback only: {exc}")
            ch.arbiter_indeterminate_regions = list(segments)
            # These are broad, recall-oriented computational proposals, not publishable
            # points. Mark them indeterminate so the whole-point resolver can recover
            # coherent audio, but never return them as ball-primary output.
            return []
        # Once a checkpoint is present, silently publishing unvalidated candidates makes
        # an "accurate" run indistinguishable from a failed fallback. Fail the run instead.
        raise RuntimeError(f"ball arbiter failed with {weights}: {exc}") from exc
    return segments


def _trim_ball_ends(input_path, segments, ch, cfg, progress,
                    cancel_check: CancelCheck = lambda: None) -> List[Segment]:
    """Opt-in (TrackNet + calibration): trim each rally end to the point-ending bounce."""
    if not (cfg.ball_weights and cfg.court_corners and ch.onsets.size and segments):
        return segments
    try:
        from .fusion.ball_end import refine_ends_with_ball
        from .signals.ball import resolve_device
        from .signals.court import Court
        dev = resolve_device()
        slow = " — slow" if str(dev) == "cpu" else ""
        progress(f"ball point-end: tracking ball over each rally on {dev}{slow}")
        court = Court.calibrate(*cfg.court_corners)
        segments = refine_ends_with_ball(
            input_path, segments, court, cfg.ball_weights,
            min_rally_s=cfg.min_rally_s, tail_s=cfg.ball_tail_s,
            max_extend_s=cfg.ball_max_extend_s,
            inference_batch_size=cfg.ball_inference_batch_size,
            progress=progress, cancel_check=cancel_check)
        ch.stages["ball_end"] = {"status": "used", "weights": cfg.ball_weights}
    except Exception as exc:  # pragma: no cover
        ch.stages["ball_end"] = {"status": "failed", "reason": str(exc)}
        progress(f"  ball point-end failed: {exc}")
        if not cfg.allow_degraded:
            raise RuntimeError(f"ball point-end failed: {exc}") from exc
    return segments


def _filter_nonplay(segments, cfg, progress) -> List[Segment]:
    """Drop warm-up (before skip_intro) and temporally isolated (non-match) points."""
    n_before = len(segments)
    if cfg.skip_intro_s > 0:
        segments = [(s, e) for s, e in segments if s >= cfg.skip_intro_s]
    if cfg.drop_isolated:
        segments = drop_isolated_points(segments, cfg.isolation_gap_s)
    if len(segments) < n_before:
        progress(f"  non-play filter dropped {n_before - len(segments)} isolated/intro points")
    return segments


def _validate_match_state(input_path, segments, ch, cfg, progress, *, weights=None,
                          cancel_check: CancelCheck = lambda: None) -> List[Segment]:
    """Require an observed serve event inside match play, never receiver pose alone."""
    if cfg.play_mode == "casual":
        ch.stages["match_state"] = {
            "status": "skipped", "mode": "casual",
            "reason": "tennis match sequence rules disabled for casual rallying",
        }
        return segments
    if not segments or (ch.onsets.size == 0 and ch.player_serve_hints.size == 0):
        ch.stages["match_state"] = {
            "status": "skipped", "mode": cfg.play_mode,
            "reason": "no point candidates or strike times",
        }
        return segments
    try:
        from .fusion.match_state import validate_match_sequence
        from .signals.player import observe_position_setups, observe_serve_setups
        from .signals.serve import observe_ball_serves

        progress("match-state validation: checking server pose near early impacts")
        # Audio-missed far-side serves use a player-reaction hint only as a probe time.
        # The hint does not confirm anything: pose/position and TrackNet still evaluate it
        # under the same match rules as an acoustic impact.
        validation_onsets = np.unique(np.r_[ch.onsets, ch.player_serve_hints])
        observations = observe_serve_setups(
            input_path, segments, validation_onsets, cfg, cancel_check=cancel_check)
        progress("match-state validation: checking stationary baseline player setup")
        position_observations = observe_position_setups(
            segments,
            validation_onsets,
            ch.player_samples,
            cfg,
            court=ch.court,
            frame_size=ch.frame_size,
        )
        if len(position_observations) != len(observations):
            raise RuntimeError("position observations do not align with point candidates")
        observations = [
            replace(
                pose,
                position_checked=position.checked,
                position_setup_evidence=position.setup_evidence,
                position_best_strike=position.best_strike,
                position_setup_strikes=position.setup_strikes,
                position_score=position.score,
                position_server_end=position.server_end,
                position_server_span=position.server_span,
                position_player_tracks=position.player_tracks,
                position_stable_tracks=position.stable_tracks,
                position_stable_fraction=position.stable_fraction,
                setup_evidence=(pose.setup_evidence or position.setup_evidence),
            )
            for pose, position in zip(observations, position_observations)
        ]
        if weights:
            progress("match-state validation: checking ball toss/serve motion")
            ball_observations = observe_ball_serves(
                input_path, segments, validation_onsets, weights, cfg,
                cancel_check=cancel_check)
            if len(ball_observations) != len(observations):
                raise RuntimeError("ball serve observations do not align with point candidates")
            observations = [
                replace(
                    pose,
                    ball_checked=ball.checked,
                    ball_serve_evidence=ball.confirmed,
                    ball_best_strike=ball.best_strike,
                    ball_coverage=ball.coverage,
                    ball_vertical_span=ball.vertical_span,
                    ball_outgoing_span=ball.outgoing_span,
                    ball_ordered_evidence=ball.ordered,
                    ball_measured_samples=ball.measured_samples,
                )
                for pose, ball in zip(observations, ball_observations)
            ]
        elif cfg.play_mode == "match" and not cfg.allow_degraded:
            raise RuntimeError(
                "match mode requires TrackNet weights to confirm far-side serves")
        protected = {
            index for index, point in enumerate(segments)
            if any(_point_fully_evaluated(point, region, ch.onsets)
                   for region in ch.arbiter_accepted_regions)
        }
        before = len(segments)
        segments, stage = validate_match_sequence(
            segments, validation_onsets, observations, cfg,
            protected_indices=protected,
        )
        # A credible rally trajectory does not prove it started with a serve (warm-up and
        # cooperative feeds also rally). Keep this for diagnostics, never as a bypass.
        stage["trajectory_accepted_indices"] = sorted(protected)
        stage["player_serve_hints"] = [
            round(float(value), 3) for value in ch.player_serve_hints]
        stage["pose_device"] = getattr(ch.detector, "device", None)
        stage["ball_inference_batch_size"] = cfg.ball_inference_batch_size
        ch.stages["match_state"] = stage
        removed = before - len(segments)
        if removed:
            progress(f"  serve validator rejected {removed} candidate(s) without a serve")
        else:
            progress("  match-state sequence is consistent or inconclusive")
    except Exception as exc:  # pragma: no cover - optional heavy dependency/runtime
        ch.stages["match_state"] = {
            "status": "failed", "mode": cfg.play_mode, "reason": str(exc),
        }
        progress(f"  match-state validation failed: {exc}")
        if cfg.play_mode == "match" and not cfg.allow_degraded:
            raise RuntimeError(f"match-state validation failed: {exc}") from exc
    return segments


def _write_output(input_path, output_path, json_path, result, info, cfg, progress) -> None:
    """Atomically publish video first and its describing JSON sidecar last."""
    def write_sidecar() -> None:
        if not json_path:
            return
        sidecar_path = Path(json_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = sidecar_path.with_name(f".{sidecar_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp_json.open("w") as fh:
                json.dump(result.sidecar(), fh, indent=2)
            os.replace(tmp_json, sidecar_path)
        finally:
            tmp_json.unlink(missing_ok=True)
        progress(f"wrote {json_path}")

    if not output_path:
        write_sidecar()
        return
    segments = result.segments
    if not segments:
        Path(output_path).unlink(missing_ok=True)
        progress("no rally segments found -> not writing output video")
    else:
        render_segments = add_real_context(
            segments, info.duration_s,
            cfg.point_start_buffer_s, cfg.point_end_buffer_s)
        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex}.tmp{dst.suffix or '.mp4'}")
        try:
            if cfg.reencode and (cfg.label_points or cfg.inter_point_gap_s > 0):
                font = find_font() if cfg.label_points else None
                if cfg.label_points and font is None:
                    progress("  no font found -> labels drawn with ffmpeg's default font")
                what = "labelled points" if cfg.label_points else "points"
                progress(f"rendering {len(segments)} {what} -> {output_path}")
                render_labeled(
                    input_path, render_segments, str(tmp_video),
                    gap_s=cfg.inter_point_gap_s,
                    label_prefix=cfg.label_prefix,
                    font=font,
                    video_height=info.height,
                    has_audio=info.has_audio,
                    draw_labels=cfg.label_points,
                )
            else:
                progress(f"cutting {len(segments)} segments -> {output_path}")
                cut_segments(input_path, render_segments, str(tmp_video), reencode=cfg.reencode)
            if not tmp_video.exists() or tmp_video.stat().st_size <= 0:
                raise RuntimeError("video renderer produced no output")
            os.replace(tmp_video, dst)
            progress(f"wrote {output_path}")
        finally:
            tmp_video.unlink(missing_ok=True)
    # Publishing metadata last ensures a successful sidecar never describes a video that
    # failed halfway through rendering. Both individual files are replaced atomically.
    write_sidecar()


# --------------------------------------------------------------------------- #
# timeline helpers (single source of truth for the analysis grid)             #
# --------------------------------------------------------------------------- #
def timeline_array(duration: float, cfg: RallyConfig) -> np.ndarray:
    return np.arange(0.0, duration, 1.0 / cfg.analysis_fps)


def timeline_size(duration: float, cfg: RallyConfig) -> int:
    return timeline_array(duration, cfg).size


def _validate_paths(input_path: str, output_path: Optional[str], json_path: Optional[str]) -> None:
    """Prevent source/output/sidecar aliases, including symlinks and hardlinks."""
    named = [("input", input_path), ("output", output_path), ("json", json_path)]
    present = [(name, Path(path)) for name, path in named if path]

    def aliases(a: Path, b: Path) -> bool:
        try:
            if a.exists() and b.exists() and os.path.samefile(a, b):
                return True
        except OSError:
            pass
        return a.resolve(strict=False) == b.resolve(strict=False)

    for i, (name_a, path_a) in enumerate(present):
        for name_b, path_b in present[i + 1:]:
            if aliases(path_a, path_b):
                raise ValueError(
                    f"{name_a} and {name_b} paths must differ (would overwrite data): {path_a}")


def trim(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[RallyConfig] = None,
    *,
    json_path: Optional[str] = None,
    detect_players: bool = True,
    progress: Progress = lambda _msg: None,
    cancel_check: CancelCheck = lambda: None,
) -> RallyResult:
    """Analyse ``input_path`` and (if ``output_path`` given) write the trimmed video.

    Expected unavailable optional sources abstain. Unexpected failures in enabled stages
    are fatal unless ``cfg.allow_degraded`` explicitly permits a partial result.
    """
    cfg = cfg or RallyConfig()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    _validate_paths(input_path, output_path, json_path)

    cancel_check()
    progress(f"probing {input_path}")
    info = probe(input_path)
    duration = info.duration_s
    if duration <= 0:
        raise RuntimeError("could not determine video duration")

    timeline = timeline_array(duration, cfg)
    progress(f"duration={duration:.1f}s  analysis frames={timeline.size}  "
             f"({cfg.analysis_fps} fps)")

    # ---- gather channels ----------------------------------------------------
    ch = _Channels(frame_size=(int(info.width), int(info.height)))
    _audio_channel(input_path, info, timeline, cfg, ch, progress, cancel_check)
    cancel_check()
    _visual_channels(input_path, cfg, timeline, detect_players, ch, progress, cancel_check)
    cancel_check()
    if not ch.used:
        raise RuntimeError(
            "no usable channels (need at least an audio track or OpenCV) — cannot segment"
        )
    _ball_channel(input_path, timeline, cfg, ch, progress, cancel_check)
    cancel_check()
    _pose_channel(input_path, timeline, cfg, ch, progress, cancel_check)
    cancel_check()
    failures = {name: stage for name, stage in ch.stages.items()
                if stage.get("status") == "failed"}
    if failures and not cfg.allow_degraded:
        detail = "; ".join(f"{name}: {stage.get('reason', 'unknown error')}"
                           for name, stage in failures.items())
        raise RuntimeError(f"enabled analysis stage failed ({detail}); "
                           "set allow_degraded=True to accept a partial result")

    # ---- decide + refine ----------------------------------------------------
    arbiter_weights = None
    if cfg.ball_arbiter:
        from .signals.ball import discover_ball_weights
        arbiter_weights = cfg.ball_weights or discover_ball_weights()
        if not arbiter_weights:
            ch.stages["ball_arbiter"] = {"status": "unavailable", "reason": "no weights"}
    use_ball_arbiter = bool(arbiter_weights)
    segments = _derive_points(
        ch, duration, cfg, progress, for_ball_arbiter=use_ball_arbiter)
    if use_ball_arbiter:
        # Candidate pre-padding and serve_times let the detailed trajectory verifier find
        # the serve. Moving starts after proposal budgeting would invalidate its workload
        # cap and can enter omitted intervals.
        ch.stages["serve_anchor"] = {
            "status": "skipped", "reason": "ball arbiter owns serve boundary recovery"}
    else:
        segments = _anchor_serves(input_path, segments, ch, cfg, progress)
    if cfg.ball_arbiter:
        # ball-primary: the trajectory validates each candidate and sets its bounds
        segments = _ball_arbiter(
            input_path, segments, ch, cfg, progress, weights=arbiter_weights,
            cancel_check=cancel_check)
        indeterminate_fallback, suppressed_rejects, superseded_accepts = (
            _indeterminate_audio_fallback(ch))
        all_fallback = sorted([*ch.arbiter_audio_fallback, *indeterminate_fallback])
        if all_fallback:
            primary_count = len(segments)
            partial_primary_count = sum(
                any(_overlaps(primary, fallback) for fallback in all_fallback)
                for primary in segments
            )
            segments = _merge_point_sources(segments, all_fallback)
            fallback_stage = ch.stages["arbiter_audio_fallback"]
            fallback_stage.update({
                "status": "used", "ball_primary_points": primary_count,
                "combined_points": len(segments),
                "omitted_points_used": len(ch.arbiter_audio_fallback),
                "indeterminate_points_used": len(indeterminate_fallback),
                "explicit_reject_points_suppressed": suppressed_rejects,
                "accepted_audio_points_superseded": superseded_accepts,
                "partial_ball_points_deferred_to_audio": partial_primary_count,
            })
        elif "arbiter_audio_fallback" in ch.stages:
            ch.stages["arbiter_audio_fallback"].update({
                "status": "none_used",
                "indeterminate_points_used": 0,
                "explicit_reject_points_suppressed": suppressed_rejects,
                "accepted_audio_points_superseded": superseded_accepts,
            })
    else:
        # audio-primary (legacy): only trim rally ends by the ball, if configured
        segments = _trim_ball_ends(
            input_path, segments, ch, cfg, progress, cancel_check)
    cancel_check()
    segments = _apply_ball_end_hints(segments, ch, cfg)
    segments = _validate_match_state(
        input_path, segments, ch, cfg, progress, weights=arbiter_weights,
        cancel_check=cancel_check)
    cancel_check()
    segments = _filter_nonplay(segments, cfg, progress)

    kept = total_kept_seconds(segments)
    progress(f"decoded {len(segments)} points, {kept:.1f}s kept of {duration:.1f}s")

    result = RallyResult(
        input_path=input_path,
        output_path=output_path if segments else None,
        segments=segments,
        total_seconds=duration,
        kept_seconds=kept,
        compression_ratio=(kept / duration) if duration else 0.0,
        channels_used=ch.used,
        n_strikes=ch.n_strikes,
        strike_times=[float(t) for t in ch.onsets],
        stages=ch.stages,
        config=asdict(cfg),
    )

    _write_output(input_path, output_path, json_path, result, info, cfg, progress)
    return result
