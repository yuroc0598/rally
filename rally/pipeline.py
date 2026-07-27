"""End-to-end orchestration: video in -> rally-only video (+ JSON sidecar) out.

``trim`` is a thin orchestrator; each analysis/refinement stage is its own
function below so the flow reads top-to-bottom:

    probe -> [audio | visual | ball | pose] channels -> co-decide (fuse)
          -> derive points -> anchor serves
          -> ball arbiter: validate + bound each candidate   (default; ball_arbiter)
             OR legacy trim-ball-ends                          (when ball_arbiter off)
          -> drop non-play -> write sidecar + cut
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import RallyConfig
from .io.ffmpeg import cut_segments, find_font, load_audio_mono, probe, render_labeled
from .signals.audio import detect_strikes, strike_rhythm_features
from .fusion.score import rally_probability
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

    def sidecar(self) -> dict:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "total_seconds": round(self.total_seconds, 3),
            "kept_seconds": round(self.kept_seconds, 3),
            "compression_ratio": round(self.compression_ratio, 4),
            "n_rallies": len(self.segments),
            "n_strikes": self.n_strikes,
            "channels_used": self.channels_used,
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
    detector: Optional[object] = None   # shared YOLO PlayerDetector (loaded once)


# --------------------------------------------------------------------------- #
# analysis stages: each fills part of _Channels                               #
# --------------------------------------------------------------------------- #
def _audio_channel(input_path, info, timeline, cfg, ch, progress) -> None:
    """Ball-strike detection + strike-rhythm features (the primary rally cue)."""
    if not info.has_audio:
        return
    try:
        progress("decoding audio and detecting ball strikes")
        pcm = load_audio_mono(input_path, cfg.audio_sr)
        ch.onsets = detect_strikes(pcm, cfg.audio_sr, cfg)
        ch.n_strikes = int(ch.onsets.size)
        progress(f"  {ch.n_strikes} strikes detected")
        # Only vote when there are strikes. With zero strikes the rhythm features are all
        # zero; adding them as a confident-0 channel would drag the fused probability down
        # (a confident "no rally" everywhere) instead of abstaining. Let the other channels
        # decide when audio has nothing to say.
        if ch.n_strikes > 0:
            ch.audio_rate, ch.audio_reg = strike_rhythm_features(ch.onsets, timeline, cfg)
            ch.used.append("audio")
        else:
            progress("  no strikes -> audio abstains from the fusion")
    except Exception as exc:  # pragma: no cover
        progress(f"  audio channel failed: {exc}")


def _visual_channels(input_path, cfg, timeline, detect_players, ch, progress) -> None:
    """Frame-diff motion, camera-motion flag, player-geometry + near-player track."""
    from .signals.visual import analyze_visual, opencv_available

    if not opencv_available():  # pragma: no cover
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
        vis = analyze_visual(input_path, cfg, timeline, detector, progress=progress)
        ch.motion = vis["motion"]
        ch.camera_moving = vis["camera_moving"]
        ch.geometry = vis["geometry"]
        ch.near_track = vis.get("near_track")
        ch.used.append("motion")
        if ch.geometry is not None:
            ch.used.append("geometry")
    except Exception as exc:  # pragma: no cover
        progress(f"  visual channel failed: {exc}")


def _ball_channel(input_path, timeline, cfg, ch, progress) -> None:
    """Opt-in: track the ball over the whole video as a co-deciding vote."""
    if not (cfg.ball_channel and cfg.ball_weights and cfg.court_corners):
        return
    try:
        from .signals.ball import ball_in_play_channel, track_tracknet
        progress("ball-in-play channel: tracking ball over the whole video (slow on CPU)")
        btrack = track_tracknet(input_path, cfg.ball_weights)
        ch.ball = ball_in_play_channel(btrack, timeline)
        ch.used.append("ball")
    except Exception as exc:  # pragma: no cover
        progress(f"  ball channel failed: {exc}")


def _pose_channel(input_path, timeline, cfg, ch, progress) -> None:
    """Opt-in: near-player pose activity as a confidence-weighted vote."""
    if not cfg.player_pose:
        return
    try:
        from .signals.player import pose_activity_track
        progress("player pose channel: tracking pose over the video (slow)")
        ch.pose, ch.pose_conf = pose_activity_track(
            input_path, timeline, fps_a=cfg.player_pose_fps)
        ch.used.append("pose")
    except Exception as exc:  # pragma: no cover
        progress(f"  pose channel failed: {exc}")


# --------------------------------------------------------------------------- #
# decision + refinement stages: probability -> points -> refined points       #
# --------------------------------------------------------------------------- #
def _derive_points(ch, duration, cfg, progress) -> List[Segment]:
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
    regions = segments_from_prob(prob, cfg.analysis_fps, cfg, total_s=duration)
    onsets = ch.onsets
    serve_window = cfg.serve_attach_window_s if cfg.serve_attach else 0.0

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
        progress(f"  serve set-up moved {moved}/{len(segments)} point starts to the serve")
    except Exception as exc:  # pragma: no cover
        progress(f"  court serve detection failed: {exc}")
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


def _ball_arbiter(input_path, segments, ch, cfg, progress) -> List[Segment]:
    """Ball-primary decision: track the ball inside each candidate window and keep only
    the real rallies, bounded to their serve start / point-end (see fusion.ball_verify)."""
    if not (cfg.ball_arbiter and segments):
        return segments
    from .signals.ball import discover_ball_weights
    weights = cfg.ball_weights or discover_ball_weights()
    if not weights:
        progress("  ball arbiter: no TrackNet weights (set --ball-weights or drop one in "
                 "models/) -> skipping validation")
        return segments
    try:
        from .fusion.ball_verify import verify_segments
        court = _resolve_court(input_path, cfg, progress)
        if court is None:
            progress("  ball arbiter: no court -> in-play + bounce structure only")
        progress("ball arbiter: validating candidates by ball trajectory (slow on CPU)")
        verdict_kwargs = dict(
            min_speed_px_s=cfg.arbiter_min_speed_px_s, min_conf=cfg.arbiter_min_conf,
            min_in_play_frac=cfg.arbiter_min_in_play_frac,
            min_in_play_span_s=cfg.arbiter_min_in_play_span_s,
            min_bounces=cfg.arbiter_min_bounces, min_rally_s=cfg.min_rally_s,
            toss_preroll_s=cfg.toss_preroll_s, tail_s=cfg.ball_tail_s,
        )
        segments = verify_segments(
            input_path, segments, court=court, weights_path=weights,
            pre_pad_s=cfg.arbiter_pre_pad_s, post_pad_s=cfg.arbiter_post_pad_s,
            max_extend_s=cfg.ball_max_extend_s, verdict_kwargs=verdict_kwargs,
            progress=progress)
    except Exception as exc:  # pragma: no cover
        progress(f"  ball arbiter failed: {exc}")
    return segments


def _trim_ball_ends(input_path, segments, ch, cfg, progress) -> List[Segment]:
    """Opt-in (TrackNet + calibration): trim each rally end to the point-ending bounce."""
    if not (cfg.ball_weights and cfg.court_corners and ch.onsets.size and segments):
        return segments
    try:
        from .fusion.ball_end import refine_ends_with_ball
        from .signals.court import Court
        progress("ball point-end: tracking ball over each rally (slow on CPU)")
        court = Court.calibrate(*cfg.court_corners)
        segments = refine_ends_with_ball(
            input_path, segments, court, cfg.ball_weights,
            min_rally_s=cfg.min_rally_s, tail_s=cfg.ball_tail_s,
            max_extend_s=cfg.ball_max_extend_s, progress=progress)
    except Exception as exc:  # pragma: no cover
        progress(f"  ball point-end failed: {exc}")
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


def _write_output(input_path, output_path, json_path, result, info, cfg, progress) -> None:
    """Write the JSON sidecar and (if requested) render/cut the trimmed video."""
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(result.sidecar(), fh, indent=2)
        progress(f"wrote {json_path}")

    if not output_path:
        return
    segments = result.segments
    if not segments:
        progress("no rally segments found -> not writing output video")
    elif cfg.label_points or cfg.inter_point_gap_s > 0:
        font = find_font() if cfg.label_points else None
        if cfg.label_points and font is None:
            progress("  no font found -> labels drawn with ffmpeg's default font")
        what = "labelled points" if cfg.label_points else "points"
        progress(f"rendering {len(segments)} {what} -> {output_path}")
        render_labeled(
            input_path, segments, output_path,
            gap_s=cfg.inter_point_gap_s,
            label_prefix=cfg.label_prefix,
            font=font,
            video_height=info.height,
            has_audio=info.has_audio,
            draw_labels=cfg.label_points,
        )
        progress(f"wrote {output_path}")
    else:
        progress(f"cutting {len(segments)} segments -> {output_path}")
        cut_segments(input_path, segments, output_path, reencode=cfg.reencode)
        progress(f"wrote {output_path}")


# --------------------------------------------------------------------------- #
# timeline helpers (single source of truth for the analysis grid)             #
# --------------------------------------------------------------------------- #
def timeline_array(duration: float, cfg: RallyConfig) -> np.ndarray:
    return np.arange(0.0, duration, 1.0 / cfg.analysis_fps)


def timeline_size(duration: float, cfg: RallyConfig) -> int:
    return timeline_array(duration, cfg).size


def trim(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[RallyConfig] = None,
    *,
    json_path: Optional[str] = None,
    detect_players: bool = True,
    progress: Progress = lambda _msg: None,
) -> RallyResult:
    """Analyse ``input_path`` and (if ``output_path`` given) write the trimmed video.

    Degrades gracefully: uses whatever of {audio, motion+camera, player-geometry,
    ball, pose} is actually available/enabled in the environment.
    """
    cfg = cfg or RallyConfig()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    progress(f"probing {input_path}")
    info = probe(input_path)
    duration = info.duration_s
    if duration <= 0:
        raise RuntimeError("could not determine video duration")

    timeline = timeline_array(duration, cfg)
    progress(f"duration={duration:.1f}s  analysis frames={timeline.size}  "
             f"({cfg.analysis_fps} fps)")

    # ---- gather channels ----------------------------------------------------
    ch = _Channels()
    _audio_channel(input_path, info, timeline, cfg, ch, progress)
    _visual_channels(input_path, cfg, timeline, detect_players, ch, progress)
    if not ch.used:
        raise RuntimeError(
            "no usable channels (need at least an audio track or OpenCV) — cannot segment"
        )
    _ball_channel(input_path, timeline, cfg, ch, progress)
    _pose_channel(input_path, timeline, cfg, ch, progress)

    # ---- decide + refine ----------------------------------------------------
    segments = _derive_points(ch, duration, cfg, progress)
    segments = _anchor_serves(input_path, segments, ch, cfg, progress)
    if cfg.ball_arbiter:
        # ball-primary: the trajectory validates each candidate and sets its bounds
        segments = _ball_arbiter(input_path, segments, ch, cfg, progress)
    else:
        # audio-primary (legacy): only trim rally ends by the ball, if configured
        segments = _trim_ball_ends(input_path, segments, ch, cfg, progress)
    segments = _filter_nonplay(segments, cfg, progress)

    kept = total_kept_seconds(segments)
    progress(f"decoded {len(segments)} points, {kept:.1f}s kept of {duration:.1f}s")

    result = RallyResult(
        input_path=input_path,
        output_path=output_path,
        segments=segments,
        total_seconds=duration,
        kept_seconds=kept,
        compression_ratio=(kept / duration) if duration else 0.0,
        channels_used=ch.used,
        n_strikes=ch.n_strikes,
    )

    _write_output(input_path, output_path, json_path, result, info, cfg, progress)
    return result
