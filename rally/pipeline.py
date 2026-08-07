"""Continuous all-player pose tennis point trimming with no audio or ball inference.

The pipeline has one direction of evidence:

    court -> persistent target-player identities -> shared pose timeline
        -> service/stroke events -> tennis point state -> positive between-point cut

RTMPose COCO-17 has no racket keypoint, so ordinary strokes are inferred from ordered
wrist motion on identified players.  Point discovery scans every tracked match player on
both court ends and can recover an exchange whose serve is outside the video or missed by
pose. Audio is only preserved in output; no bounce, line, let, or in/out claim is made.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import RallyConfig
from .fusion.points import total_kept_seconds
from .io.ffmpeg import probe
from .io.publish import write_output as _write_output
from .pipeline_types import PipelineState, RallyResult

Segment = tuple[float, float]
Progress = Callable[[str], None]
CancelCheck = Callable[[], None]
SignalCallback = Callable[[dict], None]


@contextmanager
def _timed(state: PipelineState, name: str, progress: Progress):
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        state.timings[name] = state.timings.get(name, 0.0) + elapsed
        progress(f"timing {name}: {elapsed:.3f}s")


def player_timeline_array(duration: float, cfg: RallyConfig) -> np.ndarray:
    """Times owned by the one expensive target-player identity pass."""
    return np.arange(0.0, duration, 1.0 / cfg.player_fps)


def _validate_paths(input_path: str, output_path: str | None, json_path: str | None) -> None:
    named = [("input", input_path), ("output", output_path), ("json", json_path)]
    present = [(name, Path(path)) for name, path in named if path]

    def aliases(left: Path, right: Path) -> bool:
        try:
            if left.exists() and right.exists() and os.path.samefile(left, right):
                return True
        except OSError:
            pass
        return left.resolve(strict=False) == right.resolve(strict=False)

    for index, (left_name, left) in enumerate(present):
        for right_name, right in present[index + 1:]:
            if aliases(left, right):
                raise ValueError(
                    f"{left_name} and {right_name} paths must differ: {left}")


def _resolve_court(input_path: str, cfg: RallyConfig, progress: Progress):
    if cfg.court_corners is not None:
        from .signals.court import Court

        return Court.calibrate(*cfg.court_corners)
    if not cfg.court_auto:
        return None
    try:
        from .signals.court_detect import detect_court

        court = detect_court(input_path, cfg, progress=progress)
        if court is not None:
            progress("  target court located")
            return court
    except Exception as exc:
        raise RuntimeError(f"court detection failed: {exc}") from exc
    raise RuntimeError("target court could not be located")


def _visual_channels(
    input_path: str,
    cfg: RallyConfig,
    timeline: np.ndarray,
    detect_players: bool,
    state: PipelineState,
    progress: Progress,
    cancel_check: CancelCheck,
) -> None:
    from .signals.player import PlayerDetector
    from .signals.visual import analyze_visual, opencv_available

    if not opencv_available():
        raise RuntimeError("OpenCV is required for the vision-first pipeline")
    detector = PlayerDetector(model=cfg.player_detection_model) if detect_players else None
    if detector is None or not detector.available:
        reason = "player detection was disabled" if detector is None else detector.error
        raise RuntimeError(f"target-player detector is unavailable: {reason}")
    progress("tracking target-court player identities")
    visual = analyze_visual(
        input_path,
        cfg,
        timeline,
        detector,
        court=state.court,
        progress=progress,
        cancel_check=cancel_check,
    )
    state.player_track_samples = visual.get("player_track_samples") or []
    state.frame_size = visual.get("frame_size") or state.frame_size
    if not state.player_track_samples:
        raise RuntimeError("no persistent target-court player tracks were measured")
    state.used.append("target_player_identities")
    state.stages["visual"] = {
        "status": "used",
        "detection_model": cfg.player_detection_model,
        "target_court_filtered": bool(visual.get("target_court_filtered")),
        "identity_tracking": "botsort_reid_plus_clothing",
        "tracking_frames": len(state.player_track_samples),
        "racket_observations": int(visual.get("racket_observations") or 0),
        "racket_detection": "same_pass_coco_tennis_racket_class",
        "device": detector.device,
    }


def _continuous_pose_stage(
    input_path: str,
    duration: float,
    state: PipelineState,
    cfg: RallyConfig,
    progress: Progress,
    cancel_check: CancelCheck,
    publish_stage: Callable[..., None],
) -> tuple[list[Segment], list[dict]]:
    """Build one shared pose timeline and decode all point/event stages from it."""
    from .fusion.tennis_state import decode_pose_points
    from .signals.player import create_pose_runtime
    from .signals.pose_timeline import (
        build_pose_timeline,
        detect_serves,
        detect_strokes,
    )

    pose_runtime = create_pose_runtime(cfg)
    progress("building shared all-player pose timeline")
    timeline = build_pose_timeline(
        input_path,
        duration,
        cfg,
        pose_runtime=pose_runtime,
        court=state.court,
        player_track_samples=state.player_track_samples,
        match=state.match_profile,
        progress_callback=progress,
        cancel_check=cancel_check,
    )
    between_intervals: list[list[float]] = []
    engaged_intervals: list[list[float]] = []
    between_start: float | None = None
    engaged_start: float | None = None
    prior_time = 0.0
    for frame in timeline.frames:
        frame_time = float(frame["time"])
        if frame.get("between_like") and between_start is None:
            between_start = frame_time
        elif not frame.get("between_like") and between_start is not None:
            between_intervals.append([round(between_start, 3), round(prior_time, 3)])
            between_start = None
        if frame.get("engaged_like") and engaged_start is None:
            engaged_start = frame_time
        elif not frame.get("engaged_like") and engaged_start is not None:
            engaged_intervals.append([round(engaged_start, 3), round(prior_time, 3)])
            engaged_start = None
        prior_time = frame_time
    if between_start is not None:
        between_intervals.append([round(between_start, 3), round(prior_time, 3)])
    if engaged_start is not None:
        engaged_intervals.append([round(engaged_start, 3), round(prior_time, 3)])
    state.stages["pose_timeline"] = {
        "status": "used",
        "backend": "rtmpose_coco17_in_tracker_owned_boxes",
        "coarse_fps": cfg.pose_timeline_fps,
        "refine_fps": cfg.pose_refine_fps,
        "refine_motion_windows_per_minute": cfg.pose_refine_motion_windows_per_minute,
        "coarse_frames": timeline.coarse_frames,
        "refined_frames": timeline.refined_frames,
        "pose_records": timeline.sampled_boxes,
        "actors": sorted(timeline.records_by_actor),
        "pose_records_by_actor": {
            actor: len(records)
            for actor, records in sorted(timeline.records_by_actor.items())
        },
        "frames_with_pose": sum(
            int(frame.get("visible_players", 0) > 0) for frame in timeline.frames),
        "frames_with_both_ends": sum(
            set(frame.get("ends") or []) == {"near", "far"}
            for frame in timeline.frames),
        "between_like_intervals": between_intervals,
        "engaged_like_intervals": engaged_intervals,
        "reused_player_detections": True,
        "audio_used": False,
        "ball_tracking_used": False,
    }
    publish_stage("pose_timeline")
    serves = detect_serves(timeline, cfg)
    raw_actions, stroke_episodes = detect_strokes(timeline, serves, cfg)
    segments, points, reports = decode_pose_points(
        timeline, serves, raw_actions, stroke_episodes, duration, cfg)
    state.stages["serve_pose"] = reports["serve_pose"]
    observations = state.stages["serve_pose"].get("observations") or []
    state.serve_times = np.asarray(
        [float(item["first_strike"]) for item in observations if item.get("accepted")],
        dtype=float,
    )
    publish_stage("serve_pose")
    state.stages["candidate_generation"] = reports["candidate_generation"]
    publish_stage("candidate_generation")
    state.stages["racket_actions"] = reports["racket_actions"]
    publish_stage("racket_actions", points=points, segments=segments)
    state.stages["endpoints"] = reports["endpoints"]
    publish_stage("endpoints", points=points, segments=segments)
    state.stages["quality_control"] = reports["quality_control"]
    publish_stage("quality_control", points=points, segments=segments)
    state.used.extend(["shared_all_player_pose_timeline", "pose_tennis_state_decoder"])
    if cfg.skip_intro_s > 0:
        retained = [
            (segment, point) for segment, point in zip(segments, points, strict=True)
            if segment[0] >= cfg.skip_intro_s
        ]
        segments = [item[0] for item in retained]
        points = [item[1] for item in retained]
    return segments, points


def trim(
    input_path: str,
    output_path: str | None = None,
    cfg: RallyConfig | None = None,
    *,
    json_path: str | None = None,
    detect_players: bool = True,
    progress: Progress = lambda _message: None,
    cancel_check: CancelCheck = lambda: None,
    signal_callback: SignalCallback = lambda _snapshot: None,
) -> RallyResult:
    cfg = cfg or RallyConfig()
    started = time.perf_counter()
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    _validate_paths(input_path, output_path, json_path)
    state = PipelineState()

    progress(f"probing {input_path}")
    with _timed(state, "probe", progress):
        info = probe(input_path)
    duration = float(info.duration_s)
    if duration <= 0:
        raise RuntimeError("could not determine video duration")
    timeline = player_timeline_array(duration, cfg)
    state.frame_size = (int(info.width), int(info.height))

    def publish_signals(
        current_stage: str, *, points: list[dict] | None = None,
        segments: list[Segment] | None = None,
    ) -> None:
        """Expose only completed, JSON-compatible evidence to an optional observer."""
        snapshot = {
            "current_stage": current_stage,
            "total_seconds": round(duration, 3),
            "n_serves": int(state.serve_times.size),
            "serve_times": [round(float(value), 3) for value in state.serve_times],
            "stages": deepcopy(state.stages),
            "match": deepcopy(state.match_profile),
            "points": deepcopy(points or []),
            "segments": [
                {"index": index, "start": round(start, 3), "end": round(end, 3),
                 "duration": round(end - start, 3)}
                for index, (start, end) in enumerate(segments or [])
            ],
            "timings_seconds": {
                name: round(float(seconds), 3)
                for name, seconds in state.timings.items()
            },
        }
        try:
            signal_callback(snapshot)
        except Exception as exc:  # noqa: BLE001 - inspector failure must not fail analysis
            progress(f"signal inspector publication skipped: {exc}")

    state.stages["audio"] = {
        "status": "disabled",
        "reason": "mixed-court audio cannot be attributed to the target court",
        "preserved_in_output": bool(info.has_audio),
    }
    publish_signals("probe")

    cancel_check()
    with _timed(state, "court_detection", progress):
        state.court = _resolve_court(input_path, cfg, progress)
    state.stages["court"] = {
        "status": "used" if state.court is not None else "unavailable",
        "source": "manual" if cfg.court_corners is not None else "automatic",
        **({"corners": np.asarray(state.court.corners_img).round(3).tolist()}
           if state.court is not None else {}),
    }
    publish_signals("court")
    cancel_check()
    with _timed(state, "visual", progress):
        _visual_channels(
            input_path, cfg, timeline, detect_players, state, progress, cancel_check)
    publish_signals("visual")

    if state.court is not None and state.frame_size is not None:
        from .fusion.player_identity import identify_match_players, infer_match_format

        state.match_format_evidence = infer_match_format(
            state.player_track_samples,
            state.court,
            state.frame_size,
        )
        inferred = str(state.match_format_evidence.get("format", "unknown"))
        state.match_format = inferred if inferred in {"singles", "doubles"} else "unknown"
        state.match_profile = identify_match_players(
            court=state.court,
            frame_size=state.frame_size,
            player_track_samples=state.player_track_samples,
            format_evidence=state.match_format_evidence,
        )
    state.stages["match_format"] = {
        "status": "used" if state.match_format != "unknown" else "indeterminate",
        **state.match_format_evidence,
    }
    publish_signals("match_format")

    cancel_check()
    with _timed(state, "pose_timeline_and_decode", progress):
        segments, points = _continuous_pose_stage(
            input_path, duration, state, cfg, progress, cancel_check,
            publish_signals)

    kept = total_kept_seconds(segments)
    state.timings["analysis_total"] = time.perf_counter() - started
    progress(f"decoded {len(segments)} points, {kept:.1f}s kept of {duration:.1f}s")
    result = RallyResult(
        input_path=input_path,
        output_path=output_path if segments else None,
        segments=segments,
        total_seconds=duration,
        kept_seconds=kept,
        compression_ratio=kept / duration if duration else 0.0,
        channels_used=state.used,
        n_serves=int(state.serve_times.size),
        serve_times=[float(value) for value in state.serve_times],
        stages=state.stages,
        timings=state.timings,
        config=asdict(cfg),
        match=state.match_profile,
        points=points,
    )
    _write_output(
        input_path,
        output_path,
        json_path,
        result,
        info,
        cfg,
        progress,
        pipeline_started=started,
    )
    return result
