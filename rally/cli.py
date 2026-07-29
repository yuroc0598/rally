"""Command-line entry point:  python -m rally.cli INPUT [-o OUTPUT] [options]"""

from __future__ import annotations

import argparse
import math
import os
import sys

from .config import RallyConfig
from .pipeline import trim


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rally",
        description="Trim a tennis match recording down to rally (live-play) segments only.",
    )
    p.add_argument("input", help="input video file")
    p.add_argument("-o", "--output", help="output (trimmed) video file")
    p.add_argument("--json", dest="json_path", help="write segment metadata to this JSON file")

    p.add_argument("--analysis-fps", type=float, default=None, help="visual analysis frame rate")
    p.add_argument("--min-rally", type=float, default=None, help="drop rallies shorter than N seconds")
    p.add_argument("--pad-pre", type=float, default=None, help="lead-in padding per rally (s)")
    p.add_argument("--pad-post", type=float, default=None, help="lead-out padding per rally (s)")

    p.add_argument("--static-camera", action="store_true",
                   help="preset for fixed-tripod footage where motion is uninformative: "
                        "up-weight audio and widen the strike-rhythm window (improves recall)")
    p.add_argument("--play-mode", choices=("auto", "match", "casual"), default="auto",
                   help="tennis sequence rules: auto detects match-like runs, match enables "
                        "them explicitly, casual disables serve-side validation")
    p.add_argument("--no-players", action="store_true", help="disable YOLO player-geometry channel")

    p.add_argument("--no-labels", action="store_true", help="do not draw 'Point N' labels")
    p.add_argument("--no-snap-serve", action="store_true",
                   help="do not extend segment starts back to the serve")
    p.add_argument("--no-split", action="store_true",
                   help="do not split merged regions into individual points at strike gaps")
    p.add_argument("--no-movement-merge", action="store_true",
                   help="disable the movement gate (a short low-motion lull will split a point)")
    p.add_argument("--move-thresh", type=float, default=None,
                   help="near-player displacement (frac of frame) that counts as a between-point reset")
    p.add_argument("--min-rally-strikes", type=int, default=None,
                   help="effective strikes required for a real rally (coherence filter, default 2)")
    p.add_argument("--min-rally-dur", type=float, default=None,
                   help="minimum strike-span for a real rally, seconds (default 1.0)")
    p.add_argument("--skip-intro", type=float, default=None,
                   help="drop points starting before this time, seconds (skip warm-up)")
    p.add_argument("--keep-isolated", action="store_true",
                   help="keep temporally isolated points (disable non-play isolation filter)")
    p.add_argument("--court-corners", default=None,
                   help="fixed-camera court calibration for serve detection: 4 image points "
                        "'nlx,nly;nrx,nry;netRx,netRy;netLx,netLy' "
                        "(near-left & near-right baseline corners, then net∩right & net∩left sidelines)")
    p.add_argument("--calibration", default=None,
                   help="path to a court calibration JSON (from rally.tools.calibrate --save); "
                        "alternative to --court-corners")
    p.add_argument("--ball-weights", default=None,
                   help="path to a 3-frame TrackNet .pt: enables ball-based point-end "
                        "(trim rally ends at the double-bounce / out; needs calibration, slow on CPU)")
    p.add_argument("--ball-channel", action="store_true",
                   help="also use ball-in-play as a co-deciding rally channel over the whole "
                        "video (needs --ball-weights; very slow on CPU)")
    p.add_argument("--no-ball-arbiter", action="store_true",
                   help="disable ball-tracking validation (on by default: the ball trajectory "
                        "validates each candidate as a real rally and sets its serve / point-end. "
                        "Falls back to the faster audio-primary detector automatically when "
                        "TrackNet weights / PyTorch aren't installed)")
    p.add_argument("--no-court-auto", action="store_true",
                   help="disable automatic court detection (on by default; --court-corners/"
                        "--calibration override it; turn off if it locks onto the wrong lines)")
    p.add_argument("--require-serve-evidence", action="store_true",
                   help="precision mode: ball-validated candidates also need an audio strike near serve start")
    p.add_argument("--player-pose", action="store_true",
                   help="add player pose-activity as a confidence-weighted rally vote "
                        "(YOLOv8-pose over the video; slow on CPU)")
    p.add_argument("--gap", type=float, default=None,
                   help="optional black delay between points (default 0; normally leave off)")
    p.add_argument("--start-buffer", type=float, default=None,
                   help="real source footage before each detected point start (default 1.0; max 1.0)")
    p.add_argument("--end-buffer", type=float, default=None,
                   help="real source footage after each detected point end (default 1.0; max 1.0)")
    p.add_argument("--serve-preroll", type=float, default=None,
                   help="lead-in kept before the serve strike / toss (sets toss_preroll_s, "
                        "default 1.0, and serve_preroll_s used on the --no-split path)")
    p.add_argument("--tail", type=float, default=None,
                   help="real-footage tail after the last strike / point-end cue (default 1.0)")
    p.add_argument("--hysteresis", action="store_true",
                   help="use the simple hysteresis decoder instead of the duration-aware one")
    p.add_argument("--fast", action="store_true",
                   help="stream-copy cut (fast, keyframe-aligned; labels/gaps are omitted)")
    p.add_argument("--allow-degraded", action="store_true",
                   help="continue after an enabled analysis stage fails; the sidecar records failures")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    return p


def _config_from_args(args) -> RallyConfig:
    overrides = {"play_mode": args.play_mode}
    if args.static_camera:
        # Far-court impacts are quieter in fixed baseline recordings. A slightly lower
        # local SNR gate recovers short serve/return points without weakening the general
        # preset used for noisier handheld footage.
        overrides.update(
            w_audio=0.7, w_motion=0.1, rhythm_window_s=5.0,
            strike_snr_ratio=5.5,
        )
    if args.analysis_fps is not None:
        overrides["analysis_fps"] = args.analysis_fps
    if args.min_rally is not None:
        overrides["min_rally_s"] = args.min_rally
    if args.pad_pre is not None:
        overrides["pad_pre_s"] = args.pad_pre
    if args.pad_post is not None:
        overrides["pad_post_s"] = args.pad_post
    if args.hysteresis:
        overrides["use_dp_decoder"] = False
    if args.fast:
        overrides["reencode"] = False
    if args.allow_degraded:
        overrides["allow_degraded"] = True
    if args.no_labels:
        overrides["label_points"] = False
    if args.no_snap_serve:
        overrides["snap_serve"] = False
    if args.no_split:
        overrides["point_split"] = False
    if args.no_movement_merge:
        overrides["movement_merge"] = False
    if args.move_thresh is not None:
        overrides["move_thresh"] = args.move_thresh
    if args.gap is not None:
        overrides["inter_point_gap_s"] = args.gap
    if args.start_buffer is not None:
        overrides["point_start_buffer_s"] = args.start_buffer
    if args.end_buffer is not None:
        overrides["point_end_buffer_s"] = args.end_buffer
    if args.serve_preroll is not None:
        overrides["serve_preroll_s"] = args.serve_preroll
        overrides["toss_preroll_s"] = args.serve_preroll
    if args.tail is not None:
        overrides["landing_tail_s"] = args.tail
    if args.min_rally_strikes is not None:
        overrides["min_rally_strikes"] = args.min_rally_strikes
    if args.min_rally_dur is not None:
        overrides["min_rally_dur_s"] = args.min_rally_dur
    if args.skip_intro is not None:
        overrides["skip_intro_s"] = args.skip_intro
    if args.keep_isolated:
        overrides["drop_isolated"] = False
    if args.calibration:
        from .tools.calibrate import load_calibration
        try:
            overrides["court_corners"] = tuple(load_calibration(args.calibration))
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"--calibration: could not read {args.calibration}: {exc}")
    if args.court_corners:
        try:
            pts = tuple(tuple(float(v) for v in pair.split(","))
                        for pair in args.court_corners.split(";"))
        except ValueError:
            raise SystemExit("--court-corners: each point must be 'x,y' with numeric values")
        if len(pts) != 4 or any(len(p) != 2 for p in pts) \
                or any(not math.isfinite(v) for p in pts for v in p):
            raise SystemExit("--court-corners needs exactly 4 finite 'x,y' points separated by ';'")
        overrides["court_corners"] = pts
    if args.ball_weights:
        overrides["ball_weights"] = args.ball_weights
    if args.ball_channel:
        if not args.ball_weights:
            print("[rally] warning: --ball-channel ignored without --ball-weights", file=sys.stderr)
        overrides["ball_channel"] = True
    if args.no_ball_arbiter:
        overrides["ball_arbiter"] = False
    if args.no_court_auto:
        overrides["court_auto"] = False
    if args.require_serve_evidence:
        overrides["arbiter_require_serve_evidence"] = True
    if args.player_pose:
        overrides["player_pose"] = True
    return RallyConfig(**overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    progress = (lambda _m: None) if args.quiet else (lambda m: print(f"[rally] {m}", file=sys.stderr))

    if not args.output and not args.json_path:
        print("nothing to do: pass -o OUTPUT and/or --json PATH", file=sys.stderr)
        return 2
    if args.output and os.path.abspath(args.output) == os.path.abspath(args.input):
        print("output must differ from input (would overwrite the source)", file=sys.stderr)
        return 2

    result = trim(
        args.input,
        output_path=args.output,
        cfg=cfg,
        json_path=args.json_path,
        detect_players=not args.no_players,
        progress=progress,
    )

    print(
        f"{len(result.segments)} rallies | "
        f"kept {result.kept_seconds:.1f}s of {result.total_seconds:.1f}s "
        f"({result.compression_ratio * 100:.1f}%) | "
        f"channels: {', '.join(result.channels_used)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
