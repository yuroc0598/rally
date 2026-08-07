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

    p.add_argument("--pose-fps", type=float, default=None,
                   help="coarse all-player pose frame rate")
    p.add_argument("--min-rally", type=float, default=None, help="drop rallies shorter than N seconds")

    p.add_argument(
        "--player-detection-model", default=None,
        help="Ultralytics person detector name/path (default yolo12n.pt; env: "
             "RALLY_YOLO_DETECTION_MODEL)",
    )
    p.add_argument(
        "--player-pose-model", default=None,
        help="RTMPose ONNX path/URL (env: RALLY_PLAYER_POSE_MODEL)",
    )

    p.add_argument("--no-labels", action="store_true", help="do not draw 'Point N' labels")
    p.add_argument("--skip-intro", type=float, default=None,
                   help="drop points starting before this time, seconds (skip warm-up)")
    p.add_argument("--court-corners", default=None,
                   help="fixed-camera court calibration for serve detection: 4 image points "
                        "'nlx,nly;nrx,nry;netRx,netRy;netLx,netLy' "
                        "(near-left & near-right baseline corners, then net∩right & net∩left sidelines)")
    p.add_argument("--calibration", default=None,
                   help="path to a court calibration JSON (from rally.tools.calibrate --save); "
                        "alternative to --court-corners")
    p.add_argument("--no-court-auto", action="store_true",
                   help="disable automatic court detection (on by default; --court-corners/"
                        "--calibration override it; turn off if it locks onto the wrong lines)")
    p.add_argument("--court-weights", default=None,
                   help="optional Ultralytics court-keypoint checkpoint; validated before "
                        "the classical court-detector fallback")
    p.add_argument("--gap", type=float, default=None,
                   help="optional black delay between points (default 0; normally leave off)")
    p.add_argument("--start-buffer", type=float, default=None,
                   help="real source footage before each detected point start (default 0.25; max 1.0)")
    p.add_argument("--end-buffer", type=float, default=None,
                   help="real source footage after each detected point end (default 0.25; max 1.0)")
    p.add_argument("--fast", action="store_true",
                   help="stream-copy cut (fast, keyframe-aligned; labels/gaps are omitted)")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    return p


def _config_from_args(args) -> RallyConfig:
    overrides = {}
    if args.pose_fps is not None:
        overrides["pose_timeline_fps"] = args.pose_fps
    if args.min_rally is not None:
        overrides["min_rally_s"] = args.min_rally
    if args.fast:
        overrides["reencode"] = False
    if args.no_labels:
        overrides["label_points"] = False
    if args.gap is not None:
        overrides["inter_point_gap_s"] = args.gap
    if args.start_buffer is not None:
        overrides["point_start_buffer_s"] = args.start_buffer
    if args.end_buffer is not None:
        overrides["point_end_buffer_s"] = args.end_buffer
    if args.skip_intro is not None:
        overrides["skip_intro_s"] = args.skip_intro
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
    if args.no_court_auto:
        overrides["court_auto"] = False
    if args.court_weights is not None:
        overrides["court_weights"] = args.court_weights
    if args.player_detection_model is not None:
        overrides["player_detection_model"] = args.player_detection_model
    if args.player_pose_model is not None:
        overrides["player_pose_model"] = args.player_pose_model
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
        detect_players=True,
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
