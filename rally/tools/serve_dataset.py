"""Export serve-candidate clips for review/labelling — the first step of the trained
serve-detector path (see DESIGN.md, Phase-3).

Each detected point starts at (approximately) a serve, so every point in a run's
sidecar JSON is a serve candidate. This tool cuts a short clip around each candidate
so you can quickly confirm which are true serves (and mark near/far end). Those labels
then train a light serve classifier that generalises across the match.

Usage:
    python -m rally.tools.serve_dataset match.mp4 rallies.json candidates/ [--pre 1.5 --post 3.0]

Produces candidates/cand_00.mp4 ... and candidates/labels.csv (pre-filled, index+time),
which you edit to set is_serve (1/0) and end (near/far).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

from ..io.ffmpeg import _rel, _require


def candidates_from_audio(video: str, min_strikes: int = 2):
    """All serve candidates straight from audio: the first strike of each strike
    cluster (>= min_strikes), minus the toss pre-roll. Fast — no video decode."""
    import numpy as np

    from ..signals.audio import detect_strikes
    from ..config import RallyConfig
    from ..io.ffmpeg import load_audio_mono

    cfg = RallyConfig()
    on = detect_strikes(load_audio_mono(video, cfg.audio_sr), cfg.audio_sr, cfg)
    clusters = [[float(on[0])]]
    for t in on[1:]:
        (clusters[-1].append(float(t)) if t - clusters[-1][-1] <= cfg.point_gap_s
         else clusters.append([float(t)]))
    segs = []
    for i, c in enumerate([c for c in clusters if len(c) >= min_strikes]):
        segs.append({"index": i, "start": round(max(0.0, c[0] - cfg.toss_preroll_s), 2)})
    return segs


def export_serve_candidates(
    video: str, segments_json: str, out_dir: str, *,
    pre: float = 2.5, post: float = 3.5, reencode: bool = False
) -> int:
    ffmpeg = _require("ffmpeg")
    if segments_json == "auto":
        segs = candidates_from_audio(video)
    else:
        with open(segments_json) as fh:
            segs = json.load(fh)["segments"]
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for s in segs:
        i = s["index"]
        serve_t = s["start"]  # the point start ≈ serve (minus toss pre-roll)
        clip = os.path.join(out_dir, f"cand_{i:03d}.mp4")
        ss = max(0.0, serve_t - pre)
        if reencode:
            codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                     "-vf", "scale=640:-2", "-an"]
        else:
            # fast keyframe-aligned stream copy: seconds, not minutes, for hundreds of clips.
            # The wider pre/post window keeps the serve inside despite keyframe drift.
            codec = ["-c", "copy", "-an"]
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-ss", f"{ss:.3f}",
             "-to", f"{serve_t + post:.3f}", "-i", _rel(video), *codec, _rel(clip)],
            check=True,
        )
        rows.append({"index": i, "point_start_s": round(serve_t, 2),
                     "is_serve": "", "end": ""})  # you fill is_serve (1/0), end (near/far)

    labels = os.path.join(out_dir, "labels.csv")
    with open(labels, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "point_start_s", "is_serve", "end"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rally.tools.serve_dataset",
                                description="Export serve-candidate clips for labelling.")
    p.add_argument("video")
    p.add_argument("segments_json",
                   help="a rallies.json from 'rally.cli --json', or 'auto' to derive "
                        "all candidates straight from audio")
    p.add_argument("out_dir")
    p.add_argument("--pre", type=float, default=2.5)
    p.add_argument("--post", type=float, default=3.5)
    p.add_argument("--reencode", action="store_true",
                   help="re-encode small 640px clips (slower) instead of fast keyframe copy")
    args = p.parse_args(argv)
    n = export_serve_candidates(args.video, args.segments_json, args.out_dir,
                                pre=args.pre, post=args.post, reencode=args.reencode)
    print(f"exported {n} serve candidates to {args.out_dir}/  (edit labels.csv, then we train)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
