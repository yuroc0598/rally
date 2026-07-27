"""Thin wrappers over the system ffmpeg / ffprobe binaries.

Only the system binaries are required for the audio + cut path; numpy/OpenCV cover
the rest. Everything shells out so there is no hard dependency on pyav/moviepy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(f"'{binary}' not found on PATH — install ffmpeg")
    return path


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool


def _parse_fps(rate: str | None) -> float:
    """Parse an ffprobe frame-rate string ('num/den', a bare number, or 'N/A') to fps."""
    rate = (rate or "").strip()
    num, den = rate.split("/", 1) if "/" in rate else (rate, "1")
    try:
        num_f, den_f = float(num), float(den)
    except ValueError:
        return 0.0
    return num_f / den_f if den_f else 0.0


def probe(path: str) -> VideoInfo:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError(f"no video stream in {path}")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    fps = _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate"))
    duration = float(data["format"].get("duration", 0.0)) or float(v.get("duration", 0.0))
    return VideoInfo(
        duration_s=duration, fps=fps,
        width=int(v["width"]), height=int(v["height"]), has_audio=has_audio,
    )


def load_audio_mono(path: str, sr: int) -> np.ndarray:
    """Decode the audio track to a mono float32 numpy array via an ffmpeg pipe."""
    ffmpeg = _require("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def find_font() -> str | None:
    """Locate a sans-serif TTF for drawtext labels, or None."""
    import os

    candidates = [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if out and os.path.exists(out):
            return out
    except Exception:
        pass
    return None


def _rel(p: str) -> str:
    import os

    try:
        return os.path.relpath(p)
    except ValueError:
        return p


def _escape_drawtext(s: str) -> str:
    """Escape a value for use inside a drawtext filter (backslash, colon, quote, percent)."""
    return (s.replace("\\", "\\\\").replace(":", r"\:")
             .replace("'", r"\'").replace("%", r"\%"))


def render_labeled(
    src: str,
    segments: List[Tuple[float, float]],
    dst: str,
    *,
    gap_s: float = 0.4,
    label_prefix: str = "Point",
    font: str | None = None,
    video_height: int = 1080,
    has_audio: bool = True,
    draw_labels: bool = True,
) -> None:
    """Cut, number, and concatenate rallies in one re-encode pass via filter_complex.

    Each segment is opened with a fast seek (`-ss`/`-to` before `-i`), optionally captioned
    "``label_prefix`` N" in the top-left (``draw_labels``), given a short black tail gap,
    then all are concatenated. Requires a re-encode (drawtext cannot stream-copy). Audio is
    mapped only when the source has an audio stream (``has_audio``).
    """
    ffmpeg = _require("ffmpeg")
    if not segments:
        raise ValueError("no segments to render")

    fontsize = max(24, video_height // 18)
    n = len(segments)

    cmd = [ffmpeg, "-v", "error", "-y"]
    for (start, end) in segments:
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", _rel(src)]

    chains: List[str] = []
    concat_inputs = ""
    for i in range(n):
        add_gap = gap_s > 0 and i < n - 1  # no trailing gap after the last point
        v = f"[{i}:v]setpts=PTS-STARTPTS"
        if draw_labels:
            text = _escape_drawtext(f"{label_prefix} {i + 1}")
            dt = (f"drawtext=text='{text}':x=30:y=30:fontsize={fontsize}:"
                  f"fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=14")
            if font:
                dt = f"drawtext=fontfile='{_escape_drawtext(font)}':" + dt[len("drawtext="):]
            v += f",{dt}"
        if add_gap:
            v += f",tpad=stop_duration={gap_s:.3f}:stop_mode=add:color=black"
        chains.append(v + f"[v{i}]")
        concat_inputs += f"[v{i}]"
        if has_audio:
            a = f"[{i}:a]asetpts=PTS-STARTPTS"
            if add_gap:
                a += f",apad=pad_dur={gap_s:.3f}"
            chains.append(a + f"[a{i}]")
            concat_inputs += f"[a{i}]"

    a_streams = 1 if has_audio else 0
    chains.append(f"{concat_inputs}concat=n={n}:v=1:a={a_streams}[outv]"
                  + ("[outa]" if has_audio else ""))
    graph = ";".join(chains)

    cmd += ["-filter_complex", graph, "-map", "[outv]"]
    if has_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", _rel(dst)]
    subprocess.run(cmd, check=True)


def cut_segments(
    src: str, segments: List[Tuple[float, float]], dst: str, *, reencode: bool = True
) -> None:
    """Extract ``segments`` (seconds) from ``src`` and concatenate them into ``dst``.

    Frame-accurate re-encode by default; ``reencode=False`` uses stream-copy (fast,
    but cuts snap to the nearest keyframe).
    """
    ffmpeg = _require("ffmpeg")
    if not segments:
        raise ValueError("no segments to cut")

    import os
    import tempfile

    rel = _rel  # ffmpeg paths relative to cwd: valid everywhere, sandbox-safe

    # Put intermediates next to the output rather than /tmp: avoids filling a small
    # /tmp with large clips, and keeps everything on one filesystem for a fast concat.
    dst_dir = os.path.dirname(dst) or "."
    tmpdir = tempfile.mkdtemp(prefix=".rally_cut_", dir=dst_dir)
    ext = os.path.splitext(dst)[1] or ".mp4"
    part_names: List[str] = []
    try:
        for i, (start, end) in enumerate(segments):
            name = f"part_{i:05d}{ext}"
            part = os.path.join(tmpdir, name)
            base = [ffmpeg, "-v", "error", "-y", "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}", "-i", rel(src)]
            if reencode:
                codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                         "-c:a", "aac", "-avoid_negative_ts", "make_zero"]
            else:
                codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
            subprocess.run(base + codec + [rel(part)], check=True)
            part_names.append(name)

        # concat demuxer resolves entries relative to the list file's own directory,
        # so reference parts by basename.
        listfile = os.path.join(tmpdir, "concat.txt")
        with open(listfile, "w") as fh:
            for name in part_names:
                fh.write(f"file '{name}'\n")
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", rel(listfile), "-c", "copy", rel(dst)],
            check=True,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
