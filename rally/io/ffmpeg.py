"""Thin wrappers over the ffmpeg / ffprobe binaries.

Binary resolution (see :func:`_require`), in order:
  1. an explicit override — ``$RALLY_FFMPEG`` / ``$RALLY_FFPROBE``;
  2. the first *working* ``ffmpeg``/``ffprobe`` found across ``$PATH`` and the usual
     install locations (``/usr/bin`` etc.).

Each candidate is probed with ``-version`` before use, so a broken wrapper earlier on
PATH (e.g. a shim that shells out to a container that isn't running) is skipped in favour
of a real binary further down. Set ``$RALLY_FFMPEG``/``$RALLY_FFPROBE`` to force your own
build. Everything shells out, so there is no hard dependency on pyav/moviepy.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np


def _run_cancellable(cmd: list[str], cancel_check: Optional[Callable[[], None]] = None) -> None:
    """Run a media subprocess and terminate it promptly when cancellation is requested."""
    if cancel_check is None:
        subprocess.run(cmd, check=True)
        return
    proc = subprocess.Popen(cmd)
    try:
        while proc.poll() is None:
            cancel_check()
            time.sleep(0.1)
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def add_real_context(
    segments: List[Tuple[float, float]], total_s: float,
    pre_buffer_s: float, post_buffer_s: float,
) -> List[Tuple[float, float]]:
    """Extend cuts with real source footage around each detected point.

    When adjacent buffers would overlap, the available between-point footage is split at
    its midpoint. This prevents duplicate footage while preserving as much setup and
    follow-through as the source contains. Detected point bounds remain unchanged in
    metadata; these extensions affect presentation only.
    """
    pre_buffer_s = max(0.0, min(1.0, float(pre_buffer_s)))
    post_buffer_s = max(0.0, min(1.0, float(post_buffer_s)))
    total_s = max(0.0, float(total_s))
    points = [(float(start), float(end)) for start, end in segments]
    starts = [max(0.0, start - pre_buffer_s) for start, _end in points]
    ends = [min(total_s, end + post_buffer_s) for _start, end in points]
    for i in range(len(points) - 1):
        point_end = points[i][1]
        next_start = points[i + 1][0]
        if next_start < point_end:
            continue
        if ends[i] > starts[i + 1]:
            requested = pre_buffer_s + post_buffer_s
            post_share = post_buffer_s / requested if requested else 0.5
            boundary = point_end + (next_start - point_end) * post_share
            ends[i] = min(ends[i], boundary)
            starts[i + 1] = max(starts[i + 1], boundary)
    return list(zip(starts, ends))


def add_real_postroll(
    segments: List[Tuple[float, float]], total_s: float, buffer_s: float
) -> List[Tuple[float, float]]:
    """Backward-compatible post-roll-only wrapper."""
    return add_real_context(segments, total_s, 0.0, buffer_s)


def _runs(path: Optional[str]) -> bool:
    """True if ``path -version`` actually executes (guards against broken PATH shims)."""
    if not path:
        return False
    try:
        subprocess.run([path, "-version"], capture_output=True, timeout=30, check=True)
        return True
    except Exception:
        return False


def _candidates(binary: str):
    """Yield plausible paths for ``binary``, best first.

    For ``ffmpeg`` we try imageio-ffmpeg's bundled binary first: it ships a static build
    with libx264 (browser-playable H.264), which many distro packages (e.g. ``ffmpeg-free``)
    lack. Then every ``$PATH`` entry and the usual install dirs — scanning past the first
    hit lets us skip a broken shim and find a real binary (and supplies ``ffprobe``, which
    imageio-ffmpeg does not bundle)."""
    seen = set()
    if binary == "ffmpeg":
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.isfile(exe):
                seen.add(exe)
                yield exe
        except Exception:
            pass
    dirs = os.environ.get("PATH", "").split(os.pathsep)
    dirs += ["/usr/bin", "/usr/local/bin", "/bin", "/opt/homebrew/bin"]
    for d in dirs:
        if not d:
            continue
        cand = os.path.join(d, binary)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            yield cand


def _ffmpeg_score(path: str) -> int:
    """Rank an ffmpeg build by capability: +2 for libx264 (browser H.264), +1 for drawtext
    (burned-in labels). Lets us pick a full build over a minimal one when both are present."""
    try:
        enc = subprocess.run([path, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
        flt = subprocess.run([path, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 0
    score = 2 if "libx264" in enc else 0
    if any(l.split()[1:2] == ["drawtext"] for l in flt.splitlines() if l.split()):
        score += 1
    return score


@functools.lru_cache(maxsize=None)
def _require(binary: str) -> str:
    """Resolve 'ffmpeg'/'ffprobe' to a working executable, cached for the process."""
    override = os.environ.get(f"RALLY_{binary.upper()}")
    if _runs(override):
        return override  # type: ignore[return-value]

    if binary == "ffmpeg":
        # Prefer the most capable working ffmpeg (libx264 + drawtext) over a minimal one.
        best, best_score = None, -1
        for cand in _candidates(binary):
            if not _runs(cand):
                continue
            score = _ffmpeg_score(cand)
            if score > best_score:
                best, best_score = cand, score
        if best is not None:
            return best
    else:
        for cand in _candidates(binary):
            if _runs(cand):
                return cand

    raise RuntimeError(
        f"'{binary}' is unavailable: no working binary from $RALLY_{binary.upper()} or "
        f"PATH. Install ffmpeg (e.g. `sudo dnf install ffmpeg-free`, "
        f"`sudo apt install ffmpeg`, or `brew install ffmpeg`), or set "
        f"$RALLY_{binary.upper()} to a working binary."
    )


@functools.lru_cache(maxsize=None)
def _has_filter(name: str) -> bool:
    """True if the resolved ffmpeg build provides filter ``name`` (e.g. 'drawtext')."""
    try:
        ff = _require("ffmpeg")
        out = subprocess.run([ff, "-hide_banner", "-filters"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    for line in out.splitlines():
        toks = line.split()
        # each line is: "<flags> <name> <io> <description>"; match the name column
        if len(toks) >= 2 and toks[1] == name:
            return True
    return False


@functools.lru_cache(maxsize=None)
def _video_encoder() -> Tuple[str, Tuple[str, ...]]:
    """Pick a *working* H.264 encoder + its rate-control args (probed once, cached).

    We don't trust ``-encoders`` listings: ``ffmpeg-free`` advertises ``libopenh264`` but
    its runtime lib is often absent, so it fails only at encode time. Instead we do a tiny
    test-encode of each candidate and take the first that actually runs. Order prefers
    browser-playable H.264 (``libx264`` → ``libopenh264``); ``mpeg4`` is a last resort so
    output is at least produced (note: MPEG-4 Part 2 does NOT play in an HTML5 ``<video>``).
    """
    ff = _require("ffmpeg")
    candidates = (
        ("libx264", ("-preset", "veryfast", "-crf", "20")),
        ("libopenh264", ("-b:v", "6M")),
        ("mpeg4", ("-q:v", "3")),
    )
    for codec, cargs in candidates:
        try:
            rc = subprocess.run(
                [ff, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=0.2",
                 "-c:v", codec, *cargs, "-f", "null", "-"],
                capture_output=True, timeout=30,
            ).returncode
            if rc == 0:
                return codec, cargs
        except Exception:
            continue
    return "libx264", ("-preset", "veryfast", "-crf", "20")  # nothing verified; try anyway


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
    """Read video metadata. Prefer ffprobe; fall back to OpenCV when ffprobe is missing or
    can't decode the stream (e.g. the LGPL 'ffmpeg-free' ffprobe, which ships no H.264
    decoder). OpenCV reads the same streams the pipeline already decodes, so the fallback
    covers exactly the inputs rally handles."""
    try:
        return _probe_ffprobe(path)
    except Exception:
        return _probe_opencv(path)


def _probe_ffprobe(path: str) -> VideoInfo:
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


def _probe_opencv(path: str) -> VideoInfo:
    """ffprobe-free metadata: OpenCV for the video stream, ffmpeg's own log for audio."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {path} (no ffprobe, and OpenCV failed too)")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return VideoInfo(duration_s=duration, fps=fps, width=width, height=height,
                     has_audio=_has_audio(path))


def _has_audio(path: str) -> bool:
    """Detect an audio stream via ffmpeg's own probe log (works with the imageio build)."""
    try:
        ff = _require("ffmpeg")
        log = subprocess.run([ff, "-hide_banner", "-i", path],
                             capture_output=True, text=True, timeout=120).stderr
        return "Audio:" in log
    except Exception:
        return False


def load_audio_mono(path: str, sr: int) -> np.ndarray:
    """Decode the audio track to a mono float32 numpy array via an ffmpeg pipe."""
    ffmpeg = _require("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def iter_audio_mono(path: str, sr: int, *, chunk_s: float = 60.0) -> Iterator[np.ndarray]:
    """Stream mono float32 PCM from ffmpeg in bounded chunks.

    Unlike :func:`load_audio_mono`, this never buffers the complete decoded track.  It is
    the production path for multi-hour recordings; the eager helper remains useful for
    small clips and callers that explicitly need an array.
    """
    if sr <= 0 or chunk_s <= 0:
        raise ValueError("sr and chunk_s must be positive")
    ffmpeg = _require("ffmpeg")
    samples_per_chunk = max(1, int(round(sr * chunk_s)))
    bytes_per_chunk = samples_per_chunk * np.dtype(np.float32).itemsize
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
         "-f", "f32le", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stderr_tail = bytearray()

    def drain_stderr() -> None:
        while True:
            block = proc.stderr.read(4096)
            if not block:
                return
            stderr_tail.extend(block)
            if len(stderr_tail) > 64 * 1024:
                del stderr_tail[:-64 * 1024]

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    try:
        try:
            while True:
                data = bytearray()
                while len(data) < bytes_per_chunk:
                    block = proc.stdout.read(bytes_per_chunk - len(data))
                    if not block:
                        break
                    data.extend(block)
                # ffmpeg emits complete float32 samples; ignore no bytes silently only at EOF.
                usable = len(data) - (len(data) % np.dtype(np.float32).itemsize)
                if usable:
                    yield np.frombuffer(memoryview(data)[:usable], dtype=np.float32).copy()
                if len(data) < bytes_per_chunk:
                    break
            rc = proc.wait()
            stderr_thread.join(timeout=5)
            if rc:
                detail = bytes(stderr_tail).decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg audio decode failed ({rc}): {detail[-1000:]}")
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover - wedged external process
                    proc.kill()
                    proc.wait()
    finally:
        proc.stderr.close()
        stderr_thread.join(timeout=5)


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
    cancel_check: Optional[Callable[[], None]] = None,
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

    # drawtext needs libfreetype, which some ffmpeg builds omit (e.g. the imageio-ffmpeg
    # static build we use for libx264). Rather than fail the whole render, drop the labels
    # and still produce the trimmed, browser-playable video.
    if draw_labels and not _has_filter("drawtext"):
        import warnings

        warnings.warn(
            "ffmpeg build has no 'drawtext' filter — rendering rallies without burned-in "
            "point labels. Use an ffmpeg built with libfreetype to get labels.",
            RuntimeWarning, stacklevel=2,
        )
        draw_labels = False

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
    vcodec, vargs = _video_encoder()
    cmd += ["-c:v", vcodec, *vargs, _rel(dst)]
    _run_cancellable(cmd, cancel_check)


def cut_segments(
    src: str, segments: List[Tuple[float, float]], dst: str, *, reencode: bool = True,
    cancel_check: Optional[Callable[[], None]] = None,
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
                vcodec, vargs = _video_encoder()
                codec = ["-c:v", vcodec, *vargs,
                         "-c:a", "aac", "-avoid_negative_ts", "make_zero"]
            else:
                codec = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
            _run_cancellable(base + codec + [rel(part)], cancel_check)
            part_names.append(name)

        # concat demuxer resolves entries relative to the list file's own directory,
        # so reference parts by basename.
        listfile = os.path.join(tmpdir, "concat.txt")
        with open(listfile, "w") as fh:
            for name in part_names:
                fh.write(f"file '{name}'\n")
        _run_cancellable(
            [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", rel(listfile), "-c", "copy", rel(dst)],
            cancel_check,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
