"""End-to-end integration test on a synthetic video.

Builds a black video whose audio track has ball-strike-like bursts only during two
"rally" windows, then runs the full pipeline (audio + motion channels; YOLO disabled)
and checks that it recovers those windows and writes a shorter output file.

Skipped automatically if ffmpeg or the scientific stack is unavailable.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from rally.io.ffmpeg import _require, _video_encoder  # noqa: E402

try:  # resolve a *working* ffmpeg (skips broken PATH shims); skip cleanly if none
    FFMPEG = _require("ffmpeg")
except Exception:
    pytest.skip("ffmpeg not available", allow_module_level=True)

from scipy.io import wavfile  # noqa: E402

from rally.config import RallyConfig  # noqa: E402
from rally.pipeline import trim  # noqa: E402


@pytest.fixture
def scratch_dir():
    """A working directory that a child ffmpeg process can also see.

    Some sandboxes remap absolute paths (and isolate /tmp) for subprocesses. If a
    child ffmpeg cannot read a file we just wrote by its absolute path, the whole
    cut/mux path is untestable here, so we skip rather than report a false failure.
    """
    import shutil as _sh
    import subprocess as _sp

    local = os.path.join(os.path.dirname(__file__), "_scratch")
    os.makedirs(local, exist_ok=True)
    probe = os.path.join(local, "_probe.wav")
    wavfile.write(probe, SR, np.zeros(SR // 10, dtype=np.int16))
    rc = _sp.run([FFMPEG, "-v", "error", "-i", os.path.abspath(probe),
                  "-f", "null", "-"], capture_output=True).returncode
    if rc != 0:
        _sh.rmtree(local, ignore_errors=True)
        pytest.skip("sandbox remaps absolute paths for subprocesses; run in a normal env")
    try:
        yield local
    finally:
        _sh.rmtree(local, ignore_errors=True)


SR = 22050
DURATION = 40.0
RALLIES = [(12.0, 20.0), (32.0, 38.0)]  # ground-truth rally windows


def _make_audio(path):
    n = int(DURATION * SR)
    x = 0.0005 * np.random.default_rng(0).standard_normal(n)  # quiet ambient floor
    for (start, end) in RALLIES:
        t = start
        while t < end:
            i0 = int(t * SR)
            burst = int(0.02 * SR)
            idx = np.arange(burst)
            tone = np.sin(2 * np.pi * 3000 * idx / SR) * np.exp(-idx / (0.004 * SR))
            if i0 + burst <= n:
                x[i0:i0 + burst] += tone
            t += 0.8  # ~one strike every 0.8 s -> regular rally rhythm
    pcm16 = np.clip(x, -1, 1)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    wavfile.write(path, SR, pcm16)


def _make_video(audio_path, out_path):
    vcodec, vargs = _video_encoder()
    subprocess.run(
        [FFMPEG, "-v", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=320x240:r=25:d={DURATION}",
         "-i", audio_path,
         "-c:v", vcodec, *vargs, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         out_path],
        check=True,
    )


def test_end_to_end(scratch_dir):
    audio = os.path.join(scratch_dir, "a.wav")
    src = os.path.join(scratch_dir, "match.mp4")
    out = os.path.join(scratch_dir, "rallies.mp4")
    js = os.path.join(scratch_dir, "rallies.json")
    _make_audio(audio)
    _make_video(audio, src)

    # ball_arbiter is on by default but needs TrackNet weights and is CPU-slow; this test
    # exercises the audio/motion segmentation path, so opt out for speed/determinism.
    cfg = RallyConfig(analysis_fps=5.0, pad_pre_s=0.5, pad_post_s=0.5, ball_arbiter=False)
    result = trim(src, output_path=out, cfg=cfg, json_path=js, detect_players=False)

    # recovered roughly the right amount of play and compressed the video
    assert "audio" in result.channels_used
    assert result.n_strikes >= 10
    assert 0.2 < result.compression_ratio < 0.75
    assert 1 <= len(result.segments) <= 3

    # every ground-truth rally is covered by some detected segment
    for (gs, ge) in RALLIES:
        mid = (gs + ge) / 2
        assert any(s <= mid <= e for (s, e) in result.segments), f"missed rally at {mid}s"

    # artefacts written and output is a real, shorter file
    assert os.path.isfile(out) and os.path.getsize(out) > 0
    assert os.path.isfile(js)
    assert os.path.getsize(out) < os.path.getsize(src)
