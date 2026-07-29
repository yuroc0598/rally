#!/usr/bin/env bash
#
# One-shot setup for rally-web.
#
# Installs all Python dependencies and fetches the TrackNet ball-tracking weights so
# ball-arbiter mode (the accurate, default detector) works out of the box. Idempotent —
# safe to re-run; it skips the download if the weights are already present.
#
#   ./setup.sh
#
# Overrides (env vars):
#   PYTHON=python3.12        pick a specific interpreter
#   WEIGHTS_DRIVE_ID=<id>    use a different Google Drive file id for the weights
#   WEIGHTS_SHA256=<digest>   expected identity (set empty only for an intentionally unpinned custom model)
#
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
WEIGHTS_DRIVE_ID="${WEIGHTS_DRIVE_ID:-1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl}"
if [ "${WEIGHTS_SHA256+x}" = x ]; then
  WEIGHTS_SHA256="$WEIGHTS_SHA256"
elif [ "$WEIGHTS_DRIVE_ID" = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl" ]; then
  WEIGHTS_SHA256="c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076"
else
  WEIGHTS_SHA256=""
fi
WEIGHTS_PATH="models/tracknet.pt"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[setup] error: '$PY' not found. Install Python 3 or set PYTHON=<interpreter>." >&2
  exit 1
fi
echo "[setup] interpreter: $("$PY" --version 2>&1) ($(command -v "$PY"))"

# Resolve a WORKING pip. We can't just run '$PY -m pip' blindly: some interpreters
# ship without the pip module, and some sandboxed 'pip'/'pip3' shims print an error
# but still exit 0 (so they'd silently install nothing). pip_works() runs a real
# probe and rejects both cases.
#
# Order: '$PY -m pip' (keeps packages in the selected interpreter), otherwise a
# project-local venv created by that same interpreter. The interpreter that owns those
# packages is what runs the app + fetch_models, so setup never guesses from a standalone
# pip executable that may belong to another Python.
pip_works() {
  local out
  out="$("$@" --version 2>&1)" || return 1
  case "$out" in
    pip\ *) return 0 ;;   # e.g. "pip 25.0.1 from ..."
    *) return 1 ;;        # "No module named pip", sandbox "not allowed" banner, etc.
  esac
}

PIP=()
if pip_works "$PY" -m pip; then
  PIP=("$PY" -m pip)
  echo "[setup] pip: $PY -m pip"
else
  VENV_DIR="${VENV_DIR:-.venv}"
  echo "[setup] no working pip module for '$PY' — creating venv at $VENV_DIR ..."
  "$PY" -m venv "$VENV_DIR" || {
    echo "[setup] error: could not create a venv with '$PY' (need the 'venv' module)." >&2
    echo "[setup]        install pip for '$PY' or set PYTHON=<interpreter with pip>." >&2
    exit 1
  }
  PY="$VENV_DIR/bin/python"
  "$PY" -m pip install --upgrade pip >/dev/null
  PIP=("$PY" -m pip)
  echo "[setup] pip: $PY -m pip (venv)"
fi

# 1) dependencies. Installed via the resolved pip; the app's interpreter ($PY) is
#    the same one that owns them.
echo "[setup] installing rally and all headless-compatible runtime features from pyproject.toml ..."
"${PIP[@]}" install -e ".[server]"

# The GUI and headless OpenCV wheels install into the same cv2/ directory and cannot
# coexist. Force the server-safe distribution to be the sole installed OpenCV build.
echo "[setup] pinning opencv to the headless build (avoids libxcb/libGL on servers) ..."
"${PIP[@]}" uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
"${PIP[@]}" install --force-reinstall "opencv-python-headless>=4.8"

# Ultralytics declares the GUI OpenCV distribution, but uses the same cv2 API and works with
# the headless wheel. It was installed above with the server extra; removing only the GUI
# distribution and force-reinstalling headless keeps one deterministic cv2 implementation
# while retaining player/pose inference for match-state validation.
echo "[setup] player geometry + serve-pose validation enabled with headless OpenCV."

# 2) ffmpeg (media I/O). rally shells out to ffmpeg (not a Python lib) and reads video
#    metadata with ffprobe when present, else OpenCV. The encoding ffmpeg comes from
#    imageio-ffmpeg (installed above — a static build WITH libx264 for browser-playable
#    H.264). ffmpeg_ok() proves the real capability end-to-end: encode a tiny H.264 clip
#    and read it back with rally's probe(). Only if that fails do we try to install a
#    system ffmpeg (HTTP(S)_PROXY is passed through to the root install via sudo -E).
ffmpeg_ok() {
  "$PY" - <<'PYEOF' >/dev/null 2>&1
import os, subprocess, tempfile
from rally.io.ffmpeg import _require, _video_encoder, probe
ff = _require("ffmpeg")
codec, cargs = _video_encoder()
assert codec in ("libx264", "libopenh264"), f"no browser-H.264 encoder (got {codec})"
d = tempfile.mkdtemp()
clip = os.path.join(d, "t.mp4")
subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi",
                "-i", "color=c=black:s=64x64:r=5:d=1",
                "-c:v", codec, *cargs, "-pix_fmt", "yuv420p", clip], check=True)
info = probe(clip)
assert info.width == 64 and info.fps > 0, info
PYEOF
}
_ffdesc() { "$PY" -c "from rally.io.ffmpeg import _require,_video_encoder; print(_require('ffmpeg'), '| codec', _video_encoder()[0])"; }
if ffmpeg_ok; then
  echo "[setup] ffmpeg OK (encode+probe verified): $(_ffdesc)"
else
  echo "[setup] ffmpeg not fully working — installing a system ffmpeg ..."
  if command -v feature >/dev/null 2>&1; then
    # Meta devservers: fbpkg ffmpeg is a FULL build (libx264 + drawtext + ffprobe), so it
    # also restores burned-in point labels. 'persist' keeps it on future servers.
    feature install ffmpeg && feature persist ffmpeg 2>/dev/null || true
  elif command -v dnf >/dev/null 2>&1; then
    sudo -E dnf install -y ffmpeg-free || sudo -E dnf install -y ffmpeg || true
  elif command -v apt-get >/dev/null 2>&1; then
    sudo -E apt-get update -y && sudo -E apt-get install -y ffmpeg || true
  elif command -v brew >/dev/null 2>&1; then
    brew install ffmpeg || true
  else
    echo "[setup] ⚠️  no known package manager (feature/dnf/apt/brew) found — install ffmpeg yourself." >&2
  fi
  hash -r 2>/dev/null || true
  if ffmpeg_ok; then
    echo "[setup] ffmpeg ready (encode+probe verified): $(_ffdesc)"
  else
    echo "[setup] ⚠️  ffmpeg still not fully working. Point RALLY_FFMPEG (and RALLY_FFPROBE" >&2
    echo "[setup]     if you have one) at working binaries, or install ffmpeg with libx264." >&2
  fi
fi

# 3) ball-tracking weights (auto-discovered from models/). fetch_models verifies the
#    checkpoint loads into BallTrackerNet before installing it.
#
#    These weights are OPTIONAL: they're externally hosted (Google Drive) and unlicensed,
#    and the pipeline auto-falls-back to the audio-primary detector without them. A network
#    that can't reach the host (proxy/allowlist, offline) must not fail the whole setup —
#    the deps above are the part that matters. So we warn-and-continue on fetch failure.
if [ -f "$WEIGHTS_PATH" ]; then
  echo "[setup] weights already present at $WEIGHTS_PATH — verifying ..."
  WEIGHTS_OK=1
  if [ -n "$WEIGHTS_SHA256" ]; then
    "$PY" -m rally.tools.fetch_models --verify "$WEIGHTS_PATH" --sha256 "$WEIGHTS_SHA256" || WEIGHTS_OK=0
  else
    "$PY" -m rally.tools.fetch_models --verify "$WEIGHTS_PATH" || WEIGHTS_OK=0
  fi
  if [ "$WEIGHTS_OK" = 0 ]; then
    INVALID_PATH="${WEIGHTS_PATH}.invalid.$(date +%Y%m%d%H%M%S)"
    mv "$WEIGHTS_PATH" "$INVALID_PATH"
    echo "[setup] invalid checkpoint quarantined at $INVALID_PATH" >&2
  fi
else
  echo "[setup] fetching TrackNet ball-tracking weights ..."
  WEIGHTS_OK=1
  if [ -n "$WEIGHTS_SHA256" ]; then
    "$PY" -m rally.tools.fetch_models --drive-id "$WEIGHTS_DRIVE_ID" --sha256 "$WEIGHTS_SHA256" || WEIGHTS_OK=0
  else
    "$PY" -m rally.tools.fetch_models --drive-id "$WEIGHTS_DRIVE_ID" || WEIGHTS_OK=0
  fi
fi

echo
if [ "$WEIGHTS_OK" = 1 ]; then
  echo "[setup] ✅ ready — ball-arbiter tracking is enabled by default."
else
  echo "[setup] ⚠️  ready, but TrackNet weights are missing (fetch failed above)."
  echo "  The pipeline still works — ball-arbiter auto-falls back to the audio-primary detector."
  echo "  To enable full ball-arbiter mode later, once the host is reachable:"
  echo "    $PY -m rally.tools.fetch_models --drive-id $WEIGHTS_DRIVE_ID"
  echo "  or download the .pt in a browser and install it with:"
  echo "    $PY -m rally.tools.fetch_models --verify /path/to/tracknet.pt"
fi
echo "  Web UI:  $PY -m rally.web.app        # then open http://127.0.0.1:8000"
echo "  CLI:     $PY -m rally.cli match.mp4 -o rallies.mp4 --json rallies.json"
