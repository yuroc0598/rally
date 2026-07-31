#!/usr/bin/env bash
#
# One-shot setup for rally-web.
#
# Installs all Python dependencies and prepares the three inference checkpoints used by
# the accuracy-first pipeline: TrackNet, YOLO12, and RTMPose. Idempotent — safe to re-run;
# existing checkpoints are checksum-verified and reused.
#
#   ./setup.sh
#
# Overrides (env vars):
#   PYTHON=python3.12        pick a specific interpreter
#   WEIGHTS_DRIVE_ID=<id>    use a different Google Drive file id for the weights
#   WEIGHTS_SHA256=<digest>   expected identity (set empty only for an intentionally unpinned custom model)
#   YOLO_MODEL_NAME=<name>    Ultralytics detection checkpoint (default: yolo12n.pt)
#   YOLO_SHA256=<digest>      expected YOLO identity (empty permits a custom checkpoint)
#   RTMPOSE_URL=<url>         RTMPose ONNX SDK zip download
#   RTMPOSE_ZIP_SHA256=<hex>  expected archive identity (empty permits a custom archive)
#   RTMPOSE_SHA256=<hex>      expected extracted ONNX identity
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
YOLO_MODEL_NAME="${YOLO_MODEL_NAME:-yolo12n.pt}"
YOLO_PATH="${YOLO_PATH:-models/$YOLO_MODEL_NAME}"
if [ "${YOLO_SHA256+x}" != x ]; then
  if [ "$YOLO_MODEL_NAME" = "yolo12n.pt" ]; then
    YOLO_SHA256="419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6"
  else
    YOLO_SHA256=""
  fi
fi
RTMPOSE_BASENAME="rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504"
RTMPOSE_URL="${RTMPOSE_URL:-https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/${RTMPOSE_BASENAME}.zip}"
RTMPOSE_ARCHIVE="${RTMPOSE_ARCHIVE:-models/${RTMPOSE_BASENAME}.zip}"
RTMPOSE_PATH="${RTMPOSE_PATH:-models/${RTMPOSE_BASENAME}.onnx}"
RTMPOSE_ZIP_SHA256="${RTMPOSE_ZIP_SHA256-f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741}"
RTMPOSE_SHA256="${RTMPOSE_SHA256-5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c}"

mkdir -p models

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
"${PIP[@]}" install --no-build-isolation -e ".[server]"

# The GUI and headless OpenCV wheels install into the same cv2/ directory and cannot
# coexist. Reuse a working, venv-owned headless-only install on repeated/offline runs.
# Only repair when dependency resolution actually installed a conflicting GUI wheel.
opencv_headless_ok() {
  "$PY" - <<'PYEOF' >/dev/null 2>&1
from importlib import metadata
from pathlib import Path
import sys

prefix = Path(sys.prefix).resolve()
owned = set()
for dist in metadata.distributions():
    location = Path(dist.locate_file("")).resolve()
    if location == prefix or prefix in location.parents:
        owned.add((dist.metadata.get("Name") or "").lower())
assert "opencv-python-headless" in owned
assert not ({"opencv-python", "opencv-contrib-python"} & owned)
import cv2
assert prefix in Path(cv2.__file__).resolve().parents
PYEOF
}
if opencv_headless_ok; then
  echo "[setup] headless OpenCV already healthy — reusing it."
else
  echo "[setup] pinning opencv to the headless build (avoids libxcb/libGL on servers) ..."
  "${PIP[@]}" uninstall -y opencv-python opencv-python-headless opencv-contrib-python >/dev/null 2>&1 || true
  "${PIP[@]}" install --force-reinstall "opencv-python-headless>=4.8"
fi

# Ultralytics declares the GUI OpenCV distribution, but uses the same cv2 API and works with
# the headless wheel. It was installed above with the server extra; removing only the GUI
# distribution and force-reinstalling headless keeps one deterministic cv2 implementation
# while retaining player/pose inference for match-state validation.
echo "[setup] player geometry + serve-pose validation enabled with headless OpenCV."

# RTMLib uses ONNX Runtime for RTMPose. The server extra deliberately installs the
# universally compatible CPU wheel. On an NVIDIA host, replace it with the GPU wheel so
# resolve_rtmpose_device("auto") can select CUDA. The GPU wheel retains a CPU provider,
# and a failed installation is repaired by restoring the CPU package.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  if "$PY" - <<'PYEOF' >/dev/null 2>&1
import onnxruntime as ort
assert "CUDAExecutionProvider" in ort.get_available_providers()
PYEOF
  then
    echo "[setup] ONNX Runtime CUDA provider already available."
  else
    echo "[setup] NVIDIA GPU detected — installing ONNX Runtime GPU provider ..."
    if ! "${PIP[@]}" install "onnxruntime-gpu>=1.17"; then
      # Do not uninstall the working CPU wheel before this attempt: setup must remain
      # recoverable when PyPI/DNS is temporarily unavailable.
      echo "[setup] ⚠️  onnxruntime-gpu install failed; keeping the CPU runtime." >&2
    fi
  fi
else
  echo "[setup] no NVIDIA runtime detected — RTMPose will use ONNX Runtime CPU."
fi

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

# 3) YOLO12 person-detection weights. Ultralytics owns the release download logic; setup
#    forces the download now instead of making the first video-processing job wait for it.
#    A pinned default digest prevents a changed/corrupt asset from silently affecting the
#    target-court and serve-setup signals. Custom names may set YOLO_SHA256="".
echo "[setup] preparing YOLO12 player detector at $YOLO_PATH ..."
YOLO_OK=1
if ! YOLO_MODEL_NAME="$YOLO_MODEL_NAME" YOLO_PATH="$YOLO_PATH" YOLO_SHA256="$YOLO_SHA256" \
  "$PY" - <<'PYEOF'
import hashlib
import os
import time
from pathlib import Path

from ultralytics import YOLO

name = os.environ["YOLO_MODEL_NAME"]
target = Path(os.environ["YOLO_PATH"]).expanduser().resolve()
expected = os.environ.get("YOLO_SHA256", "").strip().lower()

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

target.parent.mkdir(parents=True, exist_ok=True)
if target.is_file() and expected and digest(target) != expected:
    quarantine = target.with_name(f"{target.name}.invalid.{time.strftime('%Y%m%d%H%M%S')}")
    target.replace(quarantine)
    print(f"[setup] quarantined checksum-mismatched YOLO model at {quarantine}")

if not target.is_file():
    previous = Path.cwd()
    try:
        os.chdir(target.parent)
        YOLO(name)  # downloads a known Ultralytics asset when `name` is not local
        downloaded = target.parent / Path(name).name
        if downloaded.resolve() != target and downloaded.is_file():
            downloaded.replace(target)
    finally:
        os.chdir(previous)

if not target.is_file():
    raise RuntimeError(f"Ultralytics did not produce {target}")
actual = digest(target)
if expected and actual != expected:
    raise RuntimeError(f"YOLO SHA-256 mismatch: expected {expected}, got {actual}")
YOLO(str(target))  # parse the checkpoint now so an incompatible file fails setup
print(f"[setup] YOLO ready: {target} (sha256={actual})")
PYEOF
then
  YOLO_OK=0
  echo "[setup] ⚠️  YOLO12 download/verification failed; Ultralytics will retry on first use." >&2
fi

# 4) RTMPose-M Body7. The official SDK archive calls the graph `end2end.onnx`; the
#    application discovers it by the release basename, so extract it under that stable
#    name. Python's zipfile keeps this independent of a system `unzip` command. Downloads
#    and extraction publish atomically, and both the archive and ONNX are checksum-pinned.
echo "[setup] preparing RTMPose at $RTMPOSE_PATH ..."
RTMPOSE_OK=1
if ! RTMPOSE_URL="$RTMPOSE_URL" RTMPOSE_ARCHIVE="$RTMPOSE_ARCHIVE" \
  RTMPOSE_PATH="$RTMPOSE_PATH" RTMPOSE_ZIP_SHA256="$RTMPOSE_ZIP_SHA256" \
  RTMPOSE_SHA256="$RTMPOSE_SHA256" "$PY" - <<'PYEOF'
import hashlib
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

url = os.environ["RTMPOSE_URL"]
archive = Path(os.environ["RTMPOSE_ARCHIVE"]).expanduser().resolve()
target = Path(os.environ["RTMPOSE_PATH"]).expanduser().resolve()
archive_expected = os.environ.get("RTMPOSE_ZIP_SHA256", "").strip().lower()
target_expected = os.environ.get("RTMPOSE_SHA256", "").strip().lower()
max_bytes = 1024 * 1024 * 1024

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def quarantine(path: Path) -> None:
    moved = path.with_name(f"{path.name}.invalid.{time.strftime('%Y%m%d%H%M%S')}")
    path.replace(moved)
    print(f"[setup] quarantined checksum-mismatched model asset at {moved}")

archive.parent.mkdir(parents=True, exist_ok=True)
target.parent.mkdir(parents=True, exist_ok=True)

if target.is_file() and target_expected and digest(target) != target_expected:
    quarantine(target)

if not target.is_file():
    if archive.is_file() and archive_expected and digest(archive) != archive_expected:
        quarantine(archive)
    if not archive.is_file():
        print(f"[setup] downloading {url} -> {archive}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=".rtmpose-download.", suffix=".zip", dir=archive.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    raise RuntimeError("RTMPose archive exceeds the 1 GiB safety limit")
                copied = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    copied += len(block)
                    if copied > max_bytes:
                        raise RuntimeError("RTMPose archive exceeds the 1 GiB safety limit")
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if temporary.stat().st_size == 0:
                raise RuntimeError("RTMPose download was empty")
            actual = digest(temporary)
            if archive_expected and actual != archive_expected:
                raise RuntimeError(
                    f"RTMPose archive SHA-256 mismatch: expected {archive_expected}, got {actual}")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)

    actual_archive = digest(archive)
    if archive_expected and actual_archive != archive_expected:
        raise RuntimeError(
            f"RTMPose archive SHA-256 mismatch: expected {archive_expected}, got {actual_archive}")
    with zipfile.ZipFile(archive) as bundle:
        members = [item for item in bundle.infolist()
                   if not item.is_dir() and Path(item.filename).name == "end2end.onnx"]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one end2end.onnx in RTMPose archive, found {len(members)}")
        member = members[0]
        if member.file_size <= 0 or member.file_size > max_bytes:
            raise RuntimeError("RTMPose ONNX size is outside the allowed range")
        fd, temporary_name = tempfile.mkstemp(
            prefix=".rtmpose-extract.", suffix=".onnx", dir=target.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with bundle.open(member) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            actual = digest(temporary)
            if target_expected and actual != target_expected:
                raise RuntimeError(
                    f"RTMPose ONNX SHA-256 mismatch: expected {target_expected}, got {actual}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

actual_target = digest(target)
if target_expected and actual_target != target_expected:
    raise RuntimeError(
        f"RTMPose ONNX SHA-256 mismatch: expected {target_expected}, got {actual_target}")

# Exercise the same RTMLib constructor and COCO-17 output contract used by the pipeline.
from rally.signals.pose import resolve_rtmpose_device
from rtmlib import RTMPose

device = resolve_rtmpose_device("auto", "onnxruntime")
estimator = RTMPose(str(target), model_input_size=(192, 256), to_openpose=False,
                    backend="onnxruntime", device=device)
frame = np.zeros((360, 640, 3), dtype=np.uint8)
keypoints, scores = estimator(frame, bboxes=[[200, 40, 440, 350]])
if np.asarray(keypoints).shape != (1, 17, 2) or np.asarray(scores).shape != (1, 17):
    raise RuntimeError(
        f"unexpected RTMPose output shapes: {np.asarray(keypoints).shape}, "
        f"{np.asarray(scores).shape}")
print(f"[setup] RTMPose ready: {target} (sha256={actual_target}, device={device})")
PYEOF
then
  RTMPOSE_OK=0
  echo "[setup] ⚠️  RTMPose download/verification failed; pose evidence will abstain." >&2
  echo "[setup]      Re-run ./setup.sh when the OpenMMLab host is reachable, or place" >&2
  echo "[setup]      the official zip at $RTMPOSE_ARCHIVE and re-run setup." >&2
fi

# 5) ball-tracking weights (auto-discovered from models/). fetch_models verifies the
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
  echo "[setup] ✅ TrackNet ready — ball-arbiter tracking is enabled by default."
else
  echo "[setup] ⚠️  TrackNet weights are missing (fetch failed above)."
  echo "  The pipeline still works — ball-arbiter auto-falls back to the audio-primary detector."
  echo "  To enable full ball-arbiter mode later, once the host is reachable:"
  echo "    $PY -m rally.tools.fetch_models --drive-id $WEIGHTS_DRIVE_ID"
  echo "  or download the .pt in a browser and install it with:"
  echo "    $PY -m rally.tools.fetch_models --verify /path/to/tracknet.pt"
fi
if [ "$YOLO_OK" = 1 ]; then
  echo "[setup] ✅ YOLO player detection ready: $YOLO_PATH"
else
  echo "[setup] ⚠️  YOLO player detection weights are not ready. Re-run ./setup.sh online."
fi
if [ "$RTMPOSE_OK" = 1 ]; then
  echo "[setup] ✅ RTMPose serve-pose validation ready: $RTMPOSE_PATH"
else
  echo "[setup] ⚠️  RTMPose is not ready. Put the official archive at $RTMPOSE_ARCHIVE and re-run ./setup.sh."
fi
echo "  Web UI:  $PY -m rally.web.app        # then open http://127.0.0.1:8000"
echo "  CLI:     $PY -m rally.cli match.mp4 -o rallies.mp4 --json rallies.json"
