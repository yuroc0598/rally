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
#
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
WEIGHTS_DRIVE_ID="${WEIGHTS_DRIVE_ID:-1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl}"
WEIGHTS_PATH="models/tracknet.pt"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[setup] error: '$PY' not found. Install Python 3 or set PYTHON=<interpreter>." >&2
  exit 1
fi
echo "[setup] interpreter: $("$PY" --version 2>&1) ($(command -v "$PY"))"

# 1) dependencies. Always use 'python -m pip' so packages land in the SAME interpreter
#    that runs the app (a bare 'pip'/'pip3' can point at a different Python).
echo "[setup] installing core dependencies (numpy, scipy, opencv, ffmpeg, web) ..."
"$PY" -m pip install -r requirements.txt

echo "[setup] installing ball-arbiter deps (torch, gdown) + player detection (ultralytics) ..."
"$PY" -m pip install torch gdown ultralytics

# 2) ball-tracking weights (auto-discovered from models/). fetch_models verifies the
#    checkpoint loads into BallTrackerNet before installing it.
if [ -f "$WEIGHTS_PATH" ]; then
  echo "[setup] weights already present at $WEIGHTS_PATH — verifying ..."
  "$PY" -m rally.tools.fetch_models --verify "$WEIGHTS_PATH"
else
  echo "[setup] fetching TrackNet ball-tracking weights ..."
  "$PY" -m rally.tools.fetch_models --drive-id "$WEIGHTS_DRIVE_ID"
fi

echo
echo "[setup] ✅ ready — ball-arbiter tracking is enabled by default."
echo "  Web UI:  $PY -m rally.web.app        # then open http://127.0.0.1:8000"
echo "  CLI:     $PY -m rally.cli match.mp4 -o rallies.mp4 --json rallies.json"
