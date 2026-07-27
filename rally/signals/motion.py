"""Motion channel primitives (pure functions over grayscale frames).

* ``frame_diff_energy`` — normalized mean absolute inter-frame difference: high when
  players move dynamically, ~0 for a static scene.
* ``camera_shift_px`` — global translation magnitude between frames (phase
  correlation), used to flag camera pans / handheld motion so between-point sweeps
  are not mistaken for dynamic play.
"""

from __future__ import annotations

import numpy as np


def frame_diff_energy(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    prev_gray = np.asarray(prev_gray, dtype=np.float32)
    gray = np.asarray(gray, dtype=np.float32)
    return float(np.mean(np.abs(gray - prev_gray)) / 255.0)


def camera_shift_px(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Magnitude (pixels) of global translation. Requires OpenCV; returns 0.0 if
    unavailable so the pipeline still runs without the camera-motion channel."""
    try:
        import cv2
    except Exception:  # pragma: no cover - environment without cv2
        return 0.0
    a = np.asarray(prev_gray, dtype=np.float32)
    b = np.asarray(gray, dtype=np.float32)
    (dx, dy), _response = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy))
