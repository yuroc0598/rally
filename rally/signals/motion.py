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


def camera_shift_px(prev_gray: np.ndarray, gray: np.ndarray, *,
                    min_texture_std: float = 5.0,
                    min_response: float = 0.10) -> float:
    """Magnitude (pixels) of a *reliable* global translation estimate.

    Phase correlation is undefined on flat/near-flat images and can return a large,
    arbitrary shift with a weak correlation peak.  Such estimates are rejected as
    unavailable (``0.0``), as is the historical no-OpenCV fallback.  This keeps the
    public scalar API while preventing low-texture frames from looking like camera pans.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover - environment without cv2
        return 0.0
    a = np.asarray(prev_gray, dtype=np.float32)
    b = np.asarray(gray, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or a.size == 0:
        return 0.0
    if (not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.std(a) < min_texture_std or np.std(b) < min_texture_std):
        return 0.0
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    if (not np.isfinite(dx) or not np.isfinite(dy)
            or not np.isfinite(response) or response < min_response):
        return 0.0
    return float(np.hypot(dx, dy))
