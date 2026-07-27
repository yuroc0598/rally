"""Automatic court detection: recover the image->court homography with no manual clicking.

The camera is fixed, so we only need to find the court once. Two backends behind one
:func:`detect_court` entry point:

* **classical** (default, no weights) — court lines are the brightest near-white marks on a
  uniform surface. We threshold them, run a probabilistic Hough transform, keep the
  extreme horizontal lines (the two baselines) and extreme vertical lines (the two doubles
  sidelines), intersect them into the four outer court corners, and build the homography
  (:meth:`rally.signals.court.Court.from_image_corners`). Each candidate is scored by
  reprojecting the full court model and measuring how much of it lands on white pixels; we
  aggregate over several sampled frames and keep the best-scoring, self-consistent one.
* **keypoint model** (opt-in via ``court_weights``) — a trained court-keypoint network can
  be plugged in here for perspective-heavy or low-contrast footage; the classical path is
  the fallback.

The geometry helpers are pure and unit-testable; the frame-sampling orchestrator is
guarded and returns ``None`` on failure so the caller falls back to manual ``court_corners``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .court import Court, court_model_polylines

Line = Tuple[float, float, float, float]   # x1, y1, x2, y2


def _line_angle_deg(l: Line) -> float:
    x1, y1, x2, y2 = l
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def classify_lines(lines: List[Line], horiz_max_deg: float = 35.0,
                   vert_min_deg: float = 55.0) -> Tuple[List[Line], List[Line]]:
    """Split line segments into near-horizontal (baselines) and near-vertical (sidelines)."""
    horiz, vert = [], []
    for l in lines:
        a = _line_angle_deg(l)
        a = min(a, 180.0 - a)          # fold to [0, 90]
        if a <= horiz_max_deg:
            horiz.append(l)
        elif a >= vert_min_deg:
            vert.append(l)
    return horiz, vert


def line_intersection(a: Line, b: Line) -> Optional[Tuple[float, float]]:
    """Intersection of the infinite lines through segments ``a`` and ``b`` (None if parallel)."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


def _mid_y(l: Line) -> float:
    return 0.5 * (l[1] + l[3])


def _mid_x(l: Line) -> float:
    return 0.5 * (l[0] + l[2])


def corners_from_lines(horiz: List[Line], vert: List[Line]
                       ) -> Optional[Tuple[tuple, tuple, tuple, tuple]]:
    """Outer court corners (near-left, near-right, far-right, far-left) from extreme lines.

    The near baseline is the lowest horizontal line on screen (largest y), the far baseline
    the highest; the left/right doubles sidelines are the leftmost/rightmost verticals.
    Returns the four intersections, or ``None`` if lines are missing/degenerate.
    """
    if len(horiz) < 2 or len(vert) < 2:
        return None
    near = max(horiz, key=_mid_y)      # closest baseline (bottom of frame)
    far = min(horiz, key=_mid_y)       # far baseline (top of frame)
    left = min(vert, key=_mid_x)
    right = max(vert, key=_mid_x)
    nl = line_intersection(near, left)
    nr = line_intersection(near, right)
    fr = line_intersection(far, right)
    fl = line_intersection(far, left)
    if None in (nl, nr, fr, fl):
        return None
    return nl, nr, fr, fl


def score_court(white_mask: np.ndarray, court: Court, samples_per_m: float = 3.0,
                band_px: int = 3) -> float:
    """Fraction of reprojected court-line points that land on white pixels in ``white_mask``.

    A correct homography places the model's lines on the real painted lines, so a high
    overlap means a good fit. ``band_px`` tolerates line thickness / small error.
    """
    h, w = white_mask.shape[:2]
    hits = total = 0
    for seg in court_model_polylines():
        (x0, y0), (x1, y1) = seg[0], seg[1]
        length_m = float(np.hypot(x1 - x0, y1 - y0))
        n = max(2, int(length_m * samples_per_m))
        ts = np.linspace(0.0, 1.0, n)
        pts_court = np.stack([x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)], axis=1)
        pts_img = court.to_image(pts_court)
        for px, py in pts_img:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h:
                total += 1
                lo_y, hi_y = max(0, iy - band_px), min(h, iy + band_px + 1)
                lo_x, hi_x = max(0, ix - band_px), min(w, ix + band_px + 1)
                if white_mask[lo_y:hi_y, lo_x:hi_x].any():
                    hits += 1
    return hits / total if total else 0.0


def _white_mask(frame_bgr: np.ndarray):
    """Bright, low-saturation (white paint) pixel mask."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # high value, low saturation = white lines regardless of court colour
    return cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))


def _detect_in_frame(frame_bgr: np.ndarray, min_score: float
                     ) -> Optional[Tuple[Court, float]]:
    import cv2
    mask = _white_mask(frame_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = mask.shape[:2]
    min_len = int(0.15 * w)
    raw = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=80,
                          minLineLength=min_len, maxLineGap=20)
    if raw is None:
        return None
    lines = [tuple(map(float, l[0])) for l in raw]
    horiz, vert = classify_lines(lines)
    corners = corners_from_lines(horiz, vert)
    if corners is None:
        return None
    nl, nr, fr, fl = corners
    # sanity: corners must be ordered/convex enough to form a court quad
    if not (nl[1] > fl[1] and nr[1] > fr[1] and nr[0] > nl[0] and fr[0] > fl[0]):
        return None
    try:
        court = Court.from_image_corners(nl, nr, fr, fl)
    except Exception:
        return None
    score = score_court(mask, court)
    if score < min_score:
        return None
    return court, score


def detect_court(video: str, cfg=None, *, n_frames: int = 12, min_score: float = 0.55,
                 progress: Callable[[str], None] = lambda _m: None) -> Optional[Court]:
    """Sample frames across ``video`` and return the best-scoring court homography, or None.

    Uses the classical Hough detector. ``cfg.court_weights`` is a reserved placeholder for a
    future trained keypoint-model backend (not yet implemented); if set, we note it and use
    the classical path anyway. Returns ``None`` when no frame yields a confident court, so
    the caller falls back to manual calibration.
    """
    import cv2

    if cfg is not None and getattr(cfg, "court_weights", None):
        progress("  court_weights set but the keypoint backend isn't implemented yet "
                 "-> using classical court detection")

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(total * 0.05, total * 0.95, n_frames).astype(int)
    best: Optional[Tuple[Court, float]] = None
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        found = _detect_in_frame(frame, min_score)
        if found and (best is None or found[1] > best[1]):
            best = found
    cap.release()
    if best is None:
        return None
    progress(f"  court detected (line-overlap score {best[1]:.2f})")
    return best[0]
