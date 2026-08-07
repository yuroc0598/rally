"""Automatic court detection: recover the image->court homography with no manual clicking.

The camera is fixed, so we only need to find the court once. Two backends behind one
:func:`detect_court` entry point:

* **classical fallback** — court lines are the brightest near-white marks on a
  uniform surface. We threshold them, run a probabilistic Hough transform, keep the
  extreme horizontal lines (the two baselines) and extreme vertical lines (the two doubles
  sidelines), intersect them into the four outer court corners, and build the homography
  (:meth:`rally.signals.court.Court.from_image_corners`). Each candidate is scored by
  reprojecting the full court model and measuring how much of it lands on white pixels; we
  aggregate over several sampled frames and keep the best-scoring, self-consistent one.
* **keypoint model** — a required ResNet50 court-landmark model handles perspective-heavy
  or low-contrast footage; the independent classical path remains a geometric fallback.

The geometry helpers are pure and unit-testable; the frame-sampling orchestrator is
guarded and returns ``None`` on failure so the caller falls back to manual ``court_corners``.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .court import (
    COURT_L,
    DOUBLES_W,
    SERVICE_Y,
    SINGLES_IN,
    Court,
    court_model_polylines,
)
from ..config import DEFAULT_COURT_MODEL

Line = Tuple[float, float, float, float]   # x1, y1, x2, y2

COURT_MODEL_SOURCE_SIZE = 94_582_426
COURT_MODEL_SOURCE_SHA256 = (
    "16ebb7e46dc88247440c86b388e4f07f0d4abb76ce0a01a22925d3163f7fb7f3"
)
_COURT_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_COURT_MODEL_LOCK = threading.Lock()


def discover_court_weights(
    name: str = DEFAULT_COURT_MODEL, models_dir: Optional[str] = None,
) -> str:
    """Resolve the required court model from the shared model directory."""
    import os

    if models_dir is None:
        models_dir = os.environ.get("RALLY_MODELS_DIR")
        if not models_dir:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")
    direct = Path(name).expanduser()
    if direct.is_file():
        return str(direct.resolve())
    local = Path(models_dir).expanduser() / direct.name
    return str(local) if local.is_file() else str(direct)


def load_court_keypoint_model(weights: str, device: str = "cpu"):
    """Strictly load the pinned 14-landmark ResNet50 checkpoint."""
    import torch
    from torchvision.models import resnet50

    path = str(Path(weights).expanduser().resolve())
    key = (path, str(device))
    with _COURT_MODEL_LOCK:
        cached = _COURT_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("court checkpoint must be a plain state dictionary")
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 28)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _COURT_MODEL_CACHE[key] = model
        return model


def _resnet_keypoint_quads(
    frames: Sequence[np.ndarray], weights: str,
) -> List[Tuple[np.ndarray, float]]:
    """Run the pinned RGB/ImageNet-normalized 14-landmark model as one batch."""
    import torch
    import torch.nn.functional as functional
    from .player import resolve_yolo_device

    device = resolve_yolo_device()
    model = load_court_keypoint_model(weights, device=device)
    rgb = np.stack([frame[:, :, ::-1].copy() for frame in frames])
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = functional.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).to(device)
    with torch.inference_mode():
        predicted = model(tensor).detach().float().cpu().numpy().reshape(-1, 14, 2)

    output: List[Tuple[np.ndarray, float]] = []
    for frame, points in zip(frames, predicted):
        scaled = points.copy()
        scaled[:, 0] *= frame.shape[1] / 224.0
        scaled[:, 1] *= frame.shape[0] / 224.0
        # This checkpoint's first four landmarks are the outer doubles corners. Reducing
        # all 14 via a convex hull lets noisy internal service-line points distort a corner.
        corners = _canonical_outer_corners(scaled[:4])
        if corners is not None and valid_court_quad(corners, frame.shape):
            output.append((corners, 1.0))
        else:
            output.append((np.empty((0, 2), dtype=float), 0.0))
    return output


def _canonical_outer_corners(points: np.ndarray) -> Optional[np.ndarray]:
    """Reduce court keypoints to ``NL, NR, FR, FL`` image corners.

    Court models in the wild expose either four outer corners or all line intersections.
    Taking the convex hull supports both without baking one vendor's keypoint indices into
    the pipeline.  The result is still passed through :func:`valid_court_quad`, so an
    incomplete/occluded prediction is an abstention rather than a bad calibration.
    """
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if hull.shape[0] > 4:
        perimeter = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
        for fraction in np.linspace(0.005, 0.08, 32):
            reduced = cv2.approxPolyDP(
                hull.reshape(-1, 1, 2), fraction * perimeter, True,
            ).reshape(-1, 2)
            if reduced.shape[0] == 4:
                hull = reduced
                break
    if hull.shape[0] != 4:
        return None

    # Baseline-view footage places the near baseline below the far baseline.  Sorting the
    # two upper/lower hull vertices also makes the adapter independent of model ordering.
    by_y = hull[np.argsort(hull[:, 1])]
    far = by_y[:2][np.argsort(by_y[:2, 0])]
    near = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.asarray([near[0], near[1], far[1], far[0]], dtype=float)


def _keypoint_quads(result: Any, frame_shape, *, min_conf: float = 0.25
                    ) -> List[Tuple[np.ndarray, float]]:
    """Adapt one Ultralytics pose result into validated court quadrilaterals."""
    keypoints = getattr(result, "keypoints", None)
    xy = getattr(keypoints, "xy", None)
    if xy is None:
        return []
    xy = xy.detach().cpu().numpy() if hasattr(xy, "detach") else np.asarray(xy)
    if xy.ndim == 2:
        xy = xy[None, ...]
    conf = getattr(keypoints, "conf", None)
    if conf is not None:
        conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        if conf.ndim == 1:
            conf = conf[None, ...]
    box_conf = getattr(getattr(result, "boxes", None), "conf", None)
    if box_conf is not None:
        box_conf = (box_conf.detach().cpu().numpy()
                    if hasattr(box_conf, "detach") else np.asarray(box_conf))

    candidates: List[Tuple[np.ndarray, float]] = []
    for index, points in enumerate(xy):
        visible = np.isfinite(points).all(axis=1)
        point_conf = None
        if conf is not None and index < len(conf):
            point_conf = np.asarray(conf[index], float)
            visible &= point_conf >= min_conf
        corners = _canonical_outer_corners(points[visible])
        if corners is None or not valid_court_quad(corners, frame_shape):
            continue
        confidence = float(np.mean(point_conf[visible])) if point_conf is not None else 1.0
        if box_conf is not None and index < len(box_conf):
            confidence *= float(box_conf[index])
        candidates.append((corners, confidence))
    return candidates


def _detect_with_keypoint_model(
    frames: Sequence[np.ndarray], weights: str, *, min_score: float,
    progress: Callable[[str], None],
) -> Optional[Tuple[Court, float, int]]:
    """Return a multi-frame court consensus from the required learned checkpoint."""
    if not frames:
        return None
    path = Path(discover_court_weights(weights)).expanduser()
    if not path.is_file():
        progress(f"  court keypoint weights not found: {path}")
        return None
    learned_quads: Optional[List[Tuple[np.ndarray, float]]] = None
    try:
        learned_quads = _resnet_keypoint_quads(frames, str(path))
    except Exception as resnet_exc:
        # Keep support for explicitly configured Ultralytics court-pose checkpoints.
        try:
            from ultralytics import YOLO
            from .player import resolve_yolo_device

            model = YOLO(str(path))
            results = model.predict(
                source=list(frames), device=resolve_yolo_device(), verbose=False,
                conf=0.15,
            )
            learned_quads = [
                max(_keypoint_quads(result, frame.shape),
                    key=lambda item: item[1], default=(np.empty((0, 2)), 0.0))
                for frame, result in zip(frames, results)
            ]
        except Exception as yolo_exc:
            progress(
                "  court keypoint model unavailable "
                f"(resnet={resnet_exc}; yolo={yolo_exc})")
            return None

    candidates: List[Tuple[np.ndarray, float]] = []
    for frame, learned in zip(frames, learned_quads or []):
        edges = None
        for corners, confidence in ([learned] if learned[0].shape == (4, 2) else []):
            try:
                court = Court.from_image_corners(*corners)
            except Exception:
                continue
            if edges is None:
                import cv2
                edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 40, 120)
            # Learned confidence alone cannot distinguish a target court from a neighboring
            # one.  Require at least modest agreement with stationary painted lines.
            line_score = score_court(edges, court, band_px=max(3, frame.shape[1] // 400))
            # A regression model can return a stable, plausible-looking court in a framing
            # far outside its training distribution. Multi-frame agreement does not make
            # that geometry correct. Require the same full-template painted-line support
            # as the classical detector instead of accepting a half-strength match.
            if line_score >= min_score:
                candidates.append((corners, confidence * line_score))
    if not candidates:
        return None

    scale = float(np.hypot(*frames[0].shape[:2]))
    quads = np.stack([corners for corners, _score in candidates])
    neighbour_sets = []
    for quad in quads:
        distance = np.sqrt(np.mean(np.sum((quads - quad) ** 2, axis=2), axis=1))
        neighbour_sets.append(np.flatnonzero(distance <= 0.035 * scale))
    cluster = max(neighbour_sets, key=lambda ids: (
        ids.size, float(np.median([candidates[i][1] for i in ids]))))
    min_support = 1 if len(frames) == 1 else 2
    if cluster.size < min_support:
        progress("  court keypoint model rejected: no multi-frame geometric consensus")
        return None
    corners = np.median(quads[cluster], axis=0)
    if not valid_court_quad(corners, frames[0].shape):
        return None
    court = Court.from_image_corners(*corners)
    score = float(np.median([candidates[i][1] for i in cluster]))
    return court, score, int(cluster.size)


def normalize_hough_lines(raw) -> List[Line]:
    """Normalize OpenCV's version-dependent HoughLinesP output to ``N x 4``.

    OpenCV wheels in the wild return either ``(N, 1, 4)`` or ``(N, 4)``.  Indexing
    ``line[0]`` only works for the former and turned each line into a scalar on the latter.
    """
    if raw is None:
        return []
    arr = np.asarray(raw)
    if arr.size == 0:
        return []
    if arr.ndim == 1 and arr.shape[0] == 4:
        arr = arr.reshape(1, 4)
    elif arr.ndim >= 2 and arr.shape[-1] == 4:
        arr = arr.reshape(-1, 4)
    else:
        raise ValueError(f"unexpected HoughLinesP shape {arr.shape}")
    return [tuple(map(float, row)) for row in arr]


def _line_angle_deg(l: Line) -> float:
    x1, y1, x2, y2 = l
    return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def classify_lines(lines: List[Line], horiz_max_deg: float = 35.0,
                   vert_min_deg: float = 35.0) -> Tuple[List[Line], List[Line]]:
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


def valid_court_quad(corners, frame_shape, *, min_area_frac: float = 0.04,
                     max_area_frac: float = 0.95,
                     off_frame_margin_frac: float = 0.03,
                     max_homography_condition: float = 1e5) -> bool:
    """Reject off-frame, tiny, non-convex, or ill-conditioned court quadrilaterals.

    Hough-line intersections are infinite-line intersections; with nearly parallel or
    unrelated lines they can easily land thousands of pixels outside the image yet still
    produce an invertible homography.  Validate in normalized image coordinates so the
    thresholds do not depend on resolution.
    """
    pts = np.asarray(corners, float)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return False
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return False
    margin_x, margin_y = off_frame_margin_frac * w, off_frame_margin_frac * h
    if (np.any(pts[:, 0] < -margin_x) or np.any(pts[:, 0] > w - 1 + margin_x)
            or np.any(pts[:, 1] < -margin_y) or np.any(pts[:, 1] > h - 1 + margin_y)):
        return False

    # Shoelace area and consecutive cross products enforce a useful, convex quad.
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1))
                           - np.dot(y, np.roll(x, -1))))
    area_frac = area / float(w * h)
    if not min_area_frac <= area_frac <= max_area_frac:
        return False
    edges = np.roll(pts, -1, axis=0) - pts
    edge_len = np.linalg.norm(edges, axis=1)
    if np.min(edge_len) < 0.025 * np.hypot(w, h):
        return False
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(np.abs(crosses) < 1e-6) or not (
            np.all(crosses > 0) or np.all(crosses < 0)):
        return False

    # Expected point order is NL, NR, FR, FL.  Check the coarse perspective ordering
    # before asking OpenCV to fit a matrix.
    nl, nr, fr, fl = pts
    if not (nl[1] > fl[1] and nr[1] > fr[1]
            and nr[0] > nl[0] and fr[0] > fl[0]):
        return False
    near_width = float(np.linalg.norm(nr - nl))
    far_width = float(np.linalg.norm(fr - fl))
    # A perspective court narrows toward the far baseline.  Vanishingly thin far edges
    # are the common high-scoring degeneracy: they fold several model lines onto the same
    # real edge and appear to overlap extremely well.
    width_ratio = far_width / max(near_width, 1e-9)
    if not 0.12 <= width_ratio <= 0.95:
        return False
    try:
        import cv2
        src = (pts / np.array([w, h], float)).astype(np.float32)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        h_norm = cv2.getPerspectiveTransform(src, dst)
        condition = float(np.linalg.cond(h_norm))
    except Exception:
        return False
    return bool(np.isfinite(condition) and condition <= max_homography_condition)


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
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h:
                total += 1
                lo_y, hi_y = max(0, iy - band_px), min(h, iy + band_px + 1)
                lo_x, hi_x = max(0, ix - band_px), min(w, ix + band_px + 1)
                if white_mask[lo_y:hi_y, lo_x:hi_x].any():
                    hits += 1
    return hits / total if total else 0.0


def _surface_consistency(gray: np.ndarray, court: Court) -> float:
    """Return far/near playing-surface luminance agreement in ``[0, 1]``.

    A common baseline-camera alias maps the model's net onto the real far baseline and a
    bright fence/sign rail onto the model's far baseline. Painted-line overlap alone likes
    that geometry, but its far court half is actually dark fence. Temporal-median surface
    luminance makes the physical court and the fence alias easy to distinguish.
    """
    gray = np.asarray(gray, np.uint8)
    h, w = gray.shape[:2]
    medians: list[float] = []
    for y0, y1 in ((1.0, 10.0), (14.0, 22.0)):
        xs, ys = np.meshgrid(np.linspace(1.0, 9.97, 48), np.linspace(y0, y1, 56))
        pixels = np.asarray(court.to_image(np.column_stack((xs.ravel(), ys.ravel()))), float)
        if not np.isfinite(pixels).all():
            return 0.0
        px = np.round(pixels[:, 0]).astype(int)
        py = np.round(pixels[:, 1]).astype(int)
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if float(np.mean(inside)) < 0.9:
            return 0.0
        medians.append(float(np.median(gray[py[inside], px[inside]])))
    if max(medians) <= 1e-6:
        return 0.0
    return float(min(medians) / max(medians))


def _white_mask(frame_bgr: np.ndarray):
    """Bright, low-saturation (white paint) pixel mask."""
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # high value, low saturation = white lines regardless of court colour
    return cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))


def _line_features(line: Line, image_width: int):
    """Canonical ``(slope, intercept, length, y_at_image_centre, line)`` or None."""
    x1, y1, x2, y2 = map(float, line)
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx = x2 - x1
    if dx < 1e-6:
        return None
    slope = (y2 - y1) / dx
    intercept = y1 - slope * x1
    length = float(np.hypot(dx, y2 - y1))
    centre_y = slope * (image_width / 2.0) + intercept
    return slope, intercept, length, centre_y, (x1, y1, x2, y2)


def _cluster_by(values, distance: float, limit: int):
    """Greedily retain the longest representative from nearby line hypotheses."""
    kept = []
    for feature, key in sorted(values, key=lambda item: item[0][2], reverse=True):
        if all(float(np.hypot(*(np.asarray(key) - np.asarray(old_key)))) > distance
               for _old, old_key in kept):
            kept.append((feature, key))
            if len(kept) >= limit:
                break
    return [feature for feature, _key in kept]


def _plausible_target_alignment(corners, frame_shape) -> bool:
    """Reject shallow, laterally sheared multi-court aliases.

    In a baseline view a shallow near edge can be legitimate when the optical axis still
    follows the court. The dangerous alias combines a shallow near edge with a large
    sideways jump between the near/far baseline centres, typically by borrowing lines
    from adjacent courts. Deep near baselines remain valid regardless of that soft drift
    guard because strong perspective can amplify ordinary camera offset.
    """
    pts = np.asarray(corners, float)
    h, w = frame_shape[:2]
    near_centre = np.mean(pts[:2], axis=0)
    far_centre = np.mean(pts[2:], axis=0)
    # The general quad validator tolerates wider off-frame intersections for manual and
    # unusual views. Automatic target selection is stricter: borrowing a neighboring
    # sideline commonly puts one near corner 5--6% beyond the frame while preserving an
    # excellent line score.
    target_margin_x = 0.03 * w
    corners_near_frame = bool(
        np.all(pts[:, 0] >= -target_margin_x)
        and np.all(pts[:, 0] <= (w - 1) + target_margin_x))
    near_is_deep = near_centre[1] >= 0.78 * h
    centre_drift_is_small = abs(near_centre[0] - far_centre[0]) <= 0.08 * w
    return corners_near_frame and bool(near_is_deep or centre_drift_is_small)


def _segment_alignment(
    projected: np.ndarray,
    features,
    *,
    max_angle_deg: float,
    max_perpendicular_px: float,
    min_overlap: float,
    min_feature_length: float,
) -> bool:
    """Return whether one coherent Hough segment supports a projected court line.

    Point-wise edge overlap is not enough for court calibration: a net, fence rail, and
    painted baseline all contain edge pixels and can form a self-similar half-court alias.
    This check also requires the pixels to belong to one line with the expected direction
    and longitudinal extent.
    """
    points = np.asarray(projected, float).reshape(2, 2)
    direction = points[1] - points[0]
    projected_length = float(np.linalg.norm(direction))
    if not np.isfinite(points).all() or projected_length < 1.0:
        return False
    unit = direction / projected_length
    normal = np.asarray([-unit[1], unit[0]], float)

    for feature in features:
        _slope, _intercept, length, _centre_y, line = feature
        if length < min_feature_length:
            continue
        observed = np.asarray([[line[0], line[1]], [line[2], line[3]]], float)
        observed_direction = observed[1] - observed[0]
        observed_length = float(np.linalg.norm(observed_direction))
        if observed_length < 1.0:
            continue
        observed_unit = observed_direction / observed_length
        cosine = float(np.clip(abs(np.dot(unit, observed_unit)), 0.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        if angle > max_angle_deg:
            continue
        perpendicular = abs(float(np.dot(np.mean(observed, axis=0) - points[0], normal)))
        if perpendicular > max_perpendicular_px:
            continue
        positions = (observed - points[0]) @ unit
        overlap = max(
            0.0,
            min(projected_length, float(np.max(positions)))
            - max(0.0, float(np.min(positions))),
        )
        if overlap / projected_length >= min_overlap:
            return True
    return False


def _near_service_reference(features, near_line, frame_shape):
    """Find the deepest coherent transverse line above the selected near baseline.

    In a baseline view this is the near service line. Selecting it independently of a
    court hypothesis prevents that hypothesis from relabelling the physical net as a
    service line, which raw template-overlap scoring otherwise permits.
    """
    h, w = frame_shape[:2]
    near_y = float(near_line[3])
    candidates = [
        feature for feature in features
        if abs(feature[0]) <= 0.08
        and feature[2] >= 0.25 * w
        and near_y - 0.35 * h <= feature[3] <= near_y - 0.04 * h
    ]
    if not candidates:
        return None
    # Duplicate Hough edges of the same painted stripe are harmless. The physically
    # deepest long transverse line is the near service line; no other court cross-line
    # lies between it and the near baseline.
    return max(candidates, key=lambda feature: (feature[3], feature[2]))


def _court_landmark_consistency(features, near_line, court: Court, frame_shape) -> bool:
    """Validate service-line depth and centre-line direction for a court hypothesis."""
    w = frame_shape[1]
    service_reference = _near_service_reference(features, near_line, frame_shape)
    if service_reference is None:
        return False

    near_service = court.to_image(np.asarray([
        [SINGLES_IN, SERVICE_Y],
        [DOUBLES_W - SINGLES_IN, SERVICE_Y],
    ], float))
    if not _segment_alignment(
        near_service,
        [service_reference],
        max_angle_deg=4.0,
        max_perpendicular_px=max(5.0, 0.012 * w),
        min_overlap=0.45,
        min_feature_length=0.20 * w,
    ):
        return False

    centre_service = court.to_image(np.asarray([
        [DOUBLES_W / 2.0, SERVICE_Y],
        [DOUBLES_W / 2.0, COURT_L - SERVICE_Y],
    ], float))
    return _segment_alignment(
        centre_service,
        features,
        max_angle_deg=18.0,
        max_perpendicular_px=max(7.0, 0.020 * w),
        min_overlap=0.35,
        # Strong-perspective views can compress the complete centre-service stripe to
        # only a few percent of frame width even when it remains a coherent landmark.
        min_feature_length=max(25.0, 0.025 * w),
    )


def _detect_in_stationary_gray(gray: np.ndarray, min_score: float = 0.55
                               ) -> Optional[Tuple[Court, float]]:
    """Detect a perspective court from a temporal-median grayscale frame.

    A court's full sidelines are diagonal in a normal baseline camera view, so splitting
    Hough lines into horizontal/vertical buckets cannot recover them.  This path forms
    trapezoids from two near-horizontal baselines and opposing diagonal line families,
    then scores the *entire* projected court model against thin Canny edges.  Temporal
    median aggregation removes moving players while preserving stationary paint.
    """
    import cv2

    gray = np.asarray(gray, np.uint8)
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 40, 120)
    def hough_features(max_gap_frac: float):
        raw = cv2.HoughLinesP(
            edges, 1, np.pi / 720, threshold=20,
            # Court paint is repeatedly occluded by the net and players. Normal views can
            # bridge larger gaps; shallow views need shorter joins so lines from adjacent
            # courts are not fused into one misleading diagonal.
            minLineLength=max(60, int(0.03 * w)),
            maxLineGap=max(30, int(max_gap_frac * w)),
        )
        found = [_line_features(line, w) for line in normalize_hough_lines(raw)]
        return [feature for feature in found if feature is not None]

    def build_hypotheses(
        features, *, far_band: Tuple[float, float],
        near_band: Tuple[float, float], separation_band: Tuple[float, float],
    ) -> List[Tuple[Court, float]]:
        baseline_values = []
        for feature in features:
            slope, _intercept, length, centre_y, _line = feature
            if abs(slope) <= 0.08 and 0.35 * h <= centre_y <= 0.90 * h \
                    and length >= 0.06 * w:
                baseline_values.append((feature, (centre_y, 0.0)))
        # Long fence, net, and neighboring-court lines can outnumber the target court's
        # near baseline. Keep enough distinct y-levels for the full target baseline to
        # survive ranking; geometric pairing and full-model scoring do the real pruning.
        baselines = _cluster_by(
            baseline_values, max(5.0, 0.007 * h), limit=64)
        far_lines = [
            feature for feature in baselines
            if far_band[0] * h <= feature[3] <= far_band[1] * h
        ]
        near_lines = [
            feature for feature in baselines
            if near_band[0] * h <= feature[3] <= near_band[1] * h
        ]
        hypotheses: List[Tuple[Court, float]] = []
        band_px = max(3, int(round(0.003 * w)))
        for far in far_lines:
            for near in near_lines:
                far_y, near_y = far[3], near[3]
                if not separation_band[0] * h <= near_y - far_y <= separation_band[1] * h:
                    continue

                side_values = {"left": [], "right": []}
                for feature in features:
                    slope, intercept, length, _centre_y, _line = feature
                    # A low baseline camera can place the near doubles corners close to the
                    # image edges, producing genuinely steep sidelines. The old 0.90 cap
                    # rejected these courts and left adjacent-court motion uncalibrated.
                    if length < 0.06 * w or not 0.15 <= abs(slope) <= 1.60:
                        continue
                    x_far = (far_y - intercept) / slope
                    x_near = (near_y - intercept) / slope
                    if (slope < 0 and -0.10 * w <= x_near <= 0.40 * w
                            and 0.25 * w <= x_far <= 0.70 * w and x_near < x_far):
                        side_values["left"].append((feature, (x_near, x_far)))
                    elif (slope > 0 and 0.60 * w <= x_near <= 1.10 * w
                          and 0.30 * w <= x_far <= 0.80 * w and x_far < x_near):
                        side_values["right"].append((feature, (x_near, x_far)))

                left = _cluster_by(side_values["left"], 0.025 * w, limit=16)
                right = _cluster_by(side_values["right"], 0.025 * w, limit=16)
                for left_line in left:
                    for right_line in right:
                        corners = (
                            line_intersection(near[4], left_line[4]),
                            line_intersection(near[4], right_line[4]),
                            line_intersection(far[4], right_line[4]),
                            line_intersection(far[4], left_line[4]),
                        )
                        if not valid_court_quad(
                                corners, gray.shape, min_area_frac=0.06,
                                off_frame_margin_frac=0.06):
                            continue
                        pts = np.asarray(corners, float)
                        near_centre_x = 0.5 * (pts[0, 0] + pts[1, 0])
                        far_centre_x = 0.5 * (pts[2, 0] + pts[3, 0])
                        if not (0.30 * w <= near_centre_x <= 0.70 * w
                                and 0.35 * w <= far_centre_x <= 0.65 * w):
                            continue
                        if not _plausible_target_alignment(pts, gray.shape):
                            continue
                        try:
                            court = Court.from_image_corners(*corners)
                        except Exception:
                            continue
                        # A raw edge score can map the net/service-line subset onto the
                        # complete template. Require independently detected painted
                        # landmarks before letting such a projective alias compete.
                        if not _court_landmark_consistency(
                                features, near, court, gray.shape):
                            continue
                        score = score_court(edges, court, band_px=band_px)
                        hypotheses.append((court, score))
        return hypotheses

    # Fit several baseline-camera profiles. A single vertical band is brittle: ordinary
    # phone footage often puts the near baseline in the bottom 5--10% of the image, while
    # elevated cameras compress the entire playing rectangle toward the centre. The full
    # projected court template and ambiguity guard below remain the final judges.
    profiles = (
        # Elevated/wide-angle baseline cameras.
        # Phone cameras immediately behind the baseline often place the real near line at
        # 60--67% image height because the frame includes sky and fence above the court.
        ((0.35, 0.65), (0.58, 0.90), (0.12, 0.45)),
        # Low tripod/phone cameras with a deep, sometimes partially clipped near baseline.
        ((0.38, 0.68), (0.82, 0.97), (0.20, 0.58)),
    )
    hypotheses: List[Tuple[Court, float]] = []
    for max_gap_frac in (0.08, 0.035):
        features = hough_features(max_gap_frac)
        for far_band, near_band, separation_band in profiles:
            hypotheses.extend(build_hypotheses(
                features,
                far_band=far_band,
                near_band=near_band,
                separation_band=separation_band,
            ))
    if not hypotheses:
        return None
    # Net/service-line subsets can score better than the complete court because the same
    # painted-line pattern is projectively self-similar. If the image provides a valid
    # deeper near baseline, it owns the view; retain shallow hypotheses only for cameras
    # where no deep full-court geometry is available at all.
    deep_hypotheses = [
        candidate for candidate in hypotheses
        if float(np.mean(candidate[0].corners_img[:2, 1])) >= 0.78 * h
    ]
    if deep_hypotheses:
        hypotheses = deep_hypotheses
    hypotheses.sort(key=lambda item: item[1], reverse=True)
    best = hypotheses[0]
    if best[1] < min_score:
        return None

    # A net/service-line subset can imitate a full court under a projective transform.
    # Reject when a materially different quad scores almost as well; a fixed camera gives
    # us repeatability, not permission to choose arbitrarily among ambiguous geometries.
    diag = float(np.hypot(w, h))
    best_quad = best[0].corners_img.astype(float)
    runner_up = None
    for candidate in hypotheses[1:]:
        distance = float(np.sqrt(np.mean(np.sum(
            (candidate[0].corners_img.astype(float) - best_quad) ** 2, axis=1))))
        if distance > 0.035 * diag:
            runner_up = candidate
            break
    if runner_up is not None and best[1] - runner_up[1] < 0.04:
        return None
    return best


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
    lines = normalize_hough_lines(raw)
    horiz, vert = classify_lines(lines)
    corners = corners_from_lines(horiz, vert)
    if corners is None:
        return None
    nl, nr, fr, fl = corners
    if not valid_court_quad(corners, frame_bgr.shape):
        return None
    try:
        court = Court.from_image_corners(nl, nr, fr, fl)
    except Exception:
        return None
    if not _plausible_target_alignment(court.corners_img, frame_bgr.shape):
        return None
    score = score_court(mask, court)
    if score < min_score:
        return None
    # Per-frame extrema are the weakest fallback and can lock onto a fence/adjacent court
    # whose white rails form a neat trapezoid. Both halves of a real playing surface must
    # retain at least modest luminance agreement. When this cannot be established, abstain
    # and let multi-frame stationary geometry decide rather than poisoning every consumer.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if _surface_consistency(gray, court) < 0.55:
        return None
    return court, score


def detect_court(video: str, cfg=None, *, n_frames: int = 12, min_score: float = 0.55,
                 progress: Callable[[str], None] = lambda _m: None) -> Optional[Court]:
    """Sample frames across ``video`` and return the best-scoring court homography, or None.

    When ``cfg.court_weights`` names an Ultralytics court-keypoint checkpoint, its
    multi-frame consensus is tried first. The classical detector remains the fallback.
    Returns ``None`` when neither backend yields a confident court, so the caller safely
    abstains from target-court claims.
    """
    import cv2

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(total * 0.05, total * 0.95, n_frames).astype(int)
    early_total = min(total, int(round(130.0 * fps))) if fps > 0 else total
    early_idxs = np.linspace(
        early_total * 0.05, early_total * 0.95, n_frames).astype(int)
    candidates: List[Tuple[Court, float]] = []
    sampled_frames: List[np.ndarray] = []
    gray_frames: List[np.ndarray] = []
    early_gray_frames: List[np.ndarray] = []
    frame_shape = None
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_shape = frame.shape
        sampled_frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = _detect_in_frame(frame, min_score)
        if found:
            candidates.append(found)
    if not np.array_equal(early_idxs, idxs):
        for fi in early_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                early_gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    court_weights = getattr(cfg, "court_weights", None) if cfg is not None else None
    if court_weights:
        learned = _detect_with_keypoint_model(
            sampled_frames, str(court_weights), min_score=min_score, progress=progress)
        if learned is not None:
            court, score, support = learned
            progress(
                f"  court detected by keypoint model in {support}/{len(sampled_frames)} "
                f"sampled frames (consensus score {score:.2f})")
            return court

    # The camera is stationary: aggregate all successfully sampled frames before falling
    # back to noisier per-frame hypotheses.  Unlike simply voting on per-frame extrema,
    # this lets line fragments occluded by different players reinforce one geometry.
    aggregate = None
    stationary = None
    if len(gray_frames) >= 3:
        stationary = np.median(np.stack(gray_frames, axis=0), axis=0).astype(np.uint8)
        aggregate = _detect_in_stationary_gray(stationary, min_score=min_score)
    early_aggregate = None
    early_stationary = None
    if len(early_gray_frames) >= 3:
        early_stationary = np.median(
            np.stack(early_gray_frames, axis=0), axis=0).astype(np.uint8)
        early_aggregate = _detect_in_stationary_gray(
            early_stationary, min_score=min_score)

    # A fixed-camera court should not depend on video duration. For long recordings the
    # all-video median can accumulate lighting/player-state aliases, so reuse the clean
    # first-130-second geometry when the global median abstains or agrees with it.
    if early_aggregate is not None:
        use_early = aggregate is None
        if aggregate is not None and frame_shape is not None:
            h, w = frame_shape[:2]
            distance = float(np.sqrt(np.mean(np.sum(
                (early_aggregate[0].corners_img.astype(float)
                 - aggregate[0].corners_img.astype(float)) ** 2, axis=1))))
            use_early = distance <= 0.035 * float(np.hypot(w, h))
        if use_early:
            progress(
                f"  court detected from {len(early_gray_frames)} early stationary-frame "
                f"samples (edge-overlap score {early_aggregate[1]:.2f})")
            return early_aggregate[0]
        if aggregate is not None:
            # Both multi-frame detectors succeeded but found materially different courts.
            # Falling through to the noisiest single-frame path discards the strongest
            # evidence. Prefer the hypothesis with greater full-template line overlap.
            selected, label, count = (
                (early_aggregate, "early", len(early_gray_frames))
                if early_aggregate[1] >= aggregate[1]
                else (aggregate, "global", len(gray_frames))
            )
            progress(
                f"  court aggregate disagreement; selected {label} {count}-frame "
                f"geometry by edge-overlap score {selected[1]:.2f}")
            return selected[0]
    if aggregate is not None:
        progress(f"  court detected from {len(gray_frames)} stationary-frame samples "
                 f"(edge-overlap score {aggregate[1]:.2f})")
        return aggregate[0]
    if not candidates or frame_shape is None:
        return None

    # A single high-scoring frame is not enough: players, railings, and score graphics
    # can form an accidental quadrilateral.  Cluster normalized corner locations across
    # independently sampled frames and require repeatability.
    h, w = frame_shape[:2]
    scale = float(np.hypot(w, h))
    quads = np.stack([c.corners_img for c, _ in candidates]).astype(float)
    tolerance = 0.035 * scale
    neighbour_sets = []
    for i, quad in enumerate(quads):
        dist = np.sqrt(np.mean(np.sum((quads - quad) ** 2, axis=2), axis=1))
        neighbour_sets.append(np.flatnonzero(dist <= tolerance))
    cluster = max(neighbour_sets,
                  key=lambda ids: (ids.size,
                                   float(np.median([candidates[j][1] for j in ids]))))
    min_support = 1 if n_frames <= 1 else 2
    if cluster.size < min_support:
        progress("  court detection rejected: no multi-frame geometric consensus")
        return None

    consensus_corners = np.median(quads[cluster], axis=0)
    if not valid_court_quad(consensus_corners, frame_shape):
        return None
    try:
        court = Court.from_image_corners(*consensus_corners)
    except Exception:
        return None
    consensus_score = float(np.median([candidates[j][1] for j in cluster]))
    progress(f"  court detected in {cluster.size}/{len(candidates)} candidate frames "
             f"(median line-overlap score {consensus_score:.2f})")
    return court
