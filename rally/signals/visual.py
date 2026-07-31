"""Single-pass visual analysis: sample frames once and extract motion, camera-motion
and (optional) player-geometry channels aligned to the analysis timeline.

Decoding a 2-3 h video is the expensive step, so we do it exactly once, sampling at
``analysis_fps`` and running person detection only every few frames (``player_fps``).
Requires OpenCV; callers should guard with ``opencv_available()``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import RallyConfig
from .motion import camera_shift_px, frame_diff_energy
from .player import (
    PlayerDetector,
    estimate_court_region,
    geometry_score_from_court_persons,
    geometry_score_from_persons,
    persons_in_court,
)


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except Exception:
        return False


def _downscaled_gray(frame_bgr: np.ndarray, target_h: int):
    import cv2

    h, w = frame_bgr.shape[:2]
    if h > target_h:
        scale = target_h / h
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), target_h),
                               interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def target_court_mask(court, source_shape, output_shape, *,
                      sideline_margin_m: float = 1.5,
                      baseline_margin_m: float = 3.0) -> Optional[np.ndarray]:
    """Rasterize the target court and a small player apron at analysis resolution."""
    import cv2

    from .court import COURT_L, DOUBLES_W

    src_h, src_w = source_shape[:2]
    out_h, out_w = output_shape[:2]
    model = np.array([
        [-sideline_margin_m, -baseline_margin_m],
        [DOUBLES_W + sideline_margin_m, -baseline_margin_m],
        [DOUBLES_W + sideline_margin_m, COURT_L + baseline_margin_m],
        [-sideline_margin_m, COURT_L + baseline_margin_m],
    ], dtype=float)
    try:
        polygon = np.asarray(court.to_image(model), dtype=float).reshape(4, 2)
    except Exception:
        return None
    if not np.isfinite(polygon).all() or src_h <= 0 or src_w <= 0:
        return None
    polygon *= np.array([out_w / src_w, out_h / src_h], dtype=float)
    # Avoid integer overflow from an invalid extrapolated homography. OpenCV clips a
    # modest off-frame polygon correctly, so retain one frame of margin.
    polygon[:, 0] = np.clip(polygon[:, 0], -out_w, 2 * out_w)
    polygon[:, 1] = np.clip(polygon[:, 1], -out_h, 2 * out_h)
    points = np.round(polygon).astype(np.int32)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    # A tiny/degenerate projection is not enough geometry to interpret frame motion.
    if np.mean(mask > 0) < 0.02:
        return None
    return mask.astype(bool)


def masked_frame_diff_energy(prev_gray: np.ndarray, gray: np.ndarray,
                             mask: np.ndarray) -> float:
    """Normalized frame difference restricted to the calibrated target court."""
    a = np.asarray(prev_gray, dtype=np.float32)
    b = np.asarray(gray, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if a.shape != b.shape or m.shape != a.shape or not m.any():
        return 0.0
    return float(np.mean(np.abs(b[m] - a[m])) / 255.0)


def analyze_visual(
    path: str,
    cfg: RallyConfig,
    timeline_s: np.ndarray,
    detector: Optional[PlayerDetector] = None,
    court=None,
    progress=lambda _m: None,
    cancel_check: Callable[[], None] = lambda: None,
) -> Dict[str, Optional[np.ndarray]]:
    """Return dict with 'motion', 'camera_moving', 'geometry' arrays (length == timeline).

    ``court`` is the already-resolved target court. When court geometry was requested but
    unavailable, motion/geometry abstain rather than using whole-frame neighboring-court
    activity. If decoding stops early, the remaining timeline stays at zero.
    """
    import cv2

    timeline_s = np.asarray(timeline_s, dtype=float)
    T = timeline_s.size
    target_required = bool(cfg.target_court_required)
    target_missing = target_required and court is None
    motion_values = np.zeros(T, dtype=float)
    camera_px = np.zeros(T, dtype=float)

    use_players = detector is not None and detector.available and not target_missing
    player_stride = max(1, int(round(cfg.analysis_fps / max(cfg.player_fps, 1e-6))))
    persons_per_index: List[List[Tuple[float, float, float]]] = [[] for _ in range(T)]
    # Preserve the actual detector samples for point-level serve-position validation.
    # Reusing them avoids decoding the full video and running person detection twice.
    player_samples: List[Tuple[float, List[Tuple[float, float, float]]]] = []
    all_feet: List[Tuple[float, float]] = []
    # near-player foot track (largest bottom-half player), for the movement gate
    near_px = np.full(T, np.nan, dtype=float)
    near_py = np.full(T, np.nan, dtype=float)
    last_near: Optional[Tuple[float, float]] = None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.analysis_fps
    target_frames = np.round(timeline_s * native_fps).astype(np.int64)

    prev_gray = None
    current = 0  # index of next frame to be grabbed
    ti = 0
    detected_persons: dict[int, List[Tuple[float, float, float]]] = {}
    pending_player_frames: list[tuple[int, float, np.ndarray]] = []
    motion_mask = None
    frame_shape = None

    def flush_player_batch() -> None:
        nonlocal last_near
        if not pending_player_frames or detector is None:
            return
        frames = [item[2] for item in pending_player_frames]
        if hasattr(detector, "detect_persons_batch"):
            batches = detector.detect_persons_batch(frames)
        else:  # compatibility for lightweight/custom detectors
            batches = [detector.detect_persons(frame) for frame in frames]
        for (index, sample_time, frame), persons in zip(pending_player_frames, batches):
            court_coords = None
            if court is not None:
                persons, court_coords = persons_in_court(
                    persons, court, frame.shape[:2])
            detected_persons[index] = list(persons)
            player_samples.append((
                float(sample_time),
                [(float(x), float(y), float(area)) for x, y, area in persons],
            ))
            all_feet.extend((fx, fy) for fx, fy, _area in persons)
            if court_coords is not None:
                from .court import NET_Y
                near = [person for person, coord in zip(persons, court_coords)
                        if coord[1] <= NET_Y]
            else:
                near = [person for person in persons if person[1] > 0.5]
            if not near:
                continue
            if last_near is not None:
                ranked = sorted(
                    near, key=lambda person: np.hypot(
                        person[0] - last_near[0], person[1] - last_near[1]))
                closest = ranked[0]
                if np.hypot(closest[0] - last_near[0],
                            closest[1] - last_near[1]) <= 0.25:
                    fx, fy, _area = closest
                else:
                    fx, fy, _area = max(near, key=lambda person: person[2])
            else:
                fx, fy, _area = max(near, key=lambda person: person[2])
            last_near = (fx, fy)
            near_px[index], near_py[index] = fx, fy
        pending_player_frames.clear()

    try:
        while ti < T:
            cancel_check()
            target = int(target_frames[ti])
            # advance to the target frame using cheap grabs
            while current <= target:
                ok = cap.grab()
                if not ok:
                    target = -1
                    break
                current += 1
            if target < 0:
                progress(f"  warning: decode stopped at {ti}/{T} analysis frames "
                         f"({timeline_s[ti]:.1f}s) — remaining timeline treated as no-activity")
                break
            ok, frame = cap.retrieve()
            if not ok:
                progress(f"  warning: frame retrieve failed at {ti}/{T} "
                         f"({timeline_s[ti]:.1f}s) — remaining timeline treated as no-activity")
                break
            gray = _downscaled_gray(frame, cfg.proxy_height)
            if frame_shape is None:
                frame_shape = frame.shape[:2]
                if court is not None:
                    motion_mask = target_court_mask(court, frame.shape, gray.shape)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                if court is not None:
                    motion_values[ti] = masked_frame_diff_energy(
                        prev_gray, gray, motion_mask) if motion_mask is not None else 0.0
                elif not target_missing:
                    motion_values[ti] = frame_diff_energy(prev_gray, gray)
                camera_px[ti] = camera_shift_px(prev_gray, gray)
            prev_gray = gray

            if use_players:
                if ti % player_stride == 0:
                    pending_player_frames.append((ti, float(timeline_s[ti]), frame))
                    if len(pending_player_frames) >= 16:
                        flush_player_batch()
            ti += 1
        flush_player_batch()
    finally:
        cap.release()

    # Match the prior channel semantics: each sparse detector sample remains current until
    # the next sample, while inference itself is now batched.
    if use_players:
        last_persons: List[Tuple[float, float, float]] = []
        # Never carry the final detection beyond the last successfully decoded frame;
        # that would fabricate target-player geometry through a truncated/cancelled tail.
        for index in range(ti):
            if index in detected_persons:
                last_persons = detected_persons[index]
            persons_per_index[index] = last_persons

    camera_moving = camera_px > cfg.camera_motion_px
    motion: Optional[np.ndarray] = (
        None if target_missing or (court is not None and motion_mask is None)
        else motion_values)

    geometry: Optional[np.ndarray] = None
    if use_players:
        if court is not None and frame_shape is not None:
            geometry = np.array([
                geometry_score_from_court_persons(
                    persons_per_index[i], court, frame_shape)
                for i in range(T)
            ], dtype=float)
        else:
            region = estimate_court_region(all_feet)
            geometry = np.array(
                [geometry_score_from_persons(persons_per_index[i], region)
                 for i in range(T)], dtype=float)

    near_track = None
    if use_players and np.isfinite(near_px).any():
        near_track = (near_px, near_py)

    return {"motion": motion, "camera_moving": camera_moving,
            "geometry": geometry, "near_track": near_track,
            "player_samples": player_samples,
            "frame_size": ((frame_shape[1], frame_shape[0])
                           if frame_shape is not None else None),
            "target_court_filtered": court is not None}
