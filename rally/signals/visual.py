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
    geometry_score_from_persons,
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


def analyze_visual(
    path: str,
    cfg: RallyConfig,
    timeline_s: np.ndarray,
    detector: Optional[PlayerDetector] = None,
    progress=lambda _m: None,
    cancel_check: Callable[[], None] = lambda: None,
) -> Dict[str, Optional[np.ndarray]]:
    """Return dict with 'motion', 'camera_moving', 'geometry' arrays (length == timeline).

    'geometry' is None when no detector is available. If decoding stops early (corrupt
    frame), the remaining timeline stays at its zero init and a warning is emitted rather
    than truncating silently.
    """
    import cv2

    timeline_s = np.asarray(timeline_s, dtype=float)
    T = timeline_s.size
    motion = np.zeros(T, dtype=float)
    camera_px = np.zeros(T, dtype=float)

    use_players = detector is not None and detector.available
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
    last_persons: List[Tuple[float, float, float]] = []
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
            if prev_gray is not None and prev_gray.shape == gray.shape:
                motion[ti] = frame_diff_energy(prev_gray, gray)
                camera_px[ti] = camera_shift_px(prev_gray, gray)
            prev_gray = gray

            if use_players:
                if ti % player_stride == 0:
                    last_persons = detector.detect_persons(frame)
                    player_samples.append((
                        float(timeline_s[ti]),
                        [(float(x), float(y), float(area))
                         for x, y, area in last_persons],
                    ))
                    for (fx, fy, _a) in last_persons:
                        all_feet.append((fx, fy))
                    # Keep one near-player identity over time. Picking the largest box on
                    # every sample independently switches to spectators/partners whenever
                    # their apparent size changes and corrupts the movement-reset signal.
                    near = [p for p in last_persons if p[1] > 0.5]
                    if near:
                        if last_near is not None:
                            ranked = sorted(
                                near, key=lambda p: np.hypot(p[0] - last_near[0],
                                                             p[1] - last_near[1]))
                            closest = ranked[0]
                            if np.hypot(closest[0] - last_near[0],
                                        closest[1] - last_near[1]) <= 0.25:
                                fx, fy, _a = closest
                            else:
                                fx, fy, _a = max(near, key=lambda p: p[2])
                        else:
                            fx, fy, _a = max(near, key=lambda p: p[2])
                        last_near = (fx, fy)
                        near_px[ti], near_py[ti] = fx, fy
                persons_per_index[ti] = last_persons
            ti += 1
    finally:
        cap.release()

    camera_moving = camera_px > cfg.camera_motion_px

    geometry: Optional[np.ndarray] = None
    if use_players:
        region = estimate_court_region(all_feet)
        geometry = np.array(
            [geometry_score_from_persons(persons_per_index[i], region) for i in range(T)],
            dtype=float,
        )

    near_track = None
    if use_players and np.isfinite(near_px).any():
        near_track = (near_px, near_py)

    return {"motion": motion, "camera_moving": camera_moving,
            "geometry": geometry, "near_track": near_track,
            "player_samples": player_samples}
