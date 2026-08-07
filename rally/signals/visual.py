"""Single-pass target-player identity tracking for the shared pose timeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import RallyConfig
from .player import PlayerDetector


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def analyze_visual(
    path: str,
    cfg: RallyConfig,
    timeline_s: np.ndarray,
    detector: PlayerDetector | None = None,
    court=None,
    progress=lambda _message: None,
    cancel_check: Callable[[], None] = lambda: None,
) -> dict:
    """Track all people attributable to the selected court at ``timeline_s``.

    This pass owns person detection and identity.  Downstream pose inference reuses its
    boxes instead of decoding an unrelated motion channel or running YOLO a second time.
    Court geometry enrolls an identity; once enrolled, the same tracker ID remains valid
    while legally chasing outside the lines.
    """
    import cv2

    if detector is None or not detector.available:
        raise RuntimeError("target-player detector is unavailable")
    if court is None:
        raise RuntimeError("target court is required for player attribution")
    sample_times = np.asarray(timeline_s, dtype=float)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    target_frames = np.maximum(
        0, np.round(sample_times * native_fps).astype(np.int64))
    next_frame = int(target_frames[0]) if target_frames.size else 0
    if target_frames.size:
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

    player_track_samples: list[tuple[float, list[dict]]] = []
    pending: list[tuple[float, np.ndarray]] = []
    target_track_ids: set[int] = set()
    frame_shape: tuple[int, int] | None = None

    def flush() -> None:
        if not pending:
            return
        frames = [frame for _time, frame in pending]
        tracked_batches = detector.track_persons_batch(frames)
        if len(tracked_batches) != len(frames):
            raise RuntimeError("player tracker returned a misaligned batch")
        for (sample_time, frame), raw_tracked in zip(
            pending, tracked_batches, strict=True
        ):
            tracked = list(raw_tracked)
            persons = [
                (float(item["foot_x_norm"]), float(item["foot_y_norm"]),
                 float(item["box_area_norm"]))
                for item in tracked
            ]
            if court is not None and tracked:
                from .court import COURT_L, DOUBLES_W

                height, width = frame.shape[:2]
                feet = np.asarray([
                    [person[0] * width, person[1] * height] for person in persons
                ], dtype=float)
                coordinates = np.asarray(
                    court.to_court(feet), dtype=float).reshape(-1, 2)
                inside = (
                    np.isfinite(coordinates).all(axis=1)
                    & (coordinates[:, 0] >= -1.5)
                    & (coordinates[:, 0] <= DOUBLES_W + 1.5)
                    & (coordinates[:, 1] >= -3.0)
                    & (coordinates[:, 1] <= COURT_L + 3.0))
                retained: list[int] = []
                for index, observation in enumerate(tracked):
                    raw_id = observation.get("track_id")
                    track_id = int(raw_id) if raw_id is not None else None
                    if inside[index] and track_id is not None:
                        target_track_ids.add(track_id)
                    if inside[index] or track_id in target_track_ids:
                        retained.append(index)
                persons = [persons[index] for index in retained]
                tracked = [tracked[index] for index in retained]
            if tracked:
                player_track_samples.append((sample_time, tracked))
        pending.clear()

    try:
        for index, (sample_time, target_frame) in enumerate(
            zip(sample_times, target_frames, strict=True), 1
        ):
            cancel_check()
            ok = True
            while next_frame <= int(target_frame):
                ok = cap.grab()
                if not ok:
                    break
                next_frame += 1
            if not ok:
                progress(
                    f"  warning: player decode stopped at {index - 1}/{len(sample_times)}")
                break
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            frame_shape = frame.shape[:2]
            pending.append((float(sample_time), frame))
            if len(pending) >= 16:
                flush()
                progress(f"  player identity progress {index}/{len(sample_times)}")
        flush()
    finally:
        cap.release()

    return {
        "player_track_samples": player_track_samples,
        "frame_size": ((frame_shape[1], frame_shape[0]) if frame_shape else None),
        "target_court_filtered": court is not None,
        "racket_observations": sum(
            int(bool(person.get("racket_bbox_norm")))
            for _time, people in player_track_samples for person in people),
    }
