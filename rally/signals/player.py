"""Target-court person identity tracking and reusable body-pose features."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import DEFAULT_YOLO_DETECTION_MODEL

Region = tuple[float, float, float, float]


def resolve_yolo_device() -> str:
    from ..inference import resolve_torch_device

    return str(resolve_torch_device())


def discover_yolo_weights(
    name: str = DEFAULT_YOLO_DETECTION_MODEL,
    models_dir: str | None = None,
) -> str:
    import os

    if models_dir is None:
        models_dir = os.environ.get("RALLY_MODELS_DIR")
        if not models_dir:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")
    local = Path(models_dir) / name
    return str(local) if local.is_file() else name


class PlayerDetector:
    """YOLO person tracking with same-pass tennis-racket observations."""

    def __init__(self, model: str = DEFAULT_YOLO_DETECTION_MODEL, conf: float = 0.3):
        self.conf = conf
        self.model = None
        self.device = "cpu"
        self.error: str | None = None
        self.tracker_config = str(
            Path(__file__).resolve().parents[1] / "trackers" / "botsort_reid.yaml")
        try:
            from ultralytics import YOLO

            self.device = resolve_yolo_device()
            self.model = YOLO(discover_yolo_weights(model))
            if getattr(self.model, "task", None) not in {None, "detect"}:
                raise ValueError(
                    f"player detection requires a detect checkpoint, got {self.model.task!r}")
            names = getattr(self.model, "names", {}) or {}
            racket_label = names.get(38) if isinstance(names, dict) else (
                names[38] if len(names) > 38 else None)
            if str(racket_label).lower().replace("_", " ") != "tennis racket":
                raise ValueError(
                    "player detection checkpoint lacks COCO tennis-racket class 38")
            self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001 - expose model/runtime initialization failure
            self.error = str(exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @staticmethod
    def _clothing_signature(frame_bgr: np.ndarray, box: Sequence[float]) -> list[float]:
        """Background-resistant upper/lower-body HSV identity descriptor."""
        import cv2

        height, width = frame_bgr.shape[:2]
        x0, y0, x1, y1 = [float(value) for value in box]
        box_width, box_height = x1 - x0, y1 - y0
        x0 = int(np.clip(round(x0 + 0.18 * box_width), 0, width - 1))
        x1 = int(np.clip(round(x1 - 0.18 * box_width), x0 + 1, width))
        y0 = int(np.clip(round(y0 + 0.08 * box_height), 0, height - 1))
        y1 = int(np.clip(round(y1 - 0.08 * box_height), y0 + 1, height))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        hsv = cv2.cvtColor(
            cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2HSV)
        features: list[np.ndarray] = []
        for part in (hsv[8:34], hsv[34:58]):
            histogram = cv2.calcHist(
                [part], [0, 1, 2], None, [12, 4, 4],
                [0, 180, 0, 256, 0, 256]).reshape(-1)
            histogram /= max(float(histogram.sum()), 1.0)
            features.append(np.sqrt(histogram))
        vector = np.concatenate(features).astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-9)
        return vector.round(6).tolist()

    def track_persons_batch(self, frames: Sequence[np.ndarray]) -> list[list[dict]]:
        """Detect a chronological batch and update one persistent tracker in order.

        Ultralytics treats a list of images as one non-stream source, so its tracking
        callback deliberately reuses tracker slot zero for every result in the list.
        ``persist=True`` then carries that same state into the next chronological batch.
        """
        if self.model is None:
            return [[] for _frame in frames]
        if not frames:
            return []
        results = self.model.track(
            list(frames), persist=True, tracker=self.tracker_config,
            conf=self.conf, classes=[0, 38], verbose=False, device=self.device,
            batch=min(16, len(frames)))
        output: list[list[dict]] = []
        for frame, result in zip(frames, results, strict=True):
            height, width = frame.shape[:2]
            ids = (result.boxes.id.cpu().numpy().astype(int)
                   if result.boxes.id is not None else None)
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            raw_boxes = result.boxes.xyxy.cpu().numpy()
            racket_indices = np.flatnonzero(classes == 38)
            observations: list[dict] = []
            for index, raw_box in enumerate(raw_boxes):
                if classes[index] != 0:
                    continue
                x0, y0, x1, y1 = [float(value) for value in raw_box]
                observation = {
                    "foot_x_norm": (x0 + x1) / (2.0 * width),
                    "foot_y_norm": y1 / height,
                    "box_area_norm": ((x1 - x0) * (y1 - y0)) / (width * height),
                    "bbox_norm": [x0 / width, y0 / height, x1 / width, y1 / height],
                    "confidence": float(confidences[index]),
                    "track_id": int(ids[index]) if ids is not None else None,
                    "appearance": self._clothing_signature(frame, raw_box),
                }
                # A racket often extends beyond the person box. Associate only rackets
                # whose centre lies in a modest expansion, then keep the nearest one.
                box_width, box_height = x1 - x0, y1 - y0
                person_centre = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
                candidates: list[tuple[float, int]] = []
                for racket_index in racket_indices:
                    rx0, ry0, rx1, ry1 = raw_boxes[racket_index]
                    centre = np.asarray([(rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0])
                    if not (
                        x0 - 0.65 * box_width <= centre[0] <= x1 + 0.65 * box_width
                        and y0 - 0.35 * box_height <= centre[1] <= y1 + 0.35 * box_height
                    ):
                        continue
                    distance = float(np.linalg.norm(centre - person_centre)
                                     / max(np.hypot(box_width, box_height), 1.0))
                    candidates.append((distance, int(racket_index)))
                if candidates:
                    _distance, racket_index = min(candidates)
                    rx0, ry0, rx1, ry1 = (
                        float(value) for value in raw_boxes[racket_index])
                    observation.update({
                        "racket_bbox_norm": [
                            rx0 / width, ry0 / height, rx1 / width, ry1 / height],
                        "racket_confidence": float(confidences[racket_index]),
                    })
                observations.append(observation)
            output.append(observations)
        if len(output) != len(frames):
            raise RuntimeError("YOLO tracker returned a misaligned identity batch")
        return output


def estimate_court_region(
    all_feet: Sequence[tuple[float, float]], margin: float = 0.05
) -> Region | None:
    points = np.asarray(all_feet, dtype=float)
    if points.shape[0] < 10:
        return None
    x0, y0 = np.percentile(points, 5, axis=0)
    x1, y1 = np.percentile(points, 95, axis=0)
    region = (max(0.0, x0 - margin), max(0.0, y0 - margin),
              min(1.0, x1 + margin), min(1.0, y1 + margin))
    if region[2] <= region[0] or region[3] <= region[1]:
        return None
    return tuple(float(value) for value in region)


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _body_pose_features(pose: np.ndarray, confidence: np.ndarray, cfg):
    """Return usability, athletic-ready stance, and highest wrist above shoulders."""
    body_joints = (5, 6, 11, 12, 13, 14, 15, 16)
    if pose.shape[0] < 17 or confidence.shape[0] < 17:
        return False, False, -1.0
    if float(np.mean(confidence[list(body_joints)])) < 0.45:
        return False, False, -1.0
    knee_angles = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if min(float(confidence[index]) for index in (hip, knee, ankle)) > 0.2:
            angle = _joint_angle_deg(pose[hip], pose[knee], pose[ankle])
            if np.isfinite(angle):
                knee_angles.append(angle)
    shoulder = (pose[5] + pose[6]) / 2.0
    hip = (pose[11] + pose[12]) / 2.0
    torso = float(np.linalg.norm(hip - shoulder))
    vertical_torso = abs(float(hip[1] - shoulder[1]))
    stance = float("nan")
    if torso > 1e-6 and min(float(confidence[15]), float(confidence[16])) > 0.2:
        stance = float(np.linalg.norm(pose[15] - pose[16]) / torso)
    ready = bool(
        (knee_angles and min(knee_angles) <= cfg.match_ready_knee_deg)
        or (np.isfinite(stance) and stance >= cfg.match_ready_stance_ratio))
    ratio = -1.0
    if vertical_torso > 1e-6:
        ratio = max((
            (float(shoulder[1]) - float(pose[wrist][1])) / vertical_torso
            for wrist in (9, 10)
            if float(confidence[wrist]) > 0.2
            and (pose[wrist][0] > 0 or pose[wrist][1] > 0)
        ), default=-1.0)
    return True, ready, float(ratio)


def _minimum_knee_angle(
    pose: np.ndarray, confidence: np.ndarray
) -> float | None:
    angles: list[float] = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if min(float(confidence[index]) for index in (hip, knee, ankle)) <= 0.2:
            continue
        value = _joint_angle_deg(pose[hip], pose[knee], pose[ankle])
        if np.isfinite(value):
            angles.append(float(value))
    return min(angles) if angles else None


def create_pose_runtime(cfg):
    """Create the RTMPose runtime shared by the entire pose timeline."""
    from .pose import CroppedRTMPose

    try:
        return CroppedRTMPose(
            pose_model=cfg.player_pose_model,
            runtime=cfg.rtmpose_runtime,
            pose_device=cfg.rtmpose_device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"required RTMPose model {cfg.player_pose_model!r} failed: {exc}") from exc
