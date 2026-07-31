"""Top-down player-pose inference inside YOLO target-court crops.

YOLO remains responsible for deciding *which person* belongs to the recorded court.
RTMPose receives only those boxes and performs higher-resolution keypoint estimation on
each crop. This separation keeps neighboring courts out of the pose signal and makes a
small far-side server materially larger to the pose network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np

from ..config import DEFAULT_RTMPOSE_MODEL, DEFAULT_YOLO_DETECTION_MODEL


@dataclass(frozen=True)
class PoseFrameResult:
    """Person boxes and COCO-17 poses aligned one-to-one for one video frame."""

    boxes: np.ndarray
    keypoints: np.ndarray
    confidence: np.ndarray

    @classmethod
    def empty(cls) -> "PoseFrameResult":
        return cls(
            boxes=np.empty((0, 4), dtype=float),
            keypoints=np.empty((0, 17, 2), dtype=float),
            confidence=np.empty((0, 17), dtype=float),
        )


def discover_rtmpose_weights(
    name: str = DEFAULT_RTMPOSE_MODEL,
    models_dir: Optional[str] = None,
) -> str:
    """Prefer an operator-supplied/local ONNX checkpoint before RTMLib download.

    RTMLib URLs normally point at a zip whose extracted checkpoint has the same basename
    with ``.onnx``. Placing that ONNX file in ``models/`` therefore makes startup fully
    offline and deterministic.
    """
    configured = Path(name).expanduser()
    if configured.is_file():
        return str(configured.resolve())
    if models_dir is None:
        models_dir = os.environ.get("RALLY_MODELS_DIR")
        if not models_dir:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")
    root = Path(models_dir)
    local = root / name
    if local.is_file():
        return str(local)
    parsed = urlparse(name)
    if parsed.scheme in {"http", "https"}:
        filename = Path(parsed.path).name
        extracted_name = f"{Path(filename).stem}.onnx" if filename else ""
        extracted = root / extracted_name
        if extracted_name and extracted.is_file():
            return str(extracted)
    return name


def resolve_rtmpose_device(requested: str, runtime: str) -> str:
    """Resolve ``auto`` against the providers actually installed for RTMLib."""
    requested = requested.strip().lower()
    if runtime == "opencv":
        # Ordinary OpenCV wheels do not include CUDA DNN. An explicit cuda request is
        # retained so RTMLib can report a useful configuration error.
        return "cpu" if requested == "auto" else requested
    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("RTMPose CUDA requires onnxruntime-gpu")
        return "cpu"
    has_cuda = "CUDAExecutionProvider" in providers
    if requested == "cuda" and not has_cuda:
        raise RuntimeError(
            "RTMPose CUDA requested but onnxruntime has no CUDAExecutionProvider")
    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    return requested


class CroppedRTMPose:
    """Batched YOLO detection followed by top-down RTMPose on selected boxes."""

    def __init__(
        self,
        *,
        detection_model: str = DEFAULT_YOLO_DETECTION_MODEL,
        pose_model: str = DEFAULT_RTMPOSE_MODEL,
        runtime: str = "onnxruntime",
        pose_device: str = "auto",
        detector=None,
        estimator=None,
        detection_device: Optional[str] = None,
    ) -> None:
        if detector is None:
            from ultralytics import YOLO
            from .player import discover_yolo_weights, resolve_yolo_device

            detection_device = detection_device or resolve_yolo_device()
            detector = YOLO(discover_yolo_weights(detection_model))
            detector.to(detection_device)
        if estimator is None:
            try:
                from rtmlib import RTMPose
            except ImportError as exc:
                raise RuntimeError(
                    "RTMLib pose backend is unavailable; install rtmlib") from exc
            resolved_pose = discover_rtmpose_weights(pose_model)
            resolved_device = resolve_rtmpose_device(pose_device, runtime)
            estimator = RTMPose(
                resolved_pose,
                model_input_size=(192, 256),
                to_openpose=False,
                backend=runtime,
                device=resolved_device,
            )
            self.pose_model = resolved_pose
            self.pose_device = resolved_device
        else:
            self.pose_model = pose_model
            self.pose_device = pose_device
        self.detector = detector
        self.estimator = estimator
        self.detection_device = detection_device or "cpu"
        self.runtime = runtime

    def predict(
        self,
        frames: Sequence[np.ndarray],
        *,
        court=None,
        target_required: bool = True,
        confidence: float = 0.15,
        image_size: int = 1280,
        batch_size: int = 16,
    ) -> list[PoseFrameResult]:
        if not frames:
            return []
        detections = self.detector.predict(
            list(frames), conf=confidence, classes=[0], verbose=False,
            imgsz=int(image_size), device=self.detection_device,
            batch=min(int(batch_size), len(frames)),
        )
        if len(detections) != len(frames):
            raise RuntimeError(
                "YOLO detector returned a different number of results than pose frames")
        output: list[PoseFrameResult] = []
        for frame, detection in zip(frames, detections):
            raw_boxes = getattr(detection, "boxes", None)
            if raw_boxes is None or len(raw_boxes) == 0:
                output.append(PoseFrameResult.empty())
                continue
            values = raw_boxes.xyxy
            boxes = np.asarray(
                values.cpu().numpy() if hasattr(values, "cpu") else values,
                dtype=float,
            ).reshape(-1, 4)
            if court is not None:
                from .player import target_court_box_indices

                indices = target_court_box_indices(boxes, court, frame.shape[:2])
                boxes = boxes[indices]
            elif target_required:
                boxes = np.empty((0, 4), dtype=float)
            if boxes.size == 0:
                output.append(PoseFrameResult.empty())
                continue
            height, width = frame.shape[:2]
            boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, max(0, width - 1))
            boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, max(0, height - 1))
            valid = (boxes[:, 2] - boxes[:, 0] >= 4) & (boxes[:, 3] - boxes[:, 1] >= 8)
            boxes = boxes[valid]
            if boxes.size == 0:
                output.append(PoseFrameResult.empty())
                continue
            keypoints, scores = self.estimator(frame, bboxes=boxes.tolist())
            keypoints = np.asarray(keypoints, dtype=float).reshape(len(boxes), -1, 2)
            scores = np.asarray(scores, dtype=float).reshape(len(boxes), -1)
            if keypoints.shape[1] < 17 or scores.shape[1] < 17:
                raise RuntimeError("RTMPose result does not contain COCO-17 body joints")
            output.append(PoseFrameResult(
                boxes=boxes,
                keypoints=keypoints[:, :17],
                confidence=scores[:, :17],
            ))
        return output
