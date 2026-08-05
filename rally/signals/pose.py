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


def rtmpose_execution_providers(runtime: str = "onnxruntime") -> list[str]:
    """Return active provider names without making capability checks fatal."""
    if runtime != "onnxruntime":
        return [runtime]
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []


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

    def _predict_dynamic_onnx_batch(
        self,
        frames: Sequence[np.ndarray],
        boxes_by_frame: Sequence[np.ndarray],
        batch_size: int,
    ) -> Optional[list[PoseFrameResult]]:
        """Run all person crops in true ONNX batches when the model permits it.

        RTMLib's public ``RTMPose.__call__`` loops over boxes and invokes ONNX Runtime
        once per person. The bundled model has a dynamic batch dimension, so grouping
        preprocessed crops removes hundreds of tiny session calls on a typical match.
        Return ``None`` for fixed-batch or non-ONNX backends so the portable path below
        remains available.
        """
        estimator = self.estimator
        if getattr(estimator, "backend", None) != "onnxruntime":
            return None
        session = getattr(estimator, "session", None)
        if session is None:
            return None
        inputs = session.get_inputs()
        if not inputs:
            return None
        input_shape = getattr(inputs[0], "shape", ())
        if not input_shape:
            return None
        batch_dimension = input_shape[0]
        if isinstance(batch_dimension, int):
            return None

        keypoints_by_frame = [
            np.empty((len(boxes), 17, 2), dtype=float) for boxes in boxes_by_frame
        ]
        confidence_by_frame = [
            np.empty((len(boxes), 17), dtype=float) for boxes in boxes_by_frame
        ]
        crops: list[np.ndarray] = []
        metadata: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for frame_index, (frame, boxes) in enumerate(zip(frames, boxes_by_frame)):
            for person_index, box in enumerate(boxes):
                crop, center, scale = estimator.preprocess(frame, box.tolist())
                crops.append(np.ascontiguousarray(
                    crop.transpose(2, 0, 1), dtype=np.float32))
                metadata.append((
                    frame_index,
                    person_index,
                    np.asarray(center, dtype=float),
                    np.asarray(scale, dtype=float),
                ))

        size = max(1, int(batch_size))
        output_names = [output.name for output in session.get_outputs()]
        input_name = inputs[0].name
        for start in range(0, len(crops), size):
            stop = min(len(crops), start + size)
            values = np.stack(crops[start:stop], axis=0)
            outputs = session.run(output_names, {input_name: values})
            for offset, (frame_index, person_index, center, scale) in enumerate(
                metadata[start:stop]
            ):
                per_person = [value[offset:offset + 1] for value in outputs]
                keypoints, confidence = estimator.postprocess(
                    per_person, center, scale)
                keypoints = np.asarray(keypoints, dtype=float).reshape(-1, 2)
                confidence = np.asarray(confidence, dtype=float).reshape(-1)
                if keypoints.shape[0] < 17 or confidence.shape[0] < 17:
                    raise RuntimeError(
                        "RTMPose result does not contain COCO-17 body joints")
                keypoints_by_frame[frame_index][person_index] = keypoints[:17]
                confidence_by_frame[frame_index][person_index] = confidence[:17]

        return [
            PoseFrameResult(
                boxes=boxes,
                keypoints=keypoints_by_frame[index],
                confidence=confidence_by_frame[index],
            )
            for index, boxes in enumerate(boxes_by_frame)
        ]

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
        boxes_by_frame: list[np.ndarray] = []
        for frame, detection in zip(frames, detections):
            raw_boxes = getattr(detection, "boxes", None)
            if raw_boxes is None or len(raw_boxes) == 0:
                boxes_by_frame.append(np.empty((0, 4), dtype=float))
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
                boxes_by_frame.append(np.empty((0, 4), dtype=float))
                continue
            height, width = frame.shape[:2]
            boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, max(0, width - 1))
            boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, max(0, height - 1))
            valid = (boxes[:, 2] - boxes[:, 0] >= 4) & (boxes[:, 3] - boxes[:, 1] >= 8)
            boxes = boxes[valid]
            if boxes.size == 0:
                boxes = np.empty((0, 4), dtype=float)
            boxes_by_frame.append(boxes)

        batched = self._predict_dynamic_onnx_batch(
            frames, boxes_by_frame, batch_size)
        if batched is not None:
            return batched

        output: list[PoseFrameResult] = []
        for frame, boxes in zip(frames, boxes_by_frame):
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
