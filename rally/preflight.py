"""Strict installation preflight for the vision-first web server."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .config import RallyConfig


class InstallationError(RuntimeError):
    """Raised when the server installation is incomplete or unusable."""


_REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "OpenCV headless": "cv2",
    "PyTorch": "torch",
    "TorchVision": "torchvision",
    "Ultralytics": "ultralytics",
    "LAP assignment solver": "lap",
    "RTMLib": "rtmlib",
    "ONNX Runtime": "onnxruntime",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "python-multipart": "multipart",
}


def _local_model_path(configured: str | None, discovered: str | None) -> Path | None:
    if configured:
        direct = Path(configured).expanduser()
        if direct.is_file():
            return direct.resolve()
    if discovered:
        candidate = Path(discovered).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def _check_ffmpeg() -> None:
    from .io.ffmpeg import _require, _video_encoder, probe

    executable = _require("ffmpeg")
    codec, codec_args = _video_encoder()
    if codec not in {"h264_nvenc", "libx264", "libopenh264"}:
        raise RuntimeError(f"no browser-compatible H.264 encoder (selected {codec!r})")
    with tempfile.TemporaryDirectory(prefix="rally-preflight-") as directory:
        clip = Path(directory) / "smoke.mp4"
        subprocess.run(
            [
                executable, "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=64x64:r=5:d=1", "-c:v", codec, *codec_args,
                "-pix_fmt", "yuv420p", str(clip),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        info = probe(str(clip))
        if info.width != 64 or info.fps <= 0:
            raise RuntimeError(f"encoded ffmpeg smoke test could not be probed: {info}")


def _check_yolo(path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(path))
    if getattr(model, "task", None) not in {None, "detect"}:
        raise RuntimeError(
            f"player/racket model must be a detection checkpoint, got {model.task!r}")
    names = getattr(model, "names", {}) or {}
    label = names.get(38) if isinstance(names, dict) else (
        names[38] if len(names) > 38 else None)
    if str(label).lower().replace("_", " ") != "tennis racket":
        raise RuntimeError(
            "player/racket model must expose COCO class 38 as 'tennis racket'")


def _check_court(path: Path) -> None:
    import hashlib

    import torch

    from .signals.court_detect import (
        COURT_MODEL_SOURCE_SHA256,
        COURT_MODEL_SOURCE_SIZE,
        load_court_keypoint_model,
    )

    if path.stat().st_size != COURT_MODEL_SOURCE_SIZE:
        raise RuntimeError("court checkpoint size does not match the pinned asset")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != COURT_MODEL_SOURCE_SHA256:
        raise RuntimeError("court checkpoint SHA-256 does not match the pinned asset")
    model = load_court_keypoint_model(str(path), device="cpu")
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 224, 224), dtype=torch.float32))
    if tuple(output.shape) != (1, 28) or not torch.isfinite(output).all():
        raise RuntimeError(f"unexpected court-model smoke output: {tuple(output.shape)}")


def _check_rtmpose(path: Path, cfg: RallyConfig) -> None:
    from rtmlib import RTMPose

    from .signals.pose import resolve_rtmpose_device

    device = resolve_rtmpose_device(cfg.rtmpose_device, cfg.rtmpose_runtime)
    estimator = RTMPose(
        str(path),
        model_input_size=(192, 256),
        to_openpose=False,
        backend=cfg.rtmpose_runtime,
        device=device,
    )
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    keypoints, scores = estimator(frame, bboxes=[[200, 40, 440, 350]])
    if np.asarray(keypoints).shape != (1, 17, 2) or np.asarray(scores).shape != (1, 17):
        raise RuntimeError(
            "unexpected COCO-17 output shapes: "
            f"{np.asarray(keypoints).shape}, {np.asarray(scores).shape}")


def installation_errors(
    *,
    yolo_path: str | None = None,
    rtmpose_path: str | None = None,
    court_path: str | None = None,
    check_packages: bool = True,
    check_ffmpeg: bool = True,
    load_models: bool = True,
) -> list[str]:
    """Return every setup defect found, allowing one run to report all required repairs."""
    errors: list[str] = []
    if check_packages:
        for label, module_name in _REQUIRED_IMPORTS.items():
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - report every broken dependency
                errors.append(f"required package {label} is unusable: {exc}")

    cfg = RallyConfig()
    try:
        from .signals.player import discover_yolo_weights

        yolo_discovered = discover_yolo_weights(cfg.player_detection_model)
    except Exception as exc:  # noqa: BLE001 - discovery failures are preflight output
        yolo_discovered = None
        errors.append(f"YOLO discovery failed: {exc}")
    yolo = _local_model_path(
        yolo_path or cfg.player_detection_model,
        None if yolo_path else yolo_discovered,
    )

    try:
        from .signals.pose import discover_rtmpose_weights

        pose_discovered = discover_rtmpose_weights(cfg.player_pose_model)
    except Exception as exc:  # noqa: BLE001 - discovery failures are preflight output
        pose_discovered = None
        errors.append(f"RTMPose discovery failed: {exc}")
    pose = _local_model_path(
        rtmpose_path or cfg.player_pose_model,
        None if rtmpose_path else pose_discovered,
    )

    try:
        from .signals.court_detect import discover_court_weights

        court_discovered = discover_court_weights(cfg.court_weights)
    except Exception as exc:  # noqa: BLE001 - discovery failures are preflight output
        court_discovered = None
        errors.append(f"court-model discovery failed: {exc}")
    court = _local_model_path(
        court_path or cfg.court_weights,
        None if court_path else court_discovered,
    )

    required_models = [
        ("YOLO player detector", yolo, _check_yolo),
        ("RTMPose", pose, lambda path: _check_rtmpose(path, cfg)),
        ("court keypoint detector", court, _check_court),
    ]
    for label, path, validator in required_models:
        if path is None:
            errors.append(f"required {label} model is missing")
            continue
        if load_models:
            try:
                validator(path)
            except Exception as exc:  # noqa: BLE001 - validators cross library boundaries
                errors.append(f"required {label} model is unusable ({path}): {exc}")

    if check_ffmpeg:
        try:
            _check_ffmpeg()
        except Exception as exc:  # noqa: BLE001 - ffmpeg diagnostics are preflight output
            errors.append(f"required ffmpeg H.264 runtime is unusable: {exc}")
    return errors


def require_server_install(**kwargs) -> None:
    """Refuse startup unless every accuracy-first server dependency is usable."""
    errors = installation_errors(**kwargs)
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise InstallationError(
            "Rally server setup is incomplete; refusing to start:\n"
            f"{detail}\n"
            "Run ./setup.sh successfully before launching the server."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rally.preflight",
        description="Verify every package, binary, and model required by rally-web.",
    )
    parser.add_argument("--yolo")
    parser.add_argument("--rtmpose")
    parser.add_argument("--court")
    args = parser.parse_args(argv)
    try:
        require_server_install(
            yolo_path=args.yolo,
            rtmpose_path=args.rtmpose,
            court_path=args.court,
        )
    except InstallationError as exc:
        parser.exit(1, f"[preflight] {exc}\n")
    print("[preflight] all required packages, models, and media runtimes are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
