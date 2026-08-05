"""Strict installation preflight for the accuracy-first web server.

The analysis pipeline can expose individual optional channels to library/CLI callers, but
the web server is an accuracy-first product surface.  It must never start from a partial
installation and silently turn a requested match analysis into audio-only segmentation.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

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
    "Ultralytics": "ultralytics",
    "RTMLib": "rtmlib",
    "ONNX Runtime": "onnxruntime",
    "gdown": "gdown",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "python-multipart": "multipart",
    "HTTPX": "httpx",
}


def _local_model_path(configured: Optional[str], discovered: Optional[str]) -> Optional[Path]:
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


def _check_tracknet(path: Path) -> None:
    import torch

    from .vendor.tracknet_torch import BallTrackerNet

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    state = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    model = BallTrackerNet()
    model.load_state_dict(state)


def _check_yolo(path: Path) -> None:
    from ultralytics import YOLO

    model = YOLO(str(path))
    if getattr(model, "task", None) not in {None, "detect"}:
        raise RuntimeError(
            f"player model must be a detection checkpoint, got {model.task!r}")


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
    tracknet_path: Optional[str] = None,
    yolo_path: Optional[str] = None,
    rtmpose_path: Optional[str] = None,
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
            except Exception as exc:
                errors.append(f"required package {label} is unusable: {exc}")

    cfg = RallyConfig()
    try:
        from .signals.ball import discover_ball_weights

        ball_discovered = discover_ball_weights()
    except Exception as exc:
        ball_discovered = None
        errors.append(f"TrackNet discovery failed: {exc}")
    configured_ball = tracknet_path or cfg.ball_weights
    ball = _local_model_path(
        configured_ball,
        None if configured_ball else ball_discovered,
    )

    try:
        from .signals.player import discover_yolo_weights

        yolo_discovered = discover_yolo_weights(cfg.player_detection_model)
    except Exception as exc:
        yolo_discovered = None
        errors.append(f"YOLO discovery failed: {exc}")
    yolo = _local_model_path(
        yolo_path or cfg.player_detection_model,
        None if yolo_path else yolo_discovered,
    )

    try:
        from .signals.pose import discover_rtmpose_weights

        pose_discovered = discover_rtmpose_weights(cfg.player_pose_model)
    except Exception as exc:
        pose_discovered = None
        errors.append(f"RTMPose discovery failed: {exc}")
    pose = _local_model_path(
        rtmpose_path or cfg.player_pose_model,
        None if rtmpose_path else pose_discovered,
    )

    required_models = (
        ("TrackNet", ball, _check_tracknet),
        ("YOLO player detector", yolo, _check_yolo),
        ("RTMPose", pose, lambda path: _check_rtmpose(path, cfg)),
    )
    for label, path, validator in required_models:
        if path is None:
            errors.append(f"required {label} model is missing")
            continue
        if load_models:
            try:
                validator(path)
            except Exception as exc:
                errors.append(f"required {label} model is unusable ({path}): {exc}")

    if check_ffmpeg:
        try:
            _check_ffmpeg()
        except Exception as exc:
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rally.preflight",
        description="Verify every package, binary, and model required by rally-web.",
    )
    parser.add_argument("--tracknet")
    parser.add_argument("--yolo")
    parser.add_argument("--rtmpose")
    args = parser.parse_args(argv)
    try:
        require_server_install(
            tracknet_path=args.tracknet,
            yolo_path=args.yolo,
            rtmpose_path=args.rtmpose,
        )
    except InstallationError as exc:
        parser.exit(1, f"[preflight] {exc}\n")
    print("[preflight] all required packages, models, and media runtimes are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
