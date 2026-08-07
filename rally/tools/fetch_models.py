"""Download and strictly verify the pipeline's pinned inference checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

def _max_model_bytes() -> int:
    return int(os.environ.get("RALLY_MAX_MODEL_BYTES", str(1024 * 1024 * 1024)))


def _check_size(path: str) -> None:
    size = os.path.getsize(path)
    if size <= 0 or size > _max_model_bytes():
        raise RuntimeError(f"checkpoint size {size} is outside the allowed range")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_digest(path: str, expected: Optional[str]) -> str:
    actual = _sha256(path)
    if expected and actual.lower() != expected.lower():
        raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _verify_court(path: str) -> None:
    from ..signals.court_detect import load_court_keypoint_model

    model = load_court_keypoint_model(path, device="cpu")
    if model is None:
        raise RuntimeError("court model construction returned no model")
    print(f"[fetch_models] OK: {path} loads into the 14-landmark ResNet50")


def _drive_id_from_url(url: str) -> Optional[str]:
    """Extract a Google Drive file ID from common share-URL shapes, else None."""
    if "drive.google.com" not in url:
        return None
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", url) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", url)
    return m.group(1) if m else None


def _gdown_command(file_id: str, dest: str) -> list[str]:
    """Build the gdown 5/6-compatible command (the file ID is positional)."""
    return [sys.executable, "-m", "gdown", file_id, "-O", dest]


def _download_drive(file_id: str, dest: str) -> None:
    """Download with gdown in a monitored child so size/time limits are enforced live."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive download needs gdown (it handles the confirm-token that a plain "
            "HTTP fetch can't). Install it with 'pip install gdown', or download the file in "
            "a browser and pass it via --verify."
        ) from exc
    print(f"[fetch_models] downloading Google Drive id={file_id} -> {dest}")
    timeout_s = float(os.environ.get("RALLY_MODEL_DOWNLOAD_TIMEOUT", "900"))
    if timeout_s <= 0:
        raise RuntimeError("RALLY_MODEL_DOWNLOAD_TIMEOUT must be positive")
    dest_path = os.path.abspath(dest)
    part_dir = os.path.dirname(dest_path) or "."
    existing_parts = set(Path(part_dir).glob("*.part"))
    proc = subprocess.Popen(_gdown_command(file_id, dest))
    started = time.monotonic()
    try:
        while proc.poll() is None:
            new_parts = set(Path(part_dir).glob("*.part")) - existing_parts
            written = (os.path.getsize(dest) if os.path.exists(dest) else 0)
            written += sum(p.stat().st_size for p in new_parts if p.exists())
            if written > _max_model_bytes():
                raise RuntimeError("checkpoint exceeds RALLY_MAX_MODEL_BYTES")
            if time.monotonic() - started > timeout_s:
                raise RuntimeError(f"checkpoint download exceeded {timeout_s:.0f}s")
            time.sleep(0.2)
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for part in set(Path(part_dir).glob("*.part")) - existing_parts:
            part.unlink(missing_ok=True)
        raise
    if proc.returncode or not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError("gdown did not produce a file (check the ID / sharing permissions)")
    _check_size(dest)


def _download(url: str, dest: str) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[fetch_models] downloading {url} -> {dest}")
    timeout_s = float(os.environ.get("RALLY_MODEL_DOWNLOAD_TIMEOUT", "900"))
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=min(60, timeout_s)) as r, open(dest, "wb") as fh:  # noqa: S310
        declared = r.headers.get("Content-Length")
        if declared and int(declared) > _max_model_bytes():
            raise RuntimeError("checkpoint exceeds RALLY_MAX_MODEL_BYTES")
        copied = 0
        while True:
            if time.monotonic() - started > timeout_s:
                raise RuntimeError(f"checkpoint download exceeded {timeout_s:.0f}s")
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > _max_model_bytes():
                raise RuntimeError("checkpoint exceeds RALLY_MAX_MODEL_BYTES")
            fh.write(chunk)
    _check_size(dest)


def _copy_bounded(source: str, dest: str) -> None:
    _check_size(source)
    with open(source, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _install_verified(source: str, dest: str, expected_digest: Optional[str], verifier) -> str:
    """Verify a sibling temporary and atomically publish it over the final checkpoint."""
    dest_dir = os.path.dirname(os.path.abspath(dest)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model.", suffix=".checkpoint", dir=dest_dir)
    os.close(fd)
    try:
        _copy_bounded(source, temporary)
        verifier(temporary)
        digest = _check_digest(temporary, expected_digest)
        with open(temporary, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(temporary, dest)
        return digest
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="rally.tools.fetch_models",
        description="Download and verify a pinned inference checkpoint.",
    )
    p.add_argument("--backend", choices=("court",), default="court")
    p.add_argument("--url", help="direct URL or Google Drive share URL")
    p.add_argument("--drive-id", help="Google Drive file ID (uses gdown)")
    p.add_argument("--verify", metavar="CHECKPOINT",
                   help="verify an existing checkpoint instead of downloading")
    p.add_argument("--sha256", help="required SHA-256 identity")
    p.add_argument("--dest", help="installation path; defaults to the backend's models path")
    args = p.parse_args(argv)
    if args.dest is None:
        args.dest = "models/court_keypoints_resnet50.pth"
    if args.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        p.error("--sha256 must be exactly 64 hexadecimal characters")

    if not args.url and not args.drive_id and not args.verify:
        p.print_help()
        print("\n[fetch_models] nothing to do: pass --drive-id, --url, or --verify.",
              file=sys.stderr)
        return 2

    try:
        drive_id = args.drive_id or (_drive_id_from_url(args.url) if args.url else None)
        expected_digest = args.sha256
        if args.verify:
            if os.path.abspath(args.verify) == os.path.abspath(args.dest):
                _check_size(args.verify)
                _verify_court(args.verify)
                digest = _check_digest(args.verify, expected_digest)
            else:
                digest = _install_verified(
                    args.verify, args.dest, expected_digest, verifier=_verify_court)
                print(f"[fetch_models] installed verified weights -> {args.dest}")
        else:
            dest_dir = os.path.dirname(os.path.abspath(args.dest)) or "."
            os.makedirs(dest_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=".model-download.", suffix=".checkpoint", dir=dest_dir)
            os.close(fd)
            try:
                if drive_id:
                    _download_drive(drive_id, temporary)
                else:
                    _download(args.url, temporary)
                _verify_court(temporary)
                digest = _check_digest(temporary, expected_digest)
                with open(temporary, "rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(temporary, args.dest)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        print(f"[fetch_models] failed: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch_models] ready - sha256={digest}; installed {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
