"""Fetch / verify the pretrained TrackNet ball-tracking weights used by ball-arbiter mode.

Ball-primary detection (on by default; ``--no-ball-arbiter`` disables it) needs a 3-frame
TrackNet checkpoint compatible
with :class:`rally.vendor.tracknet_torch.BallTrackerNet`. Weights are NOT bundled with the
repo. This helper downloads a checkpoint from a URL you provide and verifies it loads into
the architecture, saving it where the pipeline auto-discovers it (``models/tracknet.pt``).

The vendored architecture matches the ``yastrebksv/TrackNet`` port (3 stacked frames ->
256-channel heatmap). Its weights are on Google Drive; usage examples::

    # Google Drive (handles the large-file confirm-token via gdown):
    python -m rally.tools.fetch_models --drive-id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl
    python -m rally.tools.fetch_models --url https://drive.google.com/file/d/<ID>/view

    # a plain direct URL:
    python -m rally.tools.fetch_models --url <DIRECT_URL_TO_PT>

    # verify (and install) a file you already downloaded:
    python -m rally.tools.fetch_models --verify /path/to/model.pt

Verification instantiates BallTrackerNet and calls ``load_state_dict`` (with
``weights_only=True`` — the checkpoint must be a plain tensor state-dict), so a mismatched
or tampered file fails loudly instead of silently producing garbage tracks.

LICENSE NOTE: the ``yastrebksv/TrackNet`` repo has no LICENSE file ("unofficial
implementation"), so the weights are not confirmed free-to-use — fine for personal/research
experimentation, but get clearance before redistributing or shipping them in a product.
"""

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

KNOWN_DRIVE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
KNOWN_SHA256 = "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076"


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


def _verify(path: str) -> None:
    """Load ``path`` into BallTrackerNet to confirm architecture compatibility."""
    import torch

    from ..vendor.tracknet_torch import BallTrackerNet

    sd = torch.load(path, map_location="cpu", weights_only=True)
    state = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    model = BallTrackerNet()
    model.load_state_dict(state)   # raises on key/shape mismatch
    print(f"[fetch_models] OK: {path} loads into BallTrackerNet")


def _drive_id_from_url(url: str) -> Optional[str]:
    """Extract a Google Drive file ID from common share-URL shapes, else None."""
    if "drive.google.com" not in url:
        return None
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", url) or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", url)
    return m.group(1) if m else None


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
    proc = subprocess.Popen([sys.executable, "-m", "gdown", "--id", file_id, "-O", dest])
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


def _install_verified(source: str, dest: str, expected_digest: Optional[str] = None) -> str:
    """Verify a sibling temporary and atomically publish it over the final checkpoint."""
    dest_dir = os.path.dirname(os.path.abspath(dest)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tracknet.", suffix=".pt", dir=dest_dir)
    os.close(fd)
    try:
        _copy_bounded(source, temporary)
        _verify(temporary)
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
        description="Download/verify pretrained TrackNet weights for ball-arbiter mode.",
    )
    p.add_argument("--url", help="direct URL, or a Google Drive share URL, to a TrackNet .pt")
    p.add_argument("--drive-id", help="Google Drive file ID of the .pt (uses gdown)")
    p.add_argument("--verify", metavar="PT", help="verify an already-downloaded .pt instead of downloading")
    p.add_argument("--sha256", help="required SHA-256 identity (the known Drive model is pinned automatically)")
    p.add_argument("--dest", default="models/tracknet.pt",
                   help="where to save/copy the weights (default: models/tracknet.pt, auto-discovered)")
    args = p.parse_args(argv)
    if args.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        p.error("--sha256 must be exactly 64 hexadecimal characters")

    if not args.url and not args.drive_id and not args.verify:
        p.print_help()
        print("\n[fetch_models] nothing to do: pass --drive-id, --url, or --verify.\n"
              "  Known TrackNet weights (yastrebksv/TrackNet, unlicensed — personal use):\n"
              "    python -m rally.tools.fetch_models --drive-id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl",
              file=sys.stderr)
        return 2

    try:
        drive_id = args.drive_id or (_drive_id_from_url(args.url) if args.url else None)
        expected_digest = args.sha256 or (KNOWN_SHA256 if drive_id == KNOWN_DRIVE_ID else None)
        if args.verify:
            if os.path.abspath(args.verify) == os.path.abspath(args.dest):
                _check_size(args.verify)
                _verify(args.verify)
                digest = _check_digest(args.verify, expected_digest)
            else:
                digest = _install_verified(args.verify, args.dest, expected_digest)
                print(f"[fetch_models] installed verified weights -> {args.dest}")
        else:
            dest_dir = os.path.dirname(os.path.abspath(args.dest)) or "."
            os.makedirs(dest_dir, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=".tracknet-download.", suffix=".pt", dir=dest_dir)
            os.close(fd)
            try:
                if drive_id:
                    _download_drive(drive_id, temporary)
                else:
                    _download(args.url, temporary)
                _verify(temporary)
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
    print(f"[fetch_models] ready — sha256={digest}; ball-arbiter will auto-discover {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
