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
import os
import re
import shutil
import sys
from typing import Optional


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
    """Download a Google Drive file, handling the large-file confirm-token via gdown."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive download needs gdown (it handles the confirm-token that a plain "
            "HTTP fetch can't). Install it with 'pip install gdown', or download the file in "
            "a browser and pass it via --verify."
        ) from exc
    print(f"[fetch_models] downloading Google Drive id={file_id} -> {dest}")
    out = gdown.download(id=file_id, output=dest, quiet=False)
    if not out or not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError("gdown did not produce a file (check the ID / sharing permissions)")


def _download(url: str, dest: str) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[fetch_models] downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as fh:  # noqa: S310 (user-supplied URL)
        shutil.copyfileobj(r, fh)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="rally.tools.fetch_models",
        description="Download/verify pretrained TrackNet weights for ball-arbiter mode.",
    )
    p.add_argument("--url", help="direct URL, or a Google Drive share URL, to a TrackNet .pt")
    p.add_argument("--drive-id", help="Google Drive file ID of the .pt (uses gdown)")
    p.add_argument("--verify", metavar="PT", help="verify an already-downloaded .pt instead of downloading")
    p.add_argument("--dest", default="models/tracknet.pt",
                   help="where to save/copy the weights (default: models/tracknet.pt, auto-discovered)")
    args = p.parse_args(argv)

    if not args.url and not args.drive_id and not args.verify:
        p.print_help()
        print("\n[fetch_models] nothing to do: pass --drive-id, --url, or --verify.\n"
              "  Known TrackNet weights (yastrebksv/TrackNet, unlicensed — personal use):\n"
              "    python -m rally.tools.fetch_models --drive-id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl",
              file=sys.stderr)
        return 2

    try:
        if args.verify:
            _verify(args.verify)
            if os.path.abspath(args.verify) != os.path.abspath(args.dest):
                os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
                shutil.copyfile(args.verify, args.dest)
                print(f"[fetch_models] copied verified weights -> {args.dest}")
        else:
            drive_id = args.drive_id or (_drive_id_from_url(args.url) if args.url else None)
            if drive_id:
                _download_drive(drive_id, args.dest)
            else:
                _download(args.url, args.dest)
            _verify(args.dest)
    except Exception as exc:
        print(f"[fetch_models] failed: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch_models] ready — ball-arbiter will auto-discover {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
