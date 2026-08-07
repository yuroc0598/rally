"""Shared inference-device selection.

Keep hardware policy outside individual detectors so every PyTorch model follows the same
explicit CUDA/MPS/CPU rules.
"""

from __future__ import annotations

import os


def resolve_torch_device(torch_module=None):
    """Return the requested usable PyTorch device, preferring CUDA then Apple MPS."""
    if torch_module is None:
        import torch as torch_module

    requested = os.environ.get("RALLY_DEVICE", "").strip().lower()
    if requested:
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            print(
                f"[inference] RALLY_DEVICE={requested} requested but CUDA is unavailable "
                "- using CPU"
            )
            return torch_module.device("cpu")
        if requested.startswith("mps") and not _mps_available(torch_module):
            print(
                f"[inference] RALLY_DEVICE={requested} requested but MPS is unavailable "
                "- using CPU"
            )
            return torch_module.device("cpu")
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if _mps_available(torch_module):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _mps_available(torch_module) -> bool:
    try:
        return bool(
            torch_module.backends.mps.is_built()
            and torch_module.backends.mps.is_available()
        )
    except (AttributeError, RuntimeError):
        return False
