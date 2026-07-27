"""rally — trim a tennis match recording down to rally (live-play) segments only.

Phase-1 pipeline (see README): audio ball-strike detection + player geometry +
motion  ->  per-frame rally probability  ->  duration-aware segment-model decode
->  ffmpeg cut. Heavy/optional dependencies (OpenCV, YOLO) degrade gracefully.
"""

from .config import RallyConfig

__all__ = ["RallyConfig"]
__version__ = "0.1.0"
