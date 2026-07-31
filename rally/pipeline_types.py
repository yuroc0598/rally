"""Typed state exchanged by pipeline stages.

The orchestrator still passes one object for low-overhead in-process analysis, but signal
evidence and arbiter/refinement state are declared separately so their ownership is clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

Segment = tuple[float, float]


@dataclass
class RallyResult:
    input_path: str
    output_path: Optional[str]
    segments: list[Segment]
    total_seconds: float
    kept_seconds: float
    compression_ratio: float
    channels_used: list[str] = field(default_factory=list)
    n_strikes: int = 0
    strike_times: list[float] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def sidecar(self) -> dict:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "total_seconds": round(self.total_seconds, 3),
            "kept_seconds": round(self.kept_seconds, 3),
            "compression_ratio": round(self.compression_ratio, 4),
            "n_rallies": len(self.segments),
            "n_strikes": self.n_strikes,
            "strike_times": [round(float(t), 3) for t in self.strike_times],
            "channels_used": self.channels_used,
            "stages": self.stages,
            "config": self.config,
            "segments": [
                {"index": index, "start": round(start, 3), "end": round(end, 3),
                 "duration": round(end - start, 3)}
                for index, (start, end) in enumerate(self.segments)
            ],
        }


@dataclass
class SignalEvidence:
    """Raw/aligned signal channels and shared detector outputs."""

    audio_rate: Optional[np.ndarray] = None
    audio_reg: Optional[np.ndarray] = None
    geometry: Optional[np.ndarray] = None
    motion: Optional[np.ndarray] = None
    camera_moving: Optional[np.ndarray] = None
    near_track: Optional[tuple[np.ndarray, np.ndarray]] = None
    ball: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    pose_conf: Optional[np.ndarray] = None
    onsets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    n_strikes: int = 0
    used: list[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    detector: Optional[Any] = None
    player_samples: list = field(default_factory=list)
    court: Optional[Any] = None
    frame_size: Optional[tuple[int, int]] = None
    player_serve_hints: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    player_proposal_hints: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))


@dataclass
class ArbiterEvidence:
    """Proposal ownership, verdict regions, and expensive reusable tracks."""

    arbiter_audio_fallback: list[Segment] = field(default_factory=list)
    arbiter_selected_audio_fallback: list[Segment] = field(default_factory=list)
    arbiter_accepted_regions: list[Segment] = field(default_factory=list)
    arbiter_indeterminate_regions: list[Segment] = field(default_factory=list)
    arbiter_fragmented_regions: list[Segment] = field(default_factory=list)
    arbiter_rejected_regions: list[Segment] = field(default_factory=list)
    ball_end_hints: list[tuple[Segment, float]] = field(default_factory=list)
    ball_track_cache: list = field(default_factory=list)
    player_recovered_points: list[Segment] = field(default_factory=list)


@dataclass
class PipelineState(SignalEvidence, ArbiterEvidence):
    """Combined state object retained for the current sequential orchestrator."""

