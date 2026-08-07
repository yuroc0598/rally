"""Typed state exchanged by the sequential vision pipeline."""

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
    n_serves: int = 0
    serve_times: list[float] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)
    points: list[dict] = field(default_factory=list)

    def sidecar(self) -> dict:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "total_seconds": round(self.total_seconds, 3),
            "kept_seconds": round(self.kept_seconds, 3),
            "compression_ratio": round(self.compression_ratio, 4),
            "n_rallies": len(self.segments),
            "n_serves": self.n_serves,
            "serve_times": [round(float(t), 3) for t in self.serve_times],
            "channels_used": self.channels_used,
            "stages": self.stages,
            "timings_seconds": {
                name: round(float(seconds), 3)
                for name, seconds in self.timings.items()
            },
            "config": self.config,
            "analysis_schema_version": "rally.pose_timeline_points.v3",
            "match": self.match,
            "points": self.points,
            "segments": [
                {"index": index, "start": round(start, 3), "end": round(end, 3),
                 "duration": round(end - start, 3)}
                for index, (start, end) in enumerate(self.segments)
            ],
        }


@dataclass
class SignalEvidence:
    """Raw/aligned signal channels and shared detector outputs."""

    serve_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    used: list[str] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    player_track_samples: list = field(default_factory=list)
    court: Optional[Any] = None
    frame_size: Optional[tuple[int, int]] = None
    match_format: str = "unknown"
    match_format_evidence: dict = field(default_factory=dict)
    match_profile: dict = field(default_factory=dict)
@dataclass
class PipelineState(SignalEvidence):
    """State retained by the sequential pose-first orchestrator."""
