"""Model-independent observations consumed by tennis point decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Segment = tuple[float, float]


@dataclass(frozen=True)
class ServeSetupObservation:
    """Pose, formation, ball, court, and optional learned evidence for one candidate."""

    point: Segment
    first_strike: float
    side: Optional[str]
    side_confidence: float
    near_x: Optional[float]
    near_x_std: Optional[float]
    sampled_frames: int
    pose_frames: int
    ready_frames: int
    serve_motion: bool
    setup_evidence: bool
    observable: bool
    overhead_frames: int = 0
    overhead_max_ratio: float = 0.0
    overhead_strikes: tuple[float, ...] = ()
    position_checked: bool = False
    position_setup_evidence: bool = False
    position_best_strike: Optional[float] = None
    position_setup_strikes: tuple[float, ...] = ()
    position_score: float = 0.0
    position_server_end: Optional[str] = None
    position_server_span: Optional[float] = None
    position_player_tracks: int = 0
    position_stable_tracks: int = 0
    position_stable_fraction: float = 0.0
    ball_checked: bool = False
    ball_serve_evidence: bool = False
    ball_best_strike: Optional[float] = None
    ball_coverage: float = 0.0
    ball_vertical_span: float = 0.0
    ball_outgoing_span: float = 0.0
    ball_ordered_evidence: bool = False
    ball_measured_samples: int = 0
    target_court_filtered: bool = False
    receiver_reaction_evidence: bool = False
    receiver_reaction_time: Optional[float] = None
    learned_serve_checked: bool = False
    learned_serve_evidence: bool = False
    learned_serve_score: Optional[float] = None

@dataclass(frozen=True)
class PositionSetupObservation:
    """Player-formation evidence immediately before an early impact."""

    point: Segment
    best_strike: Optional[float]
    setup_strikes: tuple[float, ...]
    checked: bool
    setup_evidence: bool
    score: float
    server_end: Optional[str]
    server_span: Optional[float]
    player_tracks: int
    stable_tracks: int
    stable_fraction: float
    sampled_frames: int
