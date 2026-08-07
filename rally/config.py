"""Tunable configuration for the rally-trimming pipeline.

Every threshold that a user might reasonably want to change lives here so the rest
of the code stays free of magic numbers. Defaults are tuned for a fixed-tripod
single-camera recording of a singles match (the scenario in the design doc).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from numbers import Real

DEFAULT_YOLO_DETECTION_MODEL = "yolo12n.pt"
DEFAULT_COURT_MODEL = "court_keypoints_resnet50.pth"
DEFAULT_RTMPOSE_MODEL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)


def _model_from_env(name: str, default: str) -> str:
    """Resolve a non-empty model override when each config instance is created."""
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _optional_path_from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _pose_model_from_env() -> str:
    return _optional_path_from_env("RALLY_PLAYER_POSE_MODEL") or DEFAULT_RTMPOSE_MODEL


@dataclass
class RallyConfig:
    # ---- sampling -----------------------------------------------------------
    player_fps: float = 2.0  # frame rate for (expensive) person detection

    # ---- continuous player-pose timeline ------------------------------------
    # Audio is deliberately not an analysis input. Player boxes from the identity pass
    # are reused for one all-court pose timeline; likely arm actions receive a short,
    # higher-rate refinement pass. Point discovery must never depend on one receiver.
    pose_timeline_fps: float = 6.0
    pose_refine_fps: float = 12.0
    pose_refine_pre_s: float = 0.65
    pose_refine_post_s: float = 0.70
    pose_refine_motion_windows_per_minute: float = 18.0
    pose_track_box_max_gap_s: float = 0.45
    pose_serve_overhead_ratio: float = 0.25
    pose_serve_min_hand_speed_body_s: float = 0.65
    pose_serve_setup_span_m: float = 1.8
    pose_serve_min_score: float = 0.66
    pose_serve_unreturned_min_score: float = 0.76
    pose_serve_min_wrist_rise: float = 0.22
    pose_serve_min_follow_body: float = 0.18
    pose_serve_min_follow_horizontal_body: float = 0.15
    pose_serve_knee_bend_deg: float = 155.0
    pose_serve_leg_drive_deg: float = 12.0
    # A cut starts at the measured racket-arm load/toss preparation.  The bounded
    # fallback prevents a stationary baseline wait from becoming point footage.
    pose_serve_setup_max_gap_s: float = 0.80
    pose_serve_setup_fallback_s: float = 2.5
    # RTMPose has no racket keypoint, so wrist motion is an explicit racket-hand proxy.
    # Proposals are smoothed and collapsed into physical stroke episodes.
    pose_action_pre_s: float = 0.45
    pose_action_post_s: float = 0.50
    pose_action_smoothing_s: float = 0.18
    pose_action_min_speed_body_s: float = 0.85
    pose_action_min_backswing_body: float = 0.18
    pose_action_min_follow_body: float = 0.18
    pose_action_min_through_body: float = 0.38
    pose_action_min_confidence: float = 0.66
    pose_action_nms_s: float = 0.70
    pose_stroke_episode_gap_s: float = 0.95
    pose_compact_stroke_net_distance_m: float = 3.5
    pose_service_attempt_mask_s: float = 0.70
    pose_service_retry_max_gap_s: float = 20.0
    pose_service_retry_state_max_gap_s: float = 5.0
    pose_service_impossible_end_switch_s: float = 30.0
    pose_first_return_max_s: float = 4.5
    # This is an observation-association limit, not a tennis point duration rule. It may
    # span one missed pose contact, but strict near/far alternation is still required.
    pose_exchange_max_gap_s: float = 6.0
    # Point state is classified over overlapping, identity-preserving windows.  A
    # service sequence or time-ordered cross-court response starts LIVE; ready/reaction
    # activity can maintain it, but generic walking cannot start it.
    pose_live_window_s: float = 2.0
    pose_live_response_min_s: float = 0.15
    pose_live_response_max_s: float = 1.75
    pose_live_min_two_sided_fraction: float = 0.45
    pose_live_min_end_ready_fraction: float = 0.20
    pose_live_min_end_activity_fraction: float = 0.20
    pose_live_arm_activity_speed_body_s: float = 0.55
    pose_live_court_activity_speed_m_s: float = 0.55
    pose_live_min_relaxed_fraction: float = 0.60
    pose_live_unknown_bridge_s: float = 1.0
    pose_live_candidate_min_s: float = 5.5
    pose_exchange_start_lookback_s: float = 6.0
    pose_point_min_visible_players: int = 2
    # Do not interpret the receiver's travel time as a point ending.  Endpoint evidence
    # is only considered after a normal opponent-response interval has elapsed.
    pose_point_reset_delay_s: float = 1.00
    # Between-point players may walk or retrieve balls, so court speed is intentionally
    # not an upper bound.  The positive transition is sustained relaxed/non-hitting pose.
    pose_between_min_s: float = 1.00
    # In singles this requires both observed players to have left a ready stance; in
    # doubles it likewise prevents one still-ready player from declaring the point over.
    pose_between_max_ready_fraction: float = 0.30
    pose_between_max_median_wrist_speed_body_s: float = 1.75
    pose_live_search_s: float = 5.0
    pose_endpoint_max_unexplained_tail_s: float = 4.0
    pose_unreturned_transition_search_s: float = 5.0
    pose_exchange_min_actions: int = 3
    # Batch-level fail-closed guard. Near-continuous windows indicate that candidates
    # tiled the source rather than isolated tennis points; such a result must not be
    # published as a highlight video.
    pose_quality_max_retention_fraction: float = 0.90
    pose_quality_max_median_gap_s: float = 0.75
    pose_quality_max_zero_stroke_fraction: float = 0.65
    # YOLO12 detects/crops target-court players. RTMLib then performs top-down pose
    # estimation inside those boxes; this is substantially more useful for small far-side
    # servers than whole-frame pose inference.
    player_detection_model: str = field(
        default_factory=lambda: _model_from_env(
            "RALLY_YOLO_DETECTION_MODEL", DEFAULT_YOLO_DETECTION_MODEL
        )
    )
    player_pose_model: str = field(default_factory=_pose_model_from_env)
    rtmpose_runtime: str = field(
        default_factory=lambda: _model_from_env("RALLY_RTMPOSE_RUNTIME", "onnxruntime").lower()
    )
    rtmpose_device: str = field(
        default_factory=lambda: _model_from_env("RALLY_RTMPOSE_DEVICE", "auto").lower()
    )
    min_rally_s: float = 2.0  # drop rally segments shorter than this

    # ---- match posture context ---------------------------------------------
    match_ready_stance_ratio: float = 0.85
    match_ready_knee_deg: float = 150.0
    # ---- court geometry ------------------------------------------------------
    court_corners: tuple | None = None  # ((nlx,nly),(nrx,nry),(netrx,netry),(netlx,netly))
    serve_baseline_y_m: float = 1.5  # court_y below this = at/behind near baseline

    # ---- court localization --------------------------------------------------
    court_auto: bool = True  # auto-detect the court homography
    # Learned court localization is a required first pass. The independent multi-frame
    # line/homography detector remains a fallback, not a substitute for missing setup.
    court_weights: str = field(
        default_factory=lambda: _model_from_env("RALLY_COURT_WEIGHTS", DEFAULT_COURT_MODEL)
    )
    # ---- non-play exclusion -------------------------------------------------
    skip_intro_s: float = 0.0  # drop points starting before this time (manual warm-up skip)

    # ---- output presentation ------------------------------------------------
    label_points: bool = True  # draw "Point N" in the top-left of each rally
    label_prefix: str = "Point"
    point_start_buffer_s: float = 0.25  # small encoding-safe source context
    point_end_buffer_s: float = 0.25
    # Artificial black frames make point transitions feel broken. Keep this opt-in only;
    # point_end_buffer_s provides the transition context instead.
    inter_point_gap_s: float = 0.0

    # ---- cut ----------------------------------------------------------------
    reencode: bool = True  # True = frame-accurate re-encode; False = fast stream-copy

    def __post_init__(self) -> None:
        def finite_numbers(value) -> bool:
            if isinstance(value, bool) or value is None or isinstance(value, str):
                return True
            if isinstance(value, Real):
                return math.isfinite(float(value))
            if isinstance(value, (tuple, list)):
                return all(finite_numbers(item) for item in value)
            return True

        for name, value in vars(self).items():
            if not finite_numbers(value):
                raise ValueError(f"{name} must contain only finite numbers")
        for name in ("player_detection_model", "player_pose_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty model name or path")
        if self.rtmpose_runtime not in {"onnxruntime", "opencv"}:
            raise ValueError("rtmpose_runtime must be 'onnxruntime' or 'opencv'")
        if self.rtmpose_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("rtmpose_device must be 'auto', 'cpu', or 'cuda'")
        for name in (
            "player_fps",
            "pose_timeline_fps",
            "pose_refine_fps",
            "pose_live_window_s",
            "min_rally_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "pose_refine_pre_s",
            "pose_refine_post_s",
            "pose_track_box_max_gap_s",
            "pose_refine_motion_windows_per_minute",
            "pose_serve_setup_max_gap_s",
            "pose_serve_setup_fallback_s",
            "pose_action_pre_s",
            "pose_action_post_s",
            "pose_action_smoothing_s",
            "pose_action_nms_s",
            "pose_stroke_episode_gap_s",
            "pose_service_attempt_mask_s",
            "pose_service_retry_max_gap_s",
            "pose_service_retry_state_max_gap_s",
            "pose_service_impossible_end_switch_s",
            "pose_first_return_max_s",
            "pose_exchange_max_gap_s",
            "pose_live_response_min_s",
            "pose_live_response_max_s",
            "pose_live_unknown_bridge_s",
            "pose_live_candidate_min_s",
            "pose_exchange_start_lookback_s",
            "pose_point_reset_delay_s",
            "pose_between_min_s",
            "pose_between_max_median_wrist_speed_body_s",
            "pose_live_arm_activity_speed_body_s",
            "pose_live_court_activity_speed_m_s",
            "pose_live_search_s",
            "pose_unreturned_transition_search_s",
            "pose_endpoint_max_unexplained_tail_s",
            "pose_quality_max_median_gap_s",
            "point_start_buffer_s",
            "point_end_buffer_s",
            "inter_point_gap_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        if not 0 < self.match_ready_knee_deg <= 180:
            raise ValueError("match_ready_knee_deg must be in (0, 180]")
        if self.match_ready_stance_ratio < 0:
            raise ValueError("match_ready_stance_ratio must be non-negative")
        if not 0 < self.pose_serve_knee_bend_deg <= 180:
            raise ValueError("pose_serve_knee_bend_deg must be in (0, 180]")
        if (
            self.pose_serve_min_wrist_rise < 0
            or self.pose_serve_min_follow_body < 0
            or self.pose_serve_min_follow_horizontal_body < 0
            or self.pose_serve_leg_drive_deg < 0
        ):
            raise ValueError("temporal serve thresholds must be non-negative")
        if self.pose_serve_overhead_ratio < 0 or self.pose_serve_setup_span_m < 0:
            raise ValueError("serve geometry thresholds must be non-negative")
        if self.pose_action_min_speed_body_s <= 0:
            raise ValueError("pose_action_min_speed_body_s must be positive")
        for name in (
            "pose_action_min_backswing_body",
            "pose_action_min_follow_body",
            "pose_action_min_through_body",
            "pose_compact_stroke_net_distance_m",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.pose_action_min_confidence <= 1.0:
            raise ValueError("pose_action_min_confidence must be in [0, 1]")
        for name in (
            "pose_serve_min_score",
            "pose_serve_unreturned_min_score",
            "pose_between_max_ready_fraction",
            "pose_live_min_two_sided_fraction",
            "pose_live_min_end_ready_fraction",
            "pose_live_min_end_activity_fraction",
            "pose_live_min_relaxed_fraction",
            "pose_quality_max_retention_fraction",
            "pose_quality_max_zero_stroke_fraction",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.pose_exchange_min_actions < 2:
            raise ValueError("pose_exchange_min_actions must be at least 2")
        if self.pose_live_response_max_s <= self.pose_live_response_min_s:
            raise ValueError("pose_live_response_max_s must exceed pose_live_response_min_s")
        if self.pose_point_min_visible_players < 1:
            raise ValueError("pose evidence counts must be positive")
        for name in ("point_start_buffer_s", "point_end_buffer_s"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be <= 1 second")
        if self.court_corners is not None:
            cc = self.court_corners
            if len(cc) != 4 or any(len(p) != 2 for p in cc):
                raise ValueError("court_corners must be 4 (x, y) image points")
