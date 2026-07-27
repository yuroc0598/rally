"""Tunable configuration for the rally-trimming pipeline.

Every threshold that a user might reasonably want to change lives here so the rest
of the code stays free of magic numbers. Defaults are tuned for a fixed-tripod
single-camera recording of a singles match (the scenario in the design doc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RallyConfig:
    # ---- sampling -----------------------------------------------------------
    analysis_fps: float = 5.0          # frame rate for motion / geometry analysis
    player_fps: float = 2.0            # frame rate for (expensive) person detection
    proxy_height: int = 360            # downscale analysis frames to this height

    # ---- audio: ball-strike (racket impact) detection -----------------------
    audio_sr: int = 22050
    strike_band_hz: tuple[float, float] = (1500.0, 6000.0)  # impact transient band
    strike_min_gap_s: float = 0.30     # refractory gap between two strikes
    strike_sensitivity: float = 1.6    # peak height = local median + k * local std of envelope
    strike_snr_ratio: float = 6.0      # AND peak must exceed this * local median envelope (SNR gate)
    audio_block_s: float = 4.0         # window for the LOCAL adaptive strike threshold
    #   (adaptive thresholding tracks a changing noise floor — crowd/applause vs. quiet —
    #    so loud sections don't flood false strikes and quiet sections don't miss soft hits)
    rhythm_window_s: float = 3.0       # trailing window used to score strike rhythm
    strikes_full_score: float = 3.0    # #strikes in window that saturates audio score

    # ---- co-decision scoring weights (renormalised over channels available) --
    # audio strikes + ball-in-play are *discriminative* for "is a rally happening" (they
    # only occur during play) -> weighted high. Player-geometry and motion are supporting
    # cues (two players are on court between points too), so they vote but don't dominate.
    # Each source votes weighted by its per-frame *confidence* (see rally_probability),
    # so a source only sways the decision when it's sure. Discriminative sources (audio,
    # ball, pose-activity) carry more weight; static geometry is a lighter supporting vote
    # (two players are on court between points too, so it's confidently non-discriminative).
    w_audio: float = 0.6
    w_ball: float = 0.5         # ball-in-play (needs ball_channel + weights)
    w_pose: float = 0.3         # player pose-activity (needs player_pose; self-gates by confidence)
    w_geometry: float = 0.15    # player presence (supporting)
    w_motion: float = 0.1
    ball_channel: bool = False  # run ball tracking over the whole video as a fusion vote
    player_pose: bool = False   # run pose over the video as a confidence-weighted vote (slow)
    player_pose_fps: float = 2.0

    # ---- motion / camera ----------------------------------------------------
    motion_full_score: float = 0.04    # frame-diff energy (0..1) that saturates motion score
    camera_motion_px: float = 2.5      # global shift (px, on proxy) flagged as camera move

    # ---- decode -------------------------------------------------------------
    smooth_window_s: float = 1.0
    enter_threshold: float = 0.55      # hysteresis: enter RALLY above this
    exit_threshold: float = 0.35       # hysteresis: leave RALLY below this
    min_rally_s: float = 2.0           # drop rally segments shorter than this
    merge_gap_s: float = 1.5           # merge rally segments closer than this
    pad_pre_s: float = 1.0             # lead-in padding added to each kept segment
    pad_post_s: float = 1.5            # lead-out padding added to each kept segment

    # ---- serve capture ------------------------------------------------------
    # Rally detection triggers only once the strike rhythm has built up, so the
    # decoded start lags the serve (the point's first strike). Snap the start back
    # to that first strike and add a short pre-roll for the toss/serve motion.
    snap_serve: bool = True
    serve_lookback_s: float = 6.0      # search this far before the decoded start for the serve
    serve_preroll_s: float = 1.8       # include this much before the serve strike

    # ---- point splitting / strike-bounding ----------------------------------
    # Detection is recall-oriented and its padded region edges include inter-point
    # walking. Re-bound each point tightly to its ball strikes: cluster strikes within
    # a region (split at gaps > point_gap_s) and keep only [first-toss .. last+tail].
    point_split: bool = True
    point_gap_s: float = 2.5           # strike-to-strike silence that marks a candidate boundary
    toss_preroll_s: float = 1.0        # lead-in kept before the serve strike (the toss)
    landing_tail_s: float = 1.2        # tail kept after the last strike (ball lands / point ends)

    # ---- rally coherence (reject false points) ------------------------------
    # A real rally is several strikes with an even cadence over a real span. Stray
    # sounds, echoes, and ball-bounce doublets are not. Count "effective" strikes
    # (echo/bounce transients within echo_collapse_s folded into one) and require a
    # minimum count and duration.
    echo_collapse_s: float = 0.35      # transients closer than this = one event (bounce/echo)
    min_rally_strikes: int = 2         # effective strikes required for a real rally
    min_rally_dur_s: float = 1.0       # min span of the strike cluster

    # ---- serve attach (don't cut the serve) ---------------------------------
    # The serve is often a lone strike separated from the rally by more than point_gap_s
    # (serve -> ball travels -> return), so rhythm clustering splits it off and the point
    # would start mid-rally. Re-anchor the point's start to that preceding isolated serve
    # strike (within serve_attach_window_s, itself preceded by silence).
    serve_attach: bool = True
    serve_attach_window_s: float = 4.0

    # ---- court-geometry serve detection (opt-in; needs calibration + YOLO) ---
    # One-time fixed-camera calibration: 4 image points (px) = near-left & near-right
    # baseline corners, then net∩right-sideline & net∩left-sideline. With it, the near
    # player is tracked in court metres and each point's start is anchored to the serve
    # set-up (near player set at/behind the baseline, still, just before the rally) —
    # capturing the serve even for far-side serves (the near receiver sets up in sync).
    court_corners: Optional[tuple] = None   # ((nlx,nly),(nrx,nry),(netrx,netry),(netlx,netly))
    serve_track_fps: float = 5.0
    serve_setup_lookback_s: float = 6.0
    serve_baseline_y_m: float = 1.5         # court_y below this = at/behind near baseline
    serve_still_speed_mps: float = 0.6
    serve_setup_preroll_s: float = 0.8
    serve_max_lead_s: float = 2.5           # cap serve lead-in (avoids pre-serve loiter)

    # ---- ball point-end (opt-in; needs TrackNet weights + calibration + ideally GPU) --
    # Track the ball over each rally and trim its end to the point-ending event (double
    # bounce on one side / ball lands out). CPU-slow (~0.3s/frame) — runs only over kept
    # rally segments, not the whole video.
    ball_weights: Optional[str] = None      # path to a 3-frame TrackNet .pt
    ball_tail_s: float = 0.8                 # kept after the point-ending bounce
    ball_max_extend_s: float = 3.0           # how far past the current end to look

    # ---- non-play exclusion -------------------------------------------------
    drop_isolated: bool = True         # drop points with no neighbouring point nearby
    isolation_gap_s: float = 120.0     # "nearby" window; a lone point beyond this is likely non-match
    skip_intro_s: float = 0.0          # drop points starting before this time (manual warm-up skip)

    # Movement gate against over-splitting: a candidate boundary is only a *real* point
    # break if the players actually reset. A short gap with little near-player movement is
    # a mid-rally lull (e.g. a lob), not a new point, so merge across it. (Requires the
    # player-detection channel; falls back to gap-only splitting without it.)
    movement_merge: bool = True
    merge_max_gap_s: float = 4.0       # only movement-merge gaps shorter than this
    move_thresh: float = 0.15          # near-player displacement (frac of frame) that means "reset"

    # ---- output presentation ------------------------------------------------
    label_points: bool = True          # draw "Point N" in the top-left of each rally
    label_prefix: str = "Point"
    inter_point_gap_s: float = 0.4     # brief black delay inserted between points

    # duration-aware segment-model decoder (part (a) of the design)
    use_dp_decoder: bool = True
    rally_dur_prior_s: tuple[float, float] = (7.0, 6.0)   # (mean, std) of rally length
    gap_dur_prior_s: tuple[float, float] = (18.0, 15.0)   # (mean, std) of dead-time length
    # Cap on any single decoded segment. Must exceed the longest expected GAP (segments
    # alternate RALLY/GAP, so a gap longer than this is force-split, injecting a fake
    # rally): gap prior is (mean 18, std 15), so ~90s covers a 3-sigma changeover.
    max_segment_s: float = 90.0
    transition_penalty: float = 0.4    # log-penalty per segment boundary (discourages churn)
    duration_prior_weight: float = 0.15  # how strongly duration priors pull the decode

    # ---- cut ----------------------------------------------------------------
    reencode: bool = True              # True = frame-accurate re-encode; False = fast stream-copy

    def __post_init__(self) -> None:
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        for name in ("analysis_fps", "player_fps", "audio_sr", "player_pose_fps",
                     "serve_track_fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.proxy_height <= 0:
            raise ValueError("proxy_height must be positive")
        for name in ("enter_threshold", "exit_threshold", "move_thresh"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("w_audio", "w_ball", "w_pose", "w_geometry", "w_motion"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("pad_pre_s", "pad_post_s", "min_rally_s", "max_segment_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_rally_s > self.max_segment_s:
            raise ValueError("min_rally_s must be <= max_segment_s")
        if self.court_corners is not None:
            cc = self.court_corners
            if len(cc) != 4 or any(len(p) != 2 for p in cc):
                raise ValueError("court_corners must be 4 (x, y) image points")
