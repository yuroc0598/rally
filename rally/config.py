"""Tunable configuration for the rally-trimming pipeline.

Every threshold that a user might reasonably want to change lives here so the rest
of the code stays free of magic numbers. Defaults are tuned for a fixed-tripod
single-camera recording of a singles match (the scenario in the design doc).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional


@dataclass
class RallyConfig:
    # Unexpected failures in an enabled stage are fatal by default. Set this only when a
    # caller explicitly prefers a lower-quality partial result over a failed run.
    allow_degraded: bool = False
    # Accuracy-first is the default product behaviour.  It keeps every cheap proposal and
    # every short audio hypothesis until the tennis-sequence decoder has seen it.  Set this
    # false only for an explicitly throughput-oriented batch job that accepts lower recall.
    accuracy_mode: bool = True

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
    # Speech consonants can clear the amplitude/SNR gates. A plausible racket contact must
    # additionally be either sharply impulsive (high crest factor) or broadband/noise-like
    # (high spectral flatness) in a short local window.
    strike_min_crest_factor: float = 2.7
    strike_min_spectral_flatness: float = 0.40
    audio_block_s: float = 4.0         # window for the LOCAL adaptive strike threshold
    #   (adaptive thresholding tracks a changing noise floor — crowd/applause vs. quiet —
    #    so loud sections don't flood false strikes and quiet sections don't miss soft hits)
    rhythm_window_s: float = 3.0       # trailing window used to score strike rhythm
    strikes_full_score: float = 3.0    # #strikes in window that saturates audio score

    # ---- co-decision scoring weights (absolute evidence strengths) -----------
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
    landing_tail_s: float = 1.0        # real footage kept after the last strike / point-end cue

    # ---- rally coherence (reject false points) ------------------------------
    # A real rally is several strikes with an even cadence over a real span. Stray
    # sounds, echoes, and ball-bounce doublets are not. Count "effective" strikes
    # (echo/bounce transients within echo_collapse_s folded into one) and require a
    # minimum count and duration.
    echo_collapse_s: float = 0.35      # transients closer than this = one event (bounce/echo)
    min_rally_strikes: int = 2         # effective strikes required for a real rally
    min_rally_dur_s: float = 1.0       # min span of the strike cluster

    # ---- audio-only serve attachment (unsafe heuristic; opt-in) --------------
    # An isolated transient before a coherent cluster may be a serve, but audio alone
    # cannot distinguish it from a bounce, shoe, speech consonant, or the prior point.
    # Keep this disabled unless a caller explicitly accepts that precision trade-off.
    # The output renderer independently keeps one second of real setup footage.
    serve_attach: bool = False
    serve_attach_window_s: float = 4.0

    # ---- match-state validation --------------------------------------------
    # ``auto`` only applies tennis sequence rules inside a run bounded by multiple
    # ball-confirmed serves, so leading/trailing warm-up remains unconstrained. ``match``
    # requires serve evidence for every point; ``casual`` disables the check. Screen-side
    # and receiver-ready observations are diagnostics, never proof that a serve occurred.
    play_mode: str = "auto"             # auto | match | casual
    match_setup_fps: float = 5.0
    match_setup_lookback_s: float = 1.8
    match_setup_end_before_strike_s: float = 0.2
    match_pose_imgsz: int = 1280
    match_side_center_margin: float = 0.08
    match_side_max_std: float = 0.05
    match_ready_stance_ratio: float = 0.85
    match_ready_knee_deg: float = 150.0
    match_min_ready_frames: int = 4
    # Player-position setup, measured from detections already collected by the visual
    # pass. Coordinates use court space when calibration succeeds and conservative
    # image-space baseline bands otherwise.
    match_position_post_strike_s: float = 0.15
    match_position_min_frames: int = 3
    match_position_min_players: int = 2
    match_position_track_step: float = 0.18
    match_position_max_span: float = 0.045
    match_position_min_stable_fraction: float = 0.75
    match_position_min_score: float = 0.25
    match_position_court_baseline_depth: float = 0.08
    match_position_far_baseline_y: tuple[float, float] = (0.36, 0.56)
    match_position_near_baseline_y: tuple[float, float] = (0.72, 0.98)
    # Serve confirmation is an event, not a static pose: a near-side overhead motion or
    # sustained in-court TrackNet motion with meaningful vertical travel around an early
    # impact. Receiver posture and deuce/ad side are supporting diagnostics only.
    match_overhead_wrist_ratio: float = 0.35
    match_overhead_window_s: float = 0.45
    match_serve_strikes_to_check: int = 3
    match_ball_serve_pre_s: float = 1.2
    match_ball_serve_post_s: float = 0.6
    match_ball_min_coverage: float = 0.18
    # Ball-only confirmation is deliberately stricter for a one-impact hypothesis. A
    # short pickup/pass can barely clear the general TrackNet thresholds; without pose or
    # stationary-baseline corroboration it needs substantially more measured coverage.
    match_ball_min_single_strike_coverage: float = 0.35
    match_ball_min_vertical_span: float = 0.035
    match_ball_min_outgoing_span: float = 0.012
    match_ball_court_x: tuple[float, float] = (0.12, 0.88)
    match_ball_court_y: tuple[float, float] = (0.25, 0.85)
    match_auto_min_serve_anchors: int = 3
    match_phase_max_gap_s: float = 100.0
    # Candidate fragments on the same service side can be a fault/retry or a rally whose
    # contacts were missed.  Merge them before applying match validity and recover the
    # visible serve-preparation lead-in from the selected serve contact.
    match_fragment_merge_gap_s: float = 6.5
    match_attempt_merge_min_gap_s: float = 8.0
    match_attempt_merge_gap_s: float = 15.0
    match_point_start_preroll_s: float = 4.0
    # A quiet far-side serve can be absent from the audio channel. The near receiver is
    # normally still through the service motion and starts moving shortly after contact,
    # so a stable-to-active transition opens a recall-only TrackNet proposal. It is never
    # sufficient by itself; match validation still requires serve-ball/setup evidence.
    match_player_activity_speed: float = 0.018
    match_player_activity_min_s: float = 0.8
    match_player_stable_s: float = 1.5
    match_player_stable_span: float = 0.020
    match_receiver_reaction_lag_s: float = 0.5
    match_player_hint_near_y_min: float = 0.55

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
    ball_inference_batch_size: int = 4      # batch CNN forwards; trajectory decode stays ordered
    ball_tail_s: float = 1.0                 # real footage kept after the point-ending bounce
    ball_max_extend_s: float = 3.0           # how far past the current end to look
    # A local TrackNet component can end at an occlusion in the middle of a long rally.
    # End hints may tighten modest audio overrun, but must not erase a large portion of a
    # point merely because the later trajectory is fragmented.
    ball_end_hint_max_uncalibrated_trim_s: float = 5.0

    # ---- ball-as-arbiter (SwingVision-style; needs TrackNet weights; court optional) --
    # The cheap channels only *propose* candidate windows; the ball trajectory then decides
    # which are real rallies and sets each one's serve start / point-end. On CPU we track
    # only inside candidates (padded), so this stays affordable. Court homography (manual
    # court_corners or court_auto) unlocks net-crossing / in-out geometry; without it the
    # verdict leans on in-play span + bounce count alone.
    # On by default because trajectory evidence is more discriminative than cheap cues,
    # though accuracy still requires independent real-video evaluation. If PyTorch or the
    # TrackNet weights aren't present,
    # the pipeline logs a note and falls back to the audio-primary detector automatically,
    # so the user never has to think about it. Disable only to force the fast/audio path.
    ball_arbiter: bool = True
    # On by default: when the ball arbiter runs, auto-detecting the court is cheap next to
    # ball tracking and can add net-crossing / in-out evidence; it falls back to
    # no-court if it can't find one confidently. Manual court_corners, if given, take
    # precedence. Disable only if the detector locks onto the wrong lines.
    court_auto: bool = True                    # auto-detect the court homography
    court_weights: Optional[str] = None       # reserved: future court-keypoint model (not yet implemented; classical is used)
    arbiter_pre_pad_s: float = 2.0            # track this far before each candidate (catch the serve)
    arbiter_post_pad_s: float = 2.0           # track this far after (catch the point end)
    arbiter_min_speed_px_s: float = 25.0      # image speed above which the ball counts as "live"
    arbiter_min_conf: float = 0.3             # trajectory confidence required to trust a sample
    arbiter_min_in_play_frac: float = 0.35    # fraction of the window the ball must be live
    arbiter_min_in_play_span_s: float = 1.5   # live-ball span required for a real rally
    arbiter_min_bounces: int = 2              # bounces that stand in for a net crossing (no court)
    # TrackNet commonly loses the ball briefly behind a player or at the net.  Components
    # separated by at most this interval are evaluated jointly instead of forcing a global
    # indeterminate verdict merely because measured samples are not frame-contiguous.
    arbiter_fragment_join_gap_s: float = 0.80
    # Recovery of an audio-missed point is allowed only when a player-derived serve hint
    # leads into a fragmented but structurally credible ball component.
    arbiter_player_recovery_min_coverage: float = 0.55
    # Bound expensive TrackNet work when cheap evidence saturates on camera shake/crowds.
    # Motion only opens proposals below the active-fraction guard; remaining proposals are
    # ranked and capped in both individual and aggregate duration.
    arbiter_motion_max_active_frac: float = 0.35
    arbiter_max_candidate_s: float = 45.0
    # Keep short clips complete and avoid an abrupt 80% discard at every duration: the
    # total tracking budget is at least this many seconds (clipped to video duration).
    arbiter_min_total_s: float = 120.0
    arbiter_max_total_fraction: float = 0.20
    arbiter_max_total_s: float = 600.0
    # Small diversity bonus used after audio coherence/strike support when choosing which
    # over-budget windows receive TrackNet. Prevents equal/noisy evidence selecting only
    # the beginning of a long match.
    arbiter_diversity_weight: float = 0.12
    # Optional precision mode: require both an audio strike and a ball origin near a
    # baseline. It can reject some warm-up, but is not a learned serve classifier and can
    # miss points with bad audio/court calibration.
    arbiter_require_serve_evidence: bool = False

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
    point_start_buffer_s: float = 1.0  # real source footage before each detected point start
    point_end_buffer_s: float = 1.0    # real source footage after each detected point end
    # Artificial black frames make point transitions feel broken. Keep this opt-in only;
    # point_end_buffer_s provides the transition context instead.
    inter_point_gap_s: float = 0.0

    # duration-aware segment-model decoder (part (a) of the design)
    use_dp_decoder: bool = True
    rally_dur_prior_s: tuple[float, float] = (7.0, 6.0)   # (mean, std) of rally length
    gap_dur_prior_s: tuple[float, float] = (18.0, 15.0)   # (mean, std) of dead-time length
    # Computational cap for evaluating a duration prior. A saturated state can continue
    # with the same label, so this does not force a semantic boundary or fake rally.
    max_segment_s: float = 90.0
    transition_penalty: float = 0.4    # log-penalty per segment boundary (discourages churn)
    duration_prior_weight: float = 0.15  # how strongly duration priors pull the decode

    # ---- cut ----------------------------------------------------------------
    reencode: bool = True              # True = frame-accurate re-encode; False = fast stream-copy

    def __post_init__(self) -> None:
        # Reject NaN/inf before range comparisons: every comparison with NaN is false,
        # otherwise one bad web/API value can silently turn decoder scores into NaN.
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
        if self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold")
        for name in ("analysis_fps", "player_fps", "audio_sr", "player_pose_fps",
                     "serve_track_fps", "match_setup_fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.proxy_height <= 0:
            raise ValueError("proxy_height must be positive")
        if self.ball_inference_batch_size < 1:
            raise ValueError("ball_inference_batch_size must be positive")
        for name in ("enter_threshold", "exit_threshold", "move_thresh"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("w_audio", "w_ball", "w_pose", "w_geometry", "w_motion"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.strike_min_crest_factor <= 0:
            raise ValueError("strike_min_crest_factor must be positive")
        if not 0.0 <= self.strike_min_spectral_flatness <= 1.0:
            raise ValueError("strike_min_spectral_flatness must be in [0, 1]")
        for name in (
            "pad_pre_s", "pad_post_s", "min_rally_s", "max_segment_s",
            "smooth_window_s", "merge_gap_s", "strike_min_gap_s", "audio_block_s",
            "rhythm_window_s", "point_gap_s", "toss_preroll_s", "landing_tail_s",
            "echo_collapse_s", "min_rally_dur_s", "serve_attach_window_s",
            "match_setup_lookback_s", "match_setup_end_before_strike_s",
            "match_phase_max_gap_s", "match_fragment_merge_gap_s",
            "match_attempt_merge_min_gap_s", "match_attempt_merge_gap_s",
            "match_point_start_preroll_s",
            "match_player_activity_speed", "match_player_activity_min_s",
            "match_player_stable_s", "match_player_stable_span",
            "match_receiver_reaction_lag_s",
            "match_overhead_window_s", "match_position_post_strike_s",
            "match_position_track_step", "match_position_max_span",
            "match_ball_serve_pre_s",
            "match_ball_serve_post_s",
            "arbiter_pre_pad_s", "arbiter_post_pad_s", "arbiter_min_in_play_span_s",
            "arbiter_fragment_join_gap_s",
            "arbiter_max_candidate_s", "arbiter_min_total_s", "arbiter_max_total_s",
            "point_start_buffer_s", "point_end_buffer_s", "inter_point_gap_s",
            "ball_tail_s", "ball_max_extend_s",
            "ball_end_hint_max_uncalibrated_trim_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_segment_s <= 0:
            raise ValueError("max_segment_s must be positive")
        if self.play_mode not in {"auto", "match", "casual"}:
            raise ValueError("play_mode must be 'auto', 'match', or 'casual'")
        if self.match_attempt_merge_min_gap_s > self.match_attempt_merge_gap_s:
            raise ValueError(
                "match_attempt_merge_min_gap_s must not exceed match_attempt_merge_gap_s")
        if self.match_pose_imgsz <= 0:
            raise ValueError("match_pose_imgsz must be positive")
        if not 0.0 <= self.match_side_center_margin < 0.5:
            raise ValueError("match_side_center_margin must be in [0, 0.5)")
        if self.match_side_max_std < 0:
            raise ValueError("match_side_max_std must be non-negative")
        if not 0 < self.match_ready_knee_deg <= 180:
            raise ValueError("match_ready_knee_deg must be in (0, 180]")
        if self.match_ready_stance_ratio < 0:
            raise ValueError("match_ready_stance_ratio must be non-negative")
        if not 0.0 <= self.match_position_min_stable_fraction <= 1.0:
            raise ValueError("match_position_min_stable_fraction must be in [0, 1]")
        if not 0.0 <= self.match_position_min_score <= 1.0:
            raise ValueError("match_position_min_score must be in [0, 1]")
        if not 0.0 <= self.match_position_court_baseline_depth <= 0.5:
            raise ValueError("match_position_court_baseline_depth must be in [0, 0.5]")
        for name in ("match_position_far_baseline_y", "match_position_near_baseline_y"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1] <= 1.0:
                raise ValueError(f"{name} must be increasing bounds in [0, 1]")
        if self.match_overhead_wrist_ratio < 0:
            raise ValueError("match_overhead_wrist_ratio must be non-negative")
        if not 0.0 <= self.match_ball_min_coverage <= 1.0:
            raise ValueError("match_ball_min_coverage must be in [0, 1]")
        if not 0.0 <= self.match_ball_min_single_strike_coverage <= 1.0:
            raise ValueError(
                "match_ball_min_single_strike_coverage must be in [0, 1]")
        if not 0.0 <= self.match_ball_min_vertical_span <= 1.0:
            raise ValueError("match_ball_min_vertical_span must be in [0, 1]")
        if not 0.0 <= self.match_ball_min_outgoing_span <= 1.0:
            raise ValueError("match_ball_min_outgoing_span must be in [0, 1]")
        if not 0.0 <= self.match_player_hint_near_y_min <= 1.0:
            raise ValueError("match_player_hint_near_y_min must be in [0, 1]")
        for name in ("match_ball_court_x", "match_ball_court_y"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1] <= 1.0:
                raise ValueError(f"{name} must be increasing bounds in [0, 1]")
        if (self.match_min_ready_frames < 1 or self.match_position_min_frames < 2
                or self.match_position_min_players < 1
                or self.match_serve_strikes_to_check < 1
                or self.match_auto_min_serve_anchors < 2):
            raise ValueError("match serve/setup counts must be positive")
        if self.min_rally_s > self.max_segment_s:
            raise ValueError("min_rally_s must be <= max_segment_s")
        for name in (
            "landing_tail_s", "ball_tail_s", "point_start_buffer_s", "point_end_buffer_s"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must be <= 1 second")
        lo, hi = self.strike_band_hz
        if not (0 < lo < hi < self.audio_sr / 2):
            raise ValueError("strike_band_hz must satisfy 0 < low < high < audio_sr/2")
        for name in ("arbiter_min_conf", "arbiter_min_in_play_frac",
                     "arbiter_motion_max_active_frac", "arbiter_max_total_fraction",
                     "arbiter_diversity_weight",
                     "arbiter_player_recovery_min_coverage"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.arbiter_max_candidate_s <= 0 or self.arbiter_max_total_s <= 0:
            raise ValueError("arbiter candidate/total workload limits must be positive")
        if self.min_rally_strikes < 1 or self.arbiter_min_bounces < 0:
            raise ValueError("strike/bounce counts must be non-negative (rally strikes >= 1)")
        for name, prior in (("rally_dur_prior_s", self.rally_dur_prior_s),
                            ("gap_dur_prior_s", self.gap_dur_prior_s)):
            if len(prior) != 2 or prior[0] < 0 or prior[1] <= 0:
                raise ValueError(f"{name} must be a non-negative mean and positive std")
        if self.court_corners is not None:
            cc = self.court_corners
            if len(cc) != 4 or any(len(p) != 2 for p in cc):
                raise ValueError("court_corners must be 4 (x, y) image points")
