"""Player module: detection, court-coordinate tracking, and serve set-up — one home.

Consolidates what used to be split across ``players.py`` (detection + geometry) and
``serve.py`` (near-player court track + serve set-up). Mirrors ``ball.py``: a clean
``PlayerTracker`` produces a court-metre track of the near player (speed-limited across
neighbouring frames so nobody teleports), and the pure helpers below turn detections into
the geometry channel and the serve-start anchor.

Signals this module provides to the pipeline:
* **geometry** — two opposed players on court (a supporting rally vote).
* **near-player court track + speed** — for the movement gate (point splitting) and the
  serve set-up (point-start anchor).

Pose is used narrowly for a high-resolution near-player overhead check around early
impacts. It is never used as general rally evidence; wide/far-side serves are confirmed
by the dedicated TrackNet ball-motion check instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Person = Tuple[float, float, float]          # (foot_x_norm, foot_y_norm, box_area_norm)
Region = Tuple[float, float, float, float]   # (x0, y0, x1, y1) normalized
Segment = Tuple[float, float]


def resolve_yolo_device() -> str:
    """Use the same explicit CUDA-first policy as TrackNet, with safe CPU fallback."""
    from .ball import resolve_device

    return str(resolve_device())


@dataclass(frozen=True)
class ServeSetupObservation:
    """Visible pre-strike state used by the match-sequence validator.

    ``side`` is deliberately screen-left/screen-right, not deuce/ad. Without a reliable
    court orientation and server identity, assigning tennis names would manufacture
    certainty; alternation only needs the two stable screen-side states.
    """

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

    @property
    def confirmed_serve(self) -> bool:
        # A raised wrist alone is vulnerable to pose hallucination and ordinary
        # overhead actions. It confirms a serve only in a stationary baseline setup.
        # TrackNet's sustained toss/serve flight remains independently sufficient.
        aligned_pose_setup = any(
            abs(pose_strike - position_strike) <= 1e-6
            for pose_strike in self.overhead_strikes
            for position_strike in self.position_setup_strikes
        )
        return bool(self.ball_serve_evidence or aligned_pose_setup)


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


def discover_yolo_weights(name: str = "yolov8n.pt", models_dir: Optional[str] = None) -> str:
    """Resolve a YOLO weight name to ``models/<name>`` when it lives there (we keep all
    weights — TrackNet + YOLO — in models/), else return the bare name so ultralytics can
    use its own cache or download it. Mirrors ``ball.discover_ball_weights``."""
    import os
    from pathlib import Path

    if models_dir is None:
        models_dir = os.environ.get("RALLY_MODELS_DIR")
        if not models_dir:
            models_dir = str(Path(__file__).resolve().parents[2] / "models")
    local = Path(models_dir) / name
    return str(local) if local.is_file() else name


# ---------------------------------------------------------------------------
# detection + geometry channel
# ---------------------------------------------------------------------------
class PlayerDetector:
    """Wraps a YOLO person detector. ``available`` is False when YOLO is missing."""

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.3):
        self.conf = conf
        self.model = None
        self.device = "cpu"
        try:  # pragma: no cover - optional heavy dependency
            from ultralytics import YOLO

            self.device = resolve_yolo_device()
            self.model = YOLO(discover_yolo_weights(model))
            self.model.to(self.device)
        except Exception:
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect_persons(self, frame_bgr: np.ndarray) -> List[Person]:  # pragma: no cover
        """Normalized foot points for detected persons (class 0 = person)."""
        if self.model is None:
            return []
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(
            frame_bgr, conf=self.conf, classes=[0], verbose=False,
            device=self.device)
        persons: List[Person] = []
        for r in res:
            for box in r.boxes.xyxy.cpu().numpy():
                x0, y0, x1, y1 = box
                persons.append(((x0 + x1) / 2.0 / w, y1 / h, ((x1 - x0) * (y1 - y0)) / (w * h)))
        return persons


def estimate_court_region(all_feet: Sequence[Tuple[float, float]],
                          margin: float = 0.05) -> Optional[Region]:
    """Robust central region (5th-95th percentile of foot points) + margin."""
    pts = np.asarray(all_feet, dtype=float)
    if pts.shape[0] < 10:
        return None
    x0, y0 = np.percentile(pts, 5, axis=0)
    x1, y1 = np.percentile(pts, 95, axis=0)
    x0, y0 = max(0.0, x0 - margin), max(0.0, y0 - margin)
    x1, y1 = min(1.0, x1 + margin), min(1.0, y1 + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return (float(x0), float(y0), float(x1), float(y1))


def _in_region(p: Person, region: Region) -> bool:
    x, y, _ = p
    x0, y0, x1, y1 = region
    return x0 <= x <= x1 and y0 <= y <= y1


def geometry_score_from_persons(persons: Sequence[Person], region: Optional[Region]) -> float:
    """Score a frame's player configuration in [0, 1] (two opposed players -> 1.0)."""
    if region is None:
        return 0.0
    inside = [p for p in persons if _in_region(p, region)]
    n = len(inside)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.2
    if n >= 3:
        return 0.3
    mid = (region[1] + region[3]) / 2.0
    opposed = (inside[0][1] < mid) != (inside[1][1] < mid)
    return 1.0 if opposed else 0.5


# ---------------------------------------------------------------------------
# near-player court tracking (speed-limited) + serve set-up
# ---------------------------------------------------------------------------
def track_near_player(video: str, court, fps_a: float = 5.0, model=None):
    """Detect the near player each analysis frame -> (times, court_x, court_y) in metres.

    Near player = the largest person box whose foot is in the lower part of the frame.
    NaN where not detected; pass through :func:`clean_track` to speed-limit and smooth.
    Pass a preloaded ``model`` (a YOLO instance) to avoid re-loading the detector.
    """
    import cv2

    if model is None:
        from ultralytics import YOLO
        model = YOLO(discover_yolo_weights("yolov8n.pt"))
    device = resolve_yolo_device()
    model.to(device)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_a
    stride = max(1, int(round(fps / fps_a)))
    ts, cx, cy = [], [], []
    fi = 0
    try:
        while True:
            if not cap.grab():
                break
            if fi % stride == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                h = fr.shape[0]
                r = model.predict(
                    fr, conf=0.3, classes=[0], verbose=False, device=device)[0]
                boxes = [b for b in r.boxes.xyxy.cpu().numpy() if b[3] / h > 0.60]
                ts.append(fi / fps)
                if boxes:
                    b = max(boxes, key=lambda z: (z[2] - z[0]) * (z[3] - z[1]))
                    c = court.to_court([[(b[0] + b[2]) / 2, b[3]]])[0]
                    cx.append(float(c[0])); cy.append(float(c[1]))
                else:
                    cx.append(np.nan); cy.append(np.nan)
            fi += 1
    finally:
        cap.release()
    return np.array(ts), np.array(cx), np.array(cy)


def clean_track(times: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                speed_limit_mps: float = 8.0, smooth_win: int = 5,
                max_gap_dt_s: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Speed-limit + median-smooth a court-coordinate track (nobody teleports).

    The speed test uses the time since the last *accepted* point, capped at ``max_gap_dt_s``
    so a long detection dropout can't make a far jump look slow (a genuine teleport after a
    3 s gap would otherwise pass); rejected points are interpolated instead.
    """
    times = np.asarray(times, float)
    cx = np.array(cx, float)
    cy = np.array(cy, float)
    n = times.size
    gx = np.full(n, np.nan)
    gy = np.full(n, np.nan)
    last_i = None
    for i in range(n):
        if not (np.isfinite(cx[i]) and np.isfinite(cy[i])):
            continue
        if last_i is None:
            gx[i], gy[i] = cx[i], cy[i]
            last_i = i
        else:
            dt = min(max(times[i] - times[last_i], 1e-3), max_gap_dt_s)
            if np.hypot(cx[i] - gx[last_i], cy[i] - gy[last_i]) / dt <= speed_limit_mps:
                gx[i], gy[i] = cx[i], cy[i]
                last_i = i
    idx = np.arange(n)
    good = np.isfinite(gx)
    if good.sum() >= 2:
        gx = np.interp(idx, idx[good], gx[good])
        gy = np.interp(idx, idx[good], gy[good])
    if smooth_win > 1 and n >= smooth_win:
        from scipy.signal import medfilt
        k = smooth_win + (smooth_win + 1) % 2       # force odd
        if k > n:                                    # kernel must not exceed sample count
            k = n if n % 2 == 1 else n - 1
        if k >= 3:
            gx = medfilt(gx, k)
            gy = medfilt(gy, k)
    return gx, gy


def court_speed(times: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    times = np.asarray(times, float)
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    sp = np.zeros(times.size)
    for i in range(1, times.size):
        dt = max(times[i] - times[i - 1], 1e-3)
        sp[i] = np.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / dt
    return sp


def find_serve_start(rally_first_strike: float, times: np.ndarray, cy: np.ndarray,
                     speed: np.ndarray, *, lookback_s: float = 6.0, baseline_y: float = 1.5,
                     still_speed: float = 0.6, preroll_s: float = 0.8,
                     max_lead_s: float = 2.5) -> Optional[float]:
    """Anchor a point's start on the near player's serve/receive set-up (still at/behind
    the baseline before the rally); capped at ``max_lead_s`` to drop pre-serve loiter."""
    times = np.asarray(times, float)
    m = (times >= rally_first_strike - lookback_s) & (times <= rally_first_strike)
    if not m.any():
        return None
    idx = np.where(m)[0]
    setmask = (cy[idx] < baseline_y) & (speed[idx] < still_speed)
    if not setmask.any():
        return None
    run_end = idx[np.where(setmask)[0][-1]]
    j = run_end
    while j - 1 in idx and (cy[j - 1] < baseline_y) and (speed[j - 1] < still_speed):
        j -= 1
    anchor = max(float(times[j]) - preroll_s, rally_first_strike - max_lead_s)
    return max(0.0, anchor)


def refine_starts_with_serve(points: List[Segment], onsets: np.ndarray, times: np.ndarray,
                             cy: np.ndarray, speed: np.ndarray, **kw) -> List[Segment]:
    """Extend each point's start back to the detected near-player serve set-up."""
    onsets = np.sort(np.asarray(onsets, float))
    out: List[Segment] = []
    prev_end = -1e9
    for s, e in points:
        rally = onsets[(onsets >= s) & (onsets <= e)]
        first = float(rally.min()) if rally.size else s
        anchor = find_serve_start(first, times, cy, speed, **kw)
        new_s = s if anchor is None else min(s, anchor)
        out.append((max(new_s, prev_end), e))
        prev_end = e
    return out


# ---------------------------------------------------------------------------
# pose-activity channel (near player) — a confidence-weighted vote
# ---------------------------------------------------------------------------
def pose_activity_track(
    video: str, timeline: np.ndarray, fps_a: float = 2.0,
    cancel_check: Callable[[], None] = lambda: None,
):
    """Per-analysis-frame pose-activity score + confidence for the near player.

    Uses YOLOv8-pose. "Activity" = how far the racket arm is raised above the hips
    (0 = arms hanging / standing between points; ~1 = wrist at shoulder height, hitting/
    serving). Confidence = mean keypoint confidence × player size, so a small/blurred far
    player contributes ~0 (drops out of the fusion) while a clear near player votes fully.
    Returns (pose_score, pose_conf) aligned to ``timeline`` (forward-filled between samples).
    """
    import cv2
    from ultralytics import YOLO

    timeline = np.asarray(timeline, float)
    score = np.zeros(timeline.size, float)
    conf = np.zeros(timeline.size, float)
    try:
        model = YOLO(discover_yolo_weights("yolov8n-pose.pt"))
        device = resolve_yolo_device()
        model.to(device)
    except Exception:
        return score, conf

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_a
    stride = max(1, int(round(fps / fps_a)))
    samp_t, samp_s, samp_c = [], [], []
    fi = 0
    try:
        while True:
            cancel_check()
            if not cap.grab():
                break
            if fi % stride == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                h, w = fr.shape[:2]
                r = model.predict(
                    fr, conf=0.3, classes=[0], verbose=False, device=device)[0]
                s_val, c_val = 0.0, 0.0
                if r.keypoints is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    kp = r.keypoints.xy.cpu().numpy()
                    kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                    # near player = largest box with foot in lower half
                    cand = [i for i in range(len(boxes)) if boxes[i][3] / h > 0.55]
                    if cand:
                        i = max(cand, key=lambda j: (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]))
                        k = kp[i]
                        sh_y = np.mean([k[5][1], k[6][1]])
                        hip_y = np.mean([k[11][1], k[12][1]])
                        # only trust wrists YOLO actually localized: a missing keypoint is
                        # (0,0), and min() would read it as a wrist at the image top ->
                        # spurious "arm fully raised". Fall back to no activity signal.
                        wrists = []
                        for wj in (9, 10):
                            wconf = kc[i][wj] if kc is not None else 1.0
                            if wconf > 0.2 and (k[wj][0] > 0 or k[wj][1] > 0):
                                wrists.append(k[wj][1])
                        torso = abs(hip_y - sh_y) + 1e-6
                        s_val = float(np.clip((hip_y - min(wrists)) / torso, 0.0, 1.0)) if wrists else 0.0
                        if kc is not None:
                            kq = float(np.mean([kc[i][j] for j in (5, 6, 9, 10, 11, 12)]))
                        else:
                            kq = 0.5
                        area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) / (w * h)
                        c_val = float(np.clip(kq * np.clip(area * 40, 0.0, 1.0), 0.0, 1.0))
                samp_t.append(fi / fps); samp_s.append(s_val); samp_c.append(c_val)
            fi += 1
    finally:
        cap.release()
    if not samp_t:
        return score, conf
    samp_t = np.array(samp_t)
    ss, sc = np.array(samp_s), np.array(samp_c)
    # nearest-sample fill onto the analysis timeline (searchsorted alone gives the *next*
    # sample -> a forward bias of up to 1/fps_a; pick whichever neighbour is closer)
    if samp_t.size == 1:
        idx = np.zeros(timeline.size, dtype=int)
    else:
        pos = np.clip(np.searchsorted(samp_t, timeline), 1, samp_t.size - 1)
        left = pos - 1
        idx = np.where((timeline - samp_t[left]) <= (samp_t[pos] - timeline), left, pos)
    return ss[idx], sc[idx]


# ---------------------------------------------------------------------------
# point-level serve / receiver setup observation
# ---------------------------------------------------------------------------
def _position_frame_coordinates(persons, court=None, frame_size=None) -> np.ndarray:
    """Return filtered player foot positions in normalized court/view coordinates."""
    raw = np.asarray([[person[0], person[1]] for person in persons], dtype=float)
    if raw.size == 0:
        return np.empty((0, 2), dtype=float)
    calibrated = court is not None and frame_size is not None
    if calibrated:
        width, height = frame_size
        pixels = raw * np.array([float(width), float(height)])
        from .court import COURT_L, DOUBLES_W

        raw = court.to_court(pixels) / np.array([DOUBLES_W, COURT_L])
        keep = (
            np.isfinite(raw).all(axis=1)
            & (raw[:, 0] >= -0.15) & (raw[:, 0] <= 1.15)
            & (raw[:, 1] >= -0.15) & (raw[:, 1] <= 1.15)
        )
    else:
        # Conservative central-court envelope. It excludes most spectators and people
        # behind the fence while retaining a server just behind either baseline.
        keep = (
            np.isfinite(raw).all(axis=1)
            & (raw[:, 0] >= 0.04) & (raw[:, 0] <= 0.96)
            & (raw[:, 1] >= 0.30) & (raw[:, 1] <= 0.995)
        )
    return raw[keep]


def classify_position_setup(
    point: Segment,
    first_strike: float,
    player_samples,
    cfg,
    *,
    court=None,
    frame_size=None,
) -> PositionSetupObservation:
    """Measure a stationary baseline formation before a candidate's first impact.

    Person detections are associated by nearest foot position. A setup is present when
    at least one stable track occupies a baseline band and the configured fraction of
    all reliably observed players stays stable. This is supporting evidence only; a
    dynamic pose or ball-flight event is still required to confirm a serve.
    """
    point = (float(point[0]), float(point[1]))
    first_strike = float(first_strike)
    start = max(0.0, first_strike - float(cfg.match_setup_lookback_s))
    end = min(float(point[1]), first_strike + cfg.match_position_post_strike_s)
    frames = [
        (float(sample_time), _position_frame_coordinates(
            persons, court=court, frame_size=frame_size))
        for sample_time, persons in player_samples
        if start - 1e-9 <= float(sample_time) <= end + 1e-9
    ]
    if not frames:
        return PositionSetupObservation(
            point=point, best_strike=first_strike, setup_strikes=(), checked=False,
            setup_evidence=False, score=0.0, server_end=None, server_span=None,
            player_tracks=0, stable_tracks=0, stable_fraction=0.0,
            sampled_frames=0)

    # Greedy nearest-neighbour association is sufficient over this short, fixed-camera
    # window. A deliberately generous step keeps walking players in one track so their
    # displacement is measured rather than hidden as several short tracks.
    tracks: list[dict] = []
    max_gap_s = max(0.75, 2.5 / max(float(cfg.player_fps), 1e-6))
    for sample_time, positions in frames:
        active = [
            index for index, track in enumerate(tracks)
            if sample_time - track["last_time"] <= max_gap_s
        ]
        pairs = sorted(
            (
                (float(np.linalg.norm(position - tracks[index]["last"])), index, pi)
                for index in active
                for pi, position in enumerate(positions)
            ),
            key=lambda item: item[0],
        )
        used_tracks: set[int] = set()
        used_positions: set[int] = set()
        for distance, index, pi in pairs:
            if distance > cfg.match_position_track_step:
                break
            if index in used_tracks or pi in used_positions:
                continue
            position = positions[pi]
            tracks[index]["points"].append(position)
            tracks[index]["last"] = position
            tracks[index]["last_time"] = sample_time
            used_tracks.add(index)
            used_positions.add(pi)
        for pi, position in enumerate(positions):
            if pi not in used_positions:
                tracks.append({
                    "points": [position], "last": position, "last_time": sample_time,
                })

    reliable = [
        np.asarray(track["points"], dtype=float) for track in tracks
        if len(track["points"]) >= int(cfg.match_position_min_frames)
    ]
    if len(reliable) < int(cfg.match_position_min_players):
        return PositionSetupObservation(
            point=point, best_strike=first_strike, setup_strikes=(), checked=True,
            setup_evidence=False, score=0.0, server_end=None, server_span=None,
            player_tracks=len(reliable), stable_tracks=0, stable_fraction=0.0,
            sampled_frames=len(frames))

    spans = [float(max(np.ptp(track[:, 0]), np.ptp(track[:, 1]))) for track in reliable]
    stable = [span <= cfg.match_position_max_span for span in spans]
    stable_count = int(sum(stable))
    stable_fraction = float(stable_count / len(reliable))
    calibrated = court is not None and frame_size is not None
    candidates: list[tuple[float, str]] = []
    for track, span, is_stable in zip(reliable, spans, stable):
        if not is_stable:
            continue
        median_y = float(np.median(track[:, 1]))
        if calibrated:
            depth = cfg.match_position_court_baseline_depth
            server_end = "near" if median_y <= depth else (
                "far" if median_y >= 1.0 - depth else None)
        else:
            far0, far1 = cfg.match_position_far_baseline_y
            near0, near1 = cfg.match_position_near_baseline_y
            server_end = "far" if far0 <= median_y <= far1 else (
                "near" if near0 <= median_y <= near1 else None)
        if server_end is not None:
            candidates.append((span, server_end))

    if candidates:
        server_span, server_end = min(candidates, key=lambda item: item[0])
        stability_score = float(np.clip(
            1.0 - server_span / max(cfg.match_position_max_span, 1e-9), 0.0, 1.0))
    else:
        server_span, server_end, stability_score = None, None, 0.0
    score = float(stability_score * stable_fraction)
    setup = bool(
        candidates
        and stable_fraction >= cfg.match_position_min_stable_fraction
        and score >= cfg.match_position_min_score
    )
    return PositionSetupObservation(
        point=point, best_strike=first_strike,
        setup_strikes=(first_strike,) if setup else (), checked=True,
        setup_evidence=setup, score=score, server_end=server_end,
        server_span=server_span, player_tracks=len(reliable),
        stable_tracks=stable_count, stable_fraction=stable_fraction,
        sampled_frames=len(frames))


def observe_position_setups(
    points: Sequence[Segment],
    onsets: np.ndarray,
    player_samples,
    cfg,
    *,
    court=None,
    frame_size=None,
) -> List[PositionSetupObservation]:
    """Classify position setup for each point from the shared visual-pass detections."""
    ordered = np.sort(np.asarray(onsets, dtype=float))
    observations: List[PositionSetupObservation] = []
    for point in points:
        strikes = ordered[(ordered >= point[0] - 1e-9) & (ordered <= point[1] + 1e-9)]
        considered = strikes[: int(cfg.match_serve_strikes_to_check)]
        if not considered.size:
            considered = np.array([float(point[0])])
        per_strike = [
            classify_position_setup(
                point, float(strike), player_samples, cfg,
                court=court, frame_size=frame_size)
            for strike in considered
        ]
        best = max(
            per_strike,
            key=lambda item: (
                int(item.setup_evidence), item.score,
                item.stable_fraction, item.player_tracks),
        )
        setup_strikes = tuple(
            float(item.best_strike) for item in per_strike
            if item.setup_evidence and item.best_strike is not None
        )
        observations.append(replace(
            best,
            checked=any(item.checked for item in per_strike),
            setup_evidence=bool(setup_strikes),
            setup_strikes=setup_strikes,
        ))
    return observations


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, guarded against collapsed pose keypoints."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def observe_serve_setups(
    video: str,
    points: Sequence[Segment],
    onsets: np.ndarray,
    cfg,
    cancel_check: Callable[[], None] = lambda: None,
) -> List[ServeSetupObservation]:  # pragma: no cover - exercised with optional YOLO
    """Observe the visible player setup immediately before each candidate's first hit.

    A far-side serve is usually too small for dependable arm keypoints, so receiver posture
    remains useful only as sequence context. Raw pose evidence requires the large near-side
    player's wrist to rise substantially over the shoulder near an early impact; match
    validation then pairs it with the separately measured stationary baseline formation.
    TrackNet independently handles far-side and underarm serves.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(discover_yolo_weights("yolov8n-pose.pt"))
    device = resolve_yolo_device()
    model.to(device)
    ordered_onsets = np.sort(np.asarray(onsets, dtype=float))
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")

    observations: List[ServeSetupObservation] = []
    step = 1.0 / float(cfg.match_setup_fps)
    try:
        for point in points:
            cancel_check()
            start, end = point
            strikes = ordered_onsets[
                (ordered_onsets >= start - 1e-9) & (ordered_onsets <= end + 1e-9)
            ]
            serve_strikes = strikes[: int(cfg.match_serve_strikes_to_check)]
            first = float(strikes[0]) if strikes.size else float(start)
            sample_start = max(0.0, first - float(cfg.match_setup_lookback_s))
            sample_end = max(sample_start, first - float(cfg.match_setup_end_before_strike_s))
            side_times = np.arange(sample_start, sample_end + step * 0.25, step)
            # The first accepted transient can be a bounce/feed. Sample around each of the
            # first few impacts rather than scanning the full point; this catches a later
            # serve without letting an ordinary mid-rally overhead become serve evidence.
            pose_event_times = np.array([
                float(strike) + offset
                for strike in serve_strikes
                for offset in np.arange(
                    -cfg.match_overhead_window_s,
                    cfg.match_overhead_window_s + step * 0.25,
                    step,
                )
                if start <= float(strike) + offset <= end
            ], dtype=float)
            sample_times = np.unique(np.round(
                np.concatenate((side_times, pose_event_times)), 6))

            xs: List[float] = []
            ready_frames = 0
            overhead_frames = 0
            overhead_max_ratio = 0.0
            overhead_strikes: set[float] = set()
            pose_frames = 0
            sampled_frames = 0
            for sample_time in sample_times:
                cancel_check()
                cap.set(cv2.CAP_PROP_POS_MSEC, float(sample_time) * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                sampled_frames += 1
                height, width = frame.shape[:2]
                result = model.predict(
                    frame, conf=0.15, classes=[0], verbose=False,
                    imgsz=int(cfg.match_pose_imgsz),
                    device=device,
                )[0]
                if result.keypoints is None or len(result.boxes) == 0:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                confidence = (result.keypoints.conf.cpu().numpy()
                              if result.keypoints.conf is not None else None)
                candidates = [
                    i for i, box in enumerate(boxes)
                    if box[3] / height > 0.55
                    and 0.06 < (box[0] + box[2]) / (2.0 * width) < 0.94
                ]
                if not candidates:
                    continue
                # The near player is much larger than the far player. Selecting by area
                # per point also permits the players to swap ends between games; a global
                # nearest-neighbour identity would incorrectly stick to the far player.
                selected = max(
                    candidates,
                    key=lambda i: ((boxes[i][2] - boxes[i][0])
                                   * (boxes[i][3] - boxes[i][1])),
                )
                box = boxes[selected]
                pose = keypoints[selected]
                conf = (confidence[selected] if confidence is not None
                        else np.ones(pose.shape[0], dtype=float))
                body_joints = (5, 6, 11, 12, 13, 14, 15, 16)
                body_quality = float(np.mean(conf[list(body_joints)]))
                if body_quality < 0.45:
                    continue
                pose_frames += 1
                if sample_start - 1e-6 <= sample_time <= sample_end + 1e-6:
                    xs.append(float((box[0] + box[2]) / (2.0 * width)))

                knee_angles = []
                for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
                    if min(float(conf[hip]), float(conf[knee]), float(conf[ankle])) > 0.2:
                        angle = _joint_angle_deg(pose[hip], pose[knee], pose[ankle])
                        if np.isfinite(angle):
                            knee_angles.append(angle)
                shoulder_mid = (pose[5] + pose[6]) / 2.0
                hip_mid = (pose[11] + pose[12]) / 2.0
                torso = float(np.linalg.norm(hip_mid - shoulder_mid))
                stance = float("nan")
                if torso > 1e-6 and min(float(conf[15]), float(conf[16])) > 0.2:
                    stance = float(np.linalg.norm(pose[15] - pose[16]) / torso)
                ready = ((knee_angles and min(knee_angles) <= cfg.match_ready_knee_deg)
                         or (np.isfinite(stance)
                             and stance >= cfg.match_ready_stance_ratio))
                ready_frames += int(bool(ready))

                if torso > 1e-6:
                    shoulder_y = float((pose[5][1] + pose[6][1]) / 2.0)
                    wrist_ratios = []
                    for wrist in (9, 10):
                        if (float(conf[wrist]) > 0.2
                                and (pose[wrist][0] > 0 or pose[wrist][1] > 0)):
                            wrist_ratios.append(
                                (shoulder_y - float(pose[wrist][1])) / torso)
                    ratio = max(wrist_ratios, default=-1.0)
                    near_early_impact = any(
                        abs(float(sample_time) - float(strike))
                        <= cfg.match_overhead_window_s + 1e-9
                        for strike in serve_strikes
                    )
                    if near_early_impact:
                        overhead_max_ratio = max(overhead_max_ratio, float(ratio))
                        if ratio >= cfg.match_overhead_wrist_ratio:
                            overhead_frames += 1
                            nearest_strike = min(
                                serve_strikes,
                                key=lambda strike: abs(
                                    float(sample_time) - float(strike)),
                            )
                            overhead_strikes.add(float(nearest_strike))

            near_x = float(np.median(xs)) if xs else None
            x_std = float(np.std(xs)) if xs else None
            side = None
            side_confidence = 0.0
            if (len(xs) >= 3 and x_std is not None
                    and x_std <= cfg.match_side_max_std and near_x is not None):
                distance = abs(near_x - 0.5)
                if distance >= cfg.match_side_center_margin:
                    side = "left" if near_x < 0.5 else "right"
                    side_confidence = float(np.clip(
                        (distance - cfg.match_side_center_margin) / 0.20, 0.0, 1.0))
            serve_motion = overhead_frames > 0
            setup = ready_frames >= cfg.match_min_ready_frames or serve_motion
            observations.append(ServeSetupObservation(
                point=(float(start), float(end)), first_strike=first,
                side=side, side_confidence=side_confidence,
                near_x=near_x, near_x_std=x_std, pose_frames=pose_frames,
                sampled_frames=sampled_frames,
                ready_frames=ready_frames, serve_motion=serve_motion,
                setup_evidence=bool(setup), observable=sampled_frames >= 3,
                overhead_frames=overhead_frames,
                overhead_max_ratio=overhead_max_ratio,
                overhead_strikes=tuple(sorted(overhead_strikes)),
            ))
    finally:
        cap.release()
    return observations


# ---------------------------------------------------------------------------
# cohesive interface
# ---------------------------------------------------------------------------
@dataclass
class PlayerCourtTrack:
    t: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    speed: np.ndarray


class PlayerTracker:
    """Track the near player in court metres (detection -> homography -> speed-limit).

    ``court_track`` is what the serve set-up consumes; ``detector``/``geometry`` helpers
    above feed the geometry channel (computed in the sampling pass for efficiency)."""

    def __init__(self, detector: Optional[PlayerDetector] = None):
        self.detector = detector

    def court_track(self, video: str, court, fps_a: float = 5.0,
                    speed_limit_mps: float = 8.0) -> PlayerCourtTrack:
        # reuse the detector's already-loaded YOLO model rather than loading a second copy
        model = self.detector.model if (self.detector and self.detector.available) else None
        t, cx, cy = track_near_player(video, court, fps_a, model=model)
        cx, cy = clean_track(t, cx, cy, speed_limit_mps)
        return PlayerCourtTrack(t, cx, cy, court_speed(t, cx, cy))
