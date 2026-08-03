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

import warnings
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (
    DEFAULT_RTMPOSE_MODEL,
    DEFAULT_YOLO_DETECTION_MODEL,
    DEFAULT_YOLO_POSE_MODEL,
)
from ..domain.observations import PositionSetupObservation, ServeSetupObservation
from .court import COURT_L, NET_Y

Person = Tuple[float, float, float]          # (foot_x_norm, foot_y_norm, box_area_norm)
Region = Tuple[float, float, float, float]   # (x0, y0, x1, y1) normalized
Segment = Tuple[float, float]


def resolve_yolo_device() -> str:
    """Use the same explicit CUDA-first policy as TrackNet, with safe CPU fallback."""
    from .ball import resolve_device

    return str(resolve_device())


def discover_yolo_weights(
    name: str = DEFAULT_YOLO_DETECTION_MODEL,
    models_dir: Optional[str] = None,
) -> str:
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

    def __init__(self, model: str = DEFAULT_YOLO_DETECTION_MODEL, conf: float = 0.3):
        self.conf = conf
        self.model = None
        self.device = "cpu"
        self.error: Optional[str] = None
        try:  # pragma: no cover - optional heavy dependency
            from ultralytics import YOLO

            self.device = resolve_yolo_device()
            self.model = YOLO(discover_yolo_weights(model))
            if getattr(self.model, "task", None) not in {None, "detect"}:
                raise ValueError(
                    f"player detection requires a detect checkpoint, got {self.model.task!r}")
            self.model.to(self.device)
        except Exception as exc:
            self.error = str(exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect_persons(self, frame_bgr: np.ndarray) -> List[Person]:  # pragma: no cover
        """Normalized foot points for detected persons (class 0 = person)."""
        return self.detect_persons_batch([frame_bgr])[0]

    def detect_persons_batch(self, frames: Sequence[np.ndarray]) -> List[List[Person]]:
        """Batch person inference while preserving one result list per input frame."""
        if self.model is None:
            return [[] for _frame in frames]
        if not frames:
            return []
        res = self.model.predict(
            list(frames), conf=self.conf, classes=[0], verbose=False,
            device=self.device, batch=min(16, len(frames)))
        output: List[List[Person]] = []
        for frame_bgr, r in zip(frames, res):
            h, w = frame_bgr.shape[:2]
            persons: List[Person] = []
            for box in r.boxes.xyxy.cpu().numpy():
                x0, y0, x1, y1 = box
                persons.append(((x0 + x1) / 2.0 / w, y1 / h, ((x1 - x0) * (y1 - y0)) / (w * h)))
            output.append(persons)
        if len(output) != len(frames):
            raise RuntimeError("YOLO returned a different number of results than input frames")
        return output


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


def persons_in_court(
    persons: Sequence[Person], court, image_shape: Tuple[int, int], *,
    sideline_margin_m: float = 1.5, baseline_margin_m: float = 3.0,
) -> Tuple[List[Person], np.ndarray]:
    """Keep detections whose foot point belongs to the calibrated target court.

    YOLO foot points are normalized image coordinates. Mapping those points through the
    target homography is materially safer than learning a rectangular ROI from everyone
    seen in the video: spectators and players on a neighboring court must not expand the
    very region used to decide whether they are relevant. Margins retain servers behind
    the baseline and players pulled just outside a sideline.

    Returns the retained detections and their aligned ``(court_x, court_y)`` coordinates.
    """
    from .court import COURT_L, DOUBLES_W

    people = list(persons)
    if not people:
        return [], np.empty((0, 2), dtype=float)
    h, w = image_shape[:2]
    feet = np.array([[float(p[0]) * w, float(p[1]) * h] for p in people], dtype=float)
    try:
        coords = np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
    except Exception:
        return [], np.empty((0, 2), dtype=float)
    keep = (np.isfinite(coords).all(axis=1)
            & (coords[:, 0] >= -sideline_margin_m)
            & (coords[:, 0] <= DOUBLES_W + sideline_margin_m)
            & (coords[:, 1] >= -baseline_margin_m)
            & (coords[:, 1] <= COURT_L + baseline_margin_m))
    indices = np.flatnonzero(keep)
    return [people[int(i)] for i in indices], coords[indices]


def target_court_box_indices(
    boxes: np.ndarray, court, image_shape: Tuple[int, int], *,
    sideline_margin_m: float = 1.5, baseline_margin_m: float = 3.0,
) -> List[int]:
    """Return indices of image-space boxes whose feet belong to the target court.

    This is shared by the pose pass, whose boxes are kept aligned with keypoints and
    therefore cannot be reduced to a plain list of :class:`Person` tuples first.
    """
    from .court import COURT_L, DOUBLES_W

    values = np.asarray(boxes, dtype=float)
    if values.size == 0:
        return []
    try:
        values = values.reshape(-1, 4)
    except ValueError:
        return []
    h, w = image_shape[:2]
    if h <= 0 or w <= 0:
        return []
    feet = np.column_stack(((values[:, 0] + values[:, 2]) / 2.0, values[:, 3]))
    try:
        coords = np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
    except Exception:
        return []
    keep = (np.isfinite(coords).all(axis=1)
            & (coords[:, 0] >= -sideline_margin_m)
            & (coords[:, 0] <= DOUBLES_W + sideline_margin_m)
            & (coords[:, 1] >= -baseline_margin_m)
            & (coords[:, 1] <= COURT_L + baseline_margin_m))
    return [int(index) for index in np.flatnonzero(keep)]


def geometry_score_from_court_persons(
    persons: Sequence[Person], court, image_shape: Tuple[int, int], *,
    sideline_margin_m: float = 1.5, baseline_margin_m: float = 3.0,
) -> float:
    """Target-court player configuration score using court-metre net sides."""
    inside, coords = persons_in_court(
        persons, court, image_shape, sideline_margin_m=sideline_margin_m,
        baseline_margin_m=baseline_margin_m)
    n = len(inside)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.2
    if n >= 3:
        return 0.3
    opposed = (coords[0, 1] < NET_Y) != (coords[1, 1] < NET_Y)
    return 1.0 if opposed else 0.5


# ---------------------------------------------------------------------------
# near-player court tracking (speed-limited) + serve set-up
# ---------------------------------------------------------------------------
def track_near_player(
    video: str, court, fps_a: float = 5.0, model=None,
    model_name: str = DEFAULT_YOLO_DETECTION_MODEL,
):
    """Detect the near player each analysis frame -> (times, court_x, court_y) in metres.

    Near player = the largest person box whose foot is in the lower part of the frame.
    NaN where not detected; pass through :func:`clean_track` to speed-limit and smooth.
    Pass a preloaded ``model`` (a YOLO instance) to avoid re-loading the detector.
    """
    import cv2

    if model is None:
        from ultralytics import YOLO
        model = YOLO(discover_yolo_weights(model_name))
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
                r = model.predict(
                    fr, conf=0.3, classes=[0], verbose=False, device=device)[0]
                all_boxes = r.boxes.xyxy.cpu().numpy()
                target = target_court_box_indices(all_boxes, court, fr.shape[:2])
                feet = np.stack([
                    (all_boxes[:, 0] + all_boxes[:, 2]) / 2.0,
                    all_boxes[:, 3],
                ], axis=1) if len(all_boxes) else np.zeros((0, 2), dtype=float)
                coords = court.to_court(feet) if len(feet) else feet
                candidates = [i for i in target if coords[i, 1] <= NET_Y]
                ts.append(fi / fps)
                if candidates:
                    index = max(
                        candidates,
                        key=lambda i: ((all_boxes[i][2] - all_boxes[i][0])
                                       * (all_boxes[i][3] - all_boxes[i][1])),
                    )
                    c = coords[index]
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
    # Fill only bounded interior dropouts. np.interp over the whole array fabricates
    # stationary history before the first detection, after decoder failure, and across
    # arbitrarily long gaps—the exact pattern the serve-setup detector looks for.
    good_indices = np.flatnonzero(np.isfinite(gx) & np.isfinite(gy))
    for left, right in zip(good_indices, good_indices[1:]):
        if right <= left + 1 or times[right] - times[left] > max_gap_dt_s:
            continue
        fraction = np.linspace(0.0, 1.0, right - left + 1)
        gx[left:right + 1] = gx[left] + fraction * (gx[right] - gx[left])
        gy[left:right + 1] = gy[left] + fraction * (gy[right] - gy[left])
    if smooth_win > 1:
        from scipy.ndimage import median_filter
        finite = np.isfinite(gx) & np.isfinite(gy)
        edges = np.diff(np.r_[False, finite, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            length = int(stop - start)
            k = min(smooth_win + (smooth_win + 1) % 2,
                    length if length % 2 else length - 1)
            if k >= 3:
                gx[start:stop] = median_filter(gx[start:stop], size=k, mode="nearest")
                gy[start:stop] = median_filter(gy[start:stop], size=k, mode="nearest")
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
    model_name: str = DEFAULT_YOLO_POSE_MODEL,
    pose_backend: str = "yolo",
    detection_model: str = DEFAULT_YOLO_DETECTION_MODEL,
    rtmpose_runtime: str = "onnxruntime",
    rtmpose_device: str = "auto",
    court=None,
):
    """Per-analysis-frame pose-activity score + confidence for the near player.

    Uses the configured Ultralytics pose model. "Activity" = how far the racket arm is
    raised above the hips
    (0 = arms hanging / standing between points; ~1 = wrist at shoulder height, hitting/
    serving). Confidence = mean keypoint confidence × player size, so a small/blurred far
    player contributes ~0 (drops out of the fusion) while a clear near player votes fully.
    Returns ``(pose_score, pose_conf)`` aligned to ``timeline`` by the nearest real sample.
    Times outside the sampled video extent remain zero rather than inheriting a stale pose.
    """
    if pose_backend == "rtmlib":
        return _rtmlib_pose_activity_track(
            video, timeline, fps_a=fps_a, cancel_check=cancel_check,
            detection_model=detection_model, pose_model=model_name,
            runtime=rtmpose_runtime, pose_device=rtmpose_device, court=court)

    import cv2
    from ultralytics import YOLO

    timeline = np.asarray(timeline, float)
    score = np.zeros(timeline.size, float)
    conf = np.zeros(timeline.size, float)
    model = YOLO(discover_yolo_weights(model_name))
    if getattr(model, "task", None) not in {None, "pose"}:
        raise ValueError(f"pose activity requires a pose checkpoint, got {model.task!r}")
    device = resolve_yolo_device()
    model.to(device)

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
                    if court is not None:
                        target = target_court_box_indices(boxes, court, fr.shape[:2])
                        feet = np.stack([
                            (boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3],
                        ], axis=1)
                        coords = court.to_court(feet)
                        cand = [i for i in target if coords[i, 1] <= NET_Y]
                    else:
                        # Uncalibrated compatibility path: near player is the largest box
                        # with a foot in the lower half.
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
    half_step = 0.5 / max(float(fps_a), 1e-9)
    valid = (
        (timeline >= samp_t[0] - half_step)
        & (timeline <= samp_t[-1] + half_step)
    )
    if not np.any(valid):
        return score, conf
    target = timeline[valid]
    if samp_t.size == 1:
        idx = np.zeros(target.size, dtype=int)
    else:
        pos = np.clip(np.searchsorted(samp_t, target), 1, samp_t.size - 1)
        left = pos - 1
        idx = np.where((target - samp_t[left]) <= (samp_t[pos] - target), left, pos)
    score[valid] = ss[idx]
    conf[valid] = sc[idx]
    return score, conf


def _rtmlib_pose_activity_track(
    video: str,
    timeline: np.ndarray,
    *,
    fps_a: float,
    cancel_check: Callable[[], None],
    detection_model: str,
    pose_model: str,
    runtime: str,
    pose_device: str,
    court=None,
):
    """Bounded-memory RTMPose activity sampling on YOLO-selected player crops."""
    import cv2

    from .pose import CroppedRTMPose

    timeline = np.asarray(timeline, float)
    score = np.zeros(timeline.size, float)
    confidence_out = np.zeros(timeline.size, float)
    estimator = CroppedRTMPose(
        detection_model=detection_model,
        pose_model=pose_model,
        runtime=runtime,
        pose_device=pose_device,
    )
    cap = cv2.VideoCapture(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_a)
    stride = max(1, int(round(fps / fps_a)))
    sample_times: list[float] = []
    sample_scores: list[float] = []
    sample_confidence: list[float] = []
    pending_times: list[float] = []
    pending_frames: list[np.ndarray] = []

    def flush() -> None:
        if not pending_frames:
            return
        results = estimator.predict(
            pending_frames, court=court, target_required=court is not None,
            confidence=0.2, image_size=960, batch_size=16)
        for sample_time, frame, result in zip(
                pending_times, pending_frames, results):
            value = quality = 0.0
            boxes = result.boxes
            if len(boxes):
                if court is not None:
                    feet = np.column_stack((
                        (boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))
                    coords = np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
                    candidates = [
                        index for index in range(len(boxes))
                        if np.isfinite(coords[index]).all()
                        and coords[index, 1] <= NET_Y
                    ]
                else:
                    height = frame.shape[0]
                    candidates = [
                        index for index, box in enumerate(boxes)
                        if box[3] / height > 0.55
                    ]
                if candidates:
                    selected = max(candidates, key=lambda index: (
                        (boxes[index][2] - boxes[index][0])
                        * (boxes[index][3] - boxes[index][1])))
                    pose = result.keypoints[selected]
                    conf = result.confidence[selected]
                    shoulder_y = float(np.mean([pose[5][1], pose[6][1]]))
                    hip_y = float(np.mean([pose[11][1], pose[12][1]]))
                    wrists = [
                        float(pose[index][1]) for index in (9, 10)
                        if float(conf[index]) > 0.2
                    ]
                    torso = abs(hip_y - shoulder_y) + 1e-6
                    value = float(np.clip(
                        (hip_y - min(wrists)) / torso, 0.0, 1.0)) if wrists else 0.0
                    area = ((boxes[selected][2] - boxes[selected][0])
                            * (boxes[selected][3] - boxes[selected][1])
                            / max(1.0, float(frame.shape[0] * frame.shape[1])))
                    quality = float(np.clip(
                        np.mean(conf[[5, 6, 9, 10, 11, 12]])
                        * np.clip(area * 40.0, 0.0, 1.0), 0.0, 1.0))
            sample_times.append(sample_time)
            sample_scores.append(value)
            sample_confidence.append(quality)
        pending_times.clear()
        pending_frames.clear()

    frame_index = 0
    try:
        while True:
            cancel_check()
            if not cap.grab():
                break
            if frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                pending_times.append(frame_index / fps)
                pending_frames.append(frame)
                if len(pending_frames) >= 16:
                    flush()
            frame_index += 1
        flush()
    finally:
        cap.release()
    if not sample_times:
        return score, confidence_out
    times = np.asarray(sample_times, dtype=float)
    values = np.asarray(sample_scores, dtype=float)
    qualities = np.asarray(sample_confidence, dtype=float)
    half_step = 0.5 / max(float(fps_a), 1e-9)
    valid = (timeline >= times[0] - half_step) & (timeline <= times[-1] + half_step)
    if not np.any(valid):
        return score, confidence_out
    target = timeline[valid]
    if times.size == 1:
        indices = np.zeros(target.size, dtype=int)
    else:
        right = np.clip(np.searchsorted(times, target), 1, times.size - 1)
        left = right - 1
        indices = np.where(
            target - times[left] <= times[right] - target, left, right)
    score[valid] = values[indices]
    confidence_out[valid] = qualities[indices]
    return score, confidence_out


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


def _body_pose_features(pose: np.ndarray, confidence: np.ndarray, cfg):
    """Return ``(usable, ready_stance, overhead_wrist_ratio)`` for COCO-17 joints."""
    body_joints = (5, 6, 11, 12, 13, 14, 15, 16)
    if pose.shape[0] < 17 or confidence.shape[0] < 17:
        return False, False, -1.0
    body_quality = float(np.mean(confidence[list(body_joints)]))
    if body_quality < 0.45:
        return False, False, -1.0
    knee_angles = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if min(
            float(confidence[hip]), float(confidence[knee]),
            float(confidence[ankle]),
        ) > 0.2:
            angle = _joint_angle_deg(pose[hip], pose[knee], pose[ankle])
            if np.isfinite(angle):
                knee_angles.append(angle)
    shoulder_mid = (pose[5] + pose[6]) / 2.0
    hip_mid = (pose[11] + pose[12]) / 2.0
    torso = float(np.linalg.norm(hip_mid - shoulder_mid))
    stance = float("nan")
    if torso > 1e-6 and min(
            float(confidence[15]), float(confidence[16])) > 0.2:
        stance = float(np.linalg.norm(pose[15] - pose[16]) / torso)
    ready = bool(
        (knee_angles and min(knee_angles) <= cfg.match_ready_knee_deg)
        or (np.isfinite(stance) and stance >= cfg.match_ready_stance_ratio)
    )
    ratio = -1.0
    if torso > 1e-6:
        shoulder_y = float((pose[5][1] + pose[6][1]) / 2.0)
        wrist_ratios = [
            (shoulder_y - float(pose[wrist][1])) / torso
            for wrist in (9, 10)
            if float(confidence[wrist]) > 0.2
            and (pose[wrist][0] > 0 or pose[wrist][1] > 0)
        ]
        ratio = max(wrist_ratios, default=-1.0)
    return True, ready, float(ratio)


def observe_serve_setups(
    video: str,
    points: Sequence[Segment],
    onsets: np.ndarray,
    cfg,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] = lambda: None,
    court=None,
    detector: Optional[PlayerDetector] = None,
) -> List[ServeSetupObservation]:  # pragma: no cover - exercised with optional YOLO
    """Observe player setup and service motion around each candidate's early impacts.

    The default backend uses YOLO12 to select target-court player boxes and RTMPose on each
    crop, so both near- and far-side overhead actions can contribute. Match validation still
    pairs pose with independently measured stationary-baseline/ball evidence; a raised arm
    alone is never enough to publish a point.
    """
    import cv2

    ordered_onsets = np.sort(np.asarray(onsets, dtype=float))
    pose_backend = cfg.player_pose_backend
    try:
        if pose_backend == "rtmlib":
            from .pose import CroppedRTMPose

            model = CroppedRTMPose(
                detection_model=cfg.player_detection_model,
                pose_model=cfg.player_pose_model,
                runtime=cfg.rtmpose_runtime,
                pose_device=cfg.rtmpose_device,
                detector=(detector.model if detector is not None and detector.available
                          else None),
                detection_device=(detector.device
                                  if detector is not None and detector.available else None),
            )
            device = model.pose_device
        else:
            from ultralytics import YOLO

            model = YOLO(discover_yolo_weights(cfg.player_pose_model))
            if getattr(model, "task", None) not in {None, "pose"}:
                raise ValueError(
                    f"serve setup requires a pose checkpoint, got {model.task!r}")
            device = resolve_yolo_device()
            model.to(device)
    except Exception as exc:
        default_model = (
            DEFAULT_RTMPOSE_MODEL
            if pose_backend == "rtmlib" else DEFAULT_YOLO_POSE_MODEL)
        explicit_model = cfg.player_pose_model != default_model
        if explicit_model or "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"configured {pose_backend} serve pose model "
                f"{cfg.player_pose_model!r} failed: {exc}"
            ) from exc
        # Pose is optional evidence. Preserve position/TrackNet validation with explicit
        # zero-observation records when a default checkpoint has not been cached yet.
        warnings.warn(
            f"optional {pose_backend} serve pose model "
            f"{cfg.player_pose_model!r} is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        neutral: List[ServeSetupObservation] = []
        for index, (start, end) in enumerate(points, 1):
            strikes = ordered_onsets[
                (ordered_onsets >= start - 1e-9) & (ordered_onsets <= end + 1e-9)]
            neutral.append(ServeSetupObservation(
                point=(float(start), float(end)),
                first_strike=float(strikes[0]) if strikes.size else float(start),
                side=None, side_confidence=0.0, near_x=None, near_x_std=None,
                sampled_frames=0, pose_frames=0, ready_frames=0,
                serve_motion=False, setup_evidence=False, observable=False,
                target_court_filtered=court is not None,
            ))
            if progress_callback is not None:
                progress_callback(index, len(points))
        return neutral
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    observations: List[ServeSetupObservation] = []
    step = 1.0 / float(cfg.match_setup_fps)
    try:
        total = len(points)
        for point_index, point in enumerate(points, 1):
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
            samples: list[tuple[float, np.ndarray]] = []
            # One keyframe seek per candidate, then an ordered scan. Re-seeking for every
            # 100--250 ms sample repeatedly decodes the same GOP and dominates CUDA pose
            # inference on ordinary H.264 match recordings.
            if sample_times.size:
                target_frames = np.maximum(
                    0, np.round(sample_times * native_fps).astype(np.int64))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_frames[0]))
                next_frame = int(target_frames[0])
                for sample_time, target_frame in zip(sample_times, target_frames):
                    cancel_check()
                    ok = True
                    while next_frame <= int(target_frame):
                        ok = cap.grab()
                        if not ok:
                            break
                        next_frame += 1
                        if next_frame % 120 == 0:
                            cancel_check()
                    if not ok:
                        break
                    ok, frame = cap.retrieve()
                    if ok:
                        samples.append((float(sample_time), frame))
            sampled_frames = len(samples)
            frames = [frame for _time, frame in samples]
            if pose_backend == "rtmlib":
                results = model.predict(
                    frames,
                    court=court,
                    target_required=cfg.target_court_required,
                    confidence=0.15,
                    image_size=int(cfg.match_pose_imgsz),
                    batch_size=min(16, len(samples)),
                ) if samples else []
            else:
                results = (model.predict(
                    frames, conf=0.15, classes=[0], verbose=False,
                    imgsz=int(cfg.match_pose_imgsz), device=device,
                    batch=min(16, len(samples)),
                ) if samples else [])
            if len(results) != len(samples):
                raise RuntimeError(
                    "pose model returned a different number of results than sampled frames")
            for (sample_time, frame), result in zip(samples, results):
                height, width = frame.shape[:2]
                if pose_backend == "rtmlib":
                    boxes = result.boxes
                    keypoints = result.keypoints
                    confidence = result.confidence
                else:
                    if result.keypoints is None or len(result.boxes) == 0:
                        continue
                    boxes = result.boxes.xyxy.cpu().numpy()
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidence = (result.keypoints.conf.cpu().numpy()
                                  if result.keypoints.conf is not None else None)
                if len(boxes) == 0:
                    continue
                # RTMLib results are already target-filtered before pose inference. Repeat
                # the same guard for the legacy YOLO-pose backend to keep both paths
                # semantically identical.
                target_required = cfg.target_court_required
                target_indices = (
                    set(target_court_box_indices(boxes, court, frame.shape[:2]))
                    if court is not None else (set() if target_required else None)
                )
                candidates = [
                    i for i, box in enumerate(boxes)
                    if (target_indices is None or i in target_indices)
                    and 0.06 < (box[0] + box[2]) / (2.0 * width) < 0.94
                ]
                if not candidates:
                    continue
                feet = np.column_stack((
                    (boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))
                court_coords = (
                    np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
                    if court is not None else None
                )
                features = {}
                for index in candidates:
                    pose = keypoints[index]
                    conf = (confidence[index] if confidence is not None
                            else np.ones(pose.shape[0], dtype=float))
                    usable, ready, ratio = _body_pose_features(pose, conf, cfg)
                    if usable:
                        features[index] = (ready, ratio)
                if not features:
                    continue
                pose_frames += 1

                # Preserve deuce/ad tracking from the near player while allowing overhead
                # service evidence from either baseline. Court coordinates are more stable
                # than apparent box size when the server is on the far side.
                near_candidates = [
                    index for index in features
                    if (
                        court_coords is not None
                        and np.isfinite(court_coords[index]).all()
                        and court_coords[index, 1] <= NET_Y
                    ) or (
                        court_coords is None and boxes[index][3] / height > 0.55
                    )
                ]
                if near_candidates:
                    selected_near = max(
                        near_candidates,
                        key=lambda index: (
                            (boxes[index][2] - boxes[index][0])
                            * (boxes[index][3] - boxes[index][1])),
                    )
                    if sample_start - 1e-6 <= sample_time <= sample_end + 1e-6:
                        box = boxes[selected_near]
                        xs.append(float((box[0] + box[2]) / (2.0 * width)))
                    ready_frames += int(features[selected_near][0])

                near_early_impact = bool(serve_strikes.size) and any(
                    abs(float(sample_time) - float(strike))
                    <= cfg.match_overhead_window_s + 1e-9
                    for strike in serve_strikes
                )
                if not near_early_impact:
                    continue
                frame_overhead = False
                for index, (_ready, ratio) in features.items():
                    at_baseline = True
                    if court_coords is not None:
                        y = float(court_coords[index, 1])
                        at_baseline = bool(
                            np.isfinite(y)
                            and (y <= cfg.serve_baseline_y_m
                                 or y >= COURT_L - cfg.serve_baseline_y_m)
                        )
                    if not at_baseline:
                        continue
                    overhead_max_ratio = max(overhead_max_ratio, float(ratio))
                    frame_overhead |= ratio >= cfg.match_overhead_wrist_ratio
                if frame_overhead:
                    overhead_frames += 1
                    nearest_strike = min(
                        serve_strikes,
                        key=lambda strike: abs(float(sample_time) - float(strike)),
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
                target_court_filtered=court is not None,
            ))
            if progress_callback is not None:
                progress_callback(point_index, total)
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
                    speed_limit_mps: float = 8.0,
                    detection_model: str = DEFAULT_YOLO_DETECTION_MODEL) -> PlayerCourtTrack:
        # reuse the detector's already-loaded YOLO model rather than loading a second copy
        model = self.detector.model if (self.detector and self.detector.available) else None
        t, cx, cy = track_near_player(
            video, court, fps_a, model=model, model_name=detection_model)
        cx, cy = clean_track(t, cx, cy, speed_limit_mps)
        return PlayerCourtTrack(t, cx, cy, court_speed(t, cx, cy))
