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

Pose is intentionally not used: it was validated as unreliable on wide/night footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Person = Tuple[float, float, float]          # (foot_x_norm, foot_y_norm, box_area_norm)
Region = Tuple[float, float, float, float]   # (x0, y0, x1, y1) normalized
Segment = Tuple[float, float]


# ---------------------------------------------------------------------------
# detection + geometry channel
# ---------------------------------------------------------------------------
class PlayerDetector:
    """Wraps a YOLO person detector. ``available`` is False when YOLO is missing."""

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.3):
        self.conf = conf
        self.model = None
        try:  # pragma: no cover - optional heavy dependency
            from ultralytics import YOLO

            self.model = YOLO(model)
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
        res = self.model.predict(frame_bgr, conf=self.conf, classes=[0], verbose=False)
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
        model = YOLO("yolov8n.pt")
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
                r = model.predict(fr, conf=0.3, classes=[0], verbose=False)[0]
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
def pose_activity_track(video: str, timeline: np.ndarray, fps_a: float = 2.0):
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
        model = YOLO("yolov8n-pose.pt")
    except Exception:
        return score, conf

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_a
    stride = max(1, int(round(fps / fps_a)))
    samp_t, samp_s, samp_c = [], [], []
    fi = 0
    try:
        while True:
            if not cap.grab():
                break
            if fi % stride == 0:
                ok, fr = cap.retrieve()
                if not ok:
                    break
                h, w = fr.shape[:2]
                r = model.predict(fr, conf=0.3, classes=[0], verbose=False)[0]
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
