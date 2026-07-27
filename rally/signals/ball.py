"""Independent ball-tracking module: video -> per-frame ball position.

Two backends behind one interface:

* ``tracknet`` — the purpose-built heatmap CNN (the reliable choice). It needs a
  pretrained weights file (``weights_path``); the architecture here follows the
  TrackNet design (stacked consecutive frames -> Gaussian heatmap -> peak = ball).
* ``motion`` — a training-free classical detector (3-frame differencing + small
  fast-blob + speed-limited linking). Runs anywhere, but low recall on small/blurry
  balls (verified on this night footage) — use it as a fallback / for well-lit,
  higher-resolution clips.

The output is a :class:`BallTrack` (time, x, y image px, visibility). Feed it to
``rally.signals.ballrules`` for bounce / in-out / point-end / speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BallTrack:
    t: np.ndarray          # timestamps (s)
    x: np.ndarray          # image x (px), NaN where not detected
    y: np.ndarray          # image y (px), NaN where not detected

    @property
    def visible(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    def detection_rate(self) -> float:
        return float(self.visible.mean()) if self.t.size else 0.0


# ---------------------------------------------------------------------------
# classical motion backend (runs without weights)
# ---------------------------------------------------------------------------
def track_motion(video: str, fps_a: Optional[float] = None,
                 min_area: int = 4, max_area: int = 400,
                 diff_thresh: int = 22, speed_limit_px: float = 250.0) -> BallTrack:
    """Track the ball by 3-frame differencing + small fast-blob + trajectory linking.

    Processes at native fps (fast motion needs it). ``speed_limit_px`` rejects blobs
    that jump implausibly far from the previous ball position (players, noise).
    """
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_gray = []
    ts, xs, ys = [], [], []
    prev_ball = None

    ok, f0 = cap.read()
    ok2, f1 = cap.read()
    if not (ok and ok2):
        cap.release()
        return BallTrack(np.array([]), np.array([]), np.array([]))
    g_prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g_cur = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    idx = 1
    while True:
        ok, f2 = cap.read()
        if not ok:
            break
        g_next = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        d1 = cv2.threshold(cv2.absdiff(g_cur, g_prev), diff_thresh, 255, cv2.THRESH_BINARY)[1]
        d2 = cv2.threshold(cv2.absdiff(g_next, g_cur), diff_thresh, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(cv2.bitwise_and(d1, d2), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            if min_area < a < max_area and 0.4 < bw / max(bh, 1) < 2.6:
                cands.append((x + bw / 2.0, y + bh / 2.0, a))
        best = None
        if cands:
            if prev_ball is not None:
                near = [(cx, cy, a) for cx, cy, a in cands
                        if np.hypot(cx - prev_ball[0], cy - prev_ball[1]) <= speed_limit_px]
                pool = near or cands
                best = min(pool, key=lambda z: np.hypot(z[0] - prev_ball[0], z[1] - prev_ball[1]))
            else:
                best = max(cands, key=lambda z: z[2])
        ts.append(idx / fps)
        if best is not None:
            xs.append(best[0]); ys.append(best[1]); prev_ball = best
        else:
            xs.append(np.nan); ys.append(np.nan)
        g_prev, g_cur = g_cur, g_next
        idx += 1
    cap.release()
    return BallTrack(np.array(ts), np.array(xs), np.array(ys))


# ---------------------------------------------------------------------------
# TrackNet backend (3-frame heatmap CNN, PyTorch; needs pretrained weights)
# ---------------------------------------------------------------------------
def _decode_heatmap(feature_map_360x640: np.ndarray, prev=None,
                    speed_limit_px: float = 200.0, scale: int = 1):
    import cv2
    hm = feature_map_360x640.astype(np.uint8)
    _, binm = cv2.threshold(hm, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(binm, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    if circles is None:
        return None
    cands = [(c[0] * scale, c[1] * scale) for c in circles[0]]
    if prev is not None:
        near = [c for c in cands if np.hypot(c[0] - prev[0], c[1] - prev[1]) <= speed_limit_px]
        cands = near or cands
        return min(cands, key=lambda c: np.hypot(c[0] - prev[0], c[1] - prev[1]))
    return cands[0]


def discover_ball_weights(models_dir: str = "models") -> Optional[str]:
    """Find a bundled TrackNet checkpoint so ball mode works with no explicit path.

    Looks for ``models/tracknet*.pt`` first (the expected name), then any ``models/*.pt``
    that isn't a YOLO file. Returns the path or ``None`` if none is present.
    """
    import glob
    import os

    for pat in ("tracknet*.pt", "*tracknet*.pt", "*.pt"):
        for path in sorted(glob.glob(os.path.join(models_dir, pat))):
            if "yolo" in os.path.basename(path).lower():
                continue
            return path
    return None


def load_ball_model(weights_path: str):
    """Load the 3-frame PyTorch TrackNet once (reuse across segments)."""
    import torch

    from ..vendor.tracknet_torch import BallTrackerNet

    model = BallTrackerNet()
    # weights_only=True: the checkpoint is a plain tensor state-dict, so refuse to
    # unpickle arbitrary objects (avoids code execution from a tampered .pt file).
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    model.load_state_dict(state)
    model.eval()
    return model


def track_tracknet(video: str, weights_path: Optional[str] = None, *, model=None,
                   start_s: float = 0.0, end_s: Optional[float] = None,
                   width: int = 640, height: int = 360, speed_limit_px: float = 200.0) -> BallTrack:
    """Ball positions per frame via the 3-frame PyTorch TrackNet (BallTrackerNet).

    Three consecutive frames are stacked (9 channels) so the net sees the ball's motion —
    far more reliable for a small/blurry ball than any single-frame detector. Optionally
    restrict to ``[start_s, end_s]`` (to process one rally) and pass a preloaded ``model``.
    Pure PyTorch — no TensorFlow.
    """
    import cv2
    import torch

    if model is None:
        model = load_ball_model(weights_path)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(0, int(round((start_s - 2 / fps) * fps)))  # 2 frames of lead for the 3-stack
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    end_f = None if end_s is None else int(round(end_s * fps))
    scale_x = scale_y = None
    buf, ts, xs, ys = [], [], [], []
    prev = None
    fi = start_f
    with torch.no_grad():
        while True:
            ok, fr = cap.read()
            if not ok or (end_f is not None and fi > end_f):
                break
            if scale_x is None:
                scale_x, scale_y = fr.shape[1] / width, fr.shape[0] / height
            buf.append(cv2.resize(fr, (width, height)))
            if len(buf) > 3:
                buf.pop(0)
            ts.append(fi / fps)
            pos = None
            if len(buf) == 3:
                imgs = np.concatenate((buf[2], buf[1], buf[0]), axis=2).astype(np.float32) / 255.0
                inp = torch.from_numpy(np.rollaxis(imgs, 2, 0)[None]).float()
                fm = model(inp).argmax(dim=1).cpu().numpy().reshape((height, width))
                pv = None if prev is None else (prev[0] / scale_x, prev[1] / scale_y)
                pos = _decode_heatmap(fm, prev=pv, speed_limit_px=speed_limit_px / max(scale_x, 1))
                if pos is not None:
                    pos = (pos[0] * scale_x, pos[1] * scale_y)
            if pos is not None:
                xs.append(pos[0]); ys.append(pos[1]); prev = pos
            else:
                xs.append(np.nan); ys.append(np.nan)
            fi += 1
    cap.release()
    return BallTrack(np.array(ts), np.array(xs), np.array(ys))


def ball_in_play_channel(track: BallTrack, timeline: np.ndarray,
                         window_s: float = 1.0, min_speed_px: float = 3.0) -> np.ndarray:
    """Per-analysis-frame ball-in-play evidence (0..1) for the rally fusion.

    For each timeline time, the fraction of nearby tracked frames where the ball is both
    detected and moving (a live ball) — high during a rally, ~0 between points.
    """
    timeline = np.asarray(timeline, float)
    out = np.zeros(timeline.size, float)
    if track.t.size < 2:
        return out
    vis = track.visible
    sp = np.zeros(track.t.size)
    for i in range(1, track.t.size):
        if vis[i] and vis[i - 1]:
            dt = max(track.t[i] - track.t[i - 1], 1e-6)
            sp[i] = np.hypot(track.x[i] - track.x[i - 1], track.y[i] - track.y[i - 1]) / dt
    active = vis & (sp > min_speed_px)
    for k, t in enumerate(timeline):
        m = (track.t >= t - window_s / 2) & (track.t <= t + window_s / 2)
        if m.any():
            out[k] = float(active[m].mean())
    return out


class BallTracker:
    """Unified ball tracker. ``backend='tracknet'`` (3-frame PyTorch net + weights) is
    the reliable choice; ``backend='motion'`` runs anywhere but with low recall."""

    def __init__(self, backend: str = "motion", weights_path: Optional[str] = None):
        self.backend = backend
        self.weights_path = weights_path

    def track(self, video: str, **kw) -> BallTrack:
        if self.backend == "tracknet":
            if not self.weights_path:
                raise ValueError("tracknet backend needs weights_path (a TrackNet .pt)")
            return track_tracknet(video, self.weights_path, **kw)
        return track_motion(video, **kw)
