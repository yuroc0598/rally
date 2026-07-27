"""Rules derived from a ball trajectory: bounces, in/out, point-end, and speed.

These are pure functions over a ball track (+ optional court homography), independent of
how the track was produced (TrackNet or motion). They implement the common tennis rules
the user asked for:

* a **bounce** shows up as a local maximum of the ball's image-y (the ball reaches its
  lowest point on screen — ground contact — then rebounds);
* mapping a bounce through the court homography gives its court-metre landing position
  (valid at ground contact), so we can decide **in/out** and **which side**;
* a **point ends** when the ball bounces twice on the same side (second bounce) or lands
  out of bounds;
* **ball speed** is the court-plane displacement between samples (approximate — a single
  camera can't recover height, so this under-reads on high balls).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .court import COURT_L, DOUBLES_W, NET_Y, SINGLES_IN

Bounce = Tuple[float, float, float]   # (time_s, court_x_m, court_y_m)
Event = Tuple[float, str]             # (time_s, reason)


def detect_bounces(t: np.ndarray, y: np.ndarray, prominence_px: float = 8.0,
                   min_sep_s: float = 0.3, max_gap_s: float = 0.5) -> List[int]:
    """Bounce sample-indices = local maxima of image-y (screen-lowest points).

    NaN gaps are interpolated first. ``prominence_px`` filters shallow wobble; ``min_sep_s``
    is the refractory spacing between bounces. Peaks that fall inside a long interpolated
    gap (no real detection within ``max_gap_s``) are rejected — a straight line drawn
    across a long dropout can fabricate a peak or hide a real one.
    """
    from scipy.signal import find_peaks

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 3:
        return []
    idx = np.arange(t.size)
    good = np.isfinite(y)
    if good.sum() < 3:
        return []
    yf = np.interp(idx, idx[good], y[good])
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    distance = max(1, int(round(min_sep_s / max(dt, 1e-6))))
    peaks, _ = find_peaks(yf, prominence=prominence_px, distance=distance)
    # drop peaks that sit too far from any real (non-interpolated) sample
    max_gap = max(1, int(round(max_gap_s / max(dt, 1e-6))))
    good_idx = idx[good]
    out = []
    for p in peaks:
        j = int(np.searchsorted(good_idx, p))
        neighbours = [good_idx[j - 1] for _ in (0,) if j > 0] + \
                     [good_idx[j] for _ in (0,) if j < good_idx.size]
        if neighbours and min(abs(int(p) - int(g)) for g in neighbours) <= max_gap:
            out.append(int(p))
    return out


def bounces_in_court(t: np.ndarray, x: np.ndarray, y: np.ndarray, court,
                     **kw) -> List[Bounce]:
    """Detected bounces as (time, court_x, court_y) via the homography."""
    idxs = detect_bounces(t, y, **kw)
    out: List[Bounce] = []
    for i in idxs:
        if np.isfinite(x[i]) and np.isfinite(y[i]):
            cx, cy = court.to_court([[float(x[i]), float(y[i])]])[0]
            out.append((float(t[i]), float(cx), float(cy)))
    return out


def is_in(court_x: float, court_y: float, margin_m: float = 0.35,
          singles: bool = True) -> bool:
    """Is a court-coordinate landing inside the playing area (+ line-call margin)?"""
    x0 = (SINGLES_IN if singles else 0.0) - margin_m
    x1 = (DOUBLES_W - SINGLES_IN if singles else DOUBLES_W) + margin_m
    return (x0 <= court_x <= x1) and (-margin_m <= court_y <= COURT_L + margin_m)


def side_of(court_y: float) -> str:
    return "near" if court_y < NET_Y else "far"


def point_end_events(bounces: List[Bounce], *, double_bounce_window_s: float = 2.5,
                     margin_m: float = 0.35, singles: bool = True) -> List[Event]:
    """Point-ending events from the bounce sequence:

    * ``out``           — a bounce lands outside the court.
    * ``double_bounce`` — two consecutive bounces on the same side (the ball bounced
      twice before being returned), within ``double_bounce_window_s``.
    """
    events: List[Event] = []
    out_times = set()
    for tb, cx, cy in bounces:
        if not is_in(cx, cy, margin_m, singles):
            events.append((tb, "out"))
            out_times.add(tb)
    for i in range(1, len(bounces)):
        (t0, _, y0), (t1, _, y1) = bounces[i - 1], bounces[i]
        if t1 in out_times:
            continue  # this bounce already ends the point as 'out' (don't double-count)
        if t1 - t0 <= double_bounce_window_s and side_of(y0) == side_of(y1):
            events.append((t1, "double_bounce"))
    return sorted(events)


def first_point_end_after(bounces: List[Bounce], start_s: float, **kw) -> Optional[Event]:
    """The first point-ending event at/after ``start_s`` (for trimming a rally's end)."""
    for ev in point_end_events(bounces, **kw):
        if ev[0] >= start_s:
            return ev
    return None


def refine_end_from_events(start: float, end: float, events: List[Event],
                           min_rally_s: float = 1.5, tail_s: float = 0.8,
                           max_extend_s: float = 3.0) -> Tuple[float, Optional[str]]:
    """Trim a point's end to when the ball actually ended it (double bounce / out).

    Returns the first point-ending event at/after ``start + min_rally_s`` (so a serve
    bounce doesn't end the point) and within ``end + max_extend_s``, as
    ``(event_time + tail_s, reason)``; otherwise the original ``end`` and None.
    """
    for tb, reason in sorted(events):
        if tb >= start + min_rally_s and tb <= end + max_extend_s:
            return min(end + max_extend_s, tb + tail_s), reason
    return end, None


def ball_speed_kmh(t: np.ndarray, x: np.ndarray, y: np.ndarray, court,
                   smooth: int = 3, max_kmh: float = 250.0) -> np.ndarray:
    """Approximate ball speed (km/h) from court-plane displacement between samples.

    Single-camera + ground homography can't recover height, so this under-reads on high
    balls and is most meaningful just after a strike when the ball is low. Per-sample
    speeds above ``max_kmh`` are treated as detection noise (a ball can't exceed the
    fastest-ever serve ~260 km/h) and dropped to NaN — without this a single jumpy
    detection yields absurd values.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = t.size
    sp = np.full(n, np.nan)
    pts = np.full((n, 2), np.nan)
    vis = np.isfinite(x) & np.isfinite(y)
    if vis.any():
        cc = court.to_court(np.stack([x[vis], y[vis]], 1))
        pts[vis] = cc
    for i in range(1, n):
        if np.isfinite(pts[i, 0]) and np.isfinite(pts[i - 1, 0]):
            dt = max(t[i] - t[i - 1], 1e-6)
            v = np.hypot(*(pts[i] - pts[i - 1])) / dt * 3.6
            sp[i] = v if v <= max_kmh else np.nan   # reject implausible jumps
    if smooth > 1 and n:
        # NaN-aware median: each finite sample is smoothed over its finite neighbours only.
        # (A zero-filled medfilt would pull speeds near gaps toward zero.)
        k = smooth + (smooth + 1) % 2
        half = k // 2
        sm = np.full(n, np.nan)
        for i in range(n):
            if not np.isfinite(sp[i]):
                continue
            w = sp[max(0, i - half):i + half + 1]
            w = w[np.isfinite(w)]
            if w.size:
                sm[i] = float(np.median(w))
        sp = sm
    return sp
