"""Ball-as-arbiter: let the ball trajectory *decide* which candidate windows are real
rallies and where each one starts and ends.

This is the SwingVision-style inversion of the pipeline. The cheap channels (audio /
motion / geometry) are recall-oriented — they over-propose candidate windows, including
warm-up cooperative hitting, adjacent-court noise, and crowd transients. The ball tracker
is expensive but *discriminative*: during a real point the ball is continuously in play,
crosses the net, and bounces; between/around points it is dead. So we run the tracker only
inside each candidate (CPU-affordable), reconstruct the trajectory
(:mod:`rally.signals.trajectory`), and apply a verdict:

* **is it a rally?** — the ball must be in play for a meaningful span AND show rally
  structure (a net crossing and/or bounces). This rejects false positives that pure audio
  cannot.
* **where does it start?** — snap to the serve: the first sustained in-play instant, minus
  a short toss pre-roll.
* **where does it end?** — the first point-ending event (double bounce on one side / ball
  out), via :mod:`rally.signals.ballrules`, else the last live-ball instant plus a tail.

The verdict logic (:func:`rally_verdict`) is pure — it takes a reconstructed
:class:`~rally.signals.trajectory.SmoothTrack` (+ optional court homography) and returns a
:class:`RallyVerdict` — so it is unit-testable on synthetic trajectories with no video,
model, or weights. :func:`verify_segments` is the orchestrator that does the tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..signals.ballrules import point_end_events, refine_end_from_events
from ..signals.court import NET_Y
from ..signals.trajectory import SmoothTrack, bounces_from_velocity, smooth_track

Segment = Tuple[float, float]


@dataclass
class RallyVerdict:
    is_rally: bool
    start: float
    end: float
    n_bounces: int = 0
    n_net_crossings: int = 0
    in_play_frac: float = 0.0
    in_play_span_s: float = 0.0
    reason: str = ""


def _in_play_mask(track: SmoothTrack, min_speed_px_s: float, min_conf: float) -> np.ndarray:
    """Per-sample "the ball is a live, moving ball" — confident detection + real motion."""
    speed = np.hypot(track.vx, track.vy)
    return track.measured & (track.confidence >= min_conf) & (speed >= min_speed_px_s)


def _net_crossings(track: SmoothTrack, court, in_play: np.ndarray,
                   dead_band_m: float = 0.5) -> int:
    """Count times the ball crosses the net, in court metres.

    The ball's court-y is compared to the net line with a dead-band so jitter around the
    net doesn't inflate the count; only a clean side-to-side transition among in-play
    samples is a crossing. Returns 0 without a court (can't map to metres).
    """
    if court is None:
        return 0
    idx = np.where(in_play)[0]
    if idx.size < 2:
        return 0
    pts = np.stack([track.x[idx], track.y[idx]], axis=1)
    cy = court.to_court(pts)[:, 1]
    side = 0            # -1 near, +1 far, 0 unknown (inside dead-band)
    crossings = 0
    for v in cy:
        s = -1 if v < NET_Y - dead_band_m else (1 if v > NET_Y + dead_band_m else 0)
        if s == 0:
            continue
        if side != 0 and s != side:
            crossings += 1
        side = s
    return crossings


def rally_verdict(
    track: SmoothTrack,
    court,
    win_start: float,
    win_end: float,
    *,
    min_speed_px_s: float = 25.0,
    min_conf: float = 0.3,
    min_in_play_frac: float = 0.35,
    min_in_play_span_s: float = 1.5,
    min_bounces: int = 2,
    min_rally_s: float = 1.5,
    toss_preroll_s: float = 1.0,
    tail_s: float = 0.8,
    max_extend_s: float = 3.0,
    bounce_min_descent_px_s: float = 40.0,
    double_bounce_window_s: float = 2.5,
    margin_m: float = 0.35,
) -> RallyVerdict:
    """Decide whether a reconstructed trajectory is a real rally, and bound it.

    ``win_start``/``win_end`` are the candidate's original (pre-ball) bounds; the returned
    start/end are snapped to the ball's serve and point-end within the tracked window.

    A window is a rally iff the ball is in play for at least ``min_in_play_span_s`` and a
    fraction ``min_in_play_frac`` of the window, AND shows rally structure — at least one
    net crossing (needs a court) or ``min_bounces`` bounces. Without a court, the decision
    falls back to in-play span + bounce count (no net/out geometry available).
    """
    t = track.t
    if t.size < 3:
        return RallyVerdict(False, win_start, win_end, reason="track too short")

    in_play = _in_play_mask(track, min_speed_px_s, min_conf)
    in_play_frac = float(in_play.mean())
    live_idx = np.where(in_play)[0]
    if live_idx.size < 2:
        return RallyVerdict(False, win_start, win_end, in_play_frac=in_play_frac,
                            reason="ball never in play")
    span = float(t[live_idx[-1]] - t[live_idx[0]])

    bounce_idx = bounces_from_velocity(track, min_descent_px_s=bounce_min_descent_px_s,
                                       min_conf=min_conf)
    n_bounces = len(bounce_idx)
    n_cross = _net_crossings(track, court, in_play)

    has_structure = (n_cross >= 1) or (n_bounces >= min_bounces)
    is_rally = (in_play_frac >= min_in_play_frac and span >= min_in_play_span_s
                and has_structure)
    if not is_rally:
        return RallyVerdict(False, win_start, win_end, n_bounces=n_bounces,
                            n_net_crossings=n_cross, in_play_frac=in_play_frac,
                            in_play_span_s=span, reason="no live-ball rally structure")

    # ---- boundaries: serve start, point-end (or live-ball tail) --------------
    start = max(win_start, float(t[live_idx[0]]) - toss_preroll_s)
    end = min(win_end + max_extend_s, float(t[live_idx[-1]]) + tail_s)

    if court is not None and bounce_idx:
        bounces = []
        for i in bounce_idx:
            cx, cy = court.to_court([[float(track.x[i]), float(track.y[i])]])[0]
            bounces.append((float(t[i]), float(cx), float(cy)))
        events = point_end_events(bounces, double_bounce_window_s=double_bounce_window_s,
                                  margin_m=margin_m)
        # refine_end_from_events caps at end + max_extend_s; since `end` already includes
        # max_extend_s this is nominally 2x, but it's bounded anyway — no bounce exists past
        # the tracked window [s - pre_pad_s, e + post_pad_s], so nothing extends that far.
        new_end, reason = refine_end_from_events(start, end, events, min_rally_s=min_rally_s,
                                                 tail_s=tail_s, max_extend_s=max_extend_s)
        if reason is not None:
            end = new_end

    end = max(end, start + min_rally_s)
    return RallyVerdict(True, start, end, n_bounces=n_bounces, n_net_crossings=n_cross,
                        in_play_frac=in_play_frac, in_play_span_s=span, reason="rally")


def _dedupe_non_overlapping(segments: List[Segment]) -> List[Segment]:
    """Sort and remove overlaps, preserving each segment's (serve) start where possible."""
    out: List[Segment] = []
    for s, e in sorted(segments):
        if e <= s:
            continue
        if out:
            ps, pe = out[-1]
            if s < pe:                       # overlap: trim previous end back to this start
                if s > ps:
                    out[-1] = (ps, s)
                else:
                    out.pop()
        out.append((s, e))
    return out


def verify_segments(
    video: str,
    segments: List[Segment],
    *,
    court=None,
    weights_path: Optional[str] = None,
    model=None,
    pre_pad_s: float = 2.0,
    post_pad_s: float = 2.0,
    max_extend_s: float = 3.0,
    smooth_max_gap_s: float = 0.5,
    verdict_kwargs: Optional[dict] = None,
    progress: Callable[[str], None] = lambda _m: None,
) -> List[Segment]:
    """Track the ball inside each candidate window, then keep+bound the real rallies.

    Needs a TrackNet ``weights_path`` (or a preloaded ``model``). ``court`` is optional but
    strongly recommended — without it the verdict can't use net-crossing / in-out geometry
    and leans on in-play span + bounce count alone.
    """
    from ..signals.ball import load_ball_model, track_tracknet

    if model is None:
        if not weights_path:
            raise ValueError("verify_segments needs a TrackNet weights_path or model")
        model = load_ball_model(weights_path)

    verdict_kwargs = dict(verdict_kwargs or {})
    verdict_kwargs.setdefault("max_extend_s", max_extend_s)

    kept: List[Segment] = []
    n_reject = 0
    for s, e in segments:
        track = track_tracknet(video, model=model,
                               start_s=max(0.0, s - pre_pad_s), end_s=e + post_pad_s)
        st = smooth_track(track, max_gap_s=smooth_max_gap_s)
        v = rally_verdict(st, court, s, e, **verdict_kwargs)
        if v.is_rally:
            kept.append((v.start, v.end))
        else:
            n_reject += 1
    progress(f"  ball arbiter: kept {len(kept)}/{len(segments)} candidates "
             f"({n_reject} rejected as non-rallies)")
    return _dedupe_non_overlapping(kept)
