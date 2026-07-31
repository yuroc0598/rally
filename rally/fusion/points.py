"""Point derivation: turn coarse rally regions into tightly-bounded individual points.

:mod:`rally.fusion.decode` produces padded regions whose edges still include
inter-point walking. Live play, though, runs from the serve (first ball strike) to
the last strike; everything else is silent. So we re-bound each region to its actual
strikes — clustering them, splitting at real between-point resets, capturing the
serve, and rejecting strays/echoes — then drop temporally isolated (non-match) points.

Pure numpy in, numbers out: unit-testable without a video.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Segment = Tuple[float, float]  # (start_seconds, end_seconds)


def total_kept_seconds(segments: List[Segment]) -> float:
    return float(sum(e - s for s, e in segments))


def serve_anchor(first_strike: float, onsets: np.ndarray, serve_window_s: float,
                 point_gap_s: float) -> float | None:
    """Find the serve strike that precedes a rally's first strike, if it was split off.

    The serve is an *isolated* strike shortly before the rally: within
    ``serve_window_s`` of the rally's first strike, and itself preceded by silence
    (>= point_gap_s) so it isn't the tail of the previous point. Returns its time, or
    None if no such serve strike exists (then the rally's first strike is the anchor).
    """
    onsets = np.asarray(onsets, dtype=float)
    prev = onsets[(onsets < first_strike - 1e-6) &
                  (onsets >= first_strike - serve_window_s)]
    if prev.size == 0:
        return None
    s = float(prev.max())  # the strike immediately before the rally = serve candidate
    before = onsets[onsets < s - 1e-6]
    if before.size and (s - float(before.max())) < point_gap_s:
        return None  # not isolated -> tail of prior activity/rally, not a serve
    return s


def effective_strike_times(cluster: Sequence[float], echo_s: float) -> List[float]:
    """Return distinct contact times; a gap exactly at ``echo_s`` is still one echo."""
    ordered = sorted(float(value) for value in cluster)
    if not ordered:
        return []
    kept = [ordered[0]]
    for value in ordered[1:]:
        if value - kept[-1] > echo_s:
            kept.append(value)
    return kept


def effective_strikes(cluster: List[float], echo_s: float) -> int:
    """Strike count after folding echo/bounce transients (within echo_s of a counted hit)
    into one. ``last`` advances only when a new distinct hit is counted, so a chain of
    close transients (e.g. [0, 0.2, 0.4, 0.6] with echo_s=0.35) collapses relative to the
    last *counted* strike (-> 2), not merely the previous transient (which would give 1)."""
    return len(effective_strike_times(cluster, echo_s))


def is_coherent_rally(cluster: List[float], min_strikes: int, min_dur_s: float,
                      echo_s: float) -> bool:
    """A real rally: enough distinct strikes over a real span (rejects strays/echoes)."""
    if effective_strikes(cluster, echo_s) < min_strikes:
        return False
    return (cluster[-1] - cluster[0]) >= min_dur_s if len(cluster) > 1 else min_dur_s <= 0


def drop_isolated_points(points: List[Segment], isolation_gap_s: float) -> List[Segment]:
    """Drop points with no other point starting within isolation_gap_s (likely non-match
    incidents / stray warm-up hitting rather than continuous match play)."""
    if len(points) <= 1:
        return list(points)
    starts = [s for s, _ in points]
    out = []
    for i, (s, e) in enumerate(points):
        prev_gap = s - starts[i - 1] if i > 0 else float("inf")
        next_gap = starts[i + 1] - s if i < len(points) - 1 else float("inf")
        if min(prev_gap, next_gap) <= isolation_gap_s:
            out.append((s, e))
    return out


def _cluster_strikes(w: np.ndarray, gap_s: float) -> List[List[float]]:
    """Split a sorted strike-time array into clusters wherever the gap exceeds gap_s."""
    clusters: List[List[float]] = [[float(w[0])]]
    for t in w[1:]:
        if t - clusters[-1][-1] > gap_s:
            clusters.append([float(t)])
        else:
            clusters[-1].append(float(t))
    return clusters


def _coalesce(pts: List[Segment]) -> List[Segment]:
    """Sort and merge any overlapping second-intervals."""
    pts.sort()
    out: List[Segment] = []
    for s, e in pts:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _bound_cluster(cluster: List[float], onsets: np.ndarray, *, gap_s: float,
                   toss_preroll_s: float, landing_tail_s: float, total_s: float,
                   serve_window_s: float) -> Segment | None:
    """Bound one strike cluster to ``[first - toss_preroll, last + landing_tail]``,
    re-anchoring the start to a split-off serve strike when one is found."""
    anchor = cluster[0]
    if serve_window_s > 0:
        s = serve_anchor(cluster[0], onsets, serve_window_s, gap_s)
        if s is not None:
            anchor = s
    start = max(0.0, anchor - toss_preroll_s)
    end = min(total_s, cluster[-1] + landing_tail_s)
    return (start, end) if end > start else None


def points_from_strikes(
    regions: List[Segment],
    onsets: np.ndarray,
    gap_s: float,
    min_strikes: int,
    toss_preroll_s: float,
    landing_tail_s: float,
    total_s: float,
    echo_s: float = 0.0,
    min_dur_s: float = 0.0,
    serve_window_s: float = 0.0,
) -> List[Segment]:
    """Derive tightly-bounded points from the ball-strike times inside each region.

    Cluster the strikes within each detected region — splitting wherever the gap
    between successive strikes exceeds ``gap_s`` — and bound each cluster as
    ``[first_strike - toss_preroll_s, last_strike + landing_tail_s]``. This excludes
    the inter-point walking that padded region edges used to include. Clusters that
    aren't coherent rallies (too few strikes / too short — stray sounds, lets) are
    dropped.
    """
    onsets = np.sort(np.asarray(onsets, dtype=float))
    pts: List[Segment] = []
    for rs, re in regions:
        w = onsets[(onsets >= rs) & (onsets <= re)]
        if w.size == 0:
            continue
        for c in _cluster_strikes(w, gap_s):
            if not is_coherent_rally(c, min_strikes, min_dur_s, echo_s):
                continue
            seg = _bound_cluster(c, onsets, gap_s=gap_s, toss_preroll_s=toss_preroll_s,
                                 landing_tail_s=landing_tail_s, total_s=total_s,
                                 serve_window_s=serve_window_s)
            if seg is not None:
                pts.append(seg)
    return _coalesce(pts)


def points_from_strikes_movement(
    regions: List[Segment],
    onsets: np.ndarray,
    timeline: np.ndarray,
    near_px: np.ndarray,
    near_py: np.ndarray,
    *,
    gap_s: float,
    merge_max_gap_s: float,
    move_thresh: float,
    min_strikes: int,
    toss_preroll_s: float,
    landing_tail_s: float,
    total_s: float,
    echo_s: float = 0.0,
    min_dur_s: float = 0.0,
    serve_window_s: float = 0.0,
) -> List[Segment]:
    """Like :func:`points_from_strikes`, but a short strike-gap only breaks the point
    when the players actually *reset*.

    Within each region we cluster strikes (gap > ``gap_s``) into candidate points, then
    merge two adjacent candidates when the gap between them is short (< ``merge_max_gap_s``)
    **and** the near player barely moved across it (< ``move_thresh``) — i.e. a mid-rally
    lull (a lob), not a genuine between-point reset. Each surviving point is bounded to
    ``[first_strike - toss_preroll_s, last_strike + landing_tail_s]``.
    """
    onsets = np.sort(np.asarray(onsets, dtype=float))
    timeline = np.asarray(timeline, dtype=float)
    near_px = np.asarray(near_px, dtype=float)
    near_py = np.asarray(near_py, dtype=float)
    finite = np.isfinite(near_px)

    def _near_disp(a_last: float, b_first: float):
        """Max displacement of the near player across (a_last, b_first); None if unknown."""
        sel = finite & (timeline >= a_last - 0.5) & (timeline <= b_first + 0.5)
        xs, ys = near_px[sel], near_py[sel]
        if xs.size < 2:
            return None
        d = np.hypot(xs - xs[0], ys - ys[0])
        return float(d.max())

    pts: List[Segment] = []
    for rs, re in regions:
        w = onsets[(onsets >= rs) & (onsets <= re)]
        if w.size == 0:
            continue
        clusters = _cluster_strikes(w, gap_s)
        merged: List[List[float]] = [clusters[0]]
        for c in clusters[1:]:
            gap = c[0] - merged[-1][-1]
            disp = _near_disp(merged[-1][-1], c[0])
            is_lull = gap < merge_max_gap_s and disp is not None and disp < move_thresh
            if is_lull:
                merged[-1] = merged[-1] + c
            else:
                merged.append(c)
        for c in merged:
            if not is_coherent_rally(c, min_strikes, min_dur_s, echo_s):
                continue
            seg = _bound_cluster(c, onsets, gap_s=gap_s, toss_preroll_s=toss_preroll_s,
                                 landing_tail_s=landing_tail_s, total_s=total_s,
                                 serve_window_s=serve_window_s)
            if seg is not None:
                pts.append(seg)
    return _coalesce(pts)


def snap_serve_starts(
    segments: List[Segment],
    onsets: np.ndarray,
    lookback_s: float,
    preroll_s: float,
) -> List[Segment]:
    """Extend each segment's start back to the point's first strike (the serve).

    Used only on the ``--no-split`` fallback path (the default path bounds points to
    their strikes directly via :func:`points_from_strikes`). Rally scoring fires once
    a rhythm is established, so the decoded start sits after the serve; look back up to
    ``lookback_s`` for the earliest strike of this point and move the start to
    ``that_onset - preroll_s`` (never later than the original, never overlapping the
    previous segment, never below zero).
    """
    onsets = np.sort(np.asarray(onsets, dtype=float))
    out: List[Segment] = []
    prev_end = 0.0
    for s, e in segments:
        new_start = s
        if onsets.size:
            # don't look back past the previous point, so we snap to this point's serve
            # rather than the previous point's last strike (which would cost the preroll).
            lo = max(s - lookback_s, prev_end)
            window = onsets[(onsets >= lo) & (onsets <= s + 1.0)]
            if window.size:
                new_start = min(s, float(window.min()) - preroll_s)
        new_start = max(0.0, new_start, prev_end)
        out.append((new_start, e))
        prev_end = e
    return out
