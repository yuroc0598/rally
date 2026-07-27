"""Wire ball tracking into point trimming: trim each rally's END to the ball's
point-ending event (double bounce on one side / out of bounds).

Opt-in — needs a TrackNet weights file and a court calibration. Ball tracking is
CPU-expensive (~0.3 s/frame), so this runs the tracker only over each kept rally
segment (not the whole video); a GPU is recommended for full matches.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..signals.ball import load_ball_model, track_tracknet
from ..signals.ballrules import bounces_in_court, point_end_events, refine_end_from_events

Segment = Tuple[float, float]


def refine_ends_with_ball(
    video: str,
    segments: List[Segment],
    court,
    weights_path: str,
    *,
    min_rally_s: float = 1.5,
    tail_s: float = 0.8,
    max_extend_s: float = 3.0,
    bounce_prominence_px: float = 6.0,
    double_bounce_window_s: float = 2.5,
    margin_m: float = 0.35,
    progress: Callable[[str], None] = lambda _m: None,
) -> List[Segment]:
    """For each rally, track the ball over the segment, detect bounces, and move the end
    to the first point-ending event. Non-overlapping; keeps original end if none found."""
    model = load_ball_model(weights_path)
    out: List[Segment] = []
    n_trimmed = 0
    for s, e in segments:
        bt = track_tracknet(video, model=model, start_s=s, end_s=e + max_extend_s)
        bounces = bounces_in_court(bt.t, bt.x, bt.y, court, prominence_px=bounce_prominence_px)
        events = point_end_events(bounces, double_bounce_window_s=double_bounce_window_s,
                                  margin_m=margin_m)
        new_e, reason = refine_end_from_events(s, e, events, min_rally_s=min_rally_s,
                                               tail_s=tail_s, max_extend_s=max_extend_s)
        if reason is not None and abs(new_e - e) > 0.3:
            n_trimmed += 1
        out.append((s, max(new_e, s + min_rally_s)))
    progress(f"  ball point-end refined {n_trimmed}/{len(segments)} rally ends")
    # keep sorted & non-overlapping. On overlap, preserve THIS segment's start (it may hold
    # a serve toss captured upstream) by trimming the previous segment's extended end back,
    # rather than shoving this start forward.
    out.sort()
    clean: List[Segment] = []
    for s, e in out:
        if clean:
            ps, pe = clean[-1]
            if s < pe:
                if s > ps:
                    clean[-1] = (ps, s)   # trim previous end to this start
                else:
                    clean.pop()           # this segment subsumes the previous one
        if e > s:
            clean.append((s, e))
    return clean
