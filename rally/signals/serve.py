"""Point-level serve evidence from a purpose-built ball track.

Audio proposes candidate windows; it cannot tell a serve from a ball handoff, bounce, or
footstep. In match video, TrackNet supplies the missing event evidence: around one of the
first few impact candidates, the ball must be tracked persistently inside the court and
travel vertically enough to represent a toss/serve flight rather than a short lateral feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from .ball import BallTrack

Segment = Tuple[float, float]


@dataclass(frozen=True)
class BallServeObservation:
    point: Segment
    checked: bool
    confirmed: bool
    best_strike: float | None
    coverage: float
    vertical_span: float
    outgoing_span: float
    ordered: bool
    measured_samples: int


def classify_ball_serve(
    point: Segment,
    strikes: np.ndarray,
    track: BallTrack,
    frame_width: int,
    frame_height: int,
    cfg,
) -> BallServeObservation:
    """Classify serve-like ball motion around the first few candidate impacts.

    Coordinates are normalized before thresholding so the rule is resolution independent.
    Coverage prevents a single TrackNet hallucination from becoming evidence; vertical span
    rejects net-player feeds and horizontal handoffs. Underarm serves are still supported
    because their court-depth flight has substantial image-y travel even without a toss.
    """
    point = (float(point[0]), float(point[1]))
    strikes = np.sort(np.asarray(strikes, dtype=float))
    strikes = strikes[(strikes >= point[0] - 1e-9) & (strikes <= point[1] + 1e-9)]
    strikes = strikes[: int(cfg.match_serve_strikes_to_check)]
    if not strikes.size or frame_width <= 0 or frame_height <= 0 or track.t.size == 0:
        return BallServeObservation(
            point, True, False, None, 0.0, 0.0, 0.0, False, 0)

    x0, x1 = cfg.match_ball_court_x
    y0, y1 = cfg.match_ball_court_y
    xn = np.asarray(track.x, float) / float(frame_width)
    yn = np.asarray(track.y, float) / float(frame_height)
    visible = np.isfinite(xn) & np.isfinite(yn)
    in_court = visible & (xn >= x0) & (xn <= x1) & (yn >= y0) & (yn <= y1)

    best = (0.0, 0.0, 0, None, 0.0, 0.0, False)
    for strike in strikes:
        window = ((track.t >= float(strike) - cfg.match_ball_serve_pre_s)
                  & (track.t <= float(strike) + cfg.match_ball_serve_post_s))
        samples = int(np.sum(window & in_court))
        coverage = float(samples / max(1, int(np.sum(window))))
        vertical_span = float(np.ptp(yn[window & in_court])) if samples else 0.0
        before = (
            (track.t >= float(strike) - cfg.match_ball_serve_pre_s)
            & (track.t <= float(strike) + 0.05) & in_court
        )
        after = (
            (track.t >= float(strike) - 0.05)
            & (track.t <= float(strike) + cfg.match_ball_serve_post_s) & in_court
        )
        after_idx = np.flatnonzero(after)
        outgoing_span = 0.0
        if after_idx.size >= 2:
            # Robust endpoint displacement: medians suppress one-frame heatmap jumps.
            width_n = max(1, after_idx.size // 3)
            first_idx = after_idx[:width_n]
            last_idx = after_idx[-width_n:]
            first_xy = np.array([
                float(np.median(xn[first_idx])), float(np.median(yn[first_idx]))])
            last_xy = np.array([
                float(np.median(xn[last_idx])), float(np.median(yn[last_idx]))])
            outgoing_span = float(np.linalg.norm(last_xy - first_xy))
        ordered = bool(
            int(np.sum(before)) >= 2
            and after_idx.size >= 2
            and outgoing_span >= cfg.match_ball_min_outgoing_span
        )
        # Lexicographic score first rewards jointly clearing both required dimensions.
        score = min(1.0, coverage / max(cfg.match_ball_min_coverage, 1e-9)) * min(
            1.0, vertical_span / max(cfg.match_ball_min_vertical_span, 1e-9)) * float(ordered)
        candidate = (
            score, coverage, samples, float(strike), vertical_span,
            outgoing_span, ordered)
        if candidate[:3] > best[:3]:
            best = candidate

    _score, coverage, samples, best_strike, vertical_span, outgoing_span, ordered = best
    confirmed = (coverage >= cfg.match_ball_min_coverage
                 and vertical_span >= cfg.match_ball_min_vertical_span
                 and ordered)
    return BallServeObservation(
        point, True, bool(confirmed), best_strike,
        float(coverage), float(vertical_span), float(outgoing_span),
        bool(ordered), int(samples),
    )


def observe_ball_serves(
    video: str,
    points: Sequence[Segment],
    onsets: np.ndarray,
    weights_path: str,
    cfg,
    cancel_check: Callable[[], None] = lambda: None,
) -> list[BallServeObservation]:  # pragma: no cover - heavy model integration
    """Track only short early-point windows, loading TrackNet once for all candidates."""
    import cv2

    from .ball import load_ball_model, track_tracknet

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("could not determine frame size for serve validation")

    ordered = np.sort(np.asarray(onsets, dtype=float))
    model = load_ball_model(weights_path)
    observations: list[BallServeObservation] = []
    for point in points:
        cancel_check()
        strikes = ordered[(ordered >= point[0] - 1e-9) & (ordered <= point[1] + 1e-9)]
        considered = strikes[: int(cfg.match_serve_strikes_to_check)]
        if not considered.size:
            observations.append(BallServeObservation(
                (float(point[0]), float(point[1])), True, False, None,
                0.0, 0.0, 0.0, False, 0))
            continue
        start = max(0.0, float(considered[0]) - cfg.match_ball_serve_pre_s)
        end = min(float(point[1]), float(considered[-1]) + cfg.match_ball_serve_post_s)
        track = track_tracknet(
            video, model=model, start_s=start, end_s=end,
            batch_size=cfg.ball_inference_batch_size, cancel_check=cancel_check)
        observations.append(classify_ball_serve(
            point, considered, track, width, height, cfg))
    return observations
