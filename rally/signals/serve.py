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
TrackCache = Sequence[tuple[Segment, BallTrack]]


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
    court=None,
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

    xn = np.asarray(track.x, float) / float(frame_width)
    yn = np.asarray(track.y, float) / float(frame_height)
    visible = np.isfinite(xn) & np.isfinite(yn)
    if court is not None:
        # Image-relative bounds include portions of neighboring courts in wide match
        # footage.  Once calibrated, only samples mapping onto the selected tennis court
        # may support a serve; off-court hits and motion are background evidence.
        from .court import COURT_L, DOUBLES_W

        image_points = np.stack(
            [np.asarray(track.x, float), np.asarray(track.y, float)], axis=1)
        court_points = court.to_court(image_points)
        margin_m = 0.75
        in_court = (
            visible
            & np.isfinite(court_points).all(axis=1)
            & (court_points[:, 0] >= -margin_m)
            & (court_points[:, 0] <= DOUBLES_W + margin_m)
            & (court_points[:, 1] >= -margin_m)
            & (court_points[:, 1] <= COURT_L + margin_m)
        )
    else:
        x0, x1 = cfg.match_ball_court_x
        y0, y1 = cfg.match_ball_court_y
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


def _cached_track_window(
    cache: TrackCache | None, start: float, end: float,
) -> BallTrack | None:
    """Return a continuous raw TrackNet slice when the arbiter already decoded it."""
    if not cache:
        return None
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for (_cache_start, _cache_end), track in cache:
        if track.t.size == 0:
            continue
        mask = (track.t >= start - 0.10) & (track.t <= end + 0.10)
        if np.any(mask):
            pieces.append((track.t[mask], track.x[mask], track.y[mask]))
    if not pieces:
        return None
    t = np.concatenate([piece[0] for piece in pieces])
    x = np.concatenate([piece[1] for piece in pieces])
    y = np.concatenate([piece[2] for piece in pieces])
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    unique = np.r_[True, np.diff(t) > 1e-9]
    t, x, y = t[unique], x[unique], y[unique]
    if t.size < 3 or t[0] > start + 0.10 or t[-1] < end - 0.10:
        return None
    return BallTrack(t, x, y)


def observe_ball_serves(
    video: str,
    points: Sequence[Segment],
    onsets: np.ndarray,
    weights_path: str,
    cfg,
    court=None,
    track_cache: TrackCache | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], None] = lambda: None,
) -> list[BallServeObservation]:  # pragma: no cover - heavy model integration
    """Classify serve windows, reusing arbiter tracks before running TrackNet again."""
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
    model = None
    observations: list[BallServeObservation] = []
    cache_hits = 0
    total = len(points)
    for index, point in enumerate(points, 1):
        cancel_check()
        strikes = ordered[(ordered >= point[0] - 1e-9) & (ordered <= point[1] + 1e-9)]
        considered = strikes[: int(cfg.match_serve_strikes_to_check)]
        if not considered.size:
            observations.append(BallServeObservation(
                (float(point[0]), float(point[1])), True, False, None,
                0.0, 0.0, 0.0, False, 0))
            if progress_callback is not None:
                progress_callback(index, total, cache_hits)
            continue
        start = max(0.0, float(considered[0]) - cfg.match_ball_serve_pre_s)
        end = min(float(point[1]), float(considered[-1]) + cfg.match_ball_serve_post_s)
        track = _cached_track_window(track_cache, start, end)
        if track is not None:
            cache_hits += 1
        else:
            if model is None:
                model = load_ball_model(weights_path)
            track = track_tracknet(
                video, model=model, start_s=start, end_s=end,
                batch_size=cfg.ball_inference_batch_size,
                tracking_fps=cfg.ball_tracking_fps,
                half_precision=cfg.ball_half_precision,
                court=court,
                cancel_check=cancel_check)
        observations.append(classify_ball_serve(
            point, considered, track, width, height, cfg, court=court))
        if progress_callback is not None:
            progress_callback(index, total, cache_hits)
    return observations
