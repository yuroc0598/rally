"""Fuse per-frame channel features into a single rally probability in [0, 1].

Rule-based (Phase-1). Only the channels that are actually available contribute; the
weights are renormalised over present channels, so the pipeline still produces a
sensible score from audio + motion alone when person detection is unavailable.

Kept pure (arrays in, array out) for unit testing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import RallyConfig


def _as_array(x: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.shape[0] != n:
        raise ValueError(f"channel length {a.shape[0]} != timeline length {n}")
    return a


def audio_score(rate: np.ndarray, regularity: np.ndarray) -> np.ndarray:
    """Regular, present strike trains score highest; sparse/irregular hits score low."""
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, 1.0)
    regularity = np.clip(np.asarray(regularity, dtype=float), 0.0, 1.0)
    return np.clip(rate * (0.5 + 0.5 * regularity), 0.0, 1.0)


def motion_score(motion: np.ndarray, camera_moving: Optional[np.ndarray],
                 cfg: RallyConfig) -> np.ndarray:
    """Normalise frame-diff energy; damp it while the camera itself is moving
    (a pan/handheld sweep between points should not read as dynamic play)."""
    m = np.clip(np.asarray(motion, dtype=float) / max(cfg.motion_full_score, 1e-6), 0.0, 1.0)
    if camera_moving is not None:
        m = np.where(np.asarray(camera_moving, dtype=bool), m * 0.3, m)
    return m


def rally_probability(
    cfg: RallyConfig,
    *,
    n: int,
    audio_rate: Optional[np.ndarray] = None,
    audio_regularity: Optional[np.ndarray] = None,
    geometry: Optional[np.ndarray] = None,
    geometry_conf: Optional[np.ndarray] = None,
    motion: Optional[np.ndarray] = None,
    camera_moving: Optional[np.ndarray] = None,
    ball: Optional[np.ndarray] = None,
    ball_conf: Optional[np.ndarray] = None,
    pose: Optional[np.ndarray] = None,
    pose_conf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Confidence-weighted co-decision -> per-frame rally probability (length n).

    Each source (audio strikes, player geometry, player pose, ball-in-play, motion) votes
    with a score in [0,1] AND a per-frame confidence in [0,1]. The fused probability is::

        P = Σ (weight · confidence · score) / Σ (weight · confidence)

    so a source only influences the decision in proportion to how sure it is *right now*.
    A near-side pose that's clear votes strongly; a far/blurred pose (confidence ~0) drops
    out instead of dragging the result. No single channel gates. Sources without an
    explicit confidence default to fully confident.
    """
    audio_rate = _as_array(audio_rate, n)
    audio_regularity = _as_array(audio_regularity, n)
    geometry = _as_array(geometry, n)
    geometry_conf = _as_array(geometry_conf, n)
    motion = _as_array(motion, n)
    camera_moving = _as_array(camera_moving, n)
    ball = _as_array(ball, n)
    ball_conf = _as_array(ball_conf, n)
    pose = _as_array(pose, n)
    pose_conf = _as_array(pose_conf, n)

    ones = np.ones(n, dtype=float)
    # each channel: (weight, score, confidence)
    channels: list[tuple[float, np.ndarray, np.ndarray]] = []
    if audio_rate is not None and audio_regularity is not None:
        channels.append((cfg.w_audio, audio_score(audio_rate, audio_regularity), ones))
    if geometry is not None:
        channels.append((cfg.w_geometry, np.clip(geometry, 0.0, 1.0),
                         geometry_conf if geometry_conf is not None else ones))
    if pose is not None:
        channels.append((cfg.w_pose, np.clip(pose, 0.0, 1.0),
                         pose_conf if pose_conf is not None else ones))
    if ball is not None:
        channels.append((cfg.w_ball, np.clip(ball, 0.0, 1.0),
                         ball_conf if ball_conf is not None else ones))
    if motion is not None:
        channels.append((cfg.w_motion, motion_score(motion, camera_moving, cfg), ones))

    if not channels:
        return np.zeros(n, dtype=float)

    num = np.zeros(n, dtype=float)
    den = np.zeros(n, dtype=float)
    for w, s, c in channels:
        wc = w * np.clip(c, 0.0, 1.0)
        num += wc * s
        den += wc
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
