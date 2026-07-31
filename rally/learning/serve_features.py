"""Canonical feature extraction for serialized and live serve observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .serve_schema import FEATURE_NAMES


def _value(observation: Any, name: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def empty_features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def observation_features(observation: Any) -> dict[str, float]:
    """Map one observation object/dict into the non-audio feature schema."""
    features = empty_features()
    sampled = max(1, int(_value(observation, "sampled_frames", 0) or 0))
    pose_frames = int(_value(observation, "pose_frames", 0) or 0)
    features.update({
        "pose_available": float(bool(_value(observation, "observable") or pose_frames)),
        "pose_frame_fraction": float(pose_frames / sampled),
        "pose_ready_fraction": float(int(_value(observation, "ready_frames", 0) or 0)
                                     / sampled),
        "pose_overhead_frames": float(_value(observation, "overhead_frames", 0) or 0),
        "pose_overhead_max_ratio": float(
            _value(observation, "overhead_max_ratio", 0.0) or 0.0),
        "pose_side_confidence": float(_value(observation, "side_confidence", 0.0) or 0.0),
        "position_available": float(bool(_value(observation, "position_checked"))),
        "position_score": float(_value(observation, "position_score", 0.0) or 0.0),
        "position_stable_fraction": float(
            _value(observation, "position_stable_fraction", 0.0) or 0.0),
        "position_player_tracks": float(
            _value(observation, "position_player_tracks", 0) or 0),
        "position_server_span": float(
            _value(observation, "position_server_span", 0.0) or 0.0),
        "position_end_near": float(_value(observation, "position_server_end") == "near"),
        "position_end_far": float(_value(observation, "position_server_end") == "far"),
        "court_filtered": float(bool(_value(observation, "target_court_filtered"))),
        "ball_available": float(bool(_value(observation, "ball_checked"))),
        "ball_coverage": float(_value(observation, "ball_coverage", 0.0) or 0.0),
        "ball_vertical_span": float(_value(observation, "ball_vertical_span", 0.0) or 0.0),
        "ball_outgoing_span": float(_value(observation, "ball_outgoing_span", 0.0) or 0.0),
        "ball_ordered": float(bool(_value(observation, "ball_ordered_evidence"))),
    })
    return features


def audio_features(onsets: np.ndarray, contact_time: float, cluster_gap_s: float
                   ) -> dict[str, float]:
    """Extract the schema's audio cluster nearest one expected contact."""
    values = {name: 0.0 for name in FEATURE_NAMES if name.startswith("audio_")}
    strikes = np.unique(np.sort(np.asarray(onsets, dtype=float)))
    strikes = strikes[np.isfinite(strikes)]
    if not strikes.size:
        return values
    split = np.flatnonzero(np.diff(strikes) > float(cluster_gap_s)) + 1
    clusters = [cluster for cluster in np.split(strikes, split) if cluster.size]
    firsts = np.asarray([cluster[0] for cluster in clusters], dtype=float)
    index = int(np.argmin(np.abs(firsts - float(contact_time))))
    if abs(float(firsts[index]) - float(contact_time)) > 2.0:
        return values
    cluster = clusters[index]
    intervals = np.diff(cluster)
    values.update({
        "audio_available": 1.0,
        "audio_gap_before_s": float(
            cluster[0] - clusters[index - 1][-1] if index else 30.0),
        "audio_cluster_strikes": float(cluster.size),
        "audio_cluster_duration_s": float(cluster[-1] - cluster[0]),
        "audio_interval_mean_s": float(intervals.mean()) if intervals.size else 0.0,
        "audio_interval_std_s": float(intervals.std()) if intervals.size else 0.0,
    })
    return values
