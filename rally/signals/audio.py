"""Audio channel: detect racket-ball impact "pocks" and turn them into rhythm features.

A live rally produces a quasi-periodic train of sharp transients (0.3-3 s apart);
between points there is talking / silence / applause. This is a cheap, camera-angle
invariant, and very discriminative signal — often the single strongest cue.

``detect_strikes`` returns onset timestamps (seconds). ``strike_rhythm_features``
projects those onsets onto the analysis timeline as (rate, regularity) channels.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import RallyConfig


def detect_strikes(pcm: np.ndarray, sr: int, cfg: RallyConfig) -> np.ndarray:
    """Band-pass the signal to the impact band, build an amplitude envelope, and
    peak-pick onsets. Returns a sorted array of onset times in seconds.
    """
    pcm = np.asarray(pcm, dtype=float)
    if pcm.size == 0:
        return np.zeros(0, dtype=float)

    from scipy.signal import butter, sosfiltfilt, find_peaks

    lo, hi = cfg.strike_band_hz
    hi = min(hi, sr / 2.0 - 1.0)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    band = sosfiltfilt(sos, pcm)

    env = np.abs(band)
    smooth = max(1, int(0.01 * sr))  # ~10 ms envelope smoothing
    env = np.convolve(env, np.ones(smooth) / smooth, mode="same")

    distance = max(1, int(cfg.strike_min_gap_s * sr))
    # Stage 1: permissive candidates (cheap global gate), then Stage 2: keep only those
    # that clear a LOCAL adaptive threshold. Local stats track a changing noise floor
    # (crowd/applause vs. quiet), so a global threshold neither floods loud sections nor
    # misses soft hits in quiet ones — while staying O(n) instead of a full rolling median.
    global_med = float(np.median(env)) + 1e-12
    # Floor the gate at 0.1% of the envelope peak so a near-silent track (median ~ 0) does
    # not admit a flood of tiny candidates for the O(n) local test to scan. Real strikes sit
    # near the peak, far above this floor, so normal footage is unaffected.
    height = max(2.0 * global_med, 1e-3 * float(env.max()))
    cand, _ = find_peaks(env, height=height, distance=distance)
    if cand.size == 0:
        return np.zeros(0, dtype=float)

    half = max(1, int(cfg.audio_block_s * sr / 2))
    keep = []
    for p in cand:
        seg = env[max(0, p - half):p + half]
        loc_med = float(np.median(seg))
        loc_std = float(np.std(seg))
        thr = max(loc_med + cfg.strike_sensitivity * loc_std,
                  cfg.strike_snr_ratio * loc_med)
        if env[p] >= thr:
            keep.append(p)
    return (np.array(keep, dtype=float) / sr) if keep else np.zeros(0, dtype=float)


def strike_rhythm_features(
    onsets: np.ndarray, timeline_s: np.ndarray, cfg: RallyConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """For each analysis timestamp, compute over a trailing window:

    * ``rate``       — normalised strike count (0..1), saturating at ``strikes_full_score``
    * ``regularity`` — 1/(1+CV) of inter-onset intervals (0..1); high for even rhythm

    Both are 0 where there is no strike activity.
    """
    onsets = np.sort(np.asarray(onsets, dtype=float))
    timeline_s = np.asarray(timeline_s, dtype=float)
    n = timeline_s.size
    rate = np.zeros(n, dtype=float)
    regularity = np.zeros(n, dtype=float)
    if onsets.size == 0 or n == 0:
        return rate, regularity

    win = cfg.rhythm_window_s
    lefts = timeline_s - win
    lo_idx = np.searchsorted(onsets, lefts, side="left")
    hi_idx = np.searchsorted(onsets, timeline_s, side="right")

    for i in range(n):
        window = onsets[lo_idx[i]:hi_idx[i]]
        count = window.size
        rate[i] = min(1.0, count / max(cfg.strikes_full_score, 1e-6))
        if count >= 3:
            iois = np.diff(window)
            mean_ioi = iois.mean()
            if mean_ioi > 1e-6:
                cv = iois.std() / mean_ioi
                regularity[i] = 1.0 / (1.0 + cv)
    return rate, regularity
