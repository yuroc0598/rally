"""Audio channel: detect racket-ball impact "pocks" and turn them into rhythm features.

A live rally produces a quasi-periodic train of sharp transients (0.3-3 s apart);
between points there is talking / silence / applause. This is a cheap, camera-angle
invariant, and very discriminative signal — often the single strongest cue.

``detect_strikes`` returns onset timestamps (seconds). ``strike_rhythm_features``
projects those onsets onto the analysis timeline as (rate, regularity) channels.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from ..config import RallyConfig


def _plausible_impact_shape(
    pcm: np.ndarray, band: np.ndarray, peak: int, sr: int, cfg: RallyConfig
) -> bool:
    """Reject voiced/sustained transients that resemble speech rather than a ball hit.

    Racket contact is normally either strongly impulsive in the impact band or broadband
    and noise-like. Score calls may contain loud consonants and pass the adaptive SNR gate,
    but their short spectrum remains comparatively harmonic. The two criteria are OR-ed so
    quiet real contacts are not required to be both perfectly sharp and spectrally flat.
    """
    crest_half = max(4, int(round(0.015 * sr)))
    lo, hi = max(0, peak - crest_half), min(band.size, peak + crest_half)
    local_band = np.asarray(band[lo:hi], dtype=np.float64)
    if local_band.size < 8:
        return False
    rms = float(np.sqrt(np.mean(local_band * local_band)))
    crest = float(np.max(np.abs(local_band)) / max(rms, 1e-12))
    if crest >= cfg.strike_min_crest_factor:
        return True

    spectrum_half = max(8, int(round(0.020 * sr)))
    lo, hi = max(0, peak - spectrum_half), min(pcm.size, peak + spectrum_half)
    local_pcm = np.asarray(pcm[lo:hi], dtype=np.float64)
    if local_pcm.size < 16:
        return False
    magnitude = np.abs(np.fft.rfft(local_pcm * np.hanning(local_pcm.size))) + 1e-12
    flatness = float(np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude))
    return flatness >= cfg.strike_min_spectral_flatness


def detect_strikes(pcm: np.ndarray, sr: int, cfg: RallyConfig) -> np.ndarray:
    """Band-pass the signal to the impact band, build an amplitude envelope, and
    peak-pick onsets. Returns a sorted array of onset times in seconds.
    """
    # Keep the caller's multi-hour PCM out of float64. scipy may use a temporary with the
    # SOS coefficients internally, but every persistent full-buffer array stays float32.
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0:
        return np.zeros(0, dtype=float)

    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, sosfiltfilt, find_peaks

    lo, hi = cfg.strike_band_hz
    hi = min(hi, sr / 2.0 - 1.0)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    band = np.asarray(sosfiltfilt(sos, pcm), dtype=np.float32)

    env = np.abs(band, dtype=np.float32)
    smooth = max(1, int(0.01 * sr))  # ~10 ms envelope smoothing
    # uniform_filter1d is O(n); np.convolve's direct 220-tap pass was unnecessarily costly
    # on every audio sample.
    env = uniform_filter1d(env, size=smooth, mode="nearest")

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

    # Estimate local noise once per block, not once per candidate peak. This bounds work to
    # O(n + number_of_peaks) instead of repeatedly scanning four seconds around each peak.
    block_n = max(1, int(cfg.audio_block_s * sr))
    n_blocks = (env.size + block_n - 1) // block_n
    local_med = np.empty(n_blocks, dtype=np.float32)
    local_std = np.empty(n_blocks, dtype=np.float32)
    for bi in range(n_blocks):
        seg = env[bi * block_n:min(env.size, (bi + 1) * block_n)]
        local_med[bi] = np.median(seg)
        local_std[bi] = np.std(seg)
    keep = []
    for p in cand:
        bi = min(int(p) // block_n, n_blocks - 1)
        loc_med = float(local_med[bi])
        loc_std = float(local_std[bi])
        thr = max(loc_med + cfg.strike_sensitivity * loc_std,
                  cfg.strike_snr_ratio * loc_med)
        if env[p] >= thr and _plausible_impact_shape(pcm, band, int(p), sr, cfg):
            keep.append(p)
    return (np.array(keep, dtype=float) / sr) if keep else np.zeros(0, dtype=float)


def detect_strikes_stream(
    chunks: Iterable[np.ndarray], sr: int, cfg: RallyConfig, *, overlap_s: float | None = None
) -> np.ndarray:
    """Detect strikes from a bounded PCM stream and return global onset timestamps.

    One chunk is delayed so each core chunk is analyzed with left/right context. Only
    detections inside the core are emitted, then boundary duplicates are collapsed. Peak
    memory is therefore proportional to the chunk size rather than match duration.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    context_s = overlap_s if overlap_s is not None else max(1.0, cfg.audio_block_s / 2 + 0.25)
    context_n = max(1, int(round(context_s * sr)))
    iterator = iter(chunks)
    try:
        pending = np.asarray(next(iterator), dtype=np.float32).reshape(-1)
    except StopIteration:
        return np.zeros(0, dtype=float)

    left = np.zeros(0, dtype=np.float32)
    pending_start = 0
    found: list[float] = []

    def process(right: np.ndarray, *, final: bool = False) -> None:
        nonlocal left, pending, pending_start
        right = np.asarray(right, dtype=np.float32).reshape(-1)
        right_context = right[:context_n]
        buf = np.concatenate((left, pending, right_context))
        core0 = left.size
        core1 = core0 + pending.size
        local = detect_strikes(buf, sr, cfg)
        sample_idx = np.rint(local * sr).astype(np.int64)
        inside = (sample_idx >= core0) & (sample_idx < core1)
        for idx in sample_idx[inside]:
            found.append((pending_start + int(idx) - core0) / sr)
        left = pending[-context_n:].copy()
        pending_start += pending.size
        pending = right

    for chunk in iterator:
        process(np.asarray(chunk, dtype=np.float32))
    process(np.zeros(0, dtype=np.float32), final=True)

    if not found:
        return np.zeros(0, dtype=float)
    # Core intervals are disjoint, but the detector's refractory window can straddle a
    # boundary. Enforce it globally to make chunk size irrelevant.
    ordered = np.sort(np.asarray(found, dtype=float))
    kept = [float(ordered[0])]
    for onset in ordered[1:]:
        if float(onset) - kept[-1] >= cfg.strike_min_gap_s:
            kept.append(float(onset))
    return np.asarray(kept, dtype=float)


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
