"""Temporal decoding: turn a per-frame rally probability into coarse rally regions.

Two decoders are provided:

* ``hysteresis_decode`` — cheap two-threshold state machine + min-duration / merge /
  pad post-processing. This is the Phase-1 MVP fallback.

* ``dp_decode`` — a duration-aware segmental Viterbi (the "segment-model-lite" from
  part (a) of the design). It jointly chooses segment boundaries and RALLY/GAP
  labels to maximise::

        sum_of_frame_log_emissions  +  duration_prior(len | label)  -  transition_penalty

  Explicit per-label duration priors are the reason to prefer this over frame-HMM
  smoothing: rallies and dead-time each have characteristic lengths, which
  hysteresis + min-duration only approximate crudely.

The regions this produces are still padded/coarse; :mod:`rally.fusion.points`
re-bounds them tightly to the ball strikes. Everything here is pure (numpy in,
numbers out) so it is unit-testable without a video.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..config import RallyConfig

Segment = Tuple[float, float]  # (start_seconds, end_seconds)

_EPS = 1e-6


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge-safe reflection padding."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, pad, mode="reflect")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def hysteresis_mask(prob: np.ndarray, enter: float, exit: float) -> np.ndarray:
    """Two-threshold state machine -> boolean rally mask (prevents mid-rally flicker)."""
    prob = np.asarray(prob, dtype=float)
    mask = np.zeros(prob.shape, dtype=bool)
    active = False
    for i, p in enumerate(prob):
        if active:
            if p < exit:
                active = False
        else:
            if p >= enter:
                active = True
        mask[i] = active
    return mask


def mask_to_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs as half-open [start, end) frame-index intervals."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def _postprocess_runs(
    runs: List[Tuple[int, int]], fps: float, cfg: RallyConfig
) -> List[Tuple[int, int]]:
    """Merge near-adjacent runs, then drop runs shorter than the minimum rally length."""
    if not runs:
        return []
    merge_gap = cfg.merge_gap_s * fps
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    min_len = cfg.min_rally_s * fps
    return [(s, e) for s, e in merged if (e - s) >= min_len]


def _runs_to_padded_segments(
    runs: List[Tuple[int, int]], fps: float, cfg: RallyConfig, total_s: float
) -> List[Segment]:
    """Convert frame runs to padded, clipped, non-overlapping second-intervals."""
    segs: List[Segment] = []
    for s, e in runs:
        start = max(0.0, s / fps - cfg.pad_pre_s)
        end = min(total_s, e / fps + cfg.pad_post_s)
        if end > start:
            segs.append((start, end))
    # padding can create overlaps; coalesce them.
    segs.sort()
    coalesced: List[Segment] = []
    for start, end in segs:
        if coalesced and start <= coalesced[-1][1]:
            coalesced[-1] = (coalesced[-1][0], max(coalesced[-1][1], end))
        else:
            coalesced.append((start, end))
    return coalesced


def hysteresis_decode(prob: np.ndarray, fps: float, cfg: RallyConfig,
                      total_s: float | None = None) -> List[Segment]:
    prob = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(prob)):
        raise ValueError("rally probability must contain only finite values")
    total_s = total_s if total_s is not None else prob.size / fps
    smooth_win = int(round(cfg.smooth_window_s * fps))
    sm = moving_average(prob, smooth_win)
    mask = hysteresis_mask(sm, cfg.enter_threshold, cfg.exit_threshold)
    runs = _postprocess_runs(mask_to_runs(mask), fps, cfg)
    return _runs_to_padded_segments(runs, fps, cfg, total_s)


def _duration_logprior_table(mean_s: float, std_s: float, fps: float, lmax: int) -> np.ndarray:
    """Gaussian log-duration prior evaluated for every segment length 0..lmax (frames)."""
    d = np.arange(lmax + 1, dtype=float) / fps
    z = (d - mean_s) / max(std_s, _EPS)
    return -0.5 * z * z


_INFEASIBLE = -1e17   # scores at/below this mean "no valid segmentation reached here"


def dp_decode(prob: np.ndarray, fps: float, cfg: RallyConfig,
              total_s: float | None = None) -> List[Segment]:
    """Duration-aware segmental Viterbi over two labels (RALLY / GAP).

    Jointly chooses segment boundaries and labels to maximise
    ``Σ frame-log-emissions + duration_prior(len|label) − transition_penalty``.

    The ``min_rally_s`` (minimum rally length) and ``merge_gap_s`` (minimum gap length —
    a shorter gap forces the surrounding rallies to merge) constraints are enforced *inside*
    the DP as per-label length bounds, so its output is already the final segmentation — no
    post-hoc merge/min-length cleanup that would undo the optimality (cf. hysteresis_decode).

    Prefix sums give O(1) segment scoring and the length window is vectorised over numpy, so
    the cost is O(T · Lmax) arithmetic but only O(T) Python iterations.  ``Lmax`` is a
    computational/duration-prior cap, not a semantic one: once a segment reaches it, a
    saturated state can keep the same label for arbitrarily long.  Thus long dead periods
    do not need fake RALLY segments merely to tile the timeline.
    """
    prob = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(prob)):
        raise ValueError("rally probability must contain only finite values")
    prob = np.clip(prob, _EPS, 1.0 - _EPS)
    T = prob.size
    total_s = total_s if total_s is not None else T / fps
    if T == 0:
        return []

    smooth_win = int(round(cfg.smooth_window_s * fps))
    sm = np.clip(moving_average(prob, smooth_win), _EPS, 1.0 - _EPS)

    # prefix[k] = Σ frame-log-emissions over [0, k)  ->  segment [i, j) emission = pref[j]-pref[i]
    log_r = np.log(sm)
    log_g = np.log(1.0 - sm)
    pref_r = np.concatenate([[0.0], np.cumsum(log_r)])          # if RALLY
    pref_g = np.concatenate([[0.0], np.cumsum(log_g)])          # if GAP

    Lmax = max(1, int(round(cfg.max_segment_s * fps)))
    wdur = cfg.duration_prior_weight
    pen = cfg.transition_penalty
    dur_r = wdur * _duration_logprior_table(*cfg.rally_dur_prior_s, fps, Lmax)
    dur_g = wdur * _duration_logprior_table(*cfg.gap_dur_prior_s, fps, Lmax)
    min_r = max(1, int(round(cfg.min_rally_s * fps)))
    min_g = max(1, int(round(cfg.merge_gap_s * fps)))

    NEG = -1e18
    best_r = np.empty(T + 1); best_g = np.empty(T + 1)
    short_r = np.empty(T + 1); short_g = np.empty(T + 1)
    long_r = np.empty(T + 1); long_g = np.empty(T + 1)
    back_r = np.empty(T + 1, np.int64); back_g = np.empty(T + 1, np.int64)
    long_start_r = np.empty(T + 1, np.int64)
    long_start_g = np.empty(T + 1, np.int64)
    # A_lbl[i] = (best score of a segmentation ending at i with the OPPOSITE label) - pref_lbl[i].
    # A RALLY must follow a GAP, so A_r uses best_g; a GAP must follow a RALLY, so A_g uses best_r.
    # i == 0 is the "no predecessor" start state (score 0), enabling a first segment of either label.
    A_r = np.empty(T + 1); A_g = np.empty(T + 1)

    def _solve(min_rally: int, min_gap: int) -> float:
        best_r.fill(NEG); best_g.fill(NEG)
        short_r.fill(NEG); short_g.fill(NEG)
        long_r.fill(NEG); long_g.fill(NEG)
        back_r.fill(-1); back_g.fill(-1)
        long_start_r.fill(-1); long_start_g.fill(-1)
        A_r.fill(NEG); A_g.fill(NEG)
        A_r[0] = -pref_r[0]
        A_g[0] = -pref_g[0]
        for j in range(1, T + 1):
            lo = max(0, j - Lmax)
            # RALLY [i, j): length j-i in [min_rally, Lmax] -> i in [lo, j-min_rally]
            hi = j - min_rally
            if hi >= lo:
                ii = np.arange(lo, hi + 1)
                # i == 0 starts the sequence; it is not a label transition.
                cand = A_r[ii] + dur_r[j - ii] - pen * (ii > 0)
                m = int(np.argmax(cand))
                short_r[j] = cand[m] + pref_r[j]
                back_r[j] = ii[m]
            # GAP [i, j): length in [min_gap, Lmax]
            hi = j - min_gap
            if hi >= lo:
                ii = np.arange(lo, hi + 1)
                cand = A_g[ii] + dur_g[j - ii] - pen * (ii > 0)
                m = int(np.argmax(cand))
                short_g[j] = cand[m] + pref_g[j]
                back_g[j] = ii[m]

            # A segment at the cap may continue in the same label without another
            # duration prior or transition penalty.  Keep this separate from the
            # bounded segment state so shorter durations still receive their proper
            # duration prior.
            i = j - Lmax
            if i >= 0 and Lmax >= min_rally and A_r[i] > _INFEASIBLE:
                exact = A_r[i] + dur_r[Lmax] + pref_r[j] - pen * (i > 0)
                long_r[j] = exact
                long_start_r[j] = i
            if long_r[j - 1] > _INFEASIBLE:
                extended = long_r[j - 1] + log_r[j - 1]
                if extended > long_r[j]:
                    long_r[j] = extended
                    long_start_r[j] = long_start_r[j - 1]

            if i >= 0 and Lmax >= min_gap and A_g[i] > _INFEASIBLE:
                exact = A_g[i] + dur_g[Lmax] + pref_g[j] - pen * (i > 0)
                long_g[j] = exact
                long_start_g[j] = i
            if long_g[j - 1] > _INFEASIBLE:
                extended = long_g[j - 1] + log_g[j - 1]
                if extended > long_g[j]:
                    long_g[j] = extended
                    long_start_g[j] = long_start_g[j - 1]

            best_r[j] = max(short_r[j], long_r[j])
            best_g[j] = max(short_g[j], long_g[j])
            # publish j as a predecessor state for later segments
            if best_g[j] > _INFEASIBLE:
                A_r[j] = best_g[j] - pref_r[j]
            if best_r[j] > _INFEASIBLE:
                A_g[j] = best_r[j] - pref_g[j]
        return max(best_r[T], best_g[T])

    # short clips may not admit any constrained tiling; fall back to unconstrained lengths.
    if _solve(min_r, min_g) <= _INFEASIBLE:
        _solve(1, 1)

    # Backtrack semantic segments.  A saturated segment jumps directly to its true
    # start, so the computational cap never appears as an output boundary.
    mask = np.zeros(T, dtype=bool)
    label_rally = best_r[T] >= best_g[T]
    j = T
    while j > 0:
        if label_rally:
            i = int(long_start_r[j] if long_r[j] > short_r[j] else back_r[j])
        else:
            i = int(long_start_g[j] if long_g[j] > short_g[j] else back_g[j])
        if i < 0:
            break
        if label_rally:
            mask[i:j] = True
        j = i
        label_rally = not label_rally

    return _runs_to_padded_segments(mask_to_runs(mask), fps, cfg, total_s)


def segments_from_prob(prob: np.ndarray, fps: float, cfg: RallyConfig,
                       total_s: float | None = None) -> List[Segment]:
    """Public entry point — pick the decoder according to config."""
    if cfg.use_dp_decoder:
        return dp_decode(prob, fps, cfg, total_s)
    return hysteresis_decode(prob, fps, cfg, total_s)
