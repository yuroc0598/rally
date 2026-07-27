import numpy as np
import pytest

from rally.config import RallyConfig
from rally.fusion.decode import (
    dp_decode,
    hysteresis_decode,
    hysteresis_mask,
    mask_to_runs,
    moving_average,
    segments_from_prob,
)
from rally.fusion.points import total_kept_seconds


def _prob_with_rallies(fps, layout):
    """Build a prob signal from a list of (label, seconds); label 'r'=1.0, 'g'=0.0."""
    chunks = []
    for label, secs in layout:
        val = 0.95 if label == "r" else 0.05
        chunks.append(np.full(int(secs * fps), val))
    return np.concatenate(chunks)


def test_moving_average_preserves_length_and_smooths():
    x = np.array([0, 0, 10, 0, 0], dtype=float)
    y = moving_average(x, 3)
    assert y.shape == x.shape
    assert y[2] < 10 and y[2] > 0  # spike gets spread


def test_hysteresis_mask_prevents_flicker():
    # dips to 0.4 mid-rally: with exit=0.35 it must stay active
    prob = np.array([0.1, 0.6, 0.7, 0.4, 0.7, 0.1])
    mask = hysteresis_mask(prob, enter=0.55, exit=0.35)
    assert list(mask) == [False, True, True, True, True, False]


def test_mask_to_runs_basic():
    mask = np.array([False, True, True, False, True])
    assert mask_to_runs(mask) == [(1, 3), (4, 5)]


def test_mask_to_runs_all_true_and_all_false():
    assert mask_to_runs(np.array([True, True, True])) == [(0, 3)]
    assert mask_to_runs(np.array([False, False])) == []


def test_hysteresis_decode_finds_two_rallies():
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=False, pad_pre_s=0.0, pad_post_s=0.0,
                      min_rally_s=2.0, merge_gap_s=1.0)
    prob = _prob_with_rallies(fps, [("g", 20), ("r", 8), ("g", 20), ("r", 6), ("g", 20)])
    segs = segments_from_prob(prob, fps, cfg, total_s=len(prob) / fps)
    assert len(segs) == 2
    # first rally roughly [20, 28]
    assert segs[0][0] == pytest.approx(20, abs=0.6)
    assert segs[0][1] == pytest.approx(28, abs=0.6)


def test_min_rally_filter_drops_short_blips():
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=False, pad_pre_s=0.0, pad_post_s=0.0, min_rally_s=3.0)
    prob = _prob_with_rallies(fps, [("g", 10), ("r", 1), ("g", 10)])  # 1s blip < 3s
    segs = segments_from_prob(prob, fps, cfg)
    assert segs == []


def test_merge_gap_joins_close_rallies():
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=False, pad_pre_s=0.0, pad_post_s=0.0,
                      min_rally_s=2.0, merge_gap_s=2.0)
    # two rallies separated by only 1s -> should merge into one
    prob = _prob_with_rallies(fps, [("g", 10), ("r", 4), ("g", 1), ("r", 4), ("g", 10)])
    segs = segments_from_prob(prob, fps, cfg)
    assert len(segs) == 1


def test_padding_extends_and_clips_at_bounds():
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=False, pad_pre_s=1.0, pad_post_s=1.0, min_rally_s=2.0)
    prob = _prob_with_rallies(fps, [("r", 6), ("g", 10)])  # rally at very start
    segs = segments_from_prob(prob, fps, cfg, total_s=len(prob) / fps)
    assert segs[0][0] == 0.0                       # clipped at 0, not negative
    assert segs[0][1] == pytest.approx(7.0, abs=0.6)  # 6s + 1s post pad


def test_dp_decoder_recovers_rallies():
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=True, pad_pre_s=0.0, pad_post_s=0.0)
    prob = _prob_with_rallies(fps, [("g", 20), ("r", 8), ("g", 20), ("r", 7), ("g", 20)])
    segs = dp_decode(prob, fps, cfg, total_s=len(prob) / fps)
    assert len(segs) == 2
    assert total_kept_seconds(segs) == pytest.approx(15, abs=2.0)


def test_dp_decoder_prefers_typical_durations():
    # A borderline-noisy rally region should still be recovered as one segment.
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=True, pad_pre_s=0.0, pad_post_s=0.0)
    rng = np.random.default_rng(0)
    prob = np.concatenate([
        np.clip(0.1 + 0.1 * rng.standard_normal(int(20 * fps)), 0, 1),
        np.clip(0.8 + 0.1 * rng.standard_normal(int(8 * fps)), 0, 1),
        np.clip(0.1 + 0.1 * rng.standard_normal(int(20 * fps)), 0, 1),
    ])
    segs = dp_decode(prob, fps, cfg, total_s=len(prob) / fps)
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(20, abs=1.5)


def test_dp_merges_rallies_across_short_gap():
    # two rallies separated by a 1s gap < merge_gap_s -> the DP must not emit a short gap,
    # so it merges them into one segment (constraint handled inside the DP, no postprocess).
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=True, pad_pre_s=0.0, pad_post_s=0.0,
                      merge_gap_s=2.0, min_rally_s=2.0)
    prob = _prob_with_rallies(fps, [("g", 10), ("r", 4), ("g", 1), ("r", 4), ("g", 10)])
    segs = dp_decode(prob, fps, cfg, total_s=len(prob) / fps)
    assert len(segs) == 1


def test_dp_drops_sub_min_rally_blip():
    # a 1s rally blip < min_rally_s=3s: the DP cannot label it RALLY, so it stays GAP
    # and never appears in the output.
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=True, pad_pre_s=0.0, pad_post_s=0.0, min_rally_s=3.0)
    prob = _prob_with_rallies(fps, [("g", 10), ("r", 1), ("g", 10)])
    segs = dp_decode(prob, fps, cfg, total_s=len(prob) / fps)
    assert segs == []


def test_dp_handles_clip_shorter_than_min_constraints():
    # feasibility fallback: a clip too short for any constrained tiling must not crash.
    fps = 5.0
    cfg = RallyConfig(use_dp_decoder=True, min_rally_s=3.0, merge_gap_s=3.0)
    prob = _prob_with_rallies(fps, [("r", 1)])  # 5 frames, shorter than both minima
    segs = dp_decode(prob, fps, cfg, total_s=1.0)
    assert isinstance(segs, list)


def test_empty_input():
    cfg = RallyConfig()
    assert segments_from_prob(np.zeros(0), 5.0, cfg, total_s=0.0) == []


def test_config_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        RallyConfig(enter_threshold=0.3, exit_threshold=0.6)
