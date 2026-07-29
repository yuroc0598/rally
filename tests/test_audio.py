import numpy as np
import pytest

from rally.signals.audio import (
    _plausible_impact_shape,
    detect_strikes,
    strike_rhythm_features,
)
from rally.config import RallyConfig


def _synth_signal(sr, duration_s, strike_times, freq=3000.0):
    """Silence + short decaying tone bursts (in the impact band) at strike_times."""
    n = int(duration_s * sr)
    x = 0.001 * np.random.default_rng(0).standard_normal(n)  # faint noise floor
    t = np.arange(n) / sr
    for st in strike_times:
        burst_len = int(0.02 * sr)
        i0 = int(st * sr)
        idx = np.arange(burst_len)
        env = np.exp(-idx / (0.004 * sr))
        tone = np.sin(2 * np.pi * freq * idx / sr) * env
        if i0 + burst_len <= n:
            x[i0:i0 + burst_len] += tone
    return x


def test_detect_strikes_recovers_onsets():
    sr = 22050
    cfg = RallyConfig(audio_sr=sr)
    strikes = [1.0, 2.0, 3.0, 4.0]
    x = _synth_signal(sr, 5.0, strikes)
    found = detect_strikes(x, sr, cfg)
    assert len(found) == len(strikes)
    for got, exp in zip(sorted(found), strikes):
        assert got == pytest.approx(exp, abs=0.05)


def test_detect_strikes_empty_on_silence():
    sr = 22050
    cfg = RallyConfig(audio_sr=sr)
    x = 0.0005 * np.random.default_rng(1).standard_normal(sr * 3)
    found = detect_strikes(x, sr, cfg)
    assert len(found) <= 1  # essentially nothing


def test_impact_shape_rejects_voiced_score_call_but_keeps_racket_contact():
    sr = 22050
    cfg = RallyConfig(audio_sr=sr)
    n = int(0.2 * sr)
    center = n // 2
    t = np.arange(n) / sr

    # A voiced score call: sustained harmonic energy with no sharp broadband attack.
    voiced = sum(np.sin(2 * np.pi * 220 * harmonic * t) / harmonic
                 for harmonic in range(1, 18)).astype(np.float32)
    assert not _plausible_impact_shape(voiced, voiced, center, sr, cfg)

    # A short, decaying noise burst has the impulsive/broadband shape of contact.
    impact = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    burst = rng.standard_normal(int(0.012 * sr)).astype(np.float32)
    burst *= np.exp(-np.arange(burst.size) / (0.002 * sr))
    impact[center:center + burst.size] = burst
    assert _plausible_impact_shape(impact, impact, center, sr, cfg)


def test_rhythm_features_high_during_regular_strikes():
    cfg = RallyConfig(rhythm_window_s=3.0, strikes_full_score=3.0)
    onsets = np.array([10.0, 10.8, 11.6, 12.4, 13.2])  # even ~0.8s spacing
    timeline = np.array([5.0, 13.0, 30.0])
    rate, reg = strike_rhythm_features(onsets, timeline, cfg)
    assert rate[0] == 0.0 and reg[0] == 0.0        # before any strikes
    assert rate[1] > 0.5 and reg[1] > 0.7          # in the middle of the rally
    assert rate[2] == 0.0 and reg[2] == 0.0        # long after


def test_rhythm_features_empty_onsets():
    cfg = RallyConfig()
    rate, reg = strike_rhythm_features(np.zeros(0), np.array([1.0, 2.0]), cfg)
    assert np.all(rate == 0) and np.all(reg == 0)
