import numpy as np
import pytest

from rally.config import RallyConfig
from rally.fusion.score import audio_score, motion_score, rally_probability


def test_audio_score_rewards_regular_present_strikes():
    high = audio_score(rate=np.array([1.0]), regularity=np.array([1.0]))
    low = audio_score(rate=np.array([1.0]), regularity=np.array([0.0]))
    none = audio_score(rate=np.array([0.0]), regularity=np.array([1.0]))
    assert high[0] == pytest.approx(1.0)
    assert low[0] == pytest.approx(0.5)
    assert none[0] == pytest.approx(0.0)


def test_motion_score_damped_by_camera_motion():
    cfg = RallyConfig(motion_full_score=0.04)
    m = np.array([0.04, 0.04])
    cam = np.array([False, True])
    s = motion_score(m, cam, cfg)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.3)  # damped while camera moves


def test_rally_probability_renormalises_over_available_channels():
    cfg = RallyConfig()
    # audio-only: should equal the audio score regardless of other weights
    p = rally_probability(cfg, n=1, audio_rate=np.array([1.0]),
                          audio_regularity=np.array([1.0]))
    assert p[0] == pytest.approx(1.0)


def test_rally_probability_ball_channel_codecides():
    cfg = RallyConfig(w_audio=0.5, w_ball=0.5)
    # audio says no play, ball says in-play -> fused is the average (co-decision)
    p = rally_probability(cfg, n=1,
                          audio_rate=np.array([0.0]), audio_regularity=np.array([0.0]),
                          ball=np.array([1.0]))
    assert p[0] == pytest.approx(0.5)  # neither channel gates; they average
    # both agree -> high
    p2 = rally_probability(cfg, n=1,
                           audio_rate=np.array([1.0]), audio_regularity=np.array([1.0]),
                           ball=np.array([1.0]))
    assert p2[0] == pytest.approx(1.0)


def test_confidence_weighting_drops_out_unsure_source():
    cfg = RallyConfig(w_audio=0.5, w_pose=0.5)
    # pose is confidently wrong-free: conf=0 -> it must NOT affect the result
    p = rally_probability(cfg, n=1, audio_rate=np.array([1.0]),
                          audio_regularity=np.array([1.0]),
                          pose=np.array([0.0]), pose_conf=np.array([0.0]))
    assert p[0] == pytest.approx(1.0)  # pose dropped out; audio-only stands


def test_confidence_weighting_lets_sure_source_vote():
    cfg = RallyConfig(w_audio=0.5, w_pose=0.5)
    # audio says no, pose is sure it's play (conf=1) -> they co-decide -> ~0.5
    p = rally_probability(cfg, n=1, audio_rate=np.array([0.0]),
                          audio_regularity=np.array([0.0]),
                          pose=np.array([1.0]), pose_conf=np.array([1.0]))
    assert p[0] == pytest.approx(0.5)


def test_rally_probability_zero_without_channels():
    cfg = RallyConfig()
    p = rally_probability(cfg, n=3)
    assert np.all(p == 0.0)


def test_rally_probability_length_mismatch_raises():
    cfg = RallyConfig()
    with pytest.raises(ValueError):
        rally_probability(cfg, n=3, motion=np.array([0.1, 0.2]))


def test_rally_probability_combines_weighted():
    cfg = RallyConfig(w_audio=0.5, w_geometry=0.3, w_motion=0.2, motion_full_score=1.0)
    p = rally_probability(
        cfg, n=1,
        audio_rate=np.array([1.0]), audio_regularity=np.array([1.0]),  # audio_score=1
        geometry=np.array([0.0]),                                       # geometry=0
        motion=np.array([0.0]),                                         # motion=0
    )
    # (0.5*1 + 0.3*0 + 0.2*0) / (0.5+0.3+0.2) = 0.5
    assert p[0] == pytest.approx(0.5)
