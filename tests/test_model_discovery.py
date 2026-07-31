from rally.signals.ball import discover_ball_weights


def test_ball_discovery_never_treats_other_pt_models_as_tracknet(tmp_path):
    (tmp_path / "court.pt").touch()
    (tmp_path / "serve_classifier.pt").touch()
    (tmp_path / "yolo12n.pt").touch()
    assert discover_ball_weights(str(tmp_path)) is None

    expected = tmp_path / "custom-tracknet-v2.pt"
    expected.touch()
    assert discover_ball_weights(str(tmp_path)) == str(expected)
