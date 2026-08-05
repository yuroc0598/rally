from pathlib import Path

import pytest

from rally.preflight import InstallationError, installation_errors, require_server_install


def test_preflight_reports_all_missing_required_models(monkeypatch, tmp_path):
    monkeypatch.setenv("RALLY_MODELS_DIR", str(tmp_path))
    errors = installation_errors(
        check_packages=False, check_ffmpeg=False, load_models=False)
    assert any("TrackNet model is missing" in error for error in errors)
    assert any("YOLO player detector model is missing" in error for error in errors)
    assert any("RTMPose model is missing" in error for error in errors)


def test_server_preflight_refuses_partial_install(monkeypatch, tmp_path):
    monkeypatch.setenv("RALLY_MODELS_DIR", str(tmp_path))
    with pytest.raises(InstallationError, match="refusing to start"):
        require_server_install(
            check_packages=False, check_ffmpeg=False, load_models=False)


def test_preflight_accepts_present_models_without_deep_loading(
    monkeypatch, tmp_path: Path,
):
    monkeypatch.setenv("RALLY_MODELS_DIR", str(tmp_path))
    (tmp_path / "tracknet.pt").touch()
    (tmp_path / "yolo12n.pt").touch()
    (tmp_path / "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx").touch()
    assert installation_errors(
        check_packages=False, check_ffmpeg=False, load_models=False) == []
