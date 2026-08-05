from pathlib import Path

import pytest

from rally.tools.fetch_models import _drive_id_from_url, _gdown_command, main

FILE_ID = "1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"


@pytest.mark.parametrize("url,expected", [
    (f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing", FILE_ID),
    (f"https://drive.google.com/uc?id={FILE_ID}", FILE_ID),
    (f"https://drive.google.com/uc?export=download&id={FILE_ID}", FILE_ID),
    ("https://example.com/tracknet.pt", None),        # not a drive URL
    ("https://drive.google.com/drive/folders/abc", None),  # folder, no file id
])
def test_drive_id_from_url(url, expected):
    assert _drive_id_from_url(url) == expected


def test_main_no_args_returns_usage_code():
    assert main([]) == 2      # nothing to do -> usage exit code


def test_gdown_uses_current_positional_file_id_interface():
    command = _gdown_command(FILE_ID, "/tmp/tracknet.pt")
    assert command[2:4] == ["gdown", FILE_ID]
    assert "--id" not in command


def test_main_drive_download_routes_through_gdown(monkeypatch, tmp_path):
    """--drive-id (and a drive URL) route to the gdown path, then verify."""
    calls = {}

    def fake_drive(file_id, dest):
        calls["id"] = file_id
        calls["dest"] = dest
        open(dest, "wb").close()          # pretend a file arrived

    monkeypatch.setattr("rally.tools.fetch_models._download_drive", fake_drive)
    monkeypatch.setattr("rally.tools.fetch_models._verify", lambda p: calls.setdefault("verified", p))
    monkeypatch.setattr("rally.tools.fetch_models._check_digest",
                        lambda p, expected: calls.setdefault("digest_checked", p) or "0" * 64)

    dest = str(tmp_path / "tracknet.pt")
    rc = main(["--drive-id", FILE_ID, "--dest", dest])
    assert rc == 0
    assert calls["id"] == FILE_ID
    assert calls["dest"] == calls["verified"]
    assert Path(calls["dest"]).parent == tmp_path
    assert calls["dest"] != dest
    assert Path(dest).exists()

    # a drive share URL is parsed to the same id
    calls.clear()
    assert main(["--url", f"https://drive.google.com/file/d/{FILE_ID}/view", "--dest", dest]) == 0
    assert calls["id"] == FILE_ID
