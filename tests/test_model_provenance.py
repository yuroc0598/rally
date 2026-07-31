import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_provenance_states_unknown_tracknet_rights_and_ultralytics_options():
    text = "\n".join([
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "models" / "README.md").read_text(encoding="utf-8"),
    ])
    lowered = re.sub(r"\s+", " ", text.lower())
    assert "provenance unknown" in lowered
    assert "training-data provenance unknown" in lowered
    assert "license unknown" in lowered
    assert "agpl-3.0" in lowered
    assert "enterprise license" in lowered
    assert "network" in lowered
    assert "sha-256" in lowered and "not" in lowered and "license" in lowered


def test_readme_contains_model_provenance_warning():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "unknown provenance and unknown license" in text
