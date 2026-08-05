import json
import uuid

from fastapi.testclient import TestClient

from rally.web import app as webapp


def test_detected_match_names_are_durable_and_update_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id = str(uuid.uuid4())
    root = tmp_path / job_id
    output = root / "output"
    output.mkdir(parents=True)
    metadata = output / "rallies.json"
    match = {
        "format": "singles",
        "format_confidence": 0.98,
        "roster": [
            {"id": "P1", "name": "Player 1", "team_id": "T1"},
            {"id": "P2", "name": "Player 2", "team_id": "T2"},
        ],
        "teams": [
            {"id": "T1", "player_ids": ["P1"]},
            {"id": "T2", "player_ids": ["P2"]},
        ],
    }
    result = {"match": match, "segments": [], "points": []}
    metadata.write_text(json.dumps(result))
    job = {
        "id": job_id, "filename": "match.mp4", "status": "complete",
        "match": match, "result": result, "json_path": str(metadata),
        "output_path": None, "thumbnail_path": None, "original_path": None,
        "processing": {}, "progress": [], "labeling": {},
    }
    webapp._atomic_write_json(root / "job.json", job)

    client = TestClient(webapp.app)
    response = client.post(f"/api/jobs/{job_id}/match", json={"roster": [
        {"id": "P1", "name": "Alice"}, {"id": "P2", "name": "Bob"},
    ]})

    assert response.status_code == 200
    assert [record["name"] for record in response.json()["match"]["roster"]] == [
        "Alice", "Bob"]
    saved_job = webapp._read_json(root / "job.json", {})
    saved_sidecar = webapp._read_json(metadata, {})
    assert saved_job["match"]["roster"][0]["name"] == "Alice"
    assert saved_job["result"]["match"]["roster"][1]["name"] == "Bob"
    assert saved_sidecar["match"]["roster"][0]["name"] == "Alice"


def test_match_profile_refresh_preserves_user_names():
    existing = {"roster": [
        {"id": "P1", "name": "Alice"}, {"id": "P2", "name": "Bob"},
    ], "names_updated_at": "now"}
    detected = {"format": "singles", "roster": [
        {"id": "P1", "name": "Player 1", "team_id": "T1"},
        {"id": "P2", "name": "Player 2", "team_id": "T2"},
    ]}
    merged = webapp._merge_match_profile(existing, detected)
    assert [record["name"] for record in merged["roster"]] == ["Alice", "Bob"]
    assert merged["roster"][0]["team_id"] == "T1"
