"""Tests for the rally web UI.

Unit tests cover the pure request/response logic (config mapping, progress
monotonicity, segment normalisation, safety). One end-to-end test drives the
whole HTTP surface — upload → process → poll → edit segments → re-cut — against
a synthesised video, and auto-skips in sandboxes that can't run ffmpeg on
absolute paths (same guard the core integration test uses).
"""

import json
import os
import shutil
import subprocess
import time
import uuid

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

from fastapi.testclient import TestClient  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import rally.web.app as webapp  # noqa: E402


# --------------------------------------------------------------------------- #
# unit tests (no ffmpeg / no video)                                           #
# --------------------------------------------------------------------------- #
def test_config_from_options_maps_flags_and_numbers():
    cfg = webapp._config_from_options(
        {"static_camera": True, "hysteresis": True, "fast": True, "no_labels": True,
         "min_rally": 3.0, "serve_preroll": 2.0, "gap": 0.2}
    )
    assert cfg.w_audio == 0.7 and cfg.rhythm_window_s == 5.0   # static-camera preset
    assert cfg.use_dp_decoder is False                          # hysteresis
    assert cfg.reencode is False                                # fast
    assert cfg.label_points is False                            # no_labels
    assert cfg.min_rally_s == 3.0
    assert cfg.serve_preroll_s == 2.0 and cfg.toss_preroll_s == 2.0
    assert cfg.inter_point_gap_s == 0.2


def test_config_from_options_ball_arbiter_defaults_on():
    # both ball-arbiter and court auto-detection are ON by default (best accuracy)
    default = webapp._config_from_options({})
    assert default.ball_arbiter is True and default.court_auto is True
    # explicitly unchecked boxes are respected
    off = webapp._config_from_options({"ball_arbiter": False, "court_auto": False})
    assert off.ball_arbiter is False and off.court_auto is False


def test_capabilities_reports_ball_arbiter_availability(monkeypatch):
    client = TestClient(webapp.app)

    # no weights present -> not available, with a hint
    monkeypatch.setattr("rally.signals.ball.discover_ball_weights", lambda *a, **k: None)
    caps = client.get("/api/capabilities").json()
    assert caps["ball_arbiter"]["available"] is False
    assert caps["ball_arbiter"]["weights_present"] is False
    assert caps["ball_arbiter"]["hint"]                     # tells the user what to do
    assert caps["court_auto"]["available"] is True          # classical, always on

    # weights present + torch installed -> available
    monkeypatch.setattr("rally.signals.ball.discover_ball_weights",
                        lambda *a, **k: "models/tracknet.pt")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    caps2 = client.get("/api/capabilities").json()
    assert caps2["ball_arbiter"]["available"] is True
    assert caps2["ball_arbiter"]["weights_path"] == "models/tracknet.pt"


def test_upload_route_accepts_and_stores_ball_arbiter(monkeypatch, tmp_path):
    """The upload form fields reach the stored job options (frontend -> backend wiring)."""
    from pathlib import Path
    monkeypatch.setattr(webapp, "DATA_DIR", Path(tmp_path) / "data")
    client = TestClient(webapp.app)
    resp = client.post(
        "/api/jobs",
        files={"file": ("m.mp4", b"\x00\x01\x02\x03fakevideobytes", "video/mp4")},
        data={"run_now": "false", "ball_arbiter": "true", "court_auto": "true"},
    )
    assert resp.status_code == 200
    opts = resp.json()["options"]
    assert opts["ball_arbiter"] is True and opts["court_auto"] is True
    # and that config built from those options turns the feature on
    cfg = webapp._config_from_options(opts)
    assert cfg.ball_arbiter is True and cfg.court_auto is True


def test_stage_for_message_recognises_pipeline_lines():
    assert webapp._stage_for_message("rendering 12 points -> out.mp4")["stage"] == "rendering"
    assert webapp._stage_for_message("processing failed: boom")["percent"] == 100
    assert webapp._stage_for_message("  123 strikes detected")["stage"] == "audio"
    assert webapp._stage_for_message("something unknown")["stage"] == "running"


def test_append_progress_is_monotonic(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    (tmp_path / job_id).mkdir()
    webapp._atomic_write_json(
        webapp._job_meta_path(job_id),
        {"id": job_id, "status": "running", "progress": []},
    )
    webapp._append_progress(job_id, "rendering 4 points -> out.mp4")   # -> 92
    webapp._append_progress(job_id, "  a stray warning that maps to nothing")
    job = webapp._read_json(webapp._job_meta_path(job_id), {})
    assert job["processing"]["percent"] == 92            # not dragged back to 50
    assert "stray warning" in job["processing"]["detail"]


def test_normalise_segments_clips_sorts_and_drops_empty():
    segs = webapp._normalise_segments([[10, 14], [-2, 3], [50, 500], [8, 8]], duration=100)
    assert segs == [(0.0, 3.0), (10.0, 14.0), (50.0, 100.0)]  # clipped, sorted, zero-len dropped


def test_normalise_segments_rejects_bad_shape():
    with pytest.raises(Exception):
        webapp._normalise_segments([[1, 2, 3]], duration=100)


def test_bad_job_id_is_404():
    client = TestClient(webapp.app)
    assert client.get("/api/jobs/not-a-uuid").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404


def test_delete_processing_job_blocked(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    (tmp_path / job_id).mkdir()
    webapp._atomic_write_json(webapp._job_meta_path(job_id), {"id": job_id, "status": "running"})
    client = TestClient(webapp.app)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    assert (tmp_path / job_id).exists()


# --------------------------------------------------------------------------- #
# end-to-end (ffmpeg required)                                                #
# --------------------------------------------------------------------------- #
SR = 22050
DURATION = 40.0
RALLIES = [(12.0, 20.0), (32.0, 38.0)]


def _make_video(dirpath):
    audio = os.path.join(dirpath, "a.wav")
    src = os.path.join(dirpath, "match.mp4")
    n = int(DURATION * SR)
    x = 0.0005 * np.random.default_rng(0).standard_normal(n)
    for (start, end) in RALLIES:
        t = start
        while t < end:
            i0 = int(t * SR)
            burst = int(0.02 * SR)
            idx = np.arange(burst)
            tone = np.sin(2 * np.pi * 3000 * idx / SR) * np.exp(-idx / (0.004 * SR))
            if i0 + burst <= n:
                x[i0:i0 + burst] += tone
            t += 0.8
    wavfile.write(audio, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s=320x240:r=25:d={DURATION}", "-i", audio,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
        check=True,
    )
    return src


@pytest.fixture
def scratch(monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    local = os.path.join(os.path.dirname(__file__), "_scratch_web")
    os.makedirs(local, exist_ok=True)
    # sandbox guard: verify a child ffmpeg can read an absolute path we wrote
    probe = os.path.join(local, "_probe.wav")
    wavfile.write(probe, SR, np.zeros(SR // 10, dtype=np.int16))
    rc = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.abspath(probe), "-f", "null", "-"],
                        capture_output=True).returncode
    if rc != 0:
        shutil.rmtree(local, ignore_errors=True)
        pytest.skip("sandbox remaps absolute paths for subprocesses")
    from pathlib import Path
    monkeypatch.setattr(webapp, "DATA_DIR", Path(local).resolve() / "data")
    try:
        yield local
    finally:
        shutil.rmtree(local, ignore_errors=True)


def _wait_done(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"complete", "no_output", "failed"}:
            return job
        time.sleep(0.5)
    raise AssertionError("job did not finish in time")


def test_end_to_end_upload_process_edit(scratch):
    src = _make_video(scratch)
    client = TestClient(webapp.app)

    with open(src, "rb") as fh:
        resp = client.post(
            "/api/jobs",
            files={"file": ("match.mp4", fh, "video/mp4")},
            # ball_arbiter is on by default but is CPU-slow + needs weights; this test
            # covers the audio/motion path + HTTP surface, so opt out for speed.
            data={"detect_players": "false", "analysis_fps": "5", "min_rally": "1",
                  "ball_arbiter": "false"},
        )
    assert resp.status_code == 200
    job = resp.json()
    job_id = job["id"]
    assert job["status"] in {"queued", "running", "uploaded"}

    job = _wait_done(client, job_id)
    assert job["status"] == "complete", job.get("error")
    r = job["result"]
    assert 1 <= r["n_rallies"] <= 3
    assert job["media"]["output"] and job["media"]["output_download"]
    assert job["media"]["metadata_download"]

    # both ground-truth rallies covered
    segs = [(s["start"], s["end"]) for s in r["segments"]]
    for gs, ge in RALLIES:
        mid = (gs + ge) / 2
        assert any(s <= mid <= e for s, e in segs), f"missed rally at {mid}"

    # waveform endpoint returns strikes + segments for the timeline
    wf = client.get(f"/api/jobs/{job_id}/waveform").json()
    assert wf["duration"] > 0 and len(wf["strikes"]) >= 10
    assert len(wf["segments"]) == r["n_rallies"]

    # media is actually served
    assert client.get(job["media"]["output"]).status_code == 200
    assert client.get(job["media"]["metadata_download"]).status_code == 200

    # manual edit: keep a single explicit segment, re-cut
    edited = client.post(f"/api/jobs/{job_id}/segments", json={"segments": [[12.0, 18.0]]})
    assert edited.status_code == 200
    ej = edited.json()
    assert ej["result"]["n_rallies"] == 1
    assert ej["result"]["edited"] is True
    assert abs(ej["result"]["kept_seconds"] - 6.0) < 0.2
    assert ej["media"]["output"]  # re-rendered

    # delete
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_roster_helpers_and_suggestion():
    assert [r["id"] for r in webapp._roster_for("singles")] == ["P1", "P2"]
    assert [r["id"] for r in webapp._roster_for("doubles")] == ["P1", "P2", "P3", "P4"]
    region = (0.2, 0.5, 0.8, 1.0)  # mid_y = 0.75, mid_x = 0.5
    assert webapp._suggest_player(0.3, 0.9, region, "singles") == "P1"   # near
    assert webapp._suggest_player(0.3, 0.55, region, "singles") == "P2"  # far
    assert webapp._suggest_player(0.3, 0.9, region, "doubles") == "P1"   # near-left
    assert webapp._suggest_player(0.7, 0.55, region, "doubles") == "P4"  # far-right


def test_labeling_generation_and_endpoints(scratch, monkeypatch):
    src = _make_video(scratch)
    # deterministic 2-person detection — no YOLO / network needed
    monkeypatch.setattr(
        webapp, "_detect_boxes",
        lambda frame, conf=0.3: [(50.0, 120.0, 110.0, 220.0), (180.0, 40.0, 240.0, 140.0)],
    )
    job_id = str(uuid.uuid4())
    jd = webapp._job_dir(job_id)
    jd.mkdir(parents=True)
    orig = jd / "original.mp4"
    shutil.copy(src, orig)
    webapp._atomic_write_json(webapp._job_meta_path(job_id), {
        "id": job_id, "status": "complete", "filename": "match.mp4",
        "original_path": str(orig), "progress": [], "labeling": {"status": "idle"},
        "result": {"total_seconds": DURATION,
                   "segments": [{"index": 0, "start": 12.0, "end": 20.0},
                                {"index": 1, "start": 32.0, "end": 38.0}]},
    })

    req = webapp.LabelTaskRequest(kinds=["player_identity", "serve_motion"],
                                  max_items=4, match_type="auto", regenerate=True)
    webapp._run_label_gen(job_id, req)

    job = webapp._load_job(job_id)
    assert job["labeling"]["status"] == "ready", job["labeling"]

    client = TestClient(webapp.app)
    data = client.get(f"/api/jobs/{job_id}/label-tasks").json()
    assert [r["id"] for r in data["roster"]] == ["P1", "P2"]     # auto -> singles
    players = [t for t in data["tasks"] if t["kind"] == "player_identity"]
    serves = [t for t in data["tasks"] if t["kind"] == "serve_motion"]
    assert len(players) == 4
    assert 1 <= len(serves) <= 4
    assert players[0]["suggested_player"] in {"P1", "P2"}
    assert players[0]["media_type"] == "image" and serves[0]["media_type"] == "video"

    # assets actually served
    assert client.get(players[0]["asset_url"]).status_code == 200
    assert client.get(serves[0]["asset_url"]).status_code == 200
    # path traversal on assets is blocked
    assert client.get(f"/api/jobs/{job_id}/assets/..%2F..%2Fjob.json").status_code == 404

    # save a player label + rename roster
    assert client.post(f"/api/jobs/{job_id}/labels", json={
        "task_id": players[0]["id"], "kind": "player_identity",
        "values": {"player": "P1", "quality": "clear"}}).status_code == 200
    assert client.post(f"/api/jobs/{job_id}/labels", json={
        "task_id": serves[0]["id"], "kind": "serve_motion",
        "values": {"is_serve": "yes", "server": "P1", "side": "deuce", "end": "near"}}).status_code == 200
    ros = client.post(f"/api/jobs/{job_id}/roster",
                      json={"roster": [{"id": "P1", "name": "Alice"}, {"id": "P2", "name": "Bob"}]}).json()
    assert ros["roster"][0]["name"] == "Alice"

    # export bundles roster + tasks + labels
    exp = json.loads(client.get(f"/api/jobs/{job_id}/labels/download").content)
    assert exp["labels"][players[0]["id"]]["values"]["player"] == "P1"
    assert exp["labels"][serves[0]["id"]]["values"]["is_serve"] == "yes"
    assert any(r["name"] == "Alice" for r in exp["roster"])


def test_label_kind_validation(scratch):
    src = _make_video(scratch)
    client = TestClient(webapp.app)
    with open(src, "rb") as fh:
        job = client.post("/api/jobs", files={"file": ("m.mp4", fh, "video/mp4")},
                          data={"run_now": "false"}).json()
    r = client.post(f"/api/jobs/{job['id']}/label-tasks", json={"kinds": ["bogus"]})
    assert r.status_code == 400
