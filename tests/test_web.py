"""Tests for the rally web UI.

Unit tests cover the pure request/response logic (config mapping, progress
monotonicity, segment normalisation, safety). One end-to-end test drives the
whole HTTP surface — upload → process → poll → edit segments → re-cut — against
a synthesised video, and auto-skips in sandboxes that can't run ffmpeg on
absolute paths (same guard the core integration test uses).
"""

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")
pytest.importorskip("scipy")

from fastapi.testclient import TestClient  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import rally.web.app as webapp  # noqa: E402
from rally.io.ffmpeg import _require, _video_encoder  # noqa: E402


def _ffmpeg_or_skip():
    """Resolve a working ffmpeg (skipping broken PATH shims), or skip the test."""
    try:
        return _require("ffmpeg")
    except Exception:
        pytest.skip("ffmpeg not available")


# --------------------------------------------------------------------------- #
# unit tests (no ffmpeg / no video)                                           #
# --------------------------------------------------------------------------- #
def test_default_session_directories_share_visible_root():
    assert webapp.SESSIONS_DIR == webapp.PROJECT_DIR / "sessions"
    assert webapp.DEFAULT_UPLOADS_DIR == webapp.SESSIONS_DIR / "uploads"
    assert webapp.DEFAULT_GOLDEN_RESULTS_DIR == webapp.SESSIONS_DIR / "golden"


def test_default_upload_limit_is_ten_gib(monkeypatch):
    monkeypatch.delenv("RALLY_WEB_MAX_UPLOAD_BYTES", raising=False)
    assert webapp._max_upload_bytes() == 10 * 1024 ** 3


def test_server_lifespan_requires_preflight_before_recovery(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "rally.preflight.require_server_install", lambda: calls.append("preflight"))
    monkeypatch.setattr(
        webapp, "_recover_jobs_on_startup", lambda: calls.append("recovery"))

    async def enter_lifespan():
        async with webapp._lifespan(webapp.app):
            calls.append("running")

    asyncio.run(enter_lifespan())
    assert calls == ["preflight", "recovery", "running"]


def test_server_entrypoint_refuses_incomplete_setup(monkeypatch, capsys):
    from rally.preflight import InstallationError

    def incomplete():
        raise InstallationError("missing TrackNet")

    monkeypatch.setattr("rally.preflight.require_server_install", incomplete)
    assert webapp.main([]) == 1
    assert "missing TrackNet" in capsys.readouterr().err


def test_recommended_web_workers_use_cpu_and_cuda_headroom():
    gib = 1024 ** 3
    assert webapp._recommended_web_workers(22, 80 * gib) == 4
    assert webapp._recommended_web_workers(8, 24 * gib) == 2
    assert webapp._recommended_web_workers(32, 9 * gib) == 1
    assert webapp._recommended_web_workers(32, None) == 1


def test_web_worker_count_honours_and_validates_override(monkeypatch):
    monkeypatch.setenv("RALLY_WEB_WORKERS", "3")
    assert webapp._web_worker_count() == 3
    monkeypatch.setenv("RALLY_WEB_WORKERS", "0")
    with pytest.raises(RuntimeError, match="must be positive"):
        webapp._web_worker_count()


def test_web_label_detector_defaults_to_yolo12_and_honours_overrides(monkeypatch):
    monkeypatch.delenv("RALLY_WEB_YOLO", raising=False)
    monkeypatch.delenv("RALLY_YOLO_DETECTION_MODEL", raising=False)
    assert webapp._web_yolo_model_name() == "yolo12n.pt"
    monkeypatch.setenv("RALLY_YOLO_DETECTION_MODEL", "shared.pt")
    assert webapp._web_yolo_model_name() == "shared.pt"
    monkeypatch.setenv("RALLY_WEB_YOLO", "labels.pt")
    assert webapp._web_yolo_model_name() == "labels.pt"
    monkeypatch.setenv("RALLY_WEB_WORKERS", "many")
    with pytest.raises(RuntimeError, match="must be an integer"):
        webapp._web_worker_count()


def test_upload_ui_requires_selection_and_supports_concurrent_files():
    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="videoFile"' in html and "multiple hidden" in html
    assert "required hidden" not in html
    assert 'alert("Please select at least one local video before uploading.")' in script
    assert "for (const file of files) uploadJob(file" in script
    assert "activeUploads: new Map()" in script
    assert '$("#uploadButton").disabled = true' not in script


def test_label_generation_ui_offers_independent_player_and_serve_options():
    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="labKindPlayers"' in html and "Classify players" in html
    assert 'id="labKindServe"' in html and "Generate serve motion" in html
    assert 'kinds.push("player_identity")' in script
    assert 'kinds.push("serve_motion")' in script
    assert "if (!kinds.length)" in script


def test_reprocess_ui_immediately_replaces_stale_output_with_progress_preview():
    script = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "showReprocessingState(state.current);" in script
    assert "media.output = null;" in script
    assert "result: null," in script
    assert 'selectVideoTab("processed");' in script
    assert '$("#timelineMeta").textContent = "Reprocessing — previous analysis removed"' in script


def test_golden_ui_is_separate_from_uploaded_jobs():
    html = (webapp.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (webapp.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert 'id="goldenButton"' in html
    assert 'id="goldenView"' in html
    assert 'api("/api/golden")' in script
    assert "These are pipeline evaluation runs, not uploaded jobs" in html


def test_golden_api_lists_only_root_labeled_pairs(monkeypatch, tmp_path):
    golden = tmp_path / "golden"
    results = tmp_path / "results"
    golden.mkdir()
    (golden / "input_1.mp4").write_bytes(b"input")
    (golden / "res_1.txt").write_text("Point 1: 1-2\n", encoding="utf-8")
    (golden / "input_without_label.mp4").write_bytes(b"ignored")
    nested = golden / "unlabeled"
    nested.mkdir()
    (nested / "input_2.mp4").write_bytes(b"ignored")
    (nested / "res_2.txt").write_text("Point 1: 3-4\n", encoding="utf-8")
    result = results / "input_1"
    result.mkdir(parents=True)
    (result / "rallies.json").write_text(
        json.dumps({"n_rallies": 1, "total_seconds": 3.0}), encoding="utf-8")
    (result / "rallies.mp4").write_bytes(b"output")
    monkeypatch.setattr(webapp, "GOLDEN_DIR", golden)
    monkeypatch.setattr(webapp, "GOLDEN_RESULTS_DIR", results)

    client = TestClient(webapp.app)
    body = client.get("/api/golden").json()
    assert body["total"] == 1
    assert body["datasets"][0]["id"] == "input_1"
    assert body["datasets"][0]["expected_points"] == 1
    assert body["datasets"][0]["predicted_points"] == 1
    assert client.get("/api/golden/input_1/media/input").content == b"input"
    assert client.get("/api/golden/input_1/media/output").content == b"output"
    assert client.get("/api/golden/input_2/media/input").status_code == 404


def test_marking_labels_stale_preserves_saved_revision():
    job = {
        "labeling": {
            "status": "ready", "revision": "saved-revision",
            "counts": {"serve_motion": 9},
        },
    }
    webapp._mark_labeling_stale(job)
    assert job["labeling"]["status"] == "stale"
    assert job["labeling"]["revision"] == "saved-revision"
    assert job["labeling"]["counts"] == {"serve_motion": 9}


def test_stale_label_revision_is_exportable_but_read_only(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    root = webapp._job_dir(job_id) / "label_revisions" / "rev-1" / "labels"
    root.mkdir(parents=True)
    webapp._atomic_write_json(root / "tasks.json", [])
    webapp._atomic_write_json(root / "roster.json", [{"id": "P1", "name": "Player"}])
    webapp._atomic_write_json(root / "labels.json", {})
    webapp._atomic_write_json(webapp._job_meta_path(job_id), {
        "id": job_id, "filename": "match.mp4", "status": "complete",
        "labeling": {"status": "stale", "revision": "rev-1"},
        "result": {"segments": [], "strike_times": [], "stages": {}},
    })
    client = TestClient(webapp.app)

    assert client.get(f"/api/jobs/{job_id}/labels/download").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/label-tasks").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/roster", json={
        "revision": "rev-1",
        "roster": [{"id": "P1", "name": "Changed"}],
    }).status_code == 409


def test_config_from_options_maps_flags_and_numbers():
    cfg = webapp._config_from_options(
        {"play_mode": "casual", "static_camera": True, "hysteresis": True,
         "fast": True, "no_labels": True,
         "min_rally": 3.0, "serve_preroll": 2.0, "gap": 0.2}
    )
    assert cfg.w_audio == 0.7 and cfg.rhythm_window_s == 5.0   # static-camera preset
    assert cfg.strike_snr_ratio == 5.5                           # recover quiet far hits
    assert cfg.use_dp_decoder is False                          # hysteresis
    assert cfg.reencode is False                                # fast
    assert cfg.label_points is False                            # no_labels
    assert cfg.play_mode == "casual"
    assert cfg.min_rally_s == 3.0
    assert cfg.serve_preroll_s == 2.0 and cfg.toss_preroll_s == 2.0
    assert cfg.inter_point_gap_s == 0.2


def test_default_transitions_use_real_footage_not_black_frames():
    cfg = webapp._config_from_options({})
    assert cfg.inter_point_gap_s == 0.0
    assert cfg.landing_tail_s == 1.0
    assert cfg.ball_tail_s == 1.0
    assert cfg.point_start_buffer_s == 1.0
    assert cfg.point_end_buffer_s == 1.0
    with pytest.raises(ValueError, match="landing_tail_s must be <= 1 second"):
        webapp.RallyConfig(landing_tail_s=1.01)
    with pytest.raises(ValueError, match="ball_tail_s must be <= 1 second"):
        webapp.RallyConfig(ball_tail_s=1.01)
    with pytest.raises(ValueError, match="point_end_buffer_s must be <= 1 second"):
        webapp.RallyConfig(point_end_buffer_s=1.01)
    with pytest.raises(ValueError, match="point_start_buffer_s must be <= 1 second"):
        webapp.RallyConfig(point_start_buffer_s=1.01)


def test_real_postroll_is_capped_at_next_point_and_video_end():
    assert webapp.add_real_postroll(
        [(1.0, 2.0), (2.5, 3.0), (9.5, 9.8)], 10.0, 1.0
    ) == [(1.0, 2.5), (2.5, 4.0), (9.5, 10.0)]


def test_real_context_includes_setup_and_splits_short_between_point_gap():
    assert webapp.add_real_context(
        [(1.0, 2.0), (3.0, 4.0), (7.0, 8.0)], 9.0, 1.0, 1.0
    ) == [(0.0, 2.5), (2.5, 5.0), (6.0, 9.0)]


def test_config_from_options_ball_arbiter_defaults_on():
    # Required evidence cannot be disabled by stale/forged web options.
    default = webapp._config_from_options({})
    assert default.ball_arbiter is True and default.court_auto is True
    off = webapp._config_from_options({"ball_arbiter": False, "court_auto": False})
    assert off.ball_arbiter is True and off.court_auto is True
    assert off.match_auto_fail_closed is True

    upgraded = webapp._required_web_options({
        "ball_arbiter": False, "court_auto": False, "detect_players": False,
    })
    assert upgraded["ball_arbiter"] is True
    assert upgraded["court_auto"] is True
    assert upgraded["detect_players"] is True


def test_capabilities_reports_ball_arbiter_availability(monkeypatch):
    client = TestClient(webapp.app)

    # no weights present -> not available, with a hint
    monkeypatch.setattr("rally.signals.ball.discover_ball_weights", lambda *a, **k: None)
    caps = client.get("/api/capabilities").json()
    assert caps["ball_arbiter"]["available"] is False
    assert caps["ball_arbiter"]["weights_present"] is False
    assert caps["ball_arbiter"]["hint"]                     # tells the user what to do
    assert caps["court_auto"]["available"] is True          # classical, always on
    assert set(caps["pose"]) >= {
        "available", "model_present", "device", "execution_providers", "cuda",
    }

    # weights present + torch installed -> available
    monkeypatch.setattr("rally.signals.ball.discover_ball_weights",
                        lambda *a, **k: "models/tracknet.pt")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    caps2 = client.get("/api/capabilities").json()
    assert caps2["ball_arbiter"]["available"] is True
    assert "weights_path" not in caps2["ball_arbiter"]


def test_upload_route_rejects_fake_video_bytes(monkeypatch, tmp_path):
    """An allowed filename extension is not sufficient media validation."""
    from pathlib import Path
    monkeypatch.setattr(webapp, "DATA_DIR", Path(tmp_path) / "data")
    client = TestClient(webapp.app)
    resp = client.post(
        "/api/jobs",
        files={"file": ("m.mp4", b"\x00\x01\x02\x03fakevideobytes", "video/mp4")},
        data={"run_now": "false", "ball_arbiter": "true", "court_auto": "true"},
    )
    assert resp.status_code == 400
    assert not list((Path(tmp_path) / "data").glob("*/job.json"))


def test_stage_for_message_recognises_pipeline_lines():
    assert webapp._stage_for_message("rendering 12 points -> out.mp4")["stage"] == "rendering"
    assert webapp._stage_for_message("processing failed: boom")["percent"] == 100
    assert webapp._stage_for_message("  123 strikes detected")["stage"] == "audio"
    assert webapp._stage_for_message(
        "ball tracking progress 50% (100/200s, batch 16)")["percent"] == 64
    assert webapp._stage_for_message("match pose progress 5/10")["percent"] == 79
    anchor = webapp._stage_for_message(
        "court serve detection: reusing 1200 visual-pass player samples")
    assert anchor["percent"] == 68
    assert webapp._stage_for_message(
        "serve validation progress 10/10 (10 reused TrackNet window(s))")["percent"] == 88
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


def test_append_progress_does_not_overwrite_terminal_state(monkeypatch, tmp_path):
    job_id = str(uuid.uuid4())
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    (tmp_path / job_id).mkdir()
    webapp._atomic_write_json(
        webapp._job_meta_path(job_id),
        {
            "id": job_id,
            "status": "complete",
            "progress": [],
            "processing": {
                "stage": "complete",
                "label": "Ready",
                "percent": 100,
                "detail": "9 rallies — output ready",
            },
        },
    )

    webapp._append_progress(job_id, "wrote output")

    job = webapp._read_json(webapp._job_meta_path(job_id), {})
    assert job["processing"]["stage"] == "complete"
    assert job["processing"]["percent"] == 100
    assert job["processing"]["detail"] == "9 rallies — output ready"
    assert job["progress"][-1]["message"] == "wrote output"


def test_normalise_segments_clips_sorts_and_drops_empty():
    segs = webapp._normalise_segments([[10, 14], [-2, 3], [50, 500], [8, 8]], duration=100)
    assert segs == [(0.0, 3.0), (10.0, 14.0), (50.0, 100.0)]  # clipped, sorted, zero-len dropped


def test_normalise_segments_rejects_bad_shape():
    with pytest.raises(Exception):
        webapp._normalise_segments([[1, 2, 3]], duration=100)


def test_normalise_segments_coalesces_overlaps_for_true_kept_time():
    segs = webapp._normalise_segments([[10, 20], [5, 12], [19, 25], [30, 31]], duration=40)
    assert segs == [(5.0, 25.0), (30.0, 31.0)]


def _write_job(data_dir: Path, **updates):
    job_id = str(uuid.uuid4())
    job_dir = data_dir / job_id
    job_dir.mkdir(parents=True)
    original = job_dir / "original.mp4"
    original.write_bytes(b"video")
    job = {
        "id": job_id,
        "created_at": webapp._now(),
        "updated_at": webapp._now(),
        "status": "complete",
        "filename": "match.mp4",
        "original_path": str(original),
        "thumbnail_path": None,
        "output_path": None,
        "json_path": None,
        "options": {},
        "progress": [],
        "processing": {},
        "labeling": {"status": "idle"},
        "result": {"total_seconds": 60.0, "segments": []},
        "error": None,
    }
    job.update(updates)
    webapp._atomic_write_json(job_dir / "job.json", job)
    return job_id, job_dir, job


def test_thumbnail_mutation_preserves_progress_added_during_slow_work(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, _job_dir, stale = _write_job(tmp_path)
    monkeypatch.setattr(webapp, "probe", lambda _path: SimpleNamespace(duration_s=10.0))

    def fake_frame(_src, _time, dst):
        webapp._append_progress(job_id, "progress arrived while thumbnail was rendering")
        dst.write_bytes(b"jpeg")

    monkeypatch.setattr(webapp, "_ffmpeg_frame", fake_frame)
    updated = webapp._ensure_thumbnail(stale)
    assert updated["thumbnail_path"]
    assert updated["progress"][-1]["message"].startswith("progress arrived")


def test_render_output_publishes_atomically_and_keeps_old_file_on_failure(monkeypatch, tmp_path):
    src = tmp_path / "source.mp4"
    dst = tmp_path / "rallies.mp4"
    src.write_bytes(b"source")
    dst.write_bytes(b"old-complete-video")
    cfg = webapp.RallyConfig(label_points=False, inter_point_gap_s=0, reencode=False)
    info = SimpleNamespace(height=720, has_audio=True, duration_s=10.0)

    def successful_cut(_src, _segments, temporary, reencode, cancel_check=None):
        assert _segments == [(0.0, 3.0)]
        assert Path(temporary) != dst
        assert dst.read_bytes() == b"old-complete-video"
        Path(temporary).write_bytes(b"new-complete-video")

    monkeypatch.setattr(webapp, "cut_segments", successful_cut)
    assert webapp._render_output(src, [(1, 2)], dst, cfg, info, lambda _m: None)
    assert dst.read_bytes() == b"new-complete-video"

    def failed_cut(_src, _segments, temporary, reencode, cancel_check=None):
        Path(temporary).write_bytes(b"partial")
        raise RuntimeError("encoder died")

    monkeypatch.setattr(webapp, "cut_segments", failed_cut)
    assert not webapp._render_output(src, [(2, 3)], dst, cfg, info, lambda _m: None)
    assert dst.read_bytes() == b"new-complete-video"
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_unlabelled_reencode_uses_one_pass_renderer(monkeypatch, tmp_path):
    src = tmp_path / "source.mp4"
    dst = tmp_path / "rallies.mp4"
    src.write_bytes(b"source")
    cfg = webapp.RallyConfig(label_points=False, inter_point_gap_s=0, reencode=True)
    info = SimpleNamespace(height=720, has_audio=True, duration_s=10.0)
    calls = []

    def render(_src, segments, temporary, **kwargs):
        calls.append((segments, kwargs))
        Path(temporary).write_bytes(b"one-pass")

    monkeypatch.setattr(webapp, "render_labeled", render)
    monkeypatch.setattr(
        webapp, "cut_segments",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("multi-process cut path must not run")),
    )

    assert webapp._render_output(src, [(1, 2), (4, 5)], dst, cfg, info, lambda _m: None)
    assert dst.read_bytes() == b"one-pass"
    assert len(calls) == 1
    assert calls[0][1]["draw_labels"] is False
    assert calls[0][1]["gap_s"] == 0


def test_zero_segment_edit_unpublishes_and_removes_stale_output(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, job_dir, job = _write_job(tmp_path)
    output = job_dir / "output" / "rallies.mp4"
    metadata = job_dir / "output" / "rallies.json"
    output.parent.mkdir()
    output.write_bytes(b"stale-video")
    webapp._atomic_write_json(metadata, job["result"])
    job.update(output_path=str(output), json_path=str(metadata))
    webapp._atomic_write_json(job_dir / "job.json", job)

    response = TestClient(webapp.app).post(f"/api/jobs/{job_id}/segments", json={"segments": []})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_output"
    assert body["processing"]["stage"] == "no_output"
    assert body["result"]["n_rallies"] == 0
    assert body["result"]["kept_seconds"] == 0
    assert body["media"]["output"] is None
    assert not output.exists()


def test_overlapping_segment_edit_renders_union_once(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, job_dir, _job = _write_job(tmp_path, status="failed", error="old failure")
    monkeypatch.setattr(webapp, "probe", lambda _path: SimpleNamespace(height=720, has_audio=True))
    rendered = []

    def fake_render(_src, segments, dst, _cfg, _info, _progress):
        rendered.extend(segments)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"edited-video")
        return True

    monkeypatch.setattr(webapp, "_render_output", fake_render)
    response = TestClient(webapp.app).post(
        f"/api/jobs/{job_id}/segments",
        json={"segments": [[10, 20], [5, 12], [18, 25]]},
    )
    assert response.status_code == 200
    assert rendered == [(5.0, 25.0)]
    assert response.json()["result"]["kept_seconds"] == 20.0
    assert response.json()["result"]["n_rallies"] == 1
    assert response.json()["status"] == "complete"
    assert response.json()["processing"]["stage"] == "complete"
    assert response.json()["result"]["input"] == "match.mp4"
    assert response.json()["result"]["output"] == "match_rallies.mp4"
    published = webapp._load_job(job_id)
    assert webapp._read_json(Path(published["json_path"]), {})["output"] == \
        "match_rallies.mp4"
    assert Path(published["output_path"]).exists()


def test_failed_segment_edit_retains_previous_published_result(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, job_dir, job = _write_job(tmp_path)
    output = job_dir / "output" / "rallies.mp4"
    metadata = job_dir / "output" / "rallies.json"
    output.parent.mkdir()
    output.write_bytes(b"known-good")
    webapp._atomic_write_json(metadata, job["result"])
    job.update(output_path=str(output), json_path=str(metadata))
    webapp._atomic_write_json(job_dir / "job.json", job)
    monkeypatch.setattr(webapp, "probe", lambda _path: SimpleNamespace(height=720))
    monkeypatch.setattr(webapp, "_render_output", lambda *_args, **_kwargs: False)

    response = TestClient(webapp.app, raise_server_exceptions=False).post(
        f"/api/jobs/{job_id}/segments", json={"segments": [[1.0, 2.0]]})

    assert response.status_code == 500
    retained = webapp._load_job(job_id)
    assert retained["output_path"] == str(output)
    assert retained["json_path"] == str(metadata)
    assert output.read_bytes() == b"known-good"


def test_web_sidecar_removes_absolute_host_paths():
    job = {"filename": "match.mp4"}
    clean = webapp._normalise_web_sidecar(
        job,
        {
            "input": "/srv/private/jobs/id/original.mp4",
            "output": None,
            "config": {"ball_weights": "/srv/private/models/tracknet.pt"},
        },
        output_ready=True,
    )
    assert clean == {
        "input": "match.mp4",
        "output": "match_rallies.mp4",
        "config": {"ball_weights": "tracknet.pt"},
    }


def test_web_sidecar_exposes_point_aware_output_layout_and_speed():
    job = {"filename": "match.mp4", "options": {}}
    clean = webapp._normalise_web_sidecar(
        job,
        {
            "total_seconds": 20.0,
            "segments": [
                {"index": 0, "start": 2.0, "end": 4.0, "duration": 2.0},
                {"index": 1, "start": 8.0, "end": 10.0, "duration": 2.0},
            ],
            "stages": {"ball_arbiter": {"verification": {"candidates": [{
                "output": [2.0, 4.0], "peak_ball_speed_kmh": 123.4,
                "ball_speed_estimate": {
                    "value_kmh": 123.4, "uncertainty_kmh": 42.0,
                    "uncertain": True,
                    "method": "single_camera_ground_plane_p95",
                    "limitations": ["ball height is not recovered"],
                },
            }]}}},
        },
        output_ready=True,
    )
    assert clean["output_layout"] == [
        {
            "index": 0, "source_start": 1.0, "source_end": 5.0,
            "detected_start": 2.0, "detected_end": 4.0,
            "output_start": 0.0, "output_end": 4.0,
            "peak_ball_speed_kmh": 123.4,
            "ball_speed_estimate": {
                "value_kmh": 123.4, "uncertainty_kmh": 42.0,
                "uncertain": True,
                "method": "single_camera_ground_plane_p95",
                "limitations": ["ball height is not recovered"],
            },
        },
        {
            "index": 1, "source_start": 7.0, "source_end": 11.0,
            "detected_start": 8.0, "detected_end": 10.0,
            "output_start": 4.0, "output_end": 8.0,
            "peak_ball_speed_kmh": None,
            "ball_speed_estimate": None,
        },
    ]


def test_legacy_api_and_metadata_download_are_sanitized(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, job_dir, job = _write_job(tmp_path)
    output = job_dir / "output" / "rallies.mp4"
    metadata = job_dir / "output" / "rallies.json"
    output.parent.mkdir()
    output.write_bytes(b"video")
    legacy = {
        "input": "/srv/private/original.mp4",
        "output": None,
        "total_seconds": 60.0,
        "segments": [],
        "config": {"ball_weights": "/srv/private/tracknet.pt"},
    }
    webapp._atomic_write_json(metadata, legacy)
    job.update(result=legacy, output_path=str(output), json_path=str(metadata))
    webapp._atomic_write_json(job_dir / "job.json", job)
    client = TestClient(webapp.app)

    public = client.get(f"/api/jobs/{job_id}").json()
    downloaded = client.get(public["media"]["metadata_download"]).json()

    assert public["result"]["input"] == "match.mp4"
    assert public["result"]["output"] == "match_rallies.mp4"
    assert downloaded["input"] == "match.mp4"
    assert downloaded["output"] == "match_rallies.mp4"
    assert downloaded["config"]["ball_weights"] == "tracknet.pt"
    assert webapp._read_json(metadata, None) == legacy


def test_rerun_with_no_segments_clears_previous_video(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    job_id, job_dir, job = _write_job(tmp_path)
    output = job_dir / "output" / "rallies.mp4"
    output.parent.mkdir()
    output.write_bytes(b"old-run")
    job["output_path"] = str(output)
    webapp._atomic_write_json(job_dir / "job.json", job)
    result = SimpleNamespace(
        total_seconds=60.0,
        segments=[],
        sidecar=lambda: {"total_seconds": 60.0, "segments": [], "strike_times": []},
    )
    monkeypatch.setattr(webapp, "trim", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        webapp, "probe",
        lambda _path: SimpleNamespace(fps=30.0, width=1920, height=1080, has_audio=True),
    )
    monkeypatch.setattr(webapp, "iter_audio_mono", lambda *_a, **_k: pytest.fail("decoded audio twice"))

    webapp._run_trim_job(job_id)
    saved = webapp._load_job(job_id)
    assert saved["status"] == "no_output"
    assert saved["processing"]["stage"] == "no_output"
    assert saved["output_path"] is None
    assert saved["result"]["segments"] == []
    assert saved["result"]["input"] == "match.mp4"
    assert saved["result"]["output"] is None
    assert not output.exists()


def test_terminal_but_active_reprocess_returns_retryable_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    job_id, _job_dir, _job = _write_job(tmp_path, status="complete")
    webapp._ACTIVE.add(job_id)

    response = TestClient(webapp.app).post(f"/api/jobs/{job_id}/process")

    assert response.status_code == 409
    assert "retry" in response.json()["detail"]
    assert webapp._load_job(job_id)["status"] == "complete"


def test_startup_recovers_queue_and_marks_interrupted_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_LABEL_ACTIVE", set())
    monkeypatch.setattr(webapp, "_LABEL_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_UPLOAD_RESERVED", set())
    monkeypatch.setattr(webapp, "_UPLOAD_RESERVED_BYTES", {})
    monkeypatch.setattr(webapp, "_RECOVERED_DATA_DIRS", set())
    queued_id, _queued_dir, _ = _write_job(tmp_path, status="queued")
    running_id, running_dir, running_job = _write_job(tmp_path, status="running")
    retained_output = running_dir / "output" / "rallies.mp4"
    retained_json = running_dir / "output" / "rallies.json"
    retained_output.parent.mkdir()
    retained_output.write_bytes(b"previous-video")
    retained_json.write_text("{}")
    running_job.update(
        output_path=str(retained_output), json_path=str(retained_json), result={"segments": []})
    webapp._atomic_write_json(running_dir / "job.json", running_job)
    live_id, _live_dir, _ = _write_job(tmp_path, status="running")
    labeling_id, labeling_dir, labeling_job = _write_job(tmp_path, status="complete")
    labeling_job["labeling"] = {"status": "generating", "detail": "Queued"}
    webapp._atomic_write_json(labeling_dir / "job.json", labeling_job)
    webapp._ACTIVE.add(live_id)

    class RecordingExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args):
            self.calls.append((function, args))
            return Future()

    executor = RecordingExecutor()
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)

    recovered = webapp._recover_interrupted_jobs()
    recovered_again = webapp._recover_interrupted_jobs()

    assert recovered == {"queued": 1, "interrupted": 1}
    assert recovered_again == {"queued": 0, "interrupted": 0}
    assert len(executor.calls) == 1
    submitted_job_id, submitted_attempt_id = executor.calls[0][1]
    assert submitted_job_id == queued_id
    assert submitted_attempt_id == webapp._load_job(queued_id)["active_attempt_id"]
    interrupted = webapp._load_job(running_id)
    assert interrupted["status"] == "failed"
    assert interrupted["retryable"] is True
    assert interrupted["processing"]["stage"] == "failed"
    assert interrupted["output_path"] == str(retained_output)
    assert interrupted["json_path"] == str(retained_json)
    assert "retained" in interrupted["processing"]["label"].lower()
    assert webapp._load_job(live_id)["status"] == "running"
    labeling = webapp._load_job(labeling_id)["labeling"]
    assert labeling["status"] == "failed"
    assert "server restart" in labeling["error"]


def test_duplicate_process_submissions_enqueue_only_once(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_EDIT_ACTIVE", set())
    job_id, _job_dir, _job = _write_job(tmp_path, status="uploaded")

    class RecordingExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args):
            self.calls.append((function, args))
            return Future()

    executor = RecordingExecutor()
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)
    client = TestClient(webapp.app)
    assert client.post(f"/api/jobs/{job_id}/process").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/process").status_code == 200
    assert len(executor.calls) == 1


def test_reprocess_immediately_deletes_and_unpublishes_previous_result(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_CANCEL_EVENTS", {})
    monkeypatch.setattr(webapp, "_JOB_FUTURES", {})
    job_id, job_dir, job = _write_job(tmp_path, status="complete")
    output = job_dir / "output" / "rallies.mp4"
    metadata = job_dir / "output" / "rallies.json"
    waveform = job_dir / "waveform.json"
    thumbnail = job_dir / "thumbnail.jpg"
    output.parent.mkdir()
    output.write_bytes(b"last-good-video")
    webapp._atomic_write_json(metadata, {"segments": [{"start": 1, "end": 2}]})
    webapp._atomic_write_json(waveform, {"duration": 60, "strikes": [1.5]})
    thumbnail.write_bytes(b"preview")
    job.update(
        output_path=str(output),
        json_path=str(metadata),
        thumbnail_path=str(thumbnail),
        result={"segments": [{"start": 1, "end": 2}], "n_rallies": 1},
    )
    webapp._atomic_write_json(job_dir / "job.json", job)

    class PendingExecutor:
        def __init__(self):
            self.future = Future()

        def submit(self, *_args):
            return self.future

    monkeypatch.setattr(webapp, "_EXECUTOR", PendingExecutor())
    client = TestClient(webapp.app)
    before = client.get(f"/api/jobs/{job_id}").json()
    old_output_url = before["media"]["output"]
    old_metadata_url = before["media"]["metadata_download"]

    response = client.post(f"/api/jobs/{job_id}/process")

    assert response.status_code == 200
    queued = response.json()
    assert queued["status"] == "queued"
    assert queued["result"] is None
    assert queued["media"]["output"] is None
    assert queued["media"]["metadata_download"] is None
    assert queued["media"]["thumbnail"] is not None
    assert not output.exists()
    assert not metadata.exists()
    assert not waveform.exists()
    assert client.get(old_output_url).status_code == 404
    assert client.get(old_metadata_url).status_code == 404


def test_cancel_queued_job_removes_it_before_worker_start(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_CANCEL_EVENTS", {})
    monkeypatch.setattr(webapp, "_JOB_FUTURES", {})
    job_id, _job_dir, _job = _write_job(tmp_path, status="uploaded")

    class PendingExecutor:
        def __init__(self):
            self.future = Future()

        def submit(self, *_args):
            return self.future

    executor = PendingExecutor()
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)
    client = TestClient(webapp.app)
    assert client.post(f"/api/jobs/{job_id}/process").status_code == 200

    response = client.post(f"/api/jobs/{job_id}/cancel")

    assert response.status_code == 200
    job = response.json()
    assert executor.future.cancelled()
    assert job["status"] == "cancelled"
    assert job["processing"]["stage"] == "cancelled"
    assert job["retryable"] is True
    assert job_id not in webapp._SUBMITTED


def test_cancelled_rerun_does_not_restore_previous_result(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_CANCEL_EVENTS", {})
    monkeypatch.setattr(webapp, "_JOB_FUTURES", {})
    job_id, job_dir, job = _write_job(tmp_path, status="complete")
    output = job_dir / "output" / "rallies.mp4"
    metadata = job_dir / "output" / "rallies.json"
    output.parent.mkdir()
    output.write_bytes(b"last-good-video")
    result = {"segments": [{"start": 1.0, "end": 2.0}], "n_rallies": 1}
    webapp._atomic_write_json(metadata, result)
    job.update(output_path=str(output), json_path=str(metadata), result=result)
    webapp._atomic_write_json(job_dir / "job.json", job)

    class PendingExecutor:
        def __init__(self):
            self.future = Future()

        def submit(self, *_args):
            return self.future

    executor = PendingExecutor()
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)
    client = TestClient(webapp.app)
    assert client.post(f"/api/jobs/{job_id}/process").status_code == 200

    response = client.post(f"/api/jobs/{job_id}/cancel")

    saved = response.json()
    assert saved["status"] == "cancelled"
    assert saved["result"] is None
    assert saved["media"]["output"] is None
    assert saved["media"]["metadata_download"] is None
    assert not output.exists()
    assert not metadata.exists()


def test_cancel_running_job_stops_cooperatively_and_cleans_state(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_ACTIVE", set())
    monkeypatch.setattr(webapp, "_SUBMITTED", set())
    monkeypatch.setattr(webapp, "_CANCEL_EVENTS", {})
    monkeypatch.setattr(webapp, "_JOB_FUTURES", {})
    monkeypatch.setattr(webapp, "_ensure_thumbnail", lambda job: job)
    monkeypatch.setattr(webapp, "_archive_label_artifacts", lambda _job_id: None)
    job_id, _job_dir, _job = _write_job(tmp_path, status="uploaded")
    entered = threading.Event()

    def blocking_trim(*_args, cancel_check, **_kwargs):
        entered.set()
        while True:
            cancel_check()
            time.sleep(0.01)

    monkeypatch.setattr(webapp, "trim", blocking_trim)
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)
    client = TestClient(webapp.app)
    try:
        assert client.post(f"/api/jobs/{job_id}/process").status_code == 200
        assert entered.wait(timeout=2)
        response = client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 200
        deadline = time.time() + 2
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "cancelled" and job_id not in webapp._ACTIVE:
                break
            time.sleep(0.02)
        assert job["status"] == "cancelled"
        assert job["processing"]["stage"] == "cancelled"
        assert job["error"] is None
        assert job_id not in webapp._ACTIVE
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_duplicate_label_submissions_enqueue_only_once(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_LABEL_ACTIVE", set())
    monkeypatch.setattr(webapp, "_LABEL_SUBMITTED", set())
    job_id, _job_dir, _job = _write_job(tmp_path)

    class RecordingExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args):
            self.calls.append((function, args))
            return SimpleNamespace()

    executor = RecordingExecutor()
    monkeypatch.setattr(webapp, "_EXECUTOR", executor)
    client = TestClient(webapp.app)
    assert client.post(f"/api/jobs/{job_id}/label-tasks", json={"max_items": 1}).status_code == 200
    assert client.post(f"/api/jobs/{job_id}/label-tasks", json={"max_items": 1}).status_code == 200
    assert len(executor.calls) == 1


def test_waveform_uses_pipeline_strike_times_without_audio_decode(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id, job_dir, _job = _write_job(tmp_path)
    monkeypatch.setattr(webapp, "iter_audio_mono", lambda *_a, **_k: pytest.fail("audio decoded"))
    webapp._write_waveform(
        job_id, job_dir / "original.mp4", 30.0, webapp.RallyConfig(), lambda _m: None,
        strike_times=[1.25, 2.5],
    )
    assert webapp._read_json(job_dir / "waveform.json", {}) == {
        "duration": 30.0, "strikes": [1.25, 2.5],
    }


def test_invalid_options_and_oversized_upload_leave_no_jobs(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(webapp, "DATA_DIR", data_dir)
    client = TestClient(webapp.app)
    invalid = client.post(
        "/api/jobs",
        files={"file": ("m.mp4", b"video", "video/mp4")},
        data={"run_now": "false", "analysis_fps": "0"},
    )
    assert invalid.status_code == 400
    assert not data_dir.exists()

    monkeypatch.setenv("RALLY_WEB_MAX_UPLOAD_BYTES", "3")
    oversized = client.post(
        "/api/jobs",
        files={"file": ("m.mp4", b"four", "video/mp4")},
        data={"run_now": "false"},
    )
    assert oversized.status_code == 413
    assert list(data_dir.iterdir()) == []


def test_label_task_count_is_bounded():
    job_id = str(uuid.uuid4())
    response = TestClient(webapp.app).post(
        f"/api/jobs/{job_id}/label-tasks",
        json={"max_items": webapp._MAX_LABEL_ITEMS + 1},
    )
    assert response.status_code == 422


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
    ffmpeg = _require("ffmpeg")
    vcodec, vargs = _video_encoder()
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s=320x240:r=25:d={DURATION}", "-i", audio,
         "-c:v", vcodec, *vargs, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src],
        check=True,
    )
    return src


@pytest.fixture
def scratch(monkeypatch):
    ffmpeg = _ffmpeg_or_skip()
    local = os.path.join(os.path.dirname(__file__), "_scratch_web")
    os.makedirs(local, exist_ok=True)
    # sandbox guard: verify a child ffmpeg can read an absolute path we wrote
    probe = os.path.join(local, "_probe.wav")
    wavfile.write(probe, SR, np.zeros(SR // 10, dtype=np.int16))
    rc = subprocess.run([ffmpeg, "-v", "error", "-i", os.path.abspath(probe), "-f", "null", "-"],
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


def test_end_to_end_upload_process_edit(scratch, monkeypatch):
    src = _make_video(scratch)
    client = TestClient(webapp.app)

    # The production web path mandates every model. This HTTP/render lifecycle test uses
    # a synthetic black video with no tennis geometry, so replace only its trim call with
    # the deliberately reduced library configuration.
    real_trim = webapp.trim

    def reduced_trim(input_path, output_path=None, cfg=None, **kwargs):
        kwargs["detect_players"] = False
        return real_trim(
            input_path, output_path=output_path,
            cfg=webapp.RallyConfig(
                analysis_fps=5.0, min_rally_s=1.0, ball_arbiter=False,
                court_auto=False, play_mode="casual"),
            **kwargs,
        )

    monkeypatch.setattr(webapp, "trim", reduced_trim)

    with open(src, "rb") as fh:
        resp = client.post(
            "/api/jobs",
            files={"file": ("match.mp4", fh, "video/mp4")},
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
    revision = data["revision"]
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
        "revision": revision,
        "task_id": players[0]["id"], "kind": "player_identity",
        "values": {"player": "P1", "quality": "clear"}}).status_code == 200
    assert client.post(f"/api/jobs/{job_id}/labels", json={
        "revision": revision,
        "task_id": serves[0]["id"], "kind": "serve_motion",
        "values": {"is_serve": "yes", "server": "P1", "side": "deuce", "end": "near"}}).status_code == 200
    ros = client.post(f"/api/jobs/{job_id}/roster",
                      json={"revision": revision, "roster": [{"id": "P1", "name": "Alice"}, {"id": "P2", "name": "Bob"}]}).json()
    assert ros["roster"][0]["name"] == "Alice"
    assert client.post(f"/api/jobs/{job_id}/roster", json={
        "revision": revision,
        "roster": [{"id": "x\" onmouseover=alert(1)", "name": "bad"}],
    }).status_code == 422
    assert client.post(f"/api/jobs/{job_id}/labels", json={
        "revision": revision,
        "task_id": serves[0]["id"], "kind": "serve_motion",
        "values": {"is_serve": "definitely"},
    }).status_code == 422

    # export bundles roster + tasks + labels
    exp = json.loads(client.get(f"/api/jobs/{job_id}/labels/download").content)
    assert exp["schema_version"] == "rally.web_labels.v2"
    assert exp["labels"][players[0]["id"]]["values"]["player"] == "P1"
    assert exp["labels"][serves[0]["id"]]["values"]["is_serve"] == "yes"
    assert any(r["name"] == "Alice" for r in exp["roster"])
    assert exp["feature_context"] == {
        "schema_version": "rally.serve_rule_context.v1",
        "strike_times": [],
        "segments": [{"index": 0, "start": 12.0, "end": 20.0},
                     {"index": 1, "start": 32.0, "end": 38.0}],
        "match_state": {},
    }


def test_serve_only_generation_uses_automatically_detected_durable_roster(
        monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "DATA_DIR", tmp_path)
    job_id = str(uuid.uuid4())
    root = webapp._job_dir(job_id)
    root.mkdir(parents=True)
    original = root / "original.mp4"
    original.write_bytes(b"placeholder")
    roster = [
        {"id": f"P{index}", "name": name,
         "team_id": "T1" if index <= 2 else "T2"}
        for index, name in enumerate(("Alice", "Bob", "Carol", "Dana"), 1)
    ]
    webapp._atomic_write_json(webapp._job_meta_path(job_id), {
        "id": job_id, "status": "complete", "filename": "match.mp4",
        "original_path": str(original), "progress": [],
        "labeling": {"status": "idle"},
        "match": {"format": "doubles", "roster": roster},
        "result": {"total_seconds": 20.0, "segments": [
            {"index": 0, "start": 3.0, "end": 8.0}], "stages": {}},
    })
    monkeypatch.setattr(webapp, "_generate_serve_tasks", lambda *args, **kwargs: [{
        "id": "serve_0000", "kind": "serve_motion", "title": "Serve clip 1",
        "time_s": 3.0, "media_type": "video", "asset_url": "/clip.mp4",
    }])

    webapp._run_label_gen(job_id, webapp.LabelTaskRequest(
        kinds=["serve_motion"], max_items=4, regenerate=True))

    job = webapp._load_job(job_id)
    revision = job["labeling"]["revision"]
    labels_root = root / "label_revisions" / revision / "labels"
    assert job["labeling"]["match_type"] == "doubles"
    assert [record["name"] for record in webapp._read_json(
        labels_root / "roster.json", [])] == ["Alice", "Bob", "Carol", "Dana"]
    assert [task["kind"] for task in webapp._read_json(
        labels_root / "tasks.json", [])] == ["serve_motion"]


def test_label_kind_validation(scratch):
    src = _make_video(scratch)
    client = TestClient(webapp.app)
    with open(src, "rb") as fh:
        job = client.post("/api/jobs", files={"file": ("m.mp4", fh, "video/mp4")},
                          data={"run_now": "false"}).json()
    r = client.post(f"/api/jobs/{job['id']}/label-tasks", json={"kinds": ["bogus"]})
    assert r.status_code == 400
