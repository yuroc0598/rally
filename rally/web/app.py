"""FastAPI composition root for processing jobs and the review UI.

Analysis remains in :mod:`rally.pipeline`; this module owns the mutable job lifecycle,
upload API, media publication, editing, and human-label workflows.

Design notes
------------
* **One job = one directory** under ``DATA_DIR/<uuid>/`` holding the upload,
  ``job.json`` (the single source of truth), the rendered output, the JSON
  sidecar, a thumbnail, and a cached waveform for the timeline.
* **Concurrency**: a bounded ``ThreadPoolExecutor`` runs trims; every read /
  modify / write of ``job.json`` is guarded by a process-wide ``RLock`` and
  written atomically (``os.replace``), so a crashed worker can never leave a
  half-written status file and the poller always sees a consistent job.
* **Robust rendering**: the pipeline is run analysis-only (``output_path=None``)
  so a segment list is always produced even when ffmpeg can't re-encode; the
  video is then rendered separately with graceful fallback (labelled render →
  plain cut → stream-copy). This also powers *re-export* after manual edits.
* **Safety**: job ids must be valid UUIDs (blocks path traversal) and served
  files are confined to their job directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rally.config import DEFAULT_YOLO_DETECTION_MODEL, RallyConfig
from rally.io.ffmpeg import (
    _require,
    add_real_context,
    add_real_postroll,
    cut_segments,
    find_font,
    iter_audio_mono,
    probe,
    render_labeled,
)
from rally.pipeline import timeline_array, trim
from rally.signals.audio import detect_strikes_stream
from rally.signals.player import estimate_court_region
from rally.web.golden import discover_datasets, resolve_media_path
from rally.web.schemas import (
    MAX_LABEL_ITEMS as _MAX_LABEL_ITEMS,
    LabelPayload,
    LabelTaskRequest,
    RosterUpdate,
    SegmentEdit,
)

# --------------------------------------------------------------------------- #
# module state                                                                #
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
STATIC_DIR = PACKAGE_DIR / "static"
SESSIONS_DIR = PROJECT_DIR / "sessions"
DEFAULT_UPLOADS_DIR = SESSIONS_DIR / "uploads"
DEFAULT_GOLDEN_RESULTS_DIR = SESSIONS_DIR / "golden"
DATA_DIR = Path(os.environ.get(
    "RALLY_WEB_DATA", DEFAULT_UPLOADS_DIR)).resolve()
GOLDEN_DIR = Path(os.environ.get(
    "RALLY_GOLDEN_DATA", PROJECT_DIR / "samples" / "golden")).resolve()
GOLDEN_RESULTS_DIR = Path(os.environ.get(
    "RALLY_GOLDEN_RESULTS", DEFAULT_GOLDEN_RESULTS_DIR)).resolve()


def _recommended_web_workers(cpu_count: int | None, cuda_free_bytes: int | None) -> int:
    """Choose conservative job-level concurrency for one CUDA device.

    One accuracy-mode trim uses several CPU decode/association threads and roughly 4--5
    GiB of device memory on the production models.  Budgeting four CPU cores and 10 GiB
    of currently free VRAM per worker leaves headroom for model peaks and video rendering.
    The small cap avoids turning a large shared GPU into an unbounded local job server.
    """
    if cuda_free_bytes is None:
        return 1
    cpu_slots = max(1, int(cpu_count or 1) // 4)
    gpu_slots = max(1, int(cuda_free_bytes) // (10 * 1024 ** 3))
    return max(1, min(4, cpu_slots, gpu_slots))


def _web_worker_count() -> int:
    """Resolve an explicit worker count or auto-size it from CPU and free CUDA VRAM."""
    raw = os.environ.get("RALLY_WEB_WORKERS")
    if raw is not None:
        try:
            workers = int(raw)
        except ValueError as exc:
            raise RuntimeError("RALLY_WEB_WORKERS must be an integer") from exc
        if workers <= 0:
            raise RuntimeError("RALLY_WEB_WORKERS must be positive")
        return workers

    if os.environ.get("RALLY_DEVICE", "").strip().lower() == "cpu":
        return 1
    cuda_free_bytes: int | None = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda_free_bytes = int(torch.cuda.mem_get_info()[0])
    except Exception:
        # Web startup must remain available on CPU-only or partially configured systems.
        cuda_free_bytes = None
    return _recommended_web_workers(os.cpu_count(), cuda_free_bytes)


_LOCK = threading.RLock()
_WEB_WORKERS = _web_worker_count()
_EXECUTOR = ThreadPoolExecutor(max_workers=_WEB_WORKERS, thread_name_prefix="rally-job")
_ACTIVE: set[str] = set()
_SUBMITTED: set[str] = set()
_EDIT_ACTIVE: set[str] = set()
_LABEL_ACTIVE: set[str] = set()
_LABEL_SUBMITTED: set[str] = set()
_UPLOAD_RESERVED: set[str] = set()
_UPLOAD_RESERVED_BYTES: dict[str, int] = {}
_RECOVERED_DATA_DIRS: set[Path] = set()
_CANCEL_EVENTS: dict[str, tuple[str, threading.Event]] = {}
_JOB_FUTURES: dict[str, tuple[str, Any]] = {}

_YOLO_LOCK = threading.Lock()
_YOLO_MODEL: Any = None

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_LABEL_KINDS = {"player_identity", "serve_motion"}
_ROSTER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,49}$")
_DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
_MAX_JSON_BODY_BYTES = 2 * 1024 * 1024


class _JobCancelled(RuntimeError):
    """Cooperative stop requested by the owner of a web processing job."""

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _recover_jobs_on_startup()
    yield


app = FastAPI(title="Rally — rally trimmer", lifespan=_lifespan)


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _scavenge_stale_uploads() -> None:
    """Remove abandoned temporary uploads once, during startup recovery only."""
    _ensure_data_dir()
    cutoff = datetime.now(timezone.utc).timestamp() - 3600
    for path in DATA_DIR.glob(".upload-*"):
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
        except OSError:
            pass


def _safe_filename(name: str | None) -> str:
    raw = Path(name or "upload.mp4").name
    stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in raw).strip(" .")
    return stem or "upload.mp4"


def _golden_datasets() -> list[dict[str, Any]]:
    return discover_datasets(
        GOLDEN_DIR, GOLDEN_RESULTS_DIR, video_extensions=_VIDEO_EXTS)


def _job_dir(job_id: str) -> Path:
    """Resolve a job directory, rejecting anything that isn't a real UUID.

    Validating the id here is what keeps ``job_id`` out of the filesystem as an
    attacker-controlled path segment (no ``../`` traversal).
    """
    try:
        uuid.UUID(job_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return DATA_DIR / job_id


def _job_meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_job(job_id: str) -> dict[str, Any]:
    path = _job_meta_path(job_id)
    with _LOCK:
        job = _read_json(path, None)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _mutate_job(job_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Reload and mutate one job while holding the lock.

    Callers must describe only the fields they intend to change.  This avoids a
    slow thumbnail/render/analysis path writing an old whole-job snapshot over
    progress, labels, or another route's newer changes.
    """
    with _LOCK:
        path = _job_meta_path(job_id)
        job = _read_json(path, None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        mutate(job)
        job["updated_at"] = _now()
        _atomic_write_json(path, job)
        return job


def _create_job(job: dict[str, Any]) -> None:
    """Persist a new job. Existing jobs must use :func:`_mutate_job`."""
    job["updated_at"] = _now()
    with _LOCK:
        path = _job_meta_path(job["id"])
        if path.exists():
            raise RuntimeError(f"job already exists: {job['id']}")
        _atomic_write_json(path, job)


def _mark_labeling_stale(job: dict[str, Any]) -> None:
    # Keep the immutable revision pointer and counts so human work remains recoverable and
    # downloadable after a re-analysis.  It is marked stale—not deleted—because its serve
    # clips refer to the prior segment boundaries.
    labeling = job.setdefault("labeling", {})
    labeling.update(
        status="stale",
        detail="Rally segments changed; saved labels retained; regenerate samples to relabel",
        updated_at=_now(),
    )


def _prune_label_revisions(job_id: str, keep: int = 2) -> None:
    root = _job_dir(job_id) / "label_revisions"
    if not root.exists():
        return
    revisions = sorted((p for p in root.iterdir() if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for old in revisions[keep:]:
        shutil.rmtree(old)


def _archive_label_artifacts(job_id: str) -> None:
    """Preserve, but unpublish, labels/assets tied to a previous segment revision."""
    archive_root = _job_dir(job_id) / "label_archive"
    revision = archive_root / uuid.uuid4().hex
    moved = False
    for name in ("labels", "label_assets"):
        source = _job_dir(job_id) / name
        if not source.exists():
            continue
        revision.mkdir(parents=True, exist_ok=True)
        os.replace(source, revision / name)
        moved = True
    if not moved:
        _prune_label_revisions(job_id)
        return
    # Keep only the two most recent recoverable revisions; clips can be large and repeated
    # edits must not consume disk forever.
    revisions = sorted((p for p in archive_root.iterdir() if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for old in revisions[2:]:
        shutil.rmtree(old)
    _prune_label_revisions(job_id)


# --------------------------------------------------------------------------- #
# progress: map pipeline log lines to a coarse stage + monotonic percent      #
# --------------------------------------------------------------------------- #
_STAGE_RULES = [
    ("processing stopped", "cancelled", "Stopped", 100),
    ("processing failed", "failed", "Failed", 100),
    ("no rally segments found", "no_output", "No rallies found", 100),
    ("upload complete", "uploaded", "Uploaded", 5),
    ("queued for processing", "queued", "Queued", 8),
    ("processing started", "starting", "Starting", 12),
    ("probing", "probing", "Reading video", 18),
    ("duration=", "probing", "Reading video", 22),
    ("decoding audio", "audio", "Detecting ball strikes", 32),
    ("strikes detected", "audio", "Detecting ball strikes", 40),
    ("sampling frames", "visual", "Analysing motion & players", 45),
    ("co-deciding", "deciding", "Building candidates", 52),
    ("ball arbiter", "ball_tracking", "Tracking the ball", 54),
    ("match-state validation: checking server pose", "pose", "Checking serve poses", 76),
    ("match-state validation: checking stationary", "pose", "Checking player setup", 82),
    ("match-state validation: checking ball", "serve", "Validating serves", 83),
    ("decoded", "deciding", "Points found", 89),
    ("court serve detection", "refining", "Refining serves", 89),
    ("ball point-end", "refining", "Refining rally ends", 89),
    ("computing waveform", "waveform", "Building timeline", 90),
    ("rendering", "rendering", "Rendering video", 92),
    ("cutting", "rendering", "Rendering video", 92),
    ("wrote", "writing", "Writing output", 95),
]


def _stage_for_message(message: str) -> dict[str, Any]:
    text = message.lower()
    tracking = re.search(r"ball tracking progress\s+(\d+)%", text)
    if tracking:
        completed = max(0, min(100, int(tracking.group(1))))
        return {
            "stage": "ball_tracking", "label": "Tracking the ball",
            "percent": 54 + int(round(0.21 * completed)),
        }
    pose = re.search(r"match pose progress\s+(\d+)\s*/\s*(\d+)", text)
    if pose:
        done, total = int(pose.group(1)), max(1, int(pose.group(2)))
        return {
            "stage": "pose", "label": "Checking serve poses",
            "percent": 76 + int(round(0.06 * min(100, 100 * done / total))),
        }
    serves = re.search(r"serve validation progress\s+(\d+)\s*/\s*(\d+)", text)
    if serves:
        done, total = int(serves.group(1)), max(1, int(serves.group(2)))
        return {
            "stage": "serve", "label": "Validating serves",
            "percent": 83 + int(round(0.05 * min(100, 100 * done / total))),
        }
    for needle, stage, label, percent in _STAGE_RULES:
        if needle in text:
            return {"stage": stage, "label": label, "percent": percent}
    return {"stage": "running", "label": "Processing", "percent": 50}


def _set_processing(job: dict[str, Any], stage: str, label: str, percent: int, detail: str) -> None:
    job["processing"] = {
        "stage": stage,
        "label": label,
        "percent": max(0, min(100, int(percent))),
        "detail": detail,
        "updated_at": _now(),
    }


def _append_progress(job_id: str, message: str, *, attempt_id: str | None = None) -> None:
    """Append a log line and advance the coarse progress bar (never backwards)."""
    message = str(message)
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            return
        if attempt_id is not None and job.get("active_attempt_id") != attempt_id:
            return
        log = job.setdefault("progress", [])
        log.append({"at": _now(), "message": message})
        del log[:-400]

        nxt = _stage_for_message(message)
        prev = job.get("processing") or {}
        # Late informational messages (for example, "wrote output") may arrive
        # after the worker has published its terminal state.  Keep the log, but
        # never replace Ready/Failed/No output with an earlier-looking stage.
        if prev.get("stage") in {"complete", "failed", "no_output", "cancelled"}:
            _atomic_write_json(_job_meta_path(job_id), job)
            return
        prev_pct = int(prev.get("percent") or 0)
        # keep the bar monotonic while the job is running: an unrecognised line
        # (percent 50) or a lower-percent stage must not drag it back.
        if nxt["stage"] == "running" or nxt["percent"] < prev_pct:
            if prev and prev.get("stage") not in {
                    "complete", "failed", "no_output", "cancelled"}:
                nxt = {"stage": prev.get("stage", "running"),
                       "label": prev.get("label", "Processing"),
                       "percent": prev_pct}
        _set_processing(job, nxt["stage"], nxt["label"], nxt["percent"], message)
        _atomic_write_json(_job_meta_path(job_id), job)


# --------------------------------------------------------------------------- #
# public view + media URLs                                                     #
# --------------------------------------------------------------------------- #
def _media_url(job_id: str, kind: str, path: Path, *, download: bool = False) -> str:
    version = path.stat().st_mtime_ns  # rapid atomic replacements need sub-second cache busting
    q = f"v={version}" + ("&download=1" if download else "")
    return f"/api/jobs/{job_id}/media/{kind}?{q}"


def _media_urls(job: dict[str, Any]) -> dict[str, str | None]:
    job_id = job["id"]
    urls: dict[str, str | None] = {
        "original": None, "thumbnail": None,
        "output": None, "output_download": None, "metadata_download": None,
    }
    for kind, key in (("original", "original_path"), ("output", "output_path"),
                      ("metadata", "json_path"), ("thumbnail", "thumbnail_path")):
        p = job.get(key)
        path = Path(p) if p else None
        if path and path.exists():
            try:
                url = _media_url(job_id, kind, path,
                                 download=kind == "metadata")
            except FileNotFoundError:
                continue
            if kind == "thumbnail":
                urls["thumbnail"] = url
            elif kind == "metadata":
                urls["metadata_download"] = url
            else:
                urls[kind] = url
                if kind == "output":
                    try:
                        urls["output_download"] = _media_url(
                            job_id, "output", path, download=True)
                    except FileNotFoundError:
                        urls["output"] = None
    return urls


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal ``*_path`` fields and attach browser-facing media URLs."""
    if not job:
        return {}
    data = {k: v for k, v in job.items() if not k.endswith("_path")}
    if isinstance(data.get("result"), dict):
        output = Path(job["output_path"]) if job.get("output_path") else None
        data["result"] = _normalise_web_sidecar(
            job, data["result"], output_ready=bool(output and output.exists()))
    if data.get("error") and len(str(data["error"])) > 1500:
        data["error"] = str(data["error"])[:1500].rstrip() + " ..."
    data["media"] = _media_urls(job)
    return data


def _normalise_web_sidecar(job: dict[str, Any], sidecar: dict[str, Any], *,
                           output_ready: bool) -> dict[str, Any]:
    """Return metadata that describes web-published files without host paths."""
    filename = _safe_filename(job.get("filename"))
    clean = dict(sidecar)
    clean["input"] = filename
    clean["output"] = f"{Path(filename).stem}_rallies.mp4" if output_ready else None
    if output_ready and clean.get("segments"):
        cfg = _config_from_options(job.get("options") or {})
        points = [(float(item["start"]), float(item["end"]))
                  for item in clean["segments"]]
        total_s = float(clean.get("total_seconds") or 0.0)
        clips = add_real_context(
            points, total_s, cfg.point_start_buffer_s, cfg.point_end_buffer_s)

        speed_candidates = (((clean.get("stages") or {}).get("ball_arbiter") or {})
                            .get("verification") or {}).get("candidates") or []

        def speed_for(point: tuple[float, float]) -> tuple[Optional[float], Optional[dict]]:
            best_overlap = 0.0
            best_speed: Optional[float] = None
            best_estimate: Optional[dict] = None
            for candidate in speed_candidates:
                output = candidate.get("output")
                speed = candidate.get("peak_ball_speed_kmh")
                if not output or speed is None:
                    continue
                overlap = max(0.0, min(point[1], float(output[1]))
                              - max(point[0], float(output[0])))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speed = float(speed)
                    estimate = candidate.get("ball_speed_estimate")
                    best_estimate = dict(estimate) if isinstance(estimate, dict) else None
            return best_speed, best_estimate

        cursor = 0.0
        layout = []
        gap_s = float(cfg.inter_point_gap_s)
        for index, (point, clip) in enumerate(zip(points, clips)):
            clip_duration = max(0.0, clip[1] - clip[0])
            speed, speed_estimate = speed_for(point)
            layout.append({
                "index": index,
                "source_start": round(clip[0], 3),
                "source_end": round(clip[1], 3),
                "detected_start": round(point[0], 3),
                "detected_end": round(point[1], 3),
                "output_start": round(cursor, 3),
                "output_end": round(cursor + clip_duration, 3),
                "peak_ball_speed_kmh": speed,
                "ball_speed_estimate": speed_estimate,
            })
            cursor += clip_duration
            if index < len(points) - 1:
                cursor += gap_s
        clean["output_layout"] = layout
    else:
        clean.pop("output_layout", None)

    def strip_absolute(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip_absolute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [strip_absolute(item) for item in value]
        if isinstance(value, tuple):
            return [strip_absolute(item) for item in value]
        if isinstance(value, str) and os.path.isabs(value):
            return Path(value).name
        return value

    return strip_absolute(clean)


# --------------------------------------------------------------------------- #
# config from web options (mirrors the CLI flags)                             #
# --------------------------------------------------------------------------- #
def _config_from_options(options: dict[str, Any]) -> RallyConfig:
    overrides: dict[str, Any] = {}
    if options.get("play_mode") is not None:
        overrides["play_mode"] = options["play_mode"]
    if options.get("static_camera"):
        overrides.update(
            w_audio=0.7, w_motion=0.1, rhythm_window_s=5.0,
            strike_snr_ratio=5.5,
        )
    mapping = {
        "analysis_fps": "analysis_fps",
        "min_rally": "min_rally_s",
        "skip_intro": "skip_intro_s",
        "gap": "inter_point_gap_s",
        "start_buffer": "point_start_buffer_s",
        "end_buffer": "point_end_buffer_s",
        "serve_preroll": "serve_preroll_s",
        "tail": "landing_tail_s",
    }
    for opt, cfg_name in mapping.items():
        val = options.get(opt)
        if val is not None:
            overrides[cfg_name] = val
            if opt == "serve_preroll":
                overrides["toss_preroll_s"] = val
    if options.get("no_labels"):
        overrides["label_points"] = False
    if options.get("hysteresis"):
        overrides["use_dp_decoder"] = False
    if options.get("fast"):
        overrides["reencode"] = False
    # ball-arbiter and court auto-detection are on by default (full trajectory path,
    # fallback); respect an explicitly unchecked box. Weights are auto-discovered in the
    # pipeline, which falls back to audio-primary if none are present.
    if "ball_arbiter" in options:
        overrides["ball_arbiter"] = bool(options["ball_arbiter"])
    if "court_auto" in options:
        overrides["court_auto"] = bool(options["court_auto"])
    return RallyConfig(**overrides)


def _validate_options(options: dict[str, Any]) -> None:
    """Reject invalid or pathological web options before accepting an upload."""
    non_negative = (
        "min_rally", "skip_intro", "gap", "start_buffer", "end_buffer",
        "serve_preroll", "tail")
    for name in non_negative:
        value = options.get(name)
        if value is not None and value < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be non-negative")
    analysis_fps = options.get("analysis_fps")
    if analysis_fps is not None and not 0 < analysis_fps <= 120:
        raise HTTPException(status_code=400, detail="analysis_fps must be in (0, 120]")
    try:
        _config_from_options(options)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid processing options: {exc}") from exc


def _max_upload_bytes() -> int:
    raw = os.environ.get("RALLY_WEB_MAX_UPLOAD_BYTES", str(_DEFAULT_MAX_UPLOAD_BYTES))
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError("RALLY_WEB_MAX_UPLOAD_BYTES must be an integer") from exc
    if limit <= 0:
        raise RuntimeError("RALLY_WEB_MAX_UPLOAD_BYTES must be positive")
    return limit


def _max_pending_jobs() -> int:
    try:
        value = int(os.environ.get("RALLY_WEB_MAX_PENDING_JOBS", "32"))
    except ValueError as exc:
        raise RuntimeError("RALLY_WEB_MAX_PENDING_JOBS must be an integer") from exc
    if value <= 0:
        raise RuntimeError("RALLY_WEB_MAX_PENDING_JOBS must be positive")
    return value


class _UploadTooLarge(Exception):
    pass


class _UploadLimitMiddleware:
    """Bound request bytes before Starlette buffers multipart or JSON bodies.

    The route still enforces the exact file-byte limit. This outer guard allows
    a small amount of multipart framing but prevents both chunked and declared
    requests from filling temporary storage before the endpoint runs.
    """

    _MULTIPART_OVERHEAD_BYTES = 1024 * 1024

    def __init__(self, inner_app):
        self.inner_app = inner_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.inner_app(scope, receive, send)
            return

        upload = scope.get("path") == "/api/jobs"
        file_limit = _max_upload_bytes() if upload else _MAX_JSON_BODY_BYTES
        request_limit = (file_limit + self._MULTIPART_OVERHEAD_BYTES) if upload else file_limit
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > request_limit:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds {file_limit} bytes"},
                )
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > request_limit:
                    raise _UploadTooLarge
            return message

        try:
            await self.inner_app(scope, limited_receive, send)
        except _UploadTooLarge:
            response = JSONResponse(
                status_code=413,
                content={"detail": f"request body exceeds {file_limit} bytes"},
            )
            await response(scope, receive, send)


app.add_middleware(_UploadLimitMiddleware)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid number: {value!r}") from exc
    if not math.isfinite(parsed):
        raise HTTPException(status_code=400, detail=f"invalid number: {value!r}")
    return parsed


def _max_jobs() -> int:
    value = int(os.environ.get("RALLY_WEB_MAX_JOBS", "1000"))
    if value <= 0:
        raise RuntimeError("RALLY_WEB_MAX_JOBS must be positive")
    return value


def _max_data_bytes() -> int:
    value = int(os.environ.get("RALLY_WEB_MAX_DATA_BYTES", str(100 * 1024 ** 3)))
    if value <= 0:
        raise RuntimeError("RALLY_WEB_MAX_DATA_BYTES must be positive")
    return value


def _storage_usage() -> tuple[int, int]:
    jobs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    used = 0
    for root, _dirs, files in os.walk(DATA_DIR):
        for name in files:
            try:
                used += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return len(jobs), used


def _probe_upload_isolated(path: Path, timeout_s: float = 150.0) -> dict[str, Any]:
    """Probe metadata and decode one frame in a killable child process."""
    code = (
        "import json,sys,cv2; from rally.io.ffmpeg import probe; "
        "p=sys.argv[1]; i=probe(p); c=cv2.VideoCapture(p); ok,fr=c.read(); c.release(); "
        "assert ok and fr is not None, 'could not decode a video frame'; "
        "print(json.dumps(dict(duration_s=i.duration_s,width=i.width,height=i.height)))"
    )
    try:
        done = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                              text=True, timeout=timeout_s, check=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"video probe exceeded {timeout_s:.0f}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "probe failed").strip()
        raise RuntimeError(detail[-1000:]) from exc
    return json.loads(done.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# ffmpeg helpers (thumbnail + rendering with fallback)                         #
# --------------------------------------------------------------------------- #
def _ffmpeg_frame(src: Path, time_s: float, dst: Path) -> None:
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, time_s):.3f}", "-i", str(src),
         "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(dst)],
        check=True, timeout=60,
    )


def _ensure_thumbnail(job: dict[str, Any]) -> dict[str, Any]:
    # The supplied object may have come from an unlocked directory listing. Work
    # from a fresh snapshot and later patch only thumbnail_path.
    try:
        job = _load_job(job["id"])
    except HTTPException:
        return job
    thumb = job.get("thumbnail_path")
    if thumb and Path(thumb).exists():
        return job
    original = job.get("original_path")
    if not original or not Path(original).exists():
        return job
    path = _job_dir(job["id"]) / "thumbnail.jpg"
    tmp = path.with_name(f".thumbnail.{uuid.uuid4().hex}.tmp.jpg")
    try:
        info = probe(original)
        _ffmpeg_frame(Path(original), min(max(info.duration_s / 2, 0.0), 3.0), tmp)
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError("thumbnail renderer produced no output")
        os.replace(tmp, path)
    except Exception:
        return job
    finally:
        tmp.unlink(missing_ok=True)
    try:
        return _mutate_job(job["id"], lambda current: current.update(thumbnail_path=str(path)))
    except HTTPException:
        return job


def _render_output(src: Path, segments: list[tuple[float, float]], dst: Path,
                   cfg: RallyConfig, info, progress,
                   cancel_check: Callable[[], None] = lambda: None) -> bool:
    """Render the trimmed video, degrading gracefully if ffmpeg lacks a feature.

    labelled render  →  plain re-encode cut  →  stream-copy cut. Returns True if
    a file was written. Analysis output is unaffected by a render failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real destination untouched until a complete render exists. The
    # suffix remains .mp4 so ffmpeg can infer the container from the temp name.
    tmp = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex}.tmp{dst.suffix}")
    render_segments = add_real_context(
        segments, info.duration_s,
        cfg.point_start_buffer_s, cfg.point_end_buffer_s)

    def publish_if_valid() -> bool:
        if not tmp.exists() or tmp.stat().st_size <= 0:
            raise RuntimeError("renderer produced no output")
        os.replace(tmp, dst)
        return True

    try:
        # Fast mode promises stream-copy; burned labels/gaps necessarily re-encode.
        if cfg.reencode:
            try:
                font = find_font() if cfg.label_points else None
                progress(f"rendering {len(segments)} points -> {dst.name}")
                render_labeled(str(src), render_segments, str(tmp), gap_s=cfg.inter_point_gap_s,
                               label_prefix=cfg.label_prefix, font=font,
                               video_height=info.height, has_audio=info.has_audio,
                               draw_labels=cfg.label_points,
                               cancel_check=cancel_check)
                return publish_if_valid()
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                progress(f"  one-pass render failed ({exc}); falling back to a plain cut")
        try:
            progress(f"cutting {len(segments)} segments -> {dst.name}")
            cut_segments(
                str(src), render_segments, str(tmp), reencode=cfg.reencode,
                cancel_check=cancel_check)
            return publish_if_valid()
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if cfg.reencode:
                progress(f"  re-encode cut failed ({exc}); trying a fast stream-copy")
                try:
                    cut_segments(
                        str(src), render_segments, str(tmp), reencode=False,
                        cancel_check=cancel_check)
                    return publish_if_valid()
                except Exception as exc2:
                    progress(f"  stream-copy cut failed too ({exc2})")
            else:
                progress(f"  cut failed ({exc})")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _write_waveform(job_id: str, src: Path, duration: float, cfg: RallyConfig, progress,
                    strike_times: list[float] | tuple[float, ...] | None = None,
                    destination: Path | None = None) -> None:
    """Cache ball-strike times (+ duration) so the review timeline can draw them.

    Prefer onsets already produced by analysis. Older pipeline results do not
    expose them, so retain the decode fallback for backward compatibility.
    """
    try:
        if strike_times is not None:
            progress("writing waveform from analysis strike times")
            strikes = [float(t) for t in strike_times if math.isfinite(float(t))]
        else:
            progress("computing waveform (strike timeline)")
            strikes = detect_strikes_stream(
                iter_audio_mono(str(src), cfg.audio_sr, chunk_s=60.0), cfg.audio_sr, cfg)
        data = {"duration": round(duration, 3), "strikes": [round(float(t), 3) for t in strikes]}
        _atomic_write_json(destination or (_job_dir(job_id) / "waveform.json"), data)
    except Exception as exc:
        progress(f"  waveform cache skipped ({exc})")


# --------------------------------------------------------------------------- #
# the worker                                                                   #
# --------------------------------------------------------------------------- #
def _set_job_cancelled(job: dict[str, Any], detail: str = "Stopped by user") -> None:
    output = Path(job["output_path"]) if job.get("output_path") else None
    metadata = Path(job["json_path"]) if job.get("json_path") else None
    if isinstance(job.get("result"), dict) and metadata is not None and metadata.exists():
        output_ready = bool(output is not None and output.exists())
        job["status"] = "complete" if output_ready else "no_output"
        job["retryable"] = True
        job["cancel_requested"] = False
        job["error"] = None
        _set_processing(
            job, job["status"], "Previous result retained", 100,
            f"{detail}; the last successful result is still available",
        )
        return
    percent = int((job.get("processing") or {}).get("percent") or 0)
    job["status"] = "cancelled"
    job["retryable"] = True
    job["cancel_requested"] = False
    job["error"] = None
    job["output_path"] = None
    job["json_path"] = None
    job["result"] = None
    _set_processing(job, "cancelled", "Stopped", percent, detail)


def _forget_job_future(job_id: str, attempt_id: str, future) -> None:
    with _LOCK:
        current = _JOB_FUTURES.get(job_id)
        if current == (attempt_id, future):
            _JOB_FUTURES.pop(job_id, None)


def _run_trim_job(job_id: str, attempt_id: str | None = None) -> None:
    if attempt_id is None:
        # Direct/internal callers predate durable attempt generations. Claim one so they
        # receive the same publication guarantees as executor-submitted work.
        attempt_id = uuid.uuid4().hex

        def claim(current: dict[str, Any]) -> None:
            current["active_attempt_id"] = attempt_id

        _mutate_job(job_id, claim)
    with _LOCK:
        if job_id in _ACTIVE:
            return
        cancel_entry = _CANCEL_EVENTS.get(job_id)
        if cancel_entry is None or cancel_entry[0] != attempt_id:
            cancel_entry = (attempt_id, threading.Event())
            _CANCEL_EVENTS[job_id] = cancel_entry
        cancel_event = cancel_entry[1]
        _SUBMITTED.discard(job_id)
        _ACTIVE.add(job_id)

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise _JobCancelled("processing stopped by user")

    try:
        check_cancel()
        job_dir = _job_dir(job_id)
        output_path = job_dir / "output" / "rallies.mp4"
        json_path = job_dir / "output" / "rallies.json"
        attempt_output_path = output_path.with_name(f".rallies.{attempt_id}.mp4")
        attempt_waveform_path = job_dir / f".waveform.{attempt_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def start(current: dict[str, Any]) -> None:
            if current.get("active_attempt_id") != attempt_id:
                return
            current["status"] = "running"
            current["cancel_requested"] = False
            current["error"] = None
            _mark_labeling_stale(current)
            _set_processing(current, "starting", "Starting", 12, "Preparing output files")

        job = _mutate_job(job_id, start)
        if job.get("active_attempt_id") != attempt_id:
            return
        _archive_label_artifacts(job_id)
        _append_progress(job_id, "processing started", attempt_id=attempt_id)
        # Publish an original-video still immediately so the primary processed-video
        # panel has useful visual context throughout analysis and rendering. This is
        # retained across re-runs; _ensure_thumbnail is a cheap no-op when it exists.
        _ensure_thumbnail(_load_job(job_id))

        def progress(message: str) -> None:
            check_cancel()
            _append_progress(job_id, message, attempt_id=attempt_id)
            check_cancel()

        options = job.get("options", {})
        cfg = _config_from_options(options)

        # Analysis only: always yields a segment list, even without ffmpeg encode.
        result = trim(job["original_path"], output_path=None, cfg=cfg, json_path=None,
                      detect_players=bool(options.get("detect_players", True)),
                      progress=progress, cancel_check=check_cancel)

        sidecar = result.sidecar()
        info = probe(job["original_path"])
        sidecar["info"] = {"fps": info.fps, "width": info.width,
                           "height": info.height, "has_audio": info.has_audio}

        strike_times = sidecar.get("strike_times")
        _write_waveform(job_id, Path(job["original_path"]), result.total_seconds, cfg, progress,
                        strike_times if isinstance(strike_times, (list, tuple)) else None,
                        destination=attempt_waveform_path)

        rendered = False
        if result.segments:
            rendered = _render_output(Path(job["original_path"]), result.segments,
                                      attempt_output_path, cfg, info, progress,
                                      cancel_check=check_cancel)

        check_cancel()
        output_ready = rendered and attempt_output_path.exists()
        sidecar = _normalise_web_sidecar(job, sidecar, output_ready=output_ready)
        # Generation-checked atomic publication: an attempt from a server process that
        # was superseded by a retry may finish later, but it cannot replace newer media,
        # metadata, progress, or terminal state.
        with _LOCK:
            current = _read_json(_job_meta_path(job_id), None)
            if not current or current.get("active_attempt_id") != attempt_id:
                return
            if output_ready:
                os.replace(attempt_output_path, output_path)
            else:
                output_path.unlink(missing_ok=True)
            if attempt_waveform_path.exists():
                os.replace(attempt_waveform_path, job_dir / "waveform.json")
            _atomic_write_json(json_path, sidecar)
            current["status"] = "complete" if output_ready else "no_output"
            current["retryable"] = False
            current["cancel_requested"] = False
            current["result"] = sidecar
            current["json_path"] = str(json_path)
            current["output_path"] = str(output_path) if output_ready else None
            if output_ready:
                _set_processing(current, "complete", "Ready", 100,
                                f"{len(result.segments)} rallies — output ready")
            elif not result.segments:
                _set_processing(current, "no_output", "No rallies found", 100,
                                "Processing finished but no rally segments were detected")
            else:
                _set_processing(current, "no_output", "Analysis only", 100,
                                "Segments detected but video export failed (check ffmpeg) — "
                                "JSON is available")
            current["updated_at"] = _now()
            _atomic_write_json(_job_meta_path(job_id), current)
        _ensure_thumbnail(_load_job(job_id))
        if output_ready:
            _append_progress(job_id, "wrote output", attempt_id=attempt_id)
    except Exception as exc:
        try:
            attempt_output_path.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        try:
            if cancel_event.is_set() or isinstance(exc, _JobCancelled):
                def stopped(current: dict[str, Any]) -> None:
                    if current.get("active_attempt_id") == attempt_id:
                        _set_job_cancelled(current)

                _mutate_job(job_id, stopped)
                _append_progress(
                    job_id, "processing stopped by user", attempt_id=attempt_id)
            else:
                def fail(current: dict[str, Any]) -> None:
                    if current.get("active_attempt_id") != attempt_id:
                        return
                    current["status"] = "failed"
                    current["retryable"] = True
                    current["cancel_requested"] = False
                    current["error"] = str(exc)
                    output = (Path(current["output_path"])
                              if current.get("output_path") else None)
                    metadata = (Path(current["json_path"])
                                if current.get("json_path") else None)
                    if (isinstance(current.get("result"), dict)
                            and metadata is not None and metadata.exists()):
                        current["status"] = (
                            "complete" if output is not None and output.exists()
                            else "no_output")
                        _set_processing(
                            current, current["status"], "Previous result retained", 100,
                            f"Re-run failed: {exc}",
                        )
                    else:
                        _set_processing(current, "failed", "Failed", 100, str(exc))

                _mutate_job(job_id, fail)
                _append_progress(
                    job_id, f"processing failed: {exc}", attempt_id=attempt_id)
        except HTTPException:
            pass
    finally:
        try:
            attempt_output_path.unlink(missing_ok=True)
            attempt_waveform_path.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        with _LOCK:
            _ACTIVE.discard(job_id)
            _SUBMITTED.discard(job_id)
            cancel_entry = _CANCEL_EVENTS.get(job_id)
            if cancel_entry is not None and cancel_entry[0] == attempt_id:
                _CANCEL_EVENTS.pop(job_id, None)


def _submit_job(job_id: str, *, reserved_upload: bool = False,
                _recovering: bool = False) -> dict[str, Any]:
    """Atomically reserve one executor slot per job and queue it once."""
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job_id in _EDIT_ACTIVE:
            raise HTTPException(status_code=409, detail="job segments are being edited")
        if job_id in _LABEL_SUBMITTED or job_id in _LABEL_ACTIVE:
            raise HTTPException(status_code=409, detail="job labeling samples are being generated")
        if job_id in _ACTIVE:
            # A worker publishes terminal state before doing independent thumbnail/log
            # cleanup. A new attempt cannot start until that worker leaves _ACTIVE; a 200
            # here would falsely claim the requested rerun was queued.
            if job.get("status") not in {"queued", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="processing worker is finalizing; retry shortly",
                )
            return job
        if job_id in _SUBMITTED:
            return job
        queued_work = len(_SUBMITTED | _ACTIVE | _LABEL_SUBMITTED | _LABEL_ACTIVE
                          | _UPLOAD_RESERVED)
        if not reserved_upload and not _recovering and queued_work >= _max_pending_jobs():
            raise HTTPException(status_code=503, detail="processing queue is full")
        if reserved_upload and job_id not in _UPLOAD_RESERVED:
            raise RuntimeError("upload queue reservation was lost")
        attempt_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        _CANCEL_EVENTS[job_id] = (attempt_id, cancel_event)
        job["active_attempt_id"] = attempt_id
        job["status"] = "queued"
        job["retryable"] = False
        job["cancel_requested"] = False
        job["error"] = None
        _mark_labeling_stale(job)
        job["updated_at"] = _now()
        _set_processing(job, "queued", "Queued", 8, "Waiting for a worker")
        _SUBMITTED.add(job_id)
        _UPLOAD_RESERVED.discard(job_id)
        _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        _atomic_write_json(_job_meta_path(job_id), job)

    try:
        _archive_label_artifacts(job_id)
        _append_progress(job_id, "queued for processing", attempt_id=attempt_id)
        future = _EXECUTOR.submit(_run_trim_job, job_id, attempt_id)
        with _LOCK:
            _JOB_FUTURES[job_id] = (attempt_id, future)
        future.add_done_callback(
            lambda completed, jid=job_id, aid=attempt_id: (
                _forget_job_future(jid, aid, completed)))
    except Exception as exc:
        with _LOCK:
            _SUBMITTED.discard(job_id)
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
            cancel_entry = _CANCEL_EVENTS.get(job_id)
            if cancel_entry is not None and cancel_entry[0] == attempt_id:
                _CANCEL_EVENTS.pop(job_id, None)
            _JOB_FUTURES.pop(job_id, None)

        def fail_submission(current: dict[str, Any]) -> None:
            current["status"] = "failed"
            current["retryable"] = True
            current["error"] = f"could not queue processing: {exc}"
            _set_processing(current, "failed", "Failed", 100, current["error"])

        _mutate_job(job_id, fail_submission)
        raise RuntimeError(f"could not queue processing: {exc}") from exc
    return _load_job(job_id)


def _recover_interrupted_jobs() -> dict[str, int]:
    """Recover durable queue state once per data directory after process startup.

    Queued work was accepted but never began, so it is safe to submit again. A persisted
    running job has lost its owning worker; mark it failed/retryable instead of guessing
    that partially written artifacts are valid. In-process live jobs are always skipped.
    """
    _scavenge_stale_uploads()
    root = DATA_DIR.resolve()
    queued: list[str] = []
    interrupted = 0
    with _LOCK:
        if root in _RECOVERED_DATA_DIRS:
            return {"queued": 0, "interrupted": 0}
        _RECOVERED_DATA_DIRS.add(root)
        for path in DATA_DIR.glob("*/job.json"):
            job = _read_json(path, None)
            if not job:
                continue
            job_id = str(job.get("id") or path.parent.name)
            if job_id in _ACTIVE or job_id in _SUBMITTED:
                continue
            labeling = job.get("labeling") or {}
            if (labeling.get("status") == "generating"
                    and job_id not in _LABEL_ACTIVE
                    and job_id not in _LABEL_SUBMITTED):
                message = "label generation was interrupted by a server restart; retry generation"
                labeling.update(
                    status="failed", error=message, detail=message, updated_at=_now())
                job["labeling"] = labeling
                job["updated_at"] = _now()
                _atomic_write_json(path, job)
            status = job.get("status")
            if status == "queued":
                queued.append(job_id)
            elif status == "running":
                message = "processing was interrupted by a server restart; retry processing"
                # Invalidate the former process's generation. Its non-daemon worker may
                # still unwind after the listener exits, but can no longer publish.
                job["active_attempt_id"] = None
                job["status"] = "failed"
                job["retryable"] = True
                job["error"] = message
                retained = bool(
                    isinstance(job.get("result"), dict)
                    and job.get("json_path")
                    and Path(job["json_path"]).is_file()
                )
                job["updated_at"] = _now()
                _set_processing(
                    job, "failed",
                    "Interrupted — previous result retained" if retained else "Interrupted",
                    100,
                    (f"{message}; the last successful result is still available"
                     if retained else message),
                )
                log = job.setdefault("progress", [])
                log.append({"at": _now(), "message": message})
                del log[:-400]
                _atomic_write_json(path, job)
                interrupted += 1

    submitted = 0
    for job_id in queued:
        try:
            _submit_job(job_id, _recovering=True)
            submitted += 1
        except Exception as exc:
            try:
                def fail_recovery(current: dict[str, Any]) -> None:
                    current["status"] = "failed"
                    current["retryable"] = True
                    current["error"] = f"could not recover queued processing: {exc}"
                    _set_processing(current, "failed", "Failed", 100, current["error"])

                _mutate_job(job_id, fail_recovery)
            except HTTPException:
                pass
    return {"queued": submitted, "interrupted": interrupted}


def _recover_jobs_on_startup() -> None:
    _recover_interrupted_jobs()


# --------------------------------------------------------------------------- #
# routes: pages + jobs                                                         #
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/jobs")
def list_jobs(limit: int = 100, offset: int = 0) -> JSONResponse:
    _ensure_data_dir()
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be 1..500 and offset non-negative")
    jobs = []
    paths = sorted(DATA_DIR.glob("*/job.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[offset:offset + limit]:
        job = _read_json(path, None)
        if not job:
            continue
        jobs.append(_public_job(job))
    return JSONResponse({"jobs": jobs, "total": len(paths), "limit": limit, "offset": offset})


@app.get("/api/golden")
def list_golden_datasets() -> JSONResponse:
    datasets = _golden_datasets()
    return JSONResponse({"datasets": datasets, "total": len(datasets)})


@app.get("/api/golden/{dataset_id}/media/{kind}")
def get_golden_media(dataset_id: str, kind: str, download: bool = False) -> FileResponse:
    path = resolve_media_path(
        dataset_id, kind, GOLDEN_DIR, GOLDEN_RESULTS_DIR,
        video_extensions=_VIDEO_EXTS)
    if path is None:
        raise HTTPException(status_code=404, detail="golden media not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    disposition = "attachment" if download or kind in {"metadata", "ground-truth"} else "inline"
    return FileResponse(
        path, media_type=media_type, filename=path.name,
        content_disposition_type=disposition)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    play_mode: str = Form("auto"),
    static_camera: bool = Form(False),
    detect_players: bool = Form(True),
    fast: bool = Form(False),
    hysteresis: bool = Form(False),
    no_labels: bool = Form(False),
    ball_arbiter: bool = Form(True),
    court_auto: bool = Form(True),
    run_now: bool = Form(True),
    analysis_fps: Optional[str] = Form(None),
    min_rally: Optional[str] = Form(None),
    skip_intro: Optional[str] = Form(None),
    gap: Optional[str] = Form(None),
    start_buffer: Optional[str] = Form(None),
    end_buffer: Optional[str] = Form(None),
    serve_preroll: Optional[str] = Form(None),
    tail: Optional[str] = Form(None),
) -> JSONResponse:
    # Parse and validate all options before creating a job directory or retaining
    # any upload bytes. Invalid forms must not leave orphan jobs behind.
    options = {
        "play_mode": play_mode,
        "static_camera": static_camera,
        "detect_players": detect_players,
        "fast": fast,
        "hysteresis": hysteresis,
        "no_labels": no_labels,
        "ball_arbiter": ball_arbiter,
        "court_auto": court_auto,
        "analysis_fps": _parse_optional_float(analysis_fps),
        "min_rally": _parse_optional_float(min_rally),
        "skip_intro": _parse_optional_float(skip_intro),
        "gap": _parse_optional_float(gap),
        "start_buffer": _parse_optional_float(start_buffer),
        "end_buffer": _parse_optional_float(end_buffer),
        "serve_preroll": _parse_optional_float(serve_preroll),
        "tail": _parse_optional_float(tail),
    }
    options = {k: v for k, v in options.items() if v is not None}
    _validate_options(options)

    _ensure_data_dir()
    job_id = str(uuid.uuid4())
    reserved = False
    with _LOCK:
        n_jobs, used_bytes = _storage_usage()
        if n_jobs >= _max_jobs():
            raise HTTPException(status_code=507, detail="job storage limit reached")
        available_bytes = _max_data_bytes() - used_bytes - sum(_UPLOAD_RESERVED_BYTES.values())
        if available_bytes <= 0:
            raise HTTPException(status_code=507, detail="data storage byte limit reached")
        byte_reservation = min(_max_upload_bytes(), available_bytes)
        if run_now:
            queued_work = len(_SUBMITTED | _ACTIVE | _LABEL_SUBMITTED | _LABEL_ACTIVE
                              | _UPLOAD_RESERVED)
            if queued_work >= _max_pending_jobs():
                raise HTTPException(status_code=503, detail="processing queue is full")
            _UPLOAD_RESERVED.add(job_id)
            reserved = True
        _UPLOAD_RESERVED_BYTES[job_id] = byte_reservation
    job_dir = _job_dir(job_id)
    staging_dir = DATA_DIR / f".upload-{job_id}"
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except BaseException:
        with _LOCK:
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        raise

    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in _VIDEO_EXTS:
        ext = ".mp4"
    original = staging_dir / f"original{ext}"
    uploaded = 0
    try:
        limit = byte_reservation
        with original.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                uploaded += len(chunk)
                if uploaded > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds RALLY_WEB_MAX_UPLOAD_BYTES ({limit} bytes)",
                    )
                fh.write(chunk)
        if uploaded == 0:
            raise HTTPException(status_code=400, detail="uploaded file is empty")
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        with _LOCK:
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        raise

    # An extension and MIME type are not media validation. Reject undecodable uploads now
    # instead of persisting a job that can only fail later in a worker.
    try:
        uploaded_info = await asyncio.to_thread(_probe_upload_isolated, original)
        if (uploaded_info["duration_s"] <= 0 or uploaded_info["width"] <= 0
                or uploaded_info["height"] <= 0):
            raise RuntimeError("missing usable video stream metadata")
    except BaseException as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        with _LOCK:
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise HTTPException(
            status_code=400, detail=f"uploaded file is not a readable video: {exc}") from exc

    # Only validated media becomes a visible UUID job directory.
    try:
        os.replace(staging_dir, job_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        with _LOCK:
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        raise
    original = job_dir / original.name

    job = {
        "id": job_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "uploaded",
        "filename": filename,
        "original_path": str(original),
        "thumbnail_path": None,
        "output_path": None,
        "json_path": None,
        "options": options,
        "progress": [],
        "processing": {"stage": "uploaded", "label": "Uploaded", "percent": 5,
                       "detail": "Upload complete", "updated_at": _now()},
        "labeling": {"status": "idle", "detail": "", "updated_at": _now()},
        "result": None,
        "error": None,
        "retryable": False,
        "cancel_requested": False,
    }
    try:
        _create_job(job)
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        with _LOCK:
            _UPLOAD_RESERVED.discard(job_id)
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
        raise
    # Create the preview before returning the first public job response. It is a best-effort
    # still (failure does not reject a valid upload), and lets queued jobs show their source
    # image before a worker slot becomes available.
    _ensure_thumbnail(_load_job(job_id))
    _append_progress(job_id, "upload complete")
    if run_now:
        try:
            _submit_job(job_id, reserved_upload=reserved)
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            with _LOCK:
                _UPLOAD_RESERVED.discard(job_id)
                _UPLOAD_RESERVED_BYTES.pop(job_id, None)
            raise
    else:
        with _LOCK:
            _UPLOAD_RESERVED_BYTES.pop(job_id, None)
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    return JSONResponse(_public_job(_load_job(job_id)))


def _capabilities() -> dict[str, Any]:
    """Which optional processing features are actually usable in this install.

    Lets the UI disable toggles it can't honour (e.g. ball-arbiter without TrackNet
    weights) instead of silently falling back mid-job.
    """
    import importlib.util

    torch_ok = importlib.util.find_spec("torch") is not None
    players_ok = importlib.util.find_spec("ultralytics") is not None
    rtmlib_ok = importlib.util.find_spec("rtmlib") is not None
    try:
        from rally.signals.ball import discover_ball_weights
        weights = discover_ball_weights()
    except Exception:
        weights = None
    if not torch_ok:
        hint = "install PyTorch (pip install torch) and TrackNet weights"
    elif not weights:
        hint = "no TrackNet weights found — run: python -m rally.tools.fetch_models --help"
    else:
        hint = ""
    try:
        from rally.config import RallyConfig
        from rally.signals.pose import (
            discover_rtmpose_weights,
            resolve_rtmpose_device,
            rtmpose_execution_providers,
        )

        pose_cfg = RallyConfig()
        pose_model = discover_rtmpose_weights(pose_cfg.player_pose_model)
        pose_model_present = Path(pose_model).is_file()
        pose_providers = rtmpose_execution_providers(pose_cfg.rtmpose_runtime)
        pose_device = resolve_rtmpose_device(
            pose_cfg.rtmpose_device, pose_cfg.rtmpose_runtime)
    except Exception:
        pose_model_present = False
        pose_providers = []
        pose_device = "unavailable"
    return {
        "ball_arbiter": {
            "available": bool(weights) and torch_ok,
            "weights_present": bool(weights),
            "torch_installed": torch_ok,
            "hint": hint,
        },
        # classical court detection only needs OpenCV, a core dependency
        "court_auto": {"available": True},
        "players": {
            "available": players_ok,
            "hint": ("" if players_ok else
                     "Ultralytics is not installed; match-state pose validation is unavailable"),
        },
        "pose": {
            "available": bool(players_ok and rtmlib_ok and pose_model_present),
            "backend": "rtmlib",
            "model_present": pose_model_present,
            "device": pose_device,
            "execution_providers": pose_providers,
            "cuda": "CUDAExecutionProvider" in pose_providers,
            "hint": ("" if pose_model_present else
                     "RTMPose ONNX model is missing; re-run ./setup.sh"),
        },
        "processing": {
            "workers": _WEB_WORKERS,
            "policy": ("configured" if os.environ.get("RALLY_WEB_WORKERS") is not None
                       else "auto"),
        },
    }


@app.get("/api/capabilities")
def capabilities() -> JSONResponse:
    return JSONResponse(_capabilities())


@app.post("/api/jobs/{job_id}/process")
def process_job(job_id: str) -> JSONResponse:
    return JSONResponse(_public_job(_submit_job(job_id)))


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> JSONResponse:
    """Stop a queued job immediately or cooperatively interrupt a running job."""
    stopped_immediately = False
    with _LOCK:
        path = _job_meta_path(job_id)
        job = _read_json(path, None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job_id in _EDIT_ACTIVE:
            raise HTTPException(
                status_code=409,
                detail="manual segment export cannot be stopped; previous result is retained",
            )
        if job.get("status") not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="job is not currently processing")
        if job.get("cancel_requested"):
            return JSONResponse(_public_job(job))

        attempt_id = str(job.get("active_attempt_id") or "")
        cancel_entry = _CANCEL_EVENTS.get(job_id)
        if cancel_entry is None or cancel_entry[0] != attempt_id:
            cancel_entry = (attempt_id, threading.Event())
            _CANCEL_EVENTS[job_id] = cancel_entry
        cancel_entry[1].set()

        future_entry = _JOB_FUTURES.get(job_id)
        future = (future_entry[1]
                  if future_entry is not None and future_entry[0] == attempt_id else None)
        stopped_immediately = bool(future is not None and future.cancel())
        if stopped_immediately:
            _set_job_cancelled(job)
            _SUBMITTED.discard(job_id)
            _CANCEL_EVENTS.pop(job_id, None)
        else:
            job["cancel_requested"] = True
            percent = int((job.get("processing") or {}).get("percent") or 0)
            _set_processing(
                job, "cancelling", "Stopping", percent,
                "Stopping after the current inference batch")
        job["updated_at"] = _now()
        _atomic_write_json(path, job)

    _append_progress(
        job_id,
        ("processing stopped by user" if stopped_immediately
         else "stop requested by user"),
        attempt_id=attempt_id,
    )
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}/media/{kind}")
def get_media(job_id: str, kind: str, download: bool = False):
    job = _load_job(job_id)
    paths = {"original": job.get("original_path"), "thumbnail": job.get("thumbnail_path"),
             "output": job.get("output_path"), "metadata": job.get("json_path")}
    target = paths.get(kind)
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="media not found")
    path = Path(target)
    if kind == "metadata":
        metadata = _read_json(path, None)
        if isinstance(metadata, dict):
            output = Path(job["output_path"]) if job.get("output_path") else None
            clean = _normalise_web_sidecar(
                job, metadata, output_ready=bool(output and output.exists()))
            disposition = "attachment" if download else "inline"
            filename = f"{Path(job['filename']).stem}_rallies.json"
            return JSONResponse(
                clean,
                headers={
                    "Content-Disposition": f'{disposition}; filename="{filename}"',
                },
            )
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    disposition = "attachment" if download else "inline"
    stem = Path(job["filename"]).stem
    names = {"output": f"{stem}_rallies.mp4", "metadata": f"{stem}_rallies.json"}
    return FileResponse(path, media_type=media_type,
                        filename=names.get(kind, path.name),
                        content_disposition_type=disposition)


@app.get("/api/jobs/{job_id}/waveform")
def get_waveform(job_id: str) -> JSONResponse:
    job = _load_job(job_id)
    data = _read_json(_job_dir(job_id) / "waveform.json", {"strikes": [], "duration": 0})
    result = job.get("result") or {}
    if isinstance(result.get("strike_times"), list):
        data["strikes"] = result["strike_times"]
    data["segments"] = result.get("segments", [])
    if not data.get("duration"):
        data["duration"] = result.get("total_seconds", 0)
    return JSONResponse(data)


# --------------------------------------------------------------------------- #
# manual segment editing + re-export                                          #
# --------------------------------------------------------------------------- #
def _normalise_segments(raw: list[list[float]], duration: float) -> list[tuple[float, float]]:
    segs: list[tuple[float, float]] = []
    for item in raw:
        if len(item) != 2:
            raise HTTPException(status_code=400, detail="each segment must be [start, end]")
        s, e = float(item[0]), float(item[1])
        if not (math.isfinite(s) and math.isfinite(e)):
            raise HTTPException(status_code=400, detail="segment bounds must be finite")
        s = max(0.0, s)
        e = min(e, duration) if duration else e
        if e - s > 0.05:
            segs.append((s, e))
    segs.sort()
    # The renderer concatenates intervals, so overlaps would duplicate footage
    # and make kept_seconds larger than the actual union. Coalesce them here.
    merged: list[tuple[float, float]] = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _rewrite_sidecar(job: dict[str, Any], segs: list[tuple[float, float]]) -> dict[str, Any]:
    result = job.get("result") or {}
    sidecar = _read_json(Path(job["json_path"]), {}) if job.get("json_path") else dict(result)
    if not sidecar:
        sidecar = dict(result)
    total = float(sidecar.get("total_seconds") or (job.get("result") or {}).get("total_seconds") or 0)
    kept = sum(e - s for s, e in segs)
    sidecar["segments"] = [{"index": i, "start": round(s, 3), "end": round(e, 3),
                            "duration": round(e - s, 3)} for i, (s, e) in enumerate(segs)]
    sidecar["n_rallies"] = len(segs)
    sidecar["kept_seconds"] = round(kept, 3)
    sidecar["total_seconds"] = round(total, 3)
    sidecar["compression_ratio"] = round(kept / total, 4) if total else 0.0
    sidecar["edited"] = True
    if job.get("json_path"):
        _atomic_write_json(Path(job["json_path"]), sidecar)
    return sidecar


@app.post("/api/jobs/{job_id}/segments")
def edit_segments(job_id: str, edit: SegmentEdit) -> JSONResponse:
    """Replace the segment list (manual correction). Re-renders the output video."""
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        # Terminal status is published only after result/output state is complete. The
        # worker may still be finishing an independent thumbnail/progress write, both of
        # which use field-level locked mutations and are safe alongside an edit.
        if job.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="job is still processing")
        if job_id in _LABEL_ACTIVE or job_id in _LABEL_SUBMITTED:
            raise HTTPException(status_code=409, detail="labeling samples are being generated")
        if job_id in _EDIT_ACTIVE:
            raise HTTPException(status_code=409, detail="job segments are already being edited")
        _EDIT_ACTIVE.add(job_id)

    previous_job = dict(job)
    attempt = uuid.uuid4().hex
    output_dir = _job_dir(job_id) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_out = output_dir / f"rallies.edit.{attempt}.mp4"
    attempt_json = output_dir / f"rallies.edit.{attempt}.json"
    try:
        result = job.get("result") or {}
        duration = float(result.get("total_seconds") or 0)
        segs = _normalise_segments(edit.segments, duration)
        sidecar_job = dict(job)
        # Build the replacement without mutating the currently published sidecar.
        sidecar_job["json_path"] = None
        sidecar = _rewrite_sidecar(sidecar_job, segs)
        sidecar = _normalise_web_sidecar(job, sidecar, output_ready=False)

        def begin_edit(current: dict[str, Any]) -> None:
            current["error"] = None
            current["status"] = "running"
            current["retryable"] = False
            _set_processing(current, "rendering", "Rendering video", 92,
                            "Applying manual segment edits" if segs
                            else "Saving empty segment selection")

        job = _mutate_job(job_id, begin_edit)

        ok = False
        if segs:
            try:
                cfg = _config_from_options(job.get("options", {}))
                info = probe(job["original_path"])
                ok = _render_output(
                    Path(job["original_path"]), segs, attempt_out, cfg, info,
                                    lambda m: _append_progress(job_id, m))
            except Exception as exc:
                _append_progress(job_id, f"manual render failed: {exc}")
                raise RuntimeError(f"manual video export failed: {exc}") from exc
            if not ok or not attempt_out.is_file():
                raise RuntimeError("manual video export produced no output")

        ready = bool(segs) and ok and attempt_out.exists()
        sidecar = _normalise_web_sidecar(job, sidecar, output_ready=ready)
        _atomic_write_json(attempt_json, sidecar)

        def finish_edit(current: dict[str, Any]) -> None:
            current["status"] = "complete" if ready else "no_output"
            current["retryable"] = False
            current["result"] = sidecar
            current["json_path"] = str(attempt_json)
            current["output_path"] = str(attempt_out) if ready else None
            current["error"] = None
            _mark_labeling_stale(current)
            if ready:
                _set_processing(current, "complete", "Ready", 100,
                                f"{len(segs)} edited rallies — output ready")
            elif segs:
                _set_processing(current, "no_output", "Analysis only", 100,
                                "Edited segments saved but video export failed")
            else:
                _set_processing(current, "no_output", "No segments selected", 100,
                                "All segments were removed")

        job = _mutate_job(job_id, finish_edit)
        _archive_label_artifacts(job_id)
        # The metadata pointer now owns the new immutable artifacts. Old files can be
        # removed afterward without ever leaving the job with no valid published result.
        for key in ("output_path", "json_path"):
            old = previous_job.get(key)
            if old and old not in {str(attempt_out), str(attempt_json)}:
                Path(old).unlink(missing_ok=True)
        return JSONResponse(_public_job(job))
    except Exception as exc:
        attempt_out.unlink(missing_ok=True)
        attempt_json.unlink(missing_ok=True)
        try:
            def fail_edit(current: dict[str, Any]) -> None:
                current["status"] = previous_job.get("status", "complete")
                current["retryable"] = False
                current["result"] = previous_job.get("result")
                current["json_path"] = previous_job.get("json_path")
                current["output_path"] = previous_job.get("output_path")
                current["labeling"] = previous_job.get("labeling", {})
                current["error"] = f"could not apply segment edits: {exc}"
                _set_processing(
                    current,
                    "complete" if current.get("output_path") else "no_output",
                    "Previous result retained", 100, current["error"],
                )

            _mutate_job(job_id, fail_edit)
        except HTTPException:
            pass
        raise
    finally:
        with _LOCK:
            _EDIT_ACTIVE.discard(job_id)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if (job.get("status") in {"queued", "running"} or job_id in _ACTIVE
                or job_id in _SUBMITTED or job_id in _EDIT_ACTIVE
                or job_id in _LABEL_ACTIVE or job_id in _LABEL_SUBMITTED):
            raise HTTPException(status_code=409, detail="cannot delete a busy job")
        try:
            shutil.rmtree(_job_dir(job_id))
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not delete job: {exc}") from exc
        if _job_dir(job_id).exists():
            raise HTTPException(status_code=500, detail="could not delete complete job directory")
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# labelling: generate raw samples (player crops + serve clips) to annotate     #
# --------------------------------------------------------------------------- #
def _labels_dir(job_id: str) -> Path:
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), {})
        revision = (job.get("labeling") or {}).get("revision")
    if revision:
        return _job_dir(job_id) / "label_revisions" / str(revision) / "labels"
    return _job_dir(job_id) / "labels"  # legacy/no-live-revision path


def _assets_dir(job_id: str) -> Path:
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), {})
        revision = (job.get("labeling") or {}).get("revision")
    if revision:
        return _job_dir(job_id) / "label_revisions" / str(revision) / "label_assets"
    return _job_dir(job_id) / "label_assets"


def _label_root_locked(job_id: str, *, writable: bool) -> tuple[dict[str, Any], Path]:
    """Resolve one stable revision; stale revisions are readable but never writable."""
    job = _read_json(_job_meta_path(job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    lab = job.get("labeling") or {}
    revision = lab.get("revision")
    allowed_status = {"ready"} if writable else {"ready", "stale"}
    if (lab.get("status") not in allowed_status or not revision or job_id in _LABEL_ACTIVE
            or job_id in _LABEL_SUBMITTED or job_id in _EDIT_ACTIVE
            or job.get("status") in {"queued", "running"}):
        raise HTTPException(status_code=409, detail="no stable label revision")
    return job, _job_dir(job_id) / "label_revisions" / str(revision)


def _live_label_root_locked(job_id: str) -> tuple[dict[str, Any], Path]:
    """Backward-compatible name for mutation callers requiring the ready revision."""
    return _label_root_locked(job_id, writable=True)


def _clean_label_values(kind: str, values: dict[str, Any], roster_ids: set[str]) -> dict:
    """Validate the finite annotation vocabulary before it reaches training exports."""
    allowed = {
        "player_identity": {
            "player": roster_ids,
            "quality": {"clear", "partial", "occluded", "not_player"},
        },
        "serve_motion": {
            "is_serve": {"yes", "no", "unsure"},
            "server": roster_ids | {"unknown"},
            "side": {"deuce", "ad"},
            "end": {"near", "far"},
            "serve_type": {"flat", "slice", "kick"},
            "outcome": {"in", "fault", "ace", "let"},
        },
    }[kind]
    unknown = set(values) - set(allowed) - ({"notes"} if kind == "serve_motion" else set())
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown label fields: {sorted(unknown)}")
    clean: dict[str, Any] = {}
    for key, choices in allowed.items():
        if key not in values or values[key] in (None, ""):
            continue
        value = str(values[key])
        if value not in choices:
            raise HTTPException(status_code=422, detail=f"invalid {kind} value for {key}")
        clean[key] = value
    if kind == "serve_motion" and values.get("notes") not in (None, ""):
        notes = str(values["notes"]).strip()
        if len(notes) > 1000:
            raise HTTPException(status_code=422, detail="serve notes are too long")
        clean["notes"] = notes
    return clean


def _web_yolo_model_name() -> str:
    """Label-crop detector override, then the pipeline-wide detector override."""
    for env_name in ("RALLY_WEB_YOLO", "RALLY_YOLO_DETECTION_MODEL"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_YOLO_DETECTION_MODEL


def _yolo():
    """Lazily load one shared YOLO model (used only for label-crop boxes)."""
    global _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is None:
            from ultralytics import YOLO

            from rally.signals.player import discover_yolo_weights
            _YOLO_MODEL = YOLO(discover_yolo_weights(_web_yolo_model_name()))
        return _YOLO_MODEL


def _detect_boxes(frame_bgr, conf: float = 0.3) -> list[tuple[float, float, float, float]]:
    """Person boxes as pixel (x0, y0, x1, y1). Unlike the core's foot-point
    detector we keep the full box because we need it to crop a player."""
    model = _yolo()
    # Ultralytics keeps mutable predictor state on the model. Label jobs may run in
    # parallel, so calls sharing this singleton must be serialized as well as loading.
    with _YOLO_LOCK:
        res = model.predict(frame_bgr, conf=conf, classes=[0], verbose=False)
    boxes: list[tuple[float, float, float, float]] = []
    for r in res:
        xyxy = getattr(r.boxes, "xyxy", None)
        if xyxy is None:
            continue
        for b in xyxy.cpu().numpy():
            boxes.append(tuple(float(v) for v in b[:4]))  # type: ignore[arg-type]
    return boxes


def _roster_for(match_type: str) -> list[dict[str, Any]]:
    if match_type == "doubles":
        return [
            {"id": "P1", "name": "Near left", "side": "near", "col": "left"},
            {"id": "P2", "name": "Near right", "side": "near", "col": "right"},
            {"id": "P3", "name": "Far left", "side": "far", "col": "left"},
            {"id": "P4", "name": "Far right", "side": "far", "col": "right"},
        ]
    return [
        {"id": "P1", "name": "Near player", "side": "near"},
        {"id": "P2", "name": "Far player", "side": "far"},
    ]


def _suggest_player(foot_x: float, foot_y: float, region, match_type: str) -> str:
    """Map a detection's foot point to a roster slot by court position.

    In frame coordinates the near side is lower down (larger y). Left/right split
    at the court mid-x. This only *suggests* an id — the annotator confirms it.
    """
    mid_y = (region[1] + region[3]) / 2 if region else 0.5
    mid_x = (region[0] + region[2]) / 2 if region else 0.5
    near = foot_y >= mid_y
    if match_type == "doubles":
        left = foot_x < mid_x
        return {(True, True): "P1", (True, False): "P2",
                (False, True): "P3", (False, False): "P4"}[(near, left)]
    return "P1" if near else "P2"


def _evenly_pick(items: list, limit: int) -> list:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / (limit - 1) if limit > 1 else 0
    return [items[int(round(i * step))] for i in range(limit)]


def _player_frame_times(segments: list[dict], duration: float, n: int) -> list[float]:
    """Times to grab player frames — rally midpoints (players in view), else uniform."""
    if segments:
        mids = [(float(s["start"]) + float(s["end"])) / 2 for s in segments]
        if len(mids) >= n:
            return _evenly_pick(mids, n)
        times = list(mids)
        i = 0
        while len(times) < n:  # add offset samples inside rallies for variety
            s = segments[i % len(segments)]
            span = max(0.3, float(s["end"]) - float(s["start"]))
            times.append(min(float(s["end"]) - 0.1, float(s["start"]) + (0.3 + 0.5 * (i // len(segments))) % span))
            i += 1
            if i > n * 4:
                break
        return sorted(times[:n])
    if duration <= 0:
        return []
    return [duration * (i + 1) / (n + 1) for i in range(n)]


def _serve_times(segments: list[dict], duration: float, n: int) -> list[float]:
    if segments:
        starts = [float(s["start"]) for s in segments]
        return _evenly_pick(starts, n)
    if duration <= 0:
        return []
    return [duration * (i + 1) / (n + 1) for i in range(n)]


def _ffmpeg_clip(src: Path, start_s: float, end_s: float, dst: Path) -> None:
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    end_s = max(start_s + 0.5, end_s)
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}", "-to", f"{end_s:.3f}",
         "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=640:-2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-c:a", "aac",
         "-movflags", "+faststart", str(dst)],
        check=True,
    )


def _set_labeling(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            return
        lab = job.setdefault("labeling", {})
        lab.update(fields)
        lab["updated_at"] = _now()
        _atomic_write_json(_job_meta_path(job_id), job)


def _resolve_match_type(requested: str, persons_per_frame: list[int]) -> str:
    requested = (requested or "auto").lower()
    if requested in {"singles", "doubles"}:
        return requested
    # auto: doubles if several frames clearly show 3+ people on court
    crowded = sum(1 for c in persons_per_frame if c >= 3)
    return "doubles" if crowded >= 2 else "singles"


def _generate_player_tasks(job_id: str, src: Path, segments: list[dict], duration: float,
                           max_items: int, match_type_req: str, regenerate: bool,
                           assets_dir: Optional[Path] = None):
    """Detect players, build a roster, and crop one-player pictures to annotate."""
    import cv2

    n_frames = min(max(max_items, len(segments) or 0, 6), 30)
    times = _player_frame_times(segments, duration, n_frames)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("could not open video for player detection")

    grabbed: list[tuple[float, Any, list]] = []  # (t, frame, boxes)
    feet: list[tuple[float, float]] = []
    persons_per_frame: list[int] = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            boxes = _detect_boxes(frame)
            persons_per_frame.append(len(boxes))
            for (x0, y0, x1, y1) in boxes:
                feet.append(((x0 + x1) / 2 / w, y1 / h))
            grabbed.append((t, frame, boxes))
    finally:
        cap.release()

    region = estimate_court_region(feet)
    match_type = _resolve_match_type(match_type_req, persons_per_frame)
    roster = _roster_for(match_type)

    assets = assets_dir or _assets_dir(job_id)
    tasks: list[dict[str, Any]] = []
    idx = 0
    for (t, frame, boxes) in grabbed:
        if idx >= max_items:
            break
        h, w = frame.shape[:2]
        # in-region players, largest first (closest / clearest)
        cand = []
        for (x0, y0, x1, y1) in boxes:
            fx, fy = (x0 + x1) / 2 / w, y1 / h
            if region is not None and not (region[0] <= fx <= region[2] and region[1] <= fy <= region[3]):
                continue
            cand.append(((x1 - x0) * (y1 - y0), (x0, y0, x1, y1), fx, fy))
        cand.sort(reverse=True)
        for _area, (x0, y0, x1, y1), fx, fy in cand:
            if idx >= max_items:
                break
            padx = 0.12 * (x1 - x0)
            pady = 0.10 * (y1 - y0)
            cx0 = max(0, int(x0 - padx)); cy0 = max(0, int(y0 - pady))
            cx1 = min(w, int(x1 + padx)); cy1 = min(h, int(y1 + pady))
            if cx1 - cx0 < 12 or cy1 - cy0 < 24:
                continue
            crop = frame[cy0:cy1, cx0:cx1]
            rel = f"player_{idx:04d}.jpg"
            path = assets / rel
            if regenerate or not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                    raise RuntimeError(f"could not write player crop: {path}")
            tasks.append({
                "id": f"player_{idx:04d}",
                "kind": "player_identity",
                "title": f"Player crop {idx + 1}",
                "time_s": round(float(t), 3),
                "media_type": "image",
                "asset_url": f"/api/jobs/{job_id}/assets/{rel}",
                "suggested_player": _suggest_player(fx, fy, region, match_type),
            })
            idx += 1
    return roster, match_type, tasks


def _generate_serve_tasks(job_id: str, src: Path, segments: list[dict], duration: float,
                          max_items: int, regenerate: bool, match_state: Optional[dict] = None,
                          assets_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Cut short clips around each serve moment (rally start) to classify."""
    count = min(max_items, max(1, len(segments) or max_items))
    if segments:
        selected = _evenly_pick(list(enumerate(segments)), count)
    else:
        selected = [(None, {"start": value}) for value in _serve_times([], duration, count)]
    groups = (match_state or {}).get("logical_groups") or []
    assets = assets_dir or _assets_dir(job_id)
    tasks: list[dict[str, Any]] = []
    for i, (source_index, segment) in enumerate(selected[:max_items]):
        start = float(segment["start"])
        clip_start = max(0.0, start - 1.5)
        clip_end = min(duration, start + 3.5) if duration else start + 3.5
        rel = f"serve_{i:04d}.mp4"
        path = assets / rel
        if regenerate or not path.exists():
            _ffmpeg_clip(src, clip_start, clip_end, path)
        stable_context: dict[str, Any] = {}
        if source_index is not None:
            stable_context["source_segment_index"] = int(
                segment.get("index", source_index))
            group = next((record for record in groups
                          if record.get("output")
                          and record.get("serve_member_index") is not None
                          and abs(float(record["output"][0]) - start) <= 0.01), None)
            if group is not None:
                stable_context.update({
                    "logical_group": int(group["group_index"]),
                    "match_state_observation_index": int(group["serve_member_index"]),
                })
        tasks.append({
            "id": f"serve_{i:04d}",
            "kind": "serve_motion",
            "title": f"Serve clip {i + 1}",
            "time_s": round(float(start), 3),
            "media_type": "video",
            "asset_url": f"/api/jobs/{job_id}/assets/{rel}",
            **stable_context,
        })
    return tasks


def _run_label_gen(job_id: str, req: LabelTaskRequest) -> None:
    with _LOCK:
        if job_id in _LABEL_ACTIVE:
            return
        _LABEL_SUBMITTED.discard(job_id)
        _LABEL_ACTIVE.add(job_id)
    build_root = _job_dir(job_id) / f".label-build.{uuid.uuid4().hex}"
    build_labels = build_root / "labels"
    build_assets = build_root / "label_assets"
    try:
        build_labels.mkdir(parents=True, exist_ok=True)
        build_assets.mkdir(parents=True, exist_ok=True)
        job = _load_job(job_id)
        src = Path(job["original_path"])
        if not src.exists():
            raise RuntimeError("original video missing")
        _set_labeling(job_id, status="generating", error=None,
                      detail="Generating samples", match_type=req.match_type)
        result = job.get("result") or {}
        segments = result.get("segments", [])
        try:
            duration = float(result.get("total_seconds") or probe(str(src)).duration_s)
        except Exception:
            duration = float(result.get("total_seconds") or 0)

        roster = _roster_for("singles")
        match_type = "singles"
        tasks: list[dict[str, Any]] = []

        if "player_identity" in req.kinds:
            try:
                _set_labeling(job_id, detail="Detecting players and cropping")
                roster, match_type, player_tasks = _generate_player_tasks(
                    job_id, src, segments, duration, req.max_items, req.match_type, True,
                    assets_dir=build_assets)
                tasks.extend(player_tasks)
            except (ImportError, ModuleNotFoundError):
                # A custom/mocked detector can work without Ultralytics, so discover
                # availability by actually invoking the detector. Missing optional YOLO
                # support skips only player crops; serve clips remain useful.
                _set_labeling(job_id, detail="Player crops skipped (Ultralytics unavailable)")

        if "serve_motion" in req.kinds:
            _set_labeling(job_id, detail="Cutting serve clips")
            tasks.extend(_generate_serve_tasks(
                job_id, src, segments, duration, req.max_items, True,
                match_state=((result.get("stages") or {}).get("match_state") or {}),
                assets_dir=build_assets))

        # persist roster (preserve any user-renamed names) + tasks
        existing = _read_json(_labels_dir(job_id) / "roster.json", None)
        if existing and not req.regenerate:
            names = {r["id"]: r.get("name") for r in existing if isinstance(r, dict)}
            for r in roster:
                if names.get(r["id"]):
                    r["name"] = names[r["id"]]
        _atomic_write_json(build_labels / "roster.json", roster)
        _atomic_write_json(build_labels / "tasks.json", tasks)
        stages = result.get("stages") or {}
        _atomic_write_json(build_labels / "feature_context.json", {
            "schema_version": "rally.serve_rule_context.v1",
            "strike_times": result.get("strike_times", []),
            "segments": result.get("segments", []),
            "match_state": stages.get("match_state", {}),
        })
        # New media invalidates old human answers even when stable task IDs happen to
        # repeat. Publish an explicitly empty label set with the new revision.
        _atomic_write_json(build_labels / "labels.json", {})

        n_player = sum(1 for t in tasks if t["kind"] == "player_identity")
        n_serve = sum(1 for t in tasks if t["kind"] == "serve_motion")
        revision = uuid.uuid4().hex
        revisions_root = _job_dir(job_id) / "label_revisions"
        revisions_root.mkdir(parents=True, exist_ok=True)
        published_root = revisions_root / revision
        os.replace(build_root, published_root)  # labels + assets switch as one filesystem unit

        def publish_revision(current: dict[str, Any]) -> None:
            lab = current.setdefault("labeling", {})
            lab.update(status="ready", match_type=match_type, revision=revision,
                       error=None,
                       detail=f"{n_player} player crops · {n_serve} serve clips",
                       counts={"player_identity": n_player, "serve_motion": n_serve},
                       updated_at=_now())

        _mutate_job(job_id, publish_revision)  # atomic pointer switch in job.json
        _archive_label_artifacts(job_id)       # migrate legacy dirs + prune older revisions
    except Exception as exc:
        _set_labeling(job_id, status="failed", error=str(exc), detail=str(exc))
    finally:
        shutil.rmtree(build_root, ignore_errors=True)
        with _LOCK:
            _LABEL_ACTIVE.discard(job_id)
            _LABEL_SUBMITTED.discard(job_id)


@app.post("/api/jobs/{job_id}/label-tasks")
def create_label_tasks(job_id: str, req: LabelTaskRequest) -> JSONResponse:
    bad = set(req.kinds) - _LABEL_KINDS
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown label kind(s): {', '.join(sorted(bad))}")
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if not job.get("original_path") or not Path(job["original_path"]).exists():
            raise HTTPException(status_code=404, detail="original video missing")
        if (job.get("status") in {"queued", "running"} or job_id in _SUBMITTED
                or job_id in _ACTIVE or job_id in _EDIT_ACTIVE):
            raise HTTPException(status_code=409, detail="job result is being changed")
        if job_id in _LABEL_SUBMITTED or job_id in _LABEL_ACTIVE:
            return JSONResponse(_public_job(job))
        if job.get("labeling", {}).get("status") == "ready" and not req.regenerate:
            return JSONResponse(_public_job(job))
        queued_work = len(_SUBMITTED | _ACTIVE | _LABEL_SUBMITTED | _LABEL_ACTIVE
                          | _UPLOAD_RESERVED)
        if queued_work >= _max_pending_jobs():
            raise HTTPException(status_code=503, detail="processing queue is full")
        lab = job.setdefault("labeling", {})
        lab.update(status="generating", detail="Queued", error=None, updated_at=_now())
        job["updated_at"] = _now()
        _LABEL_SUBMITTED.add(job_id)
        _atomic_write_json(_job_meta_path(job_id), job)
    try:
        _EXECUTOR.submit(_run_label_gen, job_id, req)
    except Exception:
        with _LOCK:
            _LABEL_SUBMITTED.discard(job_id)
        _set_labeling(job_id, status="failed", detail="Could not queue label generation",
                      error="executor submission failed")
        raise
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}/label-tasks")
def get_label_tasks(job_id: str) -> JSONResponse:
    with _LOCK:
        job, root = _label_root_locked(job_id, writable=False)
        labels_dir = root / "labels"
        tasks = _read_json(labels_dir / "tasks.json", [])
        roster = _read_json(labels_dir / "roster.json", [])
        labels = _read_json(labels_dir / "labels.json", {})
        revision = str((job.get("labeling") or {}).get("revision") or "")
        tasks = [
            {**task, "asset_url": f"{task['asset_url']}?revision={revision}"}
            if task.get("asset_url") else dict(task)
            for task in tasks
        ]
    return JSONResponse({
        "revision": revision, "tasks": tasks, "roster": roster, "labels": labels})


@app.post("/api/jobs/{job_id}/roster")
def update_roster(job_id: str, update: RosterUpdate) -> JSONResponse:
    for record in update.roster:
        roster_id = str(record.get("id") or "")
        if not _ROSTER_ID.fullmatch(roster_id):
            raise HTTPException(status_code=422, detail="invalid roster id")
        if len(str(record.get("name") or "")) > 100:
            raise HTTPException(status_code=422, detail="roster id/name is too long")
        if record.get("side") not in {None, "near", "far"}:
            raise HTTPException(status_code=422, detail="invalid roster side")
        if record.get("col") not in {None, "left", "right"}:
            raise HTTPException(status_code=422, detail="invalid roster column")
    clean = [{"id": str(r["id"]), "name": str(r.get("name") or r["id"]).strip(),
              "side": r.get("side"), "col": r.get("col")}
             for r in update.roster]
    with _LOCK:
        job, root = _live_label_root_locked(job_id)
        if update.revision != str((job.get("labeling") or {}).get("revision") or ""):
            raise HTTPException(status_code=409, detail="label revision changed; reload samples")
        _atomic_write_json(root / "labels" / "roster.json", clean)
    return JSONResponse({"revision": update.revision, "roster": clean})


@app.get("/api/jobs/{job_id}/labels")
def get_labels(job_id: str) -> JSONResponse:
    with _LOCK:
        _job, root = _label_root_locked(job_id, writable=False)
        labels = _read_json(root / "labels" / "labels.json", {})
    return JSONResponse({"labels": labels})


@app.post("/api/jobs/{job_id}/labels")
def save_label(job_id: str, payload: LabelPayload) -> JSONResponse:
    if payload.kind not in _LABEL_KINDS:
        raise HTTPException(status_code=400, detail="unknown label kind")
    if len(json.dumps(payload.values)) > 64 * 1024:
        raise HTTPException(status_code=413, detail="label values are too large")
    with _LOCK:
        job, root = _live_label_root_locked(job_id)
        if payload.revision != str((job.get("labeling") or {}).get("revision") or ""):
            raise HTTPException(status_code=409, detail="label revision changed; reload samples")
        tasks = _read_json(root / "labels" / "tasks.json", [])
        task = next((item for item in tasks if item.get("id") == payload.task_id), None)
        if task is None or task.get("kind") != payload.kind:
            raise HTTPException(status_code=404, detail="label task not found")
        roster = _read_json(root / "labels" / "roster.json", [])
        roster_ids = {str(record.get("id")) for record in roster if record.get("id")}
        values = _clean_label_values(payload.kind, payload.values, roster_ids)
        path = root / "labels" / "labels.json"
        labels = _read_json(path, {})
        labels[payload.task_id] = {"task_id": payload.task_id, "kind": payload.kind,
                                   "values": values, "updated_at": _now()}
        _atomic_write_json(path, labels)
    return JSONResponse({
        "revision": payload.revision, "labels": labels, "saved": payload.task_id})


@app.get("/api/jobs/{job_id}/labels/download")
def download_labels(job_id: str) -> Response:
    with _LOCK:
        job, root = _label_root_locked(job_id, writable=False)
        labels_dir = root / "labels"
        path = labels_dir / "labels.json"
        result = job.get("result") or {}
        stages = result.get("stages") or {}
        feature_context = _read_json(labels_dir / "feature_context.json", None)
        if not isinstance(feature_context, dict):
            # Legacy revisions predate context snapshots. They remain exportable for
            # audit, while stable-ID training rejects heuristic/absent joins for gating.
            feature_context = {
                "schema_version": "rally.serve_rule_context.v1",
                "strike_times": result.get("strike_times", []),
                "segments": result.get("segments", []),
                "match_state": stages.get("match_state", {}),
                "legacy_revision_context": True,
            }
        export = {"schema_version": "rally.web_labels.v2",
                  "job_id": job_id, "filename": job.get("filename"),
                  "roster": _read_json(labels_dir / "roster.json", []),
                  "tasks": _read_json(labels_dir / "tasks.json", []),
                  "labels": _read_json(path, {}),
                  # Offline training joins answers to the rule inputs measured for this
                  # match. A trained artifact still needs the held-out deployment gate.
                  "feature_context": feature_context}
        payload = json.dumps(export, indent=2).encode("utf-8")
        filename = f"{Path(job['filename']).stem}_labels.json"
    return Response(
        payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}/assets/{asset_path:path}")
def get_asset(job_id: str, asset_path: str, revision: str | None = None) -> FileResponse:
    with _LOCK:
        job, label_root = _label_root_locked(job_id, writable=False)
        current_revision = str((job.get("labeling") or {}).get("revision") or "")
        if revision is not None and revision != current_revision:
            raise HTTPException(status_code=409, detail="label revision changed; reload samples")
        root = (label_root / "label_assets").resolve()
    path = (root / asset_path).resolve()
    try:
        path.relative_to(root)               # confine to the job's asset dir
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="asset not found") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="asset not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rally-web", description="Run the rally trimmer web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-dir", default=None,
        help="uploaded-session directory (default: sessions/uploads)")
    parser.add_argument("--reload", action="store_true", help="enable uvicorn auto-reload")
    args = parser.parse_args(argv)

    if args.data_dir:
        global DATA_DIR
        DATA_DIR = Path(args.data_dir).resolve()
        os.environ["RALLY_WEB_DATA"] = str(DATA_DIR)
    _ensure_data_dir()

    import uvicorn

    uvicorn.run("rally.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
