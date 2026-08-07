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
    cut_segments,
    find_font,
    probe,
    render_labeled,
)
from rally.pipeline import trim
from rally.signals.player import estimate_court_region
from rally.web.golden import discover_datasets, resolve_media_path
from rally.web.schemas import (
    LabelPayload,
    LabelTaskRequest,
    MatchRosterUpdate,
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
        # CPU is a supported execution target; required-package/model validation happens
        # in the strict server preflight before any queued work is recovered.
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
_DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
_MAX_JSON_BODY_BYTES = 2 * 1024 * 1024


class _JobCancelled(RuntimeError):
    """Cooperative stop requested by the owner of a web processing job."""

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from rally.preflight import require_server_install

    require_server_install()
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


def _discard_published_result(job_id: str, job: dict[str, Any]) -> None:
    """Delete and unpublish every artifact from the previous full analysis.

    A reprocess is a new analysis, not an edit of the last successful result.  Once the
    request has been admitted to the queue, stale video, metadata, and waveform data must
    no longer be reachable through either the API or an old media URL.  The thumbnail and
    original upload deliberately remain available so the processing player has a preview.
    """
    job_root = _job_dir(job_id).resolve()
    candidates = [
        Path(job["output_path"]) if job.get("output_path") else None,
        Path(job["json_path"]) if job.get("json_path") else None,
        job_root / "output" / "rallies.mp4",
        job_root / "output" / "rallies.json",
        job_root / "waveform.json",
    ]
    paths: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(job_root)
        except ValueError as exc:
            raise RuntimeError(
                f"refusing to delete result outside job directory: {candidate}"
            ) from exc
        paths.add(resolved)

    for path in paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(job_root / "output" / "player_thumbnails", ignore_errors=True)
    shutil.rmtree(job_root / "output" / "signal_evidence", ignore_errors=True)
    for pattern in (".player-thumbnails.*", ".signal-evidence.*"):
        for directory in (job_root / "output").glob(pattern):
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)

    job["output_path"] = None
    job["json_path"] = None
    job["result"] = None
    job.pop("signal_snapshot", None)


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
    ("tracking target-court players", "visual", "Tracking target players", 38),
    ("building shared all-player pose timeline", "pose", "Reading all-player poses", 45),
    ("pose refinement progress", "pose", "Refining player actions", 66),
    ("decoded", "deciding", "Points found", 89),
    ("writing event timeline", "waveform", "Building timeline", 90),
    ("rendering", "rendering", "Rendering video", 92),
    ("cutting", "rendering", "Rendering video", 92),
    ("wrote", "writing", "Writing output", 95),
]


def _stage_for_message(message: str) -> dict[str, Any]:
    text = message.lower()
    refinement = re.search(r"pose refinement progress\s+(\d+)\s*/\s*(\d+)", text)
    if refinement:
        done, total = int(refinement.group(1)), max(1, int(refinement.group(2)))
        return {
            "stage": "pose", "label": "Refining player actions",
            "percent": 60 + int(round(15 * min(1.0, done / total))),
        }
    timeline = re.search(r"pose timeline progress\s+(\d+)\s*/\s*(\d+)", text)
    if timeline:
        done, total = int(timeline.group(1)), max(1, int(timeline.group(2)))
        return {
            "stage": "pose", "label": "Reading all-player poses",
            "percent": 45 + int(round(15 * min(1.0, done / total))),
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


def _player_thumbnail_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "output" / "player_thumbnails"


def _signal_artifact_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "output" / "signal_evidence"


def _snapshot_attempt_id(job: dict[str, Any]) -> str | None:
    snapshot = job.get("signal_snapshot")
    attempt_id = str(snapshot.get("attempt_id") or "") if isinstance(snapshot, dict) else ""
    return attempt_id if re.fullmatch(r"[0-9a-f]{32}", attempt_id) else None


def _player_artifact_dir(job: dict[str, Any]) -> Path:
    attempt_id = _snapshot_attempt_id(job)
    if attempt_id:
        live = _job_dir(job["id"]) / "output" / f".player-thumbnails.{attempt_id}"
        if live.is_dir():
            return live
    return _player_thumbnail_dir(job["id"])


def _serve_artifact_dir(job: dict[str, Any]) -> Path:
    attempt_id = _snapshot_attempt_id(job)
    if attempt_id:
        live = _job_dir(job["id"]) / "output" / f".signal-evidence.{attempt_id}" / "serves"
        if live.is_dir():
            return live
    return _signal_artifact_dir(job["id"]) / "serves"


def _action_artifact_dir(job: dict[str, Any]) -> Path:
    attempt_id = _snapshot_attempt_id(job)
    if attempt_id:
        live = _job_dir(job["id"]) / "output" / f".signal-evidence.{attempt_id}" / "actions"
        if live.is_dir():
            return live
    return _signal_artifact_dir(job["id"]) / "actions"


def _public_match(
    job_id: str, raw_match: Any, *, artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Attach cache-busted identity crops without persisting transient API URLs."""
    match = dict(raw_match) if isinstance(raw_match, dict) else {}
    roster = []
    root = artifact_root or _player_thumbnail_dir(job_id)
    for raw_record in match.get("roster") or []:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        player_id = str(record.get("id") or "")
        inspection_sources = (
            record.get("inspection_sources") or record.get("thumbnail_sources") or [])
        inspection_images = []
        for index, source in enumerate(inspection_sources):
            path = root / f"{player_id}_{index}.jpg"
            if not path.is_file():
                continue
            try:
                version = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            inspection_images.append({
                "url": f"/api/jobs/{job_id}/players/{player_id}/thumbnails/{index}?v={version}",
                "time_s": source.get("time_s"),
                "index": index,
            })
        thumbnail_times: set[float] = set()
        for source in record.get("thumbnail_sources") or []:
            if not isinstance(source, dict):
                continue
            try:
                value = float(source.get("time_s"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                thumbnail_times.add(round(value, 3))
        thumbnails = [
            image for image in inspection_images
            if isinstance(image.get("time_s"), (int, float))
            and round(float(image["time_s"]), 3) in thumbnail_times
        ][:3]
        if not thumbnails:
            thumbnails = inspection_images[:3]
        record["thumbnails"] = thumbnails
        record["inspection_images"] = inspection_images
        record["signal_gallery_url"] = (
            f"/jobs/{job_id}/signals/players/{player_id}")
        roster.append(record)
    match["roster"] = roster
    return match


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal ``*_path`` fields and attach browser-facing media URLs."""
    if not job:
        return {}
    data = {k: v for k, v in job.items() if not k.endswith("_path")}
    snapshot = data.pop("signal_snapshot", None)
    if isinstance(snapshot, dict):
        data["signals_live"] = {
            "current_stage": snapshot.get("current_stage"),
            "updated_at": snapshot.get("updated_at"),
        }
    if isinstance(data.get("result"), dict):
        output = Path(job["output_path"]) if job.get("output_path") else None
        data["result"] = _normalise_web_sidecar(
            job, data["result"], output_ready=bool(output and output.exists()))
        if isinstance(data["result"].get("match"), dict):
            data["result"]["match"] = _public_match(
                job["id"], data["result"]["match"],
                artifact_root=_player_artifact_dir(job))
    if isinstance(data.get("match"), dict):
        data["match"] = _public_match(
            job["id"], data["match"], artifact_root=_player_artifact_dir(job))
    if data.get("status") in {"queued", "running"}:
        # Keep the prior profile internally so user-entered names can be merged into the
        # fresh ReID roster, but never expose stale players/results during a reprocess.
        data["match"] = {"roster": []}
    if data.get("error") and len(str(data["error"])) > 1500:
        data["error"] = str(data["error"])[:1500].rstrip() + " ..."
    data["media"] = _media_urls(job)
    return data


def _render_match_player_thumbnails(
    source: Path, match: dict[str, Any], destination: Path,
) -> int:
    """Crop representative player views recorded by persistent identity tracking."""
    import cv2

    destination.mkdir(parents=True, exist_ok=True)
    if not any(
        isinstance(record, dict)
        and (record.get("inspection_sources") or record.get("thumbnail_sources"))
        for record in (match.get("roster") or [])
    ):
        return 0
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("could not open source video for player thumbnails")
    written = 0
    try:
        for record in match.get("roster") or []:
            player_id = str(record.get("id") or "")
            if not _ROSTER_ID.fullmatch(player_id):
                continue
            samples = (
                record.get("inspection_sources")
                or record.get("thumbnail_sources")
                or []
            )
            for index, sample in enumerate(samples[:40]):
                try:
                    time_s = float(sample["time_s"])
                    foot_x = float(sample["foot_x_norm"])
                    foot_y = float(sample["foot_y_norm"])
                    area = float(sample["box_area_norm"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (math.isfinite(time_s) and math.isfinite(foot_x)
                        and math.isfinite(foot_y) and math.isfinite(area) and area > 0):
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                height, width = frame.shape[:2]
                bbox = sample.get("bbox_norm")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    bx0, by0, bx1, by1 = [float(value) for value in bbox]
                    box_w = max(1.0, (bx1 - bx0) * width)
                    box_h = max(1.0, (by1 - by0) * height)
                    centre_x = 0.5 * (bx0 + bx1) * width
                    bottom = by1 * height
                else:
                    # Compatibility for results produced before full body boxes were
                    # retained by the ReID pass.
                    box_h = math.sqrt(area / 0.38) * height
                    box_w = area * width * height / max(box_h, 1.0)
                    centre_x = foot_x * width
                    bottom = foot_y * height
                box_h = max(36.0, min(float(height), box_h))
                box_w = max(18.0, min(float(width), box_w))
                x0 = max(0, int(round(centre_x - 0.72 * box_w)))
                x1 = min(width, int(round(centre_x + 0.72 * box_w)))
                y0 = max(0, int(round(bottom - 1.10 * box_h)))
                y1 = min(height, int(round(bottom + 0.06 * box_h)))
                if x1 - x0 < 16 or y1 - y0 < 32:
                    continue
                crop = frame[y0:y1, x0:x1]
                path = destination / f"{player_id}_{index}.jpg"
                temp = destination / f".{player_id}_{index}.{uuid.uuid4().hex}.jpg"
                if not cv2.imwrite(str(temp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                    raise RuntimeError(f"could not write player thumbnail: {path}")
                os.replace(temp, path)
                written += 1
    finally:
        cap.release()
    return written


def _render_serve_signal_artifacts(
    source: Path, stages: dict[str, Any], destination: Path,
) -> int:
    """Write one annotated source frame for every RTMPose serve proposal."""
    import cv2

    observations = ((stages.get("serve_pose") or {}).get("observations") or [])
    if not observations:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("could not open source video for serve-pose evidence")
    skeleton = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    written = 0
    try:
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                continue
            raw_time = observation.get("pose_evidence_time")
            if raw_time is None:
                raw_time = observation.get("first_strike")
            try:
                time_s = float(raw_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time_s):
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            accepted = bool(observation.get("accepted"))
            color = (60, 190, 90) if accepted else (60, 80, 220)
            bbox = observation.get("server_bbox_norm")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x0, y0, x1, y1 = (
                    int(round(float(bbox[0]) * width)),
                    int(round(float(bbox[1]) * height)),
                    int(round(float(bbox[2]) * width)),
                    int(round(float(bbox[3]) * height)),
                )
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            racket_bbox = observation.get("racket_bbox_norm") or []
            if len(racket_bbox) == 4:
                racket_box = tuple(
                    round(float(value) * (width if offset % 2 == 0 else height))
                    for offset, value in enumerate(racket_bbox))
                cv2.rectangle(
                    frame, (racket_box[0], racket_box[1]),
                    (racket_box[2], racket_box[3]), (0, 220, 255), 3)
            keypoints = observation.get("pose_keypoints_norm") or []
            confidence = observation.get("pose_keypoint_confidence") or []
            points: list[tuple[int, int] | None] = []
            for joint, score in zip(keypoints, confidence):
                if (not isinstance(joint, (list, tuple)) or len(joint) != 2
                        or float(score) < 0.2):
                    points.append(None)
                    continue
                point = (int(round(float(joint[0]) * width)),
                         int(round(float(joint[1]) * height)))
                points.append(point)
                cv2.circle(frame, point, 4, color, -1, lineType=cv2.LINE_AA)
            for left, right in skeleton:
                if left < len(points) and right < len(points) \
                        and points[left] is not None and points[right] is not None:
                    cv2.line(frame, points[left], points[right], color, 2,
                             lineType=cv2.LINE_AA)
            hand = observation.get("racket_hand_xy_norm")
            if isinstance(hand, (list, tuple)) and len(hand) == 2:
                hand_point = (int(round(float(hand[0]) * width)),
                              int(round(float(hand[1]) * height)))
                cv2.circle(frame, hand_point, 12, (0, 220, 255), 3,
                           lineType=cv2.LINE_AA)
            label = (
                f"{'ACCEPT' if accepted else 'REJECT'}  {time_s:.2f}s  "
                f"sequence={'yes' if observation.get('serve_sequence_evidence') else 'no'}  "
                f"rise={float(observation.get('wrist_rise_span') or 0):.2f}  "
                f"knee={int(observation.get('knee_bend_frames') or 0)}"
            )
            cv2.rectangle(frame, (0, 0), (min(width, 760), 42), (20, 20, 20), -1)
            cv2.putText(frame, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, color, 2, lineType=cv2.LINE_AA)
            path = destination / f"serve_{index}.jpg"
            temporary = destination / f".serve_{index}.{uuid.uuid4().hex}.jpg"
            if not cv2.imwrite(str(temporary), frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                raise RuntimeError(f"could not write serve-pose evidence: {path}")
            os.replace(temporary, path)
            written += 1
    finally:
        cap.release()
    return written


def _racket_actions(stages: dict[str, Any]) -> list[dict[str, Any]]:
    stage = stages.get("racket_actions") or {}
    episodes = stage.get("episodes")
    if isinstance(episodes, list):
        return [dict(raw) for raw in episodes if isinstance(raw, dict)]
    actions: list[dict[str, Any]] = []
    for decision in stage.get("decisions") or []:
        if not isinstance(decision, dict):
            continue
        episodes = decision.get("stroke_episodes")
        if isinstance(episodes, list):
            actions.extend(dict(raw) for raw in episodes if isinstance(raw, dict))
            continue
        for raw in decision.get("actions") or []:
            if isinstance(raw, dict) and raw.get("action") != "serve":
                actions.append(dict(raw))
    return actions


def _render_racket_signal_artifacts(
    source: Path, stages: dict[str, Any], destination: Path,
) -> int:
    """Write an annotated frame for each temporal wrist-motion stroke decision."""
    import cv2

    actions = _racket_actions(stages)
    if not actions:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("could not open source video for racket-action evidence")
    skeleton = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    written = 0
    try:
        for index, action in enumerate(actions):
            try:
                time_s = float(action.get("time"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(time_s):
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            accepted = bool(action.get("accepted", True))
            color = (75, 205, 95) if accepted else (70, 95, 235)
            bbox = action.get("bbox_norm") or []
            if len(bbox) == 4:
                box = tuple(int(round(float(value) * (width if offset % 2 == 0 else height)))
                            for offset, value in enumerate(bbox))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
            racket_bbox = action.get("racket_bbox_norm") or []
            if len(racket_bbox) == 4:
                racket_box = tuple(
                    round(float(value) * (width if offset % 2 == 0 else height))
                    for offset, value in enumerate(racket_bbox))
                cv2.rectangle(
                    frame, (racket_box[0], racket_box[1]),
                    (racket_box[2], racket_box[3]), (0, 220, 255), 3)
            keypoints = action.get("keypoints_norm") or []
            confidence = action.get("keypoint_confidence") or []
            points: list[tuple[int, int] | None] = []
            for joint_index in range(min(17, len(keypoints))):
                joint = keypoints[joint_index]
                score = float(confidence[joint_index]) if joint_index < len(confidence) else 0.0
                if not isinstance(joint, (list, tuple)) or len(joint) != 2 or score < 0.2:
                    points.append(None)
                    continue
                point = (int(round(float(joint[0]) * width)),
                         int(round(float(joint[1]) * height)))
                points.append(point)
                cv2.circle(frame, point, 4, color, -1, lineType=cv2.LINE_AA)
            for left, right in skeleton:
                if left < len(points) and right < len(points) \
                        and points[left] is not None and points[right] is not None:
                    cv2.line(frame, points[left], points[right], color, 2,
                             lineType=cv2.LINE_AA)
            label = (
                f"{'ACCEPT' if accepted else 'REJECT'}  "
                f"{str(action.get('action') or 'stroke').upper()}  "
                f"{action.get('actor_id') or 'unassigned'}  {time_s:.2f}s  "
                f"confidence={float(action.get('confidence') or 0):.2f}"
            )
            cv2.rectangle(frame, (0, 0), (min(width, 850), 42), (20, 20, 20), -1)
            cv2.putText(frame, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, color, 2, lineType=cv2.LINE_AA)
            path = destination / f"action_{index}.jpg"
            temporary = destination / f".action_{index}.{uuid.uuid4().hex}.jpg"
            if not cv2.imwrite(str(temporary), frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                raise RuntimeError(f"could not write racket-action evidence: {path}")
            os.replace(temporary, path)
            written += 1
    finally:
        cap.release()
    return written


def _merge_match_profile(existing: Any, detected: Any) -> dict[str, Any]:
    """Keep user names while refreshing detector-owned format, teams, and identities."""
    detected = dict(detected) if isinstance(detected, dict) else {}
    existing = dict(existing) if isinstance(existing, dict) else {}
    if not (detected.get("roster") or []) and existing.get("roster"):
        # A failed/no-point re-analysis must never erase user-entered names. There is no
        # fresh identity structure to merge, so retain the last durable match profile.
        return existing
    prior_names = {
        str(record.get("id")): str(record.get("name") or "").strip()
        for record in (existing.get("roster") or [])
        if isinstance(record, dict) and record.get("id")
    }
    roster = []
    for record in detected.get("roster") or []:
        if not isinstance(record, dict) or not record.get("id"):
            continue
        item = dict(record)
        player_id = str(item["id"])
        if prior_names.get(player_id):
            item["name"] = prior_names[player_id]
        roster.append(item)
    detected["roster"] = roster
    detected["names_updated_at"] = existing.get("names_updated_at")
    return detected


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

        point_events = clean.get("points") or []

        cursor = 0.0
        layout = []
        gap_s = float(cfg.inter_point_gap_s)
        for index, (point, clip) in enumerate(zip(points, clips)):
            clip_duration = max(0.0, clip[1] - clip[0])
            point_event = next((event for event in point_events
                                if int(event.get("index", -1)) == index), None)
            item = {
                "index": index,
                "source_start": round(clip[0], 3),
                "source_end": round(clip[1], 3),
                "detected_start": round(point[0], 3),
                "detected_end": round(point[1], 3),
                "output_start": round(cursor, 3),
                "output_end": round(cursor + clip_duration, 3),
            }
            if point_event is not None:
                item["participants"] = point_event.get("participants") or {}
                item["termination"] = point_event.get("termination") or {}
                item["actions"] = point_event.get("actions") or []
            layout.append(item)
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
    mapping = {
        "pose_fps": "pose_timeline_fps",
        "min_rally": "min_rally_s",
        "skip_intro": "skip_intro_s",
        "gap": "inter_point_gap_s",
        "start_buffer": "point_start_buffer_s",
        "end_buffer": "point_end_buffer_s",
    }
    for opt, cfg_name in mapping.items():
        val = options.get(opt)
        if val is not None:
            overrides[cfg_name] = val
    if options.get("no_labels"):
        overrides["label_points"] = False
    if options.get("fast"):
        overrides["reencode"] = False
    # Setup/startup preflight guarantees the required modern runtime dependencies.
    overrides["court_auto"] = True
    return RallyConfig(**overrides)


def _validate_options(options: dict[str, Any]) -> None:
    """Reject invalid or pathological web options before accepting an upload."""
    non_negative = (
        "min_rally", "skip_intro", "gap", "start_buffer", "end_buffer")
    for name in non_negative:
        value = options.get(name)
        if value is not None and value < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be non-negative")
    pose_fps = options.get("pose_fps")
    if pose_fps is not None and not 0 < pose_fps <= 120:
        raise HTTPException(status_code=400, detail="pose_fps must be in (0, 120]")
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
                    serve_times: list[float] | tuple[float, ...] | None = None,
                    destination: Path | None = None) -> None:
    """Cache visual serve events for the review timeline; never decode audio."""
    del src, cfg
    try:
        progress("writing event timeline from visual serve times")
        serves = [
            float(value) for value in (serve_times or ())
            if math.isfinite(float(value))
        ]
        data = {"duration": round(duration, 3),
                "serves": [round(value, 3) for value in serves]}
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

    keep_partial_artifacts = False
    try:
        check_cancel()
        job_dir = _job_dir(job_id)
        output_path = job_dir / "output" / "rallies.mp4"
        json_path = job_dir / "output" / "rallies.json"
        attempt_output_path = output_path.with_name(f".rallies.{attempt_id}.mp4")
        attempt_waveform_path = job_dir / f".waveform.{attempt_id}.json"
        attempt_player_thumbnails = output_path.parent / f".player-thumbnails.{attempt_id}"
        attempt_signal_artifacts = output_path.parent / f".signal-evidence.{attempt_id}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def start(current: dict[str, Any]) -> None:
            if current.get("active_attempt_id") != attempt_id:
                return
            current["status"] = "running"
            current["cancel_requested"] = False
            current["error"] = None
            current["signal_snapshot"] = {
                "attempt_id": attempt_id,
                "current_stage": "starting",
                "total_seconds": 0.0,
                "stages": {},
                "match": {"roster": [], "teams": []},
                "points": [],
                "segments": [],
                "updated_at": _now(),
            }
            _mark_labeling_stale(current)
            _set_processing(current, "starting", "Starting", 12, "Preparing output files")

        job = _mutate_job(job_id, start)
        if job.get("active_attempt_id") != attempt_id:
            return
        _prune_label_revisions(job_id)
        _append_progress(job_id, "processing started", attempt_id=attempt_id)
        # Publish an original-video still immediately so the primary processed-video
        # panel has useful visual context throughout analysis and rendering. This is
        # retained across re-runs; _ensure_thumbnail is a cheap no-op when it exists.
        _ensure_thumbnail(_load_job(job_id))

        def progress(message: str) -> None:
            check_cancel()
            _append_progress(job_id, message, attempt_id=attempt_id)
            check_cancel()

        options = dict(job.get("options", {}))
        cfg = _config_from_options(options)

        def publish_signal_snapshot(raw_snapshot: dict[str, Any]) -> None:
            """Publish stage data immediately, then attach optional visual artifacts."""
            check_cancel()
            snapshot = dict(raw_snapshot)
            stage = str(snapshot.get("current_stage") or "unknown")
            with _LOCK:
                current = _read_json(_job_meta_path(job_id), None)
                if not current or current.get("active_attempt_id") != attempt_id:
                    return
                match = snapshot.get("match")
                if isinstance(match, dict) and match.get("roster"):
                    snapshot["match"] = _merge_match_profile(current.get("match"), match)
                snapshot["attempt_id"] = attempt_id
                snapshot["updated_at"] = _now()
                current["signal_snapshot"] = snapshot
                current["updated_at"] = snapshot["updated_at"]
                _atomic_write_json(_job_meta_path(job_id), current)

            artifact_stage = stage in {
                "match_format", "serve_pose", "racket_actions"}
            if not artifact_stage:
                return
            artifact_errors: list[str] = []
            try:
                if stage == "serve_pose":
                    _render_serve_signal_artifacts(
                        Path(job["original_path"]), snapshot.get("stages") or {},
                        attempt_signal_artifacts / "serves")
                elif stage == "racket_actions":
                    _render_racket_signal_artifacts(
                        Path(job["original_path"]), snapshot.get("stages") or {},
                        attempt_signal_artifacts / "actions")
                else:
                    _render_match_player_thumbnails(
                        Path(job["original_path"]), snapshot.get("match") or {},
                        attempt_player_thumbnails)
            except Exception as exc:
                artifact_errors.append(str(exc))
            check_cancel()
            with _LOCK:
                current = _read_json(_job_meta_path(job_id), None)
                live = current.get("signal_snapshot") if current else None
                if (not current or current.get("active_attempt_id") != attempt_id
                        or not isinstance(live, dict)
                        or live.get("current_stage") != stage):
                    return
                live["updated_at"] = _now()
                if artifact_errors:
                    live["artifact_errors"] = artifact_errors
                current["updated_at"] = live["updated_at"]
                _atomic_write_json(_job_meta_path(job_id), current)

        # Analysis only: always yields a segment list, even without ffmpeg encode.
        result = trim(job["original_path"], output_path=None, cfg=cfg, json_path=None,
                      detect_players=True,
                      progress=progress, cancel_check=check_cancel,
                      signal_callback=publish_signal_snapshot)

        sidecar = result.sidecar()
        info = probe(job["original_path"])
        sidecar["info"] = {"fps": info.fps, "width": info.width,
                           "height": info.height, "has_audio": info.has_audio}

        serve_times = sidecar.get("serve_times")
        _write_waveform(job_id, Path(job["original_path"]), result.total_seconds, cfg, progress,
                        serve_times if isinstance(serve_times, (list, tuple)) else None,
                        destination=attempt_waveform_path)

        rendered = False
        if result.segments:
            rendered = _render_output(Path(job["original_path"]), result.segments,
                                      attempt_output_path, cfg, info, progress,
                                      cancel_check=check_cancel)

        check_cancel()
        output_ready = rendered and attempt_output_path.exists()
        sidecar = _normalise_web_sidecar(job, sidecar, output_ready=output_ready)
        if not attempt_player_thumbnails.exists():
            _render_match_player_thumbnails(
                Path(job["original_path"]), sidecar.get("match") or {},
                attempt_player_thumbnails)
        if not (attempt_signal_artifacts / "serves").exists():
            _render_serve_signal_artifacts(
                Path(job["original_path"]), sidecar.get("stages") or {},
                attempt_signal_artifacts / "serves")
        if not (attempt_signal_artifacts / "actions").exists():
            _render_racket_signal_artifacts(
                Path(job["original_path"]), sidecar.get("stages") or {},
                attempt_signal_artifacts / "actions")
        # Generation-checked atomic publication: an attempt from a server process that
        # was superseded by a retry may finish later, but it cannot replace newer media,
        # metadata, progress, or terminal state.
        with _LOCK:
            current = _read_json(_job_meta_path(job_id), None)
            if not current or current.get("active_attempt_id") != attempt_id:
                return
            match_profile = _merge_match_profile(
                current.get("match"), sidecar.get("match"))
            sidecar["match"] = match_profile
            current["match"] = match_profile
            if output_ready:
                os.replace(attempt_output_path, output_path)
            else:
                output_path.unlink(missing_ok=True)
            if attempt_waveform_path.exists():
                os.replace(attempt_waveform_path, job_dir / "waveform.json")
            published_thumbnails = _player_thumbnail_dir(job_id)
            shutil.rmtree(published_thumbnails, ignore_errors=True)
            if attempt_player_thumbnails.exists():
                os.replace(attempt_player_thumbnails, published_thumbnails)
            published_signals = _signal_artifact_dir(job_id)
            shutil.rmtree(published_signals, ignore_errors=True)
            if attempt_signal_artifacts.exists():
                os.replace(attempt_signal_artifacts, published_signals)
            _atomic_write_json(json_path, sidecar)
            current["status"] = "complete" if output_ready else "no_output"
            current["retryable"] = False
            current["cancel_requested"] = False
            current["result"] = sidecar
            current.pop("signal_snapshot", None)
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
        error = str(exc)
        try:
            partial = (_read_json(_job_meta_path(job_id), {}) or {}).get("signal_snapshot")
            keep_partial_artifacts = bool(
                isinstance(partial, dict) and (partial.get("stages") or {}))
        except (OSError, ValueError):
            keep_partial_artifacts = False
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
                    current["error"] = error
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
                            f"Re-run failed: {error}",
                        )
                    else:
                        _set_processing(current, "failed", "Failed", 100, error)

                _mutate_job(job_id, fail)
                _append_progress(
                    job_id, f"processing failed: {error}", attempt_id=attempt_id)
        except HTTPException:
            pass
    finally:
        try:
            attempt_output_path.unlink(missing_ok=True)
            attempt_waveform_path.unlink(missing_ok=True)
            if not keep_partial_artifacts:
                shutil.rmtree(attempt_player_thumbnails, ignore_errors=True)
                shutil.rmtree(attempt_signal_artifacts, ignore_errors=True)
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
        # Admission succeeded.  A full reprocess invalidates its prior publication
        # immediately; failed or cancelled attempts must not resurrect stale analysis.
        _discard_published_result(job_id, job)
        job["options"] = dict(job.get("options", {}))
        attempt_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        _CANCEL_EVENTS[job_id] = (attempt_id, cancel_event)
        job["active_attempt_id"] = attempt_id
        job["signal_snapshot"] = {
            "attempt_id": attempt_id,
            "current_stage": "queued",
            "total_seconds": 0.0,
            "stages": {},
            "match": {"roster": [], "teams": []},
            "points": [],
            "segments": [],
            "updated_at": _now(),
        }
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
        _prune_label_revisions(job_id)
        _append_progress(job_id, "queued for processing", attempt_id=attempt_id)
        future = _EXECUTOR.submit(_run_trim_job, job_id, attempt_id)
        with _LOCK:
            _JOB_FUTURES[job_id] = (attempt_id, future)
        future.add_done_callback(
            lambda completed, jid=job_id, aid=attempt_id: (
                _forget_job_future(jid, aid, completed)))
    except Exception as exc:
        error = str(exc)
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
            current["error"] = f"could not queue processing: {error}"
            _set_processing(current, "failed", "Failed", 100, current["error"])

        _mutate_job(job_id, fail_submission)
        raise RuntimeError(f"could not queue processing: {error}") from exc
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
            error = str(exc)
            try:
                def fail_recovery(
                    current: dict[str, Any], error: str = error
                ) -> None:
                    current["status"] = "failed"
                    current["retryable"] = True
                    current["error"] = f"could not recover queued processing: {error}"
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


@app.get("/jobs/{job_id}")
@app.get("/jobs/{job_id}/signals")
@app.get("/jobs/{job_id}/signals/players/{player_id}")
def job_page(job_id: str, player_id: str | None = None) -> FileResponse:
    _load_job(job_id)
    if player_id is not None and not _ROSTER_ID.fullmatch(player_id):
        raise HTTPException(status_code=404, detail="player not found")
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
    fast: bool = Form(False),
    no_labels: bool = Form(False),
    run_now: bool = Form(True),
    pose_fps: Optional[str] = Form(None),
    min_rally: Optional[str] = Form(None),
    skip_intro: Optional[str] = Form(None),
    gap: Optional[str] = Form(None),
    start_buffer: Optional[str] = Form(None),
    end_buffer: Optional[str] = Form(None),
) -> JSONResponse:
    # Parse and validate all options before creating a job directory or retaining
    # any upload bytes. Invalid forms must not leave orphan jobs behind.
    options = {
        "fast": fast,
        "no_labels": no_labels,
        "pose_fps": _parse_optional_float(pose_fps),
        "min_rally": _parse_optional_float(min_rally),
        "skip_intro": _parse_optional_float(skip_intro),
        "gap": _parse_optional_float(gap),
        "start_buffer": _parse_optional_float(start_buffer),
        "end_buffer": _parse_optional_float(end_buffer),
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
        "match": {},
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


@app.post("/api/jobs/{job_id}/match")
def update_match_roster(job_id: str, update: MatchRosterUpdate) -> JSONResponse:
    """Rename automatically detected players without changing detector-owned format."""
    supplied: dict[str, str] = {}
    for record in update.roster:
        player_id = str(record.get("id") or "")
        name = str(record.get("name") or "").strip()
        if not _ROSTER_ID.fullmatch(player_id):
            raise HTTPException(status_code=422, detail="invalid player id")
        if not name or len(name) > 100:
            raise HTTPException(status_code=422, detail="player names must be 1..100 characters")
        if player_id in supplied:
            raise HTTPException(status_code=422, detail="duplicate player id")
        supplied[player_id] = name

    with _LOCK:
        path = _job_meta_path(job_id)
        job = _read_json(path, None)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        match = dict(job.get("match") or {})
        roster = [dict(record) for record in (match.get("roster") or [])
                  if isinstance(record, dict) and record.get("id")]
        expected = {str(record["id"]) for record in roster}
        if not expected:
            raise HTTPException(status_code=409, detail="players have not been detected yet")
        if set(supplied) != expected:
            raise HTTPException(
                status_code=422,
                detail=f"roster must contain exactly: {', '.join(sorted(expected))}",
            )
        for record in roster:
            record["name"] = supplied[str(record["id"])]
        match["roster"] = roster
        match["names_updated_at"] = _now()
        job["match"] = match
        if isinstance(job.get("result"), dict):
            job["result"]["match"] = match
        job["updated_at"] = _now()
        if job.get("json_path") and Path(job["json_path"]).exists():
            sidecar = _read_json(Path(job["json_path"]), {})
            if isinstance(sidecar, dict):
                sidecar["match"] = match
                _atomic_write_json(Path(job["json_path"]), sidecar)
        _atomic_write_json(path, job)
    return JSONResponse({"match": _public_match(job_id, match)})


def _capabilities() -> dict[str, Any]:
    """Report the required processing features verified again at server startup.

    The UI still exposes per-job controls, but a missing required feature is now a startup
    error rather than a reason to launch a degraded service.
    """
    import importlib.util

    players_ok = importlib.util.find_spec("ultralytics") is not None
    rtmlib_ok = importlib.util.find_spec("rtmlib") is not None
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
        "racket_actions": {
            "available": bool(players_ok and rtmlib_ok and pose_model_present),
            "backend": "shared_rtmpose_coco17_timeline",
            "ball_tracking": False,
        },
        # classical court detection only needs OpenCV, a core dependency
        "court_auto": {"available": True},
        "players": {
            "available": players_ok,
            "hint": ("" if players_ok else
                     "Ultralytics is not installed; player tracking is unavailable"),
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


@app.get("/api/jobs/{job_id}/players/{player_id}/thumbnails/{index}")
def get_player_thumbnail(job_id: str, player_id: str, index: int) -> FileResponse:
    """Serve only tracker-owned identity thumbnails from this completed job."""
    if not _ROSTER_ID.fullmatch(player_id) or not 0 <= index < 40:
        raise HTTPException(status_code=404, detail="player thumbnail not found")
    job = _load_job(job_id)
    snapshot = job.get("signal_snapshot") if isinstance(job.get("signal_snapshot"), dict) else {}
    match = snapshot.get("match") or job.get("match") or {}
    record = next((
        item for item in match.get("roster", [])
        if isinstance(item, dict) and str(item.get("id")) == player_id
    ), None)
    sources = ((record or {}).get("inspection_sources")
               or (record or {}).get("thumbnail_sources") or [])
    if record is None or index >= len(sources):
        raise HTTPException(status_code=404, detail="player thumbnail not found")
    path = _player_artifact_dir(job) / f"{player_id}_{index}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="player thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


def _signal_payload(job: dict[str, Any]) -> dict[str, Any]:
    snapshot = job.get("signal_snapshot")
    has_snapshot = isinstance(snapshot, dict)
    live = bool(has_snapshot and job.get("status") in {"queued", "running"})
    result = snapshot if has_snapshot else (
        job.get("result") if isinstance(job.get("result"), dict) else {})
    stages = result.get("stages") if isinstance(result.get("stages"), dict) else {}
    match = _public_match(
        job["id"], result.get("match") or job.get("match") or {},
        artifact_root=_player_artifact_dir(job))
    pose_stage = stages.get("serve_pose") if isinstance(stages.get("serve_pose"), dict) else {}
    raw_pose_observations = pose_stage.get("observations") or []
    serve_root = _serve_artifact_dir(job)
    serve_observations = []
    for index, raw in enumerate(raw_pose_observations):
        if not isinstance(raw, dict):
            continue
        item = {
            "index": index,
            "time": raw.get("first_strike"),
            "pose_evidence_time": raw.get("pose_evidence_time"),
            "point": raw.get("point"),
            "pose_accepted": bool(raw.get("accepted")),
            "serve_motion": bool(raw.get("serve_motion")),
            "serve_sequence": bool(raw.get("serve_sequence_evidence")),
            "serve_sequence_score": raw.get("serve_sequence_score"),
            "overhead_frames": int(raw.get("overhead_frames") or 0),
            "overhead_ratio": raw.get("overhead_max_ratio"),
            "wrist_rise_span": raw.get("wrist_rise_span"),
            "hand_speed_body_s": raw.get("hand_speed_body_s"),
            "knee_bend_frames": int(raw.get("knee_bend_frames") or 0),
            "server_load_frames": int(raw.get("server_load_frames") or 0),
            "leg_drive_frames": int(raw.get("leg_drive_frames") or 0),
            "server_baseline_frames": int(raw.get("server_baseline_frames") or 0),
            "opposed_formation_frames": int(raw.get("opposed_formation_frames") or 0),
            "position_accepted": bool(raw.get("position_setup_evidence")),
            "position_score": raw.get("position_score"),
            "pose_frames": int(raw.get("pose_frames") or 0),
            "sampled_frames": int(raw.get("sampled_frames") or 0),
            "ready_frames": int(raw.get("ready_frames") or 0),
            "racket_hand_confidence": raw.get("racket_hand_confidence"),
            "racket_observed_frames": int(raw.get("racket_observed_frames") or 0),
            "racket_wrist_associated": bool(raw.get("racket_wrist_associated")),
            "server_end": raw.get("pose_server_end") or raw.get("position_server_end"),
            "server_court_x_m": raw.get("pose_server_court_x_m"),
        }
        path = serve_root / f"serve_{index}.jpg"
        if path.is_file():
            item["image"] = (
                f"/api/jobs/{job['id']}/signals/serves/{index}?v={path.stat().st_mtime_ns}")
        serve_observations.append(item)
    action_stage = (stages.get("racket_actions")
                    if isinstance(stages.get("racket_actions"), dict) else {})
    action_root = _action_artifact_dir(job)
    actions = _racket_actions(stages)
    for index, action in enumerate(actions):
        action["index"] = index
        path = action_root / f"action_{index}.jpg"
        if path.is_file():
            action["image"] = (
                f"/api/jobs/{job['id']}/signals/actions/{index}?v={path.stat().st_mtime_ns}")
    endpoint_stage = stages.get("endpoints") if isinstance(stages.get("endpoints"), dict) else {}
    quality_stage = (stages.get("quality_control")
                     if isinstance(stages.get("quality_control"), dict) else {})
    stage_order = (
        "audio", "court", "visual", "match_format", "pose_timeline", "serve_pose", "candidate_generation",
        "racket_actions", "endpoints", "quality_control",
    )
    stage_summary = []
    for name in stage_order:
        raw = stages.get(name)
        if isinstance(raw, dict):
            stage_summary.append({
                "name": name,
                "status": raw.get("status", "recorded"),
                "reason": raw.get("reason"),
            })
    return {
        "job": {"id": job["id"], "filename": job.get("filename"),
                "status": job.get("status"),
                "processing": job.get("processing") or {}},
        "live": live,
        "current_stage": result.get("current_stage"),
        "updated_at": result.get("updated_at") or job.get("updated_at"),
        "duration": result.get("total_seconds", 0),
        "stage_summary": stage_summary,
        "court": stages.get("court") or {},
        "visual": stages.get("visual") or {},
        "match_format": stages.get("match_format") or {},
        "pose_timeline": stages.get("pose_timeline") or {},
        "candidate_generation": stages.get("candidate_generation") or {},
        "players": match.get("roster") or [],
        "teams": match.get("teams") or [],
        "serve_pose": {**pose_stage, "observations": serve_observations},
        "racket_actions": {**action_stage, "actions": actions},
        "points": result.get("points") or [],
        "endpoints": endpoint_stage,
        "quality_control": quality_stage,
    }


@app.get("/api/jobs/{job_id}/signals")
def get_job_signals(job_id: str) -> JSONResponse:
    return JSONResponse(_signal_payload(_load_job(job_id)))


@app.get("/api/jobs/{job_id}/signals/players/{player_id}")
def get_player_signals(job_id: str, player_id: str) -> JSONResponse:
    if not _ROSTER_ID.fullmatch(player_id):
        raise HTTPException(status_code=404, detail="player not found")
    payload = _signal_payload(_load_job(job_id))
    player = next((record for record in payload["players"]
                   if str(record.get("id")) == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail="player not found")
    return JSONResponse({
        "job": payload["job"],
        "duration": payload["duration"],
        "live": payload["live"],
        "current_stage": payload["current_stage"],
        "updated_at": payload["updated_at"],
        "player": player,
    })


@app.get("/api/jobs/{job_id}/signals/serves/{index}")
def get_serve_signal_image(job_id: str, index: int) -> FileResponse:
    if not 0 <= index < 10000:
        raise HTTPException(status_code=404, detail="serve evidence not found")
    job = _load_job(job_id)
    source = (job.get("signal_snapshot")
              if isinstance(job.get("signal_snapshot"), dict)
              else (job.get("result") or {}))
    observations = (((source.get("stages") or {})
                     .get("serve_pose") or {}).get("observations") or [])
    if index >= len(observations):
        raise HTTPException(status_code=404, detail="serve evidence not found")
    path = _serve_artifact_dir(job) / f"serve_{index}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="serve evidence not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/signals/actions/{index}")
def get_racket_action_image(job_id: str, index: int) -> FileResponse:
    if not 0 <= index < 100000:
        raise HTTPException(status_code=404, detail="racket-action evidence not found")
    job = _load_job(job_id)
    source = (job.get("signal_snapshot")
              if isinstance(job.get("signal_snapshot"), dict)
              else (job.get("result") or {}))
    actions = _racket_actions(source.get("stages") or {})
    if index >= len(actions):
        raise HTTPException(status_code=404, detail="racket-action evidence not found")
    path = _action_artifact_dir(job) / f"action_{index}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="racket-action evidence not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/waveform")
def get_waveform(job_id: str) -> JSONResponse:
    job = _load_job(job_id)
    data = _read_json(_job_dir(job_id) / "waveform.json", {"serves": [], "duration": 0})
    result = job.get("result") or {}
    if isinstance(result.get("serve_times"), list):
        data["serves"] = result["serve_times"]
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
    # Point action groupings are derived from automatic boundaries. Manual edits
    # invalidate those groupings; retain the durable match roster only.
    sidecar["points"] = []
    stages = sidecar.setdefault("stages", {})
    stages["racket_actions"] = {
        "status": "stale", "reason": "manual segment boundaries changed",
    }
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
        _prune_label_revisions(job_id)
        # The metadata pointer now owns the new immutable artifacts. Old files can be
        # removed afterward without ever leaving the job with no valid published result.
        for key in ("output_path", "json_path"):
            old = previous_job.get(key)
            if old and old not in {str(attempt_out), str(attempt_json)}:
                Path(old).unlink(missing_ok=True)
        return JSONResponse(_public_job(job))
    except Exception as exc:
        error = str(exc)
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
                current["error"] = f"could not apply segment edits: {error}"
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
                           assets_dir: Path):
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

    assets = assets_dir
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
                          max_items: int, regenerate: bool, assets_dir: Path,
                          serve_pose: Optional[dict] = None) -> list[dict[str, Any]]:
    """Cut short clips around each serve moment (rally start) to classify."""
    count = min(max_items, max(1, len(segments) or max_items))
    if segments:
        selected = _evenly_pick(list(enumerate(segments)), count)
    else:
        selected = [(None, {"start": value}) for value in _serve_times([], duration, count)]
    observations = (serve_pose or {}).get("observations") or []
    assets = assets_dir
    tasks: list[dict[str, Any]] = []
    for i, (source_index, segment) in enumerate(selected[:max_items]):
        start = float(segment["start"])
        end = float(segment.get("end", start + 5.0))
        observation = next((
            (index, item) for index, item in enumerate(observations)
            if isinstance(item, dict) and item.get("accepted")
            and start <= float(item.get("first_strike", -1.0)) <= end
        ), None)
        event_time = (float(observation[1]["first_strike"])
                      if observation is not None else start)
        clip_start = max(0.0, event_time - 1.5)
        clip_end = min(duration, event_time + 3.5) if duration else event_time + 3.5
        rel = f"serve_{i:04d}.mp4"
        path = assets / rel
        if regenerate or not path.exists():
            _ffmpeg_clip(src, clip_start, clip_end, path)
        stable_context: dict[str, Any] = {}
        if source_index is not None:
            stable_context["source_segment_index"] = int(
                segment.get("index", source_index))
            if observation is not None:
                observation_index, record = observation
                stable_context.update({
                    "serve_pose_observation_index": int(observation_index),
                    "suggested_server_id": record.get("actor_id"),
                    "serve_sequence_score": record.get("serve_sequence_score"),
                })
        tasks.append({
            "id": f"serve_{i:04d}",
            "kind": "serve_motion",
            "title": f"Serve clip {i + 1}",
            "time_s": round(event_time, 3),
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

        detected_match = job.get("match") or result.get("match") or {}
        detected_format = str(detected_match.get("format") or "")
        match_type = detected_format if detected_format in {"singles", "doubles"} else "singles"
        durable_roster = [
            dict(record) for record in (detected_match.get("roster") or [])
            if isinstance(record, dict) and record.get("id")
        ]
        roster = durable_roster or _roster_for(match_type)
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
                serve_pose=((result.get("stages") or {}).get("serve_pose") or {}),
                assets_dir=build_assets))

        # persist roster (preserve any user-renamed names) + tasks
        previous_revision = (job.get("labeling") or {}).get("revision")
        previous_roster = (
            _job_dir(job_id) / "label_revisions" / str(previous_revision)
            / "labels" / "roster.json"
            if previous_revision else None)
        existing = _read_json(previous_roster, None) if previous_roster else None
        if existing and not req.regenerate:
            names = {r["id"]: r.get("name") for r in existing if isinstance(r, dict)}
            for r in roster:
                if names.get(r["id"]):
                    r["name"] = names[r["id"]]
        durable_names = {
            str(record.get("id")): str(record.get("name") or "").strip()
            for record in ((job.get("match") or {}).get("roster") or [])
            if isinstance(record, dict) and record.get("id")
        }
        for record in roster:
            if durable_names.get(str(record["id"])):
                record["name"] = durable_names[str(record["id"])]
        _atomic_write_json(build_labels / "roster.json", roster)
        _atomic_write_json(build_labels / "tasks.json", tasks)
        stages = result.get("stages") or {}
        _atomic_write_json(build_labels / "feature_context.json", {
            "schema_version": "rally.pose_serve_context.v1",
            "serve_times": result.get("serve_times", []),
            "segments": result.get("segments", []),
            "serve_pose": stages.get("serve_pose", {}),
            "pipeline_config": result.get("config", {}),
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
        _prune_label_revisions(job_id)
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
        job, root = _label_root_locked(job_id, writable=True)
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
        job, root = _label_root_locked(job_id, writable=True)
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
        feature_context = _read_json(labels_dir / "feature_context.json", None)
        if not isinstance(feature_context, dict):
            raise HTTPException(
                status_code=409,
                detail="label revision lacks feature context; regenerate label samples",
            )
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

    # Fail before uvicorn opens a listening socket. The lifespan repeats this guard so
    # direct `uvicorn rally.web.app:app` launches receive the same protection.
    from rally.preflight import InstallationError, require_server_install

    try:
        require_server_install()
    except InstallationError as exc:
        print(f"rally-web: {exc}", file=sys.stderr)
        return 1
    _ensure_data_dir()

    import uvicorn

    uvicorn.run("rally.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
