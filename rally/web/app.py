"""FastAPI web UI for the rally trimmer.

Thin layer over :func:`rally.pipeline.trim`. The core pipeline is imported and
used unchanged; everything video-specific lives in the ``rally`` package and is
treated as a black box here.

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
import json
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rally.config import RallyConfig
from rally.io.ffmpeg import _require, cut_segments, find_font, load_audio_mono, probe, render_labeled
from rally.pipeline import timeline_array, trim
from rally.signals.audio import detect_strikes
from rally.signals.player import estimate_court_region

# --------------------------------------------------------------------------- #
# module state                                                                #
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
DATA_DIR = Path(os.environ.get("RALLY_WEB_DATA", ".rally_web")).resolve()

_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get("RALLY_WEB_WORKERS", "1")))
_ACTIVE: set[str] = set()
_LABEL_ACTIVE: set[str] = set()

_YOLO_LOCK = threading.Lock()
_YOLO_MODEL: Any = None

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_LABEL_KINDS = {"player_identity", "serve_motion"}

app = FastAPI(title="Rally — rally trimmer")


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str | None) -> str:
    raw = Path(name or "upload.mp4").name
    stem = "".join(c if c.isalnum() or c in "._- " else "_" for c in raw).strip(" .")
    return stem or "upload.mp4"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _load_job(job_id: str) -> dict[str, Any]:
    path = _job_meta_path(job_id)
    with _LOCK:
        job = _read_json(path, None)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = _now()
    with _LOCK:
        _atomic_write_json(_job_meta_path(job["id"]), job)


# --------------------------------------------------------------------------- #
# progress: map pipeline log lines to a coarse stage + monotonic percent      #
# --------------------------------------------------------------------------- #
_STAGE_RULES = [
    ("processing failed", "failed", "Failed", 100),
    ("no rally segments found", "no_output", "No rallies found", 100),
    ("upload complete", "uploaded", "Uploaded", 5),
    ("queued for processing", "queued", "Queued", 8),
    ("processing started", "starting", "Starting", 12),
    ("probing", "probing", "Reading video", 18),
    ("duration=", "probing", "Reading video", 22),
    ("decoding audio", "audio", "Detecting ball strikes", 32),
    ("strikes detected", "audio", "Detecting ball strikes", 40),
    ("sampling frames", "visual", "Analysing motion & players", 52),
    ("co-deciding", "deciding", "Fusing signals", 66),
    ("decoded", "deciding", "Points found", 74),
    ("court serve detection", "refining", "Refining serves", 78),
    ("ball point-end", "refining", "Refining rally ends", 82),
    ("computing waveform", "waveform", "Building timeline", 86),
    ("rendering", "rendering", "Rendering video", 92),
    ("cutting", "rendering", "Rendering video", 92),
    ("wrote", "writing", "Writing output", 95),
]


def _stage_for_message(message: str) -> dict[str, Any]:
    text = message.lower()
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


def _append_progress(job_id: str, message: str) -> None:
    """Append a log line and advance the coarse progress bar (never backwards)."""
    message = str(message)
    with _LOCK:
        job = _read_json(_job_meta_path(job_id), None)
        if not job:
            return
        log = job.setdefault("progress", [])
        log.append({"at": _now(), "message": message})
        del log[:-400]

        nxt = _stage_for_message(message)
        prev = job.get("processing") or {}
        prev_pct = int(prev.get("percent") or 0)
        # keep the bar monotonic while the job is running: an unrecognised line
        # (percent 50) or a lower-percent stage must not drag it back.
        if nxt["stage"] == "running" or nxt["percent"] < prev_pct:
            if prev and prev.get("stage") not in {"complete", "failed", "no_output"}:
                nxt = {"stage": prev.get("stage", "running"),
                       "label": prev.get("label", "Processing"),
                       "percent": prev_pct}
        _set_processing(job, nxt["stage"], nxt["label"], nxt["percent"], message)
        _atomic_write_json(_job_meta_path(job_id), job)


# --------------------------------------------------------------------------- #
# public view + media URLs                                                     #
# --------------------------------------------------------------------------- #
def _media_url(job_id: str, kind: str, path: Path, *, download: bool = False) -> str:
    version = int(path.stat().st_mtime)  # cache-bust when the file changes
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
        if p and Path(p).exists():
            path = Path(p)
            if kind == "thumbnail":
                urls["thumbnail"] = _media_url(job_id, "thumbnail", path)
            elif kind == "metadata":
                urls["metadata_download"] = _media_url(job_id, "metadata", path, download=True)
            else:
                urls[kind] = _media_url(job_id, kind, path)
                if kind == "output":
                    urls["output_download"] = _media_url(job_id, "output", path, download=True)
    return urls


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal ``*_path`` fields and attach browser-facing media URLs."""
    if not job:
        return {}
    data = {k: v for k, v in job.items() if not k.endswith("_path")}
    if data.get("error") and len(str(data["error"])) > 1500:
        data["error"] = str(data["error"])[:1500].rstrip() + " ..."
    data["media"] = _media_urls(job)
    return data


# --------------------------------------------------------------------------- #
# config from web options (mirrors the CLI flags)                             #
# --------------------------------------------------------------------------- #
def _config_from_options(options: dict[str, Any]) -> RallyConfig:
    overrides: dict[str, Any] = {}
    if options.get("static_camera"):
        overrides.update(w_audio=0.7, w_motion=0.1, rhythm_window_s=5.0)
    mapping = {
        "analysis_fps": "analysis_fps",
        "min_rally": "min_rally_s",
        "skip_intro": "skip_intro_s",
        "gap": "inter_point_gap_s",
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
    # ball-arbiter and court auto-detection are on by default (best accuracy, graceful
    # fallback); respect an explicitly unchecked box. Weights are auto-discovered in the
    # pipeline, which falls back to audio-primary if none are present.
    if "ball_arbiter" in options:
        overrides["ball_arbiter"] = bool(options["ball_arbiter"])
    if "court_auto" in options:
        overrides["court_auto"] = bool(options["court_auto"])
    return RallyConfig(**overrides)


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


# --------------------------------------------------------------------------- #
# ffmpeg helpers (thumbnail + rendering with fallback)                         #
# --------------------------------------------------------------------------- #
def _ffmpeg_frame(src: Path, time_s: float, dst: Path) -> None:
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, time_s):.3f}", "-i", str(src),
         "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(dst)],
        check=True,
    )


def _ensure_thumbnail(job: dict[str, Any]) -> dict[str, Any]:
    thumb = job.get("thumbnail_path")
    if thumb and Path(thumb).exists():
        return job
    original = job.get("original_path")
    if not original or not Path(original).exists():
        return job
    path = _job_dir(job["id"]) / "thumbnail.jpg"
    try:
        info = probe(original)
        _ffmpeg_frame(Path(original), min(max(info.duration_s / 2, 0.0), 3.0), path)
    except Exception:
        return job
    job["thumbnail_path"] = str(path)
    _save_job(job)
    return job


def _render_output(src: Path, segments: list[tuple[float, float]], dst: Path,
                   cfg: RallyConfig, info, progress) -> bool:
    """Render the trimmed video, degrading gracefully if ffmpeg lacks a feature.

    labelled render  →  plain re-encode cut  →  stream-copy cut. Returns True if
    a file was written. Analysis output is unaffected by a render failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    want_labels = bool(cfg.label_points) or cfg.inter_point_gap_s > 0
    if want_labels:
        try:
            font = find_font() if cfg.label_points else None
            progress(f"rendering {len(segments)} points -> {dst.name}")
            render_labeled(str(src), segments, str(dst), gap_s=cfg.inter_point_gap_s,
                           label_prefix=cfg.label_prefix, font=font,
                           video_height=info.height, has_audio=info.has_audio,
                           draw_labels=cfg.label_points)
            return True
        except Exception as exc:
            progress(f"  labelled render failed ({exc}); falling back to a plain cut")
    try:
        progress(f"cutting {len(segments)} segments -> {dst.name}")
        cut_segments(str(src), segments, str(dst), reencode=cfg.reencode)
        return True
    except Exception as exc:
        if cfg.reencode:
            progress(f"  re-encode cut failed ({exc}); trying a fast stream-copy")
            try:
                cut_segments(str(src), segments, str(dst), reencode=False)
                return True
            except Exception as exc2:
                progress(f"  stream-copy cut failed too ({exc2})")
        else:
            progress(f"  cut failed ({exc})")
    return False


def _write_waveform(job_id: str, src: Path, duration: float, cfg: RallyConfig, progress) -> None:
    """Cache ball-strike times (+ duration) so the review timeline can draw them.

    The pipeline doesn't expose per-strike times on its result, so we recompute
    them once here (single audio decode) and cache to ``waveform.json``.
    """
    try:
        progress("computing waveform (strike timeline)")
        pcm = load_audio_mono(str(src), cfg.audio_sr)
        strikes = detect_strikes(pcm, cfg.audio_sr, cfg)
        data = {"duration": round(duration, 3), "strikes": [round(float(t), 3) for t in strikes]}
        _atomic_write_json(_job_dir(job_id) / "waveform.json", data)
    except Exception as exc:
        progress(f"  waveform cache skipped ({exc})")


# --------------------------------------------------------------------------- #
# the worker                                                                   #
# --------------------------------------------------------------------------- #
def _segments_as_tuples(sidecar: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(s["start"]), float(s["end"])) for s in sidecar.get("segments", [])]


def _run_trim_job(job_id: str) -> None:
    with _LOCK:
        if job_id in _ACTIVE:
            return
        _ACTIVE.add(job_id)
    try:
        job = _load_job(job_id)
        job["status"] = "running"
        job["error"] = None
        _set_processing(job, "starting", "Starting", 12, "Preparing output files")
        job_dir = _job_dir(job_id)
        output_path = job_dir / "output" / "rallies.mp4"
        json_path = job_dir / "output" / "rallies.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        job["json_path"] = str(json_path)
        _save_job(job)
        _append_progress(job_id, "processing started")

        def progress(message: str) -> None:
            _append_progress(job_id, message)

        options = job.get("options", {})
        cfg = _config_from_options(options)

        # Analysis only: always yields a segment list, even without ffmpeg encode.
        result = trim(job["original_path"], output_path=None, cfg=cfg, json_path=None,
                      detect_players=bool(options.get("detect_players", True)),
                      progress=progress)

        sidecar = result.sidecar()
        info = probe(job["original_path"])
        sidecar["info"] = {"fps": info.fps, "width": info.width,
                           "height": info.height, "has_audio": info.has_audio}
        _atomic_write_json(json_path, sidecar)

        _write_waveform(job_id, Path(job["original_path"]), result.total_seconds, cfg, progress)

        rendered = False
        if result.segments:
            rendered = _render_output(Path(job["original_path"]), result.segments,
                                      output_path, cfg, info, progress)

        job = _load_job(job_id)
        job["status"] = "complete"
        job["result"] = sidecar
        job["output_path"] = str(output_path) if output_path.exists() else None
        if rendered and output_path.exists():
            _set_processing(job, "complete", "Ready", 100,
                            f"{len(result.segments)} rallies — output ready")
            _append_progress(job_id, "wrote output")
        elif not result.segments:
            _set_processing(job, "no_output", "No rallies found", 100,
                            "Processing finished but no rally segments were detected")
        else:
            _set_processing(job, "no_output", "Analysis only", 100,
                            "Segments detected but video export failed (check ffmpeg) — "
                            "JSON is available")
        _save_job(job)
    except Exception as exc:
        job = _read_json(_job_meta_path(job_id), None)
        if job:
            job["status"] = "failed"
            job["error"] = str(exc)
            _set_processing(job, "failed", "Failed", 100, str(exc))
            _save_job(job)
            _append_progress(job_id, f"processing failed: {exc}")
    finally:
        with _LOCK:
            _ACTIVE.discard(job_id)


def _submit_job(job_id: str) -> None:
    job = _load_job(job_id)
    if job.get("status") not in {"queued", "running"}:
        job["status"] = "queued"
        _set_processing(job, "queued", "Queued", 8, "Waiting for a worker")
        _save_job(job)
        _append_progress(job_id, "queued for processing")
    _EXECUTOR.submit(_run_trim_job, job_id)


# --------------------------------------------------------------------------- #
# routes: pages + jobs                                                         #
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    _ensure_data_dir()
    jobs = []
    for path in sorted(DATA_DIR.glob("*/job.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        job = _read_json(path, None)
        if not job:
            continue
        job = _ensure_thumbnail(job)
        jobs.append(_public_job(job))
    return JSONResponse({"jobs": jobs})


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
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
    serve_preroll: Optional[str] = Form(None),
    tail: Optional[str] = Form(None),
) -> JSONResponse:
    _ensure_data_dir()
    job_id = str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)

    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in _VIDEO_EXTS:
        ext = ".mp4"
    original = job_dir / f"original{ext}"
    with original.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    if original.stat().st_size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    options = {
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
        "serve_preroll": _parse_optional_float(serve_preroll),
        "tail": _parse_optional_float(tail),
    }
    options = {k: v for k, v in options.items() if v is not None}

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
    }
    _save_job(job)
    job = _ensure_thumbnail(job)
    _append_progress(job_id, "upload complete")
    if run_now:
        _submit_job(job_id)
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    return JSONResponse(_public_job(_ensure_thumbnail(_load_job(job_id))))


def _capabilities() -> dict[str, Any]:
    """Which optional processing features are actually usable in this install.

    Lets the UI disable toggles it can't honour (e.g. ball-arbiter without TrackNet
    weights) instead of silently falling back mid-job.
    """
    import importlib.util

    torch_ok = importlib.util.find_spec("torch") is not None
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
    return {
        "ball_arbiter": {
            "available": bool(weights) and torch_ok,
            "weights_present": bool(weights),
            "weights_path": weights,
            "torch_installed": torch_ok,
            "hint": hint,
        },
        # classical court detection only needs OpenCV, a core dependency
        "court_auto": {"available": True},
    }


@app.get("/api/capabilities")
def capabilities() -> JSONResponse:
    return JSONResponse(_capabilities())


@app.post("/api/jobs/{job_id}/process")
def process_job(job_id: str) -> JSONResponse:
    job = _load_job(job_id)
    if job.get("status") in {"queued", "running"}:
        return JSONResponse(_public_job(job))
    _submit_job(job_id)
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}/media/{kind}")
def get_media(job_id: str, kind: str, download: bool = False) -> FileResponse:
    job = _load_job(job_id)
    paths = {"original": job.get("original_path"), "thumbnail": job.get("thumbnail_path"),
             "output": job.get("output_path"), "metadata": job.get("json_path")}
    target = paths.get(kind)
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="media not found")
    path = Path(target)
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
    data["segments"] = result.get("segments", [])
    if not data.get("duration"):
        data["duration"] = result.get("total_seconds", 0)
    return JSONResponse(data)


# --------------------------------------------------------------------------- #
# manual segment editing + re-export                                          #
# --------------------------------------------------------------------------- #
class SegmentEdit(BaseModel):
    segments: list[list[float]]


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
    return segs


def _rewrite_sidecar(job: dict[str, Any], segs: list[tuple[float, float]]) -> dict[str, Any]:
    sidecar = _read_json(Path(job["json_path"]), {}) if job.get("json_path") else {}
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
    job = _load_job(job_id)
    if job.get("status") in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="job is still processing")
    result = job.get("result") or {}
    duration = float(result.get("total_seconds") or 0)
    segs = _normalise_segments(edit.segments, duration)

    sidecar = _rewrite_sidecar(job, segs)
    with _LOCK:
        job = _load_job(job_id)
        job["result"] = sidecar
        _save_job(job)

    if segs:
        cfg = _config_from_options(job.get("options", {}))
        out = _job_dir(job_id) / "output" / "rallies.mp4"
        info = probe(job["original_path"])
        ok = _render_output(Path(job["original_path"]), segs, out, cfg, info,
                            lambda m: _append_progress(job_id, m))
        with _LOCK:
            job = _load_job(job_id)
            job["output_path"] = str(out) if (ok and out.exists()) else None
            _save_job(job)
    return JSONResponse(_public_job(_load_job(job_id)))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    job = _load_job(job_id)
    if job.get("status") in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="cannot delete a processing job")
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# labelling: generate raw samples (player crops + serve clips) to annotate     #
# --------------------------------------------------------------------------- #
class LabelTaskRequest(BaseModel):
    kinds: list[str] = ["player_identity", "serve_motion"]
    max_items: int = 10
    match_type: str = "auto"          # auto | singles | doubles
    regenerate: bool = False


class LabelPayload(BaseModel):
    task_id: str
    kind: str
    values: dict[str, Any] = {}


class RosterUpdate(BaseModel):
    roster: list[dict[str, Any]]


def _labels_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "labels"


def _assets_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "label_assets"


def _yolo():
    """Lazily load one shared YOLO model (used only for label-crop boxes)."""
    global _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is None:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(os.environ.get("RALLY_WEB_YOLO", "yolov8n.pt"))
        return _YOLO_MODEL


def _detect_boxes(frame_bgr, conf: float = 0.3) -> list[tuple[float, float, float, float]]:
    """Person boxes as pixel (x0, y0, x1, y1). Unlike the core's foot-point
    detector we keep the full box because we need it to crop a player."""
    res = _yolo().predict(frame_bgr, conf=conf, classes=[0], verbose=False)
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
                           max_items: int, match_type_req: str, regenerate: bool):
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

    assets = _assets_dir(job_id)
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
                cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
                          max_items: int, regenerate: bool) -> list[dict[str, Any]]:
    """Cut short clips around each serve moment (rally start) to classify."""
    times = _serve_times(segments, duration, min(max_items, max(1, len(segments) or max_items)))
    assets = _assets_dir(job_id)
    tasks: list[dict[str, Any]] = []
    for i, start in enumerate(times[:max_items]):
        clip_start = max(0.0, start - 1.5)
        clip_end = min(duration, start + 3.5) if duration else start + 3.5
        rel = f"serve_{i:04d}.mp4"
        path = assets / rel
        if regenerate or not path.exists():
            _ffmpeg_clip(src, clip_start, clip_end, path)
        tasks.append({
            "id": f"serve_{i:04d}",
            "kind": "serve_motion",
            "title": f"Serve clip {i + 1}",
            "time_s": round(float(start), 3),
            "media_type": "video",
            "asset_url": f"/api/jobs/{job_id}/assets/{rel}",
        })
    return tasks


def _run_label_gen(job_id: str, req: LabelTaskRequest) -> None:
    with _LOCK:
        if job_id in _LABEL_ACTIVE:
            return
        _LABEL_ACTIVE.add(job_id)
    try:
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
            _set_labeling(job_id, detail="Detecting players and cropping")
            roster, match_type, player_tasks = _generate_player_tasks(
                job_id, src, segments, duration, req.max_items, req.match_type, req.regenerate)
            tasks.extend(player_tasks)

        if "serve_motion" in req.kinds:
            _set_labeling(job_id, detail="Cutting serve clips")
            tasks.extend(_generate_serve_tasks(
                job_id, src, segments, duration, req.max_items, req.regenerate))

        # persist roster (preserve any user-renamed names) + tasks
        roster_path = _labels_dir(job_id) / "roster.json"
        existing = _read_json(roster_path, None)
        if existing and not req.regenerate:
            names = {r["id"]: r.get("name") for r in existing if isinstance(r, dict)}
            for r in roster:
                if names.get(r["id"]):
                    r["name"] = names[r["id"]]
        _atomic_write_json(roster_path, roster)
        _atomic_write_json(_labels_dir(job_id) / "tasks.json", tasks)

        n_player = sum(1 for t in tasks if t["kind"] == "player_identity")
        n_serve = sum(1 for t in tasks if t["kind"] == "serve_motion")
        _set_labeling(job_id, status="ready", match_type=match_type,
                      detail=f"{n_player} player crops · {n_serve} serve clips",
                      counts={"player_identity": n_player, "serve_motion": n_serve})
    except Exception as exc:
        _set_labeling(job_id, status="failed", error=str(exc), detail=str(exc))
    finally:
        with _LOCK:
            _LABEL_ACTIVE.discard(job_id)


@app.post("/api/jobs/{job_id}/label-tasks")
def create_label_tasks(job_id: str, req: LabelTaskRequest) -> JSONResponse:
    job = _load_job(job_id)
    bad = set(req.kinds) - _LABEL_KINDS
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown label kind(s): {', '.join(sorted(bad))}")
    if not job.get("original_path") or not Path(job["original_path"]).exists():
        raise HTTPException(status_code=404, detail="original video missing")
    lab = job.get("labeling") or {}
    if lab.get("status") == "generating":
        return JSONResponse(_public_job(job))
    _set_labeling(job_id, status="generating", detail="Queued", error=None)
    _EXECUTOR.submit(_run_label_gen, job_id, req)
    return JSONResponse(_public_job(_load_job(job_id)))


@app.get("/api/jobs/{job_id}/label-tasks")
def get_label_tasks(job_id: str) -> JSONResponse:
    _load_job(job_id)
    tasks = _read_json(_labels_dir(job_id) / "tasks.json", [])
    roster = _read_json(_labels_dir(job_id) / "roster.json", [])
    labels = _read_json(_labels_dir(job_id) / "labels.json", {})
    return JSONResponse({"tasks": tasks, "roster": roster, "labels": labels})


@app.post("/api/jobs/{job_id}/roster")
def update_roster(job_id: str, update: RosterUpdate) -> JSONResponse:
    _load_job(job_id)
    clean = [{"id": str(r.get("id")), "name": str(r.get("name") or r.get("id")),
              "side": r.get("side"), "col": r.get("col")}
             for r in update.roster if r.get("id")]
    _atomic_write_json(_labels_dir(job_id) / "roster.json", clean)
    return JSONResponse({"roster": clean})


@app.get("/api/jobs/{job_id}/labels")
def get_labels(job_id: str) -> JSONResponse:
    _load_job(job_id)
    return JSONResponse({"labels": _read_json(_labels_dir(job_id) / "labels.json", {})})


@app.post("/api/jobs/{job_id}/labels")
def save_label(job_id: str, payload: LabelPayload) -> JSONResponse:
    _load_job(job_id)
    path = _labels_dir(job_id) / "labels.json"
    with _LOCK:
        labels = _read_json(path, {})
        labels[payload.task_id] = {"task_id": payload.task_id, "kind": payload.kind,
                                   "values": payload.values, "updated_at": _now()}
        _atomic_write_json(path, labels)
    return JSONResponse({"labels": labels, "saved": payload.task_id})


@app.get("/api/jobs/{job_id}/labels/download")
def download_labels(job_id: str) -> FileResponse:
    job = _load_job(job_id)
    path = _labels_dir(job_id) / "labels.json"
    if not path.exists():
        _atomic_write_json(path, {})
    # fold in the roster + task metadata so the export is self-describing
    export = {"job_id": job_id, "filename": job.get("filename"),
              "roster": _read_json(_labels_dir(job_id) / "roster.json", []),
              "tasks": _read_json(_labels_dir(job_id) / "tasks.json", []),
              "labels": _read_json(path, {})}
    export_path = _labels_dir(job_id) / "labels_export.json"
    _atomic_write_json(export_path, export)
    return FileResponse(export_path, media_type="application/json",
                        filename=f"{Path(job['filename']).stem}_labels.json",
                        content_disposition_type="attachment")


@app.get("/api/jobs/{job_id}/assets/{asset_path:path}")
def get_asset(job_id: str, asset_path: str) -> FileResponse:
    root = _assets_dir(job_id).resolve()
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
    parser.add_argument("--data-dir", default=None, help="runtime upload/output directory")
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
