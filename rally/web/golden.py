"""Discovery and path resolution for retained golden-evaluation artifacts.

This module deliberately knows nothing about FastAPI or uploaded jobs.  Keeping the
read-only evaluation gallery separate prevents its filesystem rules from being mixed with
the mutable job store in :mod:`rally.web.app`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _media_url(dataset_id: str, kind: str, path: Path) -> str:
    return (
        f"/api/golden/{quote(dataset_id, safe='')}/media/{kind}"
        f"?v={path.stat().st_mtime_ns}"
    )


def discover_datasets(
    golden_dir: Path,
    results_dir: Path,
    *,
    video_extensions: Iterable[str],
) -> list[dict[str, Any]]:
    """Return labeled root-level input/result pairs and retained run metadata.

    Discovery is non-recursive by design: nested ``unlabeled`` and scratch directories
    never become public gallery entries.
    """
    if not golden_dir.is_dir():
        return []
    extensions = {str(extension).lower() for extension in video_extensions}
    datasets: list[dict[str, Any]] = []
    for source in sorted(golden_dir.glob("input_*")):
        if not source.is_file() or source.suffix.lower() not in extensions:
            continue
        suffix = source.stem.removeprefix("input_")
        annotation = golden_dir / f"res_{suffix}.txt"
        if not annotation.is_file():
            continue
        text = annotation.read_text(encoding="utf-8", errors="replace")
        expected_points = len(re.findall(
            r'["\']?point_index["\']?\s*:', text, re.I))
        if not expected_points:
            expected_points = len(re.findall(
                r"^\s*Point\s+\d+\s*:", text, re.I | re.M))
        dataset_id = source.stem
        result_dir = results_dir / dataset_id
        sidecar_path = result_dir / "rallies.json"
        output_path = result_dir / "rallies.mp4"
        sidecar = _read_json(sidecar_path)
        datasets.append({
            "id": dataset_id,
            "name": source.name,
            "annotation": annotation.name,
            "expected_points": expected_points,
            "status": (
                "ready" if output_path.is_file()
                else "analysis_only" if sidecar_path.is_file()
                else "not_run"
            ),
            "predicted_points": (
                int(sidecar.get("n_rallies", 0))
                if isinstance(sidecar, dict) else None
            ),
            "duration": (
                float(sidecar.get("total_seconds", 0.0))
                if isinstance(sidecar, dict) else None
            ),
            "media": {
                "input": _media_url(dataset_id, "input", source),
                "output": (
                    _media_url(dataset_id, "output", output_path)
                    if output_path.is_file() else None
                ),
                "metadata": (
                    _media_url(dataset_id, "metadata", sidecar_path)
                    if sidecar_path.is_file() else None
                ),
                "ground_truth": _media_url(
                    dataset_id, "ground-truth", annotation),
            },
        })
    return datasets


def resolve_media_path(
    dataset_id: str,
    kind: str,
    golden_dir: Path,
    results_dir: Path,
    *,
    video_extensions: Iterable[str],
) -> Path | None:
    """Resolve one gallery asset after proving that its labeled dataset exists."""
    datasets = discover_datasets(
        golden_dir, results_dir, video_extensions=video_extensions)
    if not any(dataset["id"] == dataset_id for dataset in datasets):
        return None
    suffix = dataset_id.removeprefix("input_")
    source_candidates = [
        path for path in golden_dir.glob(f"input_{suffix}.*")
        if path.is_file() and path.suffix.lower()
        in {str(extension).lower() for extension in video_extensions}
    ]
    paths = {
        "input": source_candidates[0] if len(source_candidates) == 1 else None,
        "ground-truth": golden_dir / f"res_{suffix}.txt",
        "output": results_dir / dataset_id / "rallies.mp4",
        "metadata": results_dir / dataset_id / "rallies.json",
    }
    path = paths.get(kind)
    return path if path is not None and path.is_file() else None
