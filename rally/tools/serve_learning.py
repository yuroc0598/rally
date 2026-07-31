"""Offline serve-learning dataset adapters.

It converts independent human labels plus recorded match diagnostics into a grouped,
versioned training dataset. Shared schemas and the deployment gate live in
``rally.learning.serve_schema`` so offline tools and runtime cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..config import RallyConfig
from ..learning.serve_schema import (
    ARTIFACT_SCHEMA,
    CONTEXT_SCHEMA,
    DATASET_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    assert_live_eligible,
    live_gate,
)
from ..learning.serve_features import (
    audio_features as schema_audio_features,
    empty_features,
    observation_features,
)

_DYNAMIC_RULE_SOURCES = {
    "overhead_pose_with_position_setup",
    "robust_target_court_overhead_pose",
    "target_court_receiver_reaction",
    "tracknet_ball_motion",
}


@dataclass(frozen=True)
class WebJobSpec:
    match_id: str
    video: str
    labels: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _source_identity(path: str) -> str:
    """Stable content identity when available, resolved-path identity for test adapters."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return f"path:{source}"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _local_job_context(path: Path) -> dict[str, Any]:
    """Recover feature context when the caller points at a live ``labels.json``."""
    for parent in (path.parent, *path.parents):
        job_path = parent / "job.json"
        if not job_path.exists():
            continue
        job = _read_json(job_path)
        result = job.get("result") or {}
        return {
            "job_id": job.get("id"),
            "filename": job.get("filename"),
            "feature_context": {
                "schema_version": CONTEXT_SCHEMA,
                "strike_times": result.get("strike_times", []),
                "segments": result.get("segments", []),
                "match_state": (result.get("stages") or {}).get("match_state", {}),
            },
        }
    return {}


def load_web_label_export(path: str | Path) -> dict[str, Any]:
    """Load either a downloaded ``labels_export`` or a revision's raw labels file."""
    source = Path(path)
    data = _read_json(source)
    if isinstance(data, dict) and isinstance(data.get("labels"), dict):
        export = dict(data)
    elif isinstance(data, dict):
        tasks_path = source.with_name("tasks.json")
        if not tasks_path.exists():
            raise ValueError(f"raw labels file has no sibling tasks.json: {source}")
        export = {
            "schema_version": "rally.web_labels.raw.v1",
            "tasks": _read_json(tasks_path),
            "labels": data,
            **_local_job_context(source),
        }
    else:
        raise ValueError(f"label export must be a JSON object: {source}")
    if not isinstance(export.get("tasks"), list) or not isinstance(export.get("labels"), dict):
        raise ValueError(f"label export is missing tasks/labels: {source}")
    return export


def _binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"yes", "1", "true", "serve"}:
        return 1
    if text in {"no", "0", "false", "non-serve", "nonserve"}:
        return 0
    return None


def labelled_serve_tasks(export: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = {str(task.get("id")): task for task in export["tasks"]
             if isinstance(task, dict) and task.get("kind") == "serve_motion"}
    rows: list[dict[str, Any]] = []
    for task_id, record in export["labels"].items():
        if task_id not in tasks or not isinstance(record, dict):
            continue
        if record.get("kind") not in {None, "serve_motion"}:
            continue
        values = record.get("values") or {}
        label = _binary_label(values.get("is_serve"))
        if label is None:
            continue
        task = tasks[task_id]
        time_s = float(task.get("time_s"))
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError(f"serve task {task_id} has invalid time_s")
        rows.append({
            "task_id": task_id,
            "time_s": time_s,
            "label": label,
            **({"source_segment_index": int(task["source_segment_index"])}
               if task.get("source_segment_index") is not None else {}),
            **({
                "logical_group": int(task["logical_group"]),
                "match_state_observation_index": int(
                    task["match_state_observation_index"]),
            } if task.get("logical_group") is not None
               and task.get("match_state_observation_index") is not None else {}),
            "annotation": {
                key: values[key] for key in (
                    "server", "side", "end", "outcome", "serve_type", "notes")
                if values.get(key) not in (None, "")
            },
        })
    return sorted(rows, key=lambda row: (row["time_s"], row["task_id"]))


def _empty_features() -> dict[str, float]:
    return empty_features()


def _context_observation(context: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    state = context.get("match_state") or {}
    observations = state.get("observations") or []
    groups = state.get("logical_groups") or []
    stable_index = row.get("match_state_observation_index")
    if stable_index is not None:
        index = int(stable_index)
        group_index = int(row.get("logical_group", -1))
        group = next((item for item in groups
                      if int(item.get("group_index", -2)) == group_index), None)
        if (group is None or index not in group.get("member_indices", [])
                or not 0 <= index < len(observations)):
            task_id = row.get("task_id")
            raise ValueError(f"serve task {task_id} has mismatched stable context ids")
        return observations[index]

    time_s = float(row["time_s"])
    candidates: list[tuple[float, int]] = []
    for group in groups:
        output = group.get("output") or []
        member = group.get("serve_member_index")
        if len(output) >= 1 and member is not None:
            candidates.append((abs(float(output[0]) - time_s), int(member)))
    if candidates:
        distance, index = min(candidates)
        if distance <= 2.0 and 0 <= index < len(observations):
            return observations[index]
    # Legacy/partial stage: nearest point start is less exact but still safely bounded.
    by_point = []
    for index, observation in enumerate(observations):
        point = observation.get("point") or []
        if point:
            by_point.append((abs(float(point[0]) - time_s), index))
    if by_point:
        distance, index = min(by_point)
        if distance <= 6.0:
            return observations[index]
    return None


def context_features(export: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, dict]:
    context = export.get("feature_context") or {}
    context_schema = context.get("schema_version")
    if context_schema not in {None, CONTEXT_SCHEMA}:
        raise ValueError(f"unsupported serve rule context schema: {context_schema!r}")
    result: dict[str, dict] = {}
    for row in rows:
        features = _empty_features()
        observation = _context_observation(context, row)
        rule_prediction = 0
        rule_available = False
        contact_time = None
        if observation is not None:
            features.update(observation_features(observation))
            sources = set(observation.get("serve_evidence_sources") or [])
            rule_prediction = int(bool(sources & _DYNAMIC_RULE_SOURCES))
            rule_available = bool(
                observation.get("observable") or observation.get("position_checked")
                or observation.get("ball_checked"))
            first_strike = observation.get("first_strike")
            if first_strike is not None and math.isfinite(float(first_strike)):
                contact_time = float(first_strike)
        result[row["task_id"]] = {
            "features": features,
            "rule_prediction": rule_prediction,
            "rule_available": rule_available,
            "contact_time_s": contact_time,
        }
    return result


def audio_features(video: str, rows: Sequence[dict[str, Any]],
                   context: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Extract one match's audio features once, then align them to labeled tasks."""
    from ..io.ffmpeg import iter_audio_mono
    from ..signals.audio import detect_strikes_stream

    cfg = RallyConfig()
    onsets = detect_strikes_stream(
        iter_audio_mono(video, cfg.audio_sr, chunk_s=60.0), cfg.audio_sr, cfg)
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        contact = context[row["task_id"]].get("contact_time_s")
        expected = float(contact if contact is not None
                         else row["time_s"] + cfg.toss_preroll_s)
        result[row["task_id"]] = schema_audio_features(
            np.asarray(onsets, dtype=float), expected, cfg.point_gap_s)
    return result


AudioExtractor = Callable[
    [str, Sequence[dict[str, Any]], dict[str, dict]], dict[str, dict[str, float]]]


def build_training_dataset(specs: Iterable[WebJobSpec], *,
                           audio_extractor: AudioExtractor = audio_features) -> dict[str, Any]:
    """Build a finite-feature, match-grouped dataset from independent web jobs."""
    samples: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_videos: set[str] = set()
    seen_exports: set[str] = set()
    all_stable_context = True
    for spec in specs:
        match_id = str(spec.match_id).strip()
        if not match_id or match_id in seen:
            raise ValueError(f"match ids must be non-empty and unique: {match_id!r}")
        seen.add(match_id)
        video_identity = _source_identity(spec.video)
        export_identity = _source_identity(spec.labels)
        if video_identity in seen_videos or export_identity in seen_exports:
            raise ValueError(
                f"match {match_id} reuses a video or label export from another group")
        seen_videos.add(video_identity)
        seen_exports.add(export_identity)
        export = load_web_label_export(spec.labels)
        rows = labelled_serve_tasks(export)
        if not rows:
            raise ValueError(f"match {match_id} has no labeled serve_motion tasks")
        context = context_features(export, rows)
        stable_context = all(
            row.get("logical_group") is not None
            and row.get("match_state_observation_index") is not None
            for row in rows
        )
        all_stable_context &= stable_context
        audio = audio_extractor(spec.video, rows, context)
        positive = 0
        for row in rows:
            task_id = row["task_id"]
            features = dict(context[task_id]["features"])
            features.update(audio.get(task_id, {}))
            clean_features = {}
            for name in FEATURE_NAMES:
                value = float(features.get(name, 0.0))
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {name} for {match_id}/{task_id}")
                clean_features[name] = value
            positive += row["label"]
            samples.append({
                "match_id": match_id,
                "task_id": task_id,
                "time_s": round(float(row["time_s"]), 3),
                **({"source_segment_index": row["source_segment_index"]}
                   if "source_segment_index" in row else {}),
                **({
                    "logical_group": row["logical_group"],
                    "match_state_observation_index": row[
                        "match_state_observation_index"],
                } if "logical_group" in row else {}),
                "label": int(row["label"]),
                "annotation": row["annotation"],
                "features": clean_features,
                "rule_prediction": int(context[task_id]["rule_prediction"]),
                "rule_available": bool(context[task_id]["rule_available"]),
            })
        matches.append({
            "match_id": match_id,
            "video": str(Path(spec.video)),
            "labels": str(Path(spec.labels)),
            "samples": len(rows),
            "positive": int(positive),
            "negative": int(len(rows) - positive),
            "source_job_id": export.get("job_id"),
            "video_identity": video_identity,
            "label_export_identity": export_identity,
            "context_alignment": "stable_ids" if stable_context else "legacy_nearest_time",
        })
    if not samples:
        raise ValueError("training dataset contains no labeled samples")
    return {
        "schema_version": DATASET_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "created_at": _now(),
        "stable_context_alignment": bool(all_stable_context),
        "matches": matches,
        "samples": samples,
    }


def dataset_fingerprint(dataset: dict[str, Any]) -> str:
    stable = {
        "schema_version": dataset.get("schema_version"),
        "feature_schema": dataset.get("feature_schema"),
        "feature_names": dataset.get("feature_names"),
        "stable_context_alignment": dataset.get("stable_context_alignment"),
        "matches": dataset.get("matches"),
        "samples": dataset.get("samples"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def held_out_match_splits(groups: Sequence[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-match-out splits; never put samples from one match on both sides."""
    values = np.asarray(groups, dtype=object)
    unique = list(dict.fromkeys(str(value) for value in values))
    if len(unique) < 2:
        raise ValueError("grouped validation needs at least two independent matches")
    return [
        (np.flatnonzero(values != group), np.flatnonzero(values == group))
        for group in unique
    ]
