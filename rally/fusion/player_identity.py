"""Persistent target-court player identities and singles/doubles inference.

This module consumes the one BoT-SORT player pass.  It does not run detection, pose,
audio, or ball inference.  Court occupancy determines match format; tracker IDs and
clothing descriptors reconnect sparse tracklets into stable, inspectable players.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..signals.court import COURT_L, DOUBLES_W, NET_Y

PLAYER_IDENTITY_SCHEMA = "rally.player_identity.v2"


@dataclass(frozen=True)
class _Observation:
    time: float
    track_id: int | None
    court_x: float
    court_y: float
    image_x: float
    image_y: float
    area: float
    bbox: tuple[float, float, float, float] | None
    appearance: tuple[float, ...] | None

    @property
    def end(self) -> str:
        return "near" if self.court_y < NET_Y else "far"


def _court_observations(
    samples: Sequence,
    court,
    frame_size: tuple[int, int] | None,
) -> list[tuple[float, list[_Observation]]]:
    if court is None or frame_size is None:
        return []
    width, height = (int(value) for value in frame_size)
    if width <= 0 or height <= 0:
        return []
    output: list[tuple[float, list[_Observation]]] = []
    for raw_time, raw_people in samples:
        time_s = float(raw_time)
        people = [item for item in raw_people if isinstance(item, dict)]
        if not people:
            output.append((time_s, []))
            continue
        feet = np.asarray([
            [float(item.get("foot_x_norm", np.nan)) * width,
             float(item.get("foot_y_norm", np.nan)) * height]
            for item in people
        ], dtype=float)
        try:
            coordinates = np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
        except Exception:
            output.append((time_s, []))
            continue
        records: list[_Observation] = []
        for item, coordinate in zip(people, coordinates, strict=True):
            x, y = float(coordinate[0]), float(coordinate[1])
            if not (
                np.isfinite((x, y)).all()
                and -1.5 <= x <= DOUBLES_W + 1.5
                and -3.0 <= y <= COURT_L + 3.0
            ):
                continue
            raw_track = item.get("track_id")
            try:
                track_id = int(raw_track) if raw_track is not None else None
            except (TypeError, ValueError):
                track_id = None
            raw_box = item.get("bbox_norm")
            bbox = (
                tuple(float(value) for value in raw_box)
                if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4
                else None
            )
            raw_appearance = item.get("appearance")
            appearance = (
                tuple(float(value) for value in raw_appearance)
                if isinstance(raw_appearance, (list, tuple)) and raw_appearance
                else None
            )
            image_x = float(item.get("foot_x_norm", np.nan))
            image_y = float(item.get("foot_y_norm", np.nan))
            area = float(item.get("box_area_norm", 0.0))
            if not np.isfinite((image_x, image_y, area)).all() or area <= 0:
                continue
            records.append(_Observation(
                time_s, track_id, x, y, image_x, image_y, area, bbox, appearance))
        output.append((time_s, records))
    return output


def infer_match_format(
    player_track_samples: Sequence,
    court,
    frame_size: tuple[int, int] | None,
    segments: Sequence[tuple[float, float]] = (),
) -> dict[str, Any]:
    """Infer singles/doubles from repeated two-sided target-court occupancy."""
    samples = _court_observations(player_track_samples, court, frame_size)
    formations: list[tuple[int, int]] = []
    for time_s, observations in samples:
        if segments and not any(start <= time_s <= end for start, end in segments):
            continue
        near = sum(item.end == "near" for item in observations)
        far = sum(item.end == "far" for item in observations)
        if near and far:
            formations.append((near, far))
    if not formations:
        return {
            "format": "unknown", "confidence": 0.0, "player_count": 0,
            "evidence_frames": 0,
            "reason": "no two-sided target-court player formations",
        }
    count = len(formations)
    two_by_two = sum(near >= 2 and far >= 2 for near, far in formations)
    three_visible = sum(near + far >= 3 for near, far in formations)
    one_by_one = sum(near == 1 and far == 1 for near, far in formations)
    strong_ratio = two_by_two / count
    partial_ratio = three_visible / count
    doubles = bool(
        two_by_two >= max(4, int(np.ceil(0.35 * count)))
        or (two_by_two >= 3 and partial_ratio >= 0.60)
    )
    if doubles:
        match_format, players = "doubles", 4
        confidence = np.clip(
            0.55 + 0.35 * partial_ratio + 0.10 * strong_ratio, 0.0, 0.99)
    else:
        match_format, players = "singles", 2
        confidence = np.clip(
            0.55 + 0.40 * one_by_one / count - 0.20 * partial_ratio, 0.0, 0.99)
    return {
        "format": match_format,
        "confidence": round(float(confidence), 4),
        "player_count": players,
        "evidence_frames": count,
        "two_by_two_frames": two_by_two,
        "three_player_frames": three_visible,
        "one_by_one_frames": one_by_one,
        "source": "persistent_target_court_player_occupancy",
    }


def _appearance_centre(records: Sequence[_Observation]) -> np.ndarray | None:
    values = [np.asarray(item.appearance, dtype=float)
              for item in records if item.appearance]
    if not values:
        return None
    centre = np.median(np.stack(values), axis=0)
    centre /= max(float(np.linalg.norm(centre)), 1e-9)
    return centre


def _appearance_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1e-9)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    return 1.0 - float(np.clip(np.dot(left, right), -1.0, 1.0))


def _end_profile(records: Sequence[_Observation]) -> tuple[str | None, float]:
    if not records:
        return None, 0.0
    near = sum(item.end == "near" for item in records)
    dominant = "near" if near >= len(records) - near else "far"
    return dominant, max(near, len(records) - near) / len(records)


def _anchor(
    samples: Sequence[tuple[float, list[_Observation]]],
    tracklets: dict[int, list[_Observation]],
    per_end: int,
) -> list[_Observation]:
    candidates: list[tuple[float, float, list[_Observation]]] = []
    for time_s, observations in samples:
        reliable = [
            item for item in observations if item.track_id is not None
            and _end_profile(tracklets.get(item.track_id, ()))[0] == item.end
            and _end_profile(tracklets.get(item.track_id, ()))[1] >= 0.80
        ]
        near = sorted(
            (item for item in reliable if item.end == "near"),
            key=lambda item: item.area, reverse=True)
        far = sorted(
            (item for item in reliable if item.end == "far"),
            key=lambda item: item.area, reverse=True)
        if len(near) >= per_end and len(far) >= per_end:
            selected = [*near[:per_end], *far[:per_end]]
            candidates.append((sum(item.area for item in selected), -time_s, selected))
    if candidates:
        selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
        near = sorted((item for item in selected if item.end == "near"),
                      key=lambda item: item.court_x)
        far = sorted((item for item in selected if item.end == "far"),
                     key=lambda item: item.court_x)
        return [*near, *far]
    return []


def _fallback_anchor(
    tracklets: dict[int, list[_Observation]], per_end: int,
) -> list[_Observation]:
    """Choose strong non-overlapping tracklets when no complete frame exists."""
    selected: list[_Observation] = []
    for end in ("near", "far"):
        ranked: list[tuple[int, float, int, _Observation]] = []
        for track_id, records in tracklets.items():
            on_end = [item for item in records if item.end == end]
            dominant, purity = _end_profile(records)
            if not on_end or dominant != end or purity < 0.80:
                continue
            representative = max(on_end, key=lambda item: item.area)
            ranked.append((len(on_end), representative.area, track_id, representative))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected.extend(item[3] for item in ranked[:per_end])
    near = sorted((item for item in selected if item.end == "near"),
                  key=lambda item: item.court_x)
    far = sorted((item for item in selected if item.end == "far"),
                 key=lambda item: item.court_x)
    return [*near, *far] if len(near) == per_end and len(far) == per_end else []


def _inspection_sources(records: Sequence[_Observation]) -> list[dict[str, Any]]:
    centre = _appearance_centre(records)
    retained: list[_Observation] = []
    for record in sorted(records, key=lambda item: item.area, reverse=True):
        if record.bbox is None:
            continue
        if centre is not None and record.appearance:
            if _appearance_distance(centre, np.asarray(record.appearance, dtype=float)) > 0.28:
                continue
        if any(abs(record.time - prior.time) < 0.75 for prior in retained):
            continue
        retained.append(record)
        if len(retained) >= 40:
            break
    retained.sort(key=lambda item: item.time)
    return [{
        "time_s": round(item.time, 3),
        "foot_x_norm": round(item.image_x, 6),
        "foot_y_norm": round(item.image_y, 6),
        "box_area_norm": round(item.area, 8),
        "bbox_norm": [round(value, 6) for value in item.bbox or ()],
    } for item in retained]


def identify_match_players(
    *, court,
    frame_size: tuple[int, int] | None,
    player_track_samples: Sequence,
    format_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map target-court tracklets to stable player/team IDs for pose association."""
    evidence = dict(format_evidence or {})
    if evidence.get("format") not in {"singles", "doubles"}:
        evidence = infer_match_format(
            player_track_samples, court, frame_size)
    match_format = str(evidence.get("format") or "unknown")
    if match_format not in {"singles", "doubles"}:
        return {
            "schema_version": PLAYER_IDENTITY_SCHEMA,
            "format": "unknown",
            "format_confidence": float(evidence.get("confidence") or 0.0),
            "format_evidence": evidence,
            "roster": [], "teams": [], "identity_method": "unresolved",
        }

    samples = _court_observations(player_track_samples, court, frame_size)
    tracklets: dict[int, list[_Observation]] = defaultdict(list)
    for _time, observations in samples:
        for observation in observations:
            if observation.track_id is not None:
                tracklets[observation.track_id].append(observation)
    per_end = 2 if match_format == "doubles" else 1
    anchor = _anchor(samples, tracklets, per_end) or _fallback_anchor(tracklets, per_end)
    if len(anchor) != 2 * per_end:
        return {
            "schema_version": PLAYER_IDENTITY_SCHEMA,
            "format": match_format,
            "format_confidence": float(evidence.get("confidence") or 0.0),
            "format_evidence": evidence,
            "roster": [], "teams": [], "identity_method": "track_identity_unresolved",
        }

    mapping: dict[int, int] = {}
    identity_records: list[list[_Observation]] = [[] for _item in anchor]
    centres: list[np.ndarray | None] = []
    occupied_times: list[set[float]] = []
    for index, observation in enumerate(anchor):
        assert observation.track_id is not None
        track_id = observation.track_id
        mapping[track_id] = index
        records = [
            item for item in tracklets.get(track_id, [observation])
            if item.end == observation.end
        ]
        identity_records[index].extend(records)
        centres.append(_appearance_centre(records))
        occupied_times.append({round(item.time, 3) for item in records})

    for track_id, records in sorted(
        tracklets.items(), key=lambda item: len(item[1]), reverse=True
    ):
        if track_id in mapping:
            continue
        centre = _appearance_centre(records)
        if centre is None:
            continue
        dominant_end, end_purity = _end_profile(records)
        if end_purity < 0.80:
            continue
        times = {round(item.time, 3) for item in records}
        scores = [
            (_appearance_distance(identity_centre, centre), index)
            for index, identity_centre in enumerate(centres)
            if identity_centre is not None
            and dominant_end == anchor[index].end
            and not (times & occupied_times[index])
        ]
        if not scores:
            continue
        distance, identity_index = min(scores)
        if distance > 0.22:
            continue
        mapping[track_id] = identity_index
        identity_records[identity_index].extend(
            item for item in records if item.end == anchor[identity_index].end)
        occupied_times[identity_index].update(times)
        centres[identity_index] = _appearance_centre(identity_records[identity_index])

    roster: list[dict[str, Any]] = []
    for index, (seed, records) in enumerate(zip(anchor, identity_records, strict=True), 1):
        player_id = f"P{index}"
        team_id = "T1" if seed.end == "near" else "T2"
        sources = _inspection_sources(records)
        thumbnails = sorted(
            sources, key=lambda item: float(item.get("box_area_norm", 0.0)),
            reverse=True)[:3]
        roster.append({
            "id": player_id,
            "name": f"Player {index}",
            "team_id": team_id,
            "initial_end": seed.end,
            "source_track_ids": sorted(
                track_id for track_id, mapped in mapping.items() if mapped == index - 1),
            "thumbnail_sources": thumbnails,
            "inspection_sources": sources,
        })
    teams = [{
        "id": team_id,
        "player_ids": [item["id"] for item in roster if item["team_id"] == team_id],
    } for team_id in ("T1", "T2")]
    # Exact clothing re-identification is intentionally conservative, especially for
    # tiny far-court players.  Keep every spatially stable target-court tracklet available
    # to pose analysis as an explicitly named fragment instead of either discarding it or
    # pretending that it belongs to a roster player. Tennis exchange order is decoded
    # from the fragment's measured court side; only confidently mapped tracks inherit a
    # persistent player/team identity.
    track_assignments: list[dict[str, Any]] = []
    for track_id, records in sorted(tracklets.items()):
        dominant_end, end_purity = _end_profile(records)
        if dominant_end is None or end_purity < 0.80:
            continue
        identity_index = mapping.get(track_id)
        player = roster[identity_index] if identity_index is not None else None
        track_assignments.append({
            "track_id": track_id,
            "actor_id": player["id"] if player is not None else f"track_{track_id}",
            "player_id": player["id"] if player is not None else None,
            "team_id": player["team_id"] if player is not None else None,
            "dominant_court_end": dominant_end,
            "end_purity": round(float(end_purity), 4),
            "identity_status": (
                "persistent_player" if player is not None else "unresolved_track_fragment"),
        })
    return {
        "schema_version": PLAYER_IDENTITY_SCHEMA,
        "format": match_format,
        "format_confidence": float(evidence.get("confidence") or 0.0),
        "format_evidence": evidence,
        "roster": roster,
        "teams": teams,
        "track_assignments": track_assignments,
        "identity_method": "end_consistent_botsort_tracklets_plus_clothing",
    }
