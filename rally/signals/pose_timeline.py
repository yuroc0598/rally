"""Shared all-player pose timeline and pose-only tennis point decoder.

The module deliberately has no audio or ball input.  It reuses the target-player boxes
and identities from the visual pass, measures every visible match player on both court
ends, detects service/stroke sequences, and decodes point boundaries from sustained
changes in player behaviour.  A missed serve pose cannot prevent exchange discovery.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .court import COURT_L, DOUBLES_W, NET_Y
from .pose_actions import (
    _action_candidates,
    _actor_for_box,
    _pose_record,
    _racket_evidence,
    _stroke_episodes,
    _track_id_to_player,
)

Segment = tuple[float, float]


@dataclass
class PoseTimeline:
    records_by_actor: dict[str, list[dict[str, Any]]]
    frames: list[dict[str, Any]]
    coarse_frames: int
    refined_frames: int
    sampled_boxes: int


def _interpolated_people(
    time_s: float,
    times: Sequence[float],
    samples: Sequence,
    max_gap_s: float,
) -> list[dict[str, Any]]:
    if not times:
        return []
    index = bisect_left(times, time_s)
    choices = [value for value in (index - 1, index) if 0 <= value < len(times)]
    if not choices:
        return []
    nearest = min(choices, key=lambda value: abs(times[value] - time_s))
    if abs(times[nearest] - time_s) > max_gap_s:
        return []
    if abs(times[nearest] - time_s) <= 1e-6:
        return [dict(item) for item in samples[nearest][1] if isinstance(item, dict)]
    if index <= 0 or index >= len(times) or times[index] == times[index - 1]:
        return [dict(item) for item in samples[nearest][1] if isinstance(item, dict)]
    left, right = index - 1, index
    if time_s - times[left] > max_gap_s or times[right] - time_s > max_gap_s:
        return [dict(item) for item in samples[nearest][1] if isinstance(item, dict)]
    left_people = {
        int(item["track_id"]): item
        for item in samples[left][1]
        if isinstance(item, dict) and item.get("track_id") is not None
    }
    right_people = {
        int(item["track_id"]): item
        for item in samples[right][1]
        if isinstance(item, dict) and item.get("track_id") is not None
    }
    fraction = float((time_s - times[left]) / (times[right] - times[left]))
    output: list[dict[str, Any]] = []
    for track_id in sorted(set(left_people) & set(right_people)):
        before, after = left_people[track_id], right_people[track_id]
        before_box, after_box = before.get("bbox_norm"), after.get("bbox_norm")
        if not (
            isinstance(before_box, (list, tuple))
            and len(before_box) == 4
            and isinstance(after_box, (list, tuple))
            and len(after_box) == 4
        ):
            continue
        item = dict(before if fraction <= 0.5 else after)
        item["bbox_norm"] = [
            (1.0 - fraction) * float(a) + fraction * float(b)
            for a, b in zip(before_box, after_box, strict=True)
        ]
        for name in ("foot_x_norm", "foot_y_norm", "box_area_norm", "confidence"):
            if before.get(name) is not None and after.get(name) is not None:
                item[name] = (1.0 - fraction) * float(before[name]) + fraction * float(after[name])
        output.append(item)
    if output:
        matched = set(left_people) & set(right_people)
        output.extend(
            dict(item)
            for item in samples[nearest][1]
            if isinstance(item, dict)
            and item.get("track_id") is not None
            and int(item["track_id"]) not in matched
        )
        return output
    return [dict(item) for item in samples[nearest][1] if isinstance(item, dict)]


def _boxes_for_people(
    people: Sequence[dict[str, Any]], frame_shape: tuple[int, int, int]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = frame_shape[:2]
    boxes: list[list[float]] = []
    retained: list[dict[str, Any]] = []
    for person in people:
        raw = person.get("bbox_norm")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in raw)
        if not np.isfinite((x0, y0, x1, y1)).all():
            continue
        # Sparse tracker boxes are at most half a detector interval away.  A small
        # expansion tolerates that interpolation error without admitting another court.
        box_width, box_height = x1 - x0, y1 - y0
        x0 = max(0.0, x0 - 0.12 * box_width)
        x1 = min(1.0, x1 + 0.12 * box_width)
        y0 = max(0.0, y0 - 0.07 * box_height)
        y1 = min(1.0, y1 + 0.06 * box_height)
        pixels = [x0 * width, y0 * height, x1 * width, y1 * height]
        if pixels[2] - pixels[0] < 4 or pixels[3] - pixels[1] < 8:
            continue
        boxes.append(pixels)
        retained.append(person)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), retained


def _read_pose_samples(
    video: str,
    sample_times: Sequence[float],
    *,
    pose_runtime,
    court,
    player_track_samples: Sequence,
    match: dict[str, Any],
    cfg,
    cancel_check: Callable[[], None],
    progress_callback: Callable[[int, int], None] | None,
) -> list[dict[str, Any]]:
    """Decode ordered times and infer poses in tracker-owned boxes."""
    import cv2

    ordered_samples = sorted(player_track_samples, key=lambda item: float(item[0]))
    track_times = [float(item[0]) for item in ordered_samples]
    raw_to_player = _track_id_to_player(match)
    actor_to_team = {
        str(item["id"]): str(item["team_id"])
        for item in match.get("roster") or []
        if isinstance(item, dict) and item.get("id") and item.get("team_id")
    }
    actor_to_team.update(
        {
            str(item["actor_id"]): str(item["team_id"])
            for item in match.get("track_assignments") or []
            if isinstance(item, dict) and item.get("actor_id") and item.get("team_id")
        }
    )
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    times = np.unique(np.round(np.asarray(sample_times, dtype=float), 6))
    target_frames = np.maximum(0, np.round(times * native_fps).astype(np.int64))
    output: list[dict[str, Any]] = []
    next_frame = int(target_frames[0]) if target_frames.size else 0
    if target_frames.size:
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
    try:
        for chunk_start in range(0, len(times), 48):
            cancel_check()
            chunk_times = times[chunk_start : chunk_start + 48]
            chunk_targets = target_frames[chunk_start : chunk_start + 48]
            frames: list[np.ndarray] = []
            actual_times: list[float] = []
            people_by_frame: list[list[dict[str, Any]]] = []
            boxes_by_frame: list[np.ndarray] = []
            for sample_time, target in zip(chunk_times, chunk_targets, strict=True):
                target_frame = int(target)
                if target_frame < next_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    next_frame = target_frame
                ok = True
                while next_frame <= target_frame:
                    ok = cap.grab()
                    if not ok:
                        break
                    next_frame += 1
                if not ok:
                    break
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                people = _interpolated_people(
                    float(sample_time),
                    track_times,
                    ordered_samples,
                    float(cfg.pose_track_box_max_gap_s),
                )
                boxes, retained = _boxes_for_people(people, frame.shape)
                frames.append(frame)
                actual_times.append(float(sample_time))
                people_by_frame.append(retained)
                boxes_by_frame.append(boxes)
            results = pose_runtime.predict_boxes(
                frames, boxes_by_frame, batch_size=min(32, max(1, len(frames)))
            )
            if len(results) != len(frames):
                raise RuntimeError("pose runtime returned a misaligned timeline batch")
            for sample_time, frame, result, tracked in zip(
                actual_times, frames, results, people_by_frame, strict=True
            ):
                if len(result.boxes) != len(tracked):
                    raise RuntimeError("tracker boxes and pose results are not aligned")
                if len(result.boxes):
                    height, width = frame.shape[:2]
                    feet = np.asarray(
                        [
                            [
                                float(person.get("foot_x_norm", (box[0] + box[2]) / (2.0 * width)))
                                * width,
                                float(person.get("foot_y_norm", box[3] / height)) * height,
                            ]
                            for box, person in zip(result.boxes, tracked, strict=True)
                        ],
                        dtype=float,
                    )
                    court_coords = np.asarray(court.to_court(feet), dtype=float).reshape(-1, 2)
                else:
                    court_coords = np.empty((0, 2), dtype=float)
                height, width = frame.shape[:2]
                for box, pose, confidence, coordinate, person in zip(
                    result.boxes,
                    result.keypoints,
                    result.confidence,
                    court_coords,
                    tracked,
                    strict=True,
                ):
                    if not np.isfinite(coordinate).all():
                        continue
                    box_norm = [
                        float(box[0] / width),
                        float(box[1] / height),
                        float(box[2] / width),
                        float(box[3] / height),
                    ]
                    actor_id, raw_track, association_iou = _actor_for_box(
                        box_norm, [person], raw_to_player
                    )
                    if actor_id is None:
                        continue
                    record = _pose_record(
                        sample_time,
                        pose,
                        confidence,
                        box,
                        (height, width),
                        coordinate,
                        actor_id,
                        raw_track,
                        association_iou,
                        cfg,
                    )
                    if record is None:
                        continue
                    record["team_id"] = actor_to_team.get(actor_id)
                    record.update(_racket_evidence(record, person))
                    output.append(record)
            if progress_callback is not None:
                progress_callback(min(chunk_start + 48, len(times)), len(times))
    finally:
        cap.release()
    return output


def _enrich_kinematics(
    records_by_actor: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for actor, raw_records in records_by_actor.items():
        records = [dict(item) for item in sorted(raw_records, key=lambda item: float(item["time"]))]
        for index, current in enumerate(records):
            court_speeds: list[float] = []
            wrist_speeds: list[float] = []
            wrist_speeds_by_hand: dict[str, list[float]] = defaultdict(list)
            for left, right in ((index - 1, index), (index, index + 1)):
                if left < 0 or right >= len(records):
                    continue
                dt = float(records[right]["time"]) - float(records[left]["time"])
                if not 0.0 < dt <= 0.55:
                    continue
                court_speeds.append(
                    float(
                        np.hypot(
                            float(records[right]["court_x_m"]) - float(records[left]["court_x_m"]),
                            float(records[right]["court_y_m"]) - float(records[left]["court_y_m"]),
                        )
                        / dt
                    )
                )
                for hand in ("left", "right"):
                    a = records[left]["wrists_body"].get(hand)
                    b = records[right]["wrists_body"].get(hand)
                    if a is not None and b is not None:
                        speed = float(np.linalg.norm(np.asarray(b) - np.asarray(a)) / dt)
                        wrist_speeds.append(speed)
                        wrist_speeds_by_hand[hand].append(speed)
            current["court_speed_m_s"] = max(court_speeds, default=0.0)
            current["wrist_speed_body_s"] = max(wrist_speeds, default=0.0)
            current["wrist_speed_by_hand_body_s"] = {
                hand: max(values) for hand, values in wrist_speeds_by_hand.items()
            }
        output[actor] = records
    return output


def _merge_actor_records(
    coarse: Sequence[dict[str, Any]], refined: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in (*coarse, *refined):
        grouped[str(record["actor_id"])].append(dict(record))
    output: dict[str, list[dict[str, Any]]] = {}
    for actor, records in grouped.items():
        merged: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda item: float(item["time"])):
            if merged and float(record["time"]) - float(merged[-1]["time"]) < 0.04:
                # Refined records are appended after coarse records and therefore replace
                # an almost-identical coarse measurement.
                merged[-1] = record
            else:
                merged.append(record)
        output[actor] = merged
    return output


def _merge_windows(
    centres: Sequence[float], duration: float, pre: float, post: float
) -> list[Segment]:
    windows: list[list[float]] = []
    for centre in sorted({round(float(value), 3) for value in centres}):
        start, end = max(0.0, centre - pre), min(duration, centre + post)
        if not windows or start > windows[-1][1] + 0.05:
            windows.append([start, end])
        else:
            windows[-1][1] = max(windows[-1][1], end)
    return [(float(start), float(end)) for start, end in windows]


def _timeline_frames(
    coarse_times: np.ndarray,
    records_by_actor: dict[str, list[dict[str, Any]]],
    cfg,
) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    fps = float(cfg.pose_timeline_fps)
    for records in records_by_actor.values():
        for record in records:
            index = round(float(record["time"]) * fps)
            if 0 <= index < len(coarse_times) and (
                abs(float(record["time"]) - float(coarse_times[index])) <= 0.55 / fps
            ):
                buckets[index].append(record)
    frames: list[dict[str, Any]] = []
    for index, time_s in enumerate(coarse_times):
        records = buckets.get(index, [])
        by_actor = {str(record["actor_id"]): record for record in records}
        visible = list(by_actor.values())
        ready = sum(bool(record["ready_stance"]) for record in visible)
        wrist_speeds = [float(record["wrist_speed_body_s"]) for record in visible]
        court_speeds = [float(record["court_speed_m_s"]) for record in visible]
        activity_wrist_speed = float(np.quantile(wrist_speeds, 0.75)) if wrist_speeds else 0.0
        activity_court_speed = float(np.quantile(court_speeds, 0.75)) if court_speeds else 0.0
        median_wrist_speed = float(np.median(wrist_speeds)) if wrist_speeds else 0.0
        ready_fraction = ready / len(visible) if visible else 0.0
        ends = sorted({str(record["actor_end"]) for record in visible})
        teams = sorted({str(record["team_id"]) for record in visible if record.get("team_id")})
        relaxed_sample = bool(
            len(visible) >= int(cfg.pose_point_min_visible_players)
            and set(ends) == {"near", "far"}
            and ready_fraction <= float(cfg.pose_between_max_ready_fraction)
            and median_wrist_speed <= float(cfg.pose_between_max_median_wrist_speed_body_s)
        )
        frames.append(
            {
                "time": round(float(time_s), 3),
                "visible_players": len(visible),
                "ready_players": ready,
                "ready_fraction": round(float(ready_fraction), 4),
                "max_wrist_speed_body_s": round(max(wrist_speeds, default=0.0), 4),
                "max_court_speed_m_s": round(max(court_speeds, default=0.0), 4),
                "activity_wrist_speed_body_s": round(activity_wrist_speed, 4),
                "activity_court_speed_m_s": round(activity_court_speed, 4),
                "median_wrist_speed_body_s": round(median_wrist_speed, 4),
                "ends": ends,
                "teams": teams,
                # These are raw per-player observations, not a frame-local point-state
                # decision.  Fusion retains actor identity while aggregating overlapping
                # two-second windows.
                "players": [
                    {
                        "actor_id": str(record["actor_id"]),
                        "team_id": record.get("team_id"),
                        "end": str(record["actor_end"]),
                        "ready": bool(record["ready_stance"]),
                        "wrist_speed_body_s": round(float(record["wrist_speed_body_s"]), 4),
                        "court_speed_m_s": round(float(record["court_speed_m_s"]), 4),
                        "wrist_overhead_ratio": round(float(record["wrist_overhead_ratio"]), 4),
                        "racket_wrist_associated": bool(record.get("racket_wrist_associated")),
                    }
                    for record in sorted(visible, key=lambda item: str(item["actor_id"]))
                ],
                "relaxed_sample": relaxed_sample,
            }
        )
    return frames


def build_pose_timeline(
    video: str,
    duration: float,
    cfg,
    *,
    pose_runtime,
    court,
    player_track_samples: Sequence,
    match: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_check: Callable[[], None] = lambda: None,
) -> PoseTimeline:
    """Build a shared coarse timeline and refine observed arm-motion intervals."""
    match = match or {}
    coarse_times = np.arange(0.0, duration, 1.0 / float(cfg.pose_timeline_fps))

    def coarse_progress(done: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(f"pose timeline progress {done}/{total}")

    coarse = _read_pose_samples(
        video,
        coarse_times,
        pose_runtime=pose_runtime,
        court=court,
        player_track_samples=player_track_samples,
        match=match,
        cfg=cfg,
        cancel_check=cancel_check,
        progress_callback=coarse_progress,
    )
    grouped = _enrich_kinematics(_merge_actor_records(coarse, ()))
    # Refine only complete coarse stroke-shape proposals and baseline-overhead service
    # motion. Raw instantaneous wrist speed is dominated by pose jitter at track-fragment
    # boundaries and previously expanded this expensive pass over almost the whole video.
    refinement_centres = [
        float(candidate["time"])
        for records in grouped.values()
        for candidate in _action_candidates(records, (), cfg)
    ]
    refinement_centres.extend(
        float(record["time"])
        for records in grouped.values()
        for record in records
        if (
            (
                float(record["court_y_m"]) <= cfg.serve_baseline_y_m
                or float(record["court_y_m"]) >= COURT_L - cfg.serve_baseline_y_m
            )
            and float(record["wrist_overhead_ratio"]) >= 0.60 * float(cfg.pose_serve_overhead_ratio)
        )
    )
    # Recover coarse near-misses without returning to whole-video refinement. Keep only
    # temporally supported local speed maxima, globally de-duplicate simultaneous players,
    # and enforce an explicit per-minute inference budget.
    motion_peaks: list[tuple[float, float]] = []
    for records in grouped.values():
        ordered = sorted(records, key=lambda item: float(item["time"]))
        for index in range(2, len(ordered) - 2):
            if float(ordered[index + 2]["time"]) - float(ordered[index - 2]["time"]) > 1.0:
                continue
            speed = float(ordered[index]["wrist_speed_body_s"])
            if speed < 0.75 * float(cfg.pose_action_min_speed_body_s):
                continue
            local = ordered[index - 2 : index + 3]
            if speed < max(float(item["wrist_speed_body_s"]) for item in local):
                continue
            motion_peaks.append((speed, float(ordered[index]["time"])))
    motion_budget = int(np.ceil(duration / 60.0 * float(cfg.pose_refine_motion_windows_per_minute)))
    selected_motion_times: list[float] = []
    for _score, time_s in sorted(motion_peaks, reverse=True):
        if any(abs(time_s - prior) < 0.55 for prior in selected_motion_times):
            continue
        selected_motion_times.append(time_s)
        if len(selected_motion_times) >= motion_budget:
            break
    refinement_centres.extend(selected_motion_times)
    windows = _merge_windows(
        refinement_centres,
        duration,
        float(cfg.pose_refine_pre_s),
        float(cfg.pose_refine_post_s),
    )
    refine_times = (
        np.unique(
            np.concatenate(
                [
                    np.arange(
                        start,
                        end + 0.2594 / float(cfg.pose_refine_fps),
                        1.0 / float(cfg.pose_refine_fps),
                    )
                    for start, end in windows
                ]
            )
        )
        if windows
        else np.zeros(0, dtype=float)
    )

    def refine_progress(done: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(f"pose refinement progress {done}/{total}")

    refined = (
        _read_pose_samples(
            video,
            refine_times,
            pose_runtime=pose_runtime,
            court=court,
            player_track_samples=player_track_samples,
            match=match,
            cfg=cfg,
            cancel_check=cancel_check,
            progress_callback=refine_progress,
        )
        if refine_times.size
        else []
    )
    combined = _enrich_kinematics(_merge_actor_records(coarse, refined))
    frames = _timeline_frames(coarse_times, combined, cfg)
    return PoseTimeline(
        records_by_actor=combined,
        frames=frames,
        coarse_frames=len(coarse_times),
        refined_frames=len(refine_times),
        sampled_boxes=sum(len(records) for records in combined.values()),
    )


def _raised_hand(
    record: dict[str, Any],
    hand: str | None = None,
) -> tuple[list[float] | None, float]:
    keypoints = record.get("keypoints_norm") or []
    confidence = record.get("keypoint_confidence") or []
    indices = {"left": (9,), "right": (10,)}.get(hand, (9, 10))
    choices = [
        (keypoints[index], float(confidence[index]))
        for index in indices
        if index < len(keypoints) and index < len(confidence) and float(confidence[index]) > 0.2
    ]
    if not choices:
        return None, 0.0
    return min(choices, key=lambda item: float(item[0][1]))


def _serve_setup_start(
    records: Sequence[dict[str, Any]],
    peak: dict[str, Any],
    hand: str,
    cfg,
) -> tuple[float, dict[str, Any]]:
    """Return the observed wrist-load onset, not an entire baseline waiting period."""
    strike = float(peak["time"])
    peak_x = float(peak["court_x_m"])
    peak_end = str(peak["actor_end"])
    lookback = float(cfg.pose_serve_setup_fallback_s)
    max_gap = float(cfg.pose_serve_setup_max_gap_s)
    local_span = float(cfg.pose_serve_setup_span_m)
    eligible = sorted(
        (record for record in records if strike - lookback <= float(record["time"]) <= strike),
        key=lambda item: float(item["time"]),
        reverse=True,
    )
    run: list[dict[str, Any]] = []
    later = strike
    for record in eligible:
        time_s = float(record["time"])
        if later - time_s > max_gap:
            break
        court_y = float(record["court_y_m"])
        same_end = str(record.get("actor_end")) == peak_end
        at_baseline = bool(
            court_y <= cfg.serve_baseline_y_m or court_y >= COURT_L - cfg.serve_baseline_y_m
        )
        localized = abs(float(record["court_x_m"]) - peak_x) <= local_span
        if not (same_end and at_baseline and localized):
            break
        run.append(record)
        later = time_s
    ordered = sorted(run, key=lambda item: float(item["time"]))
    hand_records = [
        record for record in ordered if hand in record.get("wrist_overhead_by_hand", {})
    ]
    if len(hand_records) >= 3:
        # The last local low point before the overhead peak is the racket-arm loading
        # phase.  Starting there includes the service motion while excluding an
        # arbitrarily long stationary wait behind the baseline.
        overhead_values = [float(record["wrist_overhead_by_hand"][hand]) for record in hand_records]
        minimum_overhead = min(overhead_values)
        trough_index = max(
            index for index, value in enumerate(overhead_values) if value <= minimum_overhead + 0.03
        )
        trough = hand_records[trough_index]
        start = float(trough["time"])
        earlier_load = [
            record
            for record in hand_records
            if start - 0.65 <= float(record["time"]) <= start
            and record.get("minimum_knee_deg") is not None
            and float(record["minimum_knee_deg"]) <= float(cfg.pose_serve_knee_bend_deg)
        ]
        if earlier_load:
            start = min(float(record["time"]) for record in earlier_load)
        return start, {
            "source": "observed_service_arm_load_onset",
            "setup_start": round(start, 3),
            "setup_frames": len(hand_records),
            "setup_duration_s": round(strike - start, 3),
            "serve_hand": hand,
        }
    start = max(0.0, strike - float(cfg.pose_serve_setup_fallback_s))
    return start, {
        "source": "bounded_setup_fallback",
        "setup_start": round(start, 3),
        "setup_frames": len(run),
        "setup_duration_s": round(strike - start, 3),
    }


def detect_serves(timeline: PoseTimeline, cfg) -> list[dict[str, Any]]:
    """Detect one-hand, ordered service motion for every baseline player.

    A high wrist in one frame is not a serve.  The same tracked arm must rise from a
    loaded baseline setup, accelerate overhead, and continue into a follow-through.
    Knee extension or forward recovery corroborates the drive phase; formation and an
    associated racket improve confidence but are not mandatory because the far-side
    receiver and small racket are often below detector resolution.
    """
    candidates: list[dict[str, Any]] = []
    for actor, records in timeline.records_by_actor.items():
        for peak in records:
            time_s = float(peak["time"])
            court_y = float(peak["court_y_m"])
            at_baseline = bool(
                court_y <= cfg.serve_baseline_y_m or court_y >= COURT_L - cfg.serve_baseline_y_m
            )
            if not at_baseline:
                continue
            nearby_peak = [
                record for record in records if abs(float(record["time"]) - time_s) <= 0.30
            ]
            pre = [
                record
                for record in records
                if time_s - 1.35 <= float(record["time"]) <= time_s - 0.10
            ]
            post = [
                record
                for record in records
                if time_s + 0.08 <= float(record["time"]) <= time_s + 0.70
            ]
            if len(pre) < 2:
                continue
            load_frames = sum(bool(record["ready_stance"]) for record in pre)
            knee_bend_frames = sum(
                record.get("minimum_knee_deg") is not None
                and float(record["minimum_knee_deg"]) <= cfg.pose_serve_knee_bend_deg
                for record in pre
            )
            knee_values = [
                float(record["minimum_knee_deg"])
                for record in (*pre, peak, *post)
                if record.get("minimum_knee_deg") is not None
            ]
            leg_drive = max(knee_values) - min(knee_values) if len(knee_values) >= 2 else 0.0
            xs = np.asarray([float(record["court_x_m"]) for record in pre])
            ys = np.asarray([float(record["court_y_m"]) for record in pre])
            setup_span = float(max(np.ptp(xs), np.ptp(ys)))
            opposed = sum(
                set(frame.get("ends") or []) == {"near", "far"}
                for frame in timeline.frames
                if abs(float(frame["time"]) - time_s) <= 0.45
            )
            recovery = 0.0
            if post:
                last_y = float(post[-1]["court_y_m"])
                recovery = (
                    max(0.0, last_y - court_y) if court_y < NET_Y else max(0.0, court_y - last_y)
                )
            for hand in ("left", "right"):
                overhead = peak.get("wrist_overhead_by_hand", {}).get(hand)
                if overhead is None or float(overhead) < float(cfg.pose_serve_overhead_ratio):
                    continue
                local_overheads = [
                    float(record.get("wrist_overhead_by_hand", {}).get(hand, -1.0))
                    for record in nearby_peak
                ]
                if float(overhead) < max(local_overheads, default=-1.0):
                    continue
                pre_overheads = [
                    float(record.get("wrist_overhead_by_hand", {}).get(hand, -1.0))
                    for record in pre
                    if hand in record.get("wrist_overhead_by_hand", {})
                ]
                peak_wrist = peak.get("wrists_body", {}).get(hand)
                post_wrists = [
                    record.get("wrists_body", {}).get(hand)
                    for record in post
                    if record.get("wrists_body", {}).get(hand) is not None
                ]
                if not pre_overheads or peak_wrist is None or not post_wrists:
                    continue
                rise = max(0.0, float(overhead) - min(pre_overheads))
                hand_speed = max(
                    [
                        float(record.get("wrist_speed_by_hand_body_s", {}).get(hand, 0.0))
                        for record in nearby_peak
                    ],
                    default=0.0,
                )
                follow = max(
                    float(
                        np.linalg.norm(
                            np.asarray(wrist, dtype=float) - np.asarray(peak_wrist, dtype=float)
                        )
                    )
                    for wrist in post_wrists
                )
                follow_horizontal = max(
                    abs(float(wrist[0]) - float(peak_wrist[0])) for wrist in post_wrists
                )
                hand_overhead_frames = sum(
                    float(record.get("wrist_overhead_by_hand", {}).get(hand, -1.0))
                    >= float(cfg.pose_serve_overhead_ratio)
                    for record in nearby_peak
                )
                racket_frames = sum(
                    bool(record.get("racket_wrist_associated"))
                    and record.get("racket_nearest_hand") == hand
                    for record in (*pre, *nearby_peak, *post)
                )
                drive_support = bool(
                    leg_drive >= float(cfg.pose_serve_leg_drive_deg)
                    or recovery >= 0.12
                    or racket_frames > 0
                    or (opposed > 0 and follow >= 1.5 * float(cfg.pose_serve_min_follow_body))
                )
                arm_completion = bool(
                    follow >= float(cfg.pose_serve_min_follow_body)
                    and (
                        follow_horizontal >= float(cfg.pose_serve_min_follow_horizontal_body)
                        or racket_frames > 0
                        or hand_speed >= 1.25 * float(cfg.pose_serve_min_hand_speed_body_s)
                    )
                )
                ordered = bool(
                    rise >= float(cfg.pose_serve_min_wrist_rise)
                    and hand_speed >= float(cfg.pose_serve_min_hand_speed_body_s)
                    and (load_frames > 0 or knee_bend_frames > 0)
                    and arm_completion
                    and setup_span <= float(cfg.pose_serve_setup_span_m)
                    and drive_support
                )
                score = float(
                    np.clip(
                        0.18
                        * min(
                            float(overhead) / max(cfg.pose_serve_overhead_ratio, 1e-6),
                            1.0,
                        )
                        + 0.17 * min(rise / max(cfg.pose_serve_min_wrist_rise, 1e-6), 1.0)
                        + 0.18
                        * min(
                            hand_speed / max(cfg.pose_serve_min_hand_speed_body_s, 1e-6),
                            1.8,
                        )
                        / 1.8
                        + 0.10 * min(follow / max(cfg.pose_serve_min_follow_body, 1e-6), 1.0)
                        + 0.07
                        * min(
                            follow_horizontal
                            / max(cfg.pose_serve_min_follow_horizontal_body, 1e-6),
                            1.0,
                        )
                        + 0.10 * min(load_frames + knee_bend_frames, 1)
                        + 0.08
                        * min(
                            max(
                                recovery / 0.25,
                                leg_drive / max(cfg.pose_serve_leg_drive_deg, 1e-6),
                            ),
                            1.0,
                        )
                        + 0.06 * min(opposed / 2.0, 1.0)
                        + 0.04
                        * max(
                            0.0,
                            1.0 - setup_span / max(cfg.pose_serve_setup_span_m, 1e-6),
                        )
                        + 0.02 * min(racket_frames, 1),
                        0.0,
                        1.0,
                    )
                )
                accepted = bool(ordered and score >= float(cfg.pose_serve_min_score))
                hand_xy, hand_confidence = _raised_hand(peak, hand)
                setup_start, setup_evidence = _serve_setup_start(records, peak, hand, cfg)
                candidates.append(
                    {
                        "point": [round(setup_start, 3), round(time_s + 0.70, 3)],
                        "setup_start": round(setup_start, 3),
                        "setup_evidence": setup_evidence,
                        "first_strike": round(time_s, 3),
                        "pose_evidence_time": round(time_s, 3),
                        "actor_id": actor,
                        "team_id": peak.get("team_id"),
                        "accepted": accepted,
                        "serve_motion": accepted,
                        "serve_sequence_evidence": ordered,
                        "serve_sequence_score": round(score, 4),
                        "serve_hand": hand,
                        "overhead_frames": hand_overhead_frames,
                        "overhead_max_ratio": round(float(overhead), 4),
                        "wrist_rise_span": round(rise, 4),
                        "hand_speed_body_s": round(hand_speed, 4),
                        "follow_through_body_lengths": round(follow, 4),
                        "follow_through_horizontal_body_lengths": round(follow_horizontal, 4),
                        "arm_completion_evidence": arm_completion,
                        "drive_phase_evidence": drive_support,
                        "knee_bend_frames": knee_bend_frames,
                        "server_load_frames": load_frames,
                        "leg_drive_frames": int(leg_drive >= cfg.pose_serve_leg_drive_deg),
                        "server_baseline_frames": len(pre) + 1,
                        "opposed_formation_frames": opposed,
                        "setup_span_m": round(setup_span, 4),
                        "recovery_toward_court_m": round(recovery, 4),
                        "pose_server_end": peak["actor_end"],
                        "pose_server_court_x_m": round(float(peak["court_x_m"]), 4),
                        "server_court_half": (
                            "left" if float(peak["court_x_m"]) < DOUBLES_W / 2.0 else "right"
                        ),
                        "server_bbox_norm": peak["bbox_norm"],
                        "racket_hand_xy_norm": hand_xy,
                        "racket_hand_confidence": round(hand_confidence, 4),
                        "racket_observed_frames": racket_frames,
                        "racket_wrist_associated": racket_frames > 0,
                        "racket_bbox_norm": next(
                            (
                                record.get("racket_bbox_norm")
                                for record in nearby_peak
                                if record.get("racket_wrist_associated")
                                and record.get("racket_nearest_hand") == hand
                            ),
                            None,
                        ),
                        "pose_keypoints_norm": peak["keypoints_norm"],
                        "pose_keypoint_confidence": peak["keypoint_confidence"],
                        "target_court_filtered": True,
                        "position_setup_evidence": setup_span <= cfg.pose_serve_setup_span_m,
                        "observable": True,
                        "sampled_frames": len(pre) + len(nearby_peak) + len(post),
                        "pose_frames": len(pre) + len(nearby_peak) + len(post),
                        "ready_frames": load_frames,
                    }
                )
    # A physical service action may create several neighbouring maxima or fallback IDs.
    selected: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            bool(item.get("accepted")),
            float(item["serve_sequence_score"]),
        ),
        reverse=True,
    ):
        if any(
            abs(float(candidate["first_strike"]) - float(prior["first_strike"])) < 1.25
            for prior in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: float(item["first_strike"]))


def detect_strokes(
    timeline: PoseTimeline, serves: Sequence[dict[str, Any]], cfg
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    serve_times = [float(item["first_strike"]) for item in serves if item.get("accepted")]
    raw: list[dict[str, Any]] = []
    for records in timeline.records_by_actor.values():
        raw.extend(_action_candidates(records, serve_times, cfg))
    raw.sort(key=lambda item: float(item["time"]))
    return raw, _stroke_episodes(raw, cfg)
