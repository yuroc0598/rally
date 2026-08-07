"""Body-pose records and local tennis stroke proxies.

COCO-17 has no racket keypoint.  These functions therefore publish measured wrist/body
motion with explicit proxy labels; point validity and temporal tennis rules live in
``pose_timeline`` rather than being duplicated here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .court import NET_Y
from .player import _body_pose_features, _joint_angle_deg, _minimum_knee_angle


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = (float(value) for value in left)
    rx0, ry0, rx1, ry1 = (float(value) for value in right)
    overlap = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0))
    union = ((lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0)
             - overlap)
    return overlap / union if union > 1e-9 else 0.0


def _track_id_to_player(match: dict[str, Any]) -> dict[int, str]:
    """Return persistent player IDs or honest per-track fragment actor IDs."""
    output: dict[int, str] = {}
    for player in match.get("roster") or []:
        if not isinstance(player, dict) or not player.get("id"):
            continue
        for raw in player.get("source_track_ids") or []:
            try:
                output[int(raw)] = str(player["id"])
            except (TypeError, ValueError):
                continue
    for assignment in match.get("track_assignments") or []:
        if not isinstance(assignment, dict) or assignment.get("track_id") is None:
            continue
        actor = assignment.get("actor_id")
        if not actor:
            continue
        try:
            output.setdefault(int(assignment["track_id"]), str(actor))
        except (TypeError, ValueError):
            continue
    return output


def _actor_for_box(
    box_norm: Sequence[float],
    tracked_people: Sequence[dict],
    raw_to_player: dict[int, str],
) -> tuple[str | None, int | None, float]:
    """Associate a pose only through the tracker-owned source box.

    Court-end rank is deliberately not an identity fallback: players cross, partners
    overlap, and a missed box must remain unknown instead of being assigned to whichever
    roster slot happens to occupy the same end of the court.
    """
    scored = [
        (_box_iou(box_norm, person["bbox_norm"]), person)
        for person in tracked_people
        if isinstance(person, dict)
        and isinstance(person.get("bbox_norm"), (list, tuple))
        and len(person["bbox_norm"]) == 4
    ]
    if scored:
        overlap, person = max(scored, key=lambda item: item[0])
        raw = person.get("track_id")
        raw_id = int(raw) if raw is not None else None
        if overlap >= 0.12 and raw_id in raw_to_player:
            return raw_to_player[raw_id], raw_id, float(overlap)
    return None, None, 0.0


def _arm_angle(pose: np.ndarray, confidence: np.ndarray, hand: str) -> float:
    shoulder, elbow, wrist = (5, 7, 9) if hand == "left" else (6, 8, 10)
    if min(float(confidence[index]) for index in (shoulder, elbow, wrist)) <= 0.2:
        return float("nan")
    return _joint_angle_deg(pose[shoulder], pose[elbow], pose[wrist])


def _pose_record(
    time_s: float,
    pose: np.ndarray,
    confidence: np.ndarray,
    box: np.ndarray,
    frame_shape: tuple[int, int],
    court_xy: np.ndarray,
    actor_id: str,
    raw_track_id: int | None,
    association_iou: float,
    cfg,
) -> dict[str, Any] | None:
    height, width = frame_shape
    usable, ready, wrist_ratio = _body_pose_features(pose, confidence, cfg)
    if not usable:
        return None
    shoulder = (pose[5] + pose[6]) / 2.0
    hip = (pose[11] + pose[12]) / 2.0
    torso = float(np.linalg.norm(hip - shoulder))
    if torso <= 1e-6:
        return None
    wrists: dict[str, list[float]] = {}
    for hand, joint in (("left", 9), ("right", 10)):
        if float(confidence[joint]) > 0.2:
            local = (pose[joint] - shoulder) / torso
            wrists[hand] = [float(local[0]), float(local[1])]
    if not wrists:
        return None
    return {
        "time": float(time_s),
        "actor_id": actor_id,
        "source_track_id": raw_track_id,
        "association_iou": float(association_iou),
        "actor_end": "near" if float(court_xy[1]) < NET_Y else "far",
        "court_x_m": float(court_xy[0]),
        "court_y_m": float(court_xy[1]),
        "bbox_norm": [float(box[0] / width), float(box[1] / height),
                      float(box[2] / width), float(box[3] / height)],
        "keypoints_norm": [
            [float(point[0] / width), float(point[1] / height)]
            for point in pose[:17]
        ],
        "keypoint_confidence": [float(value) for value in confidence[:17]],
        "wrists_body": wrists,
        "left_elbow_deg": _arm_angle(pose, confidence, "left"),
        "right_elbow_deg": _arm_angle(pose, confidence, "right"),
        "wrist_overhead_ratio": float(wrist_ratio),
        "minimum_knee_deg": _minimum_knee_angle(pose, confidence),
        "ready_stance": bool(ready),
    }


def _racket_evidence(
    record: dict[str, Any], person: dict[str, Any],
) -> dict[str, Any]:
    """Associate a same-pass COCO racket box with the nearest measured wrist."""
    raw_box = person.get("racket_bbox_norm")
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return {
            "racket_observed": False,
            "racket_wrist_associated": False,
        }
    box = [float(value) for value in raw_box]
    if not np.isfinite(box).all():
        return {"racket_observed": False, "racket_wrist_associated": False}
    keypoints = np.asarray(record.get("keypoints_norm") or [], dtype=float)
    confidence = record.get("keypoint_confidence") or []
    if keypoints.shape != (17, 2) or len(confidence) < 17:
        return {"racket_observed": True, "racket_wrist_associated": False,
                "racket_bbox_norm": box}
    shoulder = np.mean(keypoints[[5, 6]], axis=0)
    hip = np.mean(keypoints[[11, 12]], axis=0)
    torso = float(np.linalg.norm(hip - shoulder))
    centre = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
    choices = [
        (hand, float(np.linalg.norm(keypoints[index] - centre) / max(torso, 1e-6)))
        for hand, index in (("left", 9), ("right", 10))
        if float(confidence[index]) > 0.2
    ]
    hand, distance = min(choices, key=lambda item: item[1]) if choices else (None, None)
    associated = bool(distance is not None and distance <= 2.0)
    return {
        "racket_observed": True,
        "racket_wrist_associated": associated,
        "racket_bbox_norm": box,
        "racket_detection_confidence": float(person.get("racket_confidence") or 0.0),
        "racket_nearest_hand": hand,
        "racket_wrist_distance_body": (
            round(float(distance), 4) if distance is not None else None),
    }


def _smoothed_records(
    records: Sequence[dict], radius_s: float
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: float(item["time"]))
    output: list[dict[str, Any]] = []
    left = 0
    right = 0
    for current in ordered:
        time_s = float(current["time"])
        while left < len(ordered) and float(ordered[left]["time"]) < time_s - radius_s:
            left += 1
        right = max(right, left)
        while right < len(ordered) and float(ordered[right]["time"]) <= time_s + radius_s:
            right += 1
        nearby = ordered[left:right]
        item = dict(current)
        wrists: dict[str, list[float]] = {}
        for hand in ("left", "right"):
            values = [record["wrists_body"].get(hand) for record in nearby]
            values = [value for value in values if value is not None]
            if values:
                median = np.median(np.asarray(values, dtype=float), axis=0)
                wrists[hand] = [float(median[0]), float(median[1])]
        item["wrists_body"] = wrists
        output.append(item)
    return output


def _instant_speed(records: Sequence[dict], index: int, hand: str) -> float:
    speeds: list[float] = []
    for left, right in ((index - 1, index), (index, index + 1)):
        if left < 0 or right >= len(records):
            continue
        a = records[left]["wrists_body"].get(hand)
        b = records[right]["wrists_body"].get(hand)
        dt = float(records[right]["time"]) - float(records[left]["time"])
        if a is not None and b is not None and 0.0 < dt <= 0.35:
            speeds.append(float(np.linalg.norm(np.asarray(b) - np.asarray(a)) / dt))
    return max(speeds, default=0.0)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-9 else 0.0


def _action_candidates(
    records: Sequence[dict], serve_times: Sequence[float], cfg
) -> list[dict[str, Any]]:
    """Return strict local wrist-motion proposals, never asserted ball contacts."""
    if len(records) < 5:
        return []
    records = _smoothed_records(records, float(cfg.pose_action_smoothing_s))
    ordered_serves = sorted(float(value) for value in serve_times)
    output: list[dict[str, Any]] = []
    for index, current in enumerate(records):
        time_s = float(current["time"])
        if any(abs(time_s - serve) <= cfg.pose_service_attempt_mask_s
               for serve in ordered_serves):
            continue
        pre = [record for record in records
               if time_s - cfg.pose_action_pre_s <= float(record["time"]) <= time_s - 0.08]
        post = [record for record in records
                if time_s + 0.08 <= float(record["time"]) <= time_s + cfg.pose_action_post_s]
        if not pre or not post:
            continue
        classified: list[tuple] = []
        for hand in ("left", "right"):
            here = current["wrists_body"].get(hand)
            before = [item["wrists_body"].get(hand) for item in pre]
            after_values = [item["wrists_body"].get(hand) for item in post]
            before = [value for value in before if value is not None]
            after_values = [value for value in after_values if value is not None]
            if here is None or len(before) < 2 or not after_values:
                continue
            split = max(1, len(before) // 2)
            early = np.median(np.asarray(before[:split], float), axis=0)
            loaded = np.median(np.asarray(before[split:], float), axis=0)
            after = np.median(np.asarray(after_values, float), axis=0)
            here_array = np.asarray(here, float)
            preparation_vector = loaded - early
            forward_vector = here_array - loaded
            follow_vector = after - here_array
            backswing = float(np.linalg.norm(preparation_vector))
            forward = float(np.linalg.norm(forward_vector))
            follow = float(np.linalg.norm(follow_vector))
            through = float(np.linalg.norm(after - loaded))
            speed = _instant_speed(records, index, hand)
            horizontal = abs(float(after[0] - loaded[0]))
            elbow = float(current[f"{hand}_elbow_deg"])
            pre_elbows = [float(item[f"{hand}_elbow_deg"]) for item in pre
                          if np.isfinite(float(item[f"{hand}_elbow_deg"]))]
            extension_gain = elbow - float(np.median(pre_elbows)) if pre_elbows else 0.0
            preparation_cosine = _cosine(preparation_vector, forward_vector)
            follow_cosine = _cosine(forward_vector, follow_vector)
            reversal = bool(
                preparation_cosine <= -0.10
                or preparation_vector[0] * forward_vector[0] <= -0.025)
            compact_zone = abs(float(current["court_y_m"]) - NET_Y) \
                <= cfg.pose_compact_stroke_net_distance_m
            overhead = float(current["wrist_overhead_ratio"]) >= 0.15
            groundstroke = bool(
                speed >= cfg.pose_action_min_speed_body_s
                and backswing >= cfg.pose_action_min_backswing_body
                and forward >= 0.16 and follow >= cfg.pose_action_min_follow_body
                and through >= cfg.pose_action_min_through_body
                and horizontal >= 0.24 and elbow >= 85.0 and reversal
                and follow_cosine >= -0.25)
            compact = bool(
                compact_zone and speed >= 0.85 * cfg.pose_action_min_speed_body_s
                and backswing >= 0.12 and forward >= 0.14 and follow >= 0.14
                and through >= 0.30 and elbow >= 105.0 and extension_gain >= 4.0
                and reversal and follow_cosine >= -0.15)
            overhead_action = bool(
                overhead and speed >= 0.90 * cfg.pose_action_min_speed_body_s
                and backswing >= 0.12 and forward >= 0.14
                and follow >= cfg.pose_action_min_follow_body and through >= 0.30
                and reversal and follow_cosine >= -0.20)
            action_type = (
                "overhead_stroke_proxy" if overhead_action else
                "compact_stroke_proxy" if compact else
                "groundstroke_proxy" if groundstroke else None)
            if action_type is None or not np.isfinite(elbow):
                continue
            racket_associated = bool(current.get("racket_wrist_associated"))
            score = float(np.clip(
                0.30 * min(speed / cfg.pose_action_min_speed_body_s, 1.5) / 1.5
                + 0.20 * min(backswing / max(cfg.pose_action_min_backswing_body, 1e-6), 1.0)
                + 0.20 * min(follow / max(cfg.pose_action_min_follow_body, 1e-6), 1.0)
                + 0.15 * min(through / max(cfg.pose_action_min_through_body, 1e-6), 1.0)
                + 0.10 * min(elbow / 150.0, 1.0) + 0.05 * float(reversal)
                + 0.05 * float(racket_associated),
                0.0, 0.99))
            if score >= cfg.pose_action_min_confidence:
                classified.append((
                    score, backswing + forward + follow + through, action_type,
                    hand, backswing, forward, follow, through, speed, elbow,
                    extension_gain, preparation_cosine, follow_cosine, reversal,
                    compact_zone))
        if not classified:
            continue
        (score, _motion, action_type, hand, backswing, forward, follow, through,
         speed, elbow, extension_gain, preparation_cosine, follow_cosine,
         reversal, compact_zone) = max(classified, key=lambda item: (item[0], item[1]))
        output.append({
            "time": round(time_s, 3), "actor_id": current["actor_id"],
            "team_id": current.get("team_id"), "actor_end": current["actor_end"],
            "action": action_type, "hand": hand, "confidence": round(score, 4),
            "backswing_body_lengths": round(backswing, 4),
            "forward_span_body_lengths": round(forward, 4),
            "forward_speed_body_lengths_s": round(speed, 4),
            "follow_through_body_lengths": round(follow, 4),
            "through_span_body_lengths": round(through, 4),
            "elbow_extension_deg": round(elbow, 2),
            "elbow_extension_gain_deg": round(extension_gain, 2),
            "preparation_forward_cosine": round(preparation_cosine, 4),
            "forward_follow_cosine": round(follow_cosine, 4),
            "direction_reversal": reversal, "compact_motion_zone": compact_zone,
            "bbox_norm": current["bbox_norm"],
            "keypoints_norm": current["keypoints_norm"],
            "keypoint_confidence": current["keypoint_confidence"],
            "racket_observed": bool(current.get("racket_observed")),
            "racket_wrist_associated": bool(current.get("racket_wrist_associated")),
            "racket_bbox_norm": current.get("racket_bbox_norm"),
            "racket_detection_confidence": current.get("racket_detection_confidence"),
            "racket_wrist_distance_body": current.get("racket_wrist_distance_body"),
            "evidence": ["ordered_wrist_preparation", "direction_change_into_forward_swing",
                         "local_forward_acceleration", "post_contact_follow_through",
                         "target_court_player_pose",
                         ("detected_racket_near_wrist" if racket_associated
                          else "racket_not_observed_wrist_proxy"),
                         "ball_contact_not_observed"],
        })
    selected: list[dict[str, Any]] = []
    for candidate in sorted(output, key=lambda item: float(item["confidence"]), reverse=True):
        if any(candidate["actor_id"] == prior["actor_id"]
               and abs(float(candidate["time"]) - float(prior["time"]))
               < cfg.pose_action_nms_s for prior in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: float(item["time"]))


def _stroke_episodes(
    proposals: Sequence[dict[str, Any]], cfg
) -> list[dict[str, Any]]:
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        by_actor[str(proposal.get("actor_id") or "unassigned")].append(dict(proposal))
    episodes: list[dict[str, Any]] = []
    for actor_proposals in by_actor.values():
        clusters: list[list[dict[str, Any]]] = []
        for proposal in sorted(actor_proposals, key=lambda item: float(item["time"])):
            if not clusters or float(proposal["time"]) - float(clusters[-1][-1]["time"]) \
                    > cfg.pose_stroke_episode_gap_s:
                clusters.append([])
            clusters[-1].append(proposal)
        for cluster in clusters:
            strongest = dict(max(cluster, key=lambda item: float(item["confidence"])))
            strongest.update({
                "episode_start": round(float(cluster[0]["time"]), 3),
                "episode_end": round(float(cluster[-1]["time"]), 3),
                "proposal_count": len(cluster),
            })
            episodes.append(strongest)
    return sorted(episodes, key=lambda item: float(item["time"]))
