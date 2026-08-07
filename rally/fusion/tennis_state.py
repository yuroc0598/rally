"""Offline tennis-state decoding from player pose and racket-motion proxies.

Signal extraction belongs in :mod:`rally.signals`; this module owns all temporal tennis
rules.  It deliberately consumes neither audio nor ball detections.  Future context is
used to join service retries, recover serve-missed live play, and backdate a point end to
the measured cessation of live player state instead of retaining later walking footage.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

Segment = tuple[float, float]


def _opposite(end: str | None) -> str | None:
    return "far" if end == "near" else "near" if end == "far" else None


def _has_return(
    serve: dict[str, Any], episodes: Sequence[dict[str, Any]], cfg,
) -> bool:
    strike = float(serve["first_strike"])
    server_end = serve.get("pose_server_end")
    if server_end not in {"near", "far"}:
        return False
    return any(
        strike + float(cfg.pose_service_attempt_mask_s) < float(item["time"])
        <= strike + float(cfg.pose_first_return_max_s)
        and item.get("actor_end") == _opposite(str(server_end))
        for item in episodes
    )


def _same_service_context(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Match the physical service court without requiring one fragile tracker ID."""
    if left.get("pose_server_end") != right.get("pose_server_end"):
        return False
    if left.get("server_court_half") != right.get("server_court_half"):
        return False
    left_team, right_team = left.get("team_id"), right.get("team_id")
    return not (left_team and right_team and left_team != right_team)


def _service_groups(
    serves: Sequence[dict[str, Any]], episodes: Sequence[dict[str, Any]], cfg,
) -> list[list[dict[str, Any]]]:
    """Group faults/lets before selecting the attempt that actually starts output.

    A retry is a service-court event, not a raw-track-ID event.  An apparent court-end
    switch inside a few seconds is physically impossible without a changeover, so it is
    retained as conflicting evidence in the same group rather than emitted as a second
    point.
    """
    accepted = sorted(
        (dict(item) for item in serves if item.get("accepted")),
        key=lambda item: float(item["first_strike"]),
    )
    groups: list[list[dict[str, Any]]] = []
    for serve in accepted:
        if groups:
            previous = groups[-1][-1]
            gap = float(serve["first_strike"]) - float(previous["first_strike"])
            no_return = not _has_return(previous, episodes, cfg)
            same_context = _same_service_context(previous, serve)
            impossible_end_switch = bool(
                previous.get("pose_server_end") in {"near", "far"}
                and serve.get("pose_server_end") in {"near", "far"}
                and previous.get("pose_server_end") != serve.get("pose_server_end")
                and gap <= float(cfg.pose_service_impossible_end_switch_s)
            )
            if (gap <= float(cfg.pose_service_retry_max_gap_s)
                    and no_return and (same_context or impossible_end_switch)):
                serve["retry_relation"] = (
                    "same_service_court" if same_context
                    else "impossible_short_end_switch")
                groups[-1].append(serve)
                continue
        groups.append([serve])
    return groups


def _alternating_actions(
    episodes: Sequence[dict[str, Any]], start: float, stop: float,
    expected_end: str | None, cfg,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a time-bounded alternating near/far response sequence."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_time = start
    expected = expected_end
    for raw in episodes:
        time_s = float(raw["time"])
        if not start < time_s < stop:
            continue
        if not accepted and time_s <= start + float(cfg.pose_service_attempt_mask_s):
            continue
        item = dict(raw)
        court_end = item.get("actor_end")
        if court_end not in {"near", "far"}:
            item.update(accepted=False, rejection_reason="unknown_court_end")
            rejected.append(item)
            continue
        allowed = (
            float(cfg.pose_first_return_max_s) if not accepted
            else float(cfg.pose_exchange_max_gap_s))
        if time_s - last_time > allowed:
            item.update(accepted=False, rejection_reason="response_timeout")
            rejected.append(item)
            break
        if expected is not None and court_end != expected:
            item.update(accepted=False, rejection_reason="same_end_out_of_turn")
            rejected.append(item)
            continue
        item.update({
            "accepted": True,
            "sequence_index": len(accepted) + 1,
            "sequence_role": "return" if not accepted else "exchange",
        })
        accepted.append(item)
        last_time = time_s
        expected = _opposite(str(court_end))
    return accepted, rejected


def _exchange_sequences(
    episodes: Sequence[dict[str, Any]], cfg,
) -> list[list[dict[str, Any]]]:
    sequences: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw in sorted(episodes, key=lambda item: float(item["time"])):
        item = dict(raw)
        court_end = item.get("actor_end")
        if court_end not in {"near", "far"}:
            continue
        if current:
            gap = float(item["time"]) - float(current[-1]["time"])
            if gap > float(cfg.pose_exchange_max_gap_s) or (
                court_end == current[-1].get("actor_end")
            ):
                if len(current) >= int(cfg.pose_exchange_min_actions):
                    sequences.append(current)
                current = []
        current.append(item)
    if len(current) >= int(cfg.pose_exchange_min_actions):
        sequences.append(current)
    return sequences


def _live_bouts(frames: Sequence[dict[str, Any]], cfg) -> list[dict[str, Any]]:
    """Collapse pose dropouts while preserving sustained BETWEEN_POINTS resets."""
    ordered = sorted(frames, key=lambda item: float(item["time"]))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_live: float | None = None
    reset_start: float | None = None
    for frame in ordered:
        time_s = float(frame["time"])
        if frame.get("engaged_like"):
            sustained_reset = bool(
                reset_start is not None
                and time_s - reset_start >= float(cfg.pose_live_reset_break_s))
            if current and last_live is not None and (
                time_s - last_live > float(cfg.pose_live_gap_bridge_s)
                or sustained_reset
            ):
                groups.append(current)
                current = []
            current.append(frame)
            last_live = time_s
            reset_start = None
        elif frame.get("between_like"):
            if reset_start is None:
                reset_start = time_s
        else:
            reset_start = None
    if current:
        groups.append(current)
    bouts: list[dict[str, Any]] = []
    for group in groups:
        start, end = float(group[0]["time"]), float(group[-1]["time"])
        if end <= start:
            continue
        bouts.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "live_frames": len(group),
            "max_ready_fraction": max(
                float(frame.get("ready_fraction") or 0.0) for frame in group),
        })
    return bouts


def _matching_bout(
    bouts: Sequence[dict[str, Any]], start: float, stop: float,
) -> dict[str, Any] | None:
    candidates = [
        bout for bout in bouts
        if float(bout["end"]) >= start and float(bout["start"]) <= stop
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        min(float(item["end"]), stop) - max(float(item["start"]), start),
        float(item["duration"]),
    ))


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _point_hypotheses(
    timeline, serves: Sequence[dict[str, Any]], episodes: Sequence[dict[str, Any]],
    duration: float, cfg,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    bouts = _live_bouts(timeline.frames, cfg)
    groups = _service_groups(serves, episodes, cfg)
    hypotheses: list[dict[str, Any]] = []
    service_coverage: list[tuple[float, float]] = []

    for group in groups:
        returned = next((item for item in group if _has_return(item, episodes, cfg)), None)
        selected = returned or group[-1]
        strike = float(selected["first_strike"])
        bout = _matching_bout(
            bouts,
            strike - float(cfg.pose_point_reset_delay_s),
            min(duration, strike + float(cfg.pose_engaged_search_s)),
        )
        if returned is None and bout is None and float(
            selected.get("serve_sequence_score") or 0.0
        ) < float(cfg.pose_serve_unreturned_min_score):
            continue
        setup_start = float(selected.get("setup_start", selected["point"][0]))
        live_end = float(bout["end"]) if bout is not None else strike
        hypotheses.append({
            "kind": "serve",
            "start": setup_start,
            "live_start": strike,
            "live_end": live_end,
            "live_bout": bout,
            "server_end": selected.get("pose_server_end"),
            "serve": selected,
            "attempts": group,
            "classification": (
                "confirmed_rally" if returned is not None
                else "serve_led_player_activity" if bout is not None
                else "unreturned_service_point"),
        })
        coverage_start = min(
            float(item.get("setup_start", item["point"][0])) for item in group)
        service_coverage.append((coverage_start, max(live_end, strike + 1.0)))

    sequences = _exchange_sequences(episodes, cfg)
    for bout in bouts:
        start, end = float(bout["start"]), float(bout["end"])
        if float(bout["duration"]) < float(cfg.pose_live_candidate_min_s):
            continue
        if any(_overlap((start, end), interval) > 0.5 for interval in service_coverage):
            continue
        local_sequences = [
            sequence for sequence in sequences
            if start - 0.5 <= float(sequence[0]["time"])
            and float(sequence[-1]["time"]) <= end + 0.5
        ]
        seed = max(local_sequences, key=len) if local_sequences else []
        if seed:
            first = float(seed[0]["time"])
            point_start = max(start, first - float(cfg.pose_exchange_start_lookback_s))
            live_start = first
            classification = "exchange_without_observed_serve"
        else:
            point_start = start
            live_start = start
            classification = "two_sided_live_state_without_observed_serve"
        hypotheses.append({
            "kind": "exchange" if seed else "live_state",
            "start": max(0.0, point_start),
            "live_start": live_start,
            "live_end": end,
            "live_bout": bout,
            "server_end": None,
            "serve": None,
            "attempts": [],
            "seed_actions": seed,
            "classification": classification,
        })

    # A fault/let can look like a short live-state burst when the overhead frame is
    # occluded.  Two unserved bouts separated by only a few seconds cannot be separate
    # tennis points; if the first has no alternating exchange, defer to the later bout.
    state_hypotheses = sorted(
        (item for item in hypotheses if item["kind"] in {"exchange", "live_state"}),
        key=lambda item: float(item["start"]),
    )
    superseded_ids: set[int] = set()
    for left, right in pairwise(state_hypotheses):
        gap = float(right["start"]) - float(left["live_end"])
        if (left["kind"] == "live_state"
                and 0.0 <= gap <= float(cfg.pose_service_retry_state_max_gap_s)):
            superseded_ids.add(id(left))
    hypotheses = [item for item in hypotheses if id(item) not in superseded_ids]

    # Prefer a measured serve sequence when two hypotheses explain the same live bout.
    hypotheses.sort(key=lambda item: (float(item["start"]), item["kind"] != "serve"))
    deduplicated: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        interval = (float(hypothesis["live_start"]), float(hypothesis["live_end"]))
        duplicate_index = next((
            index for index, prior in enumerate(deduplicated)
            if _overlap(interval, (
                float(prior["live_start"]), float(prior["live_end"]))) > 0.5
        ), None)
        if duplicate_index is None:
            deduplicated.append(hypothesis)
        elif (hypothesis["kind"] == "serve"
              and deduplicated[duplicate_index]["kind"] != "serve"):
            deduplicated[duplicate_index] = hypothesis
    return sorted(deduplicated, key=lambda item: float(item["start"])), groups, bouts


def _sustained_between_endpoint(
    frames: Sequence[dict[str, Any]], after: float, before: float, cfg,
) -> tuple[float | None, dict[str, Any]]:
    required = max(2, int(np.ceil(
        float(cfg.pose_between_min_s) * float(cfg.pose_timeline_fps))))
    eligible = [frame for frame in frames if after <= float(frame["time"]) <= before]
    for index in range(len(eligible)):
        window = eligible[index:index + required]
        if len(window) < required:
            break
        observed = [
            frame for frame in window
            if int(frame.get("visible_players") or 0)
            >= int(cfg.pose_point_min_visible_players)
            and set(frame.get("ends") or []) == {"near", "far"}
        ]
        if len(observed) / len(window) < float(cfg.pose_between_min_fraction):
            continue
        relaxed = [
            frame for frame in observed
            if frame.get("between_like") and not frame.get("engaged_like")
        ]
        if not relaxed:
            continue
        return float(relaxed[0]["time"]), {
            "required_frames": required,
            "window_frames": len(window),
            "two_sided_observed_fraction": round(len(observed) / len(window), 4),
            "backdated_to_first_relaxed_frame": True,
        }
    return None, {"required_frames": required, "eligible_frames": len(eligible)}


def decode_pose_points(
    timeline,
    serves: Sequence[dict[str, Any]],
    raw_actions: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    duration: float,
    cfg,
) -> tuple[list[Segment], list[dict[str, Any]], dict[str, Any]]:
    """Decode the complete timeline, then publish only evidence-bounded points."""
    mutable_serves = [dict(item) for item in serves]
    for serve in mutable_serves:
        if (not serve.get("accepted")
                and float(serve.get("serve_sequence_score") or 0.0) >= 0.52
                and _has_return(serve, episodes, cfg)):
            serve.update(
                accepted=True,
                serve_motion=True,
                acceptance_support="opposite_end_return_sequence",
            )

    hypotheses, service_groups, live_bouts = _point_hypotheses(
        timeline, mutable_serves, episodes, duration, cfg)
    segments: list[Segment] = []
    points: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    endpoint_records: list[dict[str, Any]] = []

    for index, hypothesis in enumerate(hypotheses):
        next_start = (
            float(hypotheses[index + 1]["start"])
            if index + 1 < len(hypotheses) else duration)
        point_start = float(hypothesis["start"])
        live_start = float(hypothesis["live_start"])
        live_end = min(float(hypothesis["live_end"]), next_start, duration)
        if hypothesis["kind"] == "serve":
            accepted, rejected = _alternating_actions(
                episodes, live_start, next_start,
                _opposite(hypothesis.get("server_end")), cfg)
        else:
            accepted = [
                dict(item, accepted=True)
                for item in hypothesis.get("seed_actions") or []
            ]
            rejected = []

        last_racket_action = max(
            [live_start] + [float(item["time"]) for item in accepted])
        last_live_evidence = max(last_racket_action, live_end)
        transition_time, transition_evidence = _sustained_between_endpoint(
            timeline.frames,
            max(live_start, live_end),
            min(next_start, last_live_evidence
                + float(cfg.pose_endpoint_max_unexplained_tail_s)),
            cfg,
        )
        if transition_time is not None:
            point_end = transition_time
            endpoint_source = "backdated_sustained_between_transition"
            endpoint_evidence = transition_evidence
            endpoint_confidence = "high"
        elif live_end > live_start:
            point_end = live_end
            endpoint_source = "measured_live_state_cessation"
            endpoint_evidence = {
                "live_state_end": round(live_end, 3),
                "live_bout": hypothesis.get("live_bout"),
            }
            endpoint_confidence = "medium"
        else:
            point_end = min(
                next_start,
                last_racket_action + float(cfg.pose_unreturned_transition_search_s),
            )
            endpoint_source = "bounded_unreturned_service_window"
            endpoint_evidence = {"last_racket_action": round(last_racket_action, 3)}
            endpoint_confidence = "low"

        point_end = max(point_start, min(point_end, next_start, duration))
        unexplained_tail = max(0.0, point_end - last_live_evidence)
        drop_reason: str | None = None
        if unexplained_tail > float(cfg.pose_endpoint_max_unexplained_tail_s) + 1e-6:
            drop_reason = "endpoint exceeded the maximum unexplained live-evidence tail"
        elif point_end - point_start < float(cfg.min_rally_s):
            drop_reason = "validated point window is shorter than min_rally_s"
        elif (hypothesis["kind"] == "live_state"
              and hypothesis.get("live_bout") is None):
            drop_reason = "serve-less point had no sustained two-sided live state"

        decision = {
            "index": len(decisions),
            "candidate": [round(point_start, 3), round(next_start, 3)],
            "serve_time": ((hypothesis.get("serve") or {}).get("first_strike")),
            "accepted": drop_reason is None,
            "classification": hypothesis["classification"],
            "reason": drop_reason or "constrained tennis-state path",
            "endpoint": round(point_end, 3) if drop_reason is None else None,
            "endpoint_source": endpoint_source,
            "endpoint_confidence": endpoint_confidence,
            "raw_action_count": sum(
                point_start <= float(item["time"]) <= point_end for item in episodes),
            "action_count": len(accepted) + len(rejected),
            "accepted_action_count": len(accepted),
            "live_state_evidence": hypothesis.get("live_bout"),
            "rejected_stroke_episodes": rejected,
        }
        decisions.append(decision)
        if drop_reason is not None:
            continue

        attempts = []
        for attempt_index, attempt in enumerate(hypothesis.get("attempts") or []):
            selected = attempt is hypothesis.get("serve")
            attempts.append({
                "attempt_index": attempt_index,
                "time": attempt["first_strike"],
                "setup_start": attempt.get("setup_start"),
                "server_id": attempt.get("actor_id"),
                "server_team_id": attempt.get("team_id"),
                "server_end": attempt.get("pose_server_end"),
                "server_court_x_m": attempt.get("pose_server_court_x_m"),
                "server_court_half": attempt.get("server_court_half"),
                "disposition": (
                    "selected_playable_attempt" if selected
                    else "superseded_fault_let_or_conflicting_candidate"),
                "retry_relation": attempt.get("retry_relation"),
            })
        selected_serve = hypothesis.get("serve")
        service_action = ([{
            "time": selected_serve["first_strike"],
            "actor_id": selected_serve.get("actor_id"),
            "team_id": selected_serve.get("team_id"),
            "actor_end": selected_serve.get("pose_server_end"),
            "action": "serve",
            "confidence": selected_serve.get("serve_sequence_score"),
            "accepted": True,
            "bbox_norm": selected_serve.get("server_bbox_norm"),
            "keypoints_norm": selected_serve.get("pose_keypoints_norm"),
            "keypoint_confidence": selected_serve.get("pose_keypoint_confidence"),
            "racket_wrist_associated": selected_serve.get("racket_wrist_associated"),
            "racket_bbox_norm": selected_serve.get("racket_bbox_norm"),
        }] if selected_serve else [])
        actions = [*service_action, *accepted]
        point = {
            "index": len(points),
            "start": round(point_start, 3),
            "end": round(point_end, 3),
            "classification": hypothesis["classification"],
            "participants": {
                "server_id": (selected_serve or {}).get("actor_id"),
                "server_team_id": (selected_serve or {}).get("team_id"),
                "server_end": (selected_serve or {}).get("pose_server_end"),
                "active_player_ids": sorted({
                    str(item["actor_id"]) for item in actions if item.get("actor_id")
                }),
            },
            "service_attempts": attempts,
            "actions": actions,
            "state_transitions": [
                {"time": round(point_start, 3), "from": "BETWEEN_POINTS",
                 "to": "SERVE_SETUP" if selected_serve else "LIVE_POINT",
                 "reason": hypothesis["classification"]},
                *([{"time": round(live_start, 3), "from": "SERVE_SETUP",
                    "to": "LIVE_POINT", "reason": "selected_service_attempt"}]
                  if selected_serve else []),
                {"time": round(point_end, 3), "from": "LIVE_POINT",
                 "to": "BETWEEN_POINTS", "reason": endpoint_source},
            ],
            "termination": {
                "time": round(point_end, 3),
                "rule_event": "pose_inferred_live_state_cessation",
                "status": "presentation_estimate",
                "source": endpoint_source,
                "confidence": endpoint_confidence,
                "evidence": [endpoint_source, "no_audio_or_ball_claim"],
            },
        }
        decision.update({
            "service_attempts": attempts,
            "actions": actions,
            "state_transitions": point["state_transitions"],
        })
        points.append(point)
        segments.append((point_start, point_end))
        endpoint_records.append({
            "point_index": len(points) - 1,
            "presentation_time": round(point_end, 3),
            "last_racket_action_time": round(last_racket_action, 3),
            "last_live_evidence_time": round(last_live_evidence, 3),
            "visual_dead_signal": endpoint_source,
            "endpoint_confidence": endpoint_confidence,
            "evidence": endpoint_evidence,
            "tail_after_last_action_s": round(point_end - last_racket_action, 3),
            "unexplained_tail_s": round(unexplained_tail, 3),
        })

    gaps = [
        max(0.0, segments[index + 1][0] - segments[index][1])
        for index in range(len(segments) - 1)
    ]
    retention = (
        sum(max(0.0, end - start) for start, end in segments) / duration
        if duration > 0 else 0.0)
    median_gap = float(np.median(gaps)) if gaps else None
    zero_stroke_fraction = (
        sum(not point.get("actions") or len(point["actions"]) <= 1 for point in points)
        / len(points) if points else 0.0)
    violations = [
        record for record in endpoint_records
        if float(record["unexplained_tail_s"])
        > float(cfg.pose_endpoint_max_unexplained_tail_s)
    ]
    suspicious_tiling = bool(
        len(segments) >= 3
        and retention >= float(cfg.pose_quality_max_retention_fraction)
        and median_gap is not None
        and median_gap <= float(cfg.pose_quality_max_median_gap_s)
    )
    rejected_batch = bool(violations or suspicious_tiling)
    quality = {
        "status": "rejected" if rejected_batch else "passed",
        "retention_fraction": round(retention, 4),
        "accepted_point_count_before_guard": len(segments),
        "median_inter_point_gap_s": (
            round(median_gap, 4) if median_gap is not None else None),
        "zero_post_serve_stroke_fraction": round(zero_stroke_fraction, 4),
        "boundary_invariant_violations": len(violations),
        "thresholds": {
            "max_retention_fraction": cfg.pose_quality_max_retention_fraction,
            "max_median_gap_s": cfg.pose_quality_max_median_gap_s,
            "max_unexplained_tail_s": cfg.pose_endpoint_max_unexplained_tail_s,
            "warning_zero_stroke_fraction": cfg.pose_quality_max_zero_stroke_fraction,
        },
        "warnings": (["most points lack an observed post-serve racket action"]
                     if zero_stroke_fraction
                     > float(cfg.pose_quality_max_zero_stroke_fraction) else []),
        "reason": (
            "point windows violated evidence-bounded endpoint invariants"
            if violations else
            "candidate windows tiled almost the entire source"
            if suspicious_tiling else None),
        "policy": "reject impossible boundaries; report weak stroke coverage explicitly",
    }
    if rejected_batch:
        for decision in decisions:
            if decision.get("accepted"):
                decision.update(accepted=False, quality_gate_rejected=True)
        segments = []
        points = []

    accepted_keys = {
        (str(item.get("actor_id")), float(item["time"]))
        for point in points for item in point.get("actions") or []
        if item.get("action") != "serve"
    }
    episode_records = []
    for episode in episodes:
        item = dict(episode)
        item["accepted"] = (
            str(item.get("actor_id")), float(item["time"])) in accepted_keys
        if not item["accepted"]:
            item["rejection_reason"] = "not_assigned_to_constrained_state_path"
        episode_records.append(item)

    reports = {
        "serve_pose": {
            "status": "used" if any(item.get("accepted") for item in mutable_serves)
            else "no_confirmed_serves",
            "backend": "shared_rtmpose_coco17_timeline",
            "model_scope": "all_tracked_match_players_on_both_court_ends",
            "proposals": len(mutable_serves),
            "confirmed": sum(bool(item.get("accepted")) for item in mutable_serves),
            "serve_times": [
                item["first_strike"] for item in mutable_serves if item.get("accepted")],
            "rejected": sum(not bool(item.get("accepted")) for item in mutable_serves),
            "service_groups": len(service_groups),
            "decision": "multi-frame baseline load, wrist rise, overhead motion and setup onset",
            "observations": mutable_serves,
            "audio_used": False,
        },
        "candidate_generation": {
            "status": "used" if hypotheses else "no_candidates",
            "source": "offline_constrained_tennis_state_decoder",
            "count": len(hypotheses),
            "service_groups": len(service_groups),
            "live_state_bouts": live_bouts,
            "supports_serve_missed_points_anywhere": True,
            "requires_pose_confirmed_serve": False,
        },
        "racket_actions": {
            "status": "used",
            "backend": "shared_rtmpose_coco17_timeline",
            "signal": "multi_frame_wrist_motion_proxy",
            "ball_tracking_used": False,
            "input_candidates": len(hypotheses),
            "raw_action_proposals": len(raw_actions),
            "stroke_episode_count": len(episodes),
            "accepted_stroke_episodes": len(accepted_keys),
            "kept": len(segments),
            "dropped": sum(not bool(item.get("accepted")) for item in decisions),
            "episodes": episode_records,
            "decisions": decisions,
            "policy": "temporal near/far action order corroborates, but never invents, live state",
        },
        "endpoints": {
            "status": "used" if segments else "no_validated_points",
            "semantic_rule_endpoints": [],
            "visual_presentation_endpoints": endpoint_records,
            "policy": "backdate to measured live-state cessation and bound unexplained tails",
        },
        "quality_control": quality,
    }
    return segments, points, reports
