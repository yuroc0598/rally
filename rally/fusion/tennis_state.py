"""Offline tennis-state decoding from player pose and racket-motion proxies.

Signal extraction belongs in :mod:`rally.signals`; this module owns all temporal tennis
rules.  It deliberately consumes neither audio nor ball detections.  Future context is
used to join service retries, recover serve-missed live play, and backdate a point end to
the measured cessation of live player state instead of retaining later walking footage.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

Segment = tuple[float, float]


def _opposite(end: str | None) -> str | None:
    return "far" if end == "near" else "near" if end == "far" else None


def _has_return(
    serve: dict[str, Any],
    episodes: Sequence[dict[str, Any]],
    cfg,
) -> bool:
    strike = float(serve["first_strike"])
    server_end = serve.get("pose_server_end")
    if server_end not in {"near", "far"}:
        return False
    return any(
        strike + float(cfg.pose_service_attempt_mask_s)
        < float(item["time"])
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
    serves: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    cfg,
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
            if (
                gap <= float(cfg.pose_service_retry_max_gap_s)
                and no_return
                and (same_context or impossible_end_switch)
            ):
                serve["retry_relation"] = (
                    "same_service_court" if same_context else "impossible_short_end_switch"
                )
                groups[-1].append(serve)
                continue
        groups.append([serve])
    return groups


def _alternating_actions(
    episodes: Sequence[dict[str, Any]],
    start: float,
    stop: float,
    expected_end: str | None,
    cfg,
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
            float(cfg.pose_first_return_max_s)
            if not accepted
            else float(cfg.pose_exchange_max_gap_s)
        )
        if time_s - last_time > allowed:
            item.update(accepted=False, rejection_reason="response_timeout")
            rejected.append(item)
            break
        if expected is not None and court_end != expected:
            item.update(accepted=False, rejection_reason="same_end_out_of_turn")
            rejected.append(item)
            continue
        item.update(
            {
                "accepted": True,
                "sequence_index": len(accepted) + 1,
                "sequence_role": "return" if not accepted else "exchange",
            }
        )
        accepted.append(item)
        last_time = time_s
        expected = _opposite(str(court_end))
    return accepted, rejected


def _exchange_sequences(
    episodes: Sequence[dict[str, Any]],
    cfg,
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


def _event_end(event: dict[str, Any]) -> str | None:
    value = event.get("pose_server_end", event.get("actor_end"))
    return str(value) if value in {"near", "far"} else None


def _opponent_response(
    event: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    window_start: float,
    window_end: float,
    cfg,
) -> tuple[bool, float | None]:
    """Find a measured opposite-end reaction after a serve/stroke event."""
    event_time = float(event["time"])
    opposite = _opposite(_event_end(event))
    if opposite is None:
        return False, None
    start = max(window_start, event_time + float(cfg.pose_live_response_min_s))
    stop = min(window_end, event_time + float(cfg.pose_live_response_max_s))
    ready_times: list[float] = []
    for frame in frames:
        time_s = float(frame["time"])
        if not start <= time_s <= stop:
            continue
        players = [player for player in frame.get("players") or [] if player.get("end") == opposite]
        if not players:
            continue
        active = any(
            float(player.get("wrist_speed_body_s") or 0.0)
            >= float(cfg.pose_live_arm_activity_speed_body_s)
            or (
                bool(player.get("ready"))
                and float(player.get("court_speed_m_s") or 0.0)
                >= float(cfg.pose_live_court_activity_speed_m_s)
            )
            for player in players
        )
        if active:
            return True, time_s
        if any(bool(player.get("ready")) for player in players):
            ready_times.append(time_s)
    # A receiver who holds a ready posture across multiple samples is meaningful even
    # without a large translation; one isolated ready frame is not.
    if len(ready_times) >= 2:
        return True, ready_times[0]
    return False, None


def _actor_window_evidence(
    frames: Sequence[dict[str, Any]],
    end: str,
    cfg,
) -> tuple[dict[str, dict[str, float | int]], float, float]:
    """Aggregate each actor first, then combine actors without doubles dilution."""
    observed_end_frames = 0
    ready_end_frames = 0
    active_end_frames = 0
    raw: dict[str, dict[str, int]] = {}
    for frame in frames:
        players = [player for player in frame.get("players") or [] if player.get("end") == end]
        if not players:
            continue
        observed_end_frames += 1
        if any(bool(player.get("ready")) for player in players):
            ready_end_frames += 1
        if any(
            float(player.get("wrist_speed_body_s") or 0.0)
            >= float(cfg.pose_live_arm_activity_speed_body_s)
            or (
                bool(player.get("ready"))
                and float(player.get("court_speed_m_s") or 0.0)
                >= float(cfg.pose_live_court_activity_speed_m_s)
            )
            for player in players
        ):
            active_end_frames += 1
        for player in players:
            actor = str(player.get("actor_id") or "unknown")
            counts = raw.setdefault(
                actor,
                {
                    "observed_frames": 0,
                    "ready_frames": 0,
                    "arm_active_frames": 0,
                    "court_active_frames": 0,
                },
            )
            counts["observed_frames"] += 1
            counts["ready_frames"] += int(bool(player.get("ready")))
            counts["arm_active_frames"] += int(
                float(player.get("wrist_speed_body_s") or 0.0)
                >= float(cfg.pose_live_arm_activity_speed_body_s)
            )
            counts["court_active_frames"] += int(
                float(player.get("court_speed_m_s") or 0.0)
                >= float(cfg.pose_live_court_activity_speed_m_s)
            )
    actors: dict[str, dict[str, float | int]] = {}
    for actor, counts in raw.items():
        observed = max(1, counts["observed_frames"])
        actors[actor] = {
            **counts,
            "ready_fraction": round(counts["ready_frames"] / observed, 4),
            "arm_active_fraction": round(counts["arm_active_frames"] / observed, 4),
            "court_active_fraction": round(counts["court_active_frames"] / observed, 4),
        }
    denominator = max(1, observed_end_frames)
    return (
        actors,
        ready_end_frames / denominator,
        active_end_frames / denominator,
    )


def _classify_live_windows(
    frames: Sequence[dict[str, Any]],
    serves: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    cfg,
) -> list[dict[str, Any]]:
    """Classify overlapping two-second windows from all target-court players.

    Only a measured service sequence or a time-ordered stroke/opponent response can
    seed LIVE.  Two-sided ready/recovery activity can maintain an already-live point.
    Sustained relaxed posture with no tennis action is BETWEEN_POINTS; inadequate or
    contradictory observations remain UNKNOWN rather than voting either way.
    """
    ordered_frames = sorted(frames, key=lambda item: float(item["time"]))
    accepted_serves = [
        {**item, "time": float(item["first_strike"]), "event": "serve"}
        for item in serves
        if item.get("accepted")
    ]
    stroke_events = [{**item, "time": float(item["time"]), "event": "stroke"} for item in episodes]
    events = sorted((*accepted_serves, *stroke_events), key=lambda item: float(item["time"]))
    frame_times = [float(frame["time"]) for frame in ordered_frames]
    event_times = [float(event["time"]) for event in events]
    half_window = float(cfg.pose_live_window_s) / 2.0
    output: list[dict[str, Any]] = []
    for centre_frame in ordered_frames:
        centre = float(centre_frame["time"])
        window_start, window_end = centre - half_window, centre + half_window
        frame_left = bisect_left(frame_times, window_start)
        frame_right = bisect_right(frame_times, window_end)
        local_frames = ordered_frames[frame_left:frame_right]
        if not local_frames:
            continue
        event_left = bisect_left(event_times, window_start)
        event_right = bisect_right(event_times, window_end)
        local_events = events[event_left:event_right]
        two_sided = [
            frame
            for frame in local_frames
            if int(frame.get("visible_players") or 0) >= int(cfg.pose_point_min_visible_players)
            and set(frame.get("ends") or []) == {"near", "far"}
        ]
        two_sided_fraction = len(two_sided) / len(local_frames)
        relaxed_times = [
            float(frame["time"]) for frame in local_frames if frame.get("relaxed_sample")
        ]
        relaxed_fraction = len(relaxed_times) / len(local_frames)
        actor_evidence: dict[str, dict[str, dict[str, float | int]]] = {}
        end_ready: dict[str, float] = {}
        end_activity: dict[str, float] = {}
        for end in ("near", "far"):
            actors, ready_fraction, activity_fraction = _actor_window_evidence(
                local_frames, end, cfg
            )
            actor_evidence[end] = actors
            end_ready[end] = ready_fraction
            end_activity[end] = activity_fraction

        action_pairs: list[dict[str, Any]] = []
        for left, right in pairwise(local_events):
            gap = float(right["time"]) - float(left["time"])
            left_end, right_end = _event_end(left), _event_end(right)
            if (
                left_end in {"near", "far"}
                and right_end in {"near", "far"}
                and left_end != right_end
                and float(cfg.pose_live_response_min_s)
                <= gap
                <= float(cfg.pose_live_response_max_s)
            ):
                action_pairs.append(
                    {
                        "from": left_end,
                        "to": right_end,
                        "start": round(float(left["time"]), 3),
                        "response": round(float(right["time"]), 3),
                        "gap_s": round(gap, 3),
                    }
                )
        responses: list[dict[str, Any]] = []
        for event in local_events:
            found, response_time = _opponent_response(
                event, local_frames, window_start, window_end, cfg
            )
            if found:
                responses.append(
                    {
                        "event": event["event"],
                        "event_time": round(float(event["time"]), 3),
                        "event_end": _event_end(event),
                        "response_time": round(float(response_time), 3),
                    }
                )

        serve_events = [event for event in local_events if event["event"] == "serve"]
        seed_reasons: list[str] = []
        if serve_events:
            seed_reasons.append("ordered_baseline_serve_sequence")
        if action_pairs:
            seed_reasons.append("alternating_cross_court_actions")
        if responses and any(event["event"] == "stroke" for event in local_events):
            seed_reasons.append("stroke_then_opponent_reaction")
        two_sided_ready = bool(
            two_sided_fraction >= float(cfg.pose_live_min_two_sided_fraction)
            and all(
                end_ready[end] >= float(cfg.pose_live_min_end_ready_fraction)
                for end in ("near", "far")
            )
        )
        two_sided_activity = bool(
            all(
                end_activity[end] >= float(cfg.pose_live_min_end_activity_fraction)
                for end in ("near", "far")
            )
        )
        live_support = bool(
            two_sided_ready and (two_sided_activity or bool(local_events) or bool(responses))
        )
        between = bool(
            two_sided_fraction >= float(cfg.pose_live_min_two_sided_fraction)
            and relaxed_fraction >= float(cfg.pose_live_min_relaxed_fraction)
            and not local_events
            and not responses
        )
        if seed_reasons:
            state = "LIVE_SEED"
        elif live_support:
            state = "LIVE_SUPPORT"
        elif between:
            state = "BETWEEN_POINTS"
        else:
            state = "UNKNOWN"

        support_times = [
            float(frame["time"])
            for frame in local_frames
            if any(
                bool(player.get("ready"))
                or float(player.get("wrist_speed_body_s") or 0.0)
                >= float(cfg.pose_live_arm_activity_speed_body_s)
                for player in frame.get("players") or []
            )
        ]
        evidence_times = [float(event["time"]) for event in local_events]
        evidence_times.extend(support_times)
        output.append(
            {
                "time": round(centre, 3),
                "window": [round(max(0.0, window_start), 3), round(window_end, 3)],
                "state": state,
                "seed_reasons": seed_reasons,
                "two_sided_fraction": round(two_sided_fraction, 4),
                "relaxed_fraction": round(relaxed_fraction, 4),
                "end_ready_fraction": {end: round(value, 4) for end, value in end_ready.items()},
                "end_activity_fraction": {
                    end: round(value, 4) for end, value in end_activity.items()
                },
                "actors": actor_evidence if state == "LIVE_SEED" else {},
                "events": [
                    {
                        "time": round(float(event["time"]), 3),
                        "kind": event["event"],
                        "end": _event_end(event),
                        "actor_id": event.get("actor_id"),
                    }
                    for event in local_events
                ],
                "action_pairs": action_pairs,
                "responses": responses,
                "evidence_start": (round(min(evidence_times), 3) if evidence_times else None),
                "evidence_end": (round(max(evidence_times), 3) if evidence_times else None),
                "between_start": (
                    round(min(relaxed_times), 3) if between and relaxed_times else None
                ),
            }
        )
    return output


def _live_bouts(windows: Sequence[dict[str, Any]], cfg) -> list[dict[str, Any]]:
    """Decode LIVE bouts with asymmetric entry, continuation and exit rules."""
    bouts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_between: dict[str, Any] | None = None
    last_live_window_time: float | None = None
    for window in sorted(windows, key=lambda item: float(item["time"])):
        time_s = float(window["time"])
        state = window.get("state")
        if current is None:
            if state != "LIVE_SEED":
                continue
            start = float(window.get("evidence_start") or time_s)
            current = {
                "start": start,
                "end": float(window.get("evidence_end") or time_s),
                "live_windows": 1,
                "seed_reasons": set(window.get("seed_reasons") or []),
                "max_ready_fraction": max(
                    (float(value) for value in (window.get("end_ready_fraction") or {}).values()),
                    default=0.0,
                ),
            }
            last_live_window_time = time_s
            pending_between = None
            continue
        if state in {"LIVE_SEED", "LIVE_SUPPORT"}:
            current["end"] = max(float(current["end"]), float(window.get("evidence_end") or time_s))
            current["live_windows"] += 1
            current["seed_reasons"].update(window.get("seed_reasons") or [])
            current["max_ready_fraction"] = max(
                float(current["max_ready_fraction"]),
                max(
                    (float(value) for value in (window.get("end_ready_fraction") or {}).values()),
                    default=0.0,
                ),
            )
            last_live_window_time = time_s
            pending_between = None
            continue
        if state == "BETWEEN_POINTS":
            if pending_between is None:
                pending_between = window
            if time_s - float(pending_between["time"]) >= float(cfg.pose_between_min_s):
                transition = float(pending_between.get("between_start") or pending_between["time"])
                current["end"] = max(float(current["start"]), transition)
                current["end_source"] = "sustained_between_window"
                current["between_transition_time"] = transition
                current["seed_reasons"] = sorted(current["seed_reasons"])
                current["duration"] = float(current["end"]) - float(current["start"])
                bouts.append(current)
                current = None
                pending_between = None
                last_live_window_time = None
            continue
        # UNKNOWN is observation uncertainty, not evidence that the point ended.  It can
        # bridge a short occlusion but cannot keep a point alive indefinitely.
        pending_between = None
        if last_live_window_time is not None and time_s - last_live_window_time > float(
            cfg.pose_live_unknown_bridge_s
        ):
            current["end_source"] = "live_evidence_cessation"
            current["seed_reasons"] = sorted(current["seed_reasons"])
            current["duration"] = float(current["end"]) - float(current["start"])
            bouts.append(current)
            current = None
            last_live_window_time = None
    if current is not None:
        current["end_source"] = "live_evidence_cessation"
        current["seed_reasons"] = sorted(current["seed_reasons"])
        current["duration"] = float(current["end"]) - float(current["start"])
        bouts.append(current)
    return [bout for bout in bouts if float(bout["duration"]) > 0.0]


def _matching_bout(
    bouts: Sequence[dict[str, Any]],
    start: float,
    stop: float,
) -> dict[str, Any] | None:
    candidates = [
        bout for bout in bouts if float(bout["end"]) >= start and float(bout["start"]) <= stop
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            min(float(item["end"]), stop) - max(float(item["start"]), start),
            float(item["duration"]),
        ),
    )


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _point_hypotheses(
    timeline,
    serves: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    live_windows: Sequence[dict[str, Any]],
    duration: float,
    cfg,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    bouts = _live_bouts(live_windows, cfg)
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
            min(duration, strike + float(cfg.pose_live_search_s)),
        )
        if (
            returned is None
            and bout is None
            and float(selected.get("serve_sequence_score") or 0.0)
            < float(cfg.pose_serve_unreturned_min_score)
        ):
            continue
        setup_start = float(selected.get("setup_start", selected["point"][0]))
        live_end = float(bout["end"]) if bout is not None else strike
        hypotheses.append(
            {
                "kind": "serve",
                "start": setup_start,
                "live_start": strike,
                "live_end": live_end,
                "live_bout": bout,
                "server_end": selected.get("pose_server_end"),
                "serve": selected,
                "attempts": group,
                "classification": (
                    "confirmed_rally"
                    if returned is not None
                    else "serve_led_player_activity"
                    if bout is not None
                    else "unreturned_service_point"
                ),
            }
        )
        coverage_start = min(float(item.get("setup_start", item["point"][0])) for item in group)
        service_coverage.append((coverage_start, max(live_end, strike + 1.0)))

    sequences = _exchange_sequences(episodes, cfg)
    for bout in bouts:
        start, end = float(bout["start"]), float(bout["end"])
        if float(bout["duration"]) < float(cfg.pose_live_candidate_min_s):
            continue
        if any(_overlap((start, end), interval) > 0.5 for interval in service_coverage):
            continue
        local_sequences = [
            sequence
            for sequence in sequences
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
        hypotheses.append(
            {
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
            }
        )

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
        if left["kind"] == "live_state" and 0.0 <= gap <= float(
            cfg.pose_service_retry_state_max_gap_s
        ):
            superseded_ids.add(id(left))
    hypotheses = [item for item in hypotheses if id(item) not in superseded_ids]

    # Prefer a measured serve sequence when two hypotheses explain the same live bout.
    hypotheses.sort(key=lambda item: (float(item["start"]), item["kind"] != "serve"))
    deduplicated: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        interval = (float(hypothesis["live_start"]), float(hypothesis["live_end"]))
        duplicate_index = next(
            (
                index
                for index, prior in enumerate(deduplicated)
                if _overlap(interval, (float(prior["live_start"]), float(prior["live_end"]))) > 0.5
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(hypothesis)
        elif hypothesis["kind"] == "serve" and deduplicated[duplicate_index]["kind"] != "serve":
            deduplicated[duplicate_index] = hypothesis
    return sorted(deduplicated, key=lambda item: float(item["start"])), groups, bouts


def _state_intervals(windows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact window decisions for the persisted inspector payload."""
    intervals: list[dict[str, Any]] = []
    for window in windows:
        state = str(window.get("state") or "UNKNOWN")
        time_s = float(window["time"])
        if not intervals or intervals[-1]["state"] != state:
            intervals.append(
                {
                    "state": state,
                    "start": time_s,
                    "end": time_s,
                    "windows": 1,
                    "seed_reasons": list(window.get("seed_reasons") or []),
                }
            )
        else:
            intervals[-1]["end"] = time_s
            intervals[-1]["windows"] += 1
            intervals[-1]["seed_reasons"] = sorted(
                set(intervals[-1]["seed_reasons"]) | set(window.get("seed_reasons") or [])
            )
    for interval in intervals:
        interval["start"] = round(float(interval["start"]), 3)
        interval["end"] = round(float(interval["end"]), 3)
    return intervals


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
        if (
            not serve.get("accepted")
            and serve.get("serve_sequence_evidence")
            and float(serve.get("serve_sequence_score") or 0.0) >= 0.52
            and _has_return(serve, episodes, cfg)
        ):
            serve.update(
                accepted=True,
                serve_motion=True,
                acceptance_support="opposite_end_return_sequence",
            )

    live_windows = _classify_live_windows(timeline.frames, mutable_serves, episodes, cfg)
    hypotheses, service_groups, live_bouts = _point_hypotheses(
        timeline, mutable_serves, episodes, live_windows, duration, cfg
    )
    segments: list[Segment] = []
    points: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    endpoint_records: list[dict[str, Any]] = []

    for index, hypothesis in enumerate(hypotheses):
        next_start = (
            float(hypotheses[index + 1]["start"]) if index + 1 < len(hypotheses) else duration
        )
        point_start = float(hypothesis["start"])
        live_start = float(hypothesis["live_start"])
        live_end = min(float(hypothesis["live_end"]), next_start, duration)
        if hypothesis["kind"] == "serve":
            accepted, rejected = _alternating_actions(
                episodes,
                live_start,
                next_start,
                _opposite(hypothesis.get("server_end")),
                cfg,
            )
        else:
            accepted = [dict(item, accepted=True) for item in hypothesis.get("seed_actions") or []]
            rejected = []

        last_racket_action = max([live_start] + [float(item["time"]) for item in accepted])
        last_live_evidence = max(last_racket_action, live_end)
        bout = hypothesis.get("live_bout") or {}
        transition_time = bout.get("between_transition_time")
        if transition_time is not None:
            point_end = float(transition_time)
            endpoint_source = "backdated_sustained_between_window"
            endpoint_evidence = {
                "window_seconds": float(cfg.pose_live_window_s),
                "confirmation_seconds": float(cfg.pose_between_min_s),
                "backdated_to_first_relaxed_sample": True,
            }
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
        elif hypothesis["kind"] == "live_state" and hypothesis.get("live_bout") is None:
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
                point_start <= float(item["time"]) <= point_end for item in episodes
            ),
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
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "time": attempt["first_strike"],
                    "setup_start": attempt.get("setup_start"),
                    "server_id": attempt.get("actor_id"),
                    "server_team_id": attempt.get("team_id"),
                    "server_end": attempt.get("pose_server_end"),
                    "server_court_x_m": attempt.get("pose_server_court_x_m"),
                    "server_court_half": attempt.get("server_court_half"),
                    "disposition": (
                        "selected_playable_attempt"
                        if selected
                        else "superseded_fault_let_or_conflicting_candidate"
                    ),
                    "retry_relation": attempt.get("retry_relation"),
                }
            )
        selected_serve = hypothesis.get("serve")
        service_action = (
            [
                {
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
                }
            ]
            if selected_serve
            else []
        )
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
                "active_player_ids": sorted(
                    {str(item["actor_id"]) for item in actions if item.get("actor_id")}
                ),
            },
            "service_attempts": attempts,
            "actions": actions,
            "state_transitions": [
                {
                    "time": round(point_start, 3),
                    "from": "BETWEEN_POINTS",
                    "to": "SERVE_SETUP" if selected_serve else "LIVE_POINT",
                    "reason": hypothesis["classification"],
                },
                *(
                    [
                        {
                            "time": round(live_start, 3),
                            "from": "SERVE_SETUP",
                            "to": "LIVE_POINT",
                            "reason": "selected_service_attempt",
                        }
                    ]
                    if selected_serve
                    else []
                ),
                {
                    "time": round(point_end, 3),
                    "from": "LIVE_POINT",
                    "to": "BETWEEN_POINTS",
                    "reason": endpoint_source,
                },
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
        decision.update(
            {
                "service_attempts": attempts,
                "actions": actions,
                "state_transitions": point["state_transitions"],
            }
        )
        points.append(point)
        segments.append((point_start, point_end))
        endpoint_records.append(
            {
                "point_index": len(points) - 1,
                "presentation_time": round(point_end, 3),
                "last_racket_action_time": round(last_racket_action, 3),
                "last_live_evidence_time": round(last_live_evidence, 3),
                "visual_dead_signal": endpoint_source,
                "endpoint_confidence": endpoint_confidence,
                "evidence": endpoint_evidence,
                "tail_after_last_action_s": round(point_end - last_racket_action, 3),
                "unexplained_tail_s": round(unexplained_tail, 3),
            }
        )

    gaps = [
        max(0.0, segments[index + 1][0] - segments[index][1]) for index in range(len(segments) - 1)
    ]
    retention = (
        sum(max(0.0, end - start) for start, end in segments) / duration if duration > 0 else 0.0
    )
    median_gap = float(np.median(gaps)) if gaps else None
    zero_stroke_fraction = (
        sum(not point.get("actions") or len(point["actions"]) <= 1 for point in points)
        / len(points)
        if points
        else 0.0
    )
    violations = [
        record
        for record in endpoint_records
        if float(record["unexplained_tail_s"]) > float(cfg.pose_endpoint_max_unexplained_tail_s)
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
        "median_inter_point_gap_s": (round(median_gap, 4) if median_gap is not None else None),
        "zero_post_serve_stroke_fraction": round(zero_stroke_fraction, 4),
        "boundary_invariant_violations": len(violations),
        "thresholds": {
            "max_retention_fraction": cfg.pose_quality_max_retention_fraction,
            "max_median_gap_s": cfg.pose_quality_max_median_gap_s,
            "max_unexplained_tail_s": cfg.pose_endpoint_max_unexplained_tail_s,
            "warning_zero_stroke_fraction": cfg.pose_quality_max_zero_stroke_fraction,
        },
        "warnings": (
            ["most points lack an observed post-serve racket action"]
            if zero_stroke_fraction > float(cfg.pose_quality_max_zero_stroke_fraction)
            else []
        ),
        "reason": (
            "point windows violated evidence-bounded endpoint invariants"
            if violations
            else "candidate windows tiled almost the entire source"
            if suspicious_tiling
            else None
        ),
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
        for point in points
        for item in point.get("actions") or []
        if item.get("action") != "serve"
    }
    episode_records = []
    for episode in episodes:
        item = dict(episode)
        item["accepted"] = (
            str(item.get("actor_id")),
            float(item["time"]),
        ) in accepted_keys
        if not item["accepted"]:
            item["rejection_reason"] = "not_assigned_to_constrained_state_path"
        episode_records.append(item)

    reports = {
        "serve_pose": {
            "status": "used"
            if any(item.get("accepted") for item in mutable_serves)
            else "no_confirmed_serves",
            "backend": "shared_rtmpose_coco17_timeline",
            "model_scope": "all_tracked_match_players_on_both_court_ends",
            "proposals": len(mutable_serves),
            "confirmed": sum(bool(item.get("accepted")) for item in mutable_serves),
            "serve_times": [
                item["first_strike"] for item in mutable_serves if item.get("accepted")
            ],
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
        "live_windows": {
            "status": "used" if live_windows else "no_windows",
            "backend": "identity_preserving_pose_window_state_machine",
            "window_seconds": float(cfg.pose_live_window_s),
            "sample_fps": float(cfg.pose_timeline_fps),
            "stride_seconds": round(1.0 / float(cfg.pose_timeline_fps), 4),
            "window_count": len(live_windows),
            "state_counts": {
                state: sum(window.get("state") == state for window in live_windows)
                for state in ("LIVE_SEED", "LIVE_SUPPORT", "BETWEEN_POINTS", "UNKNOWN")
            },
            "state_intervals": _state_intervals(live_windows),
            # Persist one detailed representative per seed run.  Keeping all twelve
            # overlapping copies of the same event would bloat long-match sidecars.
            "seed_windows": [
                {
                    key: window.get(key)
                    for key in (
                        "time",
                        "window",
                        "seed_reasons",
                        "events",
                        "action_pairs",
                        "responses",
                        "two_sided_fraction",
                        "end_ready_fraction",
                        "end_activity_fraction",
                        "actors",
                    )
                }
                for index, window in enumerate(live_windows)
                if window.get("state") == "LIVE_SEED"
                and (index == 0 or live_windows[index - 1].get("state") != "LIVE_SEED")
            ],
            "entry_policy": (
                "ordered serve or stroke/opponent response; generic motion cannot start"
            ),
            "continuation_policy": (
                "both court ends ready with per-player activity; short unknown gaps only"
            ),
            "exit_policy": ("sustained visually relaxed two-sided windows with no tennis action"),
            "audio_speech_used": False,
            "audio_speech_reason": (
                "mixed-court speech cannot be attributed to target-court players"
            ),
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
