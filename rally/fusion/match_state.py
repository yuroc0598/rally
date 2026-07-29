"""Serve-event validation and conservative tennis match-phase inference.

Every match point starts with a serve. Audio, receiver posture, static player formation,
and deuce/ad alternation cannot prove that event: ball handoffs can create racket-like
impacts while players wait in normal positions. A candidate is therefore confirmed by a
near-server overhead pose paired with a stationary baseline setup, or by sustained in-court
TrackNet motion with meaningful vertical travel around an early impact.

Explicit ``match`` mode applies the rule to every candidate. ``auto`` first requires a run
of independently confirmed serves and applies it only between the first/last anchor of that
run, leaving unstructured warm-up outside the inferred match phase untouched.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..signals.player import ServeSetupObservation

Segment = Tuple[float, float]


def _effective_strike_count(point: Segment, onsets: np.ndarray, echo_s: float) -> int:
    strikes = np.sort(np.asarray(onsets, dtype=float))
    strikes = strikes[(strikes >= point[0] - 1e-9) & (strikes <= point[1] + 1e-9)]
    if strikes.size == 0:
        return 0
    count = 1
    last = float(strikes[0])
    for strike in strikes[1:]:
        if float(strike) - last >= echo_s:
            count += 1
            last = float(strike)
    return count


def _trim_sparse_trailing_contacts(
    point: Segment, group_end: float, onsets: np.ndarray, cfg,
) -> tuple[float, bool]:
    """Trim a dense exchange followed only by sparse pickup/reset contacts.

    A long interval inside a rally is valid when dense hitting resumes. We trim only when
    at least two remaining contacts stay sparse after a well-established dense prefix.
    Looking one tail-buffer beyond the candidate also catches a transient that fell just
    outside a strike-bounded point due to rounding.
    """
    strikes = np.sort(np.asarray(onsets, dtype=float))
    strikes = strikes[
        (strikes >= point[0] - 1e-9)
        & (strikes <= group_end + cfg.landing_tail_s + 1e-9)
    ]
    if strikes.size < cfg.min_rally_strikes + 5:
        return float(group_end), False
    effective = [float(strikes[0])]
    for strike in strikes[1:]:
        if float(strike) - effective[-1] >= cfg.echo_collapse_s:
            effective.append(float(strike))
    if len(effective) < cfg.min_rally_strikes + 5:
        return float(group_end), False
    values = np.asarray(effective, dtype=float)
    gaps = np.diff(values)
    min_prefix = cfg.min_rally_strikes + 3
    for split in range(min_prefix, values.size - 1):
        dense_gaps = gaps[:split - 1]
        dense_median = float(np.median(dense_gaps))
        reset_gap = max(float(cfg.merge_gap_s), 2.5 * dense_median)
        if gaps[split - 1] < reset_gap:
            continue
        suffix_gaps = gaps[split:]
        if suffix_gaps.size and np.all(suffix_gaps >= reset_gap):
            return min(
                float(group_end), float(values[split - 1] + cfg.landing_tail_s)), True
    return float(group_end), False


def _auto_match_phases(
    observations: Sequence[ServeSetupObservation], strike_counts: Sequence[int], cfg,
) -> list[tuple[int, int]]:
    """Runs bounded by several confirmed serves; long dead gaps split recordings."""
    # Far-side pose is often too small. Without a completed ball check, auto mode must
    # abstain rather than interpreting only near-side serves as the entire match.
    if not observations or not all(observation.ball_checked for observation in observations):
        return []
    anchors = [
        index for index, observation in enumerate(observations)
        if _independently_confirmed_serve(
            observation, strike_counts[index], cfg)
    ]
    if not anchors:
        return []
    groups: list[list[int]] = [[anchors[0]]]
    for index in anchors[1:]:
        prior = groups[-1][-1]
        if (observations[index].first_strike - observations[prior].first_strike
                > cfg.match_phase_max_gap_s):
            groups.append([index])
        else:
            groups[-1].append(index)
    return [
        (group[0], group[-1]) for group in groups
        if len(group) >= cfg.match_auto_min_serve_anchors
    ]


def _aligned_pose_strikes(observation: ServeSetupObservation) -> tuple[float, ...]:
    return tuple(sorted({
        float(pose_strike)
        for pose_strike in observation.overhead_strikes
        for position_strike in observation.position_setup_strikes
        if abs(pose_strike - position_strike) <= 1e-6
    }))


def _independently_confirmed_serve(
    observation: ServeSetupObservation, effective_strikes: int, cfg,
) -> bool:
    """Apply a stricter gate to an uncorroborated one-impact ball hypothesis."""
    if _aligned_pose_strikes(observation):
        return True
    if not observation.ball_serve_evidence:
        return False
    corroborated = bool(
        observation.serve_motion
        or observation.position_setup_evidence
        or effective_strikes >= 2
    )
    return bool(
        corroborated
        or observation.ball_coverage
        >= cfg.match_ball_min_single_strike_coverage
    )


def _serve_strength(observation: ServeSetupObservation) -> tuple[float, ...]:
    """Comparable evidence rank used only to select an anchor within one point group."""
    aligned = bool(_aligned_pose_strikes(observation))
    pose_ball = bool(observation.serve_motion and observation.ball_serve_evidence)
    ball_position = bool(
        observation.ball_serve_evidence and observation.position_setup_evidence)
    return (
        float(aligned),
        float(pose_ball),
        float(ball_position),
        float(observation.ball_serve_evidence),
        float(observation.serve_motion),
        float(observation.position_setup_evidence),
        float(observation.ball_coverage * observation.ball_vertical_span),
        float(observation.position_score),
    )


def _serve_contact(observation: ServeSetupObservation) -> float:
    aligned = _aligned_pose_strikes(observation)
    if aligned:
        return float(aligned[-1])
    if observation.serve_motion and observation.overhead_strikes:
        return float(observation.overhead_strikes[-1])
    if observation.ball_best_strike is not None:
        # Once this short window is serve-confirmed, its first accepted impact is the
        # conservative contact anchor.  A later bounce/return can have the highest raw
        # TrackNet score but must not trim away serve preparation.
        return float(min(observation.first_strike, observation.ball_best_strike))
    if observation.position_best_strike is not None:
        return float(observation.position_best_strike)
    return float(observation.first_strike)


def _fragment_groups(
    points: Sequence[Segment], observations: Sequence[ServeSetupObservation],
    strike_counts: Sequence[int], cfg,
) -> list[list[int]]:
    """Group adjacent fragments that occupy the same service-side state.

    A fault/retry and a rally with one missed contact can both contain multi-second audio
    gaps.  Consecutive tennis points should alternate side, so a short same-side run is one
    logical-point hypothesis.  Unknown side is allowed to inherit only across the same
    short window; it never joins distant candidates.
    """
    if not points:
        return []
    groups: list[list[int]] = [[0]]
    for index in range(1, len(points)):
        prior = groups[-1][-1]
        gap = float(points[index][0] - points[prior][1])
        side = observations[index].side
        known_sides = {
            observations[member].side for member in groups[-1]
            if observations[member].side is not None
        }
        # An unknown-side *dynamic* event can belong to the current attempt.  Static/noise
        # fragments may not bridge two opposite known sides or extend a completed point.
        direct_exchange_continuation = bool(
            0.0 <= gap <= cfg.merge_gap_s
            and strike_counts[prior] >= 2
            and strike_counts[index] >= 2
            and observations[prior].ball_ordered_evidence
            and observations[index].ball_ordered_evidence
        )
        if side is None:
            compatible_side = bool(
                _independently_confirmed_serve(
                    observations[index], strike_counts[index], cfg)
                or observations[index].serve_motion
                or observations[index].ball_ordered_evidence)
        else:
            compatible_side = not known_sides or side in known_sides
        if (0.0 <= gap <= cfg.match_fragment_merge_gap_s
                and (compatible_side or direct_exchange_continuation)):
            groups[-1].append(index)
        else:
            groups.append([index])

    # A fault/let retry can take longer than an ordinary missed-contact gap.  Same-side
    # adjacent groups are still one logical point; true consecutive match points alternate.
    merged: list[list[int]] = []
    for group in groups:
        if not merged:
            merged.append(list(group))
            continue
        prior_group = merged[-1]
        prior_sides = {
            observations[member].side for member in prior_group
            if observations[member].side is not None
        }
        sides = {
            observations[member].side for member in group
            if observations[member].side is not None
        }
        gap = float(points[group[0]][0] - points[prior_group[-1]][1])
        same_known_side = (
            len(prior_sides) == 1 and prior_sides == sides
        )
        prior_dynamic = any(
            observations[member].confirmed_serve
            or observations[member].serve_motion
            or observations[member].ball_ordered_evidence
            for member in prior_group
        )
        dynamic = any(
            observations[member].confirmed_serve
            or observations[member].serve_motion
            or observations[member].ball_ordered_evidence
            for member in group
        )
        if (same_known_side and prior_dynamic and dynamic
                and cfg.match_attempt_merge_min_gap_s <= gap
                <= cfg.match_attempt_merge_gap_s):
            prior_group.extend(group)
        else:
            merged.append(list(group))
    return merged


def _potential_ball_serve(observation: ServeSetupObservation, cfg) -> bool:
    """Near-threshold ordered motion may combine with setup at group level."""
    return bool(
        observation.ball_serve_evidence
        or (
            observation.ball_ordered_evidence
            and observation.ball_coverage >= 0.8 * cfg.match_ball_min_coverage
            and observation.ball_vertical_span >= cfg.match_ball_min_vertical_span
        )
    )


def _group_has_confirmed_serve(
    indices: Sequence[int], observations: Sequence[ServeSetupObservation],
    strike_counts: Sequence[int], cfg,
) -> bool:
    if any(_independently_confirmed_serve(
            observations[index], strike_counts[index], cfg) for index in indices):
        return True
    positions = [
        observations[index].position_best_strike for index in indices
        if observations[index].position_setup_evidence
        and observations[index].position_best_strike is not None
    ]
    balls = [
        _serve_contact(observations[index]) for index in indices
        if _potential_ball_serve(observations[index], cfg)
    ]
    return any(
        0.0 <= ball_time - position_time <= cfg.match_attempt_merge_gap_s
        for position_time in positions for ball_time in balls
    )


def _select_serve_member(
    indices: Sequence[int], observations: Sequence[ServeSetupObservation],
    points: Sequence[Segment], cfg,
) -> int:
    """Choose the serve attempt that anchors one merged logical point.

    Multiple independently aligned overhead+position events represent a fault/let retry,
    so the last is a valid canonical start.  Otherwise select the strongest event, with
    pose+ball evidence ahead of a static-position/ball coincidence.
    """
    aligned = [index for index in indices if _aligned_pose_strikes(observations[index])]
    if len(aligned) >= 2:
        return max(aligned, key=lambda index: _serve_contact(observations[index]))

    # Two ball-ordered fragments separated by less than the ordinary decoder merge gap are
    # one exchange even if a mid-rally pose sample assigns the opposite screen side. The
    # later fragment cannot be a new tennis point so soon; anchor in the prefix. Multiple
    # independently aligned pose+position serves above remain a real fault/let retry.
    continuation_starts = [
        current for prior, current in zip(indices, indices[1:])
        if 0.0 <= points[current][0] - points[prior][1] <= cfg.merge_gap_s
        and observations[prior].ball_ordered_evidence
        and observations[current].ball_ordered_evidence
    ]
    if continuation_starts:
        prefix = [index for index in indices if index < min(continuation_starts)]
        if prefix:
            return max(
                prefix,
                key=lambda index: (_serve_strength(observations[index]), -index),
            )
    pose_ball = [
        index for index in indices
        if observations[index].serve_motion
        and _potential_ball_serve(observations[index], cfg)
    ]
    if pose_ball:
        return max(pose_ball, key=lambda index: _serve_contact(observations[index]))

    # Audio can split a real serve into a compact serve-contact fragment immediately
    # followed by the long exchange.  The exchange itself also has strong ball motion, but
    # anchoring to its first return starts the point late.  Prefer the adjacent compact
    # serve fragment; the direct-gap requirement prevents an older feed/noise fragment
    # elsewhere in a large merged group from becoming the anchor.
    long_exchange = [
        index for index in indices if points[index][1] - points[index][0] >= 5.0
    ]
    compact_before_exchange = [
        index for index in indices
        if _potential_ball_serve(observations[index], cfg)
        and any(
            index < exchange
            and 0.0 <= points[exchange][0] - points[index][1]
            <= cfg.match_fragment_merge_gap_s
            for exchange in long_exchange
        )
    ]
    if compact_before_exchange:
        return max(compact_before_exchange)

    # A long same-side separation means the later event is the retry/actual serve.  Use the
    # latest dynamic serve hypothesis: a pickup/feed can combine weak ball motion with a
    # stationary formation and out-rank a later real serve under the generic strength
    # tuple, but it must never pull the point start back across the ensuing reset.
    long_retry = any(
        points[current][0] - points[prior][1] > cfg.match_fragment_merge_gap_s
        for prior, current in zip(indices, indices[1:])
    )
    if long_retry:
        retry_serves = [
            index for index in indices
            if observations[index].confirmed_serve
            or observations[index].serve_motion
            or _potential_ball_serve(observations[index], cfg)
        ]
        return max(
            retry_serves or list(indices),
            key=lambda index: (_serve_contact(observations[index]), index),
        )

    positions = [
        float(observations[index].position_best_strike) for index in indices
        if observations[index].position_setup_evidence
        and observations[index].position_best_strike is not None
    ]
    paired_ball = [
        index for index in indices
        if _potential_ball_serve(observations[index], cfg)
        and any(
            0.0 <= _serve_contact(observations[index]) - position_time
            <= cfg.match_attempt_merge_gap_s
            for position_time in positions
        )
    ]
    if paired_ball:
        # A compact contact immediately before a long exchange is the serve even when
        # the rally fragment has marginally higher raw TrackNet coverage.
        before_exchange = [
            index for index in paired_ball
            if any(index < exchange for exchange in long_exchange)
        ]
        if before_exchange:
            return max(before_exchange)
        return max(
            paired_ball,
            key=lambda index: (
                _serve_strength(observations[index]),
                points[index][1] - points[index][0], index),
        )
    return max(
        indices,
        key=lambda index: (_serve_strength(observations[index]), -index),
    )


def validate_match_sequence(
    points: Sequence[Segment],
    onsets: np.ndarray,
    observations: Sequence[ServeSetupObservation],
    cfg,
    *,
    protected_indices: Iterable[int] = (),
) -> tuple[List[Segment], dict]:
    """Reject candidates inside match play that lack a confirmed serve event."""
    points = [(float(start), float(end)) for start, end in points]
    if len(points) != len(observations):
        raise ValueError("match observations must align one-to-one with points")

    strike_counts = [
        _effective_strike_count(point, onsets, cfg.echo_collapse_s) for point in points
    ]
    if cfg.play_mode == "match":
        phases = [(0, len(points) - 1)] if points else []
    else:
        phases = _auto_match_phases(observations, strike_counts, cfg)
    required = {
        index for start, end in phases for index in range(start, end + 1)
    }

    groups = _fragment_groups(points, observations, strike_counts, cfg)
    dropped: set[int] = set()
    reasons: list[dict] = []
    group_records: list[dict] = []
    kept: List[Segment] = []
    prior_end = 0.0
    for group_index, indices in enumerate(groups):
        inside_phase = any(index in required for index in indices)
        confirmed = _group_has_confirmed_serve(
            indices, observations, strike_counts, cfg)
        if inside_phase and not confirmed:
            dropped.update(indices)
            representative = max(
                indices, key=lambda index: _serve_strength(observations[index]))
            observation = observations[representative]
            reasons.append({
                "index": representative,
                "member_indices": list(indices),
                "reason_code": "missing_confirmed_serve",
                "reason": ("logical match-point hypothesis has neither an overhead action "
                           "in a stationary baseline setup nor ordered serve-ball evidence"),
                "strike_count": sum(strike_counts[index] for index in indices),
                "pose_overhead_frames": observation.overhead_frames,
                "pose_overhead_max_ratio": round(observation.overhead_max_ratio, 4),
                "position_checked": observation.position_checked,
                "position_setup_evidence": observation.position_setup_evidence,
                "position_score": round(observation.position_score, 4),
                "position_server_end": observation.position_server_end,
                "position_player_tracks": observation.position_player_tracks,
                "position_stable_tracks": observation.position_stable_tracks,
                "position_stable_fraction": round(
                    observation.position_stable_fraction, 4),
                "ball_checked": observation.ball_checked,
                "ball_coverage": round(observation.ball_coverage, 4),
                "ball_vertical_span": round(observation.ball_vertical_span, 4),
            })
            group_records.append({
                "group_index": group_index, "member_indices": list(indices),
                "decision": "drop", "inside_match_phase": inside_phase,
            })
            continue
        serve_member = _select_serve_member(indices, observations, points, cfg)
        contact = _serve_contact(observations[serve_member])
        group_start = min(points[index][0] for index in indices)
        dynamic_members = [
            index for index in indices
            if _independently_confirmed_serve(
                observations[index], strike_counts[index], cfg)
            or observations[index].ball_ordered_evidence
        ]
        terminal_index = max(dynamic_members) if dynamic_members else max(indices)
        group_end = max(
            points[index][1] for index in indices if index <= terminal_index)
        group_end, sparse_tail_trimmed = _trim_sparse_trailing_contacts(
            points[terminal_index], group_end, onsets, cfg)
        # Recover serve preparation from the selected contact.  Unlike fixed audio
        # clustering this may trim score calls before the serve as well as extend a late
        # rally-only fragment back to the actual point setup.
        recovered_start = max(0.0, contact - cfg.match_point_start_preroll_s)
        start = recovered_start if (confirmed or inside_phase) else group_start
        start = max(start, prior_end)
        if group_end <= start:
            start = min(group_start, max(0.0, group_end - 1e-3))
        segment = (float(start), float(group_end))
        # A candidate wholly overlapping the preceding point can be clamped into a tiny
        # tail and used to survive as a separate output point. Minimum duration must be
        # re-applied after sequence-aware clamping, not only before this decoder.
        if segment[1] - segment[0] < cfg.min_rally_s:
            dropped.update(indices)
            reasons.append({
                "index": serve_member,
                "member_indices": list(indices),
                "reason_code": "clamped_below_minimum_duration",
                "reason": "candidate became only a short tail after prior-point overlap",
                "duration": round(segment[1] - segment[0], 3),
            })
            group_records.append({
                "group_index": group_index, "member_indices": list(indices),
                "decision": "drop", "inside_match_phase": inside_phase,
                "reason_code": "clamped_below_minimum_duration",
            })
            continue
        kept.append(segment)
        prior_end = float(group_end)
        group_records.append({
            "group_index": group_index,
            "member_indices": list(indices),
            "decision": "keep",
            "inside_match_phase": inside_phase,
            "serve_member_index": serve_member,
            "serve_contact": round(contact, 3),
            "output": [round(segment[0], 3), round(segment[1], 3)],
            "sparse_tail_trimmed": sparse_tail_trimmed,
        })

    serialised = []
    for index, observation in enumerate(observations):
        item = asdict(observation)
        aligned_pose_setup = bool(_aligned_pose_strikes(observation))
        containing_group = next(
            record for record in group_records if index in record["member_indices"])
        item.update({
            "index": index,
            "effective_strikes": strike_counts[index],
            "serve_evidence_sources": [
                source for source, present in (
                    ("stationary_baseline_setup", observation.position_setup_evidence),
                    ("overhead_pose_with_position_setup", aligned_pose_setup),
                    ("tracknet_ball_motion", observation.ball_serve_evidence),
                ) if present
            ],
            "inside_match_phase": index in required,
            "decision": ("drop" if index in dropped else
                         ("merge" if len(containing_group["member_indices"]) > 1
                          else "keep")),
            "logical_group": containing_group["group_index"],
        })
        serialised.append(item)

    protected = sorted(set(int(index) for index in protected_indices))
    stage = {
        "status": "used" if phases else "abstained",
        "mode": cfg.play_mode,
        "rule": "confirmed_serve_event_required_inside_match_phase",
        "match_phases": [
            {"start_index": start, "end_index": end} for start, end in phases
        ],
        "logical_groups": group_records,
        "side_sequence": [observation.side for observation in observations],
        "observations": serialised,
        "dropped": reasons,
        "trajectory_accepted_not_exempt": protected,
        "kept_points": len(kept),
    }
    if not phases:
        stage["reason"] = (
            "auto mode did not observe enough ball-checked serve anchors to infer match play"
        )
    return kept, stage
