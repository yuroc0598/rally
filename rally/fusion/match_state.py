"""Serve-event validation and conservative tennis match-phase inference.

Every match point starts with a serve. Static player formation and deuce/ad alternation
cannot prove that event: ball handoffs can create racket-like impacts while players wait
in normal positions. Dynamic evidence constrained to the selected court can: a stable
receiver beginning to react, a repeated overhead pose, or sustained TrackNet serve motion.

Explicit ``match`` mode applies the rule to every candidate. ``auto`` first requires a run
of independently confirmed serves and applies it only between the first/last anchor of that
run, leaving unstructured warm-up outside the inferred match phase untouched.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..domain.observations import ServeSetupObservation
from .points import effective_strike_times

Segment = Tuple[float, float]


def _effective_strike_count(point: Segment, onsets: np.ndarray, echo_s: float) -> int:
    strikes = np.sort(np.asarray(onsets, dtype=float))
    strikes = strikes[(strikes >= point[0] - 1e-9) & (strikes <= point[1] + 1e-9)]
    return len(effective_strike_times(strikes, echo_s))


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
    effective = effective_strike_times(strikes, cfg.echo_collapse_s)
    if len(effective) < cfg.min_rally_strikes + 5:
        return float(group_end), False
    values = np.asarray(effective, dtype=float)
    gaps = np.diff(values)
    # Crowd/applause or rapid speech can create a dense acoustic burst after play. Tennis
    # racket exchanges cannot sustain sub-650ms contacts across many consecutive events.
    # If such a burst follows a visible reset-sized pause, retain the last plausible
    # contact plus the configured real-footage buffer.
    for split in range(cfg.min_rally_strikes + 1, values.size - 5):
        if gaps[split - 1] < cfg.merge_gap_s:
            continue
        dense_suffix = gaps[split:]
        if dense_suffix.size >= 5 and float(np.median(dense_suffix)) < 0.65:
            return min(
                float(group_end), float(values[split - 1] + cfg.landing_tail_s)), True
    min_prefix = cfg.min_rally_strikes + 3
    for split in range(min_prefix, values.size - 1):
        dense_gaps = gaps[:split - 1]
        dense_median = float(np.median(dense_gaps))
        reset_gap = max(float(cfg.merge_gap_s), 2.0 * dense_median)
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
    """Infer candidate-contiguous match runs around several confirmed serves.

    The anchors establish that the recording contains match play.  They are not exact
    phase boundaries: a quiet far-side serve or a false pickup hypothesis can sit just
    before the first anchor or after the last one.  Extend through adjacent candidates
    separated by at most an ordinary attempt/reset gap, while retaining a real dead gap
    as the warm-up/match boundary.
    """
    # Far-side pose is often too small. Without a completed ball check, auto mode must
    # abstain rather than interpreting only near-side serves as the entire match.
    if not observations or not all(observation.ball_checked for observation in observations):
        return []
    anchors = [
        index for index, observation in enumerate(observations)
        if _independently_confirmed_serve(
            observation, strike_counts[index], cfg)
        # Repeated target-court receiver reactions are useful for locating a match
        # phase, but each eventual point still needs independent/group-level service
        # corroboration below.  This keeps phase discovery recall-oriented without
        # publishing reaction-only pickups as rallies.
        or _target_court_receiver_reaction(observation)
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
    phases = [
        (group[0], group[-1]) for group in groups
        if len(group) >= cfg.match_auto_min_serve_anchors
    ]
    expanded: list[tuple[int, int]] = []
    for phase_start, phase_end in phases:
        while phase_start > 0:
            gap = (
                observations[phase_start].point[0]
                - observations[phase_start - 1].point[1]
            )
            if gap > cfg.match_attempt_merge_gap_s:
                break
            phase_start -= 1
        while phase_end + 1 < len(observations):
            gap = (
                observations[phase_end + 1].point[0]
                - observations[phase_end].point[1]
            )
            if gap > cfg.match_attempt_merge_gap_s:
                break
            phase_end += 1
        if expanded and phase_start <= expanded[-1][1] + 1:
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], phase_end))
        else:
            expanded.append((phase_start, phase_end))

    # Anchor gaps can be long when the far-side server is small or TrackNet is fragmented.
    # That must not turn the middle of one continuously recorded match back into permissive
    # warm-up mode.  Merge established phases when the intervening *candidate stream* is
    # continuous; retain a real candidate-free gap as a recording/session boundary.
    continuous: list[tuple[int, int]] = []
    for phase_start, phase_end in expanded:
        if not continuous:
            continuous.append((phase_start, phase_end))
            continue
        prior_start, prior_end = continuous[-1]
        intervening_gaps = [
            observations[index + 1].point[0] - observations[index].point[1]
            for index in range(prior_end, phase_start)
        ]
        if (intervening_gaps
                and max(intervening_gaps) <= cfg.match_phase_max_gap_s):
            continuous[-1] = (prior_start, phase_end)
        else:
            continuous.append((phase_start, phase_end))
    return continuous


def _aligned_pose_strikes(observation: ServeSetupObservation) -> tuple[float, ...]:
    return tuple(sorted({
        float(pose_strike)
        for pose_strike in observation.overhead_strikes
        for position_strike in observation.position_setup_strikes
        if abs(pose_strike - position_strike) <= 1e-6
    }))


def _target_court_receiver_reaction(observation: ServeSetupObservation) -> bool:
    return bool(
        observation.target_court_filtered
        and observation.receiver_reaction_evidence
        and observation.receiver_reaction_time is not None
        # The reaction has already been source-window validated by the pipeline. It may
        # precede an audio-opened candidate when the quiet serve itself was missed.
        and 0.0 <= observation.receiver_reaction_time <= observation.point[1] + 1e-9
    )


def _corroborated_receiver_reaction(observation: ServeSetupObservation) -> bool:
    """A receiver movement is useful only beside independent serve structure.

    Walking, a ball feed, or motion on a neighboring court can all produce the same
    short reaction signal.  Stationary baseline setup is the independent visual cue;
    a TrackNet serve remains eligible through its own policy below.
    """
    return bool(
        _target_court_receiver_reaction(observation)
        and observation.position_setup_evidence
        and float(observation.receiver_reaction_time) >= observation.point[0] - 1.0
    )


def _extend_through_adjacent_terminal_contacts(
    group_end: float, onsets: np.ndarray, next_group_start: float,
    max_contact_gap: float, min_contacts: int, cfg,
) -> float:
    """Recover a short target exchange tail lost at a proposal ownership boundary."""
    original_end = float(group_end)
    limit = min(
        float(next_group_start), original_end + cfg.match_fragment_merge_gap_s)
    extended = original_end
    accepted = 0
    for strike in np.sort(np.asarray(onsets, dtype=float)):
        strike = float(strike)
        if strike <= original_end + 1e-9:
            continue
        if strike > limit or strike > extended + max_contact_gap:
            break
        accepted += 1
        extended = min(limit, max(extended, strike + cfg.landing_tail_s))
    return float(extended if accepted >= min_contacts else original_end)


def _robust_target_court_overhead(observation: ServeSetupObservation) -> bool:
    return bool(
        observation.target_court_filtered
        and observation.serve_motion
        and observation.overhead_frames >= 2
    )


def _stable_baseline_formation(observation: ServeSetupObservation, cfg) -> bool:
    """Allow one fragmented player track while retaining baseline/stability checks."""
    return bool(
        observation.position_setup_evidence
        or (
            observation.position_checked
            and observation.position_server_end is not None
            and observation.position_server_span is not None
            and observation.position_server_span <= cfg.match_position_max_span
            and observation.position_stable_fraction
            >= cfg.match_position_support_min_stable_fraction
            and observation.position_score >= cfg.match_position_support_min_score
        )
    )


def _supporting_baseline_formation(observation: ServeSetupObservation, cfg) -> bool:
    """Weaker formation cue that is valid only beside reaction/exchange evidence."""
    return bool(
        observation.position_setup_evidence
        or (
            observation.position_checked
            and observation.position_server_end is not None
            and observation.position_server_span is not None
            and observation.position_server_span <= cfg.match_position_max_span
            and observation.position_stable_fraction
            >= cfg.match_position_support_min_stable_fraction
            and observation.position_score >= cfg.match_position_min_score
        )
    )


def _independently_confirmed_serve(
    observation: ServeSetupObservation, effective_strikes: int, cfg,
) -> bool:
    """Apply a stricter gate to an uncorroborated one-impact ball hypothesis."""
    if observation.learned_serve_checked:
        return observation.learned_serve_evidence
    if (_aligned_pose_strikes(observation)
            or _corroborated_receiver_reaction(observation)
            or (
                _target_court_receiver_reaction(observation)
                and _stable_baseline_formation(observation, cfg)
            )
            or (
                _target_court_receiver_reaction(observation)
                and _supporting_baseline_formation(observation, cfg)
                and effective_strikes >= 2
            )
            or _robust_target_court_overhead(observation)):
        return True
    if (observation.position_setup_evidence
            and effective_strikes >= 2
            and _supporting_target_ball_motion(observation)):
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
        float(observation.learned_serve_evidence),
        float(observation.learned_serve_score or 0.0),
        float(aligned),
        float(_robust_target_court_overhead(observation)),
        float(_corroborated_receiver_reaction(observation)),
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
    if (_corroborated_receiver_reaction(observation)
            and float(observation.receiver_reaction_time) >= observation.point[0]):
        return float(observation.receiver_reaction_time)
    if observation.ball_best_strike is not None:
        # Once this short window is serve-confirmed, its first accepted impact is the
        # conservative contact anchor.  A later bounce/return can have the highest raw
        # TrackNet score but must not trim away serve preparation.
        return float(min(observation.first_strike, observation.ball_best_strike))
    if observation.position_best_strike is not None:
        return float(observation.position_best_strike)
    return float(observation.first_strike)


def _direct_pose_service(observation: ServeSetupObservation) -> bool:
    """Whether target-court body motion directly identifies a service action."""
    return bool(
        _aligned_pose_strikes(observation)
        or _robust_target_court_overhead(observation)
    )


def _service_attempt_contacts(
    index: int,
    observation: ServeSetupObservation,
    effective_strikes: int,
    cfg,
) -> list[tuple[float, int, str]]:
    """Return directly supported service contacts for retry detection.

    Multiple aligned overhead/setup contacts inside one candidate are retained separately.
    Strong overhead/ball evidence is direct; formation/reaction and marginal overhead cues
    are supporting and require the longer retry interval before they can form a pair.
    """
    aligned = _aligned_pose_strikes(observation)
    if aligned:
        return [(float(contact), index, "direct") for contact in aligned]
    if _robust_target_court_overhead(observation):
        contacts = observation.overhead_strikes or (_serve_contact(observation),)
        return [(float(contact), index, "direct") for contact in contacts]
    if (observation.ball_serve_evidence
            and _independently_confirmed_serve(
                observation, effective_strikes, cfg)):
        return [(_serve_contact(observation), index, "direct")]
    if _stable_baseline_formation(observation, cfg):
        if (_corroborated_receiver_reaction(observation)
                or observation.ball_ordered_evidence):
            return [(_serve_contact(observation), index, "supporting")]
    if (observation.target_court_filtered
            and observation.serve_motion
            and observation.overhead_strikes):
        return [(_serve_contact(observation), index, "supporting")]
    return []


def _detected_retry_serve(
    indices: Sequence[int], observations: Sequence[ServeSetupObservation],
    strike_counts: Sequence[int], cfg,
) -> Optional[tuple[int, float, float, tuple[float, ...], int]]:
    """Identify a later service attempt without claiming to classify the fault itself.

    The observable fact is a second direct service action before a completed point. Reliable
    opposite service-side labels reject the retry interpretation. The returned tuple is
    ``(member_index, retry_contact, prior_contact, all_attempt_contacts,
    prior_member_index)``.
    """
    attempts = sorted(
        (
            attempt
            for index in indices
            for attempt in _service_attempt_contacts(
                index, observations[index], strike_counts[index], cfg)
        ),
        key=lambda attempt: (attempt[0], attempt[1]),
    )
    distinct: list[tuple[float, int, str]] = []
    for contact, index, strength in attempts:
        if distinct and contact - distinct[-1][0] <= cfg.echo_collapse_s:
            # Duplicate acoustic/pose assignments around one racket contact are one attempt.
            prior_strength = distinct[-1][2]
            if strength == "direct" or prior_strength != "direct":
                distinct[-1] = (contact, index, strength)
            continue
        distinct.append((contact, index, strength))
    if len(distinct) < 2:
        return None

    min_gap = max(float(cfg.merge_gap_s), float(cfg.echo_collapse_s))
    max_gap = float(
        cfg.match_attempt_merge_gap_s + cfg.match_point_start_preroll_s)
    for current_position in range(len(distinct) - 1, 0, -1):
        contact, index, strength = distinct[current_position]
        for prior_contact, prior_index, prior_strength in reversed(
                distinct[:current_position]):
            gap = contact - prior_contact
            required_gap = (
                min_gap
                if strength == "direct" and prior_strength == "direct"
                else float(cfg.match_attempt_merge_min_gap_s)
            )
            if gap < required_gap:
                continue
            if gap > max_gap:
                break
            prior = observations[prior_index]
            current = observations[index]
            reliable_opposite = bool(
                prior.side is not None
                and current.side is not None
                and prior.side != current.side
                and prior.side_confidence >= 0.55
                and current.side_confidence >= 0.55
            )
            if reliable_opposite:
                continue
            if (prior_index != index
                    and strike_counts[prior_index] > cfg.min_rally_strikes + 1):
                # A sustained exchange followed by another serve is the next point, not a
                # fault/retry. Multiple service contacts inside one candidate are exempt.
                continue
            short_fragment_gap = float(
                current.point[0] - prior.point[1])
            current_is_long_exchange = bool(
                current.point[1] - current.point[0] >= 5.0
                and strike_counts[index] >= cfg.min_rally_strikes
                and 0.0 <= short_fragment_gap <= cfg.match_fragment_merge_gap_s
            )
            if (current_is_long_exchange
                    and not (
                        _direct_pose_service(prior)
                        and _direct_pose_service(current)
                    )):
                # A compact serve fragment followed immediately by a long return/exchange
                # is one ordinary point, not two service attempts. Require direct pose on
                # both sides before interpreting this shape as a fault and retry.
                continue
            return (
                index,
                float(contact),
                float(prior_contact),
                tuple(float(value) for value, _member, _strength in distinct),
                prior_index,
            )
    return None


def _fragment_groups(
    points: Sequence[Segment], observations: Sequence[ServeSetupObservation],
    strike_counts: Sequence[int], cfg, *, protected_indices: Iterable[int] = (),
) -> list[list[int]]:
    """Group adjacent fragments that occupy the same service-side state.

    A fault/retry and a rally with one missed contact can both contain multi-second audio
    gaps.  Consecutive tennis points should alternate side, so a short same-side run is one
    logical-point hypothesis.  Unknown side is allowed to inherit only across the same
    short window; it never joins distant candidates.
    """
    if not points:
        return []
    protected = {int(index) for index in protected_indices}
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
        # Tennis cannot finish one point, reset, call the score, and begin the next serve
        # inside the ordinary decoder merge gap. Treat such a boundary as fragmentation
        # even if pose tracking switched player identity or TrackNet missed one side.
        direct_exchange_continuation = bool(0.0 <= gap <= cfg.merge_gap_s)
        if side is None:
            compatible_side = bool(
                _independently_confirmed_serve(
                    observations[index], strike_counts[index], cfg)
                or observations[index].serve_motion
                or observations[index].ball_ordered_evidence)
        else:
            compatible_side = not known_sides or side in known_sides
        new_overhead_after_exchange = bool(
            _robust_target_court_overhead(observations[index])
            and (
                (
                    gap > cfg.merge_gap_s
                    and any(observations[member].ball_ordered_evidence
                            for member in groups[-1])
                )
                or (
                    gap >= 0.5
                    and max(points[member][1] for member in groups[-1])
                    - min(points[member][0] for member in groups[-1]) >= 8.0
                    and sum(strike_counts[member] for member in groups[-1])
                    >= cfg.min_rally_strikes + 8
                )
            )
        )
        new_service_side_after_exchange = bool(
            side is not None and known_sides and side not in known_sides
            and observations[index].target_court_filtered
            and observations[index].side_confidence >= 0.55
            and (observations[index].serve_motion or gap >= 5.5)
            and strike_counts[index] >= cfg.min_rally_strikes + 2
            and any(
                observations[member].ball_ordered_evidence
                or points[member][1] - points[member][0] >= 5.0
                for member in groups[-1]
            )
        )
        isolated_ball_fragment_after_reset = bool(
            gap >= 4.0
            and observations[index].ball_ordered_evidence
            and strike_counts[index] <= 1
            and not _independently_confirmed_serve(
                observations[index], strike_counts[index], cfg)
            and sum(strike_counts[member] for member in groups[-1])
            >= cfg.min_rally_strikes
        )
        if (not (new_overhead_after_exchange or new_service_side_after_exchange)
                and not isolated_ball_fragment_after_reset
                and gap <= cfg.match_fragment_merge_gap_s
                and (compatible_side or direct_exchange_continuation)):
            groups[-1].append(index)
        else:
            groups.append([index])

    # A fault/let retry can take longer than an ordinary missed-contact gap.  Side labels
    # help, but are not authoritative: a far-side serve may make the pose pass switch to
    # the near receiver.  Use temporal serve-attempt structure and never fold a group with
    # already accepted rally structure into its successor.
    merged: list[list[int]] = []
    for source_group_index, group in enumerate(groups):
        if not merged:
            merged.append(list(group))
            continue
        prior_group = merged[-1]
        prior_confirmed_sides = {
            observations[member].side for member in prior_group
            if observations[member].side is not None
            and _independently_confirmed_serve(
                observations[member], strike_counts[member], cfg)
        }
        current_confirmed_sides = {
            observations[member].side for member in group
            if observations[member].side is not None
            and _independently_confirmed_serve(
                observations[member], strike_counts[member], cfg)
        }
        prior_sides = prior_confirmed_sides or {
            observations[member].side for member in prior_group
            if observations[member].side is not None
        }
        sides = current_confirmed_sides or {
            observations[member].side for member in group
            if observations[member].side is not None
        }
        gap = float(points[group[0]][0] - points[prior_group[-1]][1])
        same_known_side = (
            len(prior_sides) == 1 and prior_sides == sides
        )
        reliable_opposite_sides = bool(
            len(prior_sides) == 1 and len(sides) == 1
            and prior_sides != sides
            and max(
                (observations[member].side_confidence for member in prior_group
                 if observations[member].side is not None), default=0.0,
            ) >= 0.55
            and max(
                (observations[member].side_confidence for member in group
                 if observations[member].side is not None), default=0.0,
            ) >= 0.55
        )
        protected_same_side_retry = bool(
            same_known_side
            and cfg.match_attempt_merge_min_gap_s <= gap
            <= cfg.match_attempt_merge_gap_s
        )
        pose_confirmed_same_side_retry = bool(
            same_known_side
            and not reliable_opposite_sides
            and 0.0 <= gap <= cfg.match_attempt_merge_gap_s
            and any(_direct_pose_service(observations[member])
                    for member in prior_group)
            and any(_direct_pose_service(observations[member]) for member in group)
            and sum(strike_counts[member] for member in prior_group)
            <= cfg.min_rally_strikes + 1
        )
        protected_reaction_tail = bool(
            0.0 <= gap <= 3.0
            and sum(strike_counts[member] for member in group) <= 1
            and any(_target_court_receiver_reaction(observations[member])
                    for member in group)
            and not any(
                _independently_confirmed_serve(
                    observations[member], strike_counts[member], cfg)
                for member in group
            )
        )
        if (any(member in protected for member in (*prior_group, *group))
                and not (
                    protected_same_side_retry
                    or pose_confirmed_same_side_retry
                    or protected_reaction_tail
                )):
            # A court-validated rally is normally a hard logical boundary. The one
            # exception is a short same-service-side retry: tennis service side cannot
            # repeat for the next point, while a fault/let may already have a valid ball
            # track and therefore be protected.
            merged.append(list(group))
            continue
        prior_confirmed = _group_has_confirmed_serve(
            prior_group, observations, strike_counts, cfg)
        confirmed = _group_has_confirmed_serve(
            group, observations, strike_counts, cfg)
        prior_dynamic = prior_confirmed or any(
            _independently_confirmed_serve(
                observations[member], strike_counts[member], cfg)
            or observations[member].serve_motion
            or observations[member].ball_ordered_evidence
            for member in prior_group
        )
        dynamic = confirmed or any(
            _independently_confirmed_serve(
                observations[member], strike_counts[member], cfg)
            or observations[member].serve_motion
            or observations[member].ball_ordered_evidence
            for member in group
        )
        next_group = (
            groups[source_group_index + 1]
            if source_group_index + 1 < len(groups) else None
        )
        group_strike_count = sum(strike_counts[member] for member in group)
        bridge_to_confirmed_continuation = bool(
            prior_confirmed and not confirmed
            and not reliable_opposite_sides
            and group_strike_count >= cfg.min_rally_strikes
            and 0.0 <= gap <= cfg.match_attempt_merge_gap_s
            and next_group is not None
            and 0.0 <= points[next_group[0]][0] - points[group[-1]][1]
            <= cfg.match_fragment_merge_gap_s
            and _group_has_confirmed_serve(
                next_group, observations, strike_counts, cfg)
        )
        confirmed_after_bridge = bool(
            prior_confirmed and confirmed
            and not reliable_opposite_sides
            and 0.0 <= gap <= 3.0
            and not _independently_confirmed_serve(
                observations[prior_group[-1]], strike_counts[prior_group[-1]], cfg)
        )
        weak_confirmed_continuation = bool(
            prior_confirmed and confirmed
            and not reliable_opposite_sides
            and 0.0 <= gap <= cfg.match_fragment_merge_gap_s
            and not any(
                _independently_confirmed_serve(
                    observations[member], strike_counts[member], cfg)
                for member in group
            )
        )
        near_reset_confirmed_continuation = bool(
            prior_confirmed and confirmed
            and not reliable_opposite_sides
            and cfg.match_fragment_merge_gap_s < gap
            < cfg.match_attempt_merge_min_gap_s
        )
        prior_contact = min(
            (_serve_contact(observations[member]) for member in prior_group
             if _independently_confirmed_serve(
                 observations[member], strike_counts[member], cfg)),
            default=float("inf"),
        )
        contact = min(
            (_serve_contact(observations[member]) for member in group
             if _independently_confirmed_serve(
                 observations[member], strike_counts[member], cfg)),
            default=float("inf"),
        )
        retry_without_completed_rally = bool(
            prior_confirmed and confirmed
            and not reliable_opposite_sides
            and not any(member in protected for member in prior_group)
            and cfg.match_attempt_merge_min_gap_s <= gap
            <= cfg.match_attempt_merge_gap_s
            and 0.0 <= contact - prior_contact
            and contact - prior_contact >= cfg.match_attempt_merge_gap_s
            and contact - prior_contact
            <= cfg.match_attempt_merge_gap_s + cfg.match_point_start_preroll_s
        )
        # An unconfirmed fragment shortly after a dynamic group is an exchange tail or
        # pickup, not a new serve. Keep it inside the logical point so it can contribute
        # at most the configured terminal buffer instead of becoming an output point.
        trailing_fragment = bool(
            prior_dynamic and not confirmed
            and 0.0 <= gap <= 3.0
            and sum(strike_counts[member] for member in group) <= 1
        )
        serve_then_exchange = bool(
            prior_confirmed and not confirmed
            and not reliable_opposite_sides
            and 0.0 <= gap <= cfg.match_fragment_merge_gap_s
            and (
                sum(strike_counts[member] for member in group) >= 2
                or any(_supporting_target_ball_motion(observations[member])
                       for member in group)
            )
            and (
                gap <= 3.0
                or sum(strike_counts[member] for member in group)
                >= cfg.min_rally_strikes + 2
                or any(_target_court_receiver_reaction(observations[member])
                       for member in group)
            )
        )
        same_side_retry_or_feed = bool(
            same_known_side and prior_dynamic and dynamic
            and cfg.match_attempt_merge_min_gap_s <= gap
            <= cfg.match_attempt_merge_gap_s
        )
        if (same_side_retry_or_feed
                or pose_confirmed_same_side_retry
                or retry_without_completed_rally
                or trailing_fragment
                or serve_then_exchange
                or bridge_to_confirmed_continuation
                or confirmed_after_bridge
                or weak_confirmed_continuation
                or near_reset_confirmed_continuation):
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
        or (
            observation.position_setup_evidence
            and observation.ball_coverage >= 0.8 * cfg.match_ball_min_coverage
            and observation.ball_vertical_span >= cfg.match_ball_min_vertical_span
            and observation.ball_outgoing_span >= cfg.match_ball_min_outgoing_span
        )
    )


def _supporting_target_ball_motion(observation: ServeSetupObservation) -> bool:
    """Near-threshold visual ball motion usable only beside stronger point context."""
    return bool(
        observation.target_court_filtered
        and observation.ball_checked
        and observation.ball_measured_samples >= 5
        and observation.ball_coverage >= 0.08
        and observation.ball_vertical_span >= 0.012
        and observation.ball_outgoing_span >= 0.01
    )


def _moderate_reaction_formation(
    observation: ServeSetupObservation, cfg,
) -> bool:
    """Partial baseline formation usable only with a dynamic player reaction."""
    return bool(
        observation.position_checked
        and observation.position_server_end is not None
        and observation.position_server_span is not None
        and observation.position_server_span <= cfg.match_position_max_span
        and observation.position_stable_fraction >= 0.5
        and observation.position_score >= cfg.match_position_min_score
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
        if _stable_baseline_formation(observations[index], cfg)
        and observations[index].position_best_strike is not None
    ]
    balls = [
        _serve_contact(observations[index]) for index in indices
        if _potential_ball_serve(observations[index], cfg)
    ]
    supporting_balls = [
        _serve_contact(observations[index]) for index in indices
        if _supporting_target_ball_motion(observations[index])
    ]
    paired_ball = any(
        0.0 <= ball_time - position_time <= cfg.match_attempt_merge_gap_s
        for position_time in positions for ball_time in balls
    )
    paired_supporting_ball = any(
        0.0 <= ball_time - position_time <= cfg.match_attempt_merge_gap_s
        for position_time in positions for ball_time in supporting_balls
    )
    reactions = [
        float(observations[index].receiver_reaction_time)
        for index in indices
        if _target_court_receiver_reaction(observations[index])
    ]
    paired_reaction = any(
        0.0 <= reaction_time - position_time <= cfg.match_attempt_merge_gap_s
        for position_time in positions for reaction_time in reactions
    )
    reaction_structure = bool(
        any(_moderate_reaction_formation(observations[index], cfg)
            for index in indices)
        or supporting_balls
        or any(
            _target_court_receiver_reaction(observations[index])
            and float(observations[index].receiver_reaction_time)
            < observations[index].point[0] - cfg.match_fragment_merge_gap_s
            for index in indices
        )
    )
    reaction_strike_floor = (
        cfg.min_rally_strikes if reaction_structure
        else cfg.min_rally_strikes + 3
    )
    reaction_with_exchange = bool(
        reactions
        and sum(strike_counts[index] for index in indices) >= reaction_strike_floor
    )
    service_motion_with_exchange = bool(
        any(
            observations[index].target_court_filtered
            and observations[index].serve_motion
            for index in indices
        )
        and sum(strike_counts[index] for index in indices)
        >= cfg.min_rally_strikes + 2
    )
    return (paired_ball or paired_supporting_ball
            or paired_reaction or reaction_with_exchange
            or service_motion_with_exchange)


def _select_serve_member(
    indices: Sequence[int], observations: Sequence[ServeSetupObservation],
    points: Sequence[Segment], strike_counts: Sequence[int], cfg,
) -> int:
    """Choose the serve attempt that anchors one merged logical point.

    Multiple independently aligned overhead+position events represent a fault/let retry,
    so the last is a valid canonical start.  Otherwise select the strongest event, with
    pose+ball evidence ahead of a static-position/ball coincidence.
    """
    aligned = [index for index in indices if _aligned_pose_strikes(observations[index])]
    if len(aligned) >= 2:
        return max(aligned, key=lambda index: _serve_contact(observations[index]))

    long_retry = any(
        points[current][0] - points[prior][1] > cfg.match_fragment_merge_gap_s
        for prior, current in zip(indices, indices[1:])
    )
    independently_confirmed = [
        index for index in indices
        if _independently_confirmed_serve(
            observations[index], strike_counts[index], cfg)
    ]
    if independently_confirmed and not long_retry:
        pose_ball = [
            index for index in independently_confirmed
            if observations[index].serve_motion
            and observations[index].ball_serve_evidence
        ]
        earliest = min(
            independently_confirmed,
            key=lambda index: (_serve_contact(observations[index]), index),
        )
        earlier_group_paired_ball = [
            index for index in indices
            if index < earliest
            and _potential_ball_serve(observations[index], cfg)
            and any(
                setup < index
                and _stable_baseline_formation(observations[setup], cfg)
                and 0.0 <= _serve_contact(observations[index])
                - float(observations[setup].position_best_strike)
                <= cfg.match_attempt_merge_gap_s
                for setup in indices
                if observations[setup].position_best_strike is not None
            )
        ]
        if earlier_group_paired_ball:
            return min(
                earlier_group_paired_ball,
                key=lambda index: (_serve_contact(observations[index]), index),
            )
        if pose_ball:
            strongest_pose = max(
                pose_ball,
                key=lambda index: (
                    observations[index].ball_coverage
                    * observations[index].ball_vertical_span,
                    _serve_contact(observations[index]),
                ),
            )
            early_motion = max(
                1e-9,
                observations[earliest].ball_coverage
                * observations[earliest].ball_vertical_span,
            )
            pose_motion = (
                observations[strongest_pose].ball_coverage
                * observations[strongest_pose].ball_vertical_span
            )
            if (strongest_pose != earliest
                    and pose_motion >= 2.5 * early_motion):
                # Ball bounces during serve preparation can look like a short ordered
                # flight. A substantially stronger ball track aligned with an overhead
                # action is the actual serve, not a later rally return.
                return strongest_pose
        robust_overheads = [
            index for index in independently_confirmed
            if _robust_target_court_overhead(observations[index])
        ]
        if robust_overheads and earliest not in robust_overheads:
            # An early receiver reaction or marginal TrackNet fragment can precede the
            # actual serve in the same short candidate group.  A repeated target-court
            # overhead is the direct service action and must anchor the point.
            return min(
                robust_overheads,
                key=lambda index: (_serve_contact(observations[index]), index),
            )
        if (not observations[earliest].serve_motion
                and not observations[earliest].ball_serve_evidence):
            setup_times = [
                float(observations[index].position_best_strike)
                for index in indices
                if _stable_baseline_formation(observations[index], cfg)
                and observations[index].position_best_strike is not None
            ]
            reaction_after_setup = [
                index for index in indices
                if _target_court_receiver_reaction(observations[index])
                and any(
                    0.0 <= float(observations[index].receiver_reaction_time) - setup
                    <= cfg.match_player_proposal_s
                    for setup in setup_times
                )
            ]
            if reaction_after_setup:
                return min(
                    reaction_after_setup,
                    key=lambda index: (
                        float(observations[index].receiver_reaction_time), index),
                )
        later_formation_exchanges = [
            exchange for exchange in indices
            if exchange > earliest
            and points[exchange][1] - points[exchange][0] >= 5.0
            and strike_counts[exchange] >= cfg.min_rally_strikes + 2
            and any(
                setup < exchange
                and _stable_baseline_formation(observations[setup], cfg)
                and 0.0 <= points[exchange][0] - points[setup][1]
                <= cfg.match_fragment_merge_gap_s
                for setup in indices
            )
        ]
        if later_formation_exchanges:
            # A short early overhead can be a warm-up/feed hypothesis. A later reset into
            # stationary baseline formation followed by a sustained exchange is the
            # service attempt that owns this logical point.
            return min(later_formation_exchanges)
        # A point can be split at missed audio contacts.  Once an earlier fragment is
        # independently serve-confirmed, a stronger return/rally track later in the same
        # short group must not replace it as the point start.
        return earliest

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
    if long_retry:
        paired_dynamic_serves = [
            index for index in indices
            if _potential_ball_serve(observations[index], cfg)
            and _target_court_receiver_reaction(observations[index])
            and any(
                setup < index
                and _stable_baseline_formation(observations[setup], cfg)
                and observations[setup].position_best_strike is not None
                and 0.0 <= _serve_contact(observations[index])
                - float(observations[setup].position_best_strike)
                <= cfg.match_player_proposal_s
                for setup in indices
            )
        ]
        if paired_dynamic_serves:
            # Setup + target-court reaction + ordered ball motion is already a complete
            # service hypothesis. A later static reaction is normally the exchange, not
            # evidence that the earlier serve/feed should be discarded as a false start.
            return min(
                paired_dynamic_serves,
                key=lambda index: (_serve_contact(observations[index]), index),
            )
        # When a reaction-only prelude is followed by a clearly observed target-court
        # overhead, retain that earliest real service action. Ball-only pickup/feed
        # hypotheses still fall through to the stronger/later retry rule below.
        target_overheads = [
            index for index in indices
            if _robust_target_court_overhead(observations[index])
        ]
        if target_overheads:
            return min(
                target_overheads,
                key=lambda index: (_serve_contact(observations[index]), index),
            )
        retry_serves = [
            index for index in indices
            if _independently_confirmed_serve(
                observations[index], strike_counts[index], cfg)
            or observations[index].serve_motion
            or _potential_ball_serve(observations[index], cfg)
        ]
        return max(
            retry_serves or list(indices),
            key=lambda index: (_serve_contact(observations[index]), index),
        )

    positions = [
        float(observations[index].position_best_strike) for index in indices
        if _stable_baseline_formation(observations[index], cfg)
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
    paired_reactions = [
        index for index in indices
        if _target_court_receiver_reaction(observations[index])
        and any(
            0.0 <= float(observations[index].receiver_reaction_time) - position_time
            <= cfg.match_attempt_merge_gap_s
            for position_time in positions
        )
    ]
    if paired_reactions:
        return min(
            paired_reactions,
            key=lambda index: (
                float(observations[index].receiver_reaction_time), index),
        )
    motion_members = [
        index for index in indices
        if observations[index].target_court_filtered
        and observations[index].serve_motion
    ]
    if motion_members:
        return max(
            motion_members,
            key=lambda index: (_serve_contact(observations[index]), index),
        )
    reaction_members = [
        index for index in indices
        if _target_court_receiver_reaction(observations[index])
        and float(observations[index].receiver_reaction_time)
        >= observations[index].point[0] - cfg.match_point_start_preroll_s
    ]
    if reaction_members:
        return min(
            reaction_members,
            key=lambda index: (
                float(observations[index].receiver_reaction_time), index),
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
    contact_onsets: Optional[np.ndarray] = None,
) -> tuple[List[Segment], dict]:
    """Reject candidates inside match play that lack a confirmed serve event."""
    points = [(float(start), float(end)) for start, end in points]
    protected_indices = {int(index) for index in protected_indices}
    terminal_onsets = (
        np.asarray(onsets, dtype=float)
        if contact_onsets is None else np.asarray(contact_onsets, dtype=float)
    )
    if len(points) != len(observations):
        raise ValueError("match observations must align one-to-one with points")

    strike_counts = [
        _effective_strike_count(point, onsets, cfg.echo_collapse_s) for point in points
    ]
    if cfg.play_mode == "match":
        phases = [(0, len(points) - 1)] if points else []
    else:
        phases = _auto_match_phases(observations, strike_counts, cfg)
    fail_closed_auto = bool(
        cfg.play_mode == "auto"
        and not phases
        and getattr(cfg, "match_auto_fail_closed", False)
        and observations
        and all(observation.ball_checked for observation in observations)
    )
    if fail_closed_auto:
        # Every candidate has had a real ball check. With no repeated legal service phase,
        # treating speech, pickup hits, or resting-player motion as points is a false-open
        # failure. Decode the whole stream under strict serve rules instead.
        phases = [(0, len(points) - 1)] if points else []
    if cfg.play_mode == "auto" and not phases:
        return list(points), {
            "status": "abstained",
            "mode": "auto",
            "rule": "confirmed_serve_event_required_inside_match_phase",
            "match_phases": [],
            "logical_groups": [
                {
                    "group_index": index,
                    "member_indices": [index],
                    "decision": "keep",
                    "inside_match_phase": False,
                    "output": [round(point[0], 3), round(point[1], 3)],
                }
                for index, point in enumerate(points)
            ],
            "side_sequence": [observation.side for observation in observations],
            "observations": [asdict(observation) for observation in observations],
            "dropped": [],
            "trajectory_accepted_not_exempt": [],
            "kept_points": len(points),
            "reason": (
                "auto mode did not observe enough ball-checked serve anchors "
                "to infer match play"
            ),
        }
    required = {
        index for start, end in phases for index in range(start, end + 1)
    }
    phase_min = min(required) if required else None
    phase_max = max(required) if required else None

    groups = _fragment_groups(
        points, observations, strike_counts, cfg,
        protected_indices=protected_indices)
    dropped: set[int] = set()
    reasons: list[dict] = []
    group_records: list[dict] = []
    kept: List[Segment] = []
    prior_end = 0.0
    prior_service_side: Optional[str] = None
    for group_index, indices in enumerate(groups):
        inside_phase = any(index in required for index in indices)
        group_start = min(points[index][0] for index in indices)
        group_raw_end = max(points[index][1] for index in indices)
        group_strike_count = sum(strike_counts[index] for index in indices)
        reliable_group_sides = {
            observations[index].side for index in indices
            if observations[index].side is not None
            and observations[index].side_confidence >= 0.55
        }
        sequence_inferred_serve = bool(
            inside_phase
            and prior_service_side is not None
            and len(reliable_group_sides) == 1
            and prior_service_side not in reliable_group_sides
            and group_strike_count >= cfg.min_rally_strikes + 2
            and group_start - prior_end >= 5.0
            and group_start - prior_end <= cfg.match_attempt_merge_min_gap_s
            and all(observations[index].target_court_filtered for index in indices)
        )
        confirmed = (
            _group_has_confirmed_serve(indices, observations, strike_counts, cfg)
            or sequence_inferred_serve
        )
        strong_service_action = any(
            _aligned_pose_strikes(observations[index])
            or _robust_target_court_overhead(observations[index])
            or (
                observations[index].ball_serve_evidence
                and (
                    _stable_baseline_formation(observations[index], cfg)
                    or observations[index].ball_coverage
                    >= cfg.match_ball_min_single_strike_coverage
                )
            )
            for index in indices
        )
        if getattr(cfg, "match_auto_fail_closed", False):
            # In the production web path every candidate has ball, pose, and player
            # formation evidence available. A stationary formation plus receiver motion
            # is still common during rest, changeovers, and ball collection; it may help
            # locate a serve but cannot *be* the serve. Require a measured dynamic action
            # before publishing the logical group as a point.
            confirmed = bool(confirmed and strong_service_action)
        weak_dynamic_hypothesis = bool(confirmed and not strong_service_action)
        implausibly_fast_reset = bool(
            kept
            and weak_dynamic_hypothesis
            and group_strike_count <= cfg.min_rally_strikes
            and 0.0 <= group_start - prior_end <= cfg.match_attempt_merge_gap_s
        )
        lookback_inflated_fragment = bool(
            weak_dynamic_hypothesis
            and group_raw_end - group_start < cfg.min_rally_s
        )
        if inside_phase and (implausibly_fast_reset or lookback_inflated_fragment):
            dropped.update(indices)
            reason_code = (
                "weak_serve_hypothesis_inside_reset_gap"
                if implausibly_fast_reset else
                "lookback_reaction_without_point_duration"
            )
            reasons.append({
                "index": indices[0],
                "member_indices": list(indices),
                "reason_code": reason_code,
                "reason": (
                    "reaction/ball-only activity is too close to the prior point or "
                    "too short to establish a new service point"
                ),
                "raw_duration": round(group_raw_end - group_start, 3),
                "gap_after_prior_point": (
                    round(group_start - prior_end, 3) if kept else None),
            })
            group_records.append({
                "group_index": group_index,
                "member_indices": list(indices),
                "decision": "drop",
                "inside_match_phase": True,
                "reason_code": reason_code,
            })
            continue
        # Once repeated target-court serves establish a match, a lone acoustic transient
        # immediately outside its inferred boundary is not a rally.  Keep multi-impact
        # warm-up hitting unconstrained, but do not publish pre-match chatting/background
        # noise (or a trailing pickup) as point 1 merely because auto mode preserves
        # unstructured warm-up footage.
        outside_phase_boundary = bool(
            required
            and (
                max(indices) < int(phase_min)
                or min(indices) > int(phase_max)
            )
        )
        weak_target_boundary_noise = bool(
            outside_phase_boundary
            and not confirmed
            and any(observations[index].target_court_filtered for index in indices)
            and sum(strike_counts[index] for index in indices) < cfg.min_rally_strikes
        )
        if weak_target_boundary_noise:
            dropped.update(indices)
            reasons.append({
                "index": indices[0],
                "member_indices": list(indices),
                "reason_code": "weak_boundary_noise_outside_match_phase",
                "reason": (
                    "single background transient outside the target-court match phase"
                ),
                "strike_count": sum(strike_counts[index] for index in indices),
            })
            group_records.append({
                "group_index": group_index,
                "member_indices": list(indices),
                "decision": "drop",
                "inside_match_phase": False,
                "reason_code": "weak_boundary_noise_outside_match_phase",
            })
            continue
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
                "target_court_filtered": observation.target_court_filtered,
                "receiver_reaction_evidence": observation.receiver_reaction_evidence,
                "receiver_reaction_time": observation.receiver_reaction_time,
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
        serve_member = _select_serve_member(
            indices, observations, points, strike_counts, cfg)
        retry_serve = _detected_retry_serve(
            indices, observations, strike_counts, cfg)
        if retry_serve is not None:
            # A confidently observed second service action owns the point. Never let the
            # failed first serve win a generic evidence-strength tie.
            serve_member = retry_serve[0]
        reaction_members = [
            index for index in indices
            if _target_court_receiver_reaction(observations[index])
        ]
        if (kept and reaction_members
                and not observations[serve_member].serve_motion
                and not observations[serve_member].ball_serve_evidence
                and 0.0 <= group_start - prior_end <= cfg.merge_gap_s):
            reaction_span = (
                max(float(observations[index].receiver_reaction_time)
                    for index in reaction_members)
                - min(float(observations[index].receiver_reaction_time)
                      for index in reaction_members)
            )
            if reaction_span >= cfg.match_attempt_merge_min_gap_s:
                # Activity immediately after the previous endpoint is often pickup/reset
                # motion. A later stable-to-active transition is the ensuing serve.
                serve_member = max(
                    reaction_members,
                    key=lambda index: (
                        float(observations[index].receiver_reaction_time), index),
                )
        group_position_setup = any(
            _stable_baseline_formation(observations[index], cfg)
            for index in indices)
        group_supporting_setup = any(
            _supporting_baseline_formation(observations[index], cfg)
            for index in indices)
        selected_reaction_is_group_corroborated = bool(
            (group_supporting_setup
             or group_strike_count >= cfg.min_rally_strikes)
            and _target_court_receiver_reaction(observations[serve_member])
            and not _aligned_pose_strikes(observations[serve_member])
            and not _robust_target_court_overhead(observations[serve_member])
            and not observations[serve_member].ball_serve_evidence
            and (
                not observations[serve_member].position_setup_evidence
                or float(observations[serve_member].receiver_reaction_time)
                >= observations[serve_member].point[0]
            )
        )
        contact = (
            retry_serve[1]
            if retry_serve is not None else (
                float(observations[serve_member].receiver_reaction_time)
                if selected_reaction_is_group_corroborated
                else _serve_contact(observations[serve_member])
            )
        )
        dynamic_members = [
            index for index in indices
            if index in protected_indices
            # A trajectory acceptance cannot prove that a point started with a serve,
            # but after this logical group has passed the serve gate it is strong evidence
            # that a later fragment is still live play and owns the terminal boundary.
            or _independently_confirmed_serve(
                observations[index], strike_counts[index], cfg)
            or (
                group_position_setup
                and _target_court_receiver_reaction(observations[index])
            )
            or (
                observations[index].ball_ordered_evidence
                and (
                    index == serve_member
                    or not _target_court_receiver_reaction(observations[index])
                    or float(observations[index].receiver_reaction_time)
                    <= contact + 1.0
                )
                and (
                    strike_counts[index] >= 2
                    or any(
                        prior < index
                        and 0.0 <= points[index][0] - points[prior][1] <= 3.0
                        for prior in indices
                    )
                )
            )
            or (
                _supporting_target_ball_motion(observations[index])
                and not _target_court_receiver_reaction(observations[index])
            )
            or (
                strike_counts[index] >= cfg.min_rally_strikes + 2
                and any(
                    prior < index
                    and 0.0 <= points[index][0] - points[prior][1] <= 4.0
                    for prior in indices
                )
            )
        ]
        terminal_index = max(dynamic_members) if dynamic_members else serve_member
        group_end = max(
            points[index][1] for index in indices if index <= terminal_index)
        next_group_start = (
            min(points[index][0] for index in groups[group_index + 1])
            if group_index + 1 < len(groups) else float("inf")
        )
        terminal_observation = observations[terminal_index]
        max_contact_gap = (
            cfg.point_gap_s
            if terminal_observation.ball_ordered_evidence
            and terminal_observation.ball_coverage >= 0.5
            else (
                cfg.point_gap_s
                if terminal_observation.ball_ordered_evidence
                else cfg.merge_gap_s
            )
        )
        min_terminal_contacts = (
            2
            if terminal_observation.ball_ordered_evidence
            and terminal_observation.ball_coverage >= 0.5
            else 1
        )
        # A well-covered ordered TrackNet trajectory already owns the visual endpoint.
        # Do not let nearby score calls or neighboring-court impacts extend it. Sparse or
        # unordered tracks may still use adjacent contacts as a conservative recovery.
        if not (
            terminal_observation.ball_ordered_evidence
            and terminal_observation.ball_coverage >= 0.5
        ):
            group_end = _extend_through_adjacent_terminal_contacts(
                group_end, terminal_onsets, next_group_start, max_contact_gap,
                min_terminal_contacts, cfg)
        group_end, sparse_tail_trimmed = _trim_sparse_trailing_contacts(
            points[terminal_index], group_end, terminal_onsets, cfg)
        # A final isolated impact can have too little TrackNet coverage to be dynamic even
        # though grouping already established that it belongs to this same-side point.
        # Retain only the configured one-second post-contact buffer; do not let it become
        # a new point or pull in an arbitrary reset interval.
        for index in indices:
            reaction_tail = _target_court_receiver_reaction(observations[index])
            if (index <= terminal_index
                    or (strike_counts[index] < 1 and not reaction_tail)
                    or observations[index].ball_ordered_evidence):
                continue
            terminal_side = observations[terminal_index].side
            if (terminal_side is not None
                    and observations[index].side is not None
                    and observations[index].side != terminal_side):
                break
            same_service_side = bool(
                terminal_side is not None
                and observations[index].side == terminal_side
            )
            # One unowned audio transient is not enough to extend a point. It is commonly
            # a score call, pickup, or neighboring-court strike. Preserve the one-second
            # buffer only when service-side continuity or an independent target-court
            # receiver reaction assigns the fragment to this logical point.
            multi_contact_tail = strike_counts[index] >= cfg.min_rally_strikes
            if not (same_service_side or reaction_tail or multi_contact_tail):
                continue
            trailing_gap_limit = (
                cfg.match_fragment_merge_gap_s
                if same_service_side else cfg.merge_gap_s
            )
            if reaction_tail:
                trailing_gap_limit = max(trailing_gap_limit, 3.0)
            elif multi_contact_tail:
                trailing_gap_limit = max(trailing_gap_limit, cfg.point_gap_s)
            if points[index][0] - group_end > trailing_gap_limit:
                break
            trailing_contacts = terminal_onsets
            trailing_contacts = trailing_contacts[
                (trailing_contacts >= points[index][0] - 1e-9)
                & (trailing_contacts <= points[index][1] + cfg.landing_tail_s + 1e-9)
            ]
            contact_end = (
                float(observations[index].receiver_reaction_time + cfg.landing_tail_s)
                if reaction_tail else (
                    float(np.min(trailing_contacts) + cfg.landing_tail_s)
                    if trailing_contacts.size else
                    float(observations[index].first_strike + cfg.landing_tail_s)
                )
            )
            group_end = max(group_end, min(
                points[index][1] + cfg.landing_tail_s, contact_end))
        # Recover serve preparation from the selected contact.  Unlike fixed audio
        # clustering this may trim score calls before the serve as well as extend a late
        # rally-only fragment back to the actual point setup.
        selected = observations[serve_member]
        start_preroll = cfg.match_point_start_preroll_s
        reaction_contact_selected = bool(
            _target_court_receiver_reaction(selected)
            and abs(contact - float(selected.receiver_reaction_time)) <= 1e-6
        )
        early_setup_before_reaction = bool(
            reaction_contact_selected
            and any(
                _stable_baseline_formation(observations[index], cfg)
                and observations[index].position_best_strike is not None
                and float(observations[index].position_best_strike)
                <= float(selected.receiver_reaction_time) - 1.0
                for index in indices
            )
        )
        if reaction_contact_selected and not early_setup_before_reaction:
            # Without a baseline setup established before the movement transition, there
            # is no visual support for retaining a long prelude. Keep only the normal
            # setup lookback; this trims walking while preserving a measured serve prep.
            start_preroll = min(start_preroll, cfg.match_setup_lookback_s)
        recovered_start = max(0.0, contact - start_preroll)
        position_contact_with_distant_reaction = bool(
            selected.position_setup_evidence
            and not selected.serve_motion
            and not selected.ball_serve_evidence
            and _target_court_receiver_reaction(selected)
            and float(selected.receiver_reaction_time)
            < selected.point[0] - cfg.match_point_start_preroll_s
        )
        if retry_serve is not None:
            (_retry_member, _retry_contact, prior_attempt_contact,
             _attempts, prior_attempt_member) = retry_serve
            if serve_member != prior_attempt_member:
                prior_attempt_end = float(points[prior_attempt_member][1])
                retry_candidate_start = float(points[serve_member][0])
                # When segmentation leaves only a tiny gap, its contents are reset/pickup
                # footage rather than useful service preparation. For a longer reset, keep
                # only the configured retry-serve preroll nearest the second contact.
                if (0.0 <= retry_candidate_start - prior_attempt_end
                        <= cfg.merge_gap_s):
                    retry_attempt_floor = retry_candidate_start
                else:
                    retry_attempt_floor = prior_attempt_end
            else:
                # Two service contacts may live inside one broad audio candidate. Start
                # after the failed attempt's normal tail rather than at the candidate edge.
                retry_attempt_floor = float(
                    prior_attempt_contact + cfg.landing_tail_s)
            desired_retry_start = max(
                retry_attempt_floor,
                float(contact - cfg.match_point_start_preroll_s),
            )
            # Preserve the minimum one-hit point duration when the retry ends immediately
            # (for example a double fault), but never move back across the failed attempt.
            latest_valid_start = max(
                retry_attempt_floor, float(group_end - cfg.min_rally_s))
            start = max(
                retry_attempt_floor,
                min(desired_retry_start, latest_valid_start),
            )
        else:
            start = (
                group_start
                if position_contact_with_distant_reaction else
                (recovered_start if (confirmed or inside_phase) else group_start)
            )
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
        selected_side = observations[serve_member].side
        if (selected_side is not None
                and observations[serve_member].side_confidence >= 0.55):
            prior_service_side = selected_side
        elif len(reliable_group_sides) == 1:
            prior_service_side = next(iter(reliable_group_sides))
        group_records.append({
            "group_index": group_index,
            "member_indices": list(indices),
            "decision": "keep",
            "inside_match_phase": inside_phase,
            "serve_member_index": serve_member,
            "serve_contact": round(contact, 3),
            "output": [round(segment[0], 3), round(segment[1], 3)],
            "sparse_tail_trimmed": sparse_tail_trimmed,
            **({
                "retry_serve_detected": True,
                "retry_serve_contacts": [
                    round(value, 3) for value in retry_serve[3]
                ],
                "failed_first_serve_trimmed": True,
            } if retry_serve is not None else {}),
            **({"serve_inferred_from_side_alternation": True}
               if sequence_inferred_serve else {}),
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
                    ("stable_baseline_formation",
                     _stable_baseline_formation(observation, cfg)
                     and not observation.position_setup_evidence),
                    ("overhead_pose_with_position_setup", aligned_pose_setup),
                    ("robust_target_court_overhead_pose",
                     _robust_target_court_overhead(observation)),
                    ("target_court_receiver_reaction",
                     _corroborated_receiver_reaction(observation)),
                    ("tracknet_ball_motion", observation.ball_serve_evidence),
                    ("eligible_held_out_serve_classifier",
                     observation.learned_serve_checked
                     and observation.learned_serve_evidence),
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
        "auto_fail_closed": fail_closed_auto,
    }
    if not phases:
        stage["reason"] = (
            "auto mode did not observe enough ball-checked serve anchors to infer match play"
        )
    return kept, stage
