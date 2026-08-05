"""Rule-constrained point outcomes and player attribution.

This module consumes evidence that the pipeline has already paid to compute.  It does not
run another detector: sparse target-court player observations establish persistent match
identities, the TrackNet cache supplies point trajectories, match-state observations locate
the service action, and audio impacts propose racket contacts.

The output deliberately separates an objective tennis rule event (out, second bounce,
double fault, net failure) from statistical credit (ace, clean winner, error).  When the
single-camera evidence is incomplete the decoder returns ``unknown`` instead of inventing
an actor or a line call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..signals.ball import BallTrack
from ..signals.ballrules import side_of
from ..signals.court import COURT_L, DOUBLES_W, NET_Y, SERVICE_Y, SINGLES_IN
from ..signals.trajectory import SmoothTrack, bounces_from_velocity, smooth_track

Segment = tuple[float, float]
POINT_SCHEMA_VERSION = "rally.point_events.v1"


@dataclass
class _Detection:
    x: float
    y: float
    area: float


@dataclass
class _IdentityTrack:
    player_id: str
    team_id: str
    initial_end: str
    records: list[tuple[float, float, float]] = field(default_factory=list)

    def add(self, time_s: float, x: float, y: float) -> None:
        self.records.append((float(time_s), float(x), float(y)))

    def position(self, time_s: float, max_gap_s: float = 3.0) -> Optional[np.ndarray]:
        if not self.records:
            return None
        values = np.asarray(sorted(set(self.records)), dtype=float)
        times = values[:, 0]
        right = int(np.searchsorted(times, time_s))
        if 0 < right < len(values):
            left = right - 1
            if (time_s - times[left] <= max_gap_s
                    and times[right] - time_s <= max_gap_s):
                span = max(times[right] - times[left], 1e-9)
                fraction = (time_s - times[left]) / span
                return values[left, 1:] + fraction * (values[right, 1:] - values[left, 1:])
        nearest = int(np.argmin(np.abs(times - time_s)))
        if abs(float(times[nearest]) - time_s) <= max_gap_s:
            return values[nearest, 1:].copy()
        return None


def _inside_any(time_s: float, segments: Sequence[Segment]) -> bool:
    return not segments or any(start <= time_s <= end for start, end in segments)


def _court_player_samples(
    player_samples: Sequence,
    court,
    frame_size: Optional[tuple[int, int]],
) -> list[tuple[float, list[_Detection]]]:
    if court is None or frame_size is None:
        return []
    width, height = frame_size
    if width <= 0 or height <= 0:
        return []
    output: list[tuple[float, list[_Detection]]] = []
    for raw_time, people in player_samples:
        if not people:
            output.append((float(raw_time), []))
            continue
        pixels = np.asarray(
            [[float(person[0]) * width, float(person[1]) * height] for person in people],
            dtype=float,
        )
        try:
            coordinates = np.asarray(court.to_court(pixels), dtype=float).reshape(-1, 2)
        except Exception:
            continue
        detections: list[_Detection] = []
        for person, coordinate in zip(people, coordinates):
            x, y = (float(coordinate[0]), float(coordinate[1]))
            # Tight enough to exclude spectators/adjacent courts while retaining a server
            # behind either baseline and a player pulled outside a doubles alley.
            if (np.isfinite(x) and np.isfinite(y)
                    and -1.5 <= x <= DOUBLES_W + 1.5
                    and -3.0 <= y <= COURT_L + 3.0):
                detections.append(_Detection(x, y, float(person[2])))
        output.append((float(raw_time), detections))
    return output


def infer_match_format(
    player_samples: Sequence,
    court,
    frame_size: Optional[tuple[int, int]],
    segments: Sequence[Segment] = (),
) -> dict[str, Any]:
    """Infer singles/doubles from repeated target-court formations.

    One missed far-side player must not turn doubles into singles.  Strong two-versus-two
    frames and partial three-player frames are accumulated across all accepted points.
    """
    samples = _court_player_samples(player_samples, court, frame_size)
    formations: list[tuple[int, int]] = []
    for time_s, detections in samples:
        if not _inside_any(time_s, segments):
            continue
        near = sum(detection.y < NET_Y for detection in detections)
        far = sum(detection.y >= NET_Y for detection in detections)
        if near and far:
            formations.append((near, far))
    valid = len(formations)
    if not valid:
        return {
            "format": "unknown", "confidence": 0.0, "player_count": 0,
            "evidence_frames": 0, "reason": "no two-sided target-court player formations",
        }
    two_by_two = sum(near >= 2 and far >= 2 for near, far in formations)
    three_visible = sum(near + far >= 3 for near, far in formations)
    one_by_one = sum(near == 1 and far == 1 for near, far in formations)
    strong_ratio = two_by_two / valid
    partial_ratio = three_visible / valid
    doubles = bool(
        two_by_two >= max(2, int(np.ceil(0.08 * valid)))
        or partial_ratio >= 0.30
    )
    if doubles:
        confidence = np.clip(0.55 + 0.35 * partial_ratio + 0.10 * strong_ratio, 0.0, 0.99)
        match_format, count = "doubles", 4
    else:
        singles_ratio = one_by_one / valid
        confidence = np.clip(0.55 + 0.40 * singles_ratio - 0.20 * partial_ratio, 0.0, 0.99)
        match_format, count = "singles", 2
    return {
        "format": match_format,
        "confidence": round(float(confidence), 4),
        "player_count": count,
        "evidence_frames": valid,
        "two_by_two_frames": two_by_two,
        "three_player_frames": three_visible,
        "one_by_one_frames": one_by_one,
        "source": "persistent_target_court_player_occupancy",
    }


def _match_state_format_fallback(match_state: dict[str, Any]) -> dict[str, Any]:
    """Use serve-window occupancy when sparse full-frame tracks miss one court end."""
    observations = [
        observation for observation in (match_state.get("observations") or [])
        if isinstance(observation, dict)
        and observation.get("position_checked")
        and observation.get("target_court_filtered")
    ]
    stable_counts = [
        int(observation.get("position_stable_tracks") or 0)
        for observation in observations
    ]
    three_plus = sum(count >= 3 for count in stable_counts)
    two_plus = sum(count >= 2 for count in stable_counts)
    if three_plus >= 2:
        confidence = float(np.clip(
            0.55 + 0.40 * three_plus / len(stable_counts), 0.0, 0.92))
        return {
            "format": "doubles", "confidence": round(confidence, 4),
            "player_count": 4, "evidence_frames": len(stable_counts),
            "three_plus_stable_windows": three_plus,
            "two_plus_stable_windows": two_plus,
            "source": "target_court_serve_window_stable_occupancy",
        }
    if two_plus >= 2:
        confidence = float(np.clip(
            0.55 + 0.35 * two_plus / len(stable_counts), 0.0, 0.88))
        return {
            "format": "singles", "confidence": round(confidence, 4),
            "player_count": 2, "evidence_frames": len(stable_counts),
            "three_plus_stable_windows": three_plus,
            "two_plus_stable_windows": two_plus,
            "source": "target_court_serve_window_stable_occupancy",
        }
    return {
        "format": "unknown", "confidence": 0.0, "player_count": 0,
        "evidence_frames": len(stable_counts),
        "reason": "serve-window occupancy is also insufficient",
    }


def _default_identities(match_format: str) -> tuple[
    list[_IdentityTrack], list[dict[str, Any]], list[dict[str, Any]],
]:
    """Create a nameable roster when format is known but identity tracks are sparse."""
    if match_format == "doubles":
        identities = [
            _IdentityTrack("P1", "T1", "near"),
            _IdentityTrack("P2", "T1", "near"),
            _IdentityTrack("P3", "T2", "far"),
            _IdentityTrack("P4", "T2", "far"),
        ]
        teams = [
            {"id": "T1", "player_ids": ["P1", "P2"]},
            {"id": "T2", "player_ids": ["P3", "P4"]},
        ]
    else:
        identities = [
            _IdentityTrack("P1", "T1", "near"),
            _IdentityTrack("P2", "T2", "far"),
        ]
        teams = [
            {"id": "T1", "player_ids": ["P1"]},
            {"id": "T2", "player_ids": ["P2"]},
        ]
    roster = [
        {
            "id": identity.player_id,
            "name": f"Player {identity.player_id[1:]}",
            "team_id": identity.team_id,
            "initial_end": identity.initial_end,
        }
        for identity in identities
    ]
    return identities, roster, teams


def _select_anchor_detections(
    samples: Sequence[tuple[float, list[_Detection]]], match_format: str,
    segments: Sequence[Segment] = (),
) -> tuple[int, list[_Detection]] | None:
    per_end = 2 if match_format == "doubles" else 1
    candidates: list[tuple[tuple[float, ...], int, list[_Detection]]] = []
    for index, (time_s, detections) in enumerate(samples):
        if not _inside_any(time_s, segments):
            continue
        near = sorted((d for d in detections if d.y < NET_Y), key=lambda d: d.area, reverse=True)
        far = sorted((d for d in detections if d.y >= NET_Y), key=lambda d: d.area, reverse=True)
        if len(near) < per_end or len(far) < per_end:
            continue
        selected = [*near[:per_end], *far[:per_end]]
        candidates.append(((sum(d.area for d in selected), -float(index)), index, selected))
    if not candidates:
        return None
    _score, index, selected = max(candidates, key=lambda item: item[0])
    return index, selected


def _best_assignment(previous: Sequence[np.ndarray], detections: Sequence[_Detection]):
    """Rectangular Hungarian assignment for the two/four persistent identities."""
    if not previous or not detections:
        return []
    from scipy.optimize import linear_sum_assignment

    current = np.asarray(previous, dtype=float).reshape(-1, 2)
    observed = np.asarray([[detection.x, detection.y] for detection in detections],
                          dtype=float).reshape(-1, 2)
    cost = np.linalg.norm(current[:, None, :] - observed[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(cost)
    return list(zip((int(value) for value in rows),
                    (int(value) for value in columns)))


def _build_identity_tracks(
    samples: Sequence[tuple[float, list[_Detection]]], match_format: str,
    segments: Sequence[Segment] = (),
) -> tuple[list[_IdentityTrack], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor = _select_anchor_detections(samples, match_format, segments)
    if anchor is None:
        return [], [], []
    anchor_index, selected = anchor
    near = sorted((d for d in selected if d.y < NET_Y), key=lambda d: d.x)
    far = sorted((d for d in selected if d.y >= NET_Y), key=lambda d: d.x)
    ordered = [*near, *far]
    if match_format == "doubles":
        identities = [
            _IdentityTrack("P1", "T1", "near"),
            _IdentityTrack("P2", "T1", "near"),
            _IdentityTrack("P3", "T2", "far"),
            _IdentityTrack("P4", "T2", "far"),
        ]
    else:
        identities = [
            _IdentityTrack("P1", "T1", "near"),
            _IdentityTrack("P2", "T2", "far"),
        ]
    anchor_time = samples[anchor_index][0]
    for identity, detection in zip(identities, ordered):
        identity.add(anchor_time, detection.x, detection.y)

    def walk(indices: Iterable[int]) -> None:
        current = [np.array(identity.records[-1][1:], dtype=float) for identity in identities]
        prior_time = anchor_time
        for index in indices:
            time_s, detections = samples[index]
            if not detections:
                continue
            chosen = _best_assignment(current, detections)
            dt = abs(float(time_s - prior_time))
            max_jump = min(16.0, 2.5 + 8.0 * max(dt, 0.25))
            updated = False
            for track_index, detection_index in chosen:
                detection = detections[detection_index]
                position = np.array([detection.x, detection.y], dtype=float)
                if float(np.linalg.norm(current[track_index] - position)) > max_jump:
                    continue
                identities[track_index].add(time_s, detection.x, detection.y)
                current[track_index] = position
                updated = True
            if updated:
                prior_time = time_s

    walk(range(anchor_index + 1, len(samples)))
    # Track backwards as well so an anchor chosen for visibility in the middle of a match
    # still provides identities to early points.
    for identity in identities:
        identity.records.sort()
    original_records = [list(identity.records) for identity in identities]
    for identity, records in zip(identities, original_records):
        identity.records = [min(records, key=lambda record: abs(record[0] - anchor_time))]
    walk(range(anchor_index - 1, -1, -1))
    for identity, records in zip(identities, original_records):
        identity.records.extend(records)
        identity.records = sorted(set(identity.records))

    roster = [
        {
            "id": identity.player_id,
            "name": f"Player {identity.player_id[1:]}",
            "team_id": identity.team_id,
            "initial_end": identity.initial_end,
        }
        for identity in identities
    ]
    teams = (
        [{"id": "T1", "player_ids": ["P1", "P2"]},
         {"id": "T2", "player_ids": ["P3", "P4"]}]
        if match_format == "doubles" else
        [{"id": "T1", "player_ids": ["P1"]},
         {"id": "T2", "player_ids": ["P2"]}]
    )
    return identities, roster, teams


def _raw_point_track(track_cache: Sequence, start: float, end: float) -> Optional[BallTrack]:
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _region, track in track_cache:
        times = np.asarray(getattr(track, "t", []), dtype=float)
        if not times.size:
            continue
        mask = (times >= start - 0.15) & (times <= end + 0.15)
        if np.any(mask):
            pieces.append((times[mask], np.asarray(track.x, float)[mask],
                           np.asarray(track.y, float)[mask]))
    if not pieces:
        return None
    times = np.concatenate([piece[0] for piece in pieces])
    x = np.concatenate([piece[1] for piece in pieces])
    y = np.concatenate([piece[2] for piece in pieces])
    order = np.argsort(times, kind="stable")
    times, x, y = times[order], x[order], y[order]
    unique = np.r_[True, np.diff(times) > 1e-9]
    times, x, y = times[unique], x[unique], y[unique]
    if times.size < 3:
        return None
    return BallTrack(times, x, y)


def _court_ball(track: SmoothTrack, court) -> np.ndarray:
    try:
        return np.asarray(court.to_court(np.column_stack((track.x, track.y))), float)
    except Exception:
        return np.full((track.t.size, 2), np.nan, dtype=float)


def _line_state(x: float, y: float, match_format: str, uncertainty_m: float = 0.35) -> str:
    x0 = SINGLES_IN if match_format == "singles" else 0.0
    x1 = DOUBLES_W - SINGLES_IN if match_format == "singles" else DOUBLES_W
    outside = max(x0 - x, x - x1, -y, y - COURT_L)
    if outside > uncertainty_m:
        return "out"
    distance = min(abs(x - x0), abs(x - x1), abs(y), abs(y - COURT_L))
    if outside > 0.0 or distance <= uncertainty_m:
        return "uncertain"
    return "in"


def _service_box_state(
    x: float, y: float, server_position: Optional[np.ndarray], server_end: Optional[str],
    uncertainty_m: float = 0.35,
) -> str:
    if server_position is None or server_end not in {"near", "far"}:
        return "unknown"
    if server_end == "near":
        y0, y1 = NET_Y, COURT_L - SERVICE_Y
    else:
        y0, y1 = SERVICE_Y, NET_Y
    # A legal service lands diagonally across the centre service line.
    if server_position[0] < DOUBLES_W / 2.0:
        x0, x1 = DOUBLES_W / 2.0, DOUBLES_W - SINGLES_IN
    else:
        x0, x1 = SINGLES_IN, DOUBLES_W / 2.0
    outside = max(x0 - x, x - x1, y0 - y, y - y1)
    if outside > uncertainty_m:
        return "fault"
    distance = min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1))
    if outside > 0.0 or distance <= uncertainty_m:
        return "uncertain"
    return "in"


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))


def _match_group(point: Segment, match_state: dict[str, Any]) -> Optional[dict[str, Any]]:
    groups = [group for group in (match_state.get("logical_groups") or [])
              if group.get("decision") == "keep" and group.get("output")]
    return max(groups, key=lambda group: _overlap(point, group["output"]), default=None)


def _serve_context(
    point: Segment, match_state: dict[str, Any], court_points: np.ndarray,
    track: Optional[SmoothTrack], identities: Sequence[_IdentityTrack],
) -> tuple[Optional[dict[str, Any]], float, Optional[str], Optional[np.ndarray],
           Optional[_IdentityTrack], float]:
    group = _match_group(point, match_state)
    contact = float(group.get("serve_contact")) if group and group.get("serve_contact") is not None else point[0]
    observations = match_state.get("observations") or []
    observation = None
    if group and group.get("serve_member_index") is not None:
        index = int(group["serve_member_index"])
        if 0 <= index < len(observations):
            observation = observations[index]
    server_end = observation.get("position_server_end") if observation else None
    if server_end not in {"near", "far"} and track is not None and court_points.size:
        index = int(np.argmin(np.abs(np.asarray(track.t, float) - contact)))
        if np.isfinite(court_points[index]).all():
            server_end = side_of(float(court_points[index, 1]))

    candidates: list[tuple[float, _IdentityTrack, np.ndarray]] = []
    ball_position = None
    if track is not None and court_points.size:
        index = int(np.argmin(np.abs(np.asarray(track.t, float) - contact)))
        if np.isfinite(court_points[index]).all():
            ball_position = court_points[index]
    for identity in identities:
        position = identity.position(contact)
        if position is None:
            continue
        end = "near" if position[1] < NET_Y else "far"
        if server_end is not None and end != server_end:
            continue
        baseline_distance = abs(position[1] if end == "near" else COURT_L - position[1])
        ball_distance = (float(np.linalg.norm(position - ball_position))
                         if ball_position is not None else 4.0)
        candidates.append((0.7 * baseline_distance + 0.3 * ball_distance,
                           identity, position))
    if not candidates:
        return group, contact, server_end, None, None, 0.0
    score, server, position = min(candidates, key=lambda item: item[0])
    confidence = float(np.clip(1.0 - score / 8.0, 0.25, 0.95))
    return group, contact, server_end, position, server, confidence


def _team_for_side(
    identities: Sequence[_IdentityTrack], time_s: float, side: str,
) -> Optional[str]:
    votes: dict[str, int] = {}
    for identity in identities:
        position = identity.position(time_s)
        if position is None or side_of(float(position[1])) != side:
            continue
        votes[identity.team_id] = votes.get(identity.team_id, 0) + 1
    return max(votes, key=votes.get) if votes else None


def _other_team(team_id: Optional[str], teams: Sequence[dict[str, Any]]) -> Optional[str]:
    if team_id is None:
        return None
    others = [str(team["id"]) for team in teams if team.get("id") != team_id]
    return others[0] if len(others) == 1 else None


def _contact_turn_score(track: SmoothTrack, index: int, radius: int = 3) -> float:
    if index < 1 or index >= track.t.size - 1:
        return 0.0
    pre = np.array([np.nanmedian(track.vx[max(0, index - radius):index]),
                    np.nanmedian(track.vy[max(0, index - radius):index])])
    post = np.array([np.nanmedian(track.vx[index:min(track.t.size, index + radius)]),
                     np.nanmedian(track.vy[index:min(track.t.size, index + radius)])])
    a, b = float(np.linalg.norm(pre)), float(np.linalg.norm(post))
    if min(a, b) <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(pre, post) / (a * b), -1.0, 1.0))
    return float(np.clip(np.degrees(np.arccos(cosine)) / 90.0, 0.0, 1.0))


def _assign_contact_player(
    time_s: float, ball_position: np.ndarray, identities: Sequence[_IdentityTrack],
    track_confidence: float, turn_score: float,
) -> tuple[Optional[_IdentityTrack], float, Optional[float]]:
    candidates: list[tuple[float, _IdentityTrack]] = []
    ball_side = side_of(float(ball_position[1]))
    for identity in identities:
        position = identity.position(time_s, max_gap_s=1.5)
        if position is None or side_of(float(position[1])) != ball_side:
            continue
        candidates.append((float(np.linalg.norm(position - ball_position)), identity))
    if not candidates:
        return None, 0.0, None
    distance, identity = min(candidates, key=lambda item: item[0])
    proximity = float(np.clip(1.0 - distance / 6.0, 0.0, 1.0))
    confidence = 0.50 * proximity + 0.30 * track_confidence + 0.20 * turn_score
    if confidence < 0.38:
        return None, confidence, distance
    return identity, float(np.clip(confidence, 0.0, 0.98)), distance


def _point_contacts(
    point: Segment, onsets: np.ndarray, track: Optional[SmoothTrack],
    court_points: np.ndarray, identities: Sequence[_IdentityTrack],
    serve_contact: float, server: Optional[_IdentityTrack], server_confidence: float,
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    if server is not None:
        contacts.append({
            "time": round(float(serve_contact), 3), "player_id": server.player_id,
            "team_id": server.team_id, "kind": "serve",
            "confidence": round(float(server_confidence), 4),
            "evidence": ["serve_state", "baseline_player", "ball_origin"],
        })
    if track is None or not court_points.size:
        return contacts
    times = np.asarray(track.t, float)
    for onset in np.sort(np.asarray(onsets, float)):
        onset = float(onset)
        if onset < max(point[0], serve_contact + 0.22) or onset > point[1]:
            continue
        index = int(np.argmin(np.abs(times - onset)))
        delta = abs(float(times[index]) - onset)
        if delta > 0.14 or not np.isfinite(court_points[index]).all():
            continue
        confidence = float(track.confidence[index])
        if confidence < 0.20:
            continue
        turn = _contact_turn_score(track, index)
        identity, actor_confidence, distance = _assign_contact_player(
            onset, court_points[index], identities, confidence, turn)
        if identity is None:
            continue
        if contacts and onset - float(contacts[-1]["time"]) < 0.22:
            if actor_confidence <= float(contacts[-1]["confidence"]):
                continue
            contacts.pop()
        contacts.append({
            "time": round(onset, 3), "player_id": identity.player_id,
            "team_id": identity.team_id, "kind": "racket_contact",
            "confidence": round(actor_confidence, 4),
            "ball_player_distance_m": round(float(distance), 3) if distance is not None else None,
            "evidence": ["audio_impact", "ball_motion_turn", "player_proximity"],
        })
    return contacts


def _point_bounces(
    point: Segment, track: Optional[SmoothTrack], court_points: np.ndarray,
    match_format: str,
) -> list[dict[str, Any]]:
    if track is None or not court_points.size:
        return []
    output: list[dict[str, Any]] = []
    for index in bounces_from_velocity(track):
        time_s = float(track.t[index])
        if not (point[0] <= time_s <= point[1]) or not np.isfinite(court_points[index]).all():
            continue
        x, y = (float(court_points[index, 0]), float(court_points[index, 1]))
        output.append({
            "time": round(time_s, 3), "x_m": round(x, 3), "y_m": round(y, 3),
            "side": side_of(y), "in_state": _line_state(x, y, match_format),
            "uncertainty_m": 0.35,
            "confidence": round(float(track.confidence[index]), 4),
        })
    return output


def _net_crossings(track: Optional[SmoothTrack], court_points: np.ndarray,
                   point: Segment) -> list[float]:
    if track is None or not court_points.size:
        return []
    output: list[float] = []
    for index in range(1, track.t.size):
        if not (point[0] <= track.t[index] <= point[1]):
            continue
        y0, y1 = court_points[index - 1, 1], court_points[index, 1]
        if (np.isfinite(y0) and np.isfinite(y1)
                and track.confidence[index - 1] >= 0.25
                and track.confidence[index] >= 0.25
                and (y0 - NET_Y) * (y1 - NET_Y) < 0):
            output.append(round(float(track.t[index]), 3))
    return output


def _probable_net_failure(
    point: Segment, track: Optional[SmoothTrack], court_points: np.ndarray,
    last_contact: Optional[dict[str, Any]], crossings: Sequence[float],
) -> Optional[tuple[float, float]]:
    if track is None or last_contact is None or not court_points.size:
        return None
    contact_time = float(last_contact["time"])
    if any(crossing > contact_time for crossing in crossings):
        return None
    mask = ((track.t >= contact_time) & (track.t <= point[1])
            & (track.confidence >= 0.25) & np.isfinite(court_points).all(axis=1))
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        return None
    start_side = side_of(float(court_points[indices[0], 1]))
    if any(side_of(float(court_points[index, 1])) != start_side for index in indices):
        return None
    closest = int(indices[np.argmin(np.abs(court_points[indices, 1] - NET_Y))])
    distance = abs(float(court_points[closest, 1] - NET_Y))
    if distance > 1.25:
        return None
    confidence = float(np.clip(0.80 - 0.35 * distance / 1.25, 0.45, 0.80))
    return float(track.t[closest]), confidence


def _termination(
    point: Segment, group: Optional[dict[str, Any]], serve_contact: float,
    server: Optional[_IdentityTrack], server_position: Optional[np.ndarray],
    server_end: Optional[str], contacts: Sequence[dict[str, Any]],
    bounces: Sequence[dict[str, Any]], crossings: Sequence[float],
    net_failure: Optional[tuple[float, float]], identities: Sequence[_IdentityTrack],
    teams: Sequence[dict[str, Any]], match_format: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    server_team = server.team_id if server is not None else None
    last_contact = contacts[-1] if contacts else None
    last_hitter_id = last_contact.get("player_id") if last_contact else None
    last_hitter_team = last_contact.get("team_id") if last_contact else None
    opponent_contacts = [contact for contact in contacts
                         if contact.get("team_id") not in {None, server_team}]
    service_bounce = next((bounce for bounce in bounces
                           if float(bounce["time"]) > serve_contact + 0.08), None)
    service_state = "unknown"
    if service_bounce is not None:
        service_state = _service_box_state(
            float(service_bounce["x_m"]), float(service_bounce["y_m"]),
            server_position, server_end)
    retry = bool(group and group.get("retry_serve_detected"))
    attempts: list[dict[str, Any]] = []
    attempt_times = (group.get("retry_serve_contacts") if retry and group else None) or [serve_contact]
    for index, attempt_time in enumerate(attempt_times):
        final_attempt = index == len(attempt_times) - 1
        attempts.append({
            "number": index + 1,
            "contact_time": round(float(attempt_time), 3),
            "result": (service_state if final_attempt else "retry_required"),
            "confidence": (round(float(service_bounce.get("confidence", 0.0)), 4)
                           if final_attempt and service_bounce else None),
        })

    rule_event = "unknown"
    credit = "unknown"
    event_time: Optional[float] = None
    confidence = 0.0
    error_player_id = None
    forcing_player_id = None
    winner_player_id = None
    winner_team_id = None
    evidence: list[str] = []

    # The first clearly out landing after the most recent racket contact owns the point.
    out_bounce = next((bounce for bounce in bounces
                       if bounce["in_state"] == "out"
                       and (last_contact is None
                            or float(bounce["time"]) >= float(last_contact["time"]))), None)
    second_bounce = None
    for previous, current in zip(bounces, bounces[1:]):
        if (current["side"] == previous["side"]
                and float(current["time"]) - float(previous["time"]) <= 2.5
                and current["in_state"] != "out"
                and not any(
                    float(previous["time"]) < float(contact["time"])
                    < float(current["time"])
                    for contact in contacts
                )):
            second_bounce = current
            break

    second_serve_failed = bool(
        retry and not opponent_contacts
        and (service_state == "fault" or net_failure is not None)
    )
    if second_serve_failed:
        rule_event, credit = "double_fault", "error_unknown"
        event_time = (float(service_bounce["time"]) if service_state == "fault" and service_bounce
                      else float(net_failure[0]))
        confidence = min(0.97, 0.72 + 0.20 * float(service_bounce is not None))
        error_player_id = server.player_id if server is not None else None
        winner_team_id = _other_team(server_team, teams)
        evidence = ["two_service_attempts", "failed_second_service", "no_receiver_contact"]
    elif out_bounce is not None:
        # A lone failed first serve is not a completed tennis point. Keep the observation
        # but abstain from winner attribution until a second attempt is found.
        if (not retry and not opponent_contacts and service_bounce is out_bounce
                and service_state == "fault"):
            evidence = ["first_service_fault", "second_service_not_observed"]
        else:
            rule_event, credit = "out", "error_unknown"
            event_time = float(out_bounce["time"])
            confidence = float(out_bounce.get("confidence", 0.0))
            error_player_id = last_hitter_id
            winner_team_id = _other_team(last_hitter_team, teams)
            evidence = ["measured_ball_bounce", "landing_outside_court"]
    elif net_failure is not None:
        rule_event, credit = "net_failure", "error_unknown"
        event_time, confidence = net_failure
        error_player_id = last_hitter_id
        winner_team_id = _other_team(last_hitter_team, teams)
        evidence = ["post_contact_net_approach", "no_net_crossing", "same_side_track_end"]
    elif second_bounce is not None:
        rule_event = "second_bounce"
        event_time = float(second_bounce["time"])
        confidence = float(second_bounce.get("confidence", 0.0))
        losing_team = _team_for_side(
            identities, event_time, str(second_bounce["side"]))
        winner_team_id = _other_team(losing_team, teams)
        if (server is not None and not opponent_contacts and service_state == "in"
                and last_hitter_id == server.player_id):
            credit = "ace"
            winner_player_id = server.player_id
        elif last_hitter_team is not None and winner_team_id == last_hitter_team:
            credit = "clean_winner"
            winner_player_id = last_hitter_id
        evidence = ["two_same_side_bounces", "no_intervening_racket_contact"]

    if winner_team_id is not None and winner_player_id is None and match_format == "singles":
        members = next((team.get("player_ids", []) for team in teams
                        if team.get("id") == winner_team_id), [])
        winner_player_id = members[0] if len(members) == 1 else None
    if credit == "error_unknown" and len(contacts) >= 2:
        prior = contacts[-2]
        if prior.get("team_id") != last_hitter_team:
            forcing_player_id = prior.get("player_id")

    confidence = float(np.clip(confidence, 0.0, 0.99))
    status = "confirmed" if confidence >= 0.85 else ("probable" if confidence >= 0.55 else "unknown")
    return ({
        "time": round(event_time, 3) if event_time is not None else None,
        "rule_event": rule_event,
        "credit": credit,
        "last_hitter_id": last_hitter_id,
        "winner_player_id": winner_player_id,
        "winner_team_id": winner_team_id,
        "error_player_id": error_player_id,
        "forcing_player_id": forcing_player_id,
        "confidence": round(confidence, 4),
        "status": status,
        "evidence": evidence,
        "alternatives": ([] if rule_event != "unknown" else [
            {"rule_event": "unknown", "reason": "insufficient calibrated terminal evidence"}
        ]),
    }, attempts)


def analyse_point_outcomes(
    segments: Sequence[Segment], *, track_cache: Sequence, court,
    player_samples: Sequence, frame_size: Optional[tuple[int, int]],
    onsets: np.ndarray, match_state: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return automatic match metadata and rule-decoded point records."""
    match_state = match_state or {}
    format_evidence = infer_match_format(
        player_samples, court, frame_size, segments)
    if format_evidence.get("format") == "unknown":
        fallback = _match_state_format_fallback(match_state)
        if fallback.get("format") != "unknown":
            format_evidence = fallback
    match_format = str(format_evidence["format"])
    samples = _court_player_samples(player_samples, court, frame_size)
    identities: list[_IdentityTrack] = []
    roster: list[dict[str, Any]] = []
    teams: list[dict[str, Any]] = []
    identity_method = "court_track_continuity"
    if match_format in {"singles", "doubles"}:
        identities, roster, teams = _build_identity_tracks(samples, match_format, segments)
        if not identities:
            identities, roster, teams = _default_identities(match_format)
            identity_method = "format_only_identity_unresolved"
    match = {
        "schema_version": POINT_SCHEMA_VERSION,
        "format": match_format,
        "format_confidence": format_evidence.get("confidence", 0.0),
        "format_evidence": format_evidence,
        "roster": roster,
        "teams": teams,
        "identity_method": identity_method,
    }
    points: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(segments):
        point = (float(start), float(end))
        raw_track = _raw_point_track(track_cache, start, end)
        track = smooth_track(raw_track) if raw_track is not None else None
        court_points = (_court_ball(track, court)
                        if track is not None and court is not None else np.empty((0, 2)))
        (group, serve_contact, server_end, server_position,
         server, server_confidence) = _serve_context(
            point, match_state, court_points, track, identities)
        contacts = _point_contacts(
            point, onsets, track, court_points, identities,
            serve_contact, server, server_confidence)
        bounces = _point_bounces(point, track, court_points, match_format)
        crossings = _net_crossings(track, court_points, point)
        net_failure = _probable_net_failure(
            point, track, court_points, contacts[-1] if contacts else None, crossings)
        termination, attempts = _termination(
            point, group, serve_contact, server, server_position, server_end,
            contacts, bounces, crossings, net_failure, identities, teams, match_format)
        server_team = server.team_id if server is not None else None
        receivers = [identity.player_id for identity in identities
                     if server_team is not None and identity.team_id != server_team]
        points.append({
            "index": index, "start": round(start, 3), "end": round(end, 3),
            "participants": {
                "server_id": server.player_id if server is not None else None,
                "server_team_id": server_team,
                "receiver_ids": receivers,
                "server_end": server_end,
                "server_confidence": round(server_confidence, 4),
            },
            "attempts": attempts,
            "contacts": contacts,
            "bounces": bounces,
            "net_crossings": crossings,
            "termination": termination,
        })
    return match, points
