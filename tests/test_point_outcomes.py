import numpy as np

from rally.fusion.point_outcomes import (
    _IdentityTrack,
    _termination,
    analyse_point_outcomes,
    infer_match_format,
)


class IdentityCourt:
    def to_court(self, points):
        return np.asarray(points, dtype=float).reshape(-1, 2)


def _person(x_m, y_m, area=0.1):
    return (x_m / 100.0, y_m / 100.0, area)


def test_match_format_is_automatic_and_aggregates_across_frames():
    singles = [
        (float(index), [_person(5.0, 2.0), _person(5.0, 21.0)])
        for index in range(8)
    ]
    detected = infer_match_format(
        singles, IdentityCourt(), (100, 100), [(0.0, 8.0)])
    assert detected["format"] == "singles"
    assert detected["player_count"] == 2

    doubles = [
        (float(index), [
            _person(3.0, 2.0), _person(8.0, 5.0),
            _person(3.0, 19.0), _person(8.0, 22.0),
        ])
        for index in range(8)
    ]
    detected = infer_match_format(
        doubles, IdentityCourt(), (100, 100), [(0.0, 8.0)])
    assert detected["format"] == "doubles"
    assert detected["player_count"] == 4
    assert detected["confidence"] > 0.8


def test_match_format_falls_back_to_stable_serve_window_occupancy():
    match_state = {"observations": [
        {
            "position_checked": True,
            "target_court_filtered": True,
            "position_stable_tracks": count,
        }
        for count in (4, 3, 4, 2)
    ]}

    match, points = analyse_point_outcomes(
        [], track_cache=[], court=None, player_samples=[], frame_size=None,
        onsets=np.array([]), match_state=match_state,
    )

    assert points == []
    assert match["format"] == "doubles"
    assert match["format_evidence"]["source"] == (
        "target_court_serve_window_stable_occupancy")
    assert [player["id"] for player in match["roster"]] == ["P1", "P2", "P3", "P4"]


def _singles_tracks():
    near = _IdentityTrack("P1", "T1", "near", [(0.0, 8.0, 0.0), (5.0, 8.0, 0.0)])
    far = _IdentityTrack("P2", "T2", "far", [(0.0, 3.0, 23.0), (5.0, 3.0, 23.0)])
    teams = [
        {"id": "T1", "player_ids": ["P1"]},
        {"id": "T2", "player_ids": ["P2"]},
    ]
    return near, far, teams


def test_second_failed_service_is_classified_as_double_fault():
    server, receiver, teams = _singles_tracks()
    termination, attempts = _termination(
        (0.0, 4.0),
        {"retry_serve_detected": True, "retry_serve_contacts": [0.5, 2.0]},
        2.0, server, np.array([8.0, 0.0]), "near",
        [{"time": 2.0, "player_id": "P1", "team_id": "T1", "kind": "serve"}],
        [{"time": 2.7, "x_m": 8.0, "y_m": 20.0, "side": "far",
          "in_state": "in", "confidence": 0.9}],
        [], None, [server, receiver], teams, "singles",
    )
    assert termination["rule_event"] == "double_fault"
    assert termination["error_player_id"] == "P1"
    assert termination["winner_player_id"] == "P2"
    assert [attempt["number"] for attempt in attempts] == [1, 2]
    assert attempts[0]["result"] == "retry_required"
    assert attempts[1]["result"] == "fault"


def test_out_ball_credits_error_to_last_hitter_and_winner_to_opponent():
    near, far, teams = _singles_tracks()
    contacts = [
        {"time": 0.5, "player_id": "P1", "team_id": "T1", "kind": "serve"},
        {"time": 2.0, "player_id": "P2", "team_id": "T2", "kind": "racket_contact"},
    ]
    termination, _attempts = _termination(
        (0.0, 4.0), None, 0.5, near, np.array([8.0, 0.0]), "near",
        contacts,
        [{"time": 2.8, "x_m": 12.0, "y_m": 4.0, "side": "near",
          "in_state": "out", "confidence": 0.93}],
        [], None, [near, far], teams, "singles",
    )
    assert termination["rule_event"] == "out"
    assert termination["credit"] == "error_unknown"
    assert termination["error_player_id"] == "P2"
    assert termination["winner_player_id"] == "P1"


def test_unknown_terminal_evidence_does_not_invent_a_winner():
    near, far, teams = _singles_tracks()
    termination, _attempts = _termination(
        (0.0, 4.0), None, 0.5, near, np.array([8.0, 0.0]), "near",
        [{"time": 0.5, "player_id": "P1", "team_id": "T1", "kind": "serve"}],
        [], [], None, [near, far], teams, "singles",
    )
    assert termination["rule_event"] == "unknown"
    assert termination["winner_player_id"] is None
    assert termination["winner_team_id"] is None
