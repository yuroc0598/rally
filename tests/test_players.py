import numpy as np

from rally.signals.player import (
    clean_track,
    estimate_court_region,
    geometry_score_from_court_persons,
    geometry_score_from_persons,
    persons_in_court,
    target_court_box_indices,
)
from rally.signals.court import Court


def test_estimate_court_region_from_clustered_feet():
    rng = np.random.default_rng(0)
    feet = rng.uniform(0.3, 0.7, size=(200, 2))  # players clustered centrally
    region = estimate_court_region(list(map(tuple, feet)))
    assert region is not None
    x0, y0, x1, y1 = region
    assert 0.2 < x0 < 0.4 and 0.6 < x1 < 0.8


def test_estimate_court_region_too_few_points():
    assert estimate_court_region([(0.5, 0.5)]) is None


def test_geometry_two_opposed_players_scores_high():
    region = (0.2, 0.2, 0.8, 0.8)  # mid at y=0.5
    persons = [(0.5, 0.35, 0.01), (0.5, 0.65, 0.01)]  # opposite halves
    assert geometry_score_from_persons(persons, region) == 1.0


def test_geometry_two_players_same_half():
    region = (0.2, 0.2, 0.8, 0.8)
    persons = [(0.4, 0.3, 0.01), (0.6, 0.35, 0.01)]  # both above mid
    assert geometry_score_from_persons(persons, region) == 0.5


def test_geometry_crowd_and_single_and_empty():
    region = (0.2, 0.2, 0.8, 0.8)
    assert geometry_score_from_persons(
        [(0.3, 0.3, 0), (0.4, 0.4, 0), (0.6, 0.6, 0)], region) == 0.3
    assert geometry_score_from_persons([(0.5, 0.5, 0)], region) == 0.2
    assert geometry_score_from_persons([], region) == 0.0


def test_geometry_ignores_persons_outside_court():
    region = (0.4, 0.4, 0.6, 0.6)
    persons = [(0.5, 0.45, 0), (0.05, 0.05, 0), (0.95, 0.95, 0)]  # 2 spectators outside
    assert geometry_score_from_persons(persons, region) == 0.2  # only 1 inside


def test_geometry_none_region():
    assert geometry_score_from_persons([(0.5, 0.5, 0)], None) == 0.0


def test_target_court_filter_excludes_neighboring_court_people():
    # Target court is the central trapezoid; all detections remain inside the image, but
    # the two people at the side belong to neighboring courts/spectator space.
    court = Court.from_image_corners((20, 90), (80, 90), (65, 20), (35, 20))
    people = [
        (0.50, 0.82, 0.02),  # target near player
        (0.50, 0.28, 0.01),  # target far player
        (0.05, 0.55, 0.02),  # left adjacent court
        (0.95, 0.55, 0.02),  # right adjacent court
    ]
    kept, coordinates = persons_in_court(
        people, court, (100, 100), sideline_margin_m=0.2, baseline_margin_m=0.2)
    assert kept == people[:2]
    assert coordinates.shape == (2, 2)
    assert geometry_score_from_court_persons(
        people, court, (100, 100), sideline_margin_m=0.2,
        baseline_margin_m=0.2) == 1.0


def test_neighboring_players_cannot_create_target_geometry_vote():
    court = Court.from_image_corners((20, 90), (80, 90), (65, 20), (35, 20))
    neighbors = [(0.05, 0.30, 0.02), (0.95, 0.75, 0.02)]
    assert geometry_score_from_court_persons(
        neighbors, court, (100, 100), sideline_margin_m=0.2,
        baseline_margin_m=0.2) == 0.0


def test_pose_boxes_are_filtered_to_target_court_before_selection():
    court = Court.from_image_corners((20, 90), (80, 90), (65, 20), (35, 20))
    boxes = np.array([
        [45, 45, 55, 82],   # target-court player
        [0, 30, 10, 60],    # left neighboring court
        [90, 30, 100, 60],  # right neighboring court
    ], dtype=float)
    assert target_court_box_indices(
        boxes, court, (100, 100), sideline_margin_m=0.2,
        baseline_margin_m=0.2) == [0]


def test_clean_track_preserves_edges_and_long_detection_gaps_as_missing():
    times = np.arange(8, dtype=float)
    x = np.array([np.nan, 0.1, np.nan, 0.3, np.nan, np.nan, 0.6, np.nan])
    y = x.copy()

    clean_x, clean_y = clean_track(
        times, x, y, speed_limit_mps=10.0, smooth_win=1, max_gap_dt_s=2.5)

    assert np.isnan(clean_x[0]) and np.isnan(clean_x[-1])
    assert np.isclose(clean_x[2], 0.2)
    assert np.isnan(clean_x[4:6]).all()
    assert np.array_equal(np.isnan(clean_x), np.isnan(clean_y))
