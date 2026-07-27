import numpy as np

from rally.signals.player import (
    estimate_court_region,
    geometry_score_from_persons,
)


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
