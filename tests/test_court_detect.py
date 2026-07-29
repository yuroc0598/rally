import numpy as np
import pytest

from rally.signals.court import Court
from rally.signals.court_detect import (
    classify_lines,
    corners_from_lines,
    line_intersection,
    normalize_hough_lines,
    score_court,
    valid_court_quad,
)

cv2 = pytest.importorskip("cv2")


def test_normalize_hough_lines_accepts_both_opencv_shapes():
    flat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.int32)
    nested = flat[:, None, :]
    assert normalize_hough_lines(flat) == normalize_hough_lines(nested)
    assert normalize_hough_lines(flat) == [(1.0, 2.0, 3.0, 4.0),
                                            (5.0, 6.0, 7.0, 8.0)]


def test_valid_quad_rejects_collapsed_far_baseline():
    shape = (1080, 1920, 3)
    assert valid_court_quad([(20, 840), (1900, 825), (1220, 510), (840, 510)], shape)
    assert not valid_court_quad([(20, 840), (1900, 825), (1220, 510), (1160, 510)], shape)


def test_line_intersection():
    assert line_intersection((0, 0, 10, 0), (5, -5, 5, 5)) == pytest.approx((5.0, 0.0))
    assert line_intersection((0, 0, 10, 0), (0, 1, 10, 1)) is None   # parallel


def test_classify_lines_splits_h_and_v():
    horiz, vert = classify_lines([
        (0, 0, 100, 3),      # ~horizontal
        (0, 0, 3, 100),      # ~vertical
        (0, 0, 100, 100),    # 45 deg -> neither
    ])
    assert len(horiz) == 1 and len(vert) == 1


def test_corners_from_lines_picks_extremes():
    horiz = [(0, 800, 1200, 800), (0, 300, 1200, 300)]      # near (y800) & far (y300) baselines
    vert = [(200, 0, 350, 1000), (1000, 0, 850, 1000)]      # left & right sidelines
    corners = corners_from_lines(horiz, vert)
    assert corners is not None
    nl, nr, fr, fl = corners
    assert nl[1] > fl[1] and nr[0] > nl[0]                  # near below far, right of left


def _synthetic_court():
    # a plausible perspective trapezoid of the outer doubles court
    nl, nr, fr, fl = (250, 850), (1050, 850), (900, 320), (400, 320)
    return Court.from_image_corners(nl, nr, fr, fl)


def test_score_court_high_on_matching_mask():
    court = _synthetic_court()
    mask = np.zeros((1080, 1280), np.uint8)
    from rally.signals.court import court_model_polylines
    for seg in court_model_polylines():
        p0 = court.to_image([seg[0]])[0]
        p1 = court.to_image([seg[1]])[0]
        cv2.line(mask, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 255, 2)
    good = score_court(mask, court)
    assert good > 0.9

    # a court shifted 40 px sideways should overlap far worse
    shifted = Court.from_image_corners((290, 850), (1090, 850), (940, 320), (440, 320))
    assert score_court(mask, shifted) < good


def test_from_image_corners_roundtrip():
    court = _synthetic_court()
    from rally.signals.court import DOUBLES_W, COURT_L
    # the four model corners map back to the image corners we supplied
    img = court.to_image([[0, 0], [DOUBLES_W, 0], [DOUBLES_W, COURT_L], [0, COURT_L]])
    assert img[0] == pytest.approx((250, 850), abs=1.0)
    assert img[2] == pytest.approx((900, 320), abs=1.0)
