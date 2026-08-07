"""Court geometry: homography between image pixels and real court metres.

The camera is fixed, so calibrate ONCE (:meth:`Court.calibrate`) from four clearly-visible
near-side points — the two near-baseline corners and where the net meets the two doubles
sidelines — then reuse the homography. A four-point planar homography is exact across the
whole court plane, so far-side coordinates follow from the fixed court dimensions even when
the far lines aren't visible.

Court model (metres), origin at the near-left DOUBLES corner, x across, y toward far end:
    near-left (0,0)  near-right (10.97,0)  far-right (10.97,23.77)  far-left (0,23.77)
Singles sidelines are at x=1.37 and x=9.60; service lines are at y=5.4864 and
y=18.2836; net y=11.885.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# court constants (metres)
DOUBLES_W = 10.97
COURT_L = 23.77
SINGLES_IN = 1.37          # singles sideline inset from doubles line
SERVICE_Y = 5.4864         # 18 ft: service line distance from each baseline
NET_Y = COURT_L / 2.0


@dataclass
class Court:
    H_img2court: np.ndarray   # 3x3, image px -> court metres
    H_court2img: np.ndarray   # inverse
    corners_img: np.ndarray   # 4x2 detected image corners (NL, NR, FR, FL)

    @classmethod
    def calibrate(cls, near_left, near_right, net_right, net_left) -> "Court":
        """One-time fixed-camera calibration from 4 clearly-visible near-side points:
        the two near-baseline corners and where the net meets the two doubles sidelines.
        The far half is then computed from the court's fixed dimensions."""
        import cv2
        img = np.array([near_left, near_right, net_right, net_left], dtype=np.float32)
        court = np.array([[0, 0], [DOUBLES_W, 0], [DOUBLES_W, NET_Y], [0, NET_Y]],
                         dtype=np.float32)
        H = cv2.getPerspectiveTransform(img, court)
        inverse = np.linalg.inv(H)
        outer_court = np.array(
            [[0, 0], [DOUBLES_W, 0], [DOUBLES_W, COURT_L], [0, COURT_L]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        outer_img = cv2.perspectiveTransform(outer_court, inverse).reshape(-1, 2)
        return cls(H_img2court=H, H_court2img=inverse, corners_img=outer_img)

    @classmethod
    def from_image_corners(cls, near_left, near_right, far_right, far_left) -> "Court":
        """Calibrate from all four DOUBLES court corners in the image (near & far visible).

        Used by automatic court detection, which recovers the outer court quadrilateral
        directly. Maps the four image corners to the court-model doubles corners
        (0,0)-(W,0)-(W,L)-(0,L)."""
        import cv2
        img = np.array([near_left, near_right, far_right, far_left], dtype=np.float32)
        court = np.array([[0, 0], [DOUBLES_W, 0], [DOUBLES_W, COURT_L], [0, COURT_L]],
                         dtype=np.float32)
        H = cv2.getPerspectiveTransform(img, court)
        return cls(H_img2court=H, H_court2img=np.linalg.inv(H), corners_img=img)

    def to_court(self, pts_img: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_img, dtype=float).reshape(-1, 1, 2)
        import cv2
        return cv2.perspectiveTransform(pts, self.H_img2court).reshape(-1, 2)

    def to_image(self, pts_court: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_court, dtype=float).reshape(-1, 1, 2)
        import cv2
        return cv2.perspectiveTransform(pts, self.H_court2img).reshape(-1, 2)


def court_model_polylines() -> List[np.ndarray]:
    """Court line segments in metres, for drawing/validation overlays."""
    W, L, s = DOUBLES_W, COURT_L, SINGLES_IN
    lines = [
        [(0, 0), (W, 0)], [(0, L), (W, L)],              # baselines
        [(0, 0), (0, L)], [(W, 0), (W, L)],              # doubles sidelines
        [(s, 0), (s, L)], [(W - s, 0), (W - s, L)],      # singles sidelines
        [(0, NET_Y), (W, NET_Y)],                        # net
        [(s, SERVICE_Y), (W - s, SERVICE_Y)],            # near service line
        [(s, L - SERVICE_Y), (W - s, L - SERVICE_Y)],    # far service line
        [(W / 2, SERVICE_Y), (W / 2, L - SERVICE_Y)],    # centre service line
    ]
    return [np.array(l, dtype=np.float32) for l in lines]
