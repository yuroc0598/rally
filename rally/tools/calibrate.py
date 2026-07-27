"""One-time fixed-camera court calibration: save/verify/reuse the 4 corner points.

A click-to-calibrate GUI needs a display, so this is the headless equivalent: you pass
your 4 corner estimates, it renders the court model overlaid on a real frame so you can
*see* whether they line up (adjust the numbers and re-run until the green court matches
the white lines), and it saves the calibration to JSON for reuse via `rally.cli --calibration`.

    # verify + save
    python -m rally.tools.calibrate match.mp4 --corners "234,833;1816,833;1381,613;585,613" \
        --at 12 --overlay check.png --save court.json
    # then reuse
    python -m rally.cli match.mp4 -o out.mp4 --static-camera --calibration court.json
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

Corners = List[Tuple[float, float]]


def save_calibration(path: str, corners: Corners) -> None:
    with open(path, "w") as fh:
        json.dump({"court_corners": [[float(x), float(y)] for x, y in corners]}, fh, indent=2)


def load_calibration(path: str) -> Corners:
    with open(path) as fh:
        return [tuple(p) for p in json.load(fh)["court_corners"]]


def overlay(video: str, corners: Corners, out_png: str, at_s: float = 10.0) -> None:
    """Draw the court model (from the homography of `corners`) onto a frame for visual check."""
    import cv2

    from ..signals.court import COURT_L, Court, court_model_polylines

    court = Court.calibrate(*corners)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_s * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame for overlay")
    h, w = fr.shape[:2]
    for seg in court_model_polylines():
        ip = court.to_image(seg).astype(int)
        cv2.line(fr, tuple(ip[0]), tuple(ip[1]), (0, 255, 0), 3)
    # mark the computed (possibly invisible) far-baseline corners
    for cp in ([0, COURT_L], [10.97, COURT_L]):
        p = court.to_image([cp])[0].astype(int)
        cv2.circle(fr, tuple(p), 8, (0, 0, 255), -1)
    cv2.imwrite(out_png, fr)


def _parse_corners(s: str) -> Corners:
    pts = [tuple(float(v) for v in pair.split(",")) for pair in s.split(";")]
    if len(pts) != 4:
        raise SystemExit("--corners needs exactly 4 'x,y' points separated by ';'")
    return pts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rally.tools.calibrate", description=__doc__)
    p.add_argument("video")
    p.add_argument("--corners", required=True,
                   help="'nlx,nly;nrx,nry;netRx,netRy;netLx,netLy'")
    p.add_argument("--at", type=float, default=10.0, help="frame time (s) for the overlay")
    p.add_argument("--overlay", default=None, help="write a verification overlay PNG here")
    p.add_argument("--save", default=None, help="save the calibration JSON here")
    args = p.parse_args(argv)
    corners = _parse_corners(args.corners)
    if args.overlay:
        overlay(args.video, corners, args.overlay, args.at)
        print(f"wrote overlay -> {args.overlay} (check the green court matches the white lines)")
    if args.save:
        save_calibration(args.save, corners)
        print(f"saved calibration -> {args.save}")
    if not args.overlay and not args.save:
        print("nothing to do: pass --overlay and/or --save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
