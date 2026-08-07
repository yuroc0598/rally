# Rally

Rally trims fixed-camera tennis recordings to live points using the target players' court
positions and temporal body poses. Audio is copied into rendered clips but is never
decoded or scored. Ball tracking is disabled.

## Active pipeline

1. A ResNet50 court-landmark model locates the target court.
2. YOLO12 and BoT-SORT track people associated with that court. Conservative clothing
   re-identification supplies named-player galleries; unmapped but end-stable tracks remain
   explicit `track_*` fragments for tennis-state evidence instead of being misnamed.
3. One shared RTMPose timeline measures every tracked match player on both court ends.
   Tracker-owned boxes are reused, avoiding a second person-detection pass; likely arm
   actions receive a focused 12 FPS refinement.
4. A serve requires a baseline player's ordered load, wrist rise, overhead acceleration,
   stable setup, and opposed formation. One reliable overhead frame can be sufficient
   when the surrounding temporal sequence agrees.
5. Repeated services from the same baseline end and physical court half remain one
   unresolved service-attempt group. Pose cannot call them faults or lets.
6. Body-relative wrist paths are smoothed;
   preparation, reversal, acceleration and follow-through proposals are collapsed into
   physical stroke episodes. “Compact stroke” is used instead of claiming a volley because
   no ball is observed.
7. The state decoder accepts pose-confirmed serves followed by sustained two-sided ready
   posture. Strict near/far stroke alternation can recover a serve-missed point before the
   observed service timeline; walking arm gestures cannot start a terminal point.
8. A point ends only on measured player disengagement, a sustained relaxed/non-hitting
   transition, or a compact opponent stroke followed by later serve preparation. A later
   candidate never becomes the prior point's endpoint. Near-continuous tiled output is
   rejected by a final quality guard.
9. Court speed is not an endpoint veto: players may legally chase a live ball anywhere and
   may walk after a point. The pipeline does not claim ball bounces, line calls, faults,
   lets, aces, winners, or
   point outcomes.

## Setup and run

```bash
./setup.sh
python -m rally.web.app
# open http://127.0.0.1:8000
```

The setup verifies YOLO12n, RTMPose-M Body7 ONNX, the tennis-court landmark model, and a
browser-compatible H.264 encoder. It does not download or validate a ball model.

CLI use:

```bash
rally match.mp4 -o points.mp4 --json points.json
```

The web signal inspector updates after each completed stage. It groups player crops by
identity and exposes service groups, raw motion-proposal counts, accepted/rejected stroke
episodes, state transitions, and endpoint confidence.

## Limitation

RTMPose COCO-17 estimates body joints, not the racket or ball. The action stage therefore
recognizes hitting motion, not verified racket-ball contact. With ball tracking disabled,
the system deliberately abstains from tennis-rule outcomes.
