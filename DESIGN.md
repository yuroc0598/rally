# Pose-first tennis point extraction

## Evidence policy

The pipeline uses evidence attributable to the selected court. Mixed audio is excluded
because a neighboring-court impact cannot be attributed reliably. Ball inference is
disabled, so no static bright object can become a tennis ball and no unobserved bounce or
line call can influence a cut.

## Stage graph

```text
court calibration
  -> target-player detection and persistent identity
  -> shared all-player pose timeline
  -> full-timeline service and stroke events
  -> smoothed per-player stroke episodes
  -> offline constrained tennis-state decoder
       -> service-court retry/conflict grouping
       -> full-timeline serve-missed live-state recovery
       -> evidence-bounded, backdated endpoint
  -> boundary-invariant and tiling quality guard
  -> presentation intervals
```

There is no weighted fusion score and no alternative acceptance path.

## Serve sequence

A serve hypothesis is accepted only when the same player supplies an ordered sequence:

- the player is at or behind a baseline;
- a knee load or receiver-ready stance precedes contact;
- the racket-hand wrist proxy rises materially relative to torso height;
- wrist acceleration agrees with the overhead phase;
- target players are present on both court ends and the server's pre-motion position is
  stable.

The cut begins at the observed onset of the server's contiguous baseline setup rather
than a fixed offset from the overhead peak. Receiver motion never decides where the
serve detector runs.

The inspector persists the wrist-rise span, knee-load frames, leg-drive frames, baseline
frames, opposed-formation frames, score, box, and COCO-17 skeleton.

## Stroke actions

The YOLO/BoT-SORT pass requests COCO person and tennis-racket classes together. Racket
boxes are associated with the nearest tracked player and then with the nearest RTMPose
wrist; far-side misses remain explicitly labelled wrist proxies rather than fabricated
racket observations. The coarse timeline samples RTMPose at 6 FPS in boxes already owned
by that identity pass. Complete coarse action shapes, baseline-overhead motion, and a bounded set
of temporally supported wrist-motion peaks are refined at 12 FPS. Court-end rank is never
an identity fallback. An end-stable tracker fragment that cannot be mapped confidently to
a named player remains an explicit `track_*` actor; point order uses its measured court
side while the player gallery remains conservative.

Court geometry enrolls target-player identities; it is not a movement boundary. Once an
identity is established on the target court, tracker overlap keeps that player eligible
outside every sideline/baseline apron. This permits legal chases without admitting an
unassociated neighboring-court player merely because that person is visible.

COCO-17 contains wrists but no racket landmark. A groundstroke therefore requires:

- wrist preparation/backswing;
- directional reversal into the forward phase;
- local wrist speed normalized by torso length;
- adequate through-span and elbow extension;
- post-contact follow-through.

A compact elbow-extension action near the net is called a `compact_stroke_proxy`, never a
volley: without the ball the pipeline cannot know whether contact preceded a bounce. A
high-wrist action is similarly an `overhead_stroke_proxy`. Every class requires directional
reversal and coherent follow-through. Median smoothing, non-maximum suppression and a
second episode-collapse step prevent pose jitter from producing several physical swings.

## Point acceptance and endpoint

The decoder follows the observable structure of tennis without inventing ball facts:

- a service motion creates `SERVICE_ATTEMPT`;
- a credible opposite-side return strengthens a serve-led `LIVE_POINT`;
- sustained two-sided ready posture can establish continuation when the small/far racket
  pose is missed;
- accepted stroke episodes alternate near/far court sides; same-side gestures are rejected;
- a same-end, same-half service retry remains part of the same unresolved service group
  even when the tracker fragments the player's raw ID;
- an apparent end switch inside a physically impossible interval is conflicting evidence,
  not a new point;
- a high-confidence unreturned service is retained only with a positive between-point
  transition; its tennis outcome remains unknown;
- three or more credible alternating actions can establish a point without an observed
  serve anywhere in the video;
- a long, two-sided live-state bout can recover a point when both the serve and a small
  far-player racket contact were missed.

Rally acceptance and trimming are separate. Court speed is deliberately not a dead-play
test: walking and ball retrieval are valid between-point behavior, while a player may run
outside every line during a live point. The decoder ends a point at the measured cessation
of its live-state bout or backdates a later sustained reset to its first relaxed frame.
Rejected racket gestures never extend a point, and unexplained endpoint tails are bounded.
A batch-level guard rejects impossible boundaries and near-continuous output while
reporting weak post-serve stroke coverage explicitly.

This respects the rules that a ball remains live while airborne outside the court and that
line, bounce, service-box, double-bounce and let decisions require ball evidence. The
pose-only pipeline deliberately makes none of those claims.

## Ownership

- `signals/court*.py`: court geometry.
- `signals/visual.py` and `signals/player.py`: player detection and persistent identity.
- `signals/pose_actions.py`: pose records and local wrist-motion stroke proxies.
- `signals/pose_timeline.py`: shared pose inference and temporal serve/stroke measurements.
- `fusion/tennis_state.py`: service retries, live-point state, endpoints, and publication
  quality invariants.
- `pipeline.py`: the only active stage orchestrator.
- `web/`: progressive staged evidence and annotated inspection frames.
- `io/`: probing and rendering.

## Required models

- YOLO12n for same-pass person/racket detection and BoT-SORT player tracking.
- RTMPose-M Body7 COCO-17 for temporal body pose.
- ResNet50 14-landmark tennis-court model.

`setup.sh` downloads and contract-checks only these active models.
