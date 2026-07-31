# Design: automatic tennis rally trimmer

Goal: given a 2–3 h unedited recording of a match (fixed/mostly-fixed single camera,
capturing everything — warm-up, changeovers, ball-picking, chatting), output a short
video containing **only rally (live-play) segments**, like SwingVision's rally export.

This is fundamentally **temporal segmentation**: classify time *intervals* as
`RALLY` vs `NOT_RALLY`, then cut on the boundaries. Clean, stable boundaries matter
more than raw per-frame accuracy.

## Architecture

```
Stage 0  decode + downsample (analysis fps, proxy resolution)
Stage 1  parallel channels:
           court/geometry · player detect+filter · ball track · motion/flow · audio strikes
Stage 2  per-frame feature vector (fuse channels)
Stage 3  rally scorer  ->  per-frame P(rally)
Stage 4  temporal segmentation (segment-model decode: emissions + duration + transition)
Stage 5  cut & concat from the ORIGINAL full-res source (+ JSON sidecar)
```

Key decisions:

- **Analyse at low res/fps, cut at full res.** A 3 h video at 5 fps ≈ 54k analysis
  frames — tractable on one machine. The final cut references the original file so
  quality is untouched.
- **Fuse at the frame level, decide at the interval level.**
- **Redundancy over any single cue.** Ball tracking and audio are the strongest positive
  signals; motion and optional player geometry remain supporting evidence.

Current package boundaries:

- `signals/` extracts measurements only; shared observation records live in `domain/`.
- `fusion/` owns interval scoring, proposal ownership, ball verdicts, match rules, and
  optional learned decisions. Exact interval policies are named in `fusion/intervals.py`.
- `learning/` owns versioned feature/artifact schemas used identically by offline tools
  and guarded runtime loading; `tools/` only adapts files, trains, and evaluates.
- `pipeline.py` orchestrates stages using the typed signal/arbiter state in
  `pipeline_types.py`; media publication is isolated in `io/publish.py`.
- `web/` owns job/revision state and presentation. Training labels are exported data;
  they do not bypass the held-out model gate or directly mutate detection rules.

## (a) The rally classifier: a segment model

**Reality check from the literature (read in full, see part (b)):** the two cheapest
published segmenters exploit *broadcast production conventions this footage lacks* —
SmartTennisTV keys on "camera is overhead only during play" (a director's cut);
Delakis keys on shot cuts / dissolves / replays. A raw single-camera recording has
none of these. What transfers is their **framework, audio cues, duration priors, and
scoring automaton** — the visual observations must come from the geometry/ball stack.

**Labels:** `{SERVE, RALLY, BETWEEN_POINTS, CHANGEOVER, IDLE}` (collapse to
`{SERVE,RALLY}` for the trim). Richer states model the rhythm better even for binary output.

**Objective** — segmental Viterbi over boundaries + labels:

```
(L,A)* = argmax  Σ_n [ log p(O_seg_n | a_n)     segment observation model
                     + log p(dur_n | a_n)        explicit per-label duration prior
                     + log p(a_n | a_{n-1}) ]     tennis-rhythm transition prior
```

1. **Observation `p(O_seg | a)`** — must model *within-segment temporal evolution*
   (Delakis: assuming per-frame independence collapsed precision to 6%). Use a small
   **1D-TCN / BiLSTM** over the segment's frame features (discriminative), or an inner
   HMM/GMM (generative).
2. **Duration prior `p(dur | a)`** — the main reason to beat a frame-HMM (whose
   duration is memoryless/geometric). Model as a per-label duration histogram
   (Delakis used 30 bins in seconds). This is what min-duration/merge/hysteresis only
   crudely approximate.
3. **Transition prior** — encodes rhythm (RALLY never directly follows RALLY;
   CHANGEOVER every ~2 games); SmartTennisTV's point/game/set scoring automaton lifts
   in directly as a validator.

**Feature vector per analysis frame:** geometry (players on court, opposed across
net, speeds) · ball (present, speed, net-crossings — soft/optional) · motion (flow
energy, camera-motion flag) · audio (ball-hit onsets, applause) at native rate.

**Audio fusion:** Delakis's best model (`VhmmA2gram`) fuses audio *asynchronously* as
a **bigram of events within the segment** — it captures the ordered signature
`serve-hit → rally-hits → silence/applause`. Fuse audio as a segment-level event
sequence, not concatenated per frame.

**Recommended build:** `frame features → 1D-TCN (per-frame logits) → segment-model
Viterbi (duration + transition + audio-bigram) → {SERVE,RALLY} intervals`. Learned
observations (robust) + structured decoding (interpretable, duration/rhythm-aware).

**Phase-1 implementation in this repo** replaces the learned observation model with a
rule-based `P(rally)` (`rally/fusion/score.py`) and implements the duration-aware segmental
Viterbi in `rally/fusion/decode.py::dp_decode` — the same objective above with the two
labels RALLY/GAP.

## (b) Evidence extracted from source papers

### TrackNet (arXiv 1907.03698) — ball tracker
- 640×360, **3 consecutive frames** → Gaussian heatmap (VGG16 + DeconvNet); threshold
  128 + Hough circle. Ball dia 2–12 px; positioning-error spec 5 px.
- Data: 2017 Universiade final, 20,844 frames + 16,118 from 9 courts (grass/clay/hard).

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Archana's (classical) | 92.5 | 74.5 | 82.5 |
| TrackNet 1-frame | 95.7 | 89.6 | 92.5 |
| TrackNet 3-frame | 99.8 | 96.6 | 98.2 |
| 3-frame, enriched | 99.7 | 97.3 | 98.5 |
| **3-frame, 10-fold CV** | **95.3** | **75.7** | **84.3** |

Honest cross-validated recall is 75.7% (only 2/7 occluded balls recovered) →
**ball tracking is the weak link; never a hard requirement.**

### SmartTennisTV (arXiv 1801.01430) — rally segmentation + scoring
- Rally segmenter: HOG → χ²-kernel SVM → Kalman smoothing → **F1 97.46%** (non-rally
  precision 98.94%, rally 95.41%) — *but relies on the overhead-only broadcast cue.*
- Data: 10 singles matches @720p; 1,011 annotated rallies.
- Score refinement via a point/game/set **scoring automaton** (vocab {0,15,30,40,AD})
  lifts accuracy e.g. 79.3→91.7%.

### Delakis et al. (ICME 2005) — segment models for tennis parsing
- Corpus 15 h; 4 scenes / 12 states; duration = 30-bin histogram; Viterbi window 70.

| Model | Correct | Precision | Recall |
|---|---|---|---|
| HMM video-only | 70.7 | 68.9 | 80.5 |
| HMM audio+video | 74.6 | 73.7 | 82.5 |
| **AVprod (frame-independent)** | 60.2 | **6.05** | 33.6 |
| Segment Vhmm | 76.4 | 71.0 | 80.8 |
| Segment AVhmm | 77.8 | 72.4 | 83.7 |
| **Segment VhmmA2gram (best)** | **79.2** | **75.1** | 80.1 |

Two load-bearing results: (1) per-frame independence is unusable (6% precision) →
model within-segment structure; (2) asynchronous audio-event bigram wins → fuse audio
as an ordered event sequence.

### Hawk-Eye pipeline (arXiv 2511.04126) — the standard explicit stack
- Players: configurable Ultralytics YOLO (YOLO12 detection by default) + court-polygon
  foot-point filter → top-2 by centre proximity. RTMLib applies top-down RTMPose only to
  those target-court crops, including small far-side baseline players.
- Ball: custom YOLOv5 (weak on serves/occlusion) + interpolation + Kalman.
- Court: ResNet50 **14-keypoint** regressor (~3.8 px) → homography.
- Shot detection: ball-velocity **angle change + magnitude jump**, scale-aware.

## Phased plan

- **Phase 1 (this repo):** audio strikes + player-geometry + motion → rule score →
  duration-aware segment decode → ffmpeg cut. No training data.
- **Phase 2 (implemented, ball-arbiter):** TrackNet ball tracking as the *arbiter*, not
  just an end-trimmer. Coarse-to-fine on CPU: the Phase-1 channels propose candidate
  windows, then per candidate the ball is tracked, the trajectory reconstructed
  (`signals/trajectory.py`: constant-velocity Kalman + RTS smoother → gap-fill,
  Mahalanobis outlier gating, per-sample confidence), bounce candidates requiring a
  measured 2-D velocity-vector turn, and a gap-aware verdict (`fusion/ball_verify.py`)
  accepts/rejects/abstains and bounds each window (serve start, point-end). Fragmented or
  low-coverage ball tracks abstain instead of deleting coherent audio points; explicit reliable
  contradictions still reject. Court homography via
  automatic detection (`signals/court_detect.py`: white-line Hough → outer-corner
  intersection → homography, scored by court-model reprojection overlap) or manual
  `court_corners`. On by default (auto court detection too; `--no-ball-arbiter` /
  `--no-court-auto` to disable), falling back to audio-primary if weights are absent;
  weights via
  `rally.tools.fetch_models`. An optional Ultralytics court-keypoint checkpoint can run
  first for perspective-heavy footage; predictions require multi-frame geometric and
  painted-line consensus, with the classical detector retained as fallback.
- **Phase 3 (guarded learning path implemented):** web `serve_motion` exports can be adapted
  into a versioned multi-match dataset with audio plus compatible pose/court/ball
  diagnostics. `rally.tools.serve_train` uses leave-one-match-out validation and emits a
  guarded artifact only *eligible* for live integration when it beats the current rules
  under minimum match/sample/coverage thresholds. An explicitly configured artifact is
  loaded only after runtime recomputes that gate. The remaining Phase-3 work is a learned
  TCN + segment-model classifier, and scoreboard OCR + scoring automaton to
  reject warm-up and index points.
