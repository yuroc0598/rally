# rally

Aim to trim a long, unedited tennis match recording (2–3 h, fixed single camera) down to a
short video containing the rallies while dropping warm-up, changeovers, ball
retrieval, chatting, and other dead time. Same end result as SwingVision's
rally-only export, built from open components.

This is **Phase-1** of the design in [`DESIGN.md`](DESIGN.md): a rule-based fusion of
cheap, training-data-free signals, decoded by a duration-aware segment model. It runs
today with no model downloads or labelled data.

```
video ──▶ audio ball-strike detection ┐
     └──▶ motion + camera-motion       ├─▶ per-frame rally probability
     └──▶ player geometry (optional)   ┘        │
                                                ▼
                       duration-aware segment-model decode  (part (a) of the design)
                                                │
                                                ▼
                              ffmpeg cut + concat  ──▶  rallies.mp4  (+ rallies.json)
```

## Why these signals

A rally has a distinctive multi-channel signature; no single cue is reliable alone,
so we fuse them:

| Channel | Signal during a rally | Source |
|---|---|---|
| **Audio** | quasi-periodic racket-ball "pock" transients, 0.3–3 s apart | band-pass + envelope peak-picking (`audio.py`) |
| **Motion** | dynamic player movement, camera static | frame-diff energy + phase-correlation (`motion.py`) |
| **Geometry** | exactly two players, on court, opposed across the net | YOLO person detection + court-region filter (`signals/player.py`, optional) |

Audio is first-class here: the racket impact train is cheap, camera-angle invariant,
and highly discriminative. Without TrackNet, the fallback is intentionally conservative:
supporting motion/geometry cues are never rescaled into certainty when audio is absent.

## Setup

One command installs every headless-compatible runtime dependency and prepares the
TrackNet, YOLO12, and RTMPose checkpoints used by the accuracy-first pipeline:

```bash
./setup.sh
```

It's idempotent (safe to re-run) and honours `PYTHON=<interpreter>` if you need a specific
Python. That's all a fresh clone needs — then launch `python -m rally.web.app` or use the CLI.

<details>
<summary>What setup.sh does / manual steps</summary>

```bash
python -m pip install -e ".[server]"   # core + TrackNet + YOLO/pose + web
./setup.sh                             # download, extract, checksum, load/inference checks
```

The model binaries are **not in the repo**. Setup downloads YOLO12 through Ultralytics,
downloads the official RTMPose SDK zip, extracts its `end2end.onnx` under the stable filename
the pipeline discovers, and verifies a COCO-17 inference. It also downloads TrackNet to
`models/tracknet.pt` and verifies that checkpoint loads into `BallTrackerNet`. Downloads and
extraction are atomic and the default artifacts are SHA-256 pinned. If OpenMMLab is
temporarily unreachable, leave the official zip in `models/` and re-run `./setup.sh`; the
existing archive is verified and extracted without another download.

The repo still runs when an external model host is unavailable: TrackNet falls back to the
audio-primary detector and optional RTMPose evidence abstains. Setup reports each missing
capability explicitly.

The weights come from `yastrebksv/TrackNet`, which has **no license** — fine for
personal/research use, but don't redistribute them (that's why they aren't committed or in a
release). For a durable/shippable setup, host a properly-licensed model at the same
architecture and set `WEIGHTS_DRIVE_ID` (or `--url`) accordingly.

Any checkpoint merely found in `models/` has **unknown provenance and unknown license**;
a filename or SHA-256 is identity evidence, not a license grant. Player/pose features use
Ultralytics, whose software/models are offered under **AGPL-3.0 or an Enterprise License**.
Review AGPL network/distribution obligations or obtain an enterprise license before a
deployment that cannot comply. Keep a private deployment inventory recording each model's
exact source, digest, authorship, training-data provenance, and applicable license.
</details>

## Usage

```bash
# trim to rallies + write segment metadata
python -m rally.cli match.mp4 -o rallies.mp4 --json rallies.json

# analysis only (no re-encode), just the segment list
python -m rally.cli match.mp4 --json rallies.json

# faster, keyframe-aligned cut instead of frame-accurate re-encode
# (burned labels and inter-point gaps are omitted because they require re-encoding)
python -m rally.cli match.mp4 -o rallies.mp4 --fast

# tuning
python -m rally.cli match.mp4 -o out.mp4 \
    --analysis-fps 5 --min-rally 2.5 --pad-pre 1 --pad-post 1.5 \
    --no-players --hysteresis

# match rules are auto-detected by default; disable serve-side/setup validation for
# unconstrained practice hitting
python -m rally.cli practice.mp4 -o rallies.mp4 --play-mode casual

# ball-primary (SwingVision-style) is the DEFAULT: the ball trajectory validates each
# candidate as a real rally and sets its serve start / point-end. Just install weights once:
python -m rally.tools.fetch_models --url <URL_TO_TRACKNET_PT>   # once, into models/
python -m rally.cli match.mp4 -o out.mp4        # ball tracking + auto court on by default
#   --no-ball-arbiter    force the faster audio-primary path (skip ball tracking)
#   --ball-weights PATH  use a specific checkpoint (else auto-discovered from models/)
#   --court-corners / --calibration  give the homography manually (takes precedence over auto)
#   --court-weights PATH  optional learned keypoint court detector, with classical fallback
#   --no-court-auto      disable auto court detection (if it locks onto the wrong lines)
#   --player-detection-model / --player-pose-model  explicit YOLO / RTMPose checkpoints
#   --serve-model PATH   guarded human-label-trained classifier (gate rechecked at load)
```

### Ball-arbiter mode (Phase-2, default)

Ball-primary detection is **on by default** and inverts the pipeline the way SwingVision
does it: the cheap audio/motion channels only **propose** candidate windows (high recall),
then the **ball trajectory decides**. If TrackNet weights / PyTorch aren't installed it
falls back automatically to the audio-primary path, so nothing to configure. Inside each
candidate the ball is tracked (TrackNet), the track is reconstructed
with a Kalman+RTS smoother (`signals/trajectory.py` — gap-fill, outlier-reject, per-sample
confidence), bounce candidates require a measured 2-D velocity-vector turn, and a verdict
(`fusion/ball_verify.py`) evaluates one continuous live component with gap-aware net
crossings/bounces. Candidate padding is used only for boundary recovery, not as negative
classification evidence. The verdict is tri-state: reliable structure accepts, reliable
contradiction rejects, and fragmented/low-coverage tracking is indeterminate. Only
cadence-coherent audio points survive an indeterminate or workload-omitted candidate; weak
one-off proposals do not. Per-candidate coverage, components, reason codes, and decisions are
stored in the sidecar. This rejects many audio/blob false positives, but ball motion alone
cannot prove match play: cooperative warm-up can still look identical. Use
`--require-serve-evidence` for a higher-precision audio+ball gate (with a recall tradeoff).

Audio is decoded in bounded chunks rather than buffered for the entire match. On **CPU** the
ball is tracked only inside candidate windows (not the whole video), which is
the only tractable option — full-video tracking is ~0.3 s/frame. Candidate work is ranked by
audio coherence/strike support with temporal diversity and capped by the padded union actually
tracked; coherent omitted intervals use the same conservative fallback. A court homography (auto
classical detection, **on by default**; or manual `--court-corners`, which takes precedence)
unlocks net-crossing and in/out geometry; without one the verdict falls back to in-play span
+ bounce count. Disable auto-detection with `--no-court-auto` if it misfires.

Library API:

```python
from rally import RallyConfig
from rally.pipeline import trim

result = trim("match.mp4", output_path="rallies.mp4",
              cfg=RallyConfig(min_rally_s=2.5), json_path="rallies.json")
print(result.segments, result.compression_ratio)
```

## Web UI

A browser front-end for the same pipeline lives in `rally/web` (a thin FastAPI
layer — the core `rally` package is used unchanged). Upload a match, watch it
process live, review the trimmed cut in the main player, open the original from
its secondary tab, correct the detected rallies by hand, and re-export.

```bash
python3 -m pip install -r requirements.txt     # core + ball + web dependencies
python3 -m rally.web                            # or: rally-web  -> http://127.0.0.1:8000
python3 -m rally.web --port 9000 --data-dir /tmp/rally_jobs   # options
```

What it does:

- **Upload & go** — drag-drop a video; the default trajectory path (ball tracking + auto
  court) needs no configuration when its external weights are available. An **Advanced** section exposes a subset of the CLI knobs
  (ball-arbiter/auto-court/YOLO on-off, static-camera preset, fast cut, hysteresis,
  min-rally, …). Processing runs in a background worker with **live progress**.
- **Review** — the processed cut is the primary player and the original is lazy-loaded in
  a secondary tab. Double-click the left/right sides to move between points. A top-right
  overlay shows an explicitly uncertain ground-plane ball-speed estimate and heuristic
  error scale when reliable court/trajectory evidence exists (otherwise it says
  unavailable). A single camera cannot recover ball height or full 3-D velocity. The
  **timeline** shows every detected rally band and ball-strike with a click-to-seek playhead.
- **Correct & re-export** — an editable segment table lets you nudge start/end
  times, add, or drop rallies, then **re-cut** the output from your edits.
- **Download** the trimmed `rallies.mp4` and the `rallies.json` sidecar; manage a
  gallery of past jobs.

### Labeling

The detail view has a **Labeling** panel that turns a match into raw annotation
samples and a quick keyboard-driven labeler. Click **Generate samples** and the
server produces two kinds of raw data (in a background worker):

- **Player ID** (pictures) — YOLO detects the people on court, a roster is built
  from their court positions (P1/P2 for singles, P1–P4 for doubles; rename them
  inline), and each detected player is **cropped to a single-player picture**.
  For each crop you pick which roster player it is (the position-based guess is
  pre-selected) plus a quality tag.
- **Serve motion** (short clips) — a few-second clip is cut around each serve
  moment (rally start). For each clip you mark *is it a serve*, *who served*,
  *deuce/ad side*, *near/far end*, serve type, and outcome.

Labels autosave to the server (`←/→` to move, `Enter` to save & advance) and
**Export labels** downloads a self-describing JSON bundle (roster + tasks +
labels). The server setup retains Ultralytics while pinning `cv2` to the headless OpenCV
wheel, so player geometry, serve-setup validation, and crops work without GUI libraries.
Player detection defaults to `models/yolo12n.pt`; set `RALLY_YOLO_DETECTION_MODEL` to
another Ultralytics model name or local weight file. Pose defaults to RTMLib: YOLO12 first
selects target-court player boxes, then top-down RTMPose estimates COCO-17 joints inside
each crop. This includes small far-side servers instead of limiting pose to the largest
near-side player. RTMLib uses its balanced body model by default; place the extracted
`rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` in `models/` for
offline startup, or set `RALLY_PLAYER_POSE_MODEL` to another ONNX path/URL.
`RALLY_PLAYER_POSE_BACKEND=yolo` retains the legacy Ultralytics-pose path, and the old
`RALLY_YOLO_POSE_MODEL` variable automatically selects that compatibility backend.
`RALLY_RTMPOSE_RUNTIME` selects `onnxruntime` (default) or `opencv`; RTMPose uses a CUDA
execution provider when one is installed, otherwise its cropped inference runs on CPU.
`RALLY_WEB_YOLO` remains a label-crop-only override and otherwise inherits the YOLO12
detection model.

Each job is a self-contained directory under the data dir (default `.rally_web/`,
override with `--data-dir` or `RALLY_WEB_DATA`). The server automatically runs up to four
trims concurrently, sized from CPU capacity and currently free CUDA memory; set
`RALLY_WEB_WORKERS` to override that choice. Video export needs `ffmpeg` on `PATH`; if a
label/`drawtext` render isn't available it falls back to a plain cut, and the
JSON analysis is always produced regardless.

TrackNet inference selects a batch from currently free VRAM and admits up to two concurrent
callers per server process so CPU decode can overlap. This is a concurrency limit, not a
claim that the implementation creates two explicit CUDA streams. Operators can override it with
`RALLY_BALL_BATCH_SIZE`, `RALLY_GPU_TRACK_SLOTS`, and
`RALLY_HEATMAP_DECODE_WORKERS`; oversized values can exhaust VRAM or reduce throughput.
`RALLY_TRACKNET_PIPELINE=0` disables bounded decode/inference/heatmap overlap for regression
comparison, and `RALLY_TRACKNET_WINDOW_PLAN=union` enables the opt-in connected-window plan.

The web regression checks can be run with `pytest tests/test_web.py`; they are not an
accuracy benchmark.

## The decoder (part (a) of the design)

`rally/fusion/decode.py` offers two decoders over the per-frame rally probability:

* **`dp_decode`** (default) — a duration-aware segmental Viterbi. It jointly chooses
  segment boundaries and RALLY/GAP labels to maximise
  `Σ frame-log-emission + duration_prior(len|label) − transition_penalty`, using
  prefix sums for O(T·Lmax) cost. Explicit per-label **duration priors** are the
  reason to prefer this over frame-level smoothing — rallies and dead-time each have
  characteristic lengths (see `RallyConfig.rally_dur_prior_s` / `gap_dur_prior_s`).
* **`hysteresis_decode`** (`--hysteresis`) — a cheaper two-threshold state machine
  plus min-duration / merge-gap / padding post-processing.

Tuning bias: prefer **recall** (a couple of extra seconds of dead time is far more
forgivable than a missing point). The defaults reflect that.

Audio-only attachment of an isolated pre-cluster sound as a serve is disabled by default:
without court/trajectory confirmation it can mistake a bounce, speech consonant, or prior-point
sound for a serve and retain walking/reset footage. The rendered cut instead keeps a bounded
one-second real-footage pre-roll around every detected point.

## Configuration

All thresholds live in `rally/config.py` (`RallyConfig`). Notable knobs: analysis and
player frame rates, audio strike band / sensitivity / SNR gate, channel weights,
hysteresis thresholds, duration priors, padding, and re-encode vs stream-copy.

## Offline serve-learning dataset

The web label download now includes `serve_motion` answers plus the pose, stationary
player/court, and TrackNet diagnostics that produced the corresponding rule decision.
Build a dataset from **multiple independent matches** (a match id is a validation group):

```bash
python -m rally.tools.serve_dataset from-web \
  --job match-a /video/a.mp4 /labels/a-labels.json \
  --job match-b /video/b.mp4 /labels/b-labels.json \
  --job match-c /video/c.mp4 /labels/c-labels.json \
  --out serve-training.json

pip install -e '.[training]'
python -m rally.tools.serve_train serve-training.json --model-out serve-model.joblib
```

`LABELS_JSON` may be the downloaded labels export or a local revision's `labels.json`
(with sibling `tasks.json`). Audio is decoded once per video; compatible exported
diagnostics add pose coverage/overhead, baseline stability/court filtering, and ball
coverage/flight features with explicit availability fields.

Validation is leave-one-match-out—samples from one match are never randomly divided
between train and validation. Duplicate video/export content is rejected across match IDs,
and live deployment additionally requires stable observation/group IDs rather than legacy
nearest-time joins. Every artifact records the model and current-rule metrics,
dataset fingerprint, feature schema, fold groups, and a conservative gate requiring at
least three matches/30 labels and a real held-out improvement over the rules. The live
pipeline never discovers an artifact implicitly. To activate a passing artifact, use
`--serve-model serve-model.joblib` or `RALLY_SERVE_MODEL`; runtime revalidates the exact
schema, coverage, accuracy, rule-improvement, and precision gate before loading it.
Joblib is executable pickle data, so only load an artifact produced by this workflow from
a trusted local source; the statistical gate is not a malware sandbox.

## Regression checks and independent evaluation

```bash
pytest -q
```

These author-written synthetic checks only guard implementation invariants; they do not
validate real-world detection accuracy. Follow [`EVALUATION.md`](EVALUATION.md) and use
`rally-evaluate predicted.json independently_labeled_gold.json` for a sealed real-video
holdout. No accuracy claim should be inferred from the repository test result.

- `test_decode.py`, `test_score.py`, `test_audio.py`, `test_players.py` — pure-logic
  unit tests (no video needed).
- `test_pipeline_integration.py` — synthesises a video and runs the whole pipeline.
  Auto-skips in sandboxes that remap absolute paths / isolate `/tmp` for subprocesses.

## Scope, limits, and next phases

**Phase-1 (this repo)** is deliberately training-free. Known limitations:

- **Court region is heuristic** in the default channels (percentiles of where feet
  cluster); the ball-arbiter path replaces this with a real homography (auto-detected by
  default, or manual `--court-corners`).
- **Warm-up can look like a rally** to both the audio and trajectory paths. Optional serve
  evidence improves precision, but reliable separation still needs independently validated
  serve/match-state or scoreboard information.
- **Ball tracking** is the default *arbiter* (Phase-2; `--no-ball-arbiter` to disable) but
  remains the least reliable single cue (motion blur / occlusion) — the Kalman+RTS
  reconstruction in `signals/trajectory.py` mitigates this, and on CPU it runs only
  inside candidate windows. Needs TrackNet weights (`rally.tools.fetch_models`); without
  them it falls back to the audio-primary path automatically.
- The DP decoder is pure-Python; for multi-hour videos at high `analysis_fps` it can
  take tens of seconds (use `--hysteresis` for O(T) decoding).

**Still ahead (Phase-3, see `DESIGN.md`):** a ResNet50 court-keypoint model at the
`court_detect` hook for perspective-heavy footage, a learned TCN + segment-model
classifier, and a scoreboard-OCR + tennis-scoring-automaton prior to reject warm-up and
index points.
