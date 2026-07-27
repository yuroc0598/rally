# rally

Trim a long, unedited tennis match recording (2–3 h, fixed single camera) down to a
short video containing **only the rallies** — dropping warm-up, changeovers, ball
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
and highly discriminative — the pipeline produces sensible output from **audio +
motion alone** when YOLO isn't installed.

## Setup

One command installs every dependency **and** fetches the ball-tracking weights, so
ball-arbiter mode works out of the box:

```bash
./setup.sh
```

It's idempotent (safe to re-run) and honours `PYTHON=<interpreter>` if you need a specific
Python. That's all a fresh clone needs — then launch `python -m rally.web.app` or use the CLI.

<details>
<summary>What setup.sh does / manual steps</summary>

```bash
pip install -r requirements.txt        # core: numpy, scipy, opencv, bundled ffmpeg, web
pip install torch gdown ultralytics    # ball-arbiter (torch), weight fetch (gdown), players
python -m rally.tools.fetch_models --drive-id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl
```

The TrackNet weights are **not in the repo** (large, externally hosted, and unlicensed —
`*.pt` is gitignored). They download to `models/tracknet.pt`, which the pipeline
auto-discovers; the fetch verifies the checkpoint loads into `BallTrackerNet` first. The repo
still runs without them — ball-arbiter auto-falls-back to the audio-primary detector.

The weights come from `yastrebksv/TrackNet`, which has **no license** — fine for
personal/research use, but don't redistribute them (that's why they aren't committed or in a
release). For a durable/shippable setup, host a properly-licensed model at the same
architecture and set `WEIGHTS_DRIVE_ID` (or `--url`) accordingly.
</details>

## Usage

```bash
# trim to rallies + write segment metadata
python -m rally.cli match.mp4 -o rallies.mp4 --json rallies.json

# analysis only (no re-encode), just the segment list
python -m rally.cli match.mp4 --json rallies.json

# faster, keyframe-aligned cut instead of frame-accurate re-encode
python -m rally.cli match.mp4 -o rallies.mp4 --fast

# tuning
python -m rally.cli match.mp4 -o out.mp4 \
    --analysis-fps 5 --min-rally 2.5 --pad-pre 1 --pad-post 1.5 \
    --no-players --hysteresis

# ball-primary (SwingVision-style) is the DEFAULT: the ball trajectory validates each
# candidate as a real rally and sets its serve start / point-end. Just install weights once:
python -m rally.tools.fetch_models --url <URL_TO_TRACKNET_PT>   # once, into models/
python -m rally.cli match.mp4 -o out.mp4        # ball tracking + auto court on by default
#   --no-ball-arbiter    force the faster audio-primary path (skip ball tracking)
#   --ball-weights PATH  use a specific checkpoint (else auto-discovered from models/)
#   --court-corners / --calibration  give the homography manually (takes precedence over auto)
#   --no-court-auto      disable auto court detection (if it locks onto the wrong lines)
```

### Ball-arbiter mode (Phase-2, default)

Ball-primary detection is **on by default** and inverts the pipeline the way SwingVision
does it: the cheap audio/motion channels only **propose** candidate windows (high recall),
then the **ball trajectory decides**. If TrackNet weights / PyTorch aren't installed it
falls back automatically to the audio-primary path, so nothing to configure. Inside each
candidate the ball is tracked (TrackNet), the track is reconstructed
with a Kalman+RTS smoother (`signals/trajectory.py` — gap-fill, outlier-reject, per-sample
confidence), bounces are found from the vertical-velocity reversal in court coordinates, and
a verdict (`fusion/ball_verify.py`) keeps only windows with a genuine live ball + rally
structure (net crossing / bounces), snapping each to its serve start and point-ending
bounce. This rejects warm-up and audio false positives that the audio-primary path keeps.

On **CPU** the ball is tracked only inside candidate windows (not the whole video), which is
the only tractable option — full-video tracking is ~0.3 s/frame. A court homography (auto
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
process live, review the trimmed cut next to the original, correct the detected
rallies by hand, and re-export.

```bash
python3 -m pip install -r requirements.txt     # fastapi, uvicorn, python-multipart
python3 -m rally.web                            # or: rally-web  -> http://127.0.0.1:8000
python3 -m rally.web --port 9000 --data-dir /tmp/rally_jobs   # options
```

What it does:

- **Upload & go** — drag-drop a video; the accurate defaults (ball tracking + auto
  court) need no configuration. An **Advanced** section exposes a subset of the CLI knobs
  (ball-arbiter/auto-court/YOLO on-off, static-camera preset, fast cut, hysteresis,
  min-rally, …). Processing runs in a background worker with **live progress**.
- **Review** — the original and the rallies-only cut play side by side, with a
  **timeline** showing every detected rally band and ball-strike, and a
  click-to-seek playhead.
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
labels). YOLO is only needed for the player crops; the first run downloads
`yolov8n.pt` (or point `RALLY_WEB_YOLO` at an existing weight).

Each job is a self-contained directory under the data dir (default `.rally_web/`,
override with `--data-dir` or `RALLY_WEB_DATA`); set `RALLY_WEB_WORKERS` to run
more than one trim concurrently. Video export needs `ffmpeg` on `PATH`; if a
label/`drawtext` render isn't available it falls back to a plain cut, and the
JSON analysis is always produced regardless.

Run the web tests (unit + one end-to-end upload→process→edit cycle) with
`pytest tests/test_web.py`.

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

## Configuration

All thresholds live in `rally/config.py` (`RallyConfig`). Notable knobs: analysis and
player frame rates, audio strike band / sensitivity / SNR gate, channel weights,
hysteresis thresholds, duration priors, padding, and re-encode vs stream-copy.

## Tests

```bash
pytest -q
```

- `test_decode.py`, `test_score.py`, `test_audio.py`, `test_players.py` — pure-logic
  unit tests (no video needed).
- `test_pipeline_integration.py` — synthesises a video and runs the whole pipeline.
  Auto-skips in sandboxes that remap absolute paths / isolate `/tmp` for subprocesses.

## Scope, limits, and next phases

**Phase-1 (this repo)** is deliberately training-free. Known limitations:

- **Court region is heuristic** in the default channels (percentiles of where feet
  cluster); the ball-arbiter path replaces this with a real homography (auto-detected by
  default, or manual `--court-corners`).
- **Warm-up looks like a rally** to the audio-primary path; the ball-arbiter rejects
  most of it (no serve / no point-end structure), but without a scoreboard reader some
  cooperative warm-up hitting can still slip through.
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
