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
| **Geometry** | exactly two players, on court, opposed across the net | YOLO person detection + court-region filter (`players.py`, optional) |

Audio is first-class here: the racket impact train is cheap, camera-angle invariant,
and highly discriminative — the pipeline produces sensible output from **audio +
motion alone** when YOLO isn't installed.

## Install

```bash
pip install -r requirements.txt      # numpy, scipy, opencv-python-headless
# ffmpeg + ffprobe must be on PATH
# optional player-geometry channel:
pip install ultralytics
```

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
```

Library API:

```python
from rally import RallyConfig
from rally.pipeline import trim

result = trim("match.mp4", output_path="rallies.mp4",
              cfg=RallyConfig(min_rally_s=2.5), json_path="rallies.json")
print(result.segments, result.compression_ratio)
```

## The decoder (part (a) of the design)

`rally/segment.py` offers two decoders over the per-frame rally probability:

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

- `test_segment.py`, `test_score.py`, `test_audio.py`, `test_players.py` — pure-logic
  unit tests (no video needed).
- `test_pipeline_integration.py` — synthesises a video and runs the whole pipeline.
  Auto-skips in sandboxes that remap absolute paths / isolate `/tmp` for subprocesses.

## Scope, limits, and next phases

**Phase-1 (this repo)** is deliberately training-free. Known limitations:

- **Court region is heuristic** (percentiles of where feet cluster), not a trained
  court-keypoint model — so the geometry channel is coarse.
- **Warm-up looks like a rally** geometrically; without a scoreboard reader it may be
  included.
- **No ball tracker** — the strongest positive cue but also the least reliable
  (motion blur / occlusion), so it's intentionally deferred.
- The DP decoder is pure-Python; for multi-hour videos at high `analysis_fps` it can
  take tens of seconds (use `--hysteresis` for O(T) decoding).

**Phase-2/3** (see `DESIGN.md`): swap in a ResNet50 court-keypoint homography and
TrackNet ball tracking (precision), a learned TCN + segment-model classifier, and a
scoreboard-OCR + tennis-scoring-automaton prior to reject warm-up and index points.
