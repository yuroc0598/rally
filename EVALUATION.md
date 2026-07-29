# Independent evaluation protocol

Repository tests are regression checks, not evidence that rally detection is accurate.
Accuracy claims require a sealed set of real matches labeled independently of this code.

## Corpus and split

- Use unedited real matches spanning cameras, venues, court colors, lighting, resolutions,
  player levels, singles/doubles, crowd noise, warm-up, changeovers, and adjacent courts.
- Assign whole matches—not frames—from the same venue/camera to only one of development or
  holdout. Keep the holdout annotations hidden while tuning.
- Have at least two annotators label every live point from serve preparation through the
  terminal event. Record warm-up and ambiguous intervals separately, measure agreement, and
  adjudicate disagreements without viewing detector output.
- Do not generate gold intervals from this repository's audio, ball, or decoder outputs.

Gold JSON may be a plain segment list or the same `{"segments": [...]}` shape as a rally
sidecar:

```json
{
  "segments": [
    {"start": 12.34, "end": 20.81},
    {"start": 44.02, "end": 48.77}
  ]
}
```

## Metrics

Run the detector once with frozen configuration, then compare its sidecar with gold:

```bash
rally-evaluate predicted.json gold.json --min-iou 0.3 -o metrics.json
```

Report per match and in aggregate:

- point precision, recall, and F1;
- false positives and missed gold points by category;
- start/end absolute error for matched points;
- retained dead-time seconds;
- runtime and peak resident memory;
- results by venue/camera and difficult condition, with confidence intervals.

The holdout result is the accuracy gate. Unit tests, synthetic clips, and development-match
scores may diagnose regressions but must not be presented as independent validation.
