# Local evaluation samples

This directory is reserved for local golden-evaluation videos, human ground truth, and
generated evaluation artifacts. Its data is intentionally excluded from Git because the
videos are large and may contain private recordings; only this documentation is tracked.

Place labeled datasets directly under `samples/golden/` using matching numeric suffixes:

```text
samples/golden/
  input_1.mp4       # source video; .mov is also supported
  res_1.txt         # human point-boundary annotations for input_1
  input_2.mp4
  res_2.txt
  ...
  unlabeled/        # ignored by golden discovery/evaluation
```

Each `res_N.txt` contains the points for `input_N`, with point start/end timestamps and an
optional second-serve start for a fault or let. The current checked-in golden test defines
the expected dataset identities and boundaries in
[`tests/test_golden_rallies.py`](../tests/test_golden_rallies.py).

Run the full local evaluation with:

```bash
RALLY_RUN_GOLDEN=1 pytest -q tests/test_golden_rallies.py
```

Set `RALLY_GOLDEN_ARTIFACTS=sessions/golden` to retain processed videos and analysis
sidecars for the web UI's Golden tab. Those generated artifacts are also excluded from Git.
