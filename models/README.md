# Local models

This directory holds model checkpoints used by the rally-analysis pipeline. Model binaries
are intentionally excluded from Git because they are large and have licenses/provenance
that must be reviewed independently.

Run the repository setup from the project root to prepare the standard models:

```bash
./setup.sh
```

The default setup verifies and places these files here:

- `tracknet.pt` — TrackNet-compatible tennis-ball tracker.
- `yolo12n.pt` — YOLO12 player detector.
- `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip` — official RTMPose SDK archive.
- `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` — `end2end.onnx` extracted from that archive.

`setup.sh` checksum-verifies the default artifacts and runs model load/inference checks.
See [`MODEL_PROVENANCE.md`](../MODEL_PROVENANCE.md) before redistributing or deploying any
checkpoint. Custom paths can be supplied through the model-related environment variables
documented at the top of `setup.sh`.
