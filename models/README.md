# Local models

Run `./setup.sh` from the repository root. The active pose-first pipeline verifies:

- `yolo12n.pt` — same-pass target-player/tennis-racket detection and player tracking;
- `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` — COCO-17 pose;
- `court_keypoints_resnet50.pth` — tennis-court landmarks.

Ball tracking is disabled and setup does not download or verify ball weights. Model files
are excluded from Git; verify upstream licenses and training-data provenance before
redistribution.
