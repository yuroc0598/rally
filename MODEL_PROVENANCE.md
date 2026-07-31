# Model provenance and licensing

This project can load model files from `models/` or from paths supplied by an operator.
File discovery and a SHA-256 digest identify bytes; they do **not** prove authorship,
training-data rights, model provenance, or permission to use or redistribute a model.
This is an engineering inventory, not legal advice. Verify current upstream terms and get
appropriate legal review before distribution, hosted service, or commercial use.

## TrackNet ball checkpoint

- Architecture compatibility: `rally/vendor/tracknet_torch.py` follows the
  `yastrebksv/TrackNet` / TennisProject-style three-frame TrackNet port.
- Automatically discovered or locally bundled checkpoint, including
  `models/tracknet.pt`: **provenance unknown; author unknown; training-data provenance unknown;
  license unknown**. The runtime sidecar records this as
  `weights_provenance: unknown_local_checkpoint` and `weights_license: unknown`.
- The pinned Google Drive ID and SHA-256 in `rally.tools.fetch_models` identify one file
  associated with the referenced unofficial repository. The upstream repository has no
  license grant. A matching digest does not create redistribution or commercial-use rights.
- Do not redistribute a TrackNet checkpoint or ship it in a product until its copyright,
  training-data, and model-license rights are established. Supply a checkpoint with known,
  compatible rights for production deployments.

## Ultralytics YOLO detection

Player detection uses the `ultralytics` package and may download/load the configured YOLO12
checkpoint. Ultralytics
offers its software and models under **AGPL-3.0 and an Enterprise License**. AGPL-3.0 can impose source-availability
obligations, including for modified software used over a network. If a deployment cannot
comply with AGPL-3.0, obtain an appropriate Ultralytics Enterprise License or replace this
dependency with a model/runtime whose terms fit the deployment.

## RTMLib / RTMPose

The default pose path uses RTMLib with an OpenMMLab RTMPose body checkpoint after YOLO12
has selected target-court player crops. RTMLib is MIT-licensed; model/checkpoint and
training-data terms must be reviewed independently. The default URL refers to the balanced
RTMPose-M Body7 ONNX release. For reproducible/offline operation, retain the exact ONNX
file locally, record its SHA-256, upstream source, model license, and training-data terms.
The runtime abstains from optional pose evidence when the default checkpoint is unavailable
and fails explicitly when an operator-configured checkpoint cannot be loaded.

References:

- Ultralytics licensing: <https://www.ultralytics.com/license>
- GNU AGPL-3.0: <https://www.gnu.org/licenses/agpl-3.0.html>
- RTMLib: <https://github.com/Tau-J/rtmlib>
- RTMPose: <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose>
- TrackNet-compatible source referenced by the fetch helper:
  <https://github.com/yastrebksv/TrackNet>

## Operator checklist

1. Record the exact model source, version, SHA-256, author, training-data provenance, and
   license alongside every deployed checkpoint.
2. Do not infer license permission from architecture compatibility or successful loading.
3. Recheck upstream terms when updating `ultralytics` or model weights.
4. Keep unknown-provenance checkpoints out of redistributed artifacts and production use.
