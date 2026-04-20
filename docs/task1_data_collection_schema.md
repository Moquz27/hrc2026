# Task 1 Minimal RGB-D Collection Format

`scripts/task1_collect_rgbd_labels.py` writes the small dataset needed for the
future Task 1 path:

```text
camera images -> Thinker predicts class + table-frame pose -> planner/control
```

It still builds the official Task 1 scene and still captures through the
official `RobotArticulation.get_cameras_images(step)` wrapper.

## Output Layout

Default root:

```text
$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/
```

Default folder shape:

```text
<run_id>/
  manifest.jsonl
  rgb/
    sample_000000/
      head_left.png
      head_right.png
      wrist_left.png
      wrist_right.png
  labels/
    sample_000000.json
```

If `--save-depth` is used:

```text
<run_id>/
  depth/
    sample_000000/
      head_left.npy
      head_right.npy
      wrist_left.npy
      wrist_right.npy
```

PNG is the default RGB format. If Pillow is unavailable at runtime, the
collector falls back to `.npy` for that RGB frame and records the actual path in
the manifest. `--rgb-format npy` forces `.npy` RGB storage.

## Label Schema

Each label file is intentionally shaped like the future Thinker perception
output:

```json
{
  "objects": [
    {
      "class": "A",
      "x": 0.123,
      "y": -0.045,
      "yaw": 1.57
    }
  ]
}
```

Fields:

- `class`: `A`, `B`, or `unknown`
- `x`: object position in the Task 1 table frame, meters
- `y`: object position in the Task 1 table frame, meters
- `yaw`: object yaw in the Task 1 table frame, radians

All spawned Task 1 objects appear in `objects`.

The label file does not include world pose, base-frame pose, bounding boxes,
planner targets, execution result, fail reason, sync debug, or visibility
metadata. Those were removed because they do not match the competition
perception target and make training inputs harder to inspect.

## Table Frame

The canonical label frame is the existing robot-facing Task 1 table frame:

- origin: near-left tabletop corner from the robot viewpoint
- `x`: along the near table edge from robot-left toward robot-right
- `y`: from the robot side toward the far side of the table
- `yaw`: object yaw expressed in this table frame

This keeps perception labels independent of world placement while preserving
the frame convention already used by the Task 1 planner work.

## Label Noise

Exported labels receive noise by default to reduce overfitting to perfect
simulator truth:

- XY Gaussian noise: sigma `0.005 m`
- yaw Gaussian noise: sigma `3 degrees`

Noise is applied only to the exported label JSON. The raw simulator truth used
inside the collector is not modified.

Noise CLI:

```bash
--label-noise
--no-label-noise
--label-xy-noise-sigma-m 0.005
--label-yaw-noise-sigma-deg 3.0
--noise-seed 1
```

`--label-noise` is enabled by default.

## Manifest Schema

Each line of `manifest.jsonl` is one sample:

```json
{
  "run_id": "20260421T000000Z_seed1",
  "seed": 1,
  "sample_id": "sample_000000",
  "sample_index": 0,
  "rgb": {
    "head_left": "rgb/sample_000000/head_left.png"
  },
  "label": "labels/sample_000000.json",
  "has_depth": false,
  "depth": {},
  "object_count": 4,
  "label_noise": {
    "applied": true,
    "xy_sigma_m": 0.005,
    "yaw_sigma_deg": 3.0,
    "seed": 1
  }
}
```

The manifest is only an index: sample id, image paths, label path, depth
availability, object count, run id, seed, and label-noise settings.

## Collector CLI

Common flags:

```bash
--samples 100
--sample-stride 5
--seed 1
--gui
--hold-open
--save-depth
--no-save-depth
--label-noise
--no-label-noise
--label-xy-noise-sigma-m 0.005
--label-yaw-noise-sigma-deg 3.0
--rgb-format png
```

Depth is disabled by default to keep the dataset small. Enable it only when the
future perception path will use depth.

## Removed From The Previous Heavy Format

Removed default outputs:

- `run_metadata.json`
- per-sample `metadata/`
- per-sample `sync_debug/`
- world pose labels
- base-frame pose labels
- bounding boxes
- target-bin metadata
- planner target / selected preset / selected arm
- execution result / fail reason
- evaluator placeholder output in the dataset

Why: the immediate competition dataset only needs images and table-frame
object labels. Extra runtime internals made the data harder to train on and
inspect without improving the camera -> Thinker -> planner contract.
