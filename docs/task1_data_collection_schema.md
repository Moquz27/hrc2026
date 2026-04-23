# Task 1 RGB-D Data Collection Schema

This document describes the Phase 1 Task 1 collection output written by
`scripts/task1_collect_rgbd_labels.py`.

## Scope

Implemented now:

- official Task 1 `SceneBuilder` table, part, and robot setup
- synchronized collection through `RobotArticulation.get_cameras_images(step)`
- head left/right RGB-D
- wrist left/right RGB-D
- simulator truth labels for spawned Task 1 parts
- one JSONL manifest entry per sample
- per-sample labels, runtime metadata, and synchronization debug records

Not implemented now:

- camera-first manipulation runtime
- automatic evaluator execution
- Thinker runtime integration
- Thinker-generated final grasp poses
- pixel-accurate segmentation labels
- occlusion-filtered object labeling

Simulator truth is for dataset labeling, evaluation, debugging, and controlled
bootstrapping only. It is not the final competition runtime input.

## Output Root

Default output root:

```text
$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/
```

The collector refuses to write inside the code repository unless the output
path is changed outside the repo.

## Folder Layout

```text
<run_id>/
  run_metadata.json
  manifest.jsonl
  rgb/
    sample_000000/
      head_left.npy
      head_right.npy
      wrist_left.npy
      wrist_right.npy
  depth/
    sample_000000/
      head_left.npy
      head_right.npy
      wrist_left.npy
      wrist_right.npy
  labels/
    sample_000000.json
  metadata/
    sample_000000.json
  sync_debug/
    sample_000000.json
```

RGB and depth are saved as `.npy` arrays to avoid image codec and depth-scale
ambiguity in Phase 1. Depth is enabled by default because this phase is a
synchronized RGB-D collector; `--no-save-depth` exists only for quick storage or
debug runs.

## Manifest Entry

Each line of `manifest.jsonl` is one sample. Key fields:

- `run_id`: collection run id
- `seed`: scene/scatter seed
- `sample_id`: stable sample key such as `sample_000000`
- `sample_index`: zero-based index
- `simulation_step`: step counter at capture time
- `timestamp_utc`: wall-clock capture timestamp
- `paths.labels`: relative path to sample labels JSON
- `paths.metadata`: relative path to runtime metadata JSON
- `paths.sync_debug`: relative path to synchronization debug JSON
- `cameras`: per-camera RGB/depth path, shape, dtype, and finite-count summary
- `object_count`: number of labeled spawned Task 1 objects
- `chosen_object_id`, `chosen_arm`, `chosen_preset`: runtime selection fields
  when known
- `execution_result`, `fail_reason`: collection/run status fields

The manifest is an index. Detailed truth labels and synchronization diagnostics
live in the sidecar JSON files.

## Camera Fields

Camera names are fixed to the official baseline names:

- `head_left`
- `head_right`
- `wrist_left`
- `wrist_right`

Each camera record contains:

- `source_interface`: `RobotArticulation.get_cameras_images`
- `rgb.path`: relative `.npy` path or `null`
- `rgb.shape`: array shape when available
- `rgb.dtype`: array dtype when available
- `depth.path`: relative `.npy` path or `null`
- `depth.shape`: array shape when available
- `depth.dtype`: array dtype when available
- `image_shape_hw`: `[height, width]` inferred from RGB or depth

## Label Fields

Each object label contains:

- `object_id`: stable run-local id such as `task1_part_000`
- `prim_path`: USD prim path
- `class`: `A`, `B`, or `unknown`-style future-facing class value
- `raw_class`: simulator/source class such as `part_a` or `part_b`
- `class_source`: `usd_reference_path` or creation-order fallback
- `world_pose.position_xyz_m`: world-frame position in meters
- `world_pose.orientation_xyzw`: world-frame quaternion in `[x, y, z, w]`
  order
- `world_pose.yaw_rad`, `world_pose.yaw_deg`: yaw about world `+Z`
- `world_pose.coarse_orientation`: bucket computed from world-frame yaw
- `base_frame_pose`: pose in the USD robot-root frame when available, with
  position in meters and yaw about the USD robot-root `+Z`
- `table_frame_pose`: pose in the Task 1 robot-facing table frame when
  available, with position in meters and yaw about `+z_table`
- `bbox_world`: USD world-aligned bounding box
- `target_bin`: semantic/configured bin metadata from `Part_Sorting.yaml`
- `visibility`: weak best-effort projection metadata

Base-frame labels use the USD robot-root transform and explicitly do not
replace the existing Pinocchio `CoordinateTransform` used by manipulation.

Visibility labels are not visibility truth. `visible_projection` is computed
from center projection only. The collector does not use segmentation, does not
perform true occlusion reasoning, and does not compute depth finite ratio inside
an ROI. Per-camera `bbox_projection` is a debug projection of the 3D bbox and
must not be treated as an occlusion-aware visible-object label.

## Table Frame

The table frame is the robot-facing Task 1 table convention already used by
the Task 1 planner work:

- origin: near-left tabletop corner from the robot viewpoint
- `x`: along the near edge from robot-left toward robot-right
- `y`: from the robot side toward the far side of the table
- `z`: up from the tabletop surface

When the USD robot root exists in the stage, this table frame is built from the
actual stage robot pose. The YAML `robot_position` / `robot_rotation` pose is
used only as a fallback. `run_metadata.json` records both pose sources and logs
a warning when their position or yaw differs beyond the collector thresholds.

`table_frame_pose` includes `x`, `y`, `z`, `yaw_rad`, `yaw_deg`,
`coarse_orientation`, and the full orientation quaternion in that frame.

Strict unit and yaw semantics:

- `table_frame_pose.position_xyz_m`, `x`, `y`, and `z` are meters.
- `run_metadata.scene.table_frame.x_extent_m` and `y_extent_m` are meters.
- `run_metadata.scene.table_frame.x_extent_unit` and `y_extent_unit` are table
  units, using `TABLE_UNIT_M = 0.035`.
- `table_frame_pose.yaw_rad` / `yaw_deg` is rotation about `+z_table`.
- `table_frame_pose.coarse_orientation` is computed from table-frame yaw, not
  world-frame yaw.

## Runtime Metadata

Each `metadata/<sample_id>.json` includes:

- `chosen_object_id`
- `chosen_arm`
- `chosen_preset`
- `chosen_candidate`
- `selected_object_id`, `selected_arm`, `selected_preset`, `selected_candidate`
  aliases for older Task 1 logs/scripts
- `planner_target`
- `execution_result`
- `fail_reason`
- `simulation_step`
- `sim_time_estimate_s`
- `timestamp_utc`
- `collector_only`

For pure collection runs, `execution_result` defaults to
`collection_only_no_manipulation`.

## Synchronization Debug

Each `sync_debug/<sample_id>.json` includes:

- `capture_step`
- `same_simulation_step_for_all_cameras`
- `camera_capture_order`
- `camera_capture_elapsed_s`
- per-camera availability, shape, dtype, and depth finite-value summaries
- warnings for missing RGB, missing depth, missing labels, or failed visibility
  projection

Phase 1 synchronization means the collector calls
`RobotArticulation.get_cameras_images(step)` once for a sample and does not
advance `sim_app.update()` between the camera RGB-D reads performed by that
official wrapper.

## CLI

Common flags:

```bash
python3 scripts/task1_collect_rgbd_labels.py \
  --samples 100 \
  --sample-stride 5 \
  --seed 1
```

GUI run:

```bash
python3 scripts/task1_collect_rgbd_labels.py \
  --gui \
  --hold-open \
  --samples 5 \
  --sample-stride 5 \
  --seed 1
```

Useful flags:

- `--samples`
- `--sample-stride`
- `--seed`
- `--gui`
- `--hold-open`
- `--save-depth` / `--no-save-depth`
- `--chosen-object-id`
- `--chosen-arm`
- `--chosen-preset`
- `--chosen-candidate-json`
- `--planner-target-json`
- `--execution-result`
- `--fail-reason`

## Removed From The Previous Minimal Format

The previous simplified collector exported only RGB plus table-frame `class`,
`x`, `y`, and `yaw`, with optional noisy labels. That was too narrow for the
new source-of-truth direction because Phase 2 needs enough truth to evaluate
camera-first perception, depth geometry, object selection, arm/preset
recommendations, and runtime outcomes.

Phase 1 now restores synchronized RGB-D, truth labels, minimal runtime
metadata, and sync diagnostics. It still does not alter manipulation/control
logic or make simulator truth a competition runtime input.
