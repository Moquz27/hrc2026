# Task 1 RGB-D Data Collection Schema

This document describes the Phase 1 Task 1 collection output written by
`scripts/task1_collect_rgbd_labels.py`.

## Scope

Implemented now:

- synchronized collection from the existing official `RobotArticulation`
  camera wrappers
- head left/right RGB-D
- wrist left/right RGB-D
- simulator truth labels for spawned Task 1 parts
- JSONL manifest with one entry per sample
- per-sample labels, metadata, and synchronization debug records

Not implemented now:

- camera-first manipulation runtime
- evaluator execution
- Thinker runtime integration
- Thinker-generated final grasp poses
- pixel-accurate segmentation or occlusion-filtered visibility

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
ambiguity in Phase 1.

## Manifest Entry

Each line of `manifest.jsonl` is one sample. Required fields:

- `run_id`: collection run id
- `sample_id`: stable sample key such as `sample_000000`
- `sample_index`: zero-based index
- `simulation_step`: step counter at capture time
- `timestamp_utc`: wall-clock capture timestamp
- `paths.labels`: relative path to sample labels JSON
- `paths.metadata`: relative path to runtime metadata JSON
- `paths.sync_debug`: relative path to synchronization debug JSON
- `cameras`: per-camera RGB/depth path, shape, dtype, and finite-count summary
- `object_count`: number of labeled spawned Task 1 objects
- `execution_result`: runtime result string
- `fail_reason`: failure reason when available

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
- `class`: `part_a`, `part_b`, or fallback classification
- `class_source`: `usd_reference_path` or creation-order fallback
- `world_pose.position_xyz_m`
- `world_pose.orientation_xyzw`
- `world_pose.yaw_rad`, `world_pose.yaw_deg`
- `world_pose.coarse_orientation`
- `base_frame_pose`: pose in the USD robot-root frame when available
- `table_frame_pose`: pose in the Task 1 robot-facing table frame when available
- `bbox_world`: USD world-aligned bounding box
- `target_bin`: semantic/configured bin metadata from `Part_Sorting.yaml`
- `visibility`: best-effort camera center-projection metadata

Base-frame labels use the USD robot-root transform and explicitly do not
replace the existing Pinocchio `CoordinateTransform` used by manipulation.

## Runtime Metadata

Each `metadata/<sample_id>.json` includes:

- `selected_arm`
- `selected_preset`
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

Phase 1 synchronization means no `sim_app.update()` is advanced between
individual camera RGB-D requests for a sample.
