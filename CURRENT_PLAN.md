# CURRENT_PLAN.md

## Plan Authority

This file is the active implementation source of truth for the Task 1 Walker
S2 project. If this file and older logs or context disagree, this file wins.

Codex must only implement the current phase unless the user explicitly
authorizes a phase change. After each implementation run, Codex must update
`TASK_LOG.md` and, if relevant, `CURRENT_PLAN.md` and `PROJECT_CONTEXT.md`.

## Project Direction

Task 1 final competition runtime must be camera-first.

The final runtime observation source is head and wrist RGB-D camera data.
Simulator ground truth is allowed for dataset labeling, evaluation, debugging,
and controlled bootstrapping, but not as the final competition runtime input.

Preserve:

- deterministic phase-based execution
- `DualArmIK` as the execution backend
- existing coordinate transform conventions
- useful `grasp_planner.py` logic for arm choice, TCP offsets, grasp target
  construction, and robot-specific geometry

Do not implement:

- end-to-end image-to-action control
- Thinker direct final 6D grasp pose generation
- Thinker direct joint or per-frame motion control
- silent frame-convention changes
- broad manipulation-stack rewrites

Target architecture:

```text
camera input
-> structured perception / Thinker output
-> depth + geometry + transforms
-> grasp-relevant 3D state
-> compact candidate grasp generation
-> robot-specific rescoring
-> deterministic planner
-> DualArmIK
-> execution phases
```

Overall priorities:

- correctness with respect to competition direction
- runtime stability
- simplicity and debuggability
- reproducibility
- baseline compatibility
- performance optimization only after a stable baseline exists

## Phase 0: Context Reset

Status: complete

Goal:

- record the old direction versus the new camera-first direction
- separate confirmed repository facts from inferred architecture and items that
  still need Linux runtime testing
- preserve the selected deterministic manipulation source and backend

Confirmed source file to preserve for deterministic manipulation:

- `scripts/task1_dualarmik_phase_baseline.py`

Confirmed subsystems to preserve:

- official Task 1 `SceneBuilder` setup
- official `RobotArticulation` camera wrapper
- `DualArmIK`
- `CoordinateTransform`
- deterministic grasp/planner flow and logs
- current Task 1 manipulation scripts unless a future phase explicitly patches
  them

Exit criteria status:

- met in this reset run

## Phase 1: Synchronized Camera + Truth Collector

Status: complete

Implemented in this phase:

- `scripts/task1_collect_rgbd_labels.py`
- `docs/task1_data_collection_schema.md`
- `docs/schemas/task1_thinker_structured_output.schema.json`
- `docs/schemas/task1_evaluator_io.schema.json`

Collector requirements:

- build the official Task 1 scene through the organizer-provided setup
- use `RobotArticulation.get_cameras_images(step)` for camera capture
- record head-left, head-right, wrist-left, and wrist-right RGB-D
- write simulator-truth labels for each spawned Task 1 object
- include object id, class, world pose, USD robot-root base-frame pose,
  table-frame pose, yaw/coarse orientation, target-bin metadata, and best-effort
  projection visibility
- write runtime metadata for chosen object, chosen arm, chosen preset, chosen
  candidate, planner target, execution result, fail reason, simulation step,
  and timestamp
- write synchronization debug records that help diagnose missing cameras, depth,
  shape mismatches, and capture timing
- keep all outputs under `$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/`

Current output shape:

```text
<run_id>/
  run_metadata.json
  manifest.jsonl
  rgb/<sample_id>/<camera>.npy
  depth/<sample_id>/<camera>.npy
  labels/<sample_id>.json
  metadata/<sample_id>.json
  sync_debug/<sample_id>.json
```

Important boundary:

- Phase 1 collects truth for labels and evaluation only.
- Phase 1 does not make simulator truth the competition runtime input.
- Phase 1 does not modify manipulation/control code.

Exit criteria:

- lightweight Python/schema checks pass on the development machine
- Linux Isaac Sim run can collect a small sample set
- collected sample structure matches `docs/task1_data_collection_schema.md`

Exit criteria status:

- met for the frozen Phase 1 baseline
- frozen baseline commit: `ee6ca51` (`Restore Task 1 RGB-D truth collector`)
- Linux Isaac Sim smoke run:
  `$OUTPUT_ROOT/datasets/task1_rgbd_labels/test_phase1_initfix_1`
- smoke run validation: 3 manifest entries, 4 cameras per sample, 12 RGB
  arrays, 12 depth arrays, 3 label files, 3 metadata files, 3 sync debug files,
  positive depth finite counts, and 4 labeled Task 1 objects per sample

## Phase 2: Automatic Evaluator

Status: active

Goal:

- compute perception, geometry, recommendation, and task metrics from Phase 1
  samples and runtime traces

Implemented in this phase:

- `scripts/task1_evaluate_dataset.py`

Current evaluator capabilities:

- validates `manifest.jsonl`, referenced camera arrays, labels, metadata, and
  sync debug sidecars
- verifies four-camera RGB-D completeness for head-left, head-right,
  wrist-left, and wrist-right
- loads RGB/depth `.npy` arrays, checks shapes, and summarizes depth finite
  counts
- checks label/metadata required fields and object-count consistency
- accepts optional direct prediction JSON/JSONL, Thinker output, geometry
  output, planner trace, execution log, or evaluator-I/O wrapper inputs
- computes prediction metrics when matching prediction fields are available

Current validation:

- real Phase 1 run
  `$OUTPUT_ROOT/datasets/task1_rgbd_labels/test_phase1_initfix_1` passes
  structural validation with 3 samples, 12 RGB arrays, 12 depth arrays, 3 label
  files, 3 metadata files, 3 sync debug files, complete four-camera records,
  positive depth finite counts, and object_count `[4]`
- prediction metric code path was smoke-tested with synthetic truth-derived
  predictions outside the repo; real Thinker/geometry/planner prediction inputs
  are still pending

Required metrics:

- class accuracy
- selected-object accuracy
- 2D center error
- yaw bucket accuracy
- arm recommendation accuracy
- preset recommendation accuracy
- 3D conversion error after depth + geometry
- task success rate
- wrong-bin rate
- drop rate
- cycle time

Required work:

- implement evaluator reader for `manifest.jsonl`, labels, metadata, sync debug,
  Thinker output, geometry output, planner traces, and execution logs
- validate the placeholder interface in
  `docs/schemas/task1_evaluator_io.schema.json`
- report missing or malformed fields clearly
- do not change planner, IK, or manipulation behavior in this phase

Exit criteria:

- evaluator can score a Linux-collected Phase 1 sample set without ad hoc paths

Exit criteria status:

- structurally met for Phase 1 sample validation
- prediction metric plumbing is implemented, but real prediction inputs are
  still required before metric values are meaningful competition evidence

## Phase 3: Camera-Only Baseline Without Thinker Dependency

Status: pending

Goal:

- create the first functional camera-first runtime baseline without requiring
  Thinker

Required work:

- use head RGB-D for object localization
- convert 2D/ROI plus depth into grasp-relevant 3D state using existing
  transforms
- estimate coarse yaw, width, and usable object state where possible
- generate a compact candidate grasp set
- rescore candidates using robot-specific checks and `DualArmIK`
- keep deterministic checkpoint-based execution phases

Non-goals:

- no fixed "stop every 2 seconds" policy
- no aggressive wrist retargeting during final descent
- no Thinker dependency

## Phase 4: Thinker Structured Perception Integration

Status: pending

Goal:

- connect Thinker as a structured visual understanding and decision-support
  component

Thinker output schema:

- `docs/schemas/task1_thinker_structured_output.schema.json`

Thinker may output:

- object candidates
- class
- ROI or 2D center
- coarse orientation bucket
- difficulty / occlusion
- confidence
- recommended arm
- recommended preset
- selected object id

Thinker must not output or control:

- final 6D grasp pose as the first integration target
- robot joint commands
- per-frame motion commands
- permissions to bypass geometry, IK, or safety checks

## Phase 5: Wrist Local Refinement

Status: pending

Goal:

- add wrist-camera local correction only if Phase 2 and Phase 3 metrics show it
  improves success rate

Rules:

- use wrist camera only near the target
- apply small XY/yaw corrections only
- if correction is small, update the target in place
- if correction is large, return to fine pregrasp or replan
- do not destabilize monotonic final Z descent

## Current Non-Goals

- no planner redesign
- no IK redesign
- no grasp logic tuning in the data-collection phase
- no full evaluator runtime in Phase 1
- no full Thinker runtime in Phase 1
- no segmentation training pipeline unless a later phase explicitly requires it
