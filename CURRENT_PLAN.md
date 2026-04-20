# CURRENT_PLAN.md

## Plan Authority

This file is the active implementation source of truth for the Task 1 hybrid
grasp project. If this file and older logs or context disagree, this file wins.

Codex must only implement the current phase unless the user explicitly
authorizes a phase change. After each implementation run, Codex must update
`TASK_LOG.md` and, if relevant, `CURRENT_PLAN.md` and `PROJECT_CONTEXT.md`.

## Project

Task 1 hybrid geometric grasp planner with `scene_state` now and Thinker later.

Overall priorities:

- competition reliability first
- deterministic behavior first
- simple debugging first
- preserve baseline compatibility
- modular future perception swap
- Thinker never controls joints

## Phase 0: Baseline Source Lock

Status: complete

Goal:

- choose the best existing Task 1 script as the source to fork from

Selected source file path:

- `scripts/task1_dualarmik_phase_baseline.py`

Short justification:

- Scene setup: uses official `Part_Sorting.yaml` through the organizer
  `SceneBuilder`, overrides the asset root to the verified challenge assets,
  builds Task 1 table, parts, and Walker S2, and preserves the diagnostic static
  bin collider workaround for the known composed-box physics issue.
- Robot loading: builds Walker S2 via `SceneBuilder.build_robot`, detects and
  acquires the articulation with fallback paths, loads the official startup
  joint map, and uses the official `DualArmIK` / `CoordinateTransform` classes.
- Motion structure: has an explicit gated phase sequence for pregrasp,
  alignment, descend, close, micro-lift, lift, carry/place, release, and
  retreat, with region-aware approach families and deterministic candidate
  evaluation.
- Logging quality: writes rolling and per-run logs under `LOG_ROOT`, with
  target selection, phase logs, object trace, candidate diagnostics,
  coordinate/EE-frame diagnostics, and real-grasp-center diagnostics.
- Simplicity: it is not the smallest Task 1 script, but it is the best safe
  fork point among the current candidates because it keeps the latest official
  IK path and safety gates. `scripts/task1_dualarmik_phase_nogate.py` is a
  diagnostic bypass copy, not a baseline source.

Subsystems to preserve:

- official Task 1 `SceneBuilder` table, part, and robot setup
- asset-root override through environment-derived paths
- diagnostic static bin collider workaround
- target selection records and object/category extraction
- official startup joint map and arm/gripper DOF selection
- official gripper open/close widths and sustained grip effort
- `DualArmIK` / `CoordinateTransform` loading
- coordinate transform and EE-frame diagnostics
- deterministic pregrasp candidate evaluation
- phase logging and object-centric success/failure checks
- pregrasp / align / descend / close / micro-lift / lift / carry / place /
  release / retreat phase structure
- mandatory contact and safety gates; no-gate behavior stays diagnostic only

Required outputs:

- selected source file path
- short justification based on:
  - scene setup
  - robot loading
  - motion structure
  - logging quality
  - simplicity
- list of subsystems to preserve

Exit criteria:

- source file chosen and documented

Exit criteria status:

- met in this initialization run

## Phase 1: Competition-Oriented Task 1 Data Collection

Status: active

User-authorized reset on 2026-04-21:

- preserve the existing deterministic manipulation backend
- add synchronized RGB-D and simulator-truth data collection first
- prepare evaluator and Thinker structured-output interfaces only as schemas
- do not connect Thinker to runtime control
- do not advance into model-based grasp generation

Implemented Phase 1 outputs:

- `scripts/task1_collect_rgbd_labels.py`
- `docs/task1_data_collection_schema.md`
- `docs/schemas/task1_thinker_structured_output.schema.json`
- `docs/schemas/task1_evaluator_io.schema.json`

The collector reuses the official `RobotArticulation.get_cameras_images(step)`
camera interface and writes structured samples under
`$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/`.

Current explicit non-goals:

- no camera-first manipulation runtime
- no evaluator runner
- no Thinker runtime integration
- no Thinker final grasp-pose generation
- no changes to `DualArmIK`, coordinate transforms, planner flow, or current
  manipulation logs

## Phase 1A: Minimal Hybrid Skeleton

Status: paused by user-authorized data-collection reset

Goal:

- create a new Task 1 script copied from the selected source
- get a minimal hybrid pipeline running without Thinker

Required outputs:

- new script file in `scripts/`
- `scene_state`-only perception
- normalized `object_info`
- simple candidate generation
- deterministic candidate selection
- pregrasp / descend / close / lift reuse from baseline

Exit criteria:

- script runs without breaking baseline infrastructure

## Phase 2: Dataset Validation And Evaluator Harness

Status: pending

Goal:

- validate collected Task 1 samples and make evaluator inputs/outputs concrete
- keep this phase data/evaluation oriented before further manipulation tuning

Required outputs:

- sample validator for `manifest.jsonl`, RGB/depth arrays, labels, metadata,
  and sync debug records
- evaluator input loader based on
  `docs/schemas/task1_evaluator_io.schema.json`
- minimal evaluator result writer with pass/fail/skip status and metric fields
- sync checks for missing cameras, missing depth, bad shapes, stale simulation
  step, and label/object-count mismatches
- no planner, IK, coordinate-transform, or deterministic phase execution changes

Exit criteria:

- Linux run produces a small valid collection and the evaluator harness can
  read it without ad hoc path assumptions

## Phase 3: Thinker Advisor Integration

Status: pending

Goal:

- add Thinker as optional multimodal strategy advisor

Required outputs:

- `scripts/utils/thinker_advisor.py`
- lazy model load
- multimodal payload support
- text-only fallback
- deterministic fallback on any failure
- strict JSON validation
- Thinker only allowed to:
  - rerank top-K
  - suggest arm preference
  - suggest approach family
  - suggest retry strategy
- Thinker not allowed to:
  - generate final robot pose
  - bypass geometric checks
  - control joints
  - run inside final descent loop

Exit criteria:

- Thinker can influence candidate selection safely

## Phase 4: Perception Abstraction For Future YOLO

Status: pending

Goal:

- clean provider abstraction while keeping `scene_state` as the real current
  provider
- keep Thinker/camera outputs as intermediate structured observations, not
  final grasp poses

Required outputs:

- `resolve_object_infos(...)`
- `scene_state_provider` implemented
- `yolo_provider` stub only
- explicit `perception_source` logging
- consume or validate
  `docs/schemas/task1_thinker_structured_output.schema.json` only after the
  Phase 2 evaluator/data contracts are stable

Exit criteria:

- future YOLO can replace provider without rewriting planner

## Phase 5: Reliability Tuning For Competition

Status: pending

Goal:

- optimize speed, stability, and failure handling for Task 1 scoring

Required outputs:

- threshold tuning
- failure taxonomy
- improved retry policy
- logging for benchmark runs
- multi-seed evaluation notes

Exit criteria:

- stable baseline for repeated simulation trials
