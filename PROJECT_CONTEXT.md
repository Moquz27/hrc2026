Project: HRC 2026 Robotics Competition

## Official Resources
Official external competition resources are tracked in `docs/resources.md`.
Use that file to distinguish baseline code, simulation assets, dataset resources, and the Walker S2 robot model source.

## Current Phase

- Current active task phase: Phase 1 - Task 1 synchronized RGB-D and simulator-truth data collection
- Current status: user-authorized camera-first direction reset on 2026-04-21
- Phase 1 focus: collect synchronized head/wrist RGB-D, simulator-truth labels, runtime metadata, and sync diagnostics while preserving the deterministic manipulation backend
- Phase 2 starts after the collector is run on Linux and the saved sample structure is validated against evaluator inputs
- Phase 4 remains deferred until Phase 2 proves the structured perception/evaluation contract against collected samples

Older Phase 2/Phase 3 manipulation experiments remain useful history, but the
active implementation contract is now `CURRENT_PLAN.md`.

## Camera-First Direction Reset

Old direction now considered incomplete for final competition runtime:

- scene-state and simulator-truth driven grasping is useful for debugging,
  labeling, and bootstrapping
- it must not be treated as the final online-round runtime observation source

New source-of-truth direction:

- final Task 1 runtime is camera-first
- head and wrist RGB-D cameras are the runtime observations
- simulator truth may be used internally for dataset labeling, evaluation,
  debugging, and controlled bootstrapping only
- deterministic execution, `DualArmIK`, coordinate transforms, and useful
  `grasp_planner.py` logic remain the manipulation/control backbone
- Thinker is an intermediate structured visual-understanding component, not a
  direct final grasp-pose generator and not an end-to-end controller

Confirmed from repository inspection:

- `RobotArticulation.get_cameras_images(step)` provides the official
  head-left, head-right, wrist-left, and wrist-right RGB-D path
- `DualArmIK.py` is still the primary IK/control backend
- `coordinate_utils.py` owns world/base transform conventions for manipulation
- `grasp_planner.py` contains useful deterministic arm choice, grasp target,
  TCP offset, and orientation logic that should be extended, not replaced

Inferred architecture to build in later phases:

```text
camera input
-> structured perception / Thinker output
-> depth + geometry + transforms
-> grasp-relevant 3D state
-> compact candidate grasp generation
-> robot-specific rescoring
-> deterministic planner
-> DualArmIK
-> phase-based execution
```

Still needs Linux runtime testing:

- camera capture synchronization and depth stability
- table-frame label correctness against physical Task 1 scene layout
- 2D projection quality and visibility metadata usefulness
- depth/geometry conversion error for object centers and yaw buckets
- evaluator metric definitions and thresholds
- camera-only baseline success before Thinker integration

## Plan-Driven Workflow

- `CURRENT_PLAN.md` is the active implementation contract for plan-driven task work.
- If `CURRENT_PLAN.md` disagrees with older logs or context, `CURRENT_PLAN.md` wins.
- Codex must only implement the current phase unless the user explicitly authorizes a phase change.
- After each implementation run, update `TASK_LOG.md` and update `CURRENT_PLAN.md` / `PROJECT_CONTEXT.md` when relevant.
- Current Task 1 hybrid grasp plan phase: Phase 0 baseline source lock.
- Phase 0 selected source: `scripts/task1_dualarmik_phase_baseline.py`.

Architecture:
- Mac: development machine (coding, planning, lightweight testing)
- Linux: runtime machine (simulation, evaluation, dataset, logs)

Core Workflow:
- Code is written on Mac
- Code is synced via Git
- Code is executed only on Linux
- Results are analyzed and iteration continues on Mac

Rules:
- Never run heavy simulation on Mac
- Never store dataset, checkpoints, or outputs in Git
- Always use environment variables for paths (no hardcoded local paths)
- Keep code minimal, modular, and testable
- Prioritize stability before optimization

Environment:
- Linux uses environment variables:
  HRC_ROOT, HRC_REPO, DATA_ROOT, CKPT_ROOT, OUTPUT_ROOT, LOG_ROOT
- Code must run on both Mac and Linux without modification

Development Strategy:
- Build baseline first (deterministic pipeline)
- Avoid premature optimization
- Avoid large refactors without benchmark evidence

Current Focus:
- Phase: Task 1 data collection reset
- Goal: make official Task 1 camera observations, simulator-truth labels, and evaluator/Thinker interface contracts reproducible
- No serious algorithm or ML optimization until collected samples and evaluator contracts are stable
- Task 1 continuous-motion baseline now has per-object diagnostics and one-knob tuning support in `scripts/task1_smooth_autoseed_multi_object_baseline.py`.
- Latest controlled Linux runtime sweep for seed=1 target-index=2 showed grasp-depth offsets 0.0, -0.005, and -0.010 all failed before grasp at `pre_grasp_unreachable`; next single tuning family should be approach/soft waypoint reachability, not deeper grasp or carry/place tuning.

Assumptions:
- Dataset not fully available yet
- Simulator not fully set up yet
- Focus is on reproducible pipeline, not performance

Reference:
- Full detailed context is in docs/context_full.md
- Current baseline maturity and script classification are in docs/baseline_status.md
