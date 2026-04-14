# Baseline Status

Last updated: 2026-04-14

This document defines the current competition baseline honestly. It is meant to
help a new agent find the right script quickly and avoid treating diagnostics as
task-solving code.

## Baseline Definition

Current baseline means:
- Isaac Sim can start on the Linux runtime machine.
- Required runtime environment variables are validated.
- Walker S2 can load from runtime assets outside the repo.
- Basic Walker S2 arm and gripper motion can be commanded and logged.
- The strongest current manipulation-related script is still motion sanity, not
  object transport or task solving.

Current baseline does not mean:
- Task 1 sorting is implemented.
- A/B perception or classification is implemented.
- A physical grasp, carry, and release of a dynamic workpiece is verified.
- Official task scenes are integrated.
- The official evaluator entrypoint exists.

## Script Classification

| Script | Class | Purpose | Result Evidence | Limitation |
| --- | --- | --- | --- | --- |
| `scripts/smoke_isaac.py` | diagnostic | Checks Isaac Sim startup, env vars, stage stepping, and `LOG_ROOT` writes. | `LOG_ROOT/isaac_smoke_ok.txt` | No robot, assets, task logic, or control. |
| `scripts/minimal_scene_baseline.py` | baseline | Minimal reproducible Isaac scene with one static cube and metrics output. | `LOG_ROOT/minimal_scene_baseline.log`, `OUTPUT_ROOT/metrics/minimal_scene_baseline.json` | No robot or task assets. |
| `scripts/load_walker_s2.py` | baseline | Loads Walker S2 USD, detects articulation and joints, rejects Git LFS pointers. | `LOG_ROOT/walker_s2_load_ok.txt` | Structural load only; no control or task scene. |
| `scripts/control_walker_s2_arms.py` | baseline | Verifies articulation DOF observation and small arm position commands. | `LOG_ROOT/walker_s2_arm_control_smoke.log` | Basic arm motion only; no object interaction. |
| `scripts/move_walker_s2_end_effector.py` | experiment | Tests simple right-arm Cartesian target reaching with damped least-squares IK. | `LOG_ROOT/walker_s2_end_effector_target.log` | Position-only IK can choose visually unsafe postures; not current preferred control path. |
| `scripts/grasp_static_object_smoke.py` | experiment | Tests right gripper motion and wrist poses around a fixed target cube. | `LOG_ROOT/walker_s2_static_grasp_smoke.log` | Misleading if read as grasp success: object transport is not verified. |
| `scripts/right_arm_joint_space_sanity.py` | diagnostic | Checks explicit right-arm joint-space signs, front pose, raise/lower motion, and gripper movement. | `LOG_ROOT/walker_s2_right_arm_joint_space_sanity.log` | Visual/motion sanity only; no task assets. |
| `scripts/front_seeded_manipulation_motion.py` | baseline | Current strongest manipulation-related motion baseline: repeated front-seeded right-arm joint-space phases plus gripper commands. | `LOG_ROOT/walker_s2_front_seeded_manipulation_motion.log` | No dynamic object, no physical grasp, no sorting, no official scene. |

No files are renamed in this pass. Existing script names remain stable for log
comparison and runtime repeatability. Misleading limitations are documented here
and in the relevant script docstrings.

## Baseline Maturity

| Area | Status | Evidence | Unverified |
| --- | --- | --- | --- |
| Environment baseline | Pass | Isaac smoke and minimal scene logs exist under `LOG_ROOT`; runtime paths use env vars. | Official assets beyond Walker S2 are not verified. |
| Robot-load baseline | Pass | Walker S2 loads from runtime assets, articulation root detected, 42 joints logged. | Repeated load/reset robustness in official scenes. |
| Robot-control baseline | Partial pass | Arm DOF commands, right-arm joint-space motion, and right gripper commands are logged. | Waist, left arm, two-arm coordination, collision behavior in task scenes. |
| Manipulation baseline | Not yet verified | Front-seeded motion is repeatable; gripper DOFs move. | Dynamic workpiece grasp, lift, carry, release, bin placement. |
| Task baseline | Not implemented | No task pipeline or evaluator entrypoint exists. | Task 1/2/3/4 success, perception, classification, scoring, official scene integration. |

## Quick Entry Points

Use these scripts in order when checking a fresh Linux runtime:

1. Isaac/runtime check:
   `scripts/smoke_isaac.py`
2. Minimal scene baseline:
   `scripts/minimal_scene_baseline.py`
3. Walker S2 load check:
   `scripts/load_walker_s2.py`
4. Basic arm controllability:
   `scripts/control_walker_s2_arms.py`
5. Right-arm joint sign and visual sanity:
   `scripts/right_arm_joint_space_sanity.py`
6. Current strongest manipulation-related motion:
   `scripts/front_seeded_manipulation_motion.py`

Do not use `scripts/grasp_static_object_smoke.py` as proof of object grasping.
It is a fixed-pose wrist and gripper experiment only.

## Execution Expectations

- Run Isaac Sim scripts on Linux with Isaac Sim's Python environment.
- Keep simulator installation and official assets outside the repo.
- Provide Walker S2 via `--robot-usd` or `WALKER_S2_USD`.
- Read paths from env vars: `HRC_ROOT`, `HRC_REPO`, `DATA_ROOT`, `CKPT_ROOT`,
  `OUTPUT_ROOT`, `LOG_ROOT`.
- Write logs under `LOG_ROOT`.
- Write metrics under `OUTPUT_ROOT/metrics` when metrics are produced.
- Do not commit datasets, checkpoints, videos, replays, outputs, logs, caches, or
  simulator assets.
- Treat Mac/lightweight checks as development checks only. Linux runtime logs are
  the source of truth.

Common CLI style already used:
- `--robot-usd` for Walker S2 asset path override.
- `--prim-path` for robot stage prim path.
- `--init-steps`, `--control-steps`, and `--settle-steps` for deterministic timing.
- `--gui` or `--no-headless` for visible inspection.
- `--hold-open` for manual GUI inspection after a run.

## Result Reporting Standard

Each meaningful diagnostic or baseline script should make the following obvious
from stdout and its log file:

- what was tested
- which asset or robot path was used
- which articulation, DOFs, or scene prims were selected
- what passed
- what failed
- what remains an assumption or unverified

Current logs already cover most of this for environment, robot load, arm control,
and front-seeded motion. The main missing result standard is object-centric
success reporting for manipulation and task scripts.

## Remaining Gaps

- No dynamic contact smoke test.
- No object-centric success metric.
- No Task 1 static sorting baseline.
- No official table/bin/workpiece scene loading.
- No perception or A/B label classification.
- No official baseline repo smoke integration.
- No dataset inspection.
- No evaluator-compatible entrypoint.
- No Task 2 conveyor, Task 3 insertion, or Task 4 carton-closing baseline.

## Exact Next Step

Add one minimal Task 1 dynamic-contact smoke test:

- one simple dynamic workpiece
- one simple tabletop
- two simple bin regions
- known label A/B
- one scripted right-arm pickup/drop attempt
- pass/fail based on the workpiece final pose inside the expected bin

This is the smallest next step that tests the currently unverified assumption
that the robot can physically move an object, not just move its wrist.
