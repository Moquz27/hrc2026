# HRC 2026 Walker S2 Simulation Baseline

This repository is the source-code workspace for the HRC 2026 Isaac Sim
qualification project. The current project state is baseline-building and
integration validation, not final task solving.

Read first:
- `PROJECT_CONTEXT.md`
- `docs/context_full.md`
- `docs/baseline_status.md`

Current quick checks:
- Isaac/runtime check: `scripts/smoke_isaac.py`
- Minimal scene baseline: `scripts/minimal_scene_baseline.py`
- Walker S2 load check: `scripts/load_walker_s2.py`
- Basic arm motion check: `scripts/control_walker_s2_arms.py`
- Right-arm joint-space sanity: `scripts/right_arm_joint_space_sanity.py`
- Current strongest manipulation-related motion: `scripts/front_seeded_manipulation_motion.py`

Important limitation:
- No Task 1/2/3/4 task baseline is implemented yet.
- Current manipulation-related scripts prove motion sanity only.
- Dynamic object grasp, carry, release, and sorting are not yet verified.

Runtime expectations:
- Run Isaac Sim scripts on Linux with Isaac Sim's Python environment.
- Keep simulator assets, datasets, logs, outputs, and checkpoints outside the repo.
- Use env vars for runtime paths: `HRC_ROOT`, `HRC_REPO`, `DATA_ROOT`,
  `CKPT_ROOT`, `OUTPUT_ROOT`, `LOG_ROOT`.
- Logs go under `LOG_ROOT`; metrics go under `OUTPUT_ROOT/metrics` when produced.
