Project: HRC 2026 Robotics Competition

## Official Resources
Official external competition resources are tracked in `docs/resources.md`.
Use that file to distinguish baseline code, simulation assets, dataset resources, and the Walker S2 robot model source.

## Current Phase

- Current active task phase: Phase 1 - Task 1 synchronized RGB-D and simulator-truth data collection
- Current status: user-authorized reset on 2026-04-21 to restore a competition-oriented data path before more grasp tuning
- Phase 1 focus: collect minimal Task 1 camera samples and table-frame object labels while preserving the deterministic manipulation backend
- Phase 2 starts after the collector is run on Linux and the saved sample structure is validated
- Phase 4 remains deferred until the minimal Thinker label contract is proven against collected samples

Older Phase 2/Phase 3 manipulation experiments remain useful history, but the
active implementation contract is now `CURRENT_PLAN.md`.

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
- Goal: make official Task 1 camera observations and Thinker-shaped table-frame labels reproducible
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
