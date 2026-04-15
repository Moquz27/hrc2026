Project: HRC 2026 Robotics Competition

## Official Resources
Official external competition resources are tracked in `docs/resources.md`.
Use that file to distinguish baseline code, simulation assets, dataset resources, and the Walker S2 robot model source.

## Current Phase

- Current major phase: Phase 3 - Competition Stack Integration & Validation
- Current status: next active phase after Phase 2 passed
- Phase 3 focus: make the full competition stack runnable and debuggable before algorithm optimization
- Phase 4 starts only after all Phase 3 exit criteria pass

Phase 0, Phase 1, and Phase 2 are complete. Phase 3 is integration and validation, not serious optimization. Phase 4 is reserved for algorithm and ML optimization after the stack is stable.

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
- Phase: Competition stack integration and validation
- Goal: make official resources, scenes, baseline, dataset access, and robot motion debuggable
- No serious algorithm or ML optimization until Phase 3 passes
- Task 1 continuous-motion baseline now has per-object diagnostics and one-knob tuning support in `scripts/task1_smooth_autoseed_multi_object_baseline.py`.
- Latest controlled Linux runtime sweep for seed=1 target-index=2 showed grasp-depth offsets 0.0, -0.005, and -0.010 all failed before grasp at `pre_grasp_unreachable`; next single tuning family should be approach/soft waypoint reachability, not deeper grasp or carry/place tuning.

Assumptions:
- Dataset not fully available yet
- Simulator not fully set up yet
- Focus is on reproducible pipeline, not performance

Reference:
- Full detailed context is in docs/context_full.md
- Current baseline maturity and script classification are in docs/baseline_status.md
