# Task Log

## 2026-04-13
- Created initial repo structure
- Created folders: src, scripts, configs, tests, docs
- Added AGENTS.md, PROJECT_CONTEXT.md, TASK_LOG.md
- Next step: create smoke test and basic requirements
- Added tracked placeholders for empty project directories
- Tightened repository hygiene ignores and simplified direct dependencies
- Added docs/context_full.md placeholder and updated agent context rules
- Created runtime asset directories under ~/hrc-runtime for data, checkpoints, outputs, and logs
- Corrected Linux path plan so HRC_REPO points to ~/hrc2026/repo while ~/hrc-runtime stores only runtime assets
- Added scripts/smoke_isaac.py as a minimal Isaac Sim headless smoke test that validates env vars, steps an empty stage, and writes LOG_ROOT/isaac_smoke_ok.txt
- Cleaned ~/.bashrc so a single HRC environment block is defined before the interactive-shell guard
- Verified HRC_REPO resolves to ~/hrc2026/repo and runtime assets remain under ~/hrc-runtime
- Test result: preflight passed and wrote ~/hrc-runtime/logs/isaac_smoke_preflight_ok.txt
- Test result: Isaac smoke passed with steps=20 and wrote ~/hrc-runtime/logs/isaac_smoke_ok.txt
- Infrastructure phase complete; next step is to commit this note, then begin the minimal simulation baseline in a separate change
- Started minimal simulation baseline phase with scripts/minimal_scene_baseline.py
- Purpose: launch Isaac Sim, create one static cube scene, step fixed frames, write one log and one metrics JSON outside the repo
- Constraints: no robot model, dataset, task logic, perception, or learning code
- Test result: Isaac minimal scene baseline passed with frames=60 and scene_prim=/World/StaticCube
- Runtime artifacts: LOG_ROOT/minimal_scene_baseline.log and OUTPUT_ROOT/metrics/minimal_scene_baseline.json
