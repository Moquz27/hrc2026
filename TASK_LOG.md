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
- Added GUI inspection flags to minimal scene baseline: --gui and --hold-open
- Fixed minimal scene baseline argument handling so Isaac Kit does not consume script-only flags such as --frames
- Removed experimental material binding from minimal scene baseline; kept scene to one static cube plus light for reproducibility
- Test result: restored minimal scene baseline passed on Linux with frames=3 and regenerated LOG_ROOT/minimal_scene_baseline.log plus OUTPUT_ROOT/metrics/minimal_scene_baseline.json

## 2026-04-14
- Started Phase 2 robot integration baseline with scripts/load_walker_s2.py
- Purpose: launch Isaac Sim, load Walker S2 from a configurable USD path outside the repo, inspect articulation and joints, and write LOG_ROOT/walker_s2_load_ok.txt
- Constraints: no task logic, robot control, object manipulation, perception, dataset use, or learning code
- Test result: lightweight Python compile passed; Isaac runtime validation still must be run on Linux with the Walker S2 USD asset available
- Next step: place the Walker S2 model under runtime assets, set WALKER_S2_USD or pass --robot-usd, then run the load inspection script with Isaac Sim's python.sh
- Added early Git LFS pointer detection to scripts/load_walker_s2.py so placeholder USD files fail before Isaac launch
- Runtime attempt: s2_v1.usd failed because the file is a Git LFS pointer, not a downloaded USD payload
- Runtime attempt: SubUSDs/s2_v1_physics.usd failed because the file is also a Git LFS pointer, not a downloaded USD payload
- Blocker: git-lfs is not installed on the Linux runtime, so the Walker S2 asset repository has not fetched the real USD payloads
- Phase 2 status remains in progress; next step is to install/enable git-lfs outside the code repo, pull the Walker S2 LFS assets under HRC_ROOT/assets, then rerun the same two candidate load commands
- After git-lfs install and asset pull, verified s2_v1.usd and SubUSDs/s2_v1_physics.usd are real binary USDC payloads rather than Git LFS pointers
- Runtime result: scripts/load_walker_s2.py passed with s2_v1.usd using init_steps=120
- Phase 2 result: robot loaded without crash, articulation root detected at /World/WalkerS2/base_link, joint_count=42, joint names printed, and LOG_ROOT/walker_s2_load_ok.txt written
- Joint state read warning remains: dynamic_control did not find articulation at /World/WalkerS2 and articulation wrapper fallback failed; this is non-blocking because joint state printing was optional
- Correct Walker S2 integration entrypoint for this phase: HRC_ROOT/assets/WalkerS2-Model-Challenge/WalkerS2-Model-Challenge/s2_v1.usd
- Phase 2 status: PASS for robot load and articulation inspection baseline; next step is to commit this baseline before starting any robot control or task logic
- Added docs/roadmap.md to define Phase 0 through Phase 4 with Phase 3 as the next active integration and validation phase
- Updated PROJECT_CONTEXT.md so Current Phase points to Phase 3 and Phase 4 remains gated on Phase 3 exit criteria
- Documentation-only update; no runtime code, training code, or scripts changed
