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
- Started Phase 3.1 minimal controllability baseline with scripts/control_walker_s2_arms.py
- Purpose: load Walker S2, acquire the detected articulation root, read DOF observations, send small arm position targets, and log observed motion
- Constraints: no task logic, object manipulation, perception, dataset use, training, or machine learning
- Test result: Isaac runtime smoke passed with control_steps=40, articulation_path=/World/WalkerS2/base_link, selected_arm_dof_count=8, and max_abs_delta=0.05010388046503067
- Runtime artifact: LOG_ROOT/walker_s2_arm_control_smoke.log
- Added scripts/move_walker_s2_end_effector.py for a minimal right-arm Cartesian target smoke test
- Purpose: identify the right-arm end-effector body, run simple damped least-squares IK over right-arm DOFs, and log target vs actual position
- Constraints: no task logic, object manipulation, perception, dataset use, training, or machine learning
- Test result: Isaac runtime smoke passed with end_effector_path=/World/WalkerS2/R_wrist_roll_link, target_position=[1.2711199712753296, -0.2775531601905823, 0.8609833312034607], actual_position=[1.275246500968933, -0.27807775139808655, 0.8502683043479919], and position_error=0.01149413953980893
- Runtime artifact: LOG_ROOT/walker_s2_end_effector_target.log
- Added scripts/grasp_static_object_smoke.py for a minimal fixed-target grasp primitive smoke test
- Purpose: identify right gripper DOFs, verify open/close motion, move to pre-grasp/grasp/lift poses around one fixed cube target, and log the primitive sequence
- Assumption: the cube is a static pose target; lift success measures end-effector vertical lift after gripper close, not physical object transport
- Test result: Isaac runtime smoke passed with target_pose=[1.2711199712753296, -0.2775531601905823, 0.8209833312034607], actual_grasp_pose=[1.2714508771896362, -0.27760079503059387, 0.8601263761520386], actual_end_effector_pose=[1.2779146432876587, -0.2779400646686554, 0.8996994495391846], gripper_dof_names=['R_finger1_joint', 'R_finger2_joint'], gripper_verified=true, lift_delta=0.039573073387145996, and lift_succeeded=true
- Runtime artifact: LOG_ROOT/walker_s2_static_grasp_smoke.log
- Debugged visually unsafe Cartesian reaching behavior in scripts/move_walker_s2_end_effector.py and scripts/grasp_static_object_smoke.py
- Root cause: the previous smoke used position-only wrist-link IK with one-shot targets, no front-workspace clamp, no posture bias, and no early stop/hold, so visually bad equivalent arm postures and small target chasing jitter were possible
- Added front-workspace target clamping, small IK steps, posture bias, early stop/hold behavior, debug target/end-effector markers, and per-step commanded-vs-observed right-arm joint logging
- Updated the static grasp smoke sequence to run approach_from_front -> move_down_grasp -> close gripper -> lift_up -> move_down_release -> open gripper
- End-effector assumption: R_wrist_roll_link remains the selected frame because no palm/hand rigid body was identified by the current body-name filter
- Test result: lightweight Python compile passed for the Cartesian target and static grasp scripts
- Runtime result: GUI static grasp sanity run passed with --ik-steps 16, lift_delta=0.0317690372467041, release_pose reached within tolerance, and LOG_ROOT/walker_s2_static_grasp_smoke.log recorded the new IK trace
- Current GUI result still failed visual motion sanity because the right arm could still appear backward; paused Cartesian grasp proof and switched to direct joint-space arm debugging
- Added scripts/right_arm_joint_space_sanity.py for a minimal right-arm joint-space visual demo with no Cartesian IK, no task assets, no dataset, and no ML
- Script logs right-arm DOF indices/names, right-gripper DOF indices/names, commanded targets, observed joint values, and optional one-joint-at-a-time diagnostics
- Diagnostic result: R_shoulder_roll_joint positive moved the wrist backward in +x-forward convention, R_shoulder_roll_joint negative moved forward, R_elbow_roll_joint positive moved forward, and R_elbow_roll_joint negative moved backward
- Updated the joint-space demo preset to use R_shoulder_roll_joint=-0.25 for front pose, raise by moving R_shoulder_roll_joint to -0.33, then lower back to -0.25; wrist and shoulder pitch are kept near zero
- Test result: lightweight Python compile passed for scripts/right_arm_joint_space_sanity.py
- Runtime result: diagnostic headless run passed and wrote LOG_ROOT/walker_s2_right_arm_joint_space_sanity.log
- Runtime result: GUI joint-space demo passed and is held open with sequence front_pose -> raise_slightly -> lower_slightly -> gripper_close -> gripper_open
- Added scripts/front_seeded_manipulation_motion.py for the next minimal front-seeded phased manipulation motion
- Purpose: start every cycle from the validated right-arm front pose, then run move_to_front_pose -> move_slightly_above_front_target -> move_slightly_downward -> gripper_close -> lift_slightly_upward -> gripper_open using explicit joint-space targets
- Root cause carried forward: the unsafe visual motion came from right-arm joint sign/posture assumptions and Cartesian IK freedom, so this script avoids general IK and keeps R_shoulder_roll_joint negative plus R_elbow_roll_joint slightly positive for front-facing motion
- Motion tuning: control_steps reduced to 72 and settle_steps reduced to 12, with only small shoulder/elbow target deltas, to make the motion slightly faster and more visible without introducing large overshoot
- Runtime result: headless 5-cycle run passed and wrote LOG_ROOT/walker_s2_front_seeded_manipulation_motion.log
- Stability result: all 5 cycles reported cycle_passed_motion_sanity=true, front_facing_ok=true, overhead_ok=true, backward_deviation=0.0, overhead_deviation=0.0, and phase repeatability drift stayed near 1e-6 meters
- Runtime result: GUI run passed the same 5-cycle sequence, updated LOG_ROOT/walker_s2_front_seeded_manipulation_motion.log, and is held open for visual inspection
- Performed minimal baseline standardization pass
- Added docs/baseline_status.md to classify scripts as diagnostic, baseline, or experiment and to state current baseline maturity honestly
- Updated README.md to point new agents at the current checks, strongest manipulation-related motion script, and unimplemented task gaps
- Updated PROJECT_CONTEXT.md with the baseline status reference
- Tightened docstrings for scripts/grasp_static_object_smoke.py and scripts/front_seeded_manipulation_motion.py so they do not imply verified object manipulation
- No runtime logic was refactored; current working behavior is preserved
- Test result: lightweight compile passed for scripts/grasp_static_object_smoke.py and scripts/front_seeded_manipulation_motion.py
- Next step: add a minimal Task 1 dynamic-contact smoke test with object-centric pass/fail based on final workpiece pose
- Verified official external resources setup
- Initial resource state: Walker S2 model repo already existed under HRC_ROOT/assets and was verified first; baseline repo, official assets, and official dataset were missing
- Downloaded missing resources outside the code repo: HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline, HRC_ROOT/assets/challenge2026_assets, and DATA_ROOT/challenge2026_dataset
- Added scripts/inspect_official_assets.py for filesystem inventory, size checks, key-file listing, and Git LFS pointer detection
- Added scripts/load_official_scene_smoke.py for Isaac validation of one official USD scene/object plus Walker S2 in the same stage
- Filesystem validation result: baseline repo 2.0M, Walker S2 557M, official assets 2.5G, official dataset 1.7G, and no Git LFS pointer files detected
- Isaac validation result: official table USD loaded with Walker S2; scene_prim_count=11, robot_prim_count=260, articulation_root=/World/WalkerS2/base_link, joint_count=42, and LOG_ROOT/official_scene_smoke.log reports status=official_scene_smoke_ok
- Known Isaac warnings: Walker S2 emits non-fatal material binding scope warnings, one non-existent collision mesh path warning, and one corrupted normal primvar warning; no missing-resource or broken-payload failure was observed
- Added docs/official_resources_inventory.md with resource paths, sizes, candidate task asset entrypoints, validation results, and readiness status
- Current readiness: official resources are present and usable for Phase 3 inspection; baseline repo execution, official task reset/scoring mapping, dataset schema inspection, and full task-scene composition remain unverified
- Next step: inspect the official baseline repo task configs and SceneBuilder to map asset-root handling, reset flow, action format, and intended scene composition before implementing task logic
- Added scripts/validate_task1_object_assets.py for object-level Task 1 asset validation with official table visual, simple table-top collider, one Part A, and one Part B; no robot, perception, sorting, or task logic
- Validation result: root visual assets resources/Task1_PartA.usd and resources/Part_B.usd have reasonable scale but no detected CollisionAPI/RigidBodyAPI, so they did not fall under gravity or rest on the table; treat these as not physics-ready for manipulation
- Validation result: collected variants resources/Collected_Task1_PartA_red/Task1_PartA.usd and resources/Collected_Part_B_red/Part_B.usd passed physics validation
- Part A collected result: size approximately 0.044 x 0.0225 x 0.036 m, collision_count=12, rigid_body_count=1, fell_under_gravity=true, rested_on_table=true, stable=true
- Part B collected result: size approximately 0.0348 x 0.0646 x 0.0464 m, collision_count=34, rigid_body_count=1, fell_under_gravity=true, rested_on_table=true, stable=true
- Table note: official Collected_table_v2/table_v2.usd exposes collision APIs, but this diagnostic pass/fail uses a simple tabletop collider so object physics is isolated from table-asset assumptions
- Runtime artifact: LOG_ROOT/task1_object_asset_validation.log
- Current object readiness: use collected Task 1 Part A/B variants for manipulation experiments; do not use root Task1_PartA.usd or Part_B.usd as physics-ready workpieces without adding/confirming collision and rigid body setup
- Next step: inspect official baseline SceneBuilder/configs to confirm which workpiece USD variants it uses and how it instantiates physics before writing Task 1 manipulation logic
- Inspected official baseline Task 1 asset usage in HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/source/SceneBuilder.py and Ubtech_sim/config/Part_Sorting.yaml
- Confirmed Task 1 config uses collected physics-ready asset pools, not root visual USDs: Part A uses Collected_Task1_PartA_ori_color/Task1_PartA.usd or Collected_Task1_PartA_red/Task1_PartA.usd; Part B uses Collected_Part_B_blue/Part_B.usd or Collected_Part_B_ori_color/Part_B.usd
- Confirmed Task 1 table config uses Collected_table_v2/table_v2.usd and the bin/box config uses Box_blank/box_60_40_23_cut_0.usd with lock_boxes=true
- Confirmed SceneBuilder.build_parts creates Task 1 objects via rep.create.from_usd, then applies rep.physics.rigid_body(overwrite=True), rep.physics.mass, random rotation, and rep.randomizer.scatter_2d on the hidden scatter plane
- Confirmed reset flow deletes/recreates Task 1 parts from the same asset pools, applies UsdPhysics.RigidBodyAPI and MassAPI to the root prim, randomizes pose in _create_parts_at_paths, resets boxes, and leaves scatter_after_reset as the post-world.reset pose randomization path
- Local mismatch found: the baseline repo assets submodule directory is empty/missing resources, so Part_Sorting.yaml resolves to missing files under baseline/assets/resources unless the official assets are mounted/symlinked there or root_path is changed; the verified assets currently live under HRC_ROOT/assets/challenge2026_assets/resources
- Validation result for configured pools from the verified assets tree: Part A original color, Part A red, Part B blue, and Part B original color all passed object-level physics checks with collision and rigid body present
- Current Task 1 asset readiness: safe to build manipulation experiments on collected variants from the verified official assets tree, but not safe to run the unmodified baseline checkout until its assets/resources path is fixed
- Next exact step: add a minimal baseline-scene smoke script that loads Part_Sorting.yaml through the official config loader with an explicit root_path override to HRC_ROOT/assets/challenge2026_assets/resources, builds table/box/parts only, steps physics, and logs actual spawned prim paths plus rigid-body detection
- Added scripts/validate_task1_scene_builder_scene.py to validate the composed Task 1 runtime scene produced by the official SceneBuilder with root_path overridden in memory to HRC_ROOT/assets/challenge2026_assets/resources
- Runtime result: SceneBuilder spawned 4 Task 1 parts at /Replicator/Ref_Xform_01 through /Replicator/Ref_Xform_04; by SceneBuilder order the first two are Part A and the last two are Part B
- Part physics result: all 4 spawned parts had CollisionAPI and RigidBodyAPI, mass approximately 0.2 kg, no rigid-body schema issues, fell under gravity after lift, rested on the table, and had near-zero final jitter
- Table result: SceneBuilder table at /Replicator/Ref_Xform had collision_count=2 and no rigid-body schema issues; parts rested on the table surface instead of falling through
- Box/bin result: /Root/Box had collision_count=2 but rigid_body_count=19 with invalid schema issues, including RigidBodyAPI on non-xformable material/shader prims and nested rigid bodies under /Root/Box
- Current composed Task 1 scene readiness: NOT READY for manipulation because the bin/box physics hierarchy is invalid, even though the SceneBuilder-spawned parts and table are usable
- Runtime artifact: LOG_ROOT/task1_scene_builder_validation.log
- Next exact step: inspect Box_blank/box_60_40_23_cut_0.usd and SceneBuilder._lock_box_positions to create a minimal diagnostic-only box collision fix or replacement strategy before any pick/place manipulation logic
- Added scripts/diagnose_task1_bin_physics.py for focused Task 1 bin diagnostics
- Standalone box asset result: Box_blank/box_60_40_23_cut_0.usd loaded cleanly enough for inspection, with collision_count=2, rigid_body_count=1 on /World/StandaloneBox/Group, and no rigid-body schema issues detected
- SceneBuilder box result: build_box + _lock_box_positions changed the composed /Root/Box into an invalid hierarchy with collision_count=2, rigid_body_count=19, RigidBodyAPI on non-xformable material/shader prims, and nested rigid bodies under /Root/Box
- Root cause classification: code, specifically SceneBuilder._lock_box_positions applying or preserving RigidBodyAPI too broadly under /Root/Box
- Chosen unblock strategy: diagnostic_replacement_static_bin_collider, using the official box as visual geometry only and adding simple static floor/wall colliders for manipulation validation
- Validation result: passed after stripping physics from the diagnostic visual box; official Part A fell under gravity, rested on the static bin floor, had jitter_last_30_steps_m near 5.7e-17, and did not explode
- Runtime artifact: LOG_ROOT/task1_bin_physics_diagnostic.log
- Remaining unverified: robot placement into the diagnostic bin, official scoring acceptance of diagnostic collider prims, and whether a targeted SceneBuilder._lock_box_positions fix is preferable after baseline ownership is decided
- Next exact step: build a minimal Task 1 pick-place baseline against the SceneBuilder table/parts plus the diagnostic static bin collider, with object final-pose scoring and no perception
- Added scripts/inspect_task1_pick_place_gui.py as a GUI-first Task 1 visual inspection script
- Confirmed the script uses the official Task 1 YAML at Ubtech_sim/config/Part_Sorting.yaml; it does not use task1.yaml
- Scene construction: loads Part_Sorting.yaml through the official config loader, overrides cfg["root_path"] in memory to HRC_ROOT/assets/challenge2026_assets/resources, builds only SceneBuilder table + Task 1 parts, loads Walker S2, and adds the diagnostic static bin collider instead of SceneBuilder.build_box
- Visual flow: hold_initial_view -> move_to_front_pose -> move_to_pre_grasp -> pause -> descend -> pause -> close_gripper -> pause -> lift -> pause -> move_to_bin -> pause -> open_gripper -> pause -> settle
- Debug aids: target part marker, pre-grasp marker, bin center marker, bin bbox corner markers, and final end-effector marker
- Logging: writes LOG_ROOT/task1_pick_place_gui_inspection.log with selected target prim, target bbox/pose, inferred category if available, bin bounds, robot/articulation info, phase log, gripper observations, and final object pose
- Important limitation: this is visual inspection only, not a scored baseline and not proof of grasp, transport, or successful sorting
- Runtime dry check: headless short run completed and wrote status=inspection_complete with selected_target_part_prim=/Replicator/Ref_Xform_01
- Runtime warnings observed during dry check: Isaac/PhysX still emits Walker S2 joint/property warnings in the composed scene; the script avoids applying the Part_Sorting.yaml robot pose by default because that path produced stronger invalid-transform warnings during testing
- User GUI check should verify robot/table/bin relative placement, visible target markers, arm phase direction, gripper open/close visibility, absence of collisions/explosions, and whether the destination bin position is plausible for Task 1
- Next exact step: manually run the GUI inspection with --gui --hold-open and tune only the minimal robot/bin/phase pose parameters needed before attempting object-contact pick/place
- Added scripts/task1_single_target_random_scene_baseline.py for the next minimal Task 1 manipulation baseline on the official randomized SceneBuilder scene
- Scene construction: loads Ubtech_sim/config/Part_Sorting.yaml through the official config loader, overrides cfg["root_path"] in memory to HRC_ROOT/assets/challenge2026_assets/resources, builds SceneBuilder table + randomized Task 1 parts, loads Walker S2, and uses the validated diagnostic static bin collider instead of the broken official composed box physics path
- Motion policy: strictly fixed joint-space phased right-arm targets for move_to_front_pose, move_to_pre_grasp, descend, grasp_validation/lift, move_to_bin, release, and settle; no generic Cartesian IK search is used
- Task-space targets are explicit logging/reachability gates only: pre-grasp x/y is the selected object bbox center with z at object top plus fixed clearance and fixed downward orientation; bin drop x/y is the diagnostic bin center with z at bin top plus safe drop height
- Object-centric result rules: pass only after settle if the target was lifted with the gripper, transported at least the minimum distance toward the bin, released inside the bin volume, and stable after settle; arm motion alone is never treated as task success
- Logging: every run writes LOG_ROOT/task1_single_target_random_scene_baseline.log plus a timestamped per-run log with run metadata, selected target, bin bounds, phase trace, object trace, and pass/fail flags
- Test result: lightweight Python compile passed for scripts/task1_single_target_random_scene_baseline.py
- Local runtime check: a short invocation without WALKER_S2_USD failed honestly before Isaac launch with failure_reason=runtime_error and still wrote durable logs
- Local runtime artifacts: LOG_ROOT/task1_single_target_random_scene_baseline.log and LOG_ROOT/task1_single_target_random_scene_baseline_20260414T122950Z_final_no_isaac_check.log
- Runtime limitation in this environment: Isaac Sim Python modules are not importable and WALKER_S2_USD is unset, so a real headless or GUI manipulation smoke was not run here
- Next exact step: on the Linux Isaac runtime with WALKER_S2_USD set, run python scripts/task1_single_target_random_scene_baseline.py --target-index 0 --init-steps 120 --phase-steps 120 --pause-steps 30 --settle-steps 240, then inspect the log for whether the object was truly lifted and transported
- Revised scripts/task1_single_target_random_scene_baseline.py to use a Task 1-specific front-ready manipulation posture family instead of the previous hip-side/front-seeded reaching assumption
- New strategy is strictly joint-space phased and front-workspace gated: move_to_front_ready_pose, move_to_pre_grasp_front, descend_front, close_gripper, grasp_validation, lift_front, move_to_bin_front, release, settle
- Added conservative front tabletop workspace checks, explicit object-bbox-center pre-grasp target with fixed downward orientation, explicit bin-center drop target, and additional front safety logging for observed end-effector poses
- Added Task 1 front-ready joint targets for the active right arm and a folded left-arm standby posture; no generic Cartesian IK, perception, A/B classification, or multi-object logic was added
- Why the old posture was wrong: the previous fixed pre-grasp was approximately 0.998 m from the explicit object-centered pre-grasp pose and came from a hip/side-origin posture family rather than a forward tabletop-ready stance
- Lightweight test result: python -m py_compile scripts/task1_single_target_random_scene_baseline.py passed
- Runtime test result: Isaac headless smoke with --target-index 0 built the official randomized Part_Sorting.yaml SceneBuilder table/parts, selected /Replicator/Ref_Xform_01, and wrote durable logs, but failed honestly with failure_reason=front_pose_failed before grasp
- Runtime detail: the explicit target/pre-grasp passed the front-workspace gate, but the commanded front-ready joint target did not fully reach because R_shoulder_pitch_joint and R_shoulder_roll_joint saturated near their limits; observed end-effector pose was [1.342050313949585, 0.000125114805996418, 0.9516192078590393], below table_top_z + clearance
- Runtime artifact: LOG_ROOT/task1_single_target_random_scene_baseline_20260414T124727Z_front_ready_config_pose_strict_xform.log and rolling LOG_ROOT/task1_single_target_random_scene_baseline.log
- Current truth: no object was lifted or transported; this remains a scene-build plus front-ready posture smoke, not a successful Task 1 pick-place cycle
- Next exact step: use GUI inspection to tune a reachable front-ready joint target within Walker S2 shoulder limits that places the right end effector above the tabletop before attempting pre-grasp/contact again
- Re-scanned organizer runtime resources for Task 1 in the Linux runtime and reconciled scripts/task1_single_target_random_scene_baseline.py back to official runtime facts
- Confirmed official Task 1 config is HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline/Ubtech_sim/config/Part_Sorting.yaml; Part_Sorting.yaml defines robot base placement only, while the official arm startup posture comes from IsaacSimRobotInterface._joint_value_map / RobotArticulation._joint_value_map
- Confirmed the official SceneBuilder path still produces valid randomized table + Task 1 parts but the composed official box path remains physically invalid; preserved the existing diagnostic static bin workaround instead of changing YAML scene config
- Updated the Task 1 one-object script to build the robot through official SceneBuilder.build_robot, apply the official startup joint map, use official same-sign gripper widths, keep official part randomization, and add a configurable descend clearance for contact experiments
- Iteration results: official startup and target-conditioned reach now pass for target-index 1; wrist-frame grasp failed with object_not_lifted; requested R_finger1_link as the active control body improved contact; lower R_finger1_link descend with --descend-clearance -0.025 achieved grasp_validation object_lifted=true but dropped during the following lift
- Latest deeper insertion run with --descend-clearance -0.04 --min-ee-table-clearance 0.005 regressed to object_not_lifted, so the best observed one-object state is transient lift, not retained grasp or bin placement
- Current blocker: position-only right finger closure can make transient contact/lift but does not retain the object through lift/transport; official baseline gripper handling uses force/effort during gripping, which has not yet been ported into this direct dynamic-control script
- Runtime artifacts: LOG_ROOT/task1_single_target_random_scene_baseline_20260414T165225Z_target1_finger1_lower_descend_run10.log showed object_lifted=true then dropped_during_lift; LOG_ROOT/task1_single_target_random_scene_baseline_20260414T165539Z_target1_finger1_deeper_grasp_run11.log showed object_not_lifted
- Next exact step: port the official gripper force-control behavior narrowly for the right gripper during close/validation/lift, then rerun the run10 geometry before tuning transport or classification
- Implemented minimal right-gripper effort hold in scripts/task1_single_target_random_scene_baseline.py without changing scene loading, target selection, pre-grasp/descend geometry, IK, perception, or phase structure
- Gripper control change: close_gripper still sends the official same-sign close position target, then applies sustained dynamic_control.set_dof_effort to the selected right finger DOFs during close_gripper, grasp_validation, and lift_front; release clears the effort back to 0.0
- Confirmed selected right gripper DOFs are indices [32, 33], names R_finger1_joint and R_finger2_joint
- Run 12 with conservative effort 35.0 preserved object_lifted=true but still failed with dropped_during_lift; lift delta was approximately [0.00579, -0.00360, -0.09767] m and object_retained_after_lift=false
- Run 13 with official baseline gripper effort 100.0 passed the one-object target-index 1 cycle using the run-10 geometry: object_lifted=true, object_retained_after_lift=true, object_transported=true, final_inside_bin=true, object_stable=true
- Run 13 validation/lift details: validation object delta was approximately [0.02077, 0.01216, 0.08879] m; lift delta was approximately [0.00013, 0.00021, -0.00027] m; ee_to_object_after_lift was approximately 0.04496 m
- Runtime artifact: LOG_ROOT/task1_single_target_random_scene_baseline_20260414T171839Z_target1_finger1_effort100_run13.log
- Repeat run 14 without an explicit --gripper-hold-effort override also passed with the new default 100.0 effort: object_lifted=true, object_retained_after_lift=true, object_transported=true, final_inside_bin=true, object_stable=true
- Runtime artifact: LOG_ROOT/task1_single_target_random_scene_baseline_20260414T172325Z_target1_finger1_default_effort100_run14.log
- Next exact step: begin the next narrow robustness check by repeating the same one-object target-index 1 baseline across a small seed/target sweep before changing transport, release, classification, or perception
- Created scripts/task1_smooth_autoseed_multi_object_baseline.py as a separate copied variant of scripts/task1_single_target_random_scene_baseline.py; the validated one-object script was not modified for this smooth/autoseed/multi-object pass
- Smooth variant changes are intentionally narrow: smooth-motion mode defaults on, non-contact phase pauses and IK end holds are reduced to 1 step, while close_gripper, grasp_validation, release, and settle keep physical dwell behavior; validated grasp geometry, descend clearance, and gripper force-hold behavior are preserved
- Auto-seed behavior: --seed now defaults to unset in the smooth variant; if omitted, a runtime 32-bit seed is generated and applied to Python random, NumPy, and Replicator, while an explicit --seed is still used exactly and logged
- Auto-seed validation: two scene-only no-seed runs produced different seeds and different initial object centers; artifacts are LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260414T175258Z_smooth_scene_only_autoseed_a.log and LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260414T175318Z_smooth_scene_only_autoseed_b.log
- One-object validation in the smooth script passed for the previously validated target-index 1 / seed 1 run geometry: object_lifted=true, object_retained_after_lift=true, object_transported=true, final_inside_bin=true, object_stable=true
- One-object smooth artifact: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260414T175021Z_smooth_single_target1_seed1_validation.log
- Multi-object loop validation with --seed 1 attempted all 4 spawned objects in the same official randomized scene; continuation rule is continue_after_manipulation_failure_unless_scene_or_robot_safety_fails
- Multi-object result: 2/4 succeeded, 2/4 failed, no hard stop; target 0 failed at move_to_bin_front with dropped_during_transport, target 1 succeeded, target 2 failed at grasp_validation with object_not_lifted, target 3 succeeded
- Multi-object artifact: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260414T175348Z_smooth_multi_seed1_attempt.log
- Current truth: the smooth variant preserves the validated one-object path and can continue through all spawned objects, but the all-object baseline is not yet robust
- Next exact step: inspect failed target 0 transport retention and failed target 2 grasp contact from the multi-object log, then make one narrow robustness correction at a time in the smooth variant only
- Refactored scripts/task1_smooth_autoseed_multi_object_baseline.py toward a true continuous-motion Task 1 baseline while preserving official SceneBuilder scene construction, auto-seed behavior, diagnostic static bin, right-gripper effort hold, logging, and the existing multi-object attempt loop
- Removed the active per-object stop-and-go chain where each object ran separate move_to_pre_grasp_front, descend_front, close_gripper, grasp_validation/lift_front, move_to_bin_front, release phases with blocking waits between them
- Introduced a compact internal MotionSegment plan and continuous executor for each object: current/home -> continuous_pregrasp -> continuous_grasp_depth -> continuous_lift_clearance -> continuous_prebin -> continuous_place_depth -> continuous_retreat
- Soft transit waypoints now use relaxed tolerances, blend-radius metadata, and zero non-contact hold in smooth mode; gripper close/open are event markers on grasp_depth/place_depth rather than standalone long pause phases
- Conservative behavior intentionally remains near contact: grasp_depth, lift_clearance, and place_depth are hard/slow contact windows with short configurable dwell and sustained gripper effort to protect the currently validated grasp assumptions
- Logging now records per-object cycle_time_steps, micro_stop_frames/micro_stop_samples estimates outside contact windows, continuous plan segment summaries, and per-object success/failure reason in the multi-object metrics
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Linux Isaac runtime manipulation validation was run in this edit pass; next step is to rerun the prior seed 1 target-index 1 validation and then the seed 1 all-object sweep to tune soft tolerance, contact dwell, and place depth if success regresses
- Added a focused observability pass to scripts/task1_smooth_autoseed_multi_object_baseline.py without changing grasp strategy or continuous executor structure
- Runtime logs now tag continuous-cycle diagnostics by approach, grasp_window, post_close_verify, lift, carry, place_window, and retreat; per-object results include selected_object_id, failure_phase, diagnostic_failure_kind, close/open event flags, lift-height threshold status, inferred carry-midpoint retention, prebin retention, pre-place retention, cycle time, and micro-stop counts
- Failure classification is now intended to distinguish miss_grasp, close_but_no_lift, lift_then_slip_during_carry, near_bin_place_failure, and release_timing_failure before tuning motion parameters
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Added controlled tuning support to scripts/task1_smooth_autoseed_multi_object_baseline.py with explicit one-family sweep knobs for grasp depth, contact dwell, carry stabilization, place depth, release timing, and soft tolerance
- First active tuning knob is grasp_depth because the latest logged multi-object run included object_not_lifted/close-but-no-lift style failure; default grasp_depth_offset is a small -0.005 m adjustment on top of descend_clearance, while carry/place/release tuning defaults remain neutral
- Runtime logs now print and record the active tuning knob family plus effective grasp clearance so seed/target sweeps are reversible and comparable
- Ran controlled Isaac runtime sweep for scripts/task1_smooth_autoseed_multi_object_baseline.py with seed=1, target-index=2, varying only --grasp-depth-offset across 0.0, -0.005, and -0.010
- Sweep result: all three runs failed with failure_reason=pre_grasp_unreachable, failure_phase=approach, diagnostic_failure_kind=miss_grasp, close_event_fired=true, lift_height_threshold_achieved=false, and no carry/prebin/pre-place retention fields were reached
- Depth comparison: pregrasp_error was unchanged at approximately 0.304866 m for all offsets; grasp_depth segment error worsened slightly from approximately 0.2062 m at offset 0.0 to 0.2086 m at -0.005 and 0.2111 m at -0.010, so grasp depth is not yet the main lever for this target
- Runtime artifacts: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T012103Z_depth_sweep_seed1_target2_offset_0p000.log, LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T012642Z_depth_sweep_seed1_target2_offset_m0p005.log, and LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T013205Z_depth_sweep_seed1_target2_offset_m0p010.log
- Next tuning recommendation: do not deepen grasp further for this target; next single knob family should be approach/soft waypoint reachability before contact-dwell, because the failure occurs before a valid grasp/lift comparison can be made

## 2026-04-15
- Added pregrasp geometry diagnostics to scripts/task1_smooth_autoseed_multi_object_baseline.py without changing the motion executor or tuning grasp depth, dwell, carry, place, or release behavior
- New diagnostic print/payload fields record object center, bbox top z, pregrasp target, robot base position, forward/lateral offsets, horizontal reach, vertical reach, table clearance, and full base-to-pregrasp distance
- Lightweight test result: python3 -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac runtime was run on Mac; seed=1 target-index=2 numeric geometry still needs a Linux diagnostic run or the referenced LOG_ROOT artifacts
- Added arm-aware greedy pregrasp candidate selection to scripts/task1_smooth_autoseed_multi_object_baseline.py, limited to pregrasp generation/approach selection while preserving the existing continuous descend, grasp, lift, carry, place, and release segment logic
- Active arm is selected from object Y relative to robot base Y, with a one-time fallback to the opposite arm only if all pregrasp candidates miss the pre_grasp_ee_tolerance
- Added --pregrasp-pullback-m and --pregrasp-max-bias-m, deterministic center/pullback/Z-offset candidates capped at 6, Z clamping to workspace limits, and structured per-candidate logs for active_arm, pregrasp_before_bias, pregrasp_after_bias, pullback_applied, base_to_target_xy, error_vector, error_norm, candidate_index, and fallback_triggered
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac runtime was run in this edit pass; next Linux check should rerun seed=1 target-index=2 with default pullback and then with a small --pregrasp-pullback-m sweep if candidate logs still miss tolerance
- Added scripts/watch_task1_gui.sh as a minimal Isaac GUI launcher for the current Task 1 seed=1 target-index=2 inspection case; it does not modify baseline logic and uses ISAAC_SIM_PYTHON or ISAAC_SIM_ROOT instead of hardcoded machine paths
- The helper defaults to --pregrasp-pullback-m 0.02, --grasp-depth-offset -0.020, --gui, and --hold-open so the Linux runtime can visually inspect the current contact/depth behavior; SEED, TARGET_INDEX, PREGRASP_PULLBACK_M, GRASP_DEPTH_OFFSET, and LOG_SUFFIX are environment overrides
- Fixed the Task 1 smooth baseline descend geometry so grasp/contact XY is derived from the selected reachable pregrasp candidate instead of rebuilding from the raw bbox center after candidate selection
- Split pre-contact motion into continuous_grasp_align, a soft XY alignment at pregrasp height, followed by continuous_grasp_depth, a hard final Z-only descend with close_gripper still attached only to that final vertical segment
- Added a small configurable wrist-to-pinch contact offset model via --grasp-contact-offset-x/--grasp-contact-offset-y and debug logs/markers for selected pregrasp target, grasp contact target, object center, XY contact delta, final-descend vertical-only status, and close trigger position
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Added an experimental pivot-centered grasp mode to scripts/task1_smooth_autoseed_multi_object_baseline.py behind --experimental-pivot-arc-grasp while leaving the current baseline path as the default/fallback
- Experimental mode uses the selected arm's wrist_pitch_link as the control pivot, drives a pivot anchor with shoulder_pitch/shoulder_roll/shoulder_yaw/elbow_roll, then drives a short lower-chain wrist_roll arc with elbow_yaw/wrist_pitch/wrist_roll; close_gripper is gated by an estimated pinch-center-to-object distance instead of a vague phase trigger
- Added sweepable experimental knobs: --pivot-to-pinch-distance-m, --pivot-anchor-height-offset-m, --pivot-anchor-forward-offset-m, --pivot-anchor-lateral-offset-m, --pivot-arc-contact-tolerance-m, --pivot-arc-max-steps, and --pivot-arc-close-distance-m
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime validation: seed=1 target-index=2 --pregrasp-pullback-m 0.02 --experimental-pivot-arc-grasp built and executed the experimental plan without fallback, selected R_wrist_pitch_link as pivot, passed Stage A, started Stage B, and satisfied the close distance gate with estimated pinch_center_to_object_distance approximately 0.03865 m
- Runtime result: the experimental run failed later at post_close_verify with failure_reason=object_not_lifted; geometry metrics classify the estimated pinch center as nearer the finger gap than the wrist, but no GUI visual confirmation was made in this run, so this is not a claimed grasp success
- Runtime artifact: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T074513Z_experimental_pivot_arc_seed1_target2_pb0p02.log
- Reworked the experimental pivot-arc runtime path so Stage B is no longer two separate blocking IK segments after the pivot anchor
- After experimental_pivot_anchor succeeds, the executor now captures the current upper-chain joint targets plus the pivot reference position, then continuously reapplies those upper-chain targets during every lower-chain arc IK update and during the close dwell
- Replaced experimental_pinch_arc_mid and experimental_pinch_contact with one experimental_locked_lower_chain_arc segment that follows parameterized arc waypoints with only elbow_yaw/wrist_pitch/wrist_roll commanded
- Added pivot-lock proof logs for upper_chain_locked_targets, upper_chain_lock_active, pivot_reference_position, pivot_position_after_anchor, pivot_position_during_arc_start/end, pivot_drift_per_waypoint, pivot_drift_norm_max/mean/final, and close_condition_satisfied_before_close
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac GUI/runtime validation was run for this patch yet; next validation should run seed=1 target-index=2 --pregrasp-pullback-m 0.02 --experimental-pivot-arc-grasp and inspect pivot drift plus GUI smoothness
- Narrowed the experimental locked-pivot arc runtime to reduce jerk: the lower-chain arc now performs one pre-stream endpoint estimate, then streams smooth lower-chain joint interpolation samples while continuously reapplying the upper-chain lock
- Added --pivot-arc-frame-step-updates with default 1 so the experimental arc no longer uses the normal ik_settle_steps behavior inside every arc sample
- Added proof metrics for upper_chain_joint_error_per_step, upper_chain_joint_error_max_during_arc, upper_chain_joint_error_mean_during_arc, and real arc micro_stop_frames/micro_stop_samples computed from the observed arc trace
- Cleaned lock-state log semantics: anchor logs upper_chain_lock_captured=true and upper_chain_lock_active_during_anchor=false, while the arc logs upper_chain_lock_active_during_arc=true
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac GUI/runtime validation was run for this smoothing patch yet; next validation should compare GUI stop/go behavior and inspect upper-chain joint error plus pivot drift in the log
- Unified the experimental pivot-arc runtime control style further: experimental continuous_pregrasp and experimental_pivot_anchor now use streaming joint interpolation instead of move_end_effector_to_target blocking IK
- continuous_pregrasp is a streaming hold because candidate selection has already positioned the arm at the selected pregrasp; experimental_pivot_anchor uses a deterministic upper-chain endpoint heuristic and streams shoulder/elbow targets over denser samples
- Removed the remaining finite-difference endpoint estimate from the locked lower-chain arc runtime path; Stage B now uses a deterministic lower-chain endpoint heuristic plus smooth streamed lower-joint interpolation
- Added --pivot-arc-stream-samples with default 48 and kept --pivot-arc-frame-step-updates default 1 so the experimental branch uses smaller streamed increments across pregrasp, pivot anchor, and lower-chain arc
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac GUI/runtime validation was run for this unified streaming patch yet; next GUI check should verify whether pregrasp, pivot anchor, and lower-chain arc now look consistently smooth
- Converted the Task 1 smooth manipulation runtime further toward competition-style streaming control: startup joint setup now sends position targets only, and the script no longer calls set_dof_position or move_end_effector_to_target in the smooth runtime path
- Default smooth pauses and holds are now zero, contact dwell is reduced to 2 steps, generic/pivot stream samples default to 80, and the stream frame-step defaults remain 1 so target updates do not become mini-settle blocks
- Replaced cubic smoothstep interpolation with a shared quintic minimum-jerk blend for streamed body segments and the locked lower-chain arc; the continuous-cycle executor now routes ordinary pregrasp, descend, lift, carry, place, and retreat through dense streaming target updates
- Preserved pivot/lock diagnostics for pivot_drift_norm_max/mean/final, upper_chain_joint_error_max/mean/final during arc, and micro_stop_frames/micro_stop_samples while adding explicit streaming_controller/no_blocking_ik/sample-count/frame-step metadata
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Runtime limitation: no Isaac GUI/runtime validation was run for this runtime-motion patch; next Linux GUI run should inspect visible stop/go reduction and compare pivot drift plus micro-stop metrics
- Fixed the GUI startup pose issue in scripts/task1_smooth_autoseed_multi_object_baseline.py: the robot is now seeded once from the official baseline startup joint map before manipulation begins, matching the Ubtech_sim/source/RobotArticulation.py initialization behavior instead of starting motion from the default stand pose
- The direct set_dof_position call is isolated to this initialization-only seed; normal smooth manipulation runtime remains target-streamed through set_dof_position_target and does not use blocking IK or runtime joint teleport forcing
- Reduced the post-startup target sync window from 600 steps to 5 steps because the initial joint state is now seeded directly before the streamed controller path begins
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Linux GUI validation with ISAAC_SIM_PYTHON=/home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/watch_task1_gui.sh passed the previous articulation/startup blocker: apply_official_startup_pose condition_met=true, initialization_seed_supported=true, and max_joint_error was approximately 0.00030 rad
- The same GUI run continued into streamed manipulation with streaming_controller=true/no_blocking_ik=true/stream_samples=80 and then failed later at descend_failed because the end effector remained approximately 0.276 m from the target before close; this is a grasp geometry/reach issue, not the startup-load failure
- Runtime artifact: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T095431Z_gui_watch_seed1_target2_depth_watch.log
- Tuned only the experimental pivot reach heuristics in scripts/task1_smooth_autoseed_multi_object_baseline.py while keeping the streaming controller, quintic interpolation, upper-chain lock, and locked-pivot arc structure unchanged
- _heuristic_upper_chain_anchor_endpoint now drives shoulder_pitch and elbow_roll more aggressively toward the forward target error, with slightly stronger shoulder roll/yaw lateral compensation
- _heuristic_lower_chain_arc_endpoint now permits a larger lower-chain endpoint cap and stronger wrist_pitch forward/depth response for a deeper approach
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- GUI validation note: the default scripts/watch_task1_gui.sh run does not enable --experimental-pivot-arc-grasp, so it did not exercise these heuristic functions and still failed at descend_failed with the previous approximately 0.276 m pre-close distance
- GUI validation with scripts/watch_task1_gui.sh --experimental-pivot-arc-grasp exercised experimental_pivot_anchor and experimental_locked_lower_chain_arc with streaming_controller=true/no_blocking_ik=true/stream_samples=80; the pre-close distance improved to approximately 0.210 m but still failed descend_failed
- Experimental pivot diagnostics from that run remained stable: pivot_drift_norm_max approximately 0.000080 m, pivot_drift_norm_mean approximately 0.000044 m, upper_chain_joint_error_max_during_arc approximately 0.000147 rad, and upper_chain_joint_error_mean_during_arc approximately 0.000116 rad
- Runtime artifact: LOG_ROOT/task1_smooth_autoseed_multi_object_baseline_20260415T100234Z_gui_watch_seed1_target2_depth_watch.log
- Reviewed the smooth streaming motion path and found the ordinary continuous cycle still assigned single_dls_endpoint to every conceptual segment, causing _execute_streaming_body_segment to run finite-difference endpoint trials before each streamed phase
- Replaced ordinary main-cycle single_dls_endpoint usage with a global_precomputed_joint_waypoint stream: the cycle now precomputes lightweight heuristic joint waypoints for the full pregrasp/align/descend/lift/carry/place/retreat chain once, then streams across them with one global quintic progress variable
- Added diagnostics for endpoint_rebuilds_during_cycle=0, finite_difference_jacobian_calls_during_cycle=0, single_global_stream_cycle=true, and global_quintic_minimum_jerk_over_precomputed_joint_waypoints
- Left _estimate_unlocked_stream_endpoint available for pregrasp candidate probing/fallback-style paths, but removed it from normal cycle segments and changed experimental post-arc transit segments to main_cycle_heuristic_endpoint instead of single_dls_endpoint
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Patched the remaining smooth-path jerk source in scripts/task1_smooth_autoseed_multi_object_baseline.py: the global precomputed cycle no longer piecewise-linearly blends between joint waypoints, and now samples a cubic Hermite joint chain with minmod tangents under the existing global quintic time warp
- Changed main-cycle waypoint generation from cumulative heuristic deltas to absolute target-relative endpoints from the cycle start, reducing accumulated heuristic drift and transport/approach overshoot
- Replaced unsigned world-XY forward heuristics with signed robot-base-frame forward/lateral components, active-arm mirroring, inward-lateral attenuation, and a rearward-motion guard in the upper-chain anchor, lower-chain arc, and main-cycle endpoint heuristics
- Preserved the target-only streaming controller, no blocking IK in the main smooth cycle, upper-chain lock diagnostics, pivot drift diagnostics, and micro-stop metrics while updating logs to report the cubic-Hermite global stream and absolute waypoint policy
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Tightened the experimental locked-arc close gate so close_gripper now requires pinch distance, wrist-to-object distance, and vertical wrist/object proximity, with a hard block when the wrist is more than 5 cm above the object contact z
- Routed continuous_prebin, continuous_place_depth, and continuous_retreat through direct_cartesian_target endpoint estimation followed by the existing streamed target interpolation, so post-grasp transport is no longer limited to the lightweight heuristic endpoint
- Disabled carry stabilization dwell as a non-contact pause source in both continuous executors; release/open dwell remains event-only and close/open event behavior is preserved
- Added cartesian-distance scaling to _heuristic_main_cycle_endpoint_delta so small target deltas generate proportionally smaller joint deltas and overshoot is reduced
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Added an explicit grasp-geometry layer to scripts/task1_smooth_autoseed_multi_object_baseline.py using the baseline-style planner only as a geometry reference: object targets are converted to signed robot-base components, a grasp frame is built from reach direction plus world-down, and local TCP offset is transformed into an absolute world contact target
- Refactored _grasp_contact_geometry to emit pregrasp_target_world, contact_target_world, lift_target_world, contact_z_world, R_grasp_world, tcp_offset_local, and tcp_offset_world; pregrasp/contact/lift targets now come from absolute grasp geometry instead of selected-pregrasp XY plus weak offsets
- Tightened the experimental locked lower-chain close gate to require pinch distance, wrist distance, and wrist/contact-Z alignment, with contact_z/wrist_z/pinch_dist/wrist_dist/z_error/close_gate_passed logged per close check
- Renamed the post-grasp transport endpoint strategy for continuous_prebin, continuous_place_depth, and continuous_retreat to absolute_cartesian_target; it still performs a single endpoint estimate before streamed quintic target execution and does not introduce a per-frame IK loop
- Lightweight test result: python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed
- Stabilized the default Task 1 path for phase-by-phase debugging: the normal cycle now labels approach/grasp/lift segments as main_cycle_heuristic_endpoint and keeps absolute_cartesian_target only for carry/place/retreat, so the logged endpoint strategy matches the executor actually used by the mixed default plan
- Left experimental_pivot_arc_grasp and locked lower-chain arc logic opt-in only behind --experimental-pivot-arc-grasp; default logs now report stable_phase_logged_streaming_cycle_with_absolute_transport_endpoint_targets instead of the old global-precomputed policy label
- Static sanity check result: python AST/symtable scan reported undefined_global_count=0, python -m py_compile scripts/task1_smooth_autoseed_multi_object_baseline.py passed, and python scripts/task1_smooth_autoseed_multi_object_baseline.py --help loaded successfully
- Added scripts/task1_cartesian_dls_phase_baseline.py as a new isolated Task 1 baseline that abandons the old heuristic reach default path and uses measured Isaac 3D Cartesian DLS phases
- The new script keeps official SceneBuilder setup, official startup pose loading, target selection, gripper commands, grasp geometry planning, phase diagnostics, and final bin/stability checks while leaving scripts/task1_smooth_autoseed_multi_object_baseline.py untouched
- Pregrasp candidate evaluation now restores the arm to a saved reference joint state before each candidate trial and after each trial, and logs the candidate start EE/joint state so failed trials do not contaminate the next candidate
- Failure naming is separated for close_gripper_failed, object_not_lifted, place_failed, release_failed, and object_outside_bin
- Lightweight test result: python -m py_compile scripts/task1_cartesian_dls_phase_baseline.py passed
- Runtime limitation: no Isaac GUI/runtime validation has been run for the new Cartesian DLS phase baseline yet; first validation should use seed=1 target-index=2 with --gui --hold-open and inspect the per-phase DLS diagnostics
- Fixed robot prim path selection in scripts/task1_cartesian_dls_phase_baseline.py so the default no longer treats /Root/Ref_Xform/Ref as source of truth; after scene.build_robot it validates scene.robot_prim_path, then stage-detected articulation roots, then optional --prim-path, and finally the hardcoded official path only if valid
- Added path-selection diagnostics for scene.robot_prim_path, detected articulation roots, chosen robot prim path, fallback status, and selection attempts; lightweight test result: python -m py_compile scripts/task1_cartesian_dls_phase_baseline.py passed
- Restored /Root stage parent creation immediately after _create_minimal_scene in scripts/task1_cartesian_dls_phase_baseline.py because organizer SceneBuilder composes table/parts/robot under that parent even though /Root/Ref_Xform/Ref is not safe as a robot path source of truth
- Added early stage_debug root/world validity prints and before/after scene_build_debug prints around build_table, build_parts, and build_robot; lightweight test result: python -m py_compile scripts/task1_cartesian_dls_phase_baseline.py passed
- Patched scripts/task1_dualarmik_phase_baseline.py after first DualArmIK runtime failure at pregrasp_candidate_failed: log writing now recursively converts numpy arrays/scalars before json.dumps, pregrasp candidate diagnostics now distinguish no-solution IK from solution-exceeded-tolerance, and temporary debug gates/CLI fixed-RPY overrides were added without changing TCP fallback or reverting to DLS
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed; runtime validation still needs Linux Isaac run with seed=1 target-index=2
- Patched scripts/task1_dualarmik_phase_baseline.py after definitive all_candidates_dualarmik_no_solution diagnostics: pregrasp selection now searches an explicit fixed orientation preset library per arm, builds each preset target with preset-derived approach/up axes, and logs per-preset no-solution summaries while preserving the derived orientation path as reference-only diagnostics
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help loaded the debug/pregrasp CLI flags successfully; runtime validation still needs a Linux Isaac run
- Expanded scripts/task1_dualarmik_phase_baseline.py for the confirmed all_candidates_dualarmik_no_solution pregrasp failure: default IK rotation tolerance is now 0.08, phase rotation tolerance is now 0.15, IK nullspace weight is forced to 0.0 for this debug stage, each arm now searches 20 fixed orientation presets including roll-based families, and pregrasp candidates now include lateral_positive, lateral_negative, and higher_farther variants
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help exposes the expanded candidate/debug flags; runtime validation still needs Linux Isaac logs to identify which preset family is reachable
- Patched scripts/task1_dualarmik_phase_baseline.py candidate classification so official DualArmIK internal ok=False is no longer treated as automatic no-solution: candidate solves now use separate reachability settings max_iter=140, pos_tol=0.02, rot_tol=0.20, null_weight=0.0, classify finite returned poses as catastrophic_no_solution, solved_but_not_internal_ok, candidate_error_exceeded_tolerance, or valid_candidate, and can choose a guarded forced best-effort candidate only within 0.10 m / 0.30 rad
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help exposes the candidate IK knobs; runtime validation still needs Linux Isaac logs to see whether failures are truly catastrophic or just internal non-convergence
- Stabilized scripts/task1_dualarmik_phase_baseline.py after runtime showed bizarre reachable arm postures: removed forced best-effort pregrasp promotion, reduced orientation search to two top-down presets per arm, reduced pregrasp candidates to nominal/slightly_higher/slightly_farther, removed the lateral candidate CLI knob, and tightened candidate IK reachability defaults to max_iter=120, pos_tol=0.015, rot_tol=0.12 with strict candidate success required
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help no longer exposes --candidate-lateral-offset; runtime validation still needs Linux Isaac logs to confirm posture quality
- Patched scripts/task1_dualarmik_phase_baseline.py after runtime showed both sixforce FK-vs-Isaac frame diffs at approximately 0.2200 m: audited the organizer CoordinateTransform path and added a startup frame-alignment selector that compares raw torso_link anchoring against the official compensation matrix variants from Ubtech_sim/main.py, selects the lowest sixforce alignment residual before grasp planning, and logs torso_prim_path, selected transform origin/rotation, and per-arm FK-vs-Isaac residuals after sync
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help still loads; runtime validation should first confirm ee_alignment_diagnostics_after_fix drops from the previous approximately 0.2200 m before further reach tuning
- Patched scripts/task1_dualarmik_phase_baseline.py after runtime showed the torso-compensation selector still chose official_coordinate_utils_from_torso_link_uncompensated with approximately 0.2200 m sixforce residuals on both arms: added full SE3 FK-vs-Isaac EE delta diagnostics, root-cause classification for fixed EE offset vs URDF/USD/link mismatch vs remaining torso/root mismatch, and a conditional EE-frame compensation layer that only activates when left/right local FK-to-Isaac deltas are near-constant
- The DualArmIK target path now keeps desired targets in the physical Isaac EE frame for logs/gates, converts to Pinocchio targets with target * inverse(EE_delta) only when compensation is active, and reports current FK as Pinocchio FK * EE_delta for candidate/servo error checks; orientation presets, candidate list, approach axis, TCP fallback, and servo architecture were not changed
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help loaded successfully; next Linux Isaac run should inspect robot.ee_frame_delta_diagnostics.comparison.root_cause_classification and expect compensated ee_alignment_diagnostics_after_fix.max_diff_m to drop to centimeter scale only if the mismatch is classified as fixed_ee_frame_offset_mismatch
- Promoted the tilted topdown orientation family in scripts/task1_dualarmik_phase_baseline.py after controlled Isaac sweeps showed pitch=2.80 reduced align error when selected: right_topdown_tilted_forward and left_topdown_tilted_forward are now first in the deterministic preset tables, with the previous topdown_forward and topdown_inward presets kept as fallbacks
- Slightly improved selected-orientation logging by adding selected_orientation_preset_label and selected_orientation_preset_rpy to the select_pregrasp_candidate phase details and top-level payload; approach-axis logic, TCP fallback, EE-frame compensation, candidate list, servo backend, and state machine were not changed
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, and python scripts/task1_dualarmik_phase_baseline.py --help loaded successfully; next Linux Isaac validation should rerun seed=1 target-index=2 with --arm right --approach-axis-mode pos_z and confirm the selected preset is right_topdown_tilted_forward before evaluating align/descend errors
- Checked scripts/task1_dualarmik_phase_baseline.py after an interrupted region-based grasp-family patch and repaired the incomplete wiring: removed duplicate selected-region payload assignments, kept target-mode default at contact_axis, reused one orientation preset library object in the plan log, and made servo_descend lock the selected orientation preset RPY instead of hardcoding the old vertical claw RPY
- The DualArmIK preset library is now split into right/left vertical and horizontal families, target regions are classified from forward_base with defaults near_body < 0.28, mid 0.28-0.42, far >= 0.42, and candidate selection receives only the ordered family list for that region
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed, python scripts/task1_dualarmik_phase_baseline.py --help loaded and exposes --horizontal-far-threshold / --near-body-threshold; runtime limitation: no Linux Isaac simulation validation was run for this patch yet
- Updated scripts/task1_dualarmik_phase_baseline.py from misleading vertical/horizontal preset families to approach-direction families: z_approach keeps the existing top-down presets, and world_y_approach is a small first-test calibration set for world-Y-style approach behavior
- Region policy now uses far >= 0.42 -> world_y_approach, near_body < 0.28 -> z_approach then world_y_approach, and mid -> z_approach; --far-threshold is the primary CLI name while --horizontal-far-threshold remains as a compatibility alias
- Added preset axis diagnostics to orientation records and candidate logs: approach_axis_base, approach_axis_world, dot_with_world_pos_y, and dot_with_world_neg_y, plus selected preset world-axis fields in top-level payload and descend phase details
- Preserved the existing DualArmIK candidate validation and servo_descend selected-RPY preservation; no Linux Isaac GUI/runtime validation was run, so world_y_approach RPY candidates remain calibration-oriented and must be verified visually/logged in Isaac
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed; python scripts/task1_dualarmik_phase_baseline.py --help exposes --far-threshold / --near-body-threshold; a helper check confirmed 0.27 -> near_body, 0.28/0.419 -> mid, and 0.42 -> far
- Updated scripts/task1_dualarmik_phase_baseline.py from approach-family-only planning to explicit world-space AB motion semantics, where point A is the DualArmIK physical EE/sixforce origin and point B is a fingertip/contact proxy computed as point A plus a selected local offset transformed into world space
- FAR targets now use a far_low_side_B_driven policy: prepare point B low near object/contact Z with AB table-parallel/world-Y-oriented, then advance point B to the object contact before close/lift; MID targets use mid_vertical_Z_descend, aligning point B over object WORLD XY and descending along WORLD Z while preserving the selected vertical AB orientation
- Added AB diagnostics and debug targets for point A, point B, contact B, low-side prepare B, selected motion policy, AB axis world, AB table-parallel/vertical scores, point-B pre-close error, and stop-after-lift completion; default runtime now stops after lift unless --continue-after-lift is provided
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed; python scripts/task1_dualarmik_phase_baseline.py --help exposes --far-low-side-clearance, --point-b-offset-local, and --continue-after-lift; a helper geometry check confirmed z_approach yields AB along world Z and world_y_approach candidates yield AB along world +/-Y under identity base/world rotation
- Runtime limitation: no Linux Isaac GUI/runtime validation was run for the AB motion-semantics patch; next validation should inspect whether the point-B proxy matches the actual finger mesh and whether far low-side approach visually advances from the intended world-Y side
- Patched scripts/task1_dualarmik_phase_baseline.py after far-region runtime reached region/family selection but failed pregrasp candidate strict acceptance: far candidate validation now uses region-specific acceptance tolerances of 0.07 m position and 0.28 rad rotation, with CLI overrides --far-candidate-position-tolerance and --far-candidate-rotation-tolerance
- Kept region thresholds, phase machine, EE-frame compensation, and far/mid motion policy unchanged; the relaxed values apply only inside _evaluate_pregrasp_candidates for target_region=far, while mid and near_body continue to use the existing pregrasp/rotation tolerance gate unless debug-pregrasp overrides are supplied
- Added flattened-candidate diagnostics for best_position_error_candidate, best_rotation_error_candidate, best_combined_error_candidate, per-candidate gate margins/excess, combined normalized gate error, dualarmik_success/internal_ok, and the tolerance source policy so far failures can distinguish a too-strict gate from wrong world-Y/Point-B geometry
- Lightweight test result: python -m py_compile scripts/task1_dualarmik_phase_baseline.py passed; python scripts/task1_dualarmik_phase_baseline.py --help exposes the far candidate tolerance knobs; helper check confirmed far -> 0.07/0.28 and mid/near_body -> existing 0.045/0.15 candidate acceptance defaults
- Runtime limitation: no Linux Isaac GUI/runtime validation was run for this tolerance/diagnostics patch; next far run should inspect select_pregrasp_candidate.best_*_candidate and candidate gate margin fields before changing presets or geometry
- Corrected scripts/task1_dualarmik_phase_baseline.py FAR world_y_approach preset definitions after runtime diagnostics showed the old family produced best-candidate approach_axis_world approximately world +X instead of world +/-Y
- Replaced the old roll-only / yaw-pi world_y_approach candidates with yaw-quarter-turn candidates for both arms, moving the local +Z/AB approach axis to base +/-X so the observed robot/world transform maps it to world +/-Y
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --target-index 2 --log-suffix far_world_y_correction_test
- Runtime result: target_region=far, approach_family_order=[world_y_approach], selected_approach_family=world_y_approach, selected_orientation_preset_label=right_world_y_approach_pos_y_yaw_plus_quarter, selected_orientation_preset_approach_axis_world approximately [0.0000069, 0.999999999, -0.0000527]
- Best FAR candidate diagnostics after correction all pointed to world +Y: best_position_error_candidate, best_rotation_error_candidate, and best_combined_error_candidate approach_axis_world were approximately [0.0000069, 0.999999999, -0.0000527], with candidate selection succeeding before the run later failed at far_prepare_low_side_approach
- Lightweight/static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help; no second preset correction pass was needed


- Augmented scripts/task1_dualarmik_phase_baseline.py FAR world_y_approach presets with deterministic AB-axis roll variants: each corrected world-Y base preset now expands to +90 deg, -90 deg, and reference rolls generated by post-multiplying the preset rotation about local +Z/AB before converting back to Euler XYZ
- Added FAR axial-roll diagnostics to preset/candidate/phase logs, including preset_axial_roll_variant_label, preset_axial_roll_about_ab_rad, AB_axis_world, far_low_side_prepare_B_world, and contact_point_B_world while preserving region thresholds, candidate gate, EE-frame compensation, and phase order
- Lowered FAR low-side point-B geometry: DEFAULT_FAR_LOW_SIDE_CLEARANCE changed from 0.03 m to 0.002 m, added --far-low-side-gap-above-support default 0.002 m, and FAR contact B now uses object/table support Z plus that gap instead of the top-down object-top contact Z
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --target-index 2 --log-suffix far_ab_roll_height_test
- Runtime result: target_region=far, approach_family_order=[world_y_approach], preset_count=12, selected_orientation_preset_label=right_world_y_approach_pos_y_yaw_plus_quarter_ab_roll_minus_quarter_palm_down_test, selected axial roll=-1.5707963267948966 rad, and selected approach_axis_world/AB_axis_world remained approximately [0.0000069, 0.999999999, -0.0000527]
- Candidate selection still passed after the roll/height patch; FAR B geometry was lowered from the old top-down reference contact_z_world approximately 1.08369 m to contact_point_B_world z approximately 1.01082 m, with far_low_side_prepare_B_world z approximately 1.01282 m
- Runtime still failed at far_prepare_low_side_approach with final_error approximately 0.05639 m versus pregrasp_tolerance 0.045 m; this is improved from the previous far_world_y_correction_test pregrasp final_error approximately 0.07119 m, but GUI verification is still needed to confirm whether the selected -90 deg axial roll visually rotates the palm/fingers in the desired direction


- Refined scripts/task1_dualarmik_phase_baseline.py FAR-only A/B contact geometry after GUI showed the world-Y side approach and palm roll were close but point B still stopped short while point A/wrist was too low
- Added explicit FAR knobs: --far-point-b-forward-extension default 0.012 m extends the B contact target along the current horizontalized FAR reach axis; --far-point-a-extra-height-clearance default 0.018 m offsets the point-B proxy so point A sits higher while B stays low; --far-point-b-gap-above-support default 0.002 m keeps B targeted 2 mm above the support plane
- Preserved region thresholds, far/mid policy split, world_y approach direction logic, EE-frame compensation, candidate gates, phase order, and MID z_approach behavior; logging now carries the adjusted point-B local offset, FAR reach axis, point-A clearance, point-B extension/gap, selected/contact point A/B targets, and pre-close point-B error
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --target-index 2 --log-suffix far_ab_geometry_refine_test
- Runtime result: target_region=far, selected_motion_policy=far_low_side_B_driven, selected_orientation_preset_label=right_world_y_approach_pos_y_yaw_plus_quarter_ab_roll_minus_quarter_palm_down_test, selected approach_axis_world remained approximately [0.0000069, 0.999999999, -0.0000527], candidate selection still passed, far_prepare_low_side_approach now passed with final_error approximately 0.04452 m, and execution reached close_gripper/micro_lift_probe before failing object_not_lifted
- Runtime diagnostics showed contact_point_B_world z approximately 1.01082 m, object_support_z approximately 1.00882 m, point A target z approximately 1.02883 m, but actual_point_B_world_before_close remained approximately 0.05854 m from target; next FAR iteration should focus on why the servo/IK pose does not bring the physical B proxy fully to the target before close rather than changing region or approach-family logic


- Refined scripts/task1_dualarmik_phase_baseline.py FAR posture after GUI showed the world-Y approach was basically correct but the wrist/point A needed to clear table clutter while point B stayed low near the support plane
- Added --far-ab-downward-slant-deg default 25.0 and now derive the FAR point-A/point-B height separation from the requested slant using the base point-B horizontal span, while preserving --far-point-a-extra-height-clearance default 0.018 m as a lower bound; in the validation run the applied A/B height separation was approximately 0.07228 m and the logged AB downward slant was approximately 25.00 deg
- Kept point B low with --far-point-b-gap-above-support default 0.002 m and kept the existing --far-point-b-forward-extension default 0.012 m; region thresholds, far/mid policy split, world_y approach presets, candidate gate, EE-frame compensation, phase order, and MID z_approach behavior were not changed
- Added nearest-object-first target selection with --target-selection-policy nearest as the default and --target-selection-policy index as a compatibility override; nearest ranking uses the smallest robot-base forward_base, then abs(lateral_base), then Euclidean world distance, and logs all candidate object records plus the selected nearness metric/reason
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --log-suffix far_slant_nearest_test
- Runtime result: nearest policy selected /Replicator/Ref_Xform_02 at target_index=1 with forward_base approximately 0.53655, target_region=far, selected world_y approach_axis_world remained approximately [0.0000069, 0.999999999, -0.0000527], candidate selection passed, far_prepare_low_side_approach passed with final_error approximately 0.03155 m, and far_reach_B_to_object passed with final_error approximately 0.03431 m
- Pre-close diagnostics on that run showed target point A z approximately 1.07442 m, target point B z approximately 1.00214 m, actual point B error before close approximately 0.03587 m, then close_gripper completed but micro_lift_probe failed with object_not_lifted; next work should stay focused on grasp capture/contact, not region selection or world-Y approach direction


- Refined scripts/task1_dualarmik_phase_baseline.py FAR final contact sequence after GUI showed direct side-driven B contact could push the object before grasp: FAR now plans separate far_low_side_prepare_B_world, far_xy_align_B_world, and far_descend_B_world targets, executes far_prepare_low_side_approach -> far_align_B_over_object_xy -> far_lower_B_world_z, then closes and runs the existing micro_lift_probe/far_lift path
- The FAR XY-align target keeps point B at object WORLD XY but above the object before lowering; --far-xy-align-clearance-above-object was added with default 0.035 m, while far_descend_B_world still targets support Z + --far-point-b-gap-above-support default 0.002 m
- Added/preserved FAR contact-sequence logging for selected_far_contact_sequence_policy, selected_far_low_side_prepare_B_world, selected_far_xy_align_B_world, selected_far_descend_B_world, far_xy_align_clearance_above_object_m, actual_point_B_world_before_close, and point_B_error_before_close_m; region thresholds, far/mid policy split, world_y approach presets, candidate gate, EE-frame compensation, and MID behavior were not changed
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --log-suffix far_xy_then_z_clearance35_test
- Runtime result: nearest policy selected /Replicator/Ref_Xform_02 at target_index=1 with target_region=far, selected approach_axis_world stayed approximately [0.0000069, 0.999999999, -0.0000527], candidate selection passed, and the new far_prepare_low_side_approach, far_align_B_over_object_xy, and far_lower_B_world_z phases all met their servo gates before close_gripper
- The run still failed at micro_lift_probe with object_not_lifted; log object_trace showed the selected object's center changed from approximately [0.81401, 0.33655, 1.02572] initially to [0.81438, 0.31574, 1.01644] by after_far_xy_align and then remained there, so the script now has the requested XY-then-Z structure but GUI/log verification still indicates object displacement before close


- Refined scripts/task1_dualarmik_phase_baseline.py vertical z_approach close timing after GUI showed fingers closing before the mesh/contact proxy reached the object: vertical contact point B now targets object/table support Z + --vertical-point-b-gap-above-support, default 0.001 m, instead of the previous object-top plus descend-clearance contact height
- Added --vertical-close-point-b-tolerance default 0.006 m and a vertical-only pre-close point-B gate; for MID/near-body z_approach the script now skips close_gripper and fails with vertical_contact_not_reached_before_close if actual point B is not within the gate of the contact mark
- Fixed the vertical descend early-success path: the vertical descend phase now servos against the final contact pose directly with position tolerance min(--descend-tolerance, --vertical-close-point-b-tolerance), instead of using a one-step dynamic 2 mm lowering target that could report success before the final contact mark was reached
- FAR geometry, region thresholds, world_y approach presets, candidate gates, EE-frame compensation, and nearest-target selection were not changed
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --log-suffix vertical_contact_mark_direct_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, selected_orientation_preset_label=right_z_approach_straight, selected vertical contact mark B world [0.8166943, 0.1985759, 1.0217553], object support z approximately 1.0207553, and the requested B gap was 0.001 m
- The run no longer closed the gripper early; it failed at mid_descend_world_z_keep_AB_vertical with descend_failed because final point B remained approximately [0.7702214, 0.2382643, 1.0915748], around 6.5 cm pose error / about 7 cm above the contact mark, so the next vertical iteration should address why the direct contact target is not reached rather than allowing close before contact

- Refined scripts/task1_dualarmik_phase_baseline.py vertical z_approach XY tracking after GUI diagnosis showed wrist/sixforce projection introduces a small offset from the actual finger/mesh midpoint: the vertical branch now resolves --vertical-xy-reference-link, default pgc_base_link, and uses that reference's WORLD XY for object alignment while keeping point B/finger proxy Z at support + --vertical-point-b-gap-above-support before close
- The pgc reference resolver first matches the active arm's dynamic-control body, then prefers the selected prim's USD bbox center as the mesh/reference position; it falls back to rigid-body pose only if bbox center is unavailable, and logs source, selected path, bbox center, body pose, local EE-frame offset, and whether vertical_xy_reference_active is true
- Vertical planning now adjusts the commanded point-B XY so the pgc/reference XY lands on the object WORLD XY; the vertical pre-close gate now requires both point-B contact-mark error <= --vertical-close-point-b-tolerance and pgc/reference XY error <= --vertical-xy-reference-tolerance when the reference is active
- FAR world-Y policy, region thresholds, candidate gate, EE-frame compensation, phase machine, nearest-target selection, and MID z_approach preset family were not changed
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --log-suffix vertical_pgc_bbox_xy_reference_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, selected_orientation_preset_label=right_z_approach_straight, selected_vertical_xy_reference_active=true, and the resolver selected /Root/Ref_Xform/Ref/R_pgc_base_link/R_pgc_base_link using source=dynamic_control_body_usd_bbox_center
- Runtime diagnostics showed the bbox-center reference produced a meaningful local offset [0.0000038, 0.0026389, 0.0666526] from the EE frame; the selected vertical contact mark B shifted from the raw object contact [0.8166943, 0.1985759, 1.0217553] to [0.8140555, 0.1985675, 1.0217553] so the pgc/reference XY, not the wrist/sixforce origin, aligns to the object XY
- The run still failed at mid_descend_world_z_keep_AB_vertical before close_gripper, with final point B approximately [0.7510247, 0.2036018, 1.0955817] versus target [0.8140555, 0.1985675, 1.0217553]; this is a remaining vertical reach/servo convergence problem, not an early-close regression

- Refined scripts/task1_dualarmik_phase_baseline.py vertical z_approach descend to continuously correct the commanded point-B target from the live pgc/reference mesh XY error during mid_descend_world_z_keep_AB_vertical, instead of only using the one-time pgc_base_link bbox offset from the planning step
- The vertical descend target callback now reads the current EE pose every servo tick, reconstructs the current pgc/reference world point from vertical_xy_reference_offset_local, computes object_xy - current_reference_xy, and commands nominal_contact_B_xy plus that error while keeping point B at support Z + --vertical-point-b-gap-above-support; FAR behavior and policy selection were not changed
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --log-suffix vertical_nominal_pgc_xy_feedback_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, selected_orientation_preset_label=right_z_approach_straight, selected_vertical_xy_reference_active=true, feedback samples were recorded, and the final pgc/reference XY error improved from approximately 0.0277 m in the prior one-time-offset run to approximately 0.0148 m with continuous feedback
- The run still failed before close_gripper at mid_descend_world_z_keep_AB_vertical: final point B was approximately [0.7624315, 0.1955212, 1.0938252] versus target [0.8140555, 0.1985675, 1.0217553], with final pose error approximately 0.0585 m and final rotation error approximately 0.4448 rad; next vertical work should focus on IK/servo reachability or orientation tolerance rather than allowing early close

- Refined scripts/task1_dualarmik_phase_baseline.py vertical XY reference from pgc_base_link to the active hand finger midpoint: default --vertical-xy-reference-link is now finger_midpoint, resolving R/L_finger1_link and R/L_finger2_link for the selected arm, using the midpoint of their USD bbox centers as the object-XY reference while retaining single-link override support through --vertical-xy-reference-link
- Added live vertical-reference feedback for the descend phase: mid_descend_world_z_keep_AB_vertical now re-queries the current finger1/finger2 midpoint each servo tick when finger_midpoint mode is active, logs the live component positions, and falls back to the initial EE-frame offset only if the live query fails
- Preserved FAR policy, region thresholds, candidate gates, EE-frame compensation, phase order, z_approach presets, point-B low contact Z, and the pre-close gate; the vertical gate now refers generically to the vertical XY reference rather than pgc
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --log-suffix vertical_finger_midpoint_xy_feedback_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, selected_orientation_preset_label=right_z_approach_straight, selected_vertical_xy_reference_mode=finger1_finger2_midpoint, source=finger_link_pair_midpoint; the resolver selected /Root/Ref_Xform/Ref/R_finger1_link/R_finger1_link and /Root/Ref_Xform/Ref/R_finger2_link/R_finger2_link from USD bbox centers
- Runtime diagnostics showed live feedback was active with runtime_source=live_vertical_xy_reference_query and no fallback; final point-B pose error improved slightly from the previous approximately 0.0585 m to approximately 0.0532 m, but final live finger-midpoint XY error was still large at approximately 0.0631 m, so the midpoint did not yet align to the object XY before the descend gate failed
- The run still failed before close_gripper at mid_descend_world_z_keep_AB_vertical with descend_failed; next vertical iteration should focus on why the solver cannot move the live finger midpoint far enough in XY during the low vertical descend, not on gripper close timing


- Implemented focused scripts/task1_dualarmik_phase_baseline.py execution-infrastructure refinements from the Walker S2 grasping diagnosis: servo phases can now refresh the coordinate transform from the live torso_link each tick, position convergence is logged and gated using the point-B world metric when the point-B proxy is available, and DualArmIK phase logs now report the effective IK parameters after per-phase overrides
- Added FAR outboard waypoint steering before the low-side prepare phase: far_outboard_transition is inserted only for the FAR/world_y policy, with --far-outboard-transition-offset default 0.12 m and --far-outboard-transition-clearance default 0.08 m; the FAR sequence remains strictly branched as far_outboard_transition -> far_prepare_low_side_approach -> far_align_B_over_object_xy -> far_lower_B_world_z before close
- Added FAR null-space/posture knobs without changing region thresholds or candidate gates: --far-null-weight default 0.08 applies only to FAR servo IK solves, --far-outboard-shoulder-roll-bias default 0.35 biases the neutral arm posture outward, and --far-ab-downward-slant-deg default is now 8.0 deg to keep the side approach shallower
- Made the pre-close point-B gate mandatory for FAR with --pre-close-point-b-tolerance default 0.005 m; FAR now fails before gripper close if actual point B is not within 5 mm of the contact mark, preserving the existing vertical close gate behavior with default --vertical-close-point-b-tolerance now 0.005 m
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Vertical runtime command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --log-suffix vertical_pointB_live_torso_test; result remained target_region=mid and failed at mid_descend_world_z_keep_AB_vertical, with live torso transform refresh active/no failures and final point-B world error approximately 0.1153 m
- FAR runtime command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --log-suffix far_outboard_nullspace_live_torso_test2; result target_region=far, selected_motion_policy=far_low_side_B_driven, selected FAR world-Y preset unchanged, outboard/prepare/XY-align/Z-lower phases all met servo gates, effective DualArmIK null_weight logged as 0.08 for FAR phases, then mandatory pre-close gate correctly blocked close with point_B_error_before_close_m approximately 0.0882 m versus 0.005 m tolerance
- Remaining runtime issue: the infrastructure now prevents unsafe closing, but FAR point B still does not physically reach the final contact mark before close, and MID vertical Point-B/finger-midpoint convergence remains unresolved


- Implemented semi-closed-loop DualArmIK refresh cadence in scripts/task1_dualarmik_phase_baseline.py without changing scene building, target selection, region thresholds, candidate validation, EE-frame compensation, or the phase machine
- The central _execute_dualarmik_servo_phase now solves DualArmIK into a cached q_goal, tracks that q_goal every simulation tick with the existing blend/step limiter, and only refreshes live target geometry / torso coordinate transform / IK solution at the configured cadence; disabling the mode restores the legacy solve-every-servo-tick behavior
- Added CLI controls: --ik-refresh-enable/--no-ik-refresh-enable default enabled, --ik-refresh-period default 12 ticks, and --ik-refresh-drift-threshold default 0.0 m (disabled unless set positive); phase logs now include periodic_ik_refresh_active, ik_refresh_period_ticks, ik_refresh_drift_threshold_m, ik_refresh_count, ik_refresh_ticks, ik_refresh_reasons, ik_refresh_events, target_pose_evaluation_count, target_drift_check_count, and q_goal_update_count
- Official baseline influence: Ubtech_sim/source/RobotArticulation.py solves DualArmIK from synced Isaac joint state and smooths the outgoing joint positions; this patch keeps the same official DualArmIK call path but reduces solve cadence and reuses the current servo blend/step limiter between q_goal refreshes
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --ik-refresh-period 12 --ik-refresh-drift-threshold 0.0 --log-suffix ik_refresh_period12_test
- Runtime result: target_region=far, selected_motion_policy=far_low_side_B_driven, selected_orientation_preset_label remained the corrected world-Y palm-down preset, and the run still stopped at the mandatory FAR pre-close gate rather than closing with point_B_error_before_close_m approximately 0.0877 m
- Periodic refresh diagnostics from that run: far_outboard_transition refreshed IK at ticks [0, 12, 24] (3 solves over 26 executed ticks), far_prepare_low_side_approach at [0, 12] (2 solves over 25 ticks), far_align_B_over_object_xy at [0, 12] (2 solves over 11 executed ticks before success), and far_lower_B_world_z at [0] (1 solve over 6 ticks); all had zero live coordinate refresh failures
- Remaining uncertainty: headless runtime confirms the solve cadence and gate behavior, but GUI observation is still needed to judge visible smoothness; the existing FAR Point-B contact miss remains a geometry/reach issue rather than an IK cadence issue

- Added focused real grasp-center diagnostics to scripts/task1_dualarmik_phase_baseline.py for the current small grasp-center proxy bias diagnosis: pre-close now resolves the active hand's finger1_link/finger2_link midpoint as real_grasp_center_world, logs point_B_proxy_world, real_grasp_center_world, proxy_to_real_grasp_center_delta_world, proxy_to_real_grasp_center_delta_local, component finger positions, and close_critical_error_before_close_m
- The mandatory FAR and vertical pre-close gates now use real_grasp_center_world as the close-critical metric when both finger links resolve, falling back to the existing point-B proxy only if the live finger midpoint cannot be resolved; motion policy, region thresholds, candidate gates, EE-frame compensation, and FAR/MID phase structure were not changed
- Small nearby runtime fix: _command_gripper_phase now accepts the coord_transform_refresh_fn keyword already passed by close/release calls and logs the optional refresh result, avoiding a TypeError once a pre-close gate passes
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py and python scripts/task1_dualarmik_phase_baseline.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --ik-refresh-period 12 --log-suffix real_grasp_center_bias_test
- Runtime result: target_region=far, selected_motion_policy=far_low_side_B_driven, selected_orientation_preset_label=right_world_y_approach_pos_y_yaw_plus_quarter_ab_roll_minus_quarter_palm_down_test; run failed at far_contact_not_reached_before_close as intended by the mandatory gate, now using close_critical_metric=real_grasp_center_world
- New diagnostics from that run: point_B_proxy_world=[0.8178210, 0.3098236, 1.0856105], real_grasp_center_world=[0.8208409, 0.3586808, 1.1071487], proxy_to_real_grasp_center_delta_world=[0.0030199, 0.0488571, 0.0215382], delta norm approximately 0.05348 m, proxy_to_real_grasp_center_delta_local=[-0.0217962, -0.0018732, 0.0488001], point_B_error_before_close_m approximately 0.08773, and real_grasp_center_error_before_close_m approximately 0.10754
- Interpretation: this single run shows the proxy-to-real grasp-center offset is measurable and substantial, not yet proven stable across seeds/targets; collect several more logs before adding a calibrated offset path

- Created no-gate diagnostic duplicate scripts/task1_dualarmik_phase_nogate.py from scripts/task1_dualarmik_phase_baseline.py, leaving the active baseline unchanged
- In the duplicate only, SCRIPT_NAME/LOG_STEM now write to task1_dualarmik_phase_nogate logs, NO_GATE_MODE is enabled, and _fail bypasses all non-setup gate failures while preserving fatal scene/asset/target setup failures: scene_build_failed, no_target_parts_found, target_index_out_of_range
- The duplicate forces pregrasp candidate continuation by selecting the best available candidate when strict candidate acceptance fails, logging selection_mode=nogate_best_available_candidate_forced_acceptance and no_gate_candidate_forced_acceptance=true
- Motion/contact/result gates now log and continue through failed startup pose, gripper open, pregrasp/align/descend, pre-close, close, micro-lift, lift, carry, place, release, retreat, and final settle/scoring gates; bypassed failures are recorded in no_gate_bypassed_failures
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help

- Added no-gate-only pre-close mesh touch probing to scripts/task1_dualarmik_phase_nogate.py: before closing the gripper, the duplicate now lowers the close-critical mesh metric, preferring the live finger1/finger2 midpoint real_grasp_center_world when available, toward the selected object's WORLD XY and support/table height
- The no-gate touch probe stops on selected-object bbox overlap, selected-object motion, table/support height, or possible contact/IK stall; it checks DualArmIK on each step and runs a short same-Z XY position-fix phase before close if the close-critical metric is not within the configured XY tolerance
- Added no-gate CLI knobs for the probe: --nogate-preclose-touch-enable/--no-nogate-preclose-touch-enable, --nogate-touch-max-ticks, --nogate-touch-fix-ticks, --nogate-touch-step-z, --nogate-touch-gap-above-table, --nogate-touch-xy-tolerance, --nogate-touch-object-expand, --nogate-touch-object-motion-threshold, and --nogate-touch-stall-tolerance
- Added phase logging for nogate_preclose_mesh_touch_probe including touch_detected, touch_reason, ik_right_before_close, latest_ik_success, fix_position_attempts, final_close_metric_world, final_xy_error_to_target_m, final_z_gap_above_table_m, and trace samples; the post-probe pre-close gate is logged under nogate_preclose_touch_probe while no-gate still continues through failures
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 1 --ik-refresh-period 12 --log-suffix nogate_touch_probe_smoke
- Runtime result: target_region=far, selected_motion_policy=far_low_side_B_driven, no-gate bypassed far_contact_not_reached_before_close/object_not_lifted/dropped_during_lift, and nogate_preclose_mesh_touch_probe reported touch_detected=true with touch_reason=possible_contact_or_ik_stall at tick 7; IK before close was true and no XY fix was needed
- Important caveat from that run: final close-critical XY error improved to approximately 0.003 m, but final_z_gap_above_table_m was still approximately 0.102 m, so headless logs show the probe stopped on a possible contact/stall proxy rather than confirmed physical table/object touch; GUI/runtime follow-up should focus on why the descent stalls high

- Tightened scripts/task1_dualarmik_phase_nogate.py no-gate pre-close behavior so the gripper no longer closes after an unconfirmed stall or selected-object motion alone: by default --nogate-require-table-touch-before-close is true, so safe_to_close now requires confirmed table touch, successful latest IK solve, and selected-object center projected between the active finger1_link/finger2_link segment
- The no-gate touch probe now continues lowering through possible contact/IK stall instead of treating that stall as confirmed touch, uses a larger default table-touch budget (--nogate-touch-max-ticks 360 and --nogate-post-touch-reposition-ticks 120), and forces at least one post-touch IK reposition/check before any close decision
- Added no-gate capture diagnostics for object_between_fingers_before_close, table_touch_confirmed, require_table_touch_before_close, touch_requirement_met, close_block_reason, finger segment projection fraction, object distance to the finger segment, and capture distance threshold; close_gripper is skipped with nogate_close_blocked_by_touch_probe=true when safe_to_close is false
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 1 --ik-refresh-period 12 --log-suffix nogate_require_table_before_close_check
- Runtime result: target_region=far, the probe detected selected_target_object_motion_detected and the object-between-fingers estimate became true, but table_touch_confirmed=false with final_z_gap_above_table_m approximately 0.062 m; safe_to_close=false, close_block_reason=no_confirmed_table_touch, and close_gripper was skipped instead of closing the mesh
- Remaining issue exposed by the stricter no-gate run: the hand still does not physically reach table height in headless execution before close; next work should focus on why the IK/servo descent stalls about 6 cm above the table, not on allowing the gripper to close earlier

- Added scripts/task1_dualarmik_phase_nogate.py no-gate post-close slow-lift hold phase: after a non-skipped close_gripper, nogate_post_close_slow_lift_hold_grip solves an IK target a small distance upward in WORLD Z, moves with conservative joint step limits, reissues closed finger position targets every tick, and reapplies gripper_hold_effort every tick
- Added CLI knobs for the phase: --nogate-post-close-slow-lift-enable/--no-nogate-post-close-slow-lift-enable, --nogate-post-close-slow-lift-height default 0.010 m, --nogate-post-close-slow-lift-ticks default 90, --nogate-post-close-slow-lift-blend default 0.12, --nogate-post-close-slow-lift-max-step-norm default 0.015, and --nogate-post-close-slow-lift-max-abs-joint-step default 0.008
- The phase is skipped when close_gripper is skipped or blocked by the no-gate touch/capture safety rule; the default safe-close rule still requires confirmed table touch before close unless --no-nogate-require-table-touch-before-close is explicitly used for diagnostics
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command used the diagnostic override to exercise the new post-close phase: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 1 --ik-refresh-period 12 --no-nogate-require-table-touch-before-close --log-suffix nogate_post_close_slow_lift_exec_check
- Runtime result: close_gripper was not skipped, nogate_post_close_slow_lift_hold_grip ran, DualArmIK initial solve succeeded, closed targets were reissued each tick, gripper effort was applied each tick with R_finger1_joint/R_finger2_joint effort 100.0, the slow-lift target height was 0.010 m, and the phase met its loose lift tolerance after one tick with final_error_m approximately 0.0091
- Remaining caveat: this validation used --no-nogate-require-table-touch-before-close only to exercise the new slow-lift path; with the default stricter no-gate rule, close still remains blocked until table touch is confirmed and the selected object is between finger1_link/finger2_link

- Added a focused vertical-only lateral bias correction to scripts/task1_dualarmik_phase_baseline.py and scripts/task1_dualarmik_phase_nogate.py after GUI showed right-arm vertical grasps biased left and left-arm vertical grasps biased right: new --vertical-arm-lateral-bias-correction defaults to 0.04 m, applies -robot-base-Y for the right arm and +robot-base-Y for the left arm, and shifts the vertical finger-midpoint XY reference target before candidate evaluation, live descend feedback, and pre-close gating
- The correction is logged as vertical_uncorrected_object_world_xy_target, vertical_xy_reference_target_xy_world, vertical_arm_lateral_bias_correction_m, vertical_arm_lateral_bias_correction_base_y_m, vertical_arm_lateral_bias_correction_world, and vertical_arm_lateral_bias_correction_rule; FAR/world_y behavior and region thresholds were not changed
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_baseline.py scripts/task1_dualarmik_phase_nogate.py, python scripts/task1_dualarmik_phase_baseline.py --help, and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 342634456 --target-index 2 --ik-refresh-period 12 --log-suffix vertical_lateral_bias_correction_test
- Runtime result: target_region=mid, selected_approach_family=z_approach, selected_orientation_preset_label=right_z_approach_straight, correction magnitude 0.04 m applied with base_y=-0.04 for the right arm; in that run robot-base -Y mapped to approximately world +X [0.0400, -0.0000003, 0.0], shifting selected_vertical_xy_reference_target_xy_world from uncorrected [0.8167, 0.1986] to corrected [0.8567, 0.1986]. The run still failed later at mid_descend_world_z_keep_AB_vertical, so GUI verification is still needed for grasp centering.

- Applied no-gate-only vertical descent smoothing in scripts/task1_dualarmik_phase_nogate.py: mid/near vertical descend now uses a live target function that ramps point-B target Z downward by --nogate-vertical-descend-step-z, default 0.003 m per IK refresh, while keeping the live finger-midpoint XY reference locked to the corrected object XY target
- Added no-gate vertical controls --nogate-vertical-continuous-ik-descend/--no-nogate-vertical-continuous-ik-descend and --nogate-vertical-descend-step-z; when enabled, the vertical descend phase forces IK refresh period to 1 tick via per-phase overrides and uses a tighter internal servo tolerance to avoid stopping early on an intermediate ramp target
- Added phase diagnostics for nogate_vertical_continuous_ik_descend, nogate_vertical_descend_step_z_m, nogate_vertical_ik_refresh_period_ticks, vertical_descend_servo_tolerance_m, vertical_descend_state, commanded_target_z_world, z_remaining_to_contact_m, and world_z_only_descent_target in vertical_xy_reference_feedback_samples
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 342634456 --target-index 2 --ik-refresh-period 12 --log-suffix nogate_vertical_z_ramp_to_contact_test
- Runtime result: target_region=mid, selected_approach_family=z_approach, vertical descend logged ik_refresh_period_ticks=1 with 361 IK refreshes / q_goal updates, and the commanded Z ramp reached the final contact Z exactly (final_z_reached_by_command=true, z_remaining_to_contact_m=0.0); actual final point-B error remained high at approximately 0.094 m in headless no-gate, so the commanded policy is now continuous/gradual but physical convergence still needs GUI inspection and likely IK/reach tuning.

- Updated scripts/task1_dualarmik_phase_nogate.py no-gate vertical descent speed/cadence per the slow-down request: default --nogate-vertical-ik-refresh-period is now 10 ticks and default --nogate-vertical-descend-step-z is now 0.010 m per IK refresh, giving an effective commanded Z target rate of 0.001 m/frame
- Added a servo-phase completion condition used by no-gate continuous vertical descend so the phase cannot stop after reaching the first intermediate ramp target; it now continues until the commanded Z ramp reaches the final vertical contact mark
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 342634456 --target-index 2 --ik-refresh-period 12 --log-suffix nogate_vertical_refresh10_slow_completion_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, vertical descend logged ik_refresh_period_ticks=10 with refresh ticks [0, 10, 20, ..., 360], ik_refresh_count=37, q_goal_update_count=37, nogate_vertical_descend_step_z_m=0.01, and nogate_vertical_effective_z_step_per_frame_m=0.001; the commanded Z ramp reached the final contact Z, but physical final point-B error remained high at approximately 0.131 m, so the slower cadence is implemented while convergence still needs runtime/IK tuning

- Added no-gate-only near-contact DLS handoff in scripts/task1_dualarmik_phase_nogate.py: during vertical descend, if the live close-critical gripper metric reaches within 0.04 m of the selected object's bbox surface, the DualArmIK descend stops early and nogate_near_contact_measured_dls_finish runs the measured finite-difference DLS position controller adapted from task1_single_target_random_scene_baseline
- The adapted close DLS uses the active finger1/finger2 midpoint as the preferred close metric, falls back to point_B_proxy_world if needed, and targets the existing vertical contact mark; reduced stop-go defaults are settle_steps=1, hold_steps=0, max_step_norm=0.012, max_abs_joint_step=0.006, blend=0.45, max_iters=18, and stop_tolerance=0.004 m
- Added CLI knobs --nogate-close-dls-enable, --nogate-close-dls-switch-distance, --nogate-close-dls-max-iters, --nogate-close-dls-eps, --nogate-close-dls-damping, --nogate-close-dls-max-step, --nogate-close-dls-max-abs-joint-step, --nogate-close-dls-blend, --nogate-close-dls-settle-steps, --nogate-close-dls-hold-steps, --nogate-close-dls-stop-tolerance, and --nogate-close-dls-posture-gain; main baseline remains unchanged
- Static checks passed: python -m py_compile scripts/task1_dualarmik_phase_nogate.py and python scripts/task1_dualarmik_phase_nogate.py --help
- Runtime validation command: /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_nogate.py --seed 342634456 --target-index 2 --ik-refresh-period 12 --log-suffix nogate_4cm_dls_bbox_handoff_test
- Runtime result: target_region=mid, selected_motion_policy=mid_vertical_Z_descend, vertical descend early-stopped at tick 101 with close_dls_handoff_reason=gripper_metric_within_switch_distance_of_object_bbox, real_grasp_center_world distance to object bbox approximately 0.03976 m, and distance to contact mark approximately 0.04246 m; the measured DLS phase ran with settle_steps=1/hold_steps=0 and reduced the close metric from [0.8727, 0.2066, 1.0899] to [0.8726, 0.2037, 1.0669], ending at final_error_m approximately 0.0476 versus 0.004 tolerance
- Remaining issue from that run: the following no-gate touch probe still failed safe-to-close because table_touch_confirmed=false with final_z_gap_above_table_m approximately 0.0411 m, although object_between_fingers=true; the handoff mechanism works, but DLS/contact descent still needs tuning to reach table/contact height reliably

## 2026-04-18
- Initialized plan-driven execution for the Task 1 hybrid geometric grasp project by creating `CURRENT_PLAN.md`.
- Phase 0 source lock selected `scripts/task1_dualarmik_phase_baseline.py` as the source to fork from because it has the most current official Task 1 SceneBuilder setup, Walker S2 robot loading, official DualArmIK path, deterministic gated phase structure, and detailed logging.
- Recorded in `CURRENT_PLAN.md` that Phase 0 is active, all later phases are pending, and the hybrid planner must not be implemented until the plan explicitly advances.
- Updated `AGENTS.md` with persistent rules requiring future Codex tasks to read `AGENTS.md`, `PROJECT_CONTEXT.md`, `CURRENT_PLAN.md`, and `TASK_LOG.md` before planning or editing; treating `CURRENT_PLAN.md` as the active source of truth; limiting implementation to the current phase; and updating logs/context after implementation runs.
- Updated `PROJECT_CONTEXT.md` to mention the new plan-driven workflow and the Phase 0 selected source.
- Test result: documentation-only change; no Isaac runtime or lightweight code test was run.
- Next step: after user authorizes Phase 1, copy the selected baseline into a new Task 1 hybrid script and add the minimal `scene_state`-only skeleton without Thinker.

## 2026-04-19
- Implemented Phase 1 only: minimal hybrid skeleton for Task 1 in `scripts/task1_hybrid_geometric_phase1.py`.
- Source file used: `scripts/task1_dualarmik_phase_baseline.py`; the source file was copied and left unchanged.
- New script keeps the baseline SceneBuilder scene setup, Walker S2 robot loading, articulation setup, official DualArmIK backend, phase-machine execution, gripper effort behavior, and object-centric scoring flow.
- Added table-frame convention in the new script: origin is the near-left tabletop corner from the robot viewpoint, +x_table runs robot-left to robot-right along the near edge, +y_table runs from robot toward the far table side, and +z_table points upward.
- Added table-unit convention: `TABLE_UNIT_M = 0.035`, so one table unit is 3.5 cm; object and candidate table coordinates are logged in meters and units.
- Added Phase 1 helper functions: `build_or_resolve_table_frame(...)`, `world_to_table(...)`, `table_to_world(...)`, `get_object_info_in_table_frame(...)`, `generate_approach_candidates_for_object(...)`, `fast_score_candidate(...)`, `select_best_candidate(...)`, and extended `resolve_real_grasp_center_world(...)` fallback logging in the copied script.
- Added `scene_state` object_info logging with object id, class name, world/table centers, table units, approximate size, bbox in world/table coordinates, yaw fallback, and `perception_source="scene_state"`.
- Added minimal finite candidate generation, currently 4 candidates for a forced arm or 8 for auto arm, with deterministic arm/preset/yaw records, pregrasp table coordinates, object grasp center table coordinates, coarse workspace/width sanity, score terms, and selected candidate logging.
- Existing motion backend is preserved: the selected Phase 1 candidate chooses the arm and logs the planned approach, then the copied baseline continues through existing pregrasp, descend, close, lift, place/release phases using DualArmIK.
- Intentionally not implemented in this run: full geometric hardening, rich alignment/width/symmetry/clearance/asymmetry filters, robust local closed-loop final descent, retry with next candidate, Thinker integration, YOLO/provider abstraction, and broad architecture refactors.
- Static checks passed: `python3 -m py_compile scripts/task1_hybrid_geometric_phase1.py` and `python3 scripts/task1_hybrid_geometric_phase1.py --help`.
- Runtime limitation: no Isaac Sim run was executed in this edit pass; Linux runtime validation is still needed to confirm the new table-frame logs and candidate selection in simulation.
- `CURRENT_PLAN.md` was not advanced because Phase 1 runtime exit criteria still need Linux validation that the new script runs without breaking baseline infrastructure.
- Created Vietnamese controlled-translation versions of the two generated inventory reports:
  `docs/baseline_full_inventory_vi.txt` and `docs/hrc2026_full_inventory_vi.txt`.
- Translation scope: section titles, file-block labels, common generated explanations, and summary language were localized to Vietnamese while preserving paths, identifiers, imports, CLI flags, log keys, and technical snippets where exact wording matters for grep/debugging.
- Validation result: the baseline Vietnamese report preserves 153 `FILE:` blocks and 153 relative-path blocks; the hrc2026 Vietnamese report preserves 3463 `FILE:` blocks and 3463 relative-path blocks.
- Ran Phase 1 Isaac runtime validation only for `scripts/task1_hybrid_geometric_phase1.py`; no Phase 2 planner hardening, Thinker, YOLO, or broad refactor was implemented.
- Static validation commands passed:
  `python3 -m py_compile scripts/task1_hybrid_geometric_phase1.py`
  and `python3 scripts/task1_hybrid_geometric_phase1.py --help`.
- Isaac auto smoke command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase1.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase1_validate_auto_smoke`.
- Auto smoke result: Isaac exited with code 0 and the script built the SceneBuilder table/parts/Walker S2 scene, generated object_info, selected a Phase 1 candidate, initialized the DualArmIK flow, then failed safely at the inherited mandatory gate `far_contact_not_reached_before_close`.
- Auto smoke log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase1_20260418T180558Z_phase1_validate_auto_smoke.log`.
- Table-frame runtime evidence from the auto smoke: `mapping_mode=axis_aligned_simplified`, `axis_aligned_with_world_xy=true`, `origin_world=[-0.04238909009209291, 0.011159473016877532, 1.000136132521064]`, `x_axis_world=[1, 0, 0]`, `y_axis_world=[0, 1, 0]`, and `z_axis_world=[0, 0, 1]`.
- Table convention validation: with robot forward as world +Y and robot-left as world -X, the resolved origin is the near-left tabletop corner, +x_table is world +X left-to-right across the near edge, +y_table is world +Y away from the robot, and table coordinates are sign-correct.
- Object table coordinate evidence: selected `/Replicator/Ref_Xform_03` had `center_world=[0.7058081623056034, 0.3744424401475328, 1.0437566159862197]`, `center_table_m=[0.7481972523976963, 0.3632829671306553, 0.04362048346515568]`, and `center_table_unit=[21.377064354219893, 10.37951334659015, 1.2462995275758764]`; values are positive and within the table extents.
- Auto candidate evidence: Phase 1 generated 8 valid candidates, selected `phase1_world_y_approach_right_yaw_-10`, selected arm `right`, selected approach `world_y_approach`, `pregrasp_table_m=[0.7481972523976963, 0.24328296713065528, 0.27855454404230123]`, and `object_grasp_center_table_m=[0.7481972523976963, 0.3632829671306553, 0.04362048346515568]`.
- Candidate scoring evidence: auto mode preferred right arm for this object, applied no arm-side penalty to right candidates, applied a 0.15 arm-side penalty to left candidates, and deterministically ranked `-10 deg` before `+10 deg` by candidate index tie-break.
- Real grasp-center evidence: the helper resolved `source=finger_link_pair_midpoint` with no fallback in the auto run and logged `real_grasp_center_table_m=[1.0254210761025477, 0.3650220376384708, 0.35582203015259073]`.
- Forced-right validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase1.py --seed 1 --target-selection-policy index --target-index 2 --arm right --servo-max-ticks 1 --servo-carry-ticks 1 --gripper-steps 1 --settle-steps 1 --log-suffix phase1_validate_forced_right_selection`.
- Forced-right result: Isaac exited with code 0, generated exactly 4 valid right-arm Phase 1 candidates, selected `phase1_world_y_approach_right_yaw_-10`, and then failed at `far_outboard_transition_failed` because the servo tick budget was intentionally reduced to validate selection without tuning motion.
- Forced-right log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase1_20260418T180715Z_phase1_validate_forced_right_selection.log`.
- Forced-left validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase1.py --seed 1 --target-selection-policy index --target-index 2 --arm left --servo-max-ticks 1 --servo-carry-ticks 1 --gripper-steps 1 --settle-steps 1 --log-suffix phase1_validate_forced_left_selection`.
- Forced-left result: Isaac exited with code 0, generated exactly 4 valid left-arm Phase 1 candidates, selected `phase1_world_y_approach_left_yaw_-10`, preserved the requested forced arm, and then failed at `far_outboard_transition_failed` due to the intentionally reduced servo tick budget.
- Forced-left log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase1_20260418T180736Z_phase1_validate_forced_left_selection.log`.
- Full pick/place attempt command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase1.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --continue-after-lift --log-suffix phase1_validate_auto_full_attempt`.
- Full attempt result: Isaac exited with code 0 but did not reach pick/place completion; it failed before close at `far_contact_not_reached_before_close`, so carry/place/release could not be validated in this run.
- Full attempt log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase1_20260418T180802Z_phase1_validate_auto_full_attempt.log`.
- Baseline comparison command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_dualarmik_phase_baseline.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase1_validate_baseline_compare`.
- Baseline comparison result: original `scripts/task1_dualarmik_phase_baseline.py` also failed at `far_contact_not_reached_before_close` for the same seed/target, indicating the observed pick/place blocker is inherited DualArmIK/contact geometry behavior rather than a Phase 1 table-frame or candidate-selection regression.
- Baseline comparison log:
  `/home/edward/hrc-runtime/logs/task1_dualarmik_phase_baseline_20260418T180830Z_phase1_validate_baseline_compare.log`.
- Observed Phase 1 issues: no direct evidence of x/y sign inversion, near-left origin error, left/right mirroring error in forced-arm candidate generation, incoherent candidate scoring, or real grasp-center fallback/logging failure.
- Minimal fixes applied: none. Code was not modified because runtime evidence did not identify a Phase 1-specific bug.
- Deferred intentionally: FAR contact reaching, robust descent/contact, retry with next candidate, richer geometric filters, and pick/place success tuning remain later-phase work and were not implemented in this validation pass.
- Phase 1 exit criterion status: the narrow runtime infrastructure criterion is satisfied because the new script runs in Isaac through scene build, table-frame resolution, scene_state object_info, finite candidate selection, real grasp-center logging, and the inherited DualArmIK phase flow without breaking baseline infrastructure; however, full pick/place success is not validated, so `CURRENT_PLAN.md` was not advanced in this conservative validation run.

- Implemented Phase 2 only in a new copied script: `scripts/task1_hybrid_geometric_phase2.py`.
- Source file used: `scripts/task1_hybrid_geometric_phase1.py`; the validated Phase 1 script was copied and left unchanged.
- Phase 2 purpose: address the inherited pre-close/contact blocker with contact/descent hardening while preserving the Phase 1 table frame, scene_state object_info, finite candidate generation, SceneBuilder setup, Walker S2 loading, DualArmIK backend, phase-machine structure, gripper effort behavior, and place/release flow.
- Added object grasp-frame estimation via `estimate_object_grasp_frame(...)`, using the scene_state table-frame bbox to estimate grasp center, closing axis, lateral axis, vertical axis, object width on closing axis, nominal contact points, and bottom/top table clearance values.
- Added fast vector geometric filtering with `predict_early_contact_asymmetry(...)`, `estimate_table_clearance_margin(...)`, `fast_geometric_grasp_filter(...)`, and `select_best_phase2_candidate(...)`.
- Geometric filter metrics logged per candidate: alignment error, lateral symmetry error, predicted early-contact asymmetry, width compatibility, table clearance margin, mandatory pass flags, violation score, selected candidate, and explicit least-bad warning if no candidate passes every mandatory check.
- Added Phase 2 local final descent with `final_descent_local_ik(...)`; it measures the real finger midpoint grasp center when available, falls back to point-B only if needed, locks the selected target frame, clamps XY/Z/yaw incremental steps, enforces monotonic downward Z commands, and continues using the existing DualArmIK servo infrastructure.
- Added multi-condition close gate with `evaluate_close_gate(...)`; close now requires geometric mandatory pass, real grasp-center error, lateral symmetry, predicted contact asymmetry, table clearance, orientation error, and recent XY drift stability before gripper close.
- Added gap-aware close helpers with `compute_target_gap(...)` and `execute_two_stage_close(...)`; Stage A moves toward a geometry-advised gap command and Stage B reissues the official close target with retention effort.
- Added short lift verification via `verify_short_lift(...)`; after close the script performs a bounded short lift and verifies the object center follows before continuing to the full lift/carry/place flow.
- Added deterministic recovery logging via `recover_and_retry(...)`; on Phase 2 close-gate, close, or short-lift failure the script reopens the gripper, records the current bad candidate, records remaining passed candidates if any, and logs the retry reason. The existing linear phase-machine is preserved, so full automatic re-entry into a second candidate remains a known limitation for runtime follow-up.
- New exposed Phase 2 parameters include:
  `--phase2-alignment-error-max`, `--phase2-symmetry-error-max`, `--phase2-contact-asymmetry-max`, `--phase2-table-clearance-min`, `--phase2-width-min`, `--phase2-width-max`, `--phase2-final-descent-enable`, `--phase2-final-descent-ticks`, `--phase2-descent-xy-step`, `--phase2-descent-z-step`, `--phase2-descent-yaw-step`, `--phase2-close-real-center-tolerance`, `--phase2-close-orientation-tolerance`, `--phase2-close-xy-drift-max`, `--phase2-min-target-gap`, `--phase2-target-gap-margin`, `--phase2-close-stage-a-fraction`, `--phase2-retention-steps`, `--phase2-short-lift-height`, `--phase2-short-lift-min-delta`, `--phase2-short-lift-ticks`, and `--phase2-max-retries`.
- Default Phase 2 thresholds: alignment error max 0.75 rad, symmetry error max 0.030 m, predicted contact asymmetry max 0.030 m, table clearance min 0.002 m, width range 0.010-0.090 m, close real-center tolerance 0.030 m, close orientation tolerance 0.35 rad, close XY drift max 0.018 m, descent XY step 0.006 m, descent Z step 0.003 m, yaw step 0.04 rad, final descent ticks 120, retention steps 18, short-lift min delta 0.006 m.
- Static checks passed:
  `python3 -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  and `python3 scripts/task1_hybrid_geometric_phase2.py --help`.
- Isaac runtime commands to run next on Linux:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_auto_contact_hardening_smoke`
  and, if that reaches close/lift,
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --continue-after-lift --log-suffix phase2_auto_full_attempt`.
- Additional forced-arm validation commands to run next:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm right --log-suffix phase2_forced_right_contact_hardening`
  and
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm left --log-suffix phase2_forced_left_contact_hardening`.
- Known limitations: grasp frame is bbox/table-axis based, not true mesh OBB; close gap to finger joint mapping is logged as geometry-advisory because the official gripper command uses same-sign joint targets; recovery currently logs and prepares a next candidate but does not yet re-enter the entire pregrasp/descent phase loop; thresholds are first-pass conservative and require Isaac validation before marking Phase 2 complete.
- Intentionally not implemented in this run: Thinker, YOLO/provider abstraction, heavy force-closure optimization, mesh contact modeling, broad architecture rewrite, or Phase 3+ changes.
- `CURRENT_PLAN.md` was not advanced because Phase 2 has not yet been Isaac-runtime validated.
- Added `scripts/watch_task1_phase2_gui.sh` as a minimal GUI launcher for `scripts/task1_hybrid_geometric_phase2.py`.
- The launcher follows the existing GUI script pattern: it uses `ISAAC_SIM_PYTHON` or `ISAAC_SIM_ROOT`, avoids hardcoded local Isaac paths, defaults to seed 1 / target-index 2 / arm auto, passes `--gui`, keeps the GUI open by default, and accepts additional script flags through trailing CLI arguments.
- GUI launcher overrides: `SEED`, `TARGET_SELECTION_POLICY`, `TARGET_INDEX`, `ARM`, `LOG_SUFFIX`, `CONTINUE_AFTER_LIFT=1`, and `NO_HOLD_OPEN=1`.
- Static shell check passed: `bash -n scripts/watch_task1_phase2_gui.sh`.

- Applied a focused Phase 2 contact-reference semantics fix in `scripts/task1_hybrid_geometric_phase2.py`; `scripts/task1_hybrid_geometric_phase1.py` was not changed.
- Bug addressed: the close-critical `real_grasp_center_world` / vertical XY reference no longer silently means a palm-ish finger-link midpoint. The new preferred semantics are the midpoint between the two active fingertip-end references.
- Added explicit fingertip-end resolution layers:
  1. try actual fingertip/end/distal/contact frames or prims for each active finger using transform positions only,
  2. if not present, derive calibrated fingertip-end proxies from each finger link bbox distal face, using the active palm/pgc reference to choose the distal direction,
  3. only if both fingertip-end paths fail, fall back to the old finger-link midpoint path and mark the fallback in logs.
- Distinct references now logged separately: `point_B_proxy_world`, `real_grasp_center_world`, `fingertip_midpoint_world`, `fingertip_component_positions_world`, `legacy_link_midpoint_world`, `legacy_link_midpoint_to_fingertip_midpoint_delta_world`, and `point_B_proxy_to_fingertip_midpoint_delta_world`.
- Updated vertical reference semantics: `finger_midpoint` / `fingertip_midpoint` / related aliases now resolve through the fingertip-end midpoint path, not the old finger-link midpoint path.
- Updated Phase 2 final descent semantics: `final_descent_local_ik(...)` now computes a `contact_control_offset_local` from the resolved close-critical fingertip midpoint at phase start and commands that reference during local IK descent. The old point-B offset is retained for diagnostics only unless fingertip resolution fails.
- Updated point-B log wording so point-B is explicitly a legacy proxy, not the close-critical contact truth.
- Static validation passed:
  `python3 -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  and
  `python3 scripts/task1_hybrid_geometric_phase2.py --help`.
- Isaac vertical validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 342634456 --target-selection-policy index --target-index 1 --arm auto --log-suffix phase2_fingertip_vertical_target1_validate`.
- Isaac vertical result: simulation exited normally with `status=fail`, `failure_reason=phase2_close_gate_failed`; this was a mid/vertical case with `selected_motion_policy=mid_vertical_Z_descend`, `selected_orientation_preset_label=right_z_approach_straight`, `selected_vertical_xy_reference_mode=fingertip_end_midpoint`, and `selected_vertical_xy_reference_source=calibrated_finger_link_tip_proxy_pair_midpoint`.
- Vertical fingertip evidence: actual fingertip frames were not found (`actual_fingertip_frame_attempt.resolved=false`), so both fingers used `finger_link_bbox_distal_face_local_offset_proxy`; final-descent logs showed `contact_control_reference_source=calibrated_finger_link_tip_proxy_pair_midpoint`.
- Vertical delta evidence: logged `legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m` was about 0.0585 m at first final-descent sample and about 0.0490 m later; `point_B_proxy_to_fingertip_midpoint_delta_world_norm_m` was about 0.1098 m then about 0.0978 m, confirming the old proxy and link midpoint were materially different from the new close-critical fingertip midpoint.
- Vertical behavior evidence: close gate used `close_critical_metric=real_grasp_center_world`, not point-B; it failed safely with `real_grasp_center_error_m≈0.3381` and `orientation_error_rad≈0.5495`, so this run did not close based on a palm-centered proxy. Headless logs cannot fully prove GUI visual improvement, but they confirm the close-critical reference semantics changed as intended.
- Isaac far/horizontal validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_fingertip_far_validate`.
- Isaac far/horizontal result: simulation exited normally with `status=fail`, `failure_reason=phase2_close_gate_failed`; this was a far case with `selected_motion_policy=far_low_side_B_driven` and `selected_orientation_preset_label=right_world_y_approach_pos_y_yaw_plus_quarter_ab_roll_minus_quarter_palm_down_test`.
- Far/horizontal behavior evidence: the run reached `phase2_far_final_descent_local_ik` after FarXYAlignB and logged `contact_control_reference_source=calibrated_finger_link_tip_proxy_pair_midpoint`; it did not stop at FarXYAlignB as if the palm/point-B proxy were already final contact. It still failed the Phase 2 close gate with `real_grasp_center_error_m≈0.1817` and `geometric_filter_mandatory_pass=false`.
- Far/horizontal delta evidence: final-descent logs showed `legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m≈0.0441` and `point_B_proxy_to_fingertip_midpoint_delta_world_norm_m≈0.0955`.
- Additional diagnostic command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 342634456 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_fingertip_vertical_validate`.
- Diagnostic result: this target-index was actually `target_region=far` in the current Phase 2 scene and failed earlier at `pregrasp_failed` / `far_prepare_low_side_approach`, so it was not used as the vertical validation case.
- Logs produced:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T190809Z_phase2_fingertip_vertical_target1_validate.log`,
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T190702Z_phase2_fingertip_far_validate.log`,
  and
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T190630Z_phase2_fingertip_vertical_validate.log`.
- Remaining unresolved: true fingertip USD/articulation frames were not present under the searched names, so the current reliable reference is a calibrated distal bbox-face proxy. The robot still does not complete the pick/place flow; remaining failures are Phase 2 IK/contact convergence and candidate/filter tuning, not the original palm-centered contact-reference semantics bug.
- `CURRENT_PLAN.md` was not advanced; this was a local Phase 2 bugfix/validation run, not a phase transition.

- Applied a minimal Phase 2 vertical close-trigger fix in `scripts/task1_hybrid_geometric_phase2.py`; `scripts/task1_hybrid_geometric_phase1.py` and other planner phases were not changed.
- Added `evaluate_vertical_support_or_stall_close_fallback(...)` as a secondary close gate only for non-far vertical policies. The existing `evaluate_close_gate(...)` remains the primary gate and is evaluated first.
- Added final-descent support/stall logging to `final_descent_local_ik(...)`: per-sample `measured_support_gap_m` / `commanded_support_gap_m`, plus `latest_measured_support_gap_m`, `min_measured_support_gap_m`, `recent_z_progress_m`, `recent_z_motion_abs_m`, `stalled_in_z`, `z_stall_window_sample_count`, and `recent_xy_drift_m`.
- Added pre-close support-gap fields for `vertical_actual_close_critical_gap_above_support_m`, `vertical_actual_real_grasp_center_gap_above_support_m`, and `vertical_actual_point_B_gap_above_support_m`.
- Final close decision is now logged as `phase2_final_close_decision` and allows close only if the primary Phase 2 gate passes or the vertical fallback passes. Fallback use is explicitly logged with `allowed_by=vertical_support_or_stall_fallback`.
- New fallback thresholds/CLI knobs:
  `--phase2-vertical-fallback-close-enable` default true,
  `--phase2-vertical-fallback-support-gap-max` default 0.004 m,
  `--phase2-vertical-fallback-support-gap-min` default -0.020 m,
  `--phase2-vertical-fallback-recent-z-progress-max` default 0.0015 m,
  `--phase2-vertical-fallback-min-descent-samples` default 2,
  and `--phase2-vertical-fallback-orientation-tolerance` default 0.65 rad. The primary close orientation tolerance remains 0.35 rad.
- Static checks passed:
  `python3 -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  and
  `python3 scripts/task1_hybrid_geometric_phase2.py --help`.
- Initial vertical validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 342634456 --target-selection-policy index --target-index 1 --arm auto --log-suffix phase2_vertical_support_fallback_validate`.
- Initial vertical result: the fallback was evaluated but failed conservatively because `stalled_in_z_pass=false` and `orientation_error_pass=false`; logs showed support gap was already small but stall semantics used absolute Z motion rather than downward-progress only.
- Corrected the stall semantics so `stalled_in_z` uses lack of downward support-gap progress (`recent_z_progress_m`) rather than absolute Z motion. This handles upward bounce/settling at support as "no meaningful downward progress" while still requiring small support gap, XY stability, and orientation sanity.
- Revalidation command after stall fix:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 342634456 --target-selection-policy index --target-index 1 --arm auto --log-suffix phase2_vertical_support_fallback_validate_v2`.
- Revalidation result after stall fix: fallback still failed because only `orientation_error_pass=false`; the run logged `support_gap_small_pass=true`, `stalled_in_z_pass=true`, `recent_xy_drift_stable_pass=true`, `recent_z_progress_m=0.0`, and `selected_support_gap_m≈-0.00353`.
- Added fallback-local orientation sanity tolerance so the fallback does not reuse the stricter primary exact-pose tolerance. This keeps the primary gate unchanged but lets the fallback act as a support/stall close trigger when orientation is broadly sane.
- Final vertical validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 342634456 --target-selection-policy index --target-index 1 --arm auto --log-suffix phase2_vertical_support_fallback_validate_v3`.
- Final vertical result: close gate no longer blocked the vertical case. Primary gate still failed with `real_grasp_center_error_pass=false` and primary `orientation_error_pass=false`, but fallback passed with `support_gap_small_pass=true`, `stalled_in_z_pass=true`, `recent_xy_drift_stable_pass=true`, `orientation_error_pass=true`, `selected_support_gap_m≈-0.00353`, `recent_z_progress_m=0.0`, `orientation_error_rad≈0.5495`, and fallback orientation tolerance 0.65 rad. `phase2_final_close_decision.allowed_by=vertical_support_or_stall_fallback`.
- Final vertical run proceeded through `execute_two_stage_close(...)`; two-stage close reported `condition_met=true`, `stage_a_ok=true`, `stage_b_ok=true`, and `retention_status=ok`.
- Final vertical run still failed later at `object_not_lifted` because the selected object did not follow during `verify_short_lift(...)`. This is outside the requested close-trigger fix and remains a Phase 2 grasp/retention or candidate-quality issue.
- Final vertical log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T192545Z_phase2_vertical_support_fallback_validate_v3.log`.
- Far guard validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_vertical_support_fallback_far_guard_validate_v2`.
- Far guard result: fallback remained blocked for `selected_motion_policy=far_low_side_B_driven` with `vertical_motion_policy=false`, `support_gap_small_pass=false`, `z_stall_sample_count_pass=false`, and `stalled_in_z_pass=false`; final close decision did not use the fallback.
- Far guard log:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T192705Z_phase2_vertical_support_fallback_far_guard_validate_v2.log`.
- False-close risk observed in these headless runs: no evidence of a broad unconditional table-touch close. The fallback only fired in the vertical case when support gap was small, downward progress was stalled, XY drift was within tolerance, and fallback orientation sanity passed; it stayed disabled for far/horizontal approach.
- `CURRENT_PLAN.md` was not advanced; this was a local Phase 2 bugfix and validation run, not a phase transition.

- Applied the requested strict vertical-only close rule in `scripts/task1_hybrid_geometric_phase2.py`; Phase 1 and far/horizontal planner behavior were not intentionally changed.
- Added `PHASE2_VERTICAL_TIP_TABLE_Z_CLOSE_THRESHOLD_M = 0.0005` and CLI flag `--phase2-vertical-tip-table-z-close-threshold`.
- Added `ServoEarlyStop` and a `per_tick_monitor_fn` hook in `_execute_dualarmik_servo_phase(...)` so vertical descent can stop immediately before the next joint command when a close-critical monitor fires.
- Added vertical-only fingertip table-frame monitoring in `final_descent_local_ik(...)`: it resolves the active close-critical fingertip midpoint, converts it through `world_to_table(...)`, logs `current_vertical_tip_world`, `current_vertical_tip_table`, `current_vertical_tip_table_z_m`, `vertical_tip_stop_rule_triggered`, `vertical_tip_stop_rule_threshold_m`, and `vertical_tip_stop_rule_source`, and sets `vertical_tip_reached_table_z0` / `vertical_tip_close_stop_reason` when the threshold is reached.
- The monitored fingertip reference source is the existing active Phase 2 close-critical source priority: actual fingertip frames if available, calibrated finger-link distal tip proxy midpoint if actual frames are missing, and legacy finger-link midpoint only as fallback.
- Updated final close decision routing: when a non-far vertical descent reports `vertical_tip_reached_table_z0=true`, the script skips `evaluate_close_gate(...)` and `evaluate_vertical_support_or_stall_close_fallback(...)` as decisive blockers, calls `execute_two_stage_close(...)`, and logs `phase2_final_close_decision.allowed_by=vertical_tip_table_z_leq_zero_rule`.
- Static checks passed:
  `python3 -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  and
  `python3 scripts/task1_hybrid_geometric_phase2.py --help`.
- Initial seed 376 validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 376 --log-suffix phase2_vertical_tip_z0_seed376_validate`.
- Initial seed 376 result: vertical case `selected_motion_policy=mid_vertical_Z_descend`, but the monitor only sampled on target refresh before the per-tick hook was added; it did not close, with minimum observed `current_vertical_tip_table_z_m≈0.00459` and final close decision still blocked by the old gates.
- Per-tick seed 376 validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 376 --log-suffix phase2_vertical_tip_z0_seed376_validate_v2`.
- Per-tick seed 376 result: monitor was active with 38 samples, including 34 `per_tick_monitor` samples and 4 `target_pose_fn` samples; reference source was `calibrated_finger_link_tip_proxy_pair_midpoint`, threshold was `0.0005`, but the monitored table-frame z never reached the threshold (`min≈0.00459`, last≈0.01663), so `vertical_tip_reached_table_z0=false`, no early stop occurred, and the run still failed at `phase2_close_gate_failed`.
- Exact GUI-default seed 376 target-2 validation command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 376 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_vertical_tip_z0_seed376_target2_validate`.
- Exact GUI-default seed 376 target-2 result: same vertical policy and same outcome; 38 monitor samples, source `calibrated_finger_link_tip_proxy_pair_midpoint`, threshold `0.0005`, `min current_vertical_tip_table_z_m≈0.00459`, `vertical_tip_stop_rule_triggered=false`, `vertical_tip_reached_table_z0=false`, and no close at `tip_table_z≈0` because the tip never reached table-frame z zero in headless runtime.
- Far/horizontal guard command:
  `/usr/bin/timeout 900 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix phase2_vertical_tip_z0_far_guard`.
- Far/horizontal guard result: `selected_motion_policy=far_low_side_B_driven`; `vertical_tip_stop_rule_active=false`, `vertical_tip_stop_rule_samples=[]`, `per_tick_monitor_active=false` in `phase2_far_final_descent_local_ik`, and final close decision did not use the vertical tip rule.
- Logs produced:
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T193913Z_phase2_vertical_tip_z0_seed376_validate.log`,
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T194239Z_phase2_vertical_tip_z0_seed376_validate_v2.log`,
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T194352Z_phase2_vertical_tip_z0_seed376_target2_validate.log`,
  and
  `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T194451Z_phase2_vertical_tip_z0_far_guard.log`.
- Important validation conclusion: the strict rule is implemented and logged, but seed 376 did not close at `tip_table_z≈0` because the monitored fingertip proxy never reached table-frame z <= 0.0005 in headless validation. This run intentionally did not tune descent, thresholds, support/stall fallback, candidate filtering, or any Phase 3+ behavior.
- `CURRENT_PLAN.md` was not advanced; this was a local Phase 2 bugfix/validation run, and the observed remaining blocker is vertical descent/contact convergence rather than the new close-decision rule.

- Added a visual debug marker for the legacy proxy midpoint in `scripts/task1_hybrid_geometric_phase2.py` (script already contained this implementation from earlier patch set; no behavior logic changed).
- Marker details (Phase 2 debug only, no grasp/close logic change):
  - path: `/World/DebugProxyMiddlePoint`
  - helper: `_upsert_debug_marker(...)`
  - radius: `DEBUG_PROXY_MIDDLE_POINT_MARKER_RADIUS_M = 0.012`
  - color: `DEBUG_PROXY_MIDDLE_POINT_MARKER_COLOR = (1.0, 0.12, 0.00)` (strong red)
- Runtime initialization flow keeps marker definition tied to:
  - resolved arm + articulation acquired
  - `_resolve_finger_midpoint_reference_position(...)` with token `"phase2_debug_legacy_finger_link_midpoint"`
  - marker creation/update to that first resolved midpoint.
- Runtime update flow:
  - `_update_proxy_middle_point_debug_marker(...)` is called from `final_descent_local_ik(...)` (during descent sampling / target pose updates).
- Logging added/verified:
  - `proxy_middle_point_marker_path`, `proxy_middle_point_source`, `proxy_middle_point_component_positions_world`, `proxy_middle_point_world`, `proxy_middle_point_fallback_used`.
- Validation run:
  - `/usr/bin/timeout 300 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --log-suffix marker_debug_test`
  - log: `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T195658Z_marker_debug_test.log`
  - marker observed at `/World/DebugProxyMiddlePoint` with `proxy_middle_point_source=finger_link_pair_midpoint`.
- Verified example position logged:
  - `proxy_middle_point_world = [0.734590160154168, 0.4212117902863426, 1.1625832078637426]`
  - component positions:
    - finger1 `[0.7683838032242821, 0.4209245932531235, 1.1631722203462789]`
    - finger2 `[0.700796517084054, 0.4214989873195617, 1.1619941953812063]`
  - update count reported as `proxy_middle_point_marker_update_count = 1`.
- No grasp logic, close gate, or geometric filtering behavior was modified in this visual-marker pass.

- Added phase visibility fixes for `scripts/task1_hybrid_geometric_phase2.py` (debug markers before deeper execution) so marker creation no longer depends on later phase success:
  - Added/ensured immediate creation of `/World/DebugObjectGraspCenter` right after `estimate_object_grasp_frame(...)` using `object_grasp_center_world` from the selected object grasp frame.
  - Added pregrasp target marker path `/World/DebugPregraspTarget` using the selected candidate `pregrasp_world` immediately after candidate selection (before pregrasp/descend execution).
  - Added explicit IK failure observability around pregrasp phase execution:
    - logs include `ik_target_position_world`, `ik_target_rpy`, `ik_success`, `position_error_norm`, `orientation_error_rad` in `pregrasp_result` payload;
    - prints/flags `IK_PREGRASP_FAIL` on failed primary IK, and attempts one fallback IK with `yaw=0` (logged as `fallback_attempt`).
- CLI and logging behavior changes for visibility:
  - `execute_pregrasp(...)` now returns `fallback_attempt` payload and `ik_*` diagnostics.
  - Main pregrasp logging includes `pregrasp_result` and `fallback_attempt`.
  - Marker logs include `selected_pregrasp_target_debug_marker` with marker path/source.
- Debug payload fields added/confirmed in logs:
  - `proxy_middle_point_marker_path`
  - `proxy_middle_point_source`
  - `proxy_middle_point_component_positions_world`
  - `selected_pregrasp_target_debug_marker`
  - `pregrasp_result` with IK metrics/fallback.
- GUI validation commands run:
  - `/usr/bin/timeout 300 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 2 --arm auto --gui --log-suffix marker_debug_pregrasp_fail_check`
  - `/usr/bin/timeout 300 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 1 --arm auto --gui --log-suffix marker_debug_visibility_test2 --servo-max-ticks 1 --pregrasp-tolerance 1e-6 --rot-tolerance 1e-6`
  - `/usr/bin/timeout 300 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 1 --target-selection-policy index --target-index 0 --arm auto --gui --log-suffix marker_debug_pregrasp_fail_check3`
  - `/usr/bin/timeout 300 /home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_hybrid_geometric_phase2.py --seed 376 --target-selection-policy index --target-index 2 --arm auto --gui --log-suffix marker_debug_pregrasp_fail_check4 --pregrasp-tolerance 1e-6 --servo-max-ticks 1 --rot-tolerance 1e-6`
- Logs produced:
  - `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T201206Z_marker_debug_visibility_test3.log`
  - `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T201144Z_marker_debug_visibility_test2.log`
  - `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T201330Z_marker_debug_pregrasp_fail_check3.log`
  - `/home/edward/hrc-runtime/logs/task1_hybrid_geometric_phase2_20260418T201404Z_marker_debug_pregrasp_fail_check4.log`
- Validation conclusion:
  - Both `/World/DebugObjectGraspCenter` and `/World/DebugPregraspTarget` are logged and present in runtime payloads even when execution fails before main motion phases.
  - Latest runs failed at `pregrasp_candidate_failed` or `phase2_close_gate_failed` in the tested seeds; no run in this batch reached a primary pregrasp IK failure path to demonstrate `IK_PREGRASP_FAIL` fallback outcome.
  - This change is debug-only: no planner selection, geometry filters, grasp strategy, or IK solver internals were refactored.

- Phase 2 close-decision hardening fix for "good-enough commit" semantics in `scripts/task1_hybrid_geometric_phase2.py`:
  - Changed `evaluate_close_gate(...)` to a runtime-dominant commit gate:
    - hard blockers are now:
      - `close_commit_zone_pass` (runtime grasp-center/fingertip proximity, using `phase2_close_real_center_tolerance`)
      - `table_clearance_pass`
      - `width_compatibility_pass`
      - `orientation_error_pass`
    - soft warnings (logged but not decisive) are now:
      - `alignment_pass`
      - `lateral_symmetry_pass`
      - `predicted_contact_asymmetry_pass`
      - `recent_xy_drift_stable_pass`
  - Old strict all-pass rule was replaced with `condition_met = all(hard_blockers.values())`.
  - Added explicit logging fields in the close-gate payload:
    - `hard_blockers`, `soft_warning_flags`
    - `hard_fail_reasons`, `soft_warning_reasons`
    - `soft_warning_present`, `hard_condition_passed`
    - `commit_zone_error_m`, `close_commit_zone_tolerance_m`, `close_commit_zone_source`
    - `alignment_error_rad`, `width_compatibility_m`, `width_compatibility_pass`
  - Kept fallback (`evaluate_vertical_support_or_stall_close_fallback`) in place as secondary path for vertical support-stall recovery.
  - No phase machine changes; two-stage close and recovery flow remain unchanged.
- Commands run in this fix:
  - `python3 -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  - `python3 scripts/task1_hybrid_geometric_phase2.py --help`
- Current runtime note:
  - This is a logic-logic adjustment only; runtime seed revalidation not run in this step.

## 2026-04-19
- Focused-only diagnostics for fingertip proxy correctness in `scripts/task1_hybrid_geometric_phase2.py` without changing close logic, close gate thresholds, planner selection, or phase transitions.
- Added a temporary diagnostic bypass mode:
  - new CLI flag `--phase2-diagnostic-finger-link-midpoint-bypass` (BooleanOptionalAction, default False).
  - `_resolve_real_grasp_center_world(...)` now accepts `diagnostic_finger_link_midpoint_bypass` and `include_diagnostic_comparison` and forwards them internally.
  - Public wrapper `resolve_real_grasp_center_world(...)` now exposes these diagnostic controls so call sites can enable them.
  - Main phase path passes these flags into object grasp-center resolution (`--phase2-diagnostic-finger-link-midpoint-bypass`).
- Added explicit fingertip-proxy diagnostics in close-critical pathways (log-only):
  - `_pre_close_gate(...)` now logs:
    - `tip1_world`, `tip2_world`,
    - `distance_tip_mid_to_object_m`,
    - `fingertip_reference_source_used`,
    - `diagnostic_finger_link_midpoint_bypass_requested`,
    - `diagnostic_finger_link_midpoint_bypass_compare_enabled`,
    - plus existing legacy-vs-fingertip delta fields.
  - `final_descent_local_ik(...)` now recomputes/debug-logs per sample:
    - tip component positions,
    - `tip1_world`, `tip2_world`,
    - `distance_tip_mid_to_object_m`,
    - measured/fallback fingertip semantics and component-based deltas.
- Added/strengthened comparison metadata from `_resolve_real_grasp_center_world(...)`:
  - `diagnostic_standard_midpoint_world`,
  - `diagnostic_midpoint_delta_world`,
  - `diagnostic_midpoint_delta_norm_m`,
  - `diagnostic_compare_standard_midpoint_enabled`.
- Fixed one runtime-safe code-order bug introduced in this diagnostic path (ensured `fingertip_midpoint_world` is defined before use in distance computation).
- No close-gate behavior was modified in this change set.
- Checks run:
  - `python -m py_compile scripts/task1_hybrid_geometric_phase2.py`
  - `python3 scripts/task1_hybrid_geometric_phase2.py --help`
- Next step: run a focused runtime seed with `--phase2-diagnostic-finger-link-midpoint-bypass` on Linux GUI/headless to compare
  - real fingertip midpoint path vs direct finger-link midpoint
  - and distinguish incorrect proxy geometry from motion-targeting mismatch.

- Started PHASE 2 - TASK 1 reference-aligned grasp implementation.
- Source file chosen: `scripts/task1_hybrid_geometric_phase2.py`.
- New file created: `scripts/task1_phase2_reference_aligned_grasp.py`.
- Reason external references were consulted: improve final grasp reliability by adopting proven frame/phase/contact-geometry ideas without changing the repository architecture.
- Runtime reference area created outside the repo:
  - `$HRC_ROOT/reference/task1_phase2_external_refs/`
  - `$HRC_ROOT/reference/task1_phase2_external_refs/REFERENCE_NOTES.md`
  - `$HRC_ROOT/reference/task1_phase2_external_refs/comparison_notes/task1_phase2_vs_external_refs.md`
  - `$HRC_ROOT/reference/task1_phase2_external_refs/comparison_notes/CONSTRAINED_IMPLEMENTATION_NOTE.md`
- External repositories cloned shallowly under `$HRC_ROOT/reference/task1_phase2_external_refs/external_repos/`:
  - IsaacSim at commit `aa503a9bbf92405bbbcfe5361e1c4a74fe10d689`
  - moveit2 at commit `c154f941c0b029e3037b049b5515b68e8a9b3100`
  - moveit_task_constructor at commit `e217ddfad77fa957944994ec00bbb571bf9748b0`
  - graspnet-baseline at commit `280c215129f759ed8649cb4e89fc5dfee55f4f80`
- Ideas adopted:
  - IsaacSim-style explicit staged pick/place phase separation and simulator-friendly frame logging
  - MoveIt-style separation of grasp pose, pregrasp approach, close posture, and retreat semantics
  - MoveIt Task Constructor-style grasp/IK frame distinction and clean phase boundaries
  - GraspNet-style width/contact-aware candidate reasoning while keeping scoring/filtering early
  - runtime two-finger geometry as final close-critical truth
- Ideas rejected:
  - MoveIt/MTC framework rewrite
  - ROS planning-scene/action architecture
  - IsaacSim example controller architecture
  - GraspNet neural inference, datasets, checkpoints, Open3D collision stack, and learned scoring
- Functions added in `scripts/task1_phase2_reference_aligned_grasp.py`:
  - `_resolve_actual_fingertip_pair_midpoint_reference_position(...)`
  - `_finite_world_vector_or_none(...)`
  - `_first_finite_vector_from_mapping(...)`
  - `_compute_runtime_two_finger_metrics(...)`
  - `_reference_aligned_frame_summary(...)`
- Functions changed in `scripts/task1_phase2_reference_aligned_grasp.py`:
  - `_resolve_real_grasp_center_world(...)`: fallback ladder now prefers actual fingertip frames, stable finger-link midpoint, calibrated distal proxy, then explicit proxy.
  - `_pre_close_gate(...)`: logs `tip_mid_world`, `tip_axis_world`, `tip_mid_error_to_object_grasp_center_m`, `tip_axis_alignment_error_rad`, `tip_symmetry_error_m`, `tip_z_asymmetry_m`, and `close_runtime_metrics`.
  - `final_descent_local_ik(...)`: logs close-runtime two-finger metrics per sampled local descent step and records a reference-aligned frame summary.
  - `evaluate_close_gate(...)`: close commit-zone error now prefers runtime `tip_mid_error_to_object_grasp_center_m` when available; candidate-stage imperfections remain warning-oriented.
- Existing behavior intentionally preserved:
  - original `scripts/task1_hybrid_geometric_phase2.py` untouched
  - existing phase machine
  - DualArmIK/local servo backend
  - candidate generation and early filtering
  - two-stage close
  - short-lift verification
  - deterministic recovery and retry
- Test plan for the new script:
  - first lightweight check: `python -m py_compile scripts/task1_phase2_reference_aligned_grasp.py`
  - first runtime smoke on Linux Isaac Sim: run one known failing seed/target with a unique log suffix and inspect `close_runtime_metrics`
  - focused comparison: run the same seed against `scripts/task1_hybrid_geometric_phase2.py` and `scripts/task1_phase2_reference_aligned_grasp.py`
  - verify logs contain `tip1_world`, `tip2_world`, `tip_mid_world`, `tip_axis_world`, `tip_mid_error_to_object_grasp_center_m`, `tip_axis_alignment_error_rad`, `tip_symmetry_error_m`, and `tip_z_asymmetry_m`
  - evaluate whether close failures are now caused by hard blockers only: severe clearance, impossible width, catastrophic orientation mismatch, or runtime tip-midpoint still too far
- Tests run in this implementation pass: none. Linux runtime validation remains the source of truth and was not requested for this step.

- Started PHASE 2 - TASK 1 contact-centric correction patch on top of the current Phase 2 backend.
- Source file copied: `scripts/task1_hybrid_geometric_phase2.py`.
- New file created: `scripts/task1_phase2_contact_centric_patch.py`.
- Why this patch exists:
  - final descent still relied too much on `point_B` pose semantics
  - close gate treated orientation too strictly as a hard blocker
  - no generic near-enough stalled commit fallback existed outside the vertical-specific path
  - runtime fingertip diagnostics needed XY/Z split, per-tip distances, and easier close-decision traceability
  - GUI markers needed to show true two-finger runtime geometry, not only a proxy midpoint
- What `point_B` still does:
  - remains a compatibility input for existing pose-construction helpers and legacy debug logs
  - remains useful for comparing old proxy behavior against contact-centric runtime truth
- Why `point_B` is no longer final truth:
  - new command comments/logs route final descent through runtime contact reference first, then convert to an EE pose for DualArmIK compatibility
  - close gate now prefers runtime `tip_mid_error_to_object_grasp_center_m` when available
  - close-debug summary explicitly records `point_B_is_final_truth=false`
- Functions added in `scripts/task1_phase2_contact_centric_patch.py`:
  - `_resolve_actual_fingertip_pair_midpoint_reference_position(...)`
  - `_resolve_calibrated_distal_proxy_pair_midpoint_reference_position(...)`
  - `_finite_world_vector_or_none(...)`
  - `_first_finite_vector_from_mapping(...)`
  - `_compute_runtime_two_finger_metrics(...)`
  - `_pose_for_contact_reference_world(...)`
  - `_upsert_two_finger_runtime_debug_markers(...)`
  - `_reference_comparison_payload(...)`
  - `evaluate_runtime_commit_fallback(...)`
  - `build_close_debug_summary(...)`
- Functions changed in `scripts/task1_phase2_contact_centric_patch.py`:
  - `_resolve_real_grasp_center_world(...)`: truth order is actual fingertip pair, stable finger-link midpoint, calibrated distal proxy pair, explicit fallback proxy.
  - `_pre_close_gate(...)`: logs split two-finger runtime metrics, close runtime metrics, and two-finger markers.
  - `final_descent_local_ik(...)`: command path is documented/logged as contact-reference first and point_B compatibility second; trace now splits measured vs commanded tip-mid error.
  - `evaluate_close_gate(...)`: catastrophic orientation remains hard, moderate orientation becomes a warning; commit-zone error prefers runtime tip-mid error.
  - main close-decision path: final close decision is now primary close gate OR vertical support/stall fallback OR generic runtime commit fallback.
  - vertical tip table-z monitor: logs tip1/tip2/tip-mid table-z and support/object-top gaps.
  - vertical XY descend feedback: logs reference at phase start, reference at current tick, and delta from phase start.
- New thresholds / knobs:
  - `--catastrophic-orientation-error-max-rad`
  - `--soft-orientation-warning-max-rad`
  - `--catastrophic-table-clearance-min-m`
  - `--runtime-commit-fallback-enable` / `--no-runtime-commit-fallback-enable`
  - `--runtime-commit-fallback-tip-mid-error-max-m`
  - `--runtime-commit-fallback-recent-z-progress-max-m`
  - `--runtime-commit-fallback-recent-xy-drift-max-m`
  - `--runtime-commit-fallback-min-samples`
  - `--close-debug-summary-enable` / `--no-close-debug-summary-enable`
  - `--diagnostic-two-finger-marker-enable` / `--no-diagnostic-two-finger-marker-enable`
- New logs / markers:
  - `close_debug_summary`
  - `generic_runtime_commit_fallback_gate`
  - `tip_mid_xy_error_m`
  - `tip_mid_z_error_m`
  - `tip1_to_object_grasp_center_distance_m`
  - `tip2_to_object_grasp_center_distance_m`
  - `tip1_to_object_center_distance_m`
  - `tip2_to_object_center_distance_m`
  - `measured_tip_mid_world`
  - `commanded_tip_mid_world`
  - `measured_tip_mid_to_target_delta_world`
  - `commanded_tip_mid_to_target_delta_world`
  - `/World/DebugTip1World`
  - `/World/DebugTip2World`
  - `/World/DebugTipMidWorld`
  - `/World/DebugObjectCenter`
  - `/World/DebugRuntimeObjectGraspCenter`
- Existing behavior intentionally preserved:
  - original `scripts/task1_hybrid_geometric_phase2.py` untouched
  - scene setup
  - scene-state object extraction
  - table-frame builder
  - candidate generation
  - fast scoring
  - fast geometric filter
  - DualArmIK backend
  - phase machine order
  - two-stage close
  - short-lift verification
  - deterministic retry/recovery
- Tests run in this implementation pass: none. Linux Isaac Sim remains the runtime source of truth.
- Required next test plan:
  - compare actual fingertip pair vs finger-link midpoint vs calibrated proxy stability
  - compare measured vs commanded tip-mid trajectory during final descent
  - compare old strict close gate against new practical close gate
  - create/identify a near-object stalled case and verify generic runtime commit fallback
  - verify vertical tip table-z rule fires only when tip midpoint is actually near table z reference
- Fixed a syntax error in `scripts/task1_phase2_contact_centric_patch.py` reported by runtime:
  - error: `IndentationError: unexpected indent` near line 6303
  - cause: over-indented unresolved-fingertip `else` branch inside the vertical tip table-z monitor
  - fix: aligned `current_vertical_tip_table_z`, `vertical_tip_stop_rule_triggered`, `vertical_tip_error`, and `tip_source` assignments with the rest of the `else` block
  - original `scripts/task1_hybrid_geometric_phase2.py` remains untouched
- Applied minimal horizontal/far approach stagnation fix in `scripts/task1_phase2_contact_centric_patch.py`:
  - lowered far/horizontal XY align height to `object_top_z + far_xy_align_clearance_above_object`
  - added `--horizontal-descent-xy-trigger-tolerance` with default `0.07`
  - replaced strict full-pose align failure before descent with a loose XY descent trigger
  - logs now include `align_height`, `object_top_z`, `align_clearance`, `descent_triggered`, and `xy_error_at_descent`
  - existing phase structure and DualArmIK backend remain unchanged
- Applied requested minimal critical horizontal grasp execution fix directly in `scripts/task1_hybrid_geometric_phase2.py`:
  - relaxed horizontal `far_low_side_B_driven` table clearance threshold to `-0.005`
  - allowed descent from far XY align when `xy_error < 0.06` instead of requiring perfect full-pose alignment
  - forced final descent runtime measurement to prefer `tip_mid` and use `point_B_fallback` only when tip midpoint is unavailable
  - expanded horizontal final grasp target along `closing_axis_world` by `min(max(object_width * 0.3, 0.01), 0.03)`
  - added `PHASE2_RUNTIME_COMMIT_FALLBACK_TIP_MID_ERROR_MAX_M = 0.08` and uses it as the far/horizontal close commit tolerance floor
  - added debug logs for `descent_triggered`, `xy_error`, `table_clearance_margin`, and `control_reference_source`
  - planner, candidate generation, phase structure, and DualArmIK backend were not refactored

## 2026-04-19 — Phase 2 Task 1 contact-centric horizontal descent correction

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Issue fixed: horizontal `far_low_side_B_driven` approach could stop at XY align height because table clearance, XY trigger strictness, and point_B compatibility control prevented descent from becoming contact-centric.
- Thresholds changed: generic runtime commit fallback tip-mid error max relaxed from `0.045 m` to `0.08 m`; horizontal descent trigger default set to `0.06 m`; horizontal table-clearance pass threshold added at `-0.005 m` while retaining separate catastrophic table-clearance blocking.
- Control change: final descent now uses runtime `tip_mid` as the primary measured/control reference when available, with `point_B_fallback` only when tip-mid geometry is unavailable.
- Geometry change: horizontal contact target receives a bounded closing-axis expansion before final descent, using `clamp(object_width * 0.3, 0.01, 0.03)`.
- Logging added/extended: `descent_triggered`, `xy_error`, `xy_error_at_descent_trigger`, `horizontal_descent_trigger_tolerance_used`, table-clearance pass fields, `control_reference_source`, and `measured_world_source_for_descent`.
- Test plan: run the contact-centric script in Isaac Sim with GUI/hold-open and inspect logs for horizontal descent start, tip-mid movement toward the object, reduced runtime tip-mid error, and eventual close-gate or runtime fallback pass.

## 2026-04-19 — Phase 2 Task 1 hybrid grasp-center fallback priority fix

- Active file patched: `scripts/task1_hybrid_geometric_phase2.py`.
- Issue fixed: real grasp-center resolution could allow the finger-link midpoint diagnostic/fallback path to dominate before calibrated distal fingertip proxy semantics, even though finger link origins are low-accuracy palm/joint-side references.
- Fallback priority now recorded as actual fingertip frames, calibrated distal fingertip proxy from bbox face, stable finger-link midpoint, explicit fallback proxy.
- Diagnostic bypass behavior is now comparison-only for actual/calibrated success paths; link midpoint no longer overrides actual or calibrated fingertip references.
- Added logs: `fingertip_source`, `calibrated_vs_link_midpoint_offset_m`, `calibrated_vs_link_midpoint_warning`, and diagnostic-bypass effect fields.
- Tests run: none. Linux Isaac Sim runtime remains the validation source of truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric grasp-center fallback priority fix

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Critical bug fixed: `_resolve_real_grasp_center_world(...)` returned stable finger-link midpoint before calibrated distal fingertip proxy, even though finger-link origins are low-accuracy base/joint references rather than fingertip contact references.
- Correct fallback order is now actual fingertip frames, calibrated distal fingertip proxy from bbox face, stable finger-link midpoint, explicit fallback proxy.
- Diagnostic finger-link midpoint bypass no longer overrides actual or calibrated fingertip references; it is recorded as comparison-only on successful actual/calibrated paths.
- Added diagnostics: `fingertip_source`, `calibrated_vs_link_midpoint_offset_m`, `calibrated_vs_link_midpoint_warning`, and diagnostic-bypass effect fields.
- No IK, descent, planner, candidate-generation, or phase-machine logic was changed.
- Tests run: none. Linux Isaac Sim runtime remains the validation source of truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric close safety and pose-builder cleanup

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Fixed `_pose_for_contact_reference_world(...)` so it builds through `_pose_for_point_b_world(...)` with `pose_log` instead of referencing undefined `log`.
- Updated pose-construction metadata to state that contact references are converted through a local compatibility offset and that `point_B` is not final truth.
- Added explicit fingertip-centric runtime-truth comment in `_compute_runtime_two_finger_metrics(...)`.
- Added contact/control subject logs before far and vertical `final_descent_local_ik(...)` calls.
- Added hard close refusal before `execute_two_stage_close(...)` when `close_critical_uses_real_grasp_center` is not true, preventing explicit proxy-only close authorization.
- Removed vertical tip table-z auto-pass behavior; the rule is now an auxiliary diagnostic condition and final close still depends on primary, vertical fallback, or generic runtime commit gates.
- No IK backend, planner, candidate generation, phase order, or perception logic was changed.
- Tests run: none. Linux Isaac Sim runtime remains the validation source of truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric servo metric parameter binding fix

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Fixed `_execute_dualarmik_servo_phase(...)` parameter binding bug by moving `position_metric_offset_local` and `position_metric_label` into the real function signature before `extra_details`.
- Removed the two pseudo-parameter annotation lines from the function body so the metric-stop logic no longer reads unbound locals.
- Kept the existing metric-stop logic that prefers `position_metric_offset_local`, then falls back to point_B, then EE pose.
- Fixed the `final_descent_local_ik(...)` call site so `_execute_dualarmik_servo_phase(...)` receives the actual nested `_pose_for_contact_reference_world` callback instead of an undefined `target_pose_fn` name.
- No IK backend, planner, candidate generation, phase order, or control-law changes were made.
- Tests run: none. Linux Isaac Sim runtime remains the validation source of truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric callback shadow fix

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Runtime bug fixed: `final_descent_local_ik(...)` defined a nested callback named `_pose_for_contact_reference_world`, shadowing the global pose-builder helper and causing Python to treat that name as an unbound local before the nested callback was defined.
- Renamed the zero-argument nested callback to `contact_target_pose_fn` and passed that callback into `_execute_dualarmik_servo_phase(...)`.
- Moved nominal contact-reference pose construction until after `contact_control_offset` is computed, so `_pose_for_contact_reference_world(...)` is called with a valid offset.
- Added the nominal contact-reference pose conversion log into the final descent `target_lock_frame` payload.
- No IK backend, planner, candidate generation, phase order, or control-law changes were made.
- Checks passed: `python3 -m py_compile scripts/task1_phase2_contact_centric_patch.py` and `python3 scripts/task1_phase2_contact_centric_patch.py --help`.
- Runtime limitation: no Linux Isaac Sim run was executed in this patch; Linux remains the runtime source of truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric close-truth priority fix

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Issue addressed: close gate could treat the calibrated distal fingertip proxy as close-critical truth when actual fingertip frames were unavailable, even when runtime evidence suggested that proxy could be far above and away from the visual pinch area.
- Restored the close-truth priority required by `CURRENT_PLAN.md`: actual fingertip frame pair, then stable finger-link midpoint, then calibrated distal proxy as diagnostic-only, then explicit fallback proxy.
- Calibrated distal proxy returns now set `close_critical_reference=false` with `close_critical_rejected_reason=calibrated_distal_proxy_pair_not_runtime_validated`; it can still be logged and drawn but cannot authorize close or drive contact-centric descent.
- `_compute_runtime_two_finger_metrics(...)`, `evaluate_close_gate(...)`, and `evaluate_runtime_commit_fallback(...)` now only use runtime tip-mid error as decisive close truth when `primary_runtime_truth=true`, which requires a trusted close-critical reference.
- `_pre_close_gate(...)` now distinguishes a present diagnostic `real_grasp_center_world` from a trusted close-critical grasp center; `close_critical_uses_real_grasp_center` is true only when the resolved reference is trusted.
- Final descent now ignores untrusted calibrated-proxy midpoint for the control reference and falls back to point_B compatibility until actual or stable midpoint truth is available.
- `_pose_for_contact_reference_world(...)` now overrides inherited helper metadata with `target_semantics=contact_reference_world_driven` while preserving the legacy helper's old semantics under `legacy_pose_builder_target_semantics`.
- Added close-reference debug markers for runtime comparison:
  `/World/DebugRealGraspCenter`, `/World/DebugContactPointWorld`, and `/World/DebugContactPointBWorld`, alongside the existing `DebugTip1World`, `DebugTip2World`, and `DebugTipMidWorld` markers.
- No gate thresholds were relaxed in this patch; the existing catastrophic table-clearance blocker remains intact.
- Checks passed: `python3 -m py_compile scripts/task1_phase2_contact_centric_patch.py`, `python3 scripts/task1_phase2_contact_centric_patch.py --help`, and `git diff --check -- scripts/task1_phase2_contact_centric_patch.py`.
- Runtime limitation: no Linux Isaac Sim run was executed in this patch; the next Linux run should verify marker alignment and whether close failure now reports an untrusted calibrated proxy instead of a false high-altitude close truth.

## 2026-04-19 — Phase 2 Task 1 contact-centric near-table commit gate tuning

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Issue addressed: near-table close could remain blocked when the candidate/contact geometry had a small negative table-clearance margin and the descent was already making tiny Z progress near contact.
- Relaxed default `PHASE2_TABLE_CLEARANCE_MIN_M` from `0.002` to `-0.010`, so mild tabletop penetration/surface-contact estimates no longer fail the nominal table-clearance pass.
- Relaxed default `PHASE2_CATASTROPHIC_TABLE_CLEARANCE_MIN_M` from `-0.002` to `-0.010`, so only deeper estimated penetration is treated as catastrophic.
- Relaxed both Z-stall commit thresholds from `0.0015` to `0.010`:
  `PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_Z_PROGRESS_MAX_M` and `PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M`.
- Updated CLI validation so `--phase2-table-clearance-min` may be negative as long as it remains at least as strict as `--catastrophic-table-clearance-min-m`.
- Added explicit final-descent Z debug fields in trace samples:
  `current_z_world_m`, `target_z_world_m`, `delta_z_world_m`, `commanded_z_world_m`, and `commanded_delta_z_world_m`.
- Added log wording clarifying that Z-stall near contact is a commit condition, not a descent failure, and that table clearance now allows mild tabletop penetration but still blocks beyond the catastrophic threshold.
- Did not add a default force-descend/no-gate override; forcing descent without gate checks remains too broad for the current contact-centric patch.
- Checks passed:
  `python3 -m py_compile scripts/task1_phase2_contact_centric_patch.py`,
  `python3 scripts/task1_phase2_contact_centric_patch.py --help`,
  and `python3 scripts/task1_phase2_contact_centric_patch.py --phase2-table-clearance-min -0.01 --catastrophic-table-clearance-min-m -0.01 --help`.
- Runtime limitation: no Linux Isaac Sim run was executed in this patch; next Linux run should confirm whether the previous `table_clearance_margin_m≈-0.002` and small-Z-progress case now reaches close commit or a more specific remaining blocker.

## 2026-04-19 — Phase 2 Task 1 candidate and stalled-Z fallback hardening

- Active file patched: `scripts/task1_phase2_contact_centric_patch.py`.
- Disabled least-bad candidate selection by default with `PHASE2_ALLOW_LEAST_BAD_CANDIDATE=false`; when no candidate passes all Phase 2 mandatory geometric checks, selection now fails early with `no_phase2_candidate_passed_mandatory_geometric_filter` instead of descending with an alignment/table-clearance failure.
- Kept the CLI override `--phase2-allow-least-bad-candidate` for explicit diagnostics, but the default runtime path no longer continues after `geometric_pass_candidate_count == 0`.
- Reworked vertical XY reference resolution so `finger_midpoint`/`fingertip_midpoint` uses actual fingertip pair if available, otherwise stable finger-link midpoint; calibrated distal proxy is not used as vertical XY truth.
- Allowed the support-gap/Z-stall close fallback to apply to far motion policy when `--phase2-vertical-fallback-allow-far-policy` is enabled, while still requiring small support gap, Z stall, bounded XY drift, and orientation sanity.
- Removed the final-decision `(not far_motion_policy)` block around `vertical_support_or_stall_fallback`, so a passing far stalled-Z fallback can actually authorize close.
- Checks passed:
  `python3 -m py_compile scripts/task1_phase2_contact_centric_patch.py`,
  `python3 scripts/task1_phase2_contact_centric_patch.py --help`,
  and `git diff --check -- scripts/task1_phase2_contact_centric_patch.py`.
- Current runtime status from user feedback: still not solved; robot still has not completed the intended descend/close/grasp behavior.
- Runtime limitation: no Linux Isaac Sim run was executed in this patch; next Linux run should verify that bad-orientation least-bad candidates fail before descent and that far hover cases can use stalled-Z fallback only when the logged support/XY/orientation gates pass.

## 2026-04-20 — Task 1 contact-centric algorithm explanation document

- Added Vietnamese algorithm explanation document for `scripts/task1_phase2_contact_centric_patch.py`.
- New document path: `docs/task1_phase2_contact_centric_patch_giai_trinh_thuat_toan_vi.txt`.
- Content covers the general contact-centric idea, major runtime/control/data-flow blocks, line-numbered code explanations, core pose/contact formulas, debug guidance, and a quick function map.
- Documentation-only update; no runtime code, training code, or scripts changed.
- Checks run: repo inspection only; no Isaac runtime or simulation commands were run.

## 2026-04-20 — GitHub repository Vietnamese explanation document

- Added Vietnamese repository explanation document for `https://github.com/Moquz27/hrc2026`.
- New document path: `docs/github_repo_giai_trinh_vi.md`.
- Content covers repository purpose, current phase, Task 1 plan phase, workflow, official resources, tracked file roles, script families, integration status, known risks, and next engineering direction.
- Documentation-only update; no runtime code, training code, or scripts changed.
- Checks run: repo inspection only; no Isaac runtime or simulation commands were run.

## 2026-04-20 — Contact-centric explanation Markdown conversion

- Added Markdown version of `docs/task1_phase2_contact_centric_patch_giai_trinh_thuat_toan_vi.txt`.
- New document path: `docs/task1_phase2_contact_centric_patch_giai_trinh_thuat_toan_vi.md`.
- Converted the same explanation into Markdown headings and Vietnamese text with full diacritics while preserving source line references, formulas, function names, code identifiers, paths, and technical ordering.
- Kept the original `.txt` file unchanged for audit comparison.
- Documentation-only update; no runtime code, training code, or scripts changed.

## 2026-04-21 — Phase 1 Task 1 synchronized RGB-D/truth collection reset

- Added `scripts/task1_collect_rgbd_labels.py` as a standalone Phase 1 collector.
- Collector scope: build official Task 1 SceneBuilder table/parts/robot, reuse official `RobotArticulation.get_cameras_images(step)`, save head left/right and wrist left/right RGB-D arrays, labels, metadata, sync debug records, and one JSONL manifest entry per sample.
- Output structure is under `$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/` with stable `rgb/`, `depth/`, `labels/`, `metadata/`, and `sync_debug/` folders.
- Simulator truth labels include object id, class, world pose, USD robot-root base-frame pose, Task 1 table-frame pose, yaw/coarse orientation, configured semantic bin metadata, and best-effort camera projection visibility metadata.
- Added `docs/task1_data_collection_schema.md` to document the saved dataset fields and current visibility limitations.
- Added minimal schema-only placeholders:
  - `docs/schemas/task1_thinker_structured_output.schema.json`
  - `docs/schemas/task1_evaluator_io.schema.json`
- Thinker schema is intentionally intermediate/advisory only: object candidates, class, 2D center/ROI, coarse orientation, recommended arm/preset, and confidence. It does not define final grasp poses or robot commands.
- Preserved current manipulation backend: no edits to `grasp_planner.py`, `DualArmIK.py`, `coordinate_utils.py`, `RobotArticulation.py`, or existing Task 1 manipulation scripts.
- Updated `CURRENT_PLAN.md` and `PROJECT_CONTEXT.md` to record the user-authorized Phase 1 data-collection reset.
- Tests run in this pass: lightweight syntax/schema validation only; no full Isaac Sim collection run yet.

## 2026-04-21 — Phase 1 Task 1 minimal competition dataset refinement

- Simplified `scripts/task1_collect_rgbd_labels.py` from a research-style RGB-D/truth dump into a competition-oriented dataset writer for camera -> Thinker -> planner.
- Default dataset output now contains only `manifest.jsonl`, `rgb/`, and `labels/`; `depth/` is written only with `--save-depth`.
- Label JSON is now Thinker-shaped: `{"objects": [{"class": "A"|"B", "x": table_x_m, "y": table_y_m, "yaw": table_yaw_rad}]}`.
- Removed default per-sample `metadata/`, `sync_debug/`, world pose labels, base-frame pose labels, bbox labels, target-bin metadata, planner metadata, execution result, and fail reason fields.
- Added exported-label noise by default: XY Gaussian sigma `0.005 m` and yaw Gaussian sigma `3 deg`; flags can disable or retune noise without modifying raw simulator truth.
- Simplified manifest entries to sample id, run id, seed, RGB paths, optional depth paths, label path, object count, and label-noise settings.
- Simplified `docs/task1_data_collection_schema.md` and `docs/schemas/task1_thinker_structured_output.schema.json` to match the table-frame object output contract.
- Removed the Phase 1 evaluator placeholder schema because it was premature for the minimal dataset path.
- Preserved current manipulation backend and official camera/scene paths; no edits were made to planner, IK, coordinate transforms, `RobotArticulation.py`, or existing Task 1 manipulation scripts.

## 2026-04-21 — Camera-first Task 1 direction reset and Phase 1 collector restore

- Updated `PROJECT_CONTEXT.md` and `CURRENT_PLAN.md` to make the camera-first competition runtime the active source of truth.
- Recorded the old scene-truth-driven direction as useful only for debugging, labeling, evaluation, and bootstrapping, not final competition runtime input.
- Replaced the minimal table-label-only collector with a synchronized Phase 1 RGB-D/truth collector in `scripts/task1_collect_rgbd_labels.py`.
- Collector scope now builds the official Task 1 scene, uses `RobotArticulation.get_cameras_images(step)`, and writes head left/right plus wrist left/right RGB-D arrays.
- Labels now include object id, class, raw class, world pose, USD robot-root base-frame pose, Task 1 table-frame pose, yaw/coarse orientation, configured target-bin metadata, world bbox, and best-effort center-projection visibility metadata.
- Runtime metadata now includes chosen object, chosen arm, chosen preset, chosen candidate, planner target, execution result, fail reason, simulation step, estimated sim time, and timestamp.
- Dataset output restored to a structured Phase 1 layout under `$OUTPUT_ROOT/datasets/task1_rgbd_labels/<run_id>/` with `run_metadata.json`, `manifest.jsonl`, `rgb/`, `depth/`, `labels/`, `metadata/`, and `sync_debug/`.
- Added/updated schema docs:
  - `docs/task1_data_collection_schema.md`
  - `docs/schemas/task1_thinker_structured_output.schema.json`
  - `docs/schemas/task1_evaluator_io.schema.json`
- Thinker schema is explicitly intermediate structured perception/decision support: object candidates, class, ROI/center, orientation bucket, difficulty, occlusion, confidence, recommended arm, recommended preset, and selected object id. It does not define final grasp poses or robot commands.
- Phase 2 evaluator schema is a placeholder interface only; no evaluator runtime was implemented in this phase.
- Preserved current manipulation backend: no edits to `grasp_planner.py`, `DualArmIK.py`, `coordinate_utils.py`, `RobotArticulation.py`, or existing Task 1 manipulation scripts.
- Checks passed: `python3 -m py_compile scripts/task1_collect_rgbd_labels.py`, `python3 scripts/task1_collect_rgbd_labels.py --help`, `python3 -m json.tool` for the Thinker and evaluator schema files, and `git diff --check`.
- Runtime limitation: no Isaac Sim collection run was performed on this machine.

## 2026-04-21 — Phase 1 collector frame and visibility semantics tightening

- Patched `scripts/task1_collect_rgbd_labels.py` so the Task 1 table frame is built from the actual USD stage robot-root pose when available.
- Kept the YAML `robot_position` / `robot_rotation` pose as an explicit fallback only, with run metadata recording both pose sources.
- Added config-vs-USD robot pose comparison metadata, including position/yaw deltas and warning thresholds; the collector logs `robot_pose_source_warning` when the difference exceeds the threshold.
- Made visibility semantics explicit: `visible_projection` is center-projection-only and not visibility truth; the collector records no segmentation, no true occlusion reasoning, and no depth ROI finite-ratio check.
- Added projected 3D bbox debug fields under per-camera `bbox_projection`; these are debug hints only and do not make visibility occlusion-aware.
- Tightened `docs/task1_data_collection_schema.md` around world pose, USD robot-root pose, table-frame pose, meter units, table units, yaw reference axis, and coarse-orientation bucket origin.
- Added brief schema descriptions for Thinker `orientation_bucket` and Phase 2 evaluator metric units/frame assumptions.
- Preserved current manipulation backend; no edits were made to planner, IK, coordinate transforms, `RobotArticulation.py`, or existing Task 1 manipulation scripts.
- Checks passed: `python3 -m py_compile scripts/task1_collect_rgbd_labels.py`, `python3 scripts/task1_collect_rgbd_labels.py --help`, `python3 -m json.tool` for the Thinker and evaluator schema files, and `git diff --check`.
- Runtime limitation: no Isaac Sim collection run was performed on this machine.

## 2026-04-21 — Phase 1 collector Isaac physics initialization bugfix

- Runtime blocker reproduced with Isaac Sim `python.sh`: the collector launched Isaac and built the run folder, but stopped before writing samples because `RobotArticulation.initialize()` saw no Isaac physics simulation view and failed with `AttributeError: 'NoneType' object has no attribute 'create_articulation_view'`.
- Patched only `scripts/task1_collect_rgbd_labels.py` runtime initialization ordering.
- Added an Isaac `SimulationContext` readiness step after SceneBuilder table/parts/robot creation, `rep.orchestrator.step()`, and initial updates, but before `RobotArticulation.initialize()`.
- The collector now logs `scene_built`, `physics_ready_check`, `physics_ready`, `robot_initialize_start`, and `robot_initialize_success` so future failures around camera wrapper setup are visible in the collector log.
- Preserved dataset output structure, labels, metadata, schema, official `RobotArticulation.get_cameras_images(step)` path, and manipulation/control backend behavior.
- Checks passed: `python3 -m py_compile scripts/task1_collect_rgbd_labels.py` and `python3 scripts/task1_collect_rgbd_labels.py --help`.
- Isaac runtime validation passed with `/home/edward/Projects/NVIDIA/isaac-sim/python.sh scripts/task1_collect_rgbd_labels.py --samples 3 --sample-stride 2 --seed 1 --run-id test_phase1_initfix_1`.
- Runtime output validation passed: `manifest.jsonl` has 3 entries, 12 RGB arrays, 12 depth arrays, 3 label files, 3 metadata files, 3 sync debug files, 4 cameras per sample, positive depth `finite_count` for all first-sample cameras, and label `object_count` matches 4 spawned Task 1 parts.

## 2026-04-21 — Phase 1 baseline freeze before evaluator implementation

- Git status before freeze: clean and synced with `origin/main`; no modified or untracked files were present.
- Frozen Phase 1 baseline commit: `ee6ca51` (`Restore Task 1 RGB-D truth collector`).
- Confirmed the frozen baseline is the fuller synchronized RGB-D/truth collector, not the older simplified table-label-only collector.
- Real Linux smoke run selected as the first evaluator validation input:
  `$OUTPUT_ROOT/datasets/task1_rgbd_labels/test_phase1_initfix_1`.
- Updated `CURRENT_PLAN.md` and `PROJECT_CONTEXT.md` so Phase 1 is marked complete and Phase 2 automatic evaluator work is active.
- Manipulation backend files remain out of scope: no edits to `grasp_planner.py`, `DualArmIK.py`, `coordinate_utils.py`, or `RobotArticulation.py`.

## 2026-04-21 — Phase 2 Task 1 dataset evaluator baseline

- Added `scripts/task1_evaluate_dataset.py` as the standalone Phase 2 evaluator entrypoint.
- Structural validation covers `manifest.jsonl`, referenced RGB/depth files, four-camera completeness, `.npy` loadability and shape checks, depth finite-count summaries, labels, metadata, sync debug records, and object-count consistency.
- Optional prediction inputs are supported through direct JSON/JSONL plus Thinker, geometry, planner trace, execution log, or evaluator-I/O wrapper paths.
- Implemented metrics: class accuracy, selected-object accuracy, 2D center error, yaw bucket accuracy, arm recommendation accuracy, preset recommendation accuracy, 3D/table-frame position error, task success rate, wrong-bin rate, drop rate, and cycle time.
- Real validation command:
  `python3 scripts/task1_evaluate_dataset.py --dataset-root "$OUTPUT_ROOT/datasets/task1_rgbd_labels/test_phase1_initfix_1" --report "$OUTPUT_ROOT/metrics/task1_dataset_eval_test_phase1_initfix_1.json" --strict`
- Also verified run-id resolution without an absolute dataset path:
  `python3 scripts/task1_evaluate_dataset.py --run-id test_phase1_initfix_1 --report "$OUTPUT_ROOT/metrics/task1_dataset_eval_test_phase1_initfix_1_by_run_id.json" --strict`
- Real validation result: structural PASS with 3 samples, 12 RGB arrays, 12 depth arrays, 3 label files, 3 metadata files, 3 sync debug files, complete four-camera records, depth finite counts positive for all cameras, and object_count `[4]`.
- Prediction metric status on the real run: pending because no real prediction/Thinker/geometry/planner/execution inputs were provided.
- Synthetic prediction smoke check outside the repo confirmed the implemented metric path can compute class accuracy, yaw bucket accuracy, and 3D table-frame position error from matching object predictions.
- No edits were made to `grasp_planner.py`, `DualArmIK.py`, `coordinate_utils.py`, or `RobotArticulation.py`.

## 2026-04-23 — Branch setup for input-correction experiment

- Started from `main` with uncommitted Phase 2 evaluator changes.
- Checks before committing evaluator baseline: `python3 -m py_compile scripts/task1_evaluate_dataset.py scripts/task1_collect_rgbd_labels.py`, schema JSON validation for evaluator/Thinker schemas, and `git diff --check`.
- Committed the evaluator baseline on `main` as `10593a6` with message `Add Task 1 dataset evaluator`.
- Created and switched to branch `test-camera-kiet`.
- Pushed `test-camera-kiet` to origin before starting new implementation work.

## 2026-04-23 — Task 1 AI input-correction evaluation workflow

- Added `scripts/task1_input_correction.py` as a lightweight modular correction layer for input-level fields only.
- Added `scripts/task1_run_input_correction_eval.py` for offline 10-case correction evaluation.
- Added `docs/task1_input_correction_eval_format.md` documenting the JSON case format and output layout.
- Supported correction targets: selected object id, class, 2D center, ROI, orientation bucket, recommended arm, and recommended preset.
- Correction gates: configurable confidence threshold, allowed object-id check, center/ROI large-correction rejection, enum validation, and forbidden-field ignoring for grasp poses, 3D pose/position overwrites, joint commands, motion commands, trajectories, and waypoints.
- Outputs are written outside the repo under `$OUTPUT_ROOT/test_runs/task1_input_correction_eval/` with one JSON file per case plus `summary.json`.
- Validation command:
  `python3 scripts/task1_run_input_correction_eval.py --run-id test_phase1_initfix_1 --limit 10 --output-dir "$OUTPUT_ROOT/test_runs/task1_input_correction_eval/test_phase1_initfix_1_kiet"`
- Validation result: 10 generated offline cases completed; accepted AI corrections 64, rejected AI corrections 6, cases improved 10, unchanged 0, worsened 0.
- Before/after synthetic-case metrics: class accuracy 0.6 -> 0.9, selected-object accuracy 0.7 -> 1.0, mean 2D center error 28.95 px -> 7.37 px, orientation accuracy 0.5 -> 0.8, arm recommendation accuracy 0.7 -> 1.0, preset recommendation accuracy 0.7 -> 1.0.
- Important limitation: cases use deterministic synthetic AI outputs generated from Phase 1 truth to test gating and metric plumbing; this is not real Thinker runtime performance.
- No edits were made to `grasp_planner.py`, `DualArmIK.py`, `coordinate_utils.py`, or `RobotArticulation.py`.
