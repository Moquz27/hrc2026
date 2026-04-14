#!/usr/bin/env python3
"""GUI-first Task 1 pick/place visual inspection scene.

This script opens a minimal Task 1 scene for manual Isaac Sim inspection. It
uses the official Part_Sorting.yaml through SceneBuilder for the table and Task
1 parts, loads Walker S2 in the same stage, and uses the diagnostic static bin
collider instead of the currently broken official box locking pipeline.

It is not a scored Task 1 baseline and does not verify task success.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from control_walker_s2_arms import (  # type: ignore
    _acquire_articulation,
    _hold_gui_open,
    _read_dof_observation,
    _send_position_targets,
    _start_timeline,
)
from diagnose_task1_bin_physics import (  # type: ignore
    DEFAULT_BOX_RELATIVE,
    _add_static_bin_colliders,
    _disable_physics_under,
)
from front_seeded_manipulation_motion import (  # type: ignore
    ABOVE_FRONT_TARGET_BY_NAME,
    DOWN_FRONT_TARGET_BY_NAME,
    _current_positions,
    _named_positions,
    _targets_from_map,
)
from grasp_static_object_smoke import _read_positions, _select_right_gripper_dofs  # type: ignore
from load_walker_s2 import (  # type: ignore
    DEFAULT_INIT_STEPS,
    DEFAULT_PRIM_PATH,
    _create_minimal_scene,
    _find_articulation_roots,
    _find_joint_names,
    _load_robot_reference,
    _load_simulation_app,
    _resolve_robot_usd,
    _validate_environment,
)
from move_walker_s2_end_effector import (  # type: ignore
    _body_pose_position,
    _create_debug_marker,
    _identify_end_effector_body,
    _select_right_arm_dofs,
)
from right_arm_joint_space_sanity import FRONT_POSE_BY_NAME  # type: ignore
from validate_task1_object_assets import _bbox, _physics_summary  # type: ignore
from validate_task1_scene_builder_scene import (  # type: ignore
    DEFAULT_ASSET_ROOT_RELATIVE,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_CONFIG_RELATIVE,
    _NullDataLogger,
    _category_from_reference,
    _get_stage,
    _load_official_scene_builder,
    _path_from_env_or_default,
    _reference_paths,
)


LOG_NAME = "task1_pick_place_gui_inspection.log"
DEFAULT_PHASE_STEPS = 120
DEFAULT_PAUSE_STEPS = 60
DEFAULT_SETTLE_STEPS = 180
DEFAULT_GRIPPER_DELTA = 0.03

BIN_TARGET_BY_NAME = {
    **FRONT_POSE_BY_NAME,
    "R_shoulder_roll_joint": -0.38,
    "R_elbow_roll_joint": 0.04,
}


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


def _add_reference(stage: Any, prim_path: str, usd_path: Path) -> Any:
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.GetReferences().AddReference(str(usd_path)):
        raise RuntimeError(f"Could not add reference {usd_path} at {prim_path}")
    return prim


def _set_xform(stage: Any, prim_path: str, position: list[float], rotation_xyz_deg: list[float] | None = None, scale: list[float] | None = None) -> None:
    from pxr import UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot set transform on missing prim: {prim_path}")
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(tuple(float(value) for value in position))
    if rotation_xyz_deg is not None:
        xform.SetRotate(tuple(float(value) for value in rotation_xyz_deg))
    if scale is not None:
        xform.SetScale(tuple(float(value) for value in scale))


def _create_stage_marker(stage: Any, path: str, position: list[float], radius: float, color: tuple[float, float, float]) -> str:
    return _create_debug_marker(stage, path, np.array(position, dtype=float), radius, color)


def _create_bbox_corner_markers(stage: Any, prefix: str, bbox: dict[str, list[float]], color: tuple[float, float, float]) -> list[str]:
    min_v = bbox["min"]
    max_v = bbox["max"]
    paths: list[str] = []
    for index, point in enumerate(
        (
            [min_v[0], min_v[1], min_v[2]],
            [max_v[0], min_v[1], min_v[2]],
            [min_v[0], max_v[1], min_v[2]],
            [max_v[0], max_v[1], min_v[2]],
            [min_v[0], min_v[1], max_v[2]],
            [max_v[0], min_v[1], max_v[2]],
            [min_v[0], max_v[1], max_v[2]],
            [max_v[0], max_v[1], max_v[2]],
        )
    ):
        paths.append(_create_stage_marker(stage, f"{prefix}Corner{index}", point, 0.01, color))
    return paths


def _command_arm_phase(
    phase_name: str,
    dc: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    target_positions: list[float],
    sim_app: Any,
    phase_steps: int,
    phase_log: list[dict[str, Any]],
) -> dict[str, Any]:
    start_positions = _current_positions(dc, arm_dofs)
    for step in range(1, phase_steps + 1):
        alpha = step / float(phase_steps)
        command = [
            float(start + alpha * (target - start))
            for start, target in zip(start_positions, target_positions)
        ]
        _send_position_targets(dc, arm_dofs, command)
        sim_app.update()

    observed = _current_positions(dc, arm_dofs)
    ee_position = _body_pose_position(dc, end_effector_body)
    row = {
        "phase": phase_name,
        "right_arm_dof_names": [name for _, _, name in arm_dofs],
        "commanded_joint_target_values": _named_positions(arm_dofs, target_positions),
        "observed_joint_values": _named_positions(arm_dofs, observed),
        "end_effector_position": ee_position.tolist(),
    }
    phase_log.append(row)
    print(f"phase={phase_name} end_effector={row['end_effector_position']}")
    return row


def _pause_phase(
    phase_name: str,
    dc: Any,
    end_effector_body: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    steps: int,
    phase_log: list[dict[str, Any]],
) -> dict[str, Any]:
    for _ in range(steps):
        sim_app.update()
    row = {
        "phase": phase_name,
        "end_effector_position": _body_pose_position(dc, end_effector_body).tolist(),
        "observed_gripper_joint_values": _named_positions(gripper_dofs, _read_positions(dc, gripper_dofs)),
    }
    phase_log.append(row)
    print(f"phase={phase_name}")
    return row


def _command_gripper_phase(
    phase_name: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    base_positions: list[float],
    delta: float,
    sim_app: Any,
    steps: int,
    phase_log: list[dict[str, Any]],
) -> dict[str, Any]:
    targets: list[float] = []
    for base_position, (_, _, name) in zip(base_positions, gripper_dofs):
        sign = -1.0 if "finger2" in name.lower() else 1.0
        targets.append(float(base_position + sign * delta))
    _send_position_targets(dc, gripper_dofs, targets)
    for _ in range(steps):
        sim_app.update()
    observed = _read_positions(dc, gripper_dofs)
    row = {
        "phase": phase_name,
        "right_gripper_dof_names": [name for _, _, name in gripper_dofs],
        "commanded_joint_target_values": _named_positions(gripper_dofs, targets),
        "observed_joint_values": _named_positions(gripper_dofs, observed),
    }
    phase_log.append(row)
    print(f"phase={phase_name} gripper={row['observed_joint_values']}")
    return row


def _write_log(log_root: Path, payload: dict[str, Any]) -> Path:
    log_path = log_root / LOG_NAME
    lines = [
        "status=inspection_complete",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        "not_a_scored_baseline=true",
        "task_success_verified=false",
        f"yaml_path={payload['yaml_path']}",
        f"root_path_override={payload['root_path_override']}",
        f"selected_target_part_prim={payload['target_part']['prim_path']}",
        f"robot_usd_path={payload['robot']['usd_path']}",
        f"phase_order={payload['phase_order']}",
        f"payload={json.dumps(payload, indent=2, sort_keys=True)}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--asset-root")
    parser.add_argument("--robot-usd", help="Path to Walker S2 USD. Defaults to WALKER_S2_USD.")
    parser.add_argument("--prim-path", default=DEFAULT_PRIM_PATH)
    parser.add_argument("--end-effector-body")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
    parser.add_argument("--pause-steps", type=int, default=DEFAULT_PAUSE_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-gripper-close", action="store_true")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument(
        "--use-config-robot-pose",
        action="store_true",
        help="Opt in to Part_Sorting.yaml robot pose. Default keeps Walker S2 at the previously validated origin pose.",
    )
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.init_steps < 1 or args.phase_steps < 1 or args.pause_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --phase-steps, --pause-steps, and --settle-steps must be positive")
    if args.target_index < 0:
        raise RuntimeError("--target-index must be non-negative")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be positive")
    if args.gripper_delta <= 0.0:
        raise RuntimeError("--gripper-delta must be positive")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    baseline_root = _as_path(args.baseline_root, paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
    asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
    config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
    box_usd = asset_root / DEFAULT_BOX_RELATIVE
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])
    if config_path.name != "Part_Sorting.yaml":
        raise RuntimeError(f"Wrong Task 1 config; expected Part_Sorting.yaml, got {config_path}")
    if not config_path.exists():
        raise RuntimeError(f"Task 1 config missing: {config_path}")
    if not asset_root.exists():
        raise RuntimeError(f"Asset root missing: {asset_root}")
    if not box_usd.exists():
        raise RuntimeError(f"Diagnostic bin visual USD missing: {box_usd}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
    timeline = None

    try:
        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        original_root_path = cfg.get("root_path")
        cfg["root_path"] = str(asset_root)
        apply_scatter_config(cfg)

        import omni.replicator.core as rep  # type: ignore

        if hasattr(rep, "set_global_seed"):
            rep.set_global_seed(args.seed)

        stage = _create_minimal_scene()
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        scene.build_table()
        scene.build_parts()

        bin_position = [float(value) for value in cfg["box"]["box_position"][0]]
        bin_scale = [float(value) for value in cfg["box"]["box_scale"][0]]
        _add_reference(stage, "/World/DiagnosticBinVisual", box_usd)
        _set_xform(stage, "/World/DiagnosticBinVisual", bin_position, scale=bin_scale)
        for _ in range(5):
            sim_app.update()
        bin_visual_bbox = _bbox(stage, "/World/DiagnosticBinVisual")
        removed_bin_visual_physics = _disable_physics_under(stage, "/World/DiagnosticBinVisual")
        bin_collider = _add_static_bin_colliders(stage, bin_visual_bbox)
        bin_after_bbox = _bbox(stage, "/World/DiagnosticBinVisual")

        _load_robot_reference(stage, robot_usd, args.prim_path)
        robot_cfg = cfg.get("robot", {})
        configured_robot_position = [float(value) for value in robot_cfg.get("robot_position", [0.0, 0.0, 0.0])]
        configured_robot_rotation = [float(value) for value in robot_cfg.get("robot_rotation", [0.0, 0.0, 0.0])]
        robot_position = configured_robot_position if args.use_config_robot_pose else [0.0, 0.0, 0.0]
        robot_rotation = configured_robot_rotation if args.use_config_robot_pose else [0.0, 0.0, 0.0]
        _set_xform(stage, args.prim_path, robot_position, rotation_xyz_deg=robot_rotation)

        rep.orchestrator.step()
        for _ in range(args.init_steps):
            sim_app.update()

        part_paths = list(getattr(scene, "parts_prim_paths", []))
        if not part_paths:
            raise RuntimeError("SceneBuilder did not expose any Task 1 part prim paths")
        if args.target_index >= len(part_paths):
            raise RuntimeError(f"--target-index {args.target_index} out of range for {len(part_paths)} parts")

        target_path = part_paths[args.target_index]
        target_prim = stage.GetPrimAtPath(target_path)
        target_refs = _reference_paths(target_prim) if target_prim and target_prim.IsValid() else []
        category_from_refs = _category_from_reference(target_refs)
        num_parts_per_class = int(cfg["part"].get("num_parts", 2))
        category_from_order = "part_a" if args.target_index < num_parts_per_class else "part_b"
        category = category_from_refs if category_from_refs != "unknown" else "not yet verified"
        target_initial_bbox = _bbox(stage, target_path)
        target_center = target_initial_bbox["center"]
        pre_grasp_marker = [target_center[0], target_center[1], target_initial_bbox["max"][2] + 0.12]

        marker_paths = [
            _create_stage_marker(stage, "/World/DebugTargetPart", target_center, 0.025, (1.0, 0.2, 0.1)),
            _create_stage_marker(stage, "/World/DebugPreGrasp", pre_grasp_marker, 0.025, (0.2, 0.6, 1.0)),
            _create_stage_marker(stage, "/World/DebugBinCenter", bin_after_bbox["center"], 0.03, (0.2, 1.0, 0.2)),
        ]
        marker_paths.extend(_create_bbox_corner_markers(stage, "/World/DebugBin", bin_after_bbox, (0.1, 0.9, 0.1)))

        articulation_roots: list[str] = []
        joint_names: list[str] = []
        for _ in range(args.init_steps):
            sim_app.update()
            articulation_roots = _find_articulation_roots(stage, args.prim_path)
            joint_names = _find_joint_names(stage, args.prim_path)
            if articulation_roots and joint_names:
                break
        if not articulation_roots:
            raise RuntimeError("Walker S2 loaded, but no articulation root was detected")

        articulation_path = articulation_roots[0]
        timeline = _start_timeline()
        for _ in range(args.pause_steps):
            sim_app.update()

        dc, articulation = _acquire_articulation(articulation_path)
        dof_observation = _read_dof_observation(dc, articulation)
        arm_dofs = _select_right_arm_dofs(dc, articulation, args.max_arm_dofs)
        gripper_dofs = _select_right_gripper_dofs(dc, articulation)
        end_effector_body, end_effector_name, end_effector_path = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
        )
        gripper_initial = _read_positions(dc, gripper_dofs)
        front_targets = _targets_from_map(arm_dofs, FRONT_POSE_BY_NAME)
        pre_grasp_targets = _targets_from_map(arm_dofs, ABOVE_FRONT_TARGET_BY_NAME)
        descend_targets = _targets_from_map(arm_dofs, DOWN_FRONT_TARGET_BY_NAME)
        bin_targets = _targets_from_map(arm_dofs, BIN_TARGET_BY_NAME)

        phase_log: list[dict[str, Any]] = []
        phase_order = [
            "hold_initial_view",
            "move_to_front_pose",
            "move_to_pre_grasp",
            "pause",
            "descend",
            "pause",
            "close_gripper",
            "pause",
            "lift",
            "pause",
            "move_to_bin",
            "pause",
            "open_gripper",
            "pause",
            "settle",
        ]
        _pause_phase("hold_initial_view", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        _command_arm_phase("move_to_front_pose", dc, arm_dofs, end_effector_body, front_targets, sim_app, args.phase_steps, phase_log)
        _command_arm_phase("move_to_pre_grasp", dc, arm_dofs, end_effector_body, pre_grasp_targets, sim_app, args.phase_steps, phase_log)
        _pause_phase("pause_after_pre_grasp", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        _command_arm_phase("descend", dc, arm_dofs, end_effector_body, descend_targets, sim_app, args.phase_steps, phase_log)
        _pause_phase("pause_after_descend", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        if args.skip_gripper_close:
            _pause_phase("close_gripper_skipped", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        else:
            _command_gripper_phase("close_gripper", dc, gripper_dofs, gripper_initial, -args.gripper_delta, sim_app, args.pause_steps, phase_log)
        _pause_phase("pause_after_close", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        _command_arm_phase("lift", dc, arm_dofs, end_effector_body, pre_grasp_targets, sim_app, args.phase_steps, phase_log)
        _pause_phase("pause_after_lift", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        _command_arm_phase("move_to_bin", dc, arm_dofs, end_effector_body, bin_targets, sim_app, args.phase_steps, phase_log)
        _pause_phase("pause_after_move_to_bin", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        if args.skip_release:
            _pause_phase("open_gripper_skipped", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        else:
            _command_gripper_phase("open_gripper", dc, gripper_dofs, gripper_initial, args.gripper_delta, sim_app, args.pause_steps, phase_log)
        _pause_phase("pause_after_open", dc, end_effector_body, gripper_dofs, sim_app, args.pause_steps, phase_log)
        _pause_phase("settle", dc, end_effector_body, gripper_dofs, sim_app, args.settle_steps, phase_log)

        final_bbox = _bbox(stage, target_path)
        final_ee = _body_pose_position(dc, end_effector_body).tolist()
        marker_paths.append(_create_stage_marker(stage, "/World/DebugFinalEndEffector", final_ee, 0.02, (1.0, 1.0, 0.1)))
        payload = {
            "status": "inspection_complete",
            "not_a_scored_baseline": True,
            "task_success_verified": False,
            "yaml_path": str(config_path),
            "root_path_original": original_root_path,
            "root_path_override": str(asset_root),
            "seed": args.seed,
            "scene_builder_methods": ["build_table", "build_parts"],
            "official_box_pipeline_used": False,
            "diagnostic_bin_strategy": "official_box_visual_with_physics_stripped_plus_static_floor_wall_colliders",
            "diagnostic_bin_visual_usd": str(box_usd),
            "diagnostic_bin_removed_visual_physics": removed_bin_visual_physics,
            "bin": {
                "configured_position": bin_position,
                "configured_scale": bin_scale,
                "visual_bbox": bin_after_bbox,
                "collider": bin_collider,
            },
            "table": {
                "configured_usd": cfg["table"]["table_usd"],
                "scene_table_path": "/Replicator/Ref_Xform",
                "bbox": _bbox(stage, "/Replicator/Ref_Xform"),
                "physics": _physics_summary(stage, "/Replicator/Ref_Xform"),
            },
            "parts": {
                "all_part_paths": part_paths,
                "target_index": args.target_index,
            },
            "target_part": {
                "prim_path": target_path,
                "referenced_usd_paths": target_refs,
                "category_from_reference": category_from_refs,
                "category_from_scene_builder_order": category_from_order,
                "category_for_log": category,
                "initial_bbox": target_initial_bbox,
                "final_bbox": final_bbox,
                "initial_pose": {"bbox_center": target_initial_bbox["center"]},
                "final_pose": {"bbox_center": final_bbox["center"]},
            },
            "robot": {
                "usd_path": str(robot_usd),
                "prim_path": args.prim_path,
                "configured_position_from_yaml": configured_robot_position,
                "configured_rotation_xyz_deg_from_yaml": configured_robot_rotation,
                "config_robot_pose_applied": bool(args.use_config_robot_pose),
                "position_applied": robot_position,
                "rotation_xyz_deg_applied": robot_rotation,
                "articulation_path": articulation_path,
                "joint_count": len(joint_names),
                "right_arm_dof_names": [name for _, _, name in arm_dofs],
                "right_gripper_dof_names": [name for _, _, name in gripper_dofs],
                "end_effector_name": end_effector_name,
                "end_effector_path": end_effector_path,
                "final_end_effector_position": final_ee,
                "dof_observation_sample": dof_observation[:12],
            },
            "phase_order": phase_order,
            "phase_log": phase_log,
            "debug_marker_paths": marker_paths,
            "runtime_warnings_detected_by_script": [
                "official box physics pipeline intentionally not used",
                "script does not verify object grasp or scored placement",
                "manual GUI inspection is still required",
            ],
            "final_status": "inspection_complete; not a scored baseline; not yet verified for task success",
        }
        log_path = _write_log(paths["LOG_ROOT"], payload)
        print(f"Task 1 GUI inspection complete; wrote {log_path}")
        print("status=inspection_complete not_a_scored_baseline=true task_success_verified=false")

        if (args.no_headless or args.gui) and args.hold_open:
            _hold_gui_open(sim_app)
    finally:
        if timeline is not None:
            timeline.stop()
        sim_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Task 1 GUI pick/place inspection FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
