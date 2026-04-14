#!/usr/bin/env python3
"""Minimal Walker S2 single-object grasp primitive smoke test in Isaac Sim.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script validates the next deterministic primitive after Cartesian reaching:
open the right gripper, move above one fixed cube target, descend, close the
gripper, and lift vertically. It intentionally contains no competition task
logic, dataset use, perception, or learning code.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from control_walker_s2_arms import (  # type: ignore
    _acquire_articulation,
    _hold_gui_open,
    _send_position_targets,
    _start_timeline,
)
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
    DEFAULT_DAMPING,
    DEFAULT_IK_STEPS,
    DEFAULT_MAX_STEP,
    DEFAULT_POSITION_EPS,
    DEFAULT_SETTLE_STEPS,
    _body_pose_position,
    _clamp_target_to_front_workspace,
    _create_front_workspace_debug,
    _current_positions,
    _identify_end_effector_body,
    _named_positions,
    _select_right_arm_dofs,
    move_end_effector_through_waypoints,
)


DEFAULT_OBJECT_POSITION = np.array([1.2711199712753296, -0.2775531601905823, 0.8209833312034607])
DEFAULT_OBJECT_SIZE = 0.04
DEFAULT_PRE_GRASP_OFFSET = np.array([0.0, 0.0, 0.08])
DEFAULT_GRASP_OFFSET = np.array([0.0, 0.0, 0.04])
DEFAULT_LIFT_OFFSET = np.array([0.0, 0.0, 0.12])
DEFAULT_GRIPPER_DELTA = 0.03
DEFAULT_GRIPPER_SETTLE_STEPS = 20
DEFAULT_POSE_TOLERANCE = 0.12
DEFAULT_LIFT_MIN_DELTA = 0.03
RIGHT_GRIPPER_TOKENS = ("r_finger", "right_finger", "r_thumb", "right_thumb", "r_gripper", "right_gripper")
LEFT_GRIPPER_TOKENS = ("l_finger", "left_finger", "l_thumb", "left_thumb", "l_gripper", "left_gripper")


def _create_fixed_target_cube(stage: Any, object_position: np.ndarray, object_size: float) -> str:
    from pxr import Gf, UsdGeom  # type: ignore

    target_path = "/World/FixedGraspTarget"
    cube = UsdGeom.Cube.Define(stage, target_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(float(object_size), float(object_size), float(object_size)))
    cube.AddTranslateOp().Set(
        Gf.Vec3d(
            float(object_position[0]),
            float(object_position[1]),
            float(object_position[2]),
        )
    )
    return target_path


def _select_right_gripper_dofs(dc: Any, articulation: Any) -> list[tuple[int, Any, str]]:
    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        lower = name.lower()
        if any(token in lower for token in LEFT_GRIPPER_TOKENS):
            continue
        if any(token in lower for token in RIGHT_GRIPPER_TOKENS):
            selected.append((index, dof, name))

    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
        raise RuntimeError(
            "No right gripper/finger DOFs matched the simple name filter. "
            f"Tokens={RIGHT_GRIPPER_TOKENS}; available_dof_names={all_names}"
        )
    return selected


def _read_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> list[float]:
    return [float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs]


def _command_gripper(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    base_positions: list[float],
    delta: float,
    sim_app: Any,
    settle_steps: int,
) -> list[float]:
    targets: list[float] = []
    for base_position, (_, _, name) in zip(base_positions, selected_dofs):
        sign = -1.0 if "finger2" in name.lower() else 1.0
        targets.append(float(base_position + sign * delta))

    _send_position_targets(dc, selected_dofs, targets)
    for _ in range(settle_steps):
        sim_app.update()
    return targets


def _move_to_pose(
    label: str,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    target_position: np.ndarray,
    sim_app: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    actual_position, error = move_end_effector_through_waypoints(
        dc,
        articulation,
        arm_dofs,
        end_effector_body,
        [(label, _clamp_target_to_front_workspace(target_position))],
        sim_app,
        args.ik_steps,
        args.settle_steps,
        args.position_eps,
        args.damping,
        args.max_step,
        args.posture_gain,
        args.stop_tolerance,
        args.hold_steps,
        args.ik_trace,
    )
    print(f"{label}_target={target_position.tolist()}")
    print(f"{label}_actual={actual_position.tolist()}")
    print(f"{label}_error={error}")
    if error > args.pose_tolerance:
        raise RuntimeError(
            f"{label} exceeded pose tolerance: error={error}, tolerance={args.pose_tolerance}"
        )
    return actual_position, error


def _write_log(
    log_root: Path,
    robot_usd: Path,
    robot_prim_path: str,
    articulation_path: str,
    target_path: str,
    object_position: np.ndarray,
    object_size: float,
    pre_grasp_pose: np.ndarray,
    grasp_pose: np.ndarray,
    lift_pose: np.ndarray,
    release_pose: np.ndarray,
    actual_pre_grasp: np.ndarray,
    actual_grasp: np.ndarray,
    actual_lift: np.ndarray,
    actual_release: np.ndarray,
    end_effector_name: str,
    end_effector_path: str,
    arm_names: list[str],
    arm_initial_positions: dict[str, float],
    arm_final_positions: dict[str, float],
    gripper_names: list[str],
    gripper_initial: list[float],
    gripper_open_targets: list[float],
    gripper_open_positions: list[float],
    gripper_close_targets: list[float],
    gripper_close_positions: list[float],
    gripper_release_targets: list[float],
    gripper_release_positions: list[float],
    gripper_verified: bool,
    lift_succeeded: bool,
    lift_delta: float,
    debug_marker_paths: list[str],
    ik_trace: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "walker_s2_static_grasp_smoke.log"
    log_file.write_text(
        "\n".join(
            (
                "status=walker_s2_static_grasp_smoke_ok",
                f"timestamp_utc={timestamp}",
                f"robot_usd_path={robot_usd}",
                f"robot_prim_path={robot_prim_path}",
                f"articulation_path={articulation_path}",
                f"target_path={target_path}",
                f"target_pose={object_position.tolist()}",
                f"target_size={object_size}",
                f"pre_grasp_pose={pre_grasp_pose.tolist()}",
                f"grasp_pose={grasp_pose.tolist()}",
                f"lift_pose={lift_pose.tolist()}",
                f"release_pose={release_pose.tolist()}",
                f"actual_pre_grasp_pose={actual_pre_grasp.tolist()}",
                f"actual_grasp_pose={actual_grasp.tolist()}",
                f"actual_lift_pose={actual_lift.tolist()}",
                f"actual_release_pose={actual_release.tolist()}",
                f"end_effector_name={end_effector_name}",
                f"end_effector_path={end_effector_path}",
                f"right_arm_dof_names={arm_names}",
                f"right_arm_initial_positions={arm_initial_positions}",
                f"right_arm_final_positions={arm_final_positions}",
                f"gripper_dof_names={gripper_names}",
                f"gripper_initial_positions={gripper_initial}",
                f"gripper_open_targets={gripper_open_targets}",
                f"gripper_open_positions={gripper_open_positions}",
                f"gripper_close_targets={gripper_close_targets}",
                f"gripper_close_positions={gripper_close_positions}",
                f"gripper_release_targets={gripper_release_targets}",
                f"gripper_release_positions={gripper_release_positions}",
                f"gripper_verified={str(gripper_verified).lower()}",
                f"lift_delta={lift_delta}",
                f"lift_succeeded={str(lift_succeeded).lower()}",
                f"debug_marker_paths={debug_marker_paths}",
                f"ik_trace={ik_trace}",
                "motion_sequence=approach_from_front -> move_down -> grasp -> lift_up -> move_down_release",
                "end_effector_assumption=Using the right wrist-roll link as the grasp frame because no palm/hand rigid body was identified by the current body-name filter.",
                "assumption=static target cube is a pose target; lift success measures end-effector vertical lift after gripper close, not object transport.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", help="Path to the Walker S2 root USD file. Defaults to WALKER_S2_USD.")
    parser.add_argument("--prim-path", default=DEFAULT_PRIM_PATH)
    parser.add_argument("--end-effector-body", help="Optional end-effector body name or path.")
    parser.add_argument("--object-position", nargs=3, type=float, default=DEFAULT_OBJECT_POSITION.tolist())
    parser.add_argument("--object-size", type=float, default=DEFAULT_OBJECT_SIZE)
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--ik-steps", type=int, default=DEFAULT_IK_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--gripper-settle-steps", type=int, default=DEFAULT_GRIPPER_SETTLE_STEPS)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--position-eps", type=float, default=DEFAULT_POSITION_EPS)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    parser.add_argument("--posture-gain", type=float, default=0.04)
    parser.add_argument("--stop-tolerance", type=float, default=0.015)
    parser.add_argument("--hold-steps", type=int, default=16)
    parser.add_argument("--pose-tolerance", type=float, default=DEFAULT_POSE_TOLERANCE)
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA)
    parser.add_argument("--lift-min-delta", type=float, default=DEFAULT_LIFT_MIN_DELTA)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.object_size <= 0.0:
        raise RuntimeError("--object-size must be positive")
    if args.init_steps < 1 or args.ik_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --ik-steps, and --settle-steps must be at least 1")
    if args.gripper_settle_steps < 1:
        raise RuntimeError("--gripper-settle-steps must be at least 1")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be at least 1")
    if args.gripper_delta <= 0.0:
        raise RuntimeError("--gripper-delta must be positive")
    if args.posture_gain < 0.0:
        raise RuntimeError("--posture-gain must be non-negative")
    if args.stop_tolerance <= 0.0:
        raise RuntimeError("--stop-tolerance must be positive")
    if args.hold_steps < 1:
        raise RuntimeError("--hold-steps must be at least 1")
    if args.lift_min_delta <= 0.0:
        raise RuntimeError("--lift-min-delta must be positive")
    if args.pose_tolerance <= 0.0:
        raise RuntimeError("--pose-tolerance must be positive")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path, for example /World/WalkerS2")

    # Prevent Isaac Kit from consuming this script's CLI arguments.
    sys.argv = [sys.argv[0]]

    paths = _validate_environment()
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])
    object_position = np.array(args.object_position, dtype=float)
    pre_grasp_pose = object_position + DEFAULT_PRE_GRASP_OFFSET
    grasp_pose = object_position + DEFAULT_GRASP_OFFSET
    lift_pose = object_position + DEFAULT_LIFT_OFFSET
    release_pose = grasp_pose.copy()

    show_gui = args.no_headless or args.gui
    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not show_gui})
    timeline = None

    try:
        stage = _create_minimal_scene()
        target_path = _create_fixed_target_cube(stage, object_position, args.object_size)
        _load_robot_reference(stage, robot_usd, args.prim_path)
        sim_app.update()

        articulation_roots: list[str] = []
        joint_names: list[str] = []
        for step in range(args.init_steps):
            sim_app.update()
            articulation_roots = _find_articulation_roots(stage, args.prim_path)
            joint_names = _find_joint_names(stage, args.prim_path)
            if articulation_roots and joint_names:
                break
            print(f"init_step={step + 1}/{args.init_steps}")

        if not articulation_roots:
            raise RuntimeError("Robot loaded, but no articulation root API was detected.")
        if not joint_names:
            raise RuntimeError("Robot articulation loaded, but no joint prims were detected.")

        articulation_path = articulation_roots[0]
        timeline = _start_timeline()
        for _ in range(args.settle_steps):
            sim_app.update()

        dc, articulation = _acquire_articulation(articulation_path)
        arm_dofs = _select_right_arm_dofs(dc, articulation, args.max_arm_dofs)
        arm_names = [name for _, _, name in arm_dofs]
        arm_initial_positions = _named_positions(arm_dofs, _current_positions(dc, arm_dofs))
        gripper_dofs = _select_right_gripper_dofs(dc, articulation)
        gripper_names = [name for _, _, name in gripper_dofs]
        gripper_initial = _read_positions(dc, gripper_dofs)
        gripper_open_targets = _command_gripper(
            dc,
            gripper_dofs,
            gripper_initial,
            args.gripper_delta,
            sim_app,
            args.gripper_settle_steps,
        )
        gripper_open_positions = _read_positions(dc, gripper_dofs)
        open_motion = max(abs(value - base) for value, base in zip(gripper_open_positions, gripper_initial))

        end_effector_body, end_effector_name, end_effector_path = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
        )
        debug_marker_paths = _create_front_workspace_debug(
            stage,
            _clamp_target_to_front_workspace(grasp_pose),
            _body_pose_position(dc, end_effector_body),
        )
        args.ik_trace = []

        print(f"end_effector_name={end_effector_name}")
        print(f"end_effector_path={end_effector_path}")
        print(f"right_arm_dof_names={arm_names}")
        print(f"right_arm_initial_positions={arm_initial_positions}")
        print(f"debug_marker_paths={debug_marker_paths}")

        actual_pre_grasp, _ = _move_to_pose(
            "approach_from_front",
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            pre_grasp_pose,
            sim_app,
            args,
        )
        actual_grasp, _ = _move_to_pose(
            "move_down_grasp",
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            grasp_pose,
            sim_app,
            args,
        )
        gripper_close_targets = _command_gripper(
            dc,
            gripper_dofs,
            gripper_initial,
            -args.gripper_delta,
            sim_app,
            args.gripper_settle_steps,
        )
        gripper_close_positions = _read_positions(dc, gripper_dofs)
        close_motion = max(abs(value - opened) for value, opened in zip(gripper_close_positions, gripper_open_positions))
        gripper_verified = open_motion > 1.0e-4 and close_motion > 1.0e-4
        if not gripper_verified:
            raise RuntimeError(
                "Right gripper open/close command produced too little observed motion: "
                f"open_motion={open_motion}, close_motion={close_motion}"
            )

        actual_lift, _ = _move_to_pose(
            "lift_up",
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            lift_pose,
            sim_app,
            args,
        )
        lift_delta = float(actual_lift[2] - actual_grasp[2])
        lift_succeeded = lift_delta >= args.lift_min_delta
        if not lift_succeeded:
            raise RuntimeError(
                "End-effector did not lift enough after gripper close: "
                f"lift_delta={lift_delta}, lift_min_delta={args.lift_min_delta}"
            )

        actual_release, _ = _move_to_pose(
            "move_down_release",
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            release_pose,
            sim_app,
            args,
        )
        gripper_release_targets = _command_gripper(
            dc,
            gripper_dofs,
            gripper_initial,
            args.gripper_delta,
            sim_app,
            args.gripper_settle_steps,
        )
        gripper_release_positions = _read_positions(dc, gripper_dofs)
        arm_final_positions = _named_positions(arm_dofs, _current_positions(dc, arm_dofs))

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_path,
            target_path,
            object_position,
            args.object_size,
            pre_grasp_pose,
            grasp_pose,
            lift_pose,
            release_pose,
            actual_pre_grasp,
            actual_grasp,
            actual_lift,
            actual_release,
            end_effector_name,
            end_effector_path,
            arm_names,
            arm_initial_positions,
            arm_final_positions,
            gripper_names,
            gripper_initial,
            gripper_open_targets,
            gripper_open_positions,
            gripper_close_targets,
            gripper_close_positions,
            gripper_release_targets,
            gripper_release_positions,
            gripper_verified,
            lift_succeeded,
            lift_delta,
            debug_marker_paths,
            args.ik_trace,
        )
        print(f"Walker S2 static grasp smoke OK; wrote {log_file}")

        if show_gui and args.hold_open:
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
        print(f"Walker S2 static grasp smoke FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
