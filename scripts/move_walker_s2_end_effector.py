#!/usr/bin/env python3
"""Minimal Walker S2 end-effector target smoke test in Isaac Sim.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script verifies a small Cartesian-control milestone: identify an arm
end-effector body, move it toward one 3D target with a simple damped least
squares IK loop, and log target-vs-actual position. It intentionally contains no
task logic, object manipulation, perception, dataset use, or learning code.
"""

from __future__ import annotations

import argparse
import math
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


DEFAULT_IK_STEPS = 24
DEFAULT_SETTLE_STEPS = 8
DEFAULT_TARGET_OFFSET = np.array([0.02, -0.02, 0.02], dtype=float)
DEFAULT_ERROR_TOLERANCE = 0.08
DEFAULT_POSITION_EPS = 0.01
DEFAULT_DAMPING = 0.05
DEFAULT_MAX_STEP = 0.04
DEFAULT_FRONT_X_LIMITS = (0.75, 1.35)
DEFAULT_RIGHT_Y_LIMITS = (-0.65, -0.05)
DEFAULT_Z_LIMITS = (0.55, 1.05)
DEFAULT_POSTURE_GAIN = 0.04
DEFAULT_STOP_TOLERANCE = 0.015
DEFAULT_HOLD_STEPS = 16
RIGHT_ARM_TOKENS = ("r_shoulder", "r_elbow", "r_wrist", "right_shoulder", "right_elbow", "right_wrist")
END_EFFECTOR_TOKENS = ("r_hand", "right_hand", "r_palm", "right_palm", "r_wrist", "right_wrist")
END_EFFECTOR_EXCLUDE_TOKENS = ("finger", "thumb", "sensor", "camera")


def _vector3(value: Any) -> np.ndarray:
    try:
        return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)
    except AttributeError:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def _body_pose_position(dc: Any, body: Any) -> np.ndarray:
    pose = dc.get_rigid_body_pose(body)
    return _vector3(pose.p)


def _list_articulation_bodies(dc: Any, articulation: Any) -> list[tuple[int, Any, str, str]]:
    bodies: list[tuple[int, Any, str, str]] = []
    for index in range(dc.get_articulation_body_count(articulation)):
        body = dc.get_articulation_body(articulation, index)
        bodies.append(
            (
                index,
                body,
                str(dc.get_rigid_body_name(body)),
                str(dc.get_rigid_body_path(body)),
            )
        )
    return bodies


def _identify_end_effector_body(
    dc: Any,
    articulation: Any,
    requested_body: str | None,
) -> tuple[Any, str, str]:
    bodies = _list_articulation_bodies(dc, articulation)

    if requested_body:
        requested_lower = requested_body.lower()
        for _, body, name, path in bodies:
            if requested_body == name or requested_body == path or requested_lower in path.lower():
                return body, name, path
        raise RuntimeError(
            f"Requested end-effector body was not found: {requested_body}. "
            f"Available body paths={[path for _, _, _, path in bodies]}"
        )

    candidates: list[tuple[int, Any, str, str]] = []
    for index, body, name, path in bodies:
        lower = f"{name} {path}".lower()
        if any(token in lower for token in END_EFFECTOR_EXCLUDE_TOKENS):
            continue
        if any(token in lower for token in END_EFFECTOR_TOKENS):
            candidates.append((index, body, name, path))

    if not candidates:
        raise RuntimeError(
            "Could not identify a right-arm end-effector body. "
            f"Tokens={END_EFFECTOR_TOKENS}; available body paths={[path for _, _, _, path in bodies]}"
        )

    _, body, name, path = candidates[-1]
    return body, name, path


def _select_right_arm_dofs(dc: Any, articulation: Any, max_dofs: int) -> list[tuple[int, Any, str]]:
    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        lower = name.lower()
        if any(token in lower for token in RIGHT_ARM_TOKENS):
            selected.append((index, dof, name))
        if len(selected) >= max_dofs:
            break

    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
        raise RuntimeError(
            "No right-arm DOFs matched the simple name filter. "
            f"Tokens={RIGHT_ARM_TOKENS}; available_dof_names={all_names}"
        )
    return selected


def _current_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> np.ndarray:
    return np.array([float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs], dtype=float)


def _named_positions(selected_dofs: list[tuple[int, Any, str]], positions: np.ndarray) -> dict[str, float]:
    return {name: float(position) for (_, _, name), position in zip(selected_dofs, positions)}


def _clamp_target_to_front_workspace(target_position: np.ndarray) -> np.ndarray:
    clamped = np.array(target_position, dtype=float).copy()
    clamped[0] = float(np.clip(clamped[0], *DEFAULT_FRONT_X_LIMITS))
    clamped[1] = float(np.clip(clamped[1], *DEFAULT_RIGHT_Y_LIMITS))
    clamped[2] = float(np.clip(clamped[2], *DEFAULT_Z_LIMITS))
    return clamped


def _front_workspace_ok(position: np.ndarray) -> bool:
    return (
        DEFAULT_FRONT_X_LIMITS[0] <= float(position[0]) <= DEFAULT_FRONT_X_LIMITS[1]
        and DEFAULT_RIGHT_Y_LIMITS[0] <= float(position[1]) <= DEFAULT_RIGHT_Y_LIMITS[1]
        and DEFAULT_Z_LIMITS[0] <= float(position[2]) <= DEFAULT_Z_LIMITS[1]
    )


def _create_debug_marker(stage: Any, path: str, position: np.ndarray, radius: float, color: tuple[float, float, float]) -> str:
    from pxr import Gf, UsdGeom  # type: ignore

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    return path


def _create_front_workspace_debug(stage: Any, target_position: np.ndarray, initial_position: np.ndarray) -> list[str]:
    marker_paths = [
        _create_debug_marker(stage, "/World/DebugInitialEndEffector", initial_position, 0.025, (0.2, 0.6, 1.0)),
        _create_debug_marker(stage, "/World/DebugTargetEndEffector", target_position, 0.03, (1.0, 0.2, 0.1)),
    ]
    for index, corner in enumerate(
        (
            (DEFAULT_FRONT_X_LIMITS[0], DEFAULT_RIGHT_Y_LIMITS[0], DEFAULT_Z_LIMITS[0]),
            (DEFAULT_FRONT_X_LIMITS[1], DEFAULT_RIGHT_Y_LIMITS[1], DEFAULT_Z_LIMITS[1]),
        )
    ):
        marker_paths.append(
            _create_debug_marker(
                stage,
                f"/World/DebugFrontWorkspaceCorner{index}",
                np.array(corner, dtype=float),
                0.015,
                (0.1, 0.9, 0.2),
            )
        )
    return marker_paths


def _apply_positions(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    positions: np.ndarray,
    sim_app: Any,
    settle_steps: int,
) -> None:
    _send_position_targets(dc, selected_dofs, [float(value) for value in positions])
    for _ in range(settle_steps):
        sim_app.update()


def _estimate_position_jacobian(
    dc: Any,
    articulation: Any,
    selected_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    base_positions: np.ndarray,
    base_ee_position: np.ndarray,
    sim_app: Any,
    eps: float,
    settle_steps: int,
) -> np.ndarray:
    jacobian = np.zeros((3, len(selected_dofs)), dtype=float)
    for column in range(len(selected_dofs)):
        trial = base_positions.copy()
        trial[column] += eps
        _apply_positions(dc, selected_dofs, trial, sim_app, settle_steps)
        dc.wake_up_articulation(articulation)
        moved_position = _body_pose_position(dc, end_effector_body)
        jacobian[:, column] = (moved_position - base_ee_position) / eps

    _apply_positions(dc, selected_dofs, base_positions, sim_app, settle_steps)
    dc.wake_up_articulation(articulation)
    return jacobian


def move_end_effector_to_target(
    dc: Any,
    articulation: Any,
    selected_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    target_position: np.ndarray,
    sim_app: Any,
    ik_steps: int,
    settle_steps: int,
    eps: float,
    damping: float,
    max_step: float,
    posture_positions: np.ndarray | None = None,
    posture_gain: float = DEFAULT_POSTURE_GAIN,
    stop_tolerance: float = DEFAULT_STOP_TOLERANCE,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    phase_label: str = "target",
    trace: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, float]:
    positions = _current_positions(dc, selected_dofs)
    posture = positions.copy() if posture_positions is None else posture_positions.copy()
    best_position = _body_pose_position(dc, end_effector_body)
    best_error = float(np.linalg.norm(target_position - best_position))
    best_joint_positions = positions.copy()

    for step in range(ik_steps):
        current_ee_position = _body_pose_position(dc, end_effector_body)
        error_vector = target_position - current_ee_position
        position_error = float(np.linalg.norm(error_vector))
        if position_error < best_error:
            best_position = current_ee_position.copy()
            best_error = position_error
            best_joint_positions = positions.copy()
        if position_error <= stop_tolerance:
            break

        jacobian = _estimate_position_jacobian(
            dc,
            articulation,
            selected_dofs,
            end_effector_body,
            positions,
            current_ee_position,
            sim_app,
            eps,
            settle_steps,
        )
        lhs = jacobian @ jacobian.T + (damping**2) * np.eye(3)
        delta = jacobian.T @ np.linalg.solve(lhs, error_vector)
        delta += posture_gain * (posture - positions)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step:
            delta *= max_step / delta_norm

        positions = positions + delta
        _apply_positions(dc, selected_dofs, positions, sim_app, settle_steps)
        dc.wake_up_articulation(articulation)

        updated_position = _body_pose_position(dc, end_effector_body)
        observed_positions = _current_positions(dc, selected_dofs)
        updated_error = float(np.linalg.norm(target_position - updated_position))
        print(
            f"phase={phase_label} ik_step={step + 1}/{ik_steps} "
            f"target={target_position.tolist()} actual={updated_position.tolist()} "
            f"error={updated_error} commanded_joints={_named_positions(selected_dofs, positions)} "
            f"observed_joints={_named_positions(selected_dofs, observed_positions)}"
        )
        if trace is not None:
            trace.append(
                {
                    "phase": phase_label,
                    "ik_step": step + 1,
                    "target": target_position.tolist(),
                    "actual": updated_position.tolist(),
                    "error": updated_error,
                    "commanded_joint_targets": _named_positions(selected_dofs, positions),
                    "observed_joint_positions": _named_positions(selected_dofs, observed_positions),
                    "front_workspace_ok": _front_workspace_ok(updated_position),
                }
            )
        if updated_error < best_error:
            best_position = updated_position.copy()
            best_error = updated_error
            best_joint_positions = positions.copy()
        if updated_error <= stop_tolerance:
            break

    _apply_positions(dc, selected_dofs, best_joint_positions, sim_app, hold_steps)
    return best_position, best_error


def move_end_effector_through_waypoints(
    dc: Any,
    articulation: Any,
    selected_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    waypoints: list[tuple[str, np.ndarray]],
    sim_app: Any,
    ik_steps: int,
    settle_steps: int,
    eps: float,
    damping: float,
    max_step: float,
    posture_gain: float = DEFAULT_POSTURE_GAIN,
    stop_tolerance: float = DEFAULT_STOP_TOLERANCE,
    hold_steps: int = DEFAULT_HOLD_STEPS,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, float]:
    posture_positions = _current_positions(dc, selected_dofs)
    actual_position = _body_pose_position(dc, end_effector_body)
    position_error = math.inf
    for label, waypoint in waypoints:
        actual_position, position_error = move_end_effector_to_target(
            dc,
            articulation,
            selected_dofs,
            end_effector_body,
            _clamp_target_to_front_workspace(waypoint),
            sim_app,
            ik_steps,
            settle_steps,
            eps,
            damping,
            max_step,
            posture_positions,
            posture_gain,
            stop_tolerance,
            hold_steps,
            label,
            trace,
        )
        posture_positions = _current_positions(dc, selected_dofs)
    return actual_position, position_error


def _write_log(
    log_root: Path,
    robot_usd: Path,
    robot_prim_path: str,
    articulation_path: str,
    end_effector_name: str,
    end_effector_path: str,
    selected_dof_names: list[str],
    initial_joint_positions: dict[str, float],
    initial_position: np.ndarray,
    target_position: np.ndarray,
    clamped_target_position: np.ndarray,
    actual_position: np.ndarray,
    position_error: float,
    final_joint_positions: dict[str, float],
    debug_marker_paths: list[str],
    trace: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "walker_s2_end_effector_target.log"
    log_file.write_text(
        "\n".join(
            (
                "status=walker_s2_end_effector_target_ok",
                f"timestamp_utc={timestamp}",
                f"robot_usd_path={robot_usd}",
                f"robot_prim_path={robot_prim_path}",
                f"articulation_path={articulation_path}",
                f"end_effector_name={end_effector_name}",
                f"end_effector_path={end_effector_path}",
                f"selected_dof_names={selected_dof_names}",
                f"initial_joint_positions={initial_joint_positions}",
                f"initial_position={initial_position.tolist()}",
                f"target_position={target_position.tolist()}",
                f"clamped_target_position={clamped_target_position.tolist()}",
                f"actual_position={actual_position.tolist()}",
                f"position_error={position_error}",
                f"final_joint_positions={final_joint_positions}",
                f"front_workspace_limits=x{DEFAULT_FRONT_X_LIMITS}, y{DEFAULT_RIGHT_Y_LIMITS}, z{DEFAULT_Z_LIMITS}",
                f"target_front_workspace_ok={str(_front_workspace_ok(clamped_target_position)).lower()}",
                f"actual_front_workspace_ok={str(_front_workspace_ok(actual_position)).lower()}",
                f"debug_marker_paths={debug_marker_paths}",
                f"ik_trace={trace}",
                "end_effector_assumption=Using the right wrist-roll link as the grasp frame because no palm/hand rigid body was identified by the current body-name filter.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-usd",
        help="Path to the Walker S2 root USD file. Defaults to WALKER_S2_USD.",
    )
    parser.add_argument(
        "--prim-path",
        default=DEFAULT_PRIM_PATH,
        help=f"Stage prim path for the robot reference. Default: {DEFAULT_PRIM_PATH}",
    )
    parser.add_argument("--end-effector-body", help="Optional end-effector body name or path.")
    parser.add_argument(
        "--target-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Absolute target position in world meters. Defaults to initial end-effector position plus --target-offset.",
    )
    parser.add_argument(
        "--target-offset",
        nargs=3,
        type=float,
        default=DEFAULT_TARGET_OFFSET.tolist(),
        metavar=("DX", "DY", "DZ"),
        help="Fallback target offset from the initial end-effector position in meters.",
    )
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--ik-steps", type=int, default=DEFAULT_IK_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--position-eps", type=float, default=DEFAULT_POSITION_EPS)
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    parser.add_argument("--posture-gain", type=float, default=DEFAULT_POSTURE_GAIN)
    parser.add_argument("--stop-tolerance", type=float, default=DEFAULT_STOP_TOLERANCE)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--error-tolerance", type=float, default=DEFAULT_ERROR_TOLERANCE)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.init_steps < 1:
        raise RuntimeError("--init-steps must be at least 1")
    if args.ik_steps < 1:
        raise RuntimeError("--ik-steps must be at least 1")
    if args.settle_steps < 1:
        raise RuntimeError("--settle-steps must be at least 1")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be at least 1")
    if args.position_eps <= 0.0:
        raise RuntimeError("--position-eps must be positive")
    if args.damping <= 0.0:
        raise RuntimeError("--damping must be positive")
    if args.max_step <= 0.0:
        raise RuntimeError("--max-step must be positive")
    if args.posture_gain < 0.0:
        raise RuntimeError("--posture-gain must be non-negative")
    if args.stop_tolerance <= 0.0:
        raise RuntimeError("--stop-tolerance must be positive")
    if args.hold_steps < 1:
        raise RuntimeError("--hold-steps must be at least 1")
    if args.error_tolerance <= 0.0:
        raise RuntimeError("--error-tolerance must be positive")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path, for example /World/WalkerS2")

    # Prevent Isaac Kit from consuming this script's CLI arguments.
    sys.argv = [sys.argv[0]]

    paths = _validate_environment()
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])

    show_gui = args.no_headless or args.gui
    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not show_gui})
    timeline = None

    try:
        stage = _create_minimal_scene()
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
        _read_dof_observation(dc, articulation)
        selected_dofs = _select_right_arm_dofs(dc, articulation, args.max_arm_dofs)
        selected_dof_names = [name for _, _, name in selected_dofs]
        end_effector_body, end_effector_name, end_effector_path = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
        )
        initial_position = _body_pose_position(dc, end_effector_body)
        initial_joint_positions_array = _current_positions(dc, selected_dofs)
        initial_joint_positions = _named_positions(selected_dofs, initial_joint_positions_array)
        if args.target_position is not None:
            target_position = np.array(args.target_position, dtype=float)
        else:
            target_position = initial_position + np.array(args.target_offset, dtype=float)
        clamped_target_position = _clamp_target_to_front_workspace(target_position)
        debug_marker_paths = _create_front_workspace_debug(stage, clamped_target_position, initial_position)

        print(f"articulation_path={articulation_path}")
        print(f"end_effector_name={end_effector_name}")
        print(f"end_effector_path={end_effector_path}")
        print(f"selected_dof_names={selected_dof_names}")
        print(f"initial_joint_positions={initial_joint_positions}")
        print(f"initial_position={initial_position.tolist()}")
        print(f"target_position={target_position.tolist()}")
        print(f"clamped_target_position={clamped_target_position.tolist()}")
        print(f"debug_marker_paths={debug_marker_paths}")

        trace: list[dict[str, Any]] = []
        actual_position, position_error = move_end_effector_through_waypoints(
            dc,
            articulation,
            selected_dofs,
            end_effector_body,
            [
                ("approach_front", np.array([clamped_target_position[0], clamped_target_position[1], initial_position[2]])),
                ("final_target", clamped_target_position),
            ],
            sim_app,
            args.ik_steps,
            args.settle_steps,
            args.position_eps,
            args.damping,
            args.max_step,
            args.posture_gain,
            args.stop_tolerance,
            args.hold_steps,
            trace,
        )

        final_joint_positions = _named_positions(selected_dofs, _current_positions(dc, selected_dofs))
        print(f"actual_position={actual_position.tolist()}")
        print(f"position_error={position_error}")
        if position_error > args.error_tolerance:
            raise RuntimeError(
                "End-effector target smoke exceeded tolerance: "
                f"position_error={position_error}, error_tolerance={args.error_tolerance}"
            )

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_path,
            end_effector_name,
            end_effector_path,
            selected_dof_names,
            initial_joint_positions,
            initial_position,
            target_position,
            clamped_target_position,
            actual_position,
            position_error,
            final_joint_positions,
            debug_marker_paths,
            trace,
        )
        print(f"Walker S2 end-effector target OK; wrote {log_file}")

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
        print(f"Walker S2 end-effector target FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
