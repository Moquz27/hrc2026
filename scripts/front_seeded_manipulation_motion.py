#!/usr/bin/env python3
"""Walker S2 front-seeded right-arm phased motion baseline.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script intentionally uses explicit joint-space phases instead of general
Cartesian IK. It is the current strongest manipulation-related motion baseline,
but it is still only motion sanity: no dynamic object transport, task assets,
dataset, competition logic, or ML.
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
    _read_dof_observation,
    _send_position_targets,
    _start_timeline,
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


DEFAULT_CYCLES = 5
DEFAULT_CONTROL_STEPS = 72
DEFAULT_SETTLE_STEPS = 12
DEFAULT_GRIPPER_DELTA = 0.03
DEFAULT_FRONT_MARGIN_X = 0.045
DEFAULT_MAX_Z = 1.05
DEFAULT_MAX_DRIFT = 0.04

ABOVE_FRONT_TARGET_BY_NAME = {
    **FRONT_POSE_BY_NAME,
    "R_shoulder_roll_joint": -0.34,
    "R_elbow_roll_joint": 0.035,
}
DOWN_FRONT_TARGET_BY_NAME = {
    **FRONT_POSE_BY_NAME,
    "R_shoulder_roll_joint": -0.27,
    "R_elbow_roll_joint": 0.03,
}


def _named_positions(selected_dofs: list[tuple[int, Any, str]], positions: list[float]) -> dict[str, float]:
    return {name: float(position) for (_, _, name), position in zip(selected_dofs, positions)}


def _current_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> list[float]:
    return [float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs]


def _targets_from_map(selected_dofs: list[tuple[int, Any, str]], target_by_name: dict[str, float]) -> list[float]:
    missing_names = [name for _, _, name in selected_dofs if name not in target_by_name]
    if missing_names:
        raise RuntimeError(f"Missing explicit right-arm targets for DOFs: {missing_names}")
    return [float(target_by_name[name]) for _, _, name in selected_dofs]


def _set_debug_marker(stage: Any, path: str, position: np.ndarray, color: tuple[float, float, float]) -> str:
    return _create_debug_marker(stage, path, position, 0.025, color)


def _phase_sanity(
    end_effector_position: np.ndarray,
    front_reference_x: float,
    reference_position: np.ndarray,
    max_z: float,
) -> tuple[bool, dict[str, float | bool]]:
    backward_deviation = max(0.0, float(front_reference_x - end_effector_position[0]))
    overhead_deviation = max(0.0, float(end_effector_position[2] - max_z))
    offset = float(np.linalg.norm(end_effector_position - reference_position))
    checks = {
        "backward_deviation": backward_deviation,
        "overhead_deviation": overhead_deviation,
        "offset_from_front_reference": offset,
        "front_facing_ok": backward_deviation <= 0.0,
        "overhead_ok": overhead_deviation <= 0.0,
    }
    return bool(checks["front_facing_ok"] and checks["overhead_ok"]), checks


def _step_arm_phase(
    iteration: int,
    phase_name: str,
    dc: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    target_positions: list[float],
    sim_app: Any,
    control_steps: int,
    settle_steps: int,
    front_reference_x: float,
    front_reference_position: np.ndarray,
    max_z: float,
    log_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, bool]:
    start_positions = _current_positions(dc, arm_dofs)
    commanded = list(target_positions)
    for step in range(1, control_steps + 1):
        alpha = step / float(control_steps)
        command = [
            float(start + alpha * (target - start))
            for start, target in zip(start_positions, target_positions)
        ]
        _send_position_targets(dc, arm_dofs, command)
        sim_app.update()

    for _ in range(settle_steps):
        sim_app.update()

    observed = _current_positions(dc, arm_dofs)
    ee_position = _body_pose_position(dc, end_effector_body)
    phase_ok, checks = _phase_sanity(
        ee_position,
        front_reference_x,
        front_reference_position,
        max_z,
    )
    row = {
        "iteration": iteration,
        "phase": phase_name,
        "right_arm_dof_indices": [index for index, _, _ in arm_dofs],
        "right_arm_dof_names": [name for _, _, name in arm_dofs],
        "commanded_joint_target_values": _named_positions(arm_dofs, commanded),
        "observed_joint_values": _named_positions(arm_dofs, observed),
        "end_effector_position": ee_position.tolist(),
        "front_posture_checks": checks,
        "phase_passed_motion_sanity": phase_ok,
    }
    log_rows.append(row)
    print(row)
    return ee_position, phase_ok


def _command_gripper_phase(
    iteration: int,
    phase_name: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    base_positions: list[float],
    delta: float,
    sim_app: Any,
    settle_steps: int,
    log_rows: list[dict[str, Any]],
) -> bool:
    targets: list[float] = []
    for base_position, (_, _, name) in zip(base_positions, gripper_dofs):
        sign = -1.0 if "finger2" in name.lower() else 1.0
        targets.append(float(base_position + sign * delta))
    _send_position_targets(dc, gripper_dofs, targets)
    for _ in range(settle_steps):
        sim_app.update()
    observed = _read_positions(dc, gripper_dofs)
    row = {
        "iteration": iteration,
        "phase": phase_name,
        "right_gripper_dof_indices": [index for index, _, _ in gripper_dofs],
        "right_gripper_dof_names": [name for _, _, name in gripper_dofs],
        "commanded_joint_target_values": _named_positions(gripper_dofs, targets),
        "observed_joint_values": _named_positions(gripper_dofs, observed),
        "phase_passed_motion_sanity": True,
    }
    log_rows.append(row)
    print(row)
    return True


def _update_repeatability(
    phase_references: dict[str, np.ndarray],
    phase_name: str,
    position: np.ndarray,
    max_drift: float,
) -> tuple[bool, float]:
    if phase_name not in phase_references:
        phase_references[phase_name] = position.copy()
        return True, 0.0
    drift = float(np.linalg.norm(position - phase_references[phase_name]))
    return drift <= max_drift, drift


def _write_log(
    log_root: Path,
    robot_usd: Path,
    robot_prim_path: str,
    articulation_path: str,
    end_effector_name: str,
    end_effector_path: str,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    cycles: int,
    cycle_results: list[dict[str, Any]],
    log_rows: list[dict[str, Any]],
    marker_paths: list[str],
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "walker_s2_front_seeded_manipulation_motion.log"
    log_file.write_text(
        "\n".join(
            (
                "status=walker_s2_front_seeded_manipulation_motion_ok",
                f"timestamp_utc={timestamp}",
                f"robot_usd_path={robot_usd}",
                f"robot_prim_path={robot_prim_path}",
                f"articulation_path={articulation_path}",
                f"end_effector_name={end_effector_name}",
                f"end_effector_path={end_effector_path}",
                f"right_arm_dof_indices={[index for index, _, _ in arm_dofs]}",
                f"right_arm_dof_names={[name for _, _, name in arm_dofs]}",
                f"right_gripper_dof_indices={[index for index, _, _ in gripper_dofs]}",
                f"right_gripper_dof_names={[name for _, _, name in gripper_dofs]}",
                f"cycles={cycles}",
                f"front_pose_targets={FRONT_POSE_BY_NAME}",
                f"above_front_target={ABOVE_FRONT_TARGET_BY_NAME}",
                f"down_front_target={DOWN_FRONT_TARGET_BY_NAME}",
                f"cycle_results={cycle_results}",
                f"phase_log={log_rows}",
                f"debug_marker_paths={marker_paths}",
                "motion_sequence=move_to_front_pose -> move_slightly_above_front_target -> move_slightly_downward -> gripper_close -> lift_slightly_upward -> gripper_open",
                "assumption=This remains explicit joint-space visual debugging; no Cartesian IK, task assets, dataset, or ML are used.",
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
    parser.add_argument("--end-effector-body", help="Optional body name/path used for logging.")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--control-steps", type=int, default=DEFAULT_CONTROL_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA)
    parser.add_argument("--front-margin-x", type=float, default=DEFAULT_FRONT_MARGIN_X)
    parser.add_argument("--max-z", type=float, default=DEFAULT_MAX_Z)
    parser.add_argument("--max-drift", type=float, default=DEFAULT_MAX_DRIFT)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.cycles < 1:
        raise RuntimeError("--cycles must be at least 1")
    if args.init_steps < 1 or args.control_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --control-steps, and --settle-steps must be at least 1")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be at least 1")
    if args.gripper_delta <= 0.0:
        raise RuntimeError("--gripper-delta must be positive")
    if args.front_margin_x < 0.0:
        raise RuntimeError("--front-margin-x must be non-negative")
    if args.max_drift <= 0.0:
        raise RuntimeError("--max-drift must be positive")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path, for example /World/WalkerS2")

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
        arm_dofs = _select_right_arm_dofs(dc, articulation, args.max_arm_dofs)
        gripper_dofs = _select_right_gripper_dofs(dc, articulation)
        end_effector_body, end_effector_name, end_effector_path = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
        )

        front_targets = _targets_from_map(arm_dofs, FRONT_POSE_BY_NAME)
        above_targets = _targets_from_map(arm_dofs, ABOVE_FRONT_TARGET_BY_NAME)
        down_targets = _targets_from_map(arm_dofs, DOWN_FRONT_TARGET_BY_NAME)
        gripper_initial = _read_positions(dc, gripper_dofs)

        log_rows: list[dict[str, Any]] = []
        cycle_results: list[dict[str, Any]] = []
        marker_paths: list[str] = []
        phase_references: dict[str, np.ndarray] = {}
        front_reference_position: np.ndarray | None = None
        front_reference_x = 0.0

        print(f"articulation_path={articulation_path}")
        print(f"right_arm_dof_indices={[index for index, _, _ in arm_dofs]}")
        print(f"right_arm_dof_names={[name for _, _, name in arm_dofs]}")
        print(f"right_gripper_dof_indices={[index for index, _, _ in gripper_dofs]}")
        print(f"right_gripper_dof_names={[name for _, _, name in gripper_dofs]}")

        for iteration in range(1, args.cycles + 1):
            cycle_ok = True
            front_position, phase_ok = _step_arm_phase(
                iteration,
                "move_to_front_pose",
                dc,
                arm_dofs,
                end_effector_body,
                front_targets,
                sim_app,
                args.control_steps,
                args.settle_steps,
                -float("inf"),
                _body_pose_position(dc, end_effector_body),
                args.max_z,
                log_rows,
            )
            cycle_ok = cycle_ok and phase_ok
            if iteration == 1:
                front_reference_position = front_position.copy()
                front_reference_x = float(front_position[0] - args.front_margin_x)
                marker_paths.append(_set_debug_marker(stage, "/World/DebugFrontPoseReference", front_position, (0.1, 0.8, 0.2)))

            assert front_reference_position is not None
            phase_repeatability: dict[str, float] = {}
            repeatability_ok, drift = _update_repeatability(
                phase_references,
                "move_to_front_pose",
                front_position,
                args.max_drift,
            )
            phase_repeatability["move_to_front_pose"] = drift
            cycle_ok = cycle_ok and repeatability_ok

            above_position, phase_ok = _step_arm_phase(
                iteration,
                "move_slightly_above_front_target",
                dc,
                arm_dofs,
                end_effector_body,
                above_targets,
                sim_app,
                args.control_steps,
                args.settle_steps,
                front_reference_x,
                front_reference_position,
                args.max_z,
                log_rows,
            )
            cycle_ok = cycle_ok and phase_ok
            repeatability_ok, drift = _update_repeatability(
                phase_references,
                "move_slightly_above_front_target",
                above_position,
                args.max_drift,
            )
            phase_repeatability["move_slightly_above_front_target"] = drift
            cycle_ok = cycle_ok and repeatability_ok
            if iteration == 1:
                marker_paths.append(_set_debug_marker(stage, "/World/DebugAboveFrontTarget", above_position, (0.2, 0.5, 1.0)))

            down_position, phase_ok = _step_arm_phase(
                iteration,
                "move_slightly_downward",
                dc,
                arm_dofs,
                end_effector_body,
                down_targets,
                sim_app,
                args.control_steps,
                args.settle_steps,
                front_reference_x,
                front_reference_position,
                args.max_z,
                log_rows,
            )
            cycle_ok = cycle_ok and phase_ok
            repeatability_ok, drift = _update_repeatability(
                phase_references,
                "move_slightly_downward",
                down_position,
                args.max_drift,
            )
            phase_repeatability["move_slightly_downward"] = drift
            cycle_ok = cycle_ok and repeatability_ok
            if iteration == 1:
                marker_paths.append(_set_debug_marker(stage, "/World/DebugDownFrontTarget", down_position, (1.0, 0.4, 0.1)))

            cycle_ok = cycle_ok and _command_gripper_phase(
                iteration,
                "gripper_close",
                dc,
                gripper_dofs,
                gripper_initial,
                -args.gripper_delta,
                sim_app,
                args.settle_steps,
                log_rows,
            )
            _, phase_ok = _step_arm_phase(
                iteration,
                "lift_slightly_upward",
                dc,
                arm_dofs,
                end_effector_body,
                above_targets,
                sim_app,
                args.control_steps,
                args.settle_steps,
                front_reference_x,
                front_reference_position,
                args.max_z,
                log_rows,
            )
            cycle_ok = cycle_ok and phase_ok
            lift_position = _body_pose_position(dc, end_effector_body)
            repeatability_ok, drift = _update_repeatability(
                phase_references,
                "lift_slightly_upward",
                lift_position,
                args.max_drift,
            )
            phase_repeatability["lift_slightly_upward"] = drift
            cycle_ok = cycle_ok and repeatability_ok
            cycle_ok = cycle_ok and _command_gripper_phase(
                iteration,
                "gripper_open",
                dc,
                gripper_dofs,
                gripper_initial,
                args.gripper_delta,
                sim_app,
                args.settle_steps,
                log_rows,
            )
            final_position = _body_pose_position(dc, end_effector_body)
            cycle_results.append(
                {
                    "iteration": iteration,
                    "cycle_passed_motion_sanity": cycle_ok,
                    "final_end_effector_position": final_position.tolist(),
                    "phase_repeatability_drifts": phase_repeatability,
                }
            )

        failed_cycles = [item for item in cycle_results if not item["cycle_passed_motion_sanity"]]
        if failed_cycles:
            raise RuntimeError(f"Front-seeded motion sanity failed: {failed_cycles}")

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_path,
            end_effector_name,
            end_effector_path,
            arm_dofs,
            gripper_dofs,
            args.cycles,
            cycle_results,
            log_rows,
            marker_paths,
        )
        print(f"Walker S2 front-seeded manipulation motion OK; wrote {log_file}")

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
        print(f"Walker S2 front-seeded manipulation motion FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
