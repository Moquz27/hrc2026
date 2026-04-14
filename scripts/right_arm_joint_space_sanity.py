#!/usr/bin/env python3
"""Walker S2 right-arm joint-space visual sanity demo.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script intentionally avoids Cartesian IK. It commands a small, explicit
right-arm joint-space sequence for visual sanity checking before task assets,
datasets, or manipulation logic are introduced.
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
    _identify_end_effector_body,
    _select_right_arm_dofs,
)


DEFAULT_CONTROL_STEPS = 90
DEFAULT_SETTLE_STEPS = 24
DEFAULT_GRIPPER_DELTA = 0.03
DEFAULT_DIAGNOSTIC_DELTA = 0.08
DEFAULT_FORWARD_AXIS = "x"
DEFAULT_FORWARD_SIGN = 1.0

# Conservative first-pass pose. These values are intentionally small and should
# be tuned from the per-joint diagnostic if the visual sign is wrong.
FRONT_POSE_BY_NAME = {
    "R_shoulder_pitch_joint": 0.00,
    "R_shoulder_roll_joint": -0.25,
    "R_shoulder_yaw_joint": 0.00,
    "R_elbow_roll_joint": 0.03,
    "R_elbow_yaw_joint": 0.00,
    "R_wrist_pitch_joint": 0.00,
    "R_wrist_roll_joint": 0.00,
}
RAISE_DELTAS_BY_NAME = {
    "R_shoulder_roll_joint": -0.08,
    "R_elbow_roll_joint": 0.02,
}


def _axis_index(axis: str) -> int:
    mapping = {"x": 0, "y": 1, "z": 2}
    if axis not in mapping:
        raise RuntimeError("--forward-axis must be one of x, y, or z")
    return mapping[axis]


def _named_positions(selected_dofs: list[tuple[int, Any, str]], positions: list[float]) -> dict[str, float]:
    return {name: float(position) for (_, _, name), position in zip(selected_dofs, positions)}


def _current_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> list[float]:
    return [float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs]


def _targets_from_map(
    selected_dofs: list[tuple[int, Any, str]],
    base_positions: list[float],
    target_by_name: dict[str, float],
) -> list[float]:
    targets: list[float] = []
    for base_position, (_, _, name) in zip(base_positions, selected_dofs):
        targets.append(float(target_by_name.get(name, base_position)))
    return targets


def _add_targets(
    selected_dofs: list[tuple[int, Any, str]],
    base_targets: list[float],
    deltas_by_name: dict[str, float],
) -> list[float]:
    targets: list[float] = []
    for base_target, (_, _, name) in zip(base_targets, selected_dofs):
        targets.append(float(base_target + deltas_by_name.get(name, 0.0)))
    return targets


def _step_to_targets(
    label: str,
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    targets: list[float],
    sim_app: Any,
    control_steps: int,
    log_rows: list[dict[str, Any]],
) -> list[float]:
    start_positions = _current_positions(dc, selected_dofs)
    for step in range(1, control_steps + 1):
        alpha = step / float(control_steps)
        command = [
            float(start + alpha * (target - start))
            for start, target in zip(start_positions, targets)
        ]
        _send_position_targets(dc, selected_dofs, command)
        sim_app.update()
        if step == 1 or step == control_steps or step % max(1, control_steps // 3) == 0:
            observed = _current_positions(dc, selected_dofs)
            row = {
                "phase": label,
                "step": step,
                "commanded_target_values": _named_positions(selected_dofs, command),
                "observed_joint_values": _named_positions(selected_dofs, observed),
            }
            log_rows.append(row)
            print(row)
    return _current_positions(dc, selected_dofs)


def _command_gripper(
    label: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    base_positions: list[float],
    delta: float,
    sim_app: Any,
    settle_steps: int,
    log_rows: list[dict[str, Any]],
) -> list[float]:
    targets: list[float] = []
    for base_position, (_, _, name) in zip(base_positions, gripper_dofs):
        sign = -1.0 if "finger2" in name.lower() else 1.0
        targets.append(float(base_position + sign * delta))
    _send_position_targets(dc, gripper_dofs, targets)
    for _ in range(settle_steps):
        sim_app.update()
    observed = _read_positions(dc, gripper_dofs)
    row = {
        "phase": label,
        "commanded_target_values": _named_positions(gripper_dofs, targets),
        "observed_joint_values": _named_positions(gripper_dofs, observed),
    }
    log_rows.append(row)
    print(row)
    return observed


def _run_joint_diagnostics(
    dc: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    sim_app: Any,
    base_positions: list[float],
    delta: float,
    settle_steps: int,
    forward_axis: str,
    forward_sign: float,
) -> list[dict[str, Any]]:
    axis = _axis_index(forward_axis)
    diagnostics: list[dict[str, Any]] = []
    _send_position_targets(dc, arm_dofs, base_positions)
    for _ in range(settle_steps):
        sim_app.update()
    base_ee = _body_pose_position(dc, end_effector_body)

    for column, (_, _, name) in enumerate(arm_dofs):
        for sign in (1.0, -1.0):
            command = list(base_positions)
            command[column] += sign * delta
            _send_position_targets(dc, arm_dofs, command)
            for _ in range(settle_steps):
                sim_app.update()
            moved_ee = _body_pose_position(dc, end_effector_body)
            displacement = moved_ee - base_ee
            forward_component = float(displacement[axis] * forward_sign)
            result = {
                "joint_index": arm_dofs[column][0],
                "joint_name": name,
                "command_delta": float(sign * delta),
                "base_end_effector": base_ee.tolist(),
                "moved_end_effector": moved_ee.tolist(),
                "end_effector_displacement": displacement.tolist(),
                "forward_component": forward_component,
                "causes_backward_motion": forward_component < -1.0e-4,
            }
            diagnostics.append(result)
            print(result)
            _send_position_targets(dc, arm_dofs, base_positions)
            for _ in range(settle_steps):
                sim_app.update()
    return diagnostics


def _write_log(
    log_root: Path,
    robot_usd: Path,
    robot_prim_path: str,
    articulation_path: str,
    end_effector_name: str,
    end_effector_path: str,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    initial_arm_positions: list[float],
    final_arm_positions: list[float],
    log_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "walker_s2_right_arm_joint_space_sanity.log"
    log_file.write_text(
        "\n".join(
            (
                "status=walker_s2_right_arm_joint_space_sanity_ok",
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
                f"initial_arm_positions={_named_positions(arm_dofs, initial_arm_positions)}",
                f"final_arm_positions={_named_positions(arm_dofs, final_arm_positions)}",
                f"front_pose_targets={FRONT_POSE_BY_NAME}",
                f"raise_deltas={RAISE_DELTAS_BY_NAME}",
                f"command_log={log_rows}",
                f"joint_diagnostics={diagnostics}",
                "motion_sequence=front_pose -> raise_slightly -> lower_slightly -> gripper_close -> gripper_open",
                "assumption=This is joint-space visual debugging only; no Cartesian IK, task assets, dataset, or ML are used.",
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
    parser.add_argument("--end-effector-body", help="Optional body name/path used only for diagnostics.")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--control-steps", type=int, default=DEFAULT_CONTROL_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA)
    parser.add_argument("--diagnostic", action="store_true", help="Move one right-arm joint at a time before the demo.")
    parser.add_argument("--diagnostic-delta", type=float, default=DEFAULT_DIAGNOSTIC_DELTA)
    parser.add_argument("--forward-axis", default=DEFAULT_FORWARD_AXIS, choices=("x", "y", "z"))
    parser.add_argument("--forward-sign", type=float, default=DEFAULT_FORWARD_SIGN)
    parser.add_argument("--no-demo", action="store_true", help="Run diagnostics only.")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.init_steps < 1 or args.control_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --control-steps, and --settle-steps must be at least 1")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be at least 1")
    if args.gripper_delta <= 0.0:
        raise RuntimeError("--gripper-delta must be positive")
    if args.diagnostic_delta <= 0.0:
        raise RuntimeError("--diagnostic-delta must be positive")
    if args.forward_sign not in (-1.0, 1.0):
        raise RuntimeError("--forward-sign must be 1.0 or -1.0")
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

        initial_arm_positions = _current_positions(dc, arm_dofs)
        print(f"articulation_path={articulation_path}")
        print(f"right_arm_dof_indices={[index for index, _, _ in arm_dofs]}")
        print(f"right_arm_dof_names={[name for _, _, name in arm_dofs]}")
        print(f"right_arm_initial_positions={_named_positions(arm_dofs, initial_arm_positions)}")
        print(f"right_gripper_dof_indices={[index for index, _, _ in gripper_dofs]}")
        print(f"right_gripper_dof_names={[name for _, _, name in gripper_dofs]}")
        print(f"diagnostic_end_effector={end_effector_path}")

        diagnostics: list[dict[str, Any]] = []
        if args.diagnostic:
            diagnostics = _run_joint_diagnostics(
                dc,
                arm_dofs,
                end_effector_body,
                sim_app,
                initial_arm_positions,
                args.diagnostic_delta,
                args.settle_steps,
                args.forward_axis,
                args.forward_sign,
            )

        log_rows: list[dict[str, Any]] = []
        if args.no_demo:
            final_arm_positions = _current_positions(dc, arm_dofs)
        else:
            front_targets = _targets_from_map(arm_dofs, initial_arm_positions, FRONT_POSE_BY_NAME)
            raise_targets = _add_targets(arm_dofs, front_targets, RAISE_DELTAS_BY_NAME)
            _step_to_targets("front_pose", dc, arm_dofs, front_targets, sim_app, args.control_steps, log_rows)
            _step_to_targets("raise_slightly", dc, arm_dofs, raise_targets, sim_app, args.control_steps, log_rows)
            _step_to_targets("lower_slightly", dc, arm_dofs, front_targets, sim_app, args.control_steps, log_rows)
            gripper_initial = _read_positions(dc, gripper_dofs)
            _command_gripper(
                "gripper_close",
                dc,
                gripper_dofs,
                gripper_initial,
                -args.gripper_delta,
                sim_app,
                args.settle_steps,
                log_rows,
            )
            _command_gripper(
                "gripper_open",
                dc,
                gripper_dofs,
                gripper_initial,
                args.gripper_delta,
                sim_app,
                args.settle_steps,
                log_rows,
            )
            final_arm_positions = _current_positions(dc, arm_dofs)

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_path,
            end_effector_name,
            end_effector_path,
            arm_dofs,
            gripper_dofs,
            initial_arm_positions,
            final_arm_positions,
            log_rows,
            diagnostics,
        )
        print(f"Walker S2 right-arm joint-space sanity OK; wrote {log_file}")

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
        print(f"Walker S2 right-arm joint-space sanity FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
