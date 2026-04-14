#!/usr/bin/env python3
"""Minimal Walker S2 arm control smoke test in Isaac Sim.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script verifies the next baseline milestone after robot loading: the articulation
can expose joint state and accept small arm position commands. It intentionally
contains no task logic, object manipulation, perception, dataset use, or learning
code.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


DEFAULT_CONTROL_STEPS = 120
DEFAULT_COMMAND_AMPLITUDE = 0.05
DEFAULT_MIN_MOTION = 1.0e-4
ARM_TOKENS = ("arm", "shoulder", "elbow", "wrist")
NON_ARM_TOKENS = ("leg", "hip", "knee", "ankle", "waist", "head", "neck")


def _start_timeline() -> Any:
    import omni.timeline  # type: ignore

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    return timeline


def _acquire_articulation(articulation_path: str) -> tuple[Any, Any]:
    from omni.isaac.dynamic_control import _dynamic_control  # type: ignore

    dc = _dynamic_control.acquire_dynamic_control_interface()
    articulation = dc.get_articulation(articulation_path)
    if not articulation:
        raise RuntimeError(
            "dynamic_control could not acquire the articulation at "
            f"{articulation_path}. Use the detected articulation root, not just "
            "the parent robot prim."
        )
    dc.wake_up_articulation(articulation)
    return dc, articulation


def _read_dof_observation(dc: Any, articulation: Any) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    dof_count = dc.get_articulation_dof_count(articulation)
    for index in range(dof_count):
        dof = dc.get_articulation_dof(articulation, index)
        item: dict[str, Any] = {
            "index": index,
            "name": str(dc.get_dof_name(dof)),
            "position": float(dc.get_dof_position(dof)),
        }
        try:
            item["velocity"] = float(dc.get_dof_velocity(dof))
        except Exception:
            item["velocity"] = math.nan
        observations.append(item)
    return observations


def _is_arm_dof(name: str) -> bool:
    lower_name = name.lower()
    if any(token in lower_name for token in NON_ARM_TOKENS):
        return False
    return any(token in lower_name for token in ARM_TOKENS)


def _select_arm_dofs(
    dc: Any,
    articulation: Any,
    max_dofs: int,
) -> list[tuple[int, Any, str]]:
    selected: list[tuple[int, Any, str]] = []
    dof_count = dc.get_articulation_dof_count(articulation)
    for index in range(dof_count):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        if _is_arm_dof(name):
            selected.append((index, dof, name))
        if len(selected) >= max_dofs:
            break

    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dof_count)
        ]
        raise RuntimeError(
            "No arm DOFs matched the conservative arm-name filter. "
            f"Arm tokens={ARM_TOKENS}; non-arm tokens={NON_ARM_TOKENS}; "
            f"available_dof_names={all_names}"
        )
    return selected


def _send_position_targets(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    targets: list[float],
) -> None:
    if not hasattr(dc, "set_dof_position_target"):
        raise RuntimeError("dynamic_control does not expose set_dof_position_target")
    for (_, dof, _), target in zip(selected_dofs, targets):
        dc.set_dof_position_target(dof, float(target))


def _write_log(
    log_root: Path,
    robot_usd: Path,
    robot_prim_path: str,
    articulation_path: str,
    selected_names: list[str],
    control_steps: int,
    command_amplitude: float,
    initial_positions: list[float],
    final_positions: list[float],
    max_abs_delta: float,
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "walker_s2_arm_control_smoke.log"
    log_file.write_text(
        "\n".join(
            (
                "status=walker_s2_arm_control_smoke_ok",
                f"timestamp_utc={timestamp}",
                f"robot_usd_path={robot_usd}",
                f"robot_prim_path={robot_prim_path}",
                f"articulation_path={articulation_path}",
                f"selected_arm_dof_count={len(selected_names)}",
                f"selected_arm_dof_names={selected_names}",
                f"control_steps={control_steps}",
                f"command_amplitude={command_amplitude}",
                f"initial_positions={initial_positions}",
                f"final_positions={final_positions}",
                f"max_abs_delta={max_abs_delta}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def _hold_gui_open(sim_app: Any) -> None:
    print("Holding Isaac Sim GUI open. Close the window or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        print("Ctrl+C received; closing Isaac Sim.")


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
    parser.add_argument(
        "--init-steps",
        type=int,
        default=DEFAULT_INIT_STEPS,
        help="Simulation updates to allow robot initialization.",
    )
    parser.add_argument(
        "--control-steps",
        type=int,
        default=DEFAULT_CONTROL_STEPS,
        help="Control-loop updates to run after initialization.",
    )
    parser.add_argument(
        "--command-amplitude",
        type=float,
        default=DEFAULT_COMMAND_AMPLITUDE,
        help="Small position-target offset for selected arm joints.",
    )
    parser.add_argument(
        "--max-arm-dofs",
        type=int,
        default=8,
        help="Maximum matching arm DOFs to command.",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=DEFAULT_MIN_MOTION,
        help="Minimum observed arm joint motion required for the smoke test to pass.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Launch Isaac Sim with a visible window instead of headless mode.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Alias for --no-headless; launch Isaac Sim with a visible window.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="After the smoke test, keep the GUI open until the window closes or Ctrl+C is pressed.",
    )
    args = parser.parse_args()

    if args.init_steps < 1:
        raise RuntimeError("--init-steps must be at least 1")
    if args.control_steps < 1:
        raise RuntimeError("--control-steps must be at least 1")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be at least 1")
    if args.command_amplitude <= 0.0:
        raise RuntimeError("--command-amplitude must be positive")
    if args.min_motion < 0.0:
        raise RuntimeError("--min-motion must be non-negative")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path, for example /World/WalkerS2")

    # Prevent Isaac Kit from consuming this script's CLI arguments.
    sys.argv = [sys.argv[0]]

    paths = _validate_environment()
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])

    print(f"HRC_REPO={paths['HRC_REPO']}")
    print(f"LOG_ROOT={paths['LOG_ROOT']}")
    print(f"robot_usd_path={robot_usd}")
    print(f"robot_prim_path={args.prim_path}")

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
        print(f"articulation_path={articulation_path}")
        print(f"joint_count={len(joint_names)}")

        timeline = _start_timeline()
        for _ in range(5):
            sim_app.update()

        dc, articulation = _acquire_articulation(articulation_path)
        initial_observation = _read_dof_observation(dc, articulation)
        selected_dofs = _select_arm_dofs(dc, articulation, args.max_arm_dofs)
        selected_indices = [index for index, _, _ in selected_dofs]
        selected_names = [name for _, _, name in selected_dofs]
        initial_positions = [
            float(initial_observation[index]["position"]) for index in selected_indices
        ]

        print(f"dof_count={len(initial_observation)}")
        print(f"selected_arm_dof_count={len(selected_names)}")
        print(f"selected_arm_dof_names={selected_names}")
        print(f"initial_arm_positions={initial_positions}")

        final_positions = initial_positions[:]
        max_abs_delta = 0.0
        for step in range(args.control_steps):
            phase = math.sin((step + 1) / args.control_steps * math.pi)
            targets = [
                initial + args.command_amplitude * phase
                for initial in initial_positions
            ]
            _send_position_targets(dc, selected_dofs, targets)
            sim_app.update()

            observation = _read_dof_observation(dc, articulation)
            final_positions = [float(observation[index]["position"]) for index in selected_indices]
            step_max_delta = max(
                abs(current - initial)
                for initial, current in zip(initial_positions, final_positions)
            )
            max_abs_delta = max(max_abs_delta, step_max_delta)
            if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.control_steps:
                print(
                    f"control_step={step + 1}/{args.control_steps} "
                    f"targets={targets} positions={final_positions}"
                )

        print(f"final_arm_positions={final_positions}")
        print(f"max_abs_delta={max_abs_delta}")

        if max_abs_delta < args.min_motion:
            raise RuntimeError(
                "Arm command loop ran, but observed motion was below threshold: "
                f"max_abs_delta={max_abs_delta}, min_motion={args.min_motion}"
            )

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_path,
            selected_names,
            args.control_steps,
            args.command_amplitude,
            initial_positions,
            final_positions,
            max_abs_delta,
        )
        print(f"Walker S2 arm control smoke OK; wrote {log_file}")

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
        print(f"Walker S2 arm control smoke FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
