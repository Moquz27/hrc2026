#!/usr/bin/env python3
"""Smooth auto-seed multi-object Task 1 variant.

This is a copied variant of the validated one-object baseline. It preserves the
official scene path, validated grasp/contact geometry, and gripper effort hold,
then adds runtime auto-seeding, an in-scene one-arm loop over spawned Task 1
objects, and a queued continuous waypoint cycle with short contact-only dwell.

It does not run perception, A/B classification, or Thinker/model integration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass, field
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
from front_seeded_manipulation_motion import _current_positions, _named_positions, _targets_from_map  # type: ignore
from grasp_static_object_smoke import _read_positions, _select_right_gripper_dofs  # type: ignore
from inspect_task1_pick_place_gui import _add_reference, _set_xform  # type: ignore
from load_walker_s2 import (  # type: ignore
    DEFAULT_INIT_STEPS,
    _create_minimal_scene,
    _find_articulation_roots,
    _find_joint_names,
    _load_simulation_app,
    _validate_environment,
)
from move_walker_s2_end_effector import (  # type: ignore
    DEFAULT_DAMPING,
    DEFAULT_HOLD_STEPS,
    DEFAULT_IK_STEPS,
    DEFAULT_POSITION_EPS,
    DEFAULT_POSTURE_GAIN,
    DEFAULT_SETTLE_STEPS as DEFAULT_IK_SETTLE_STEPS,
    DEFAULT_STOP_TOLERANCE,
    _body_pose_position,
    _create_debug_marker,
    _identify_end_effector_body,
    _select_right_arm_dofs,
    move_end_effector_to_target,
)
from validate_task1_object_assets import _bbox, _physics_summary  # type: ignore
from validate_task1_scene_builder_scene import (  # type: ignore
    DEFAULT_ASSET_ROOT_RELATIVE,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_CONFIG_RELATIVE,
    _NullDataLogger,
    _category_from_reference,
    _load_official_scene_builder,
    _reference_paths,
)


SCRIPT_NAME = "task1_smooth_autoseed_multi_object_baseline.py"
LOG_STEM = "task1_smooth_autoseed_multi_object_baseline"
DEFAULT_PHASE_STEPS = 120
DEFAULT_PAUSE_STEPS = 30
DEFAULT_SETTLE_STEPS = 240
DEFAULT_GRIPPER_DELTA = 0.03
OFFICIAL_GRIPPER_OPEN_WIDTH = -0.0215
OFFICIAL_GRIPPER_CLOSE_WIDTH = 0.01
DEFAULT_GRIPPER_HOLD_EFFORT = 100.0
DEFAULT_DESCEND_CLEARANCE = 0.015
DEFAULT_GRASP_DEPTH_TUNING_OFFSET = -0.005
DEFAULT_PRE_GRASP_CLEARANCE = 0.10
DEFAULT_SAFE_DROP_HEIGHT = 0.10
DEFAULT_STABLE_JITTER = 0.01
DEFAULT_MIN_LIFT_DELTA = 0.015
DEFAULT_MIN_TRANSPORT_DISTANCE = 0.08
DEFAULT_PRE_GRASP_EE_TOLERANCE = 0.25
DEFAULT_DESCEND_OBJECT_TOLERANCE = 0.16
DEFAULT_JOINT_TOLERANCE = 0.06
DEFAULT_MAX_LOCAL_JOINT_ADJUSTMENT = 0.04
DEFAULT_FRONT_WORKSPACE_X = (0.55, 1.45)
DEFAULT_FRONT_WORKSPACE_Y = (-0.75, 0.45)
DEFAULT_FRONT_WORKSPACE_Z = (0.55, 1.20)
DEFAULT_MIN_EE_TABLE_CLEARANCE = 0.025
DEFAULT_IK_MAX_STEP = 0.03
DEFAULT_CONTINUOUS_SOFT_TOLERANCE = 0.18
DEFAULT_CONTINUOUS_BLEND_RADIUS = 0.05
DEFAULT_CONTINUOUS_CONTACT_DWELL_STEPS = 8
DEFAULT_CARRY_STABILIZATION_STEPS = 0
DEFAULT_PLACE_DEPTH_OFFSET = 0.0
DEFAULT_RELEASE_TIMING_DWELL_STEPS = 0
DEFAULT_MICRO_STOP_SPEED_THRESHOLD = 0.0005
OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"

# The organizer baseline does not define arm startup pose in Part_Sorting.yaml.
# It applies this posture in IsaacSimRobotInterface.initialize(); load it from
# that official class at runtime instead of maintaining a local front-ready seed.
OFFICIAL_STARTUP_ARM_JOINT_NAMES = {
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_roll_joint",
    "L_elbow_yaw_joint",
    "L_wrist_pitch_joint",
    "L_wrist_roll_joint",
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_roll_joint",
    "R_elbow_yaw_joint",
    "R_wrist_pitch_joint",
    "R_wrist_roll_joint",
}

class RunFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class MotionSegment:
    name: str
    phase_tag: str
    target_position: np.ndarray
    waypoint_type: str
    speed_profile: str
    max_ik_steps: int
    stop_tolerance: float
    hold_steps: int
    blend_radius: float
    event_marker: str | None = None
    contact_window: bool = False
    gripper_effort: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _center_from_bbox(box: dict[str, list[float]]) -> np.ndarray:
    return np.array(box["center"], dtype=float)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pregrasp_geometry_summary(
    *,
    target_index: int,
    target_path: str,
    object_center: np.ndarray,
    bbox_top_z: float,
    pre_grasp_position: np.ndarray,
    robot_base_position: list[float],
    table_top_z: float,
) -> dict[str, Any]:
    robot_base = np.array(robot_base_position, dtype=float)
    delta = pre_grasp_position - robot_base
    return {
        "target_index": target_index,
        "target_path": target_path,
        "reference_frame": "world",
        "object_center": object_center.tolist(),
        "bbox_top_z": float(bbox_top_z),
        "pre_grasp_target": pre_grasp_position.tolist(),
        "robot_base_position": robot_base.tolist(),
        "forward_offset_x": float(delta[0]),
        "lateral_offset_y": float(delta[1]),
        "horizontal_reach_distance": float(np.linalg.norm(delta[:2])),
        "vertical_reach_requirement": float(delta[2]),
        "table_clearance": float(pre_grasp_position[2] - table_top_z),
        "full_base_to_pregrasp_distance": float(np.linalg.norm(delta)),
    }


def _print_pregrasp_geometry(summary: dict[str, Any]) -> None:
    print(f"pregrasp_geometry={json.dumps(summary, sort_keys=True)}")


def _parse_bounds(raw_values: list[float], name: str) -> tuple[float, float]:
    if len(raw_values) != 2:
        raise RuntimeError(f"{name} requires exactly two values")
    low = float(raw_values[0])
    high = float(raw_values[1])
    if low >= high:
        raise RuntimeError(f"{name} lower bound must be smaller than upper bound")
    return low, high


def _front_workspace_checks(
    position: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
    table_top_z: float,
) -> dict[str, Any]:
    checks = {
        "position": position.tolist(),
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "z_limits": list(z_limits),
        "table_top_z": table_top_z,
        "x_ok": x_limits[0] <= float(position[0]) <= x_limits[1],
        "y_ok": y_limits[0] <= float(position[1]) <= y_limits[1],
        "z_ok": z_limits[0] <= float(position[2]) <= z_limits[1],
        "above_table_ok": float(position[2]) >= table_top_z,
    }
    checks["front_workspace_ok"] = bool(checks["x_ok"] and checks["y_ok"] and checks["z_ok"] and checks["above_table_ok"])
    return checks


def _ee_front_safety_checks(
    position: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    table_top_z: float,
    min_table_clearance: float,
) -> dict[str, Any]:
    checks = {
        "position": position.tolist(),
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "table_top_z": table_top_z,
        "min_table_clearance": min_table_clearance,
        "x_ok": x_limits[0] <= float(position[0]) <= x_limits[1],
        "y_ok": y_limits[0] <= float(position[1]) <= y_limits[1],
        "above_table_clearance_ok": float(position[2]) >= table_top_z + min_table_clearance,
    }
    checks["front_safety_ok"] = bool(checks["x_ok"] and checks["y_ok"] and checks["above_table_clearance_ok"])
    return checks


def _bbox_state(stage: Any, prim_path: str) -> dict[str, Any]:
    box = _bbox(stage, prim_path)
    return {
        "bbox": box,
        "center": box["center"],
        "finite": _finite(box["min"] + box["max"] + box["center"]),
    }


def _gripper_values(dc: Any, gripper_dofs: list[tuple[int, Any, str]]) -> dict[str, float]:
    return _named_positions(gripper_dofs, _read_positions(dc, gripper_dofs))


def _apply_gripper_effort(
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    effort_value: float,
) -> dict[str, Any]:
    if not hasattr(dc, "set_dof_effort"):
        return {
            "supported": False,
            "method": None,
            "effort_value": float(effort_value),
            "dof_indices": [index for index, _, _ in gripper_dofs],
            "dof_names": [name for _, _, name in gripper_dofs],
            "errors": ["dynamic_control does not expose set_dof_effort"],
        }

    applied: dict[str, float] = {}
    errors: list[str] = []
    for index, dof, name in gripper_dofs:
        try:
            dc.set_dof_effort(dof, float(effort_value))
            applied[name] = float(effort_value)
        except Exception as exc:  # pragma: no cover - Isaac runtime API detail.
            errors.append(f"{index}:{name}: {exc}")

    return {
        "supported": not errors and len(applied) == len(gripper_dofs),
        "method": "dynamic_control.set_dof_effort",
        "effort_value": float(effort_value),
        "dof_indices": [index for index, _, _ in gripper_dofs],
        "dof_names": [name for _, _, name in gripper_dofs],
        "applied_efforts": applied,
        "errors": errors,
    }


def _run_updates_with_optional_gripper_effort(
    sim_app: Any,
    steps: int,
    counter: dict[str, int],
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    effort_value: float | None,
) -> dict[str, Any] | None:
    last_effort_result: dict[str, Any] | None = None
    for _ in range(steps):
        if effort_value is not None:
            last_effort_result = _apply_gripper_effort(dc, gripper_dofs, effort_value)
            if not last_effort_result["supported"]:
                raise RuntimeError(f"Gripper effort command failed: {last_effort_result}")
        sim_app.update()
        counter["step"] += 1
    return last_effort_result


def _select_dofs_by_target_names(
    dc: Any,
    articulation: Any,
    target_by_name: dict[str, float],
    required_names: set[str],
) -> tuple[list[tuple[int, Any, str]], list[str]]:
    selected: list[tuple[int, Any, str]] = []
    found: set[str] = set()
    missing_optional: list[str] = []
    dof_count = dc.get_articulation_dof_count(articulation)
    for index in range(dof_count):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        if name in target_by_name:
            selected.append((index, dof, name))
            found.add(name)

    missing_required = sorted(required_names - found)
    if missing_required:
        raise RuntimeError(f"Missing required front-ready DOFs: {missing_required}")
    for name in target_by_name:
        if name not in found:
            missing_optional.append(name)
    return selected, missing_optional


def _load_official_startup_joint_map(baseline_root: Path, prim_path: str, urdf_path: Path) -> dict[str, float]:
    module_path = baseline_root / "lerobot/common/robot_devices/robots/isaac_sim_robot_interface.py"
    if not module_path.exists():
        raise RuntimeError(f"Official IsaacSimRobotInterface file missing: {module_path}")
    spec = importlib.util.spec_from_file_location("official_isaac_sim_robot_interface", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official IsaacSimRobotInterface module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    IsaacSimRobotInterface = module.IsaacSimRobotInterface

    interface = IsaacSimRobotInterface(
        prim_path=prim_path,
        name=OFFICIAL_ROBOT_NAME,
        world=None,
        urdf_path=str(urdf_path),
    )
    joint_map = dict(getattr(interface, "_joint_value_map", {}))
    missing = sorted(OFFICIAL_STARTUP_ARM_JOINT_NAMES - set(joint_map))
    if missing:
        raise RuntimeError(f"Official startup joint map missing expected arm joints: {missing}")
    return {name: float(value) for name, value in joint_map.items()}


def _set_joint_positions_and_targets(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    target_positions: list[float],
) -> None:
    for (_, dof, _), target in zip(selected_dofs, target_positions):
        if hasattr(dc, "set_dof_position"):
            dc.set_dof_position(dof, float(target))
        if hasattr(dc, "set_dof_position_target"):
            dc.set_dof_position_target(dof, float(target))


def _debug_marker(stage: Any, path: str, position: list[float], radius: float, color: tuple[float, float, float]) -> str:
    return _create_debug_marker(stage, path, np.array(position, dtype=float), radius, color)


def _write_logs(log_root: Path, payload: dict[str, Any], log_suffix: str | None) -> list[str]:
    timestamp = payload["run_metadata"].get("timestamp_compact") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{log_suffix}" if log_suffix else ""
    rolling = log_root / f"{LOG_STEM}.log"
    per_run = log_root / f"{LOG_STEM}_{timestamp}{suffix}.log"
    payload["log_paths"] = [str(rolling), str(per_run)]
    text = "\n".join(
        (
            f"status={payload.get('final_status', 'fail')}",
            f"failure_reason={payload.get('failure_reason')}",
            f"timestamp_utc={payload['run_metadata'].get('timestamp_utc')}",
            f"script_name={SCRIPT_NAME}",
            f"yaml_path={payload['run_metadata'].get('yaml_path')}",
            f"root_path_override={payload['run_metadata'].get('root_path_override')}",
            f"selected_target_prim={payload.get('target', {}).get('prim_path')}",
            f"object_lifted={payload['result_flags'].get('object_lifted')}",
            f"object_retained_after_lift={payload['result_flags'].get('object_retained_after_lift')}",
            f"object_transported={payload['result_flags'].get('object_transported')}",
            f"final_inside_bin={payload['result_flags'].get('final_inside_bin')}",
            f"object_stable={payload['result_flags'].get('object_stable')}",
            f"payload={json.dumps(payload, indent=2, sort_keys=True)}",
        )
    ) + "\n"
    rolling.write_text(text, encoding="utf-8")
    per_run.write_text(text, encoding="utf-8")
    return payload["log_paths"]


def _append_phase(
    phase_log: list[dict[str, Any]],
    *,
    phase: str,
    start_step: int,
    end_step: int,
    commanded_targets: dict[str, float] | None,
    ee_position: np.ndarray | None,
    gripper_values: dict[str, float] | None,
    condition_met: bool,
    details: dict[str, Any],
) -> None:
    phase_log.append(
        {
            "phase": phase,
            "start_step": start_step,
            "end_step": end_step,
            "commanded_joint_targets": commanded_targets or {},
            "observed_end_effector_position": None if ee_position is None else ee_position.tolist(),
            "observed_gripper_joint_values": gripper_values or {},
            "condition_met": bool(condition_met),
            "details": details,
        }
    )


def _run_updates(sim_app: Any, steps: int, counter: dict[str, int]) -> None:
    for _ in range(steps):
        sim_app.update()
        counter["step"] += 1


def _active_tuning_knob(args: argparse.Namespace) -> str:
    if abs(float(args.grasp_depth_offset)) > 1e-9:
        return "grasp_depth"
    if args.carry_stabilization_steps > 0:
        return "carry_stabilization"
    if abs(float(args.place_depth_offset)) > 1e-9:
        return "place_depth"
    if args.release_timing_dwell_steps > 0:
        return "release_timing"
    if args.continuous_contact_dwell_steps != DEFAULT_CONTINUOUS_CONTACT_DWELL_STEPS:
        return "contact_dwell"
    if args.continuous_soft_tolerance != DEFAULT_CONTINUOUS_SOFT_TOLERANCE:
        return "soft_tolerance"
    return "baseline_defaults"


def _tuning_knob_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "active_knob_family": _active_tuning_knob(args),
        "policy": "one_knob_family_per_controlled_sweep",
        "targeted_failure_pattern": "miss_grasp_or_close_but_no_lift",
        "grasp_depth": {
            "descend_clearance": args.descend_clearance,
            "grasp_depth_offset": args.grasp_depth_offset,
            "effective_clearance": args.descend_clearance + args.grasp_depth_offset,
        },
        "contact_dwell": {
            "continuous_contact_dwell_steps": args.continuous_contact_dwell_steps,
        },
        "carry_stabilization": {
            "carry_stabilization_steps": args.carry_stabilization_steps,
        },
        "place_depth": {
            "safe_drop_height": args.safe_drop_height,
            "place_depth_offset": args.place_depth_offset,
        },
        "release_timing": {
            "release_timing_dwell_steps": args.release_timing_dwell_steps,
        },
        "soft_tolerance": {
            "continuous_soft_tolerance": args.continuous_soft_tolerance,
        },
    }


def _build_continuous_cycle_plan(
    *,
    pre_grasp_target: np.ndarray,
    grasp_depth_target: np.ndarray,
    lift_clearance_target: np.ndarray,
    prebin_target: np.ndarray,
    place_depth_target: np.ndarray,
    retreat_target: np.ndarray,
    args: argparse.Namespace,
) -> list[MotionSegment]:
    soft_hold_steps = 0 if args.smooth_motion else args.ik_hold_steps
    contact_hold_steps = args.continuous_contact_dwell_steps if args.smooth_motion else args.ik_hold_steps
    soft_tolerance = args.continuous_soft_tolerance if args.smooth_motion else args.ik_stop_tolerance
    return [
        MotionSegment(
            name="continuous_pregrasp",
            phase_tag="approach",
            target_position=pre_grasp_target,
            waypoint_type="soft",
            speed_profile="transit_blended",
            max_ik_steps=args.ik_steps,
            stop_tolerance=soft_tolerance,
            hold_steps=soft_hold_steps,
            blend_radius=args.continuous_blend_radius,
            details={"legacy_phase": "move_to_pre_grasp_front"},
        ),
        MotionSegment(
            name="continuous_grasp_depth",
            phase_tag="grasp_window",
            target_position=grasp_depth_target,
            waypoint_type="hard",
            speed_profile="slow_contact_approach",
            max_ik_steps=args.ik_steps,
            stop_tolerance=args.ik_stop_tolerance,
            hold_steps=contact_hold_steps,
            blend_radius=0.0,
            event_marker="close_gripper",
            contact_window=True,
            details={"legacy_phase": "descend_front"},
        ),
        MotionSegment(
            name="continuous_lift_clearance",
            phase_tag="lift",
            target_position=lift_clearance_target,
            waypoint_type="hard",
            speed_profile="slow_grasp_validation_lift",
            max_ik_steps=args.ik_steps,
            stop_tolerance=args.ik_stop_tolerance,
            hold_steps=contact_hold_steps,
            blend_radius=0.0,
            contact_window=True,
            gripper_effort=args.gripper_hold_effort,
            details={"legacy_phase": "grasp_validation_and_lift_front"},
        ),
        MotionSegment(
            name="continuous_prebin",
            phase_tag="carry",
            target_position=prebin_target,
            waypoint_type="soft",
            speed_profile="transit_blended_with_grip_effort",
            max_ik_steps=args.ik_steps,
            stop_tolerance=soft_tolerance,
            hold_steps=soft_hold_steps,
            blend_radius=args.continuous_blend_radius,
            gripper_effort=args.gripper_hold_effort,
            details={"legacy_phase": "move_to_bin_front"},
        ),
        MotionSegment(
            name="continuous_place_depth",
            phase_tag="place_window",
            target_position=place_depth_target,
            waypoint_type="hard",
            speed_profile="slow_place_release",
            max_ik_steps=args.ik_steps,
            stop_tolerance=args.ik_stop_tolerance,
            hold_steps=contact_hold_steps,
            blend_radius=0.0,
            event_marker="open_gripper",
            contact_window=True,
            gripper_effort=args.gripper_hold_effort,
            details={"legacy_phase": "release"},
        ),
        MotionSegment(
            name="continuous_retreat",
            phase_tag="retreat",
            target_position=retreat_target,
            waypoint_type="soft",
            speed_profile="transit_blended_after_release",
            max_ik_steps=args.ik_steps,
            stop_tolerance=soft_tolerance,
            hold_steps=soft_hold_steps,
            blend_radius=args.continuous_blend_radius,
            details={"legacy_phase": "retreat"},
        ),
    ]


def _trigger_motion_event(
    *,
    event_marker: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
) -> tuple[dict[str, Any], float | None]:
    if event_marker == "close_gripper":
        if args.skip_gripper_close:
            effort_result = _run_updates_with_optional_gripper_effort(
                sim_app,
                args.continuous_contact_dwell_steps,
                counter,
                dc,
                gripper_dofs,
                args.gripper_hold_effort,
            )
            return {
                "event_marker": event_marker,
                "skipped": True,
                "dwell_steps": args.continuous_contact_dwell_steps,
                "gripper_effort": effort_result,
            }, args.gripper_hold_effort
        close_targets = [OFFICIAL_GRIPPER_CLOSE_WIDTH] * len(gripper_dofs)
        _send_position_targets(dc, gripper_dofs, close_targets)
        effort_result = _run_updates_with_optional_gripper_effort(
            sim_app,
            args.continuous_contact_dwell_steps,
            counter,
            dc,
            gripper_dofs,
            args.gripper_hold_effort,
        )
        return {
            "event_marker": event_marker,
            "skipped": False,
            "dwell_steps": args.continuous_contact_dwell_steps,
            "commanded_targets": _named_positions(gripper_dofs, close_targets),
            "gripper_effort": effort_result,
        }, args.gripper_hold_effort

    if event_marker == "open_gripper":
        if args.skip_release:
            effort_result = _run_updates_with_optional_gripper_effort(
                sim_app,
                args.continuous_contact_dwell_steps,
                counter,
                dc,
                gripper_dofs,
                0.0,
            )
            return {
                "event_marker": event_marker,
                "skipped": True,
                "dwell_steps": args.continuous_contact_dwell_steps,
                "gripper_effort": effort_result,
            }, 0.0
        open_targets = [OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs)
        _send_position_targets(dc, gripper_dofs, open_targets)
        effort_result = _run_updates_with_optional_gripper_effort(
            sim_app,
            args.continuous_contact_dwell_steps,
            counter,
            dc,
            gripper_dofs,
            0.0,
        )
        return {
            "event_marker": event_marker,
            "skipped": False,
            "dwell_steps": args.continuous_contact_dwell_steps,
            "commanded_targets": _named_positions(gripper_dofs, open_targets),
            "gripper_effort": effort_result,
        }, None

    raise RuntimeError(f"Unknown motion event marker: {event_marker}")


def _micro_stop_estimate(
    *,
    start_position: np.ndarray,
    trace: list[dict[str, Any]],
    frame_span_per_trace_sample: int,
    threshold: float,
    contact_window: bool,
) -> dict[str, Any]:
    if contact_window:
        return {"micro_stop_frames": 0, "micro_stop_samples": 0}
    previous = start_position
    micro_stop_frames = 0
    micro_stop_samples = 0
    for sample in trace:
        actual = np.array(sample["actual"], dtype=float)
        speed = _distance(previous, actual) / float(max(frame_span_per_trace_sample, 1))
        if speed <= threshold:
            micro_stop_samples += 1
            micro_stop_frames += frame_span_per_trace_sample
        previous = actual
    return {
        "micro_stop_frames": micro_stop_frames,
        "micro_stop_samples": micro_stop_samples,
    }


def _execute_continuous_motion_cycle(
    *,
    target_index: int,
    target_path: str,
    stage: Any,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    segments: list[MotionSegment],
) -> dict[str, Any]:
    cycle_start_step = counter["step"]
    frame_span_per_trace_sample = (len(arm_dofs) + 1) * args.ik_settle_steps
    metrics: dict[str, Any] = {
        "target_index": target_index,
        "cycle_start_step": cycle_start_step,
        "segment_count": len(segments),
        "micro_stop_frames": 0,
        "micro_stop_samples": 0,
        "micro_stop_speed_threshold_m_per_frame": args.micro_stop_speed_threshold,
        "phase_tags": ["approach", "grasp_window", "post_close_verify", "lift", "carry", "place_window", "retreat"],
        "tuning_knobs": _tuning_knob_summary(args),
        "close_event_fired": False,
        "open_event_fired": False,
        "snapshots": {},
        "segment_summaries": [],
    }
    active_effort: float | None = None

    for segment in segments:
        start_step = counter["step"]
        start_ee = _body_pose_position(dc, end_effector_body)
        if active_effort is not None:
            effort_before = _apply_gripper_effort(dc, gripper_dofs, active_effort)
            if not effort_before["supported"]:
                raise RuntimeError(f"Gripper effort command failed before {segment.name}: {effort_before}")
        else:
            effort_before = None
        if segment.gripper_effort is not None:
            effort_before = _apply_gripper_effort(dc, gripper_dofs, segment.gripper_effort)
            if not effort_before["supported"]:
                raise RuntimeError(f"Gripper effort command failed before {segment.name}: {effort_before}")
            active_effort = segment.gripper_effort

        trace: list[dict[str, Any]] = []
        actual_position, position_error = move_end_effector_to_target(
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            segment.target_position,
            sim_app,
            segment.max_ik_steps,
            args.ik_settle_steps,
            args.ik_position_eps,
            args.ik_damping,
            args.ik_max_step,
            posture_positions=np.array(_current_positions(dc, arm_dofs), dtype=float),
            posture_gain=args.ik_posture_gain,
            stop_tolerance=segment.stop_tolerance,
            hold_steps=segment.hold_steps,
            phase_label=segment.name,
            trace=trace,
        )
        counter["step"] += len(trace) * frame_span_per_trace_sample + segment.hold_steps
        if active_effort is not None:
            effort_after = _apply_gripper_effort(dc, gripper_dofs, active_effort)
            if not effort_after["supported"]:
                raise RuntimeError(f"Gripper effort command failed after {segment.name}: {effort_after}")
        else:
            effort_after = None

        event_details = None
        if segment.event_marker is not None:
            if segment.event_marker == "open_gripper" and args.release_timing_dwell_steps > 0:
                _run_updates_with_optional_gripper_effort(
                    sim_app,
                    args.release_timing_dwell_steps,
                    counter,
                    dc,
                    gripper_dofs,
                    active_effort,
                )
            metrics["snapshots"][f"before_{segment.event_marker}"] = _bbox_state(stage, target_path)
            event_details, active_effort = _trigger_motion_event(
                event_marker=segment.event_marker,
                dc=dc,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
            )
            if segment.event_marker == "close_gripper":
                metrics["close_event_fired"] = True
            if segment.event_marker == "open_gripper":
                metrics["open_event_fired"] = True
            metrics["snapshots"][f"after_{segment.event_marker}"] = _bbox_state(stage, target_path)

        snapshot_name = f"after_{segment.name}"
        metrics["snapshots"][snapshot_name] = _bbox_state(stage, target_path)
        stop_estimate = _micro_stop_estimate(
            start_position=start_ee,
            trace=trace,
            frame_span_per_trace_sample=frame_span_per_trace_sample,
            threshold=args.micro_stop_speed_threshold,
            contact_window=segment.contact_window,
        )
        if not segment.contact_window and segment.hold_steps > 0:
            stop_estimate["micro_stop_frames"] += segment.hold_steps
        if segment.phase_tag == "carry" and args.carry_stabilization_steps > 0:
            carry_effort = active_effort if active_effort is not None else segment.gripper_effort
            carry_stabilization_effort = _run_updates_with_optional_gripper_effort(
                sim_app,
                args.carry_stabilization_steps,
                counter,
                dc,
                gripper_dofs,
                carry_effort,
            )
            metrics["snapshots"]["after_carry_stabilization"] = _bbox_state(stage, target_path)
        else:
            carry_stabilization_effort = None
        metrics["micro_stop_frames"] += int(stop_estimate["micro_stop_frames"])
        metrics["micro_stop_samples"] += int(stop_estimate["micro_stop_samples"])

        condition_met = bool(
            position_error <= segment.stop_tolerance
            or (segment.waypoint_type == "soft" and position_error <= args.pre_grasp_ee_tolerance)
        )
        segment_summary = {
            "name": segment.name,
            "phase_tag": segment.phase_tag,
            "waypoint_type": segment.waypoint_type,
            "speed_profile": segment.speed_profile,
            "blend_radius": segment.blend_radius,
            "target_position": segment.target_position.tolist(),
            "actual_position": actual_position.tolist(),
            "position_error": position_error,
            "condition_met": condition_met,
            "contact_window": segment.contact_window,
            "event_marker": segment.event_marker,
            "micro_stop_estimate": stop_estimate,
        }
        metrics["segment_summaries"].append(segment_summary)
        details = {
            "target_index": target_index,
            "continuous_motion": True,
            "phase_tag": segment.phase_tag,
            "waypoint_type": segment.waypoint_type,
            "speed_profile": segment.speed_profile,
            "blend_radius": segment.blend_radius,
            "target_position": segment.target_position.tolist(),
            "position_error": position_error,
            "ik_step_count": len(trace),
            "ik_trace": trace,
            "stop_tolerance": segment.stop_tolerance,
            "hold_steps": segment.hold_steps,
            "gripper_effort_before_segment": effort_before,
            "gripper_effort_after_segment": effort_after,
            "event": event_details,
            "carry_stabilization_steps": args.carry_stabilization_steps if segment.phase_tag == "carry" else 0,
            "carry_stabilization_effort": carry_stabilization_effort,
        }
        details.update(segment.details)
        _append_phase(
            phase_log,
            phase=segment.name,
            start_step=start_step,
            end_step=counter["step"],
            commanded_targets=_named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
            ee_position=actual_position,
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=condition_met,
            details=details,
        )
        print(
            f"phase={segment.name} waypoint_type={segment.waypoint_type} "
            f"position_error={position_error} event={segment.event_marker}"
        )

    metrics["cycle_end_step"] = counter["step"]
    metrics["cycle_time_steps"] = counter["step"] - cycle_start_step
    return metrics


def _smooth_pause_steps(args: argparse.Namespace, phase_name: str) -> int:
    if not args.smooth_motion:
        return args.pause_steps
    if phase_name in {"close_gripper", "grasp_validation", "release"}:
        return args.pause_steps
    return args.smooth_non_contact_pause_steps


def _smooth_ik_hold_steps(args: argparse.Namespace, phase_name: str) -> int:
    if not args.smooth_motion:
        return args.ik_hold_steps
    if phase_name == "grasp_validation":
        return args.ik_hold_steps
    return args.smooth_ik_hold_steps


def _command_joint_phase(
    phase_name: str,
    dc: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    target_positions: list[float],
    sim_app: Any,
    phase_steps: int,
    pause_steps: int,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    joint_tolerance: float,
    extra_details: dict[str, Any] | None = None,
) -> tuple[np.ndarray, bool]:
    start_step = counter["step"]
    start_positions = _current_positions(dc, arm_dofs)
    for step in range(1, phase_steps + 1):
        alpha = step / float(phase_steps)
        command = [
            float(start + alpha * (target - start))
            for start, target in zip(start_positions, target_positions)
        ]
        _send_position_targets(dc, arm_dofs, command)
        sim_app.update()
        counter["step"] += 1
    _run_updates(sim_app, pause_steps, counter)

    observed = _current_positions(dc, arm_dofs)
    ee_position = _body_pose_position(dc, end_effector_body)
    max_joint_error = max(abs(float(obs - target)) for obs, target in zip(observed, target_positions))
    condition_met = bool(max_joint_error <= joint_tolerance)
    details = {
        "max_joint_error": max_joint_error,
        "joint_tolerance": joint_tolerance,
        "observed_joint_values": _named_positions(arm_dofs, observed),
    }
    if extra_details:
        details.update(extra_details)
    _append_phase(
        phase_log,
        phase=phase_name,
        start_step=start_step,
        end_step=counter["step"],
        commanded_targets=_named_positions(arm_dofs, target_positions),
        ee_position=ee_position,
        gripper_values=_gripper_values(dc, gripper_dofs),
        condition_met=condition_met,
        details=details,
    )
    print(f"phase={phase_name} condition_met={condition_met} ee={ee_position.tolist()}")
    return ee_position, condition_met


def _command_ee_phase(
    phase_name: str,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    target_position: np.ndarray,
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    extra_details: dict[str, Any] | None = None,
    gripper_effort_value: float | None = None,
) -> tuple[np.ndarray, float]:
    start_step = counter["step"]
    trace: list[dict[str, Any]] = []
    phase_hold_steps = _smooth_ik_hold_steps(args, phase_name)
    gripper_effort_before = None
    if gripper_effort_value is not None:
        gripper_effort_before = _apply_gripper_effort(dc, gripper_dofs, gripper_effort_value)
        if not gripper_effort_before["supported"]:
            raise RuntimeError(f"Gripper effort command failed before {phase_name}: {gripper_effort_before}")
    actual_position, position_error = move_end_effector_to_target(
        dc,
        articulation,
        arm_dofs,
        end_effector_body,
        target_position,
        sim_app,
        args.ik_steps,
        args.ik_settle_steps,
        args.ik_position_eps,
        args.ik_damping,
        args.ik_max_step,
        posture_positions=np.array(_current_positions(dc, arm_dofs), dtype=float),
        posture_gain=args.ik_posture_gain,
        stop_tolerance=args.ik_stop_tolerance,
        hold_steps=phase_hold_steps,
        phase_label=phase_name,
        trace=trace,
    )
    gripper_effort_after = None
    if gripper_effort_value is not None:
        gripper_effort_after = _apply_gripper_effort(dc, gripper_dofs, gripper_effort_value)
        if not gripper_effort_after["supported"]:
            raise RuntimeError(f"Gripper effort command failed after {phase_name}: {gripper_effort_after}")
    counter["step"] += len(trace) * (len(arm_dofs) + 1) * args.ik_settle_steps + phase_hold_steps
    observed_positions = _current_positions(dc, arm_dofs)
    details = {
        "target_position": target_position.tolist(),
        "position_error": position_error,
        "ik_step_count": len(trace),
        "ik_trace": trace,
        "observed_joint_values": _named_positions(arm_dofs, observed_positions),
        "ik_parameters": {
            "ik_steps": args.ik_steps,
            "ik_settle_steps": args.ik_settle_steps,
            "ik_position_eps": args.ik_position_eps,
            "ik_damping": args.ik_damping,
            "ik_max_step": args.ik_max_step,
            "ik_posture_gain": args.ik_posture_gain,
            "ik_stop_tolerance": args.ik_stop_tolerance,
            "ik_hold_steps": phase_hold_steps,
            "smooth_motion": bool(args.smooth_motion),
        },
    }
    if gripper_effort_value is not None:
        details.update(
            {
                "gripper_effort_active": True,
                "gripper_effort_value": float(gripper_effort_value),
                "gripper_effort_before_phase": gripper_effort_before,
                "gripper_effort_after_phase": gripper_effort_after,
            }
        )
    if extra_details:
        details.update(extra_details)
    _append_phase(
        phase_log,
        phase=phase_name,
        start_step=start_step,
        end_step=counter["step"],
        commanded_targets=_named_positions(arm_dofs, observed_positions),
        ee_position=actual_position,
        gripper_values=_gripper_values(dc, gripper_dofs),
        condition_met=bool(position_error <= args.pre_grasp_ee_tolerance),
        details=details,
    )
    print(
        f"phase={phase_name} position_error={position_error} "
        f"condition_met={position_error <= args.pre_grasp_ee_tolerance} ee={actual_position.tolist()}"
    )
    return actual_position, position_error


def _command_gripper_phase(
    phase_name: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    target_positions: list[float],
    sim_app: Any,
    steps: int,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    skipped: bool = False,
    effort_value: float | None = None,
) -> bool:
    start_step = counter["step"]
    targets: list[float] = []
    effort_result = None
    if skipped:
        effort_result = _run_updates_with_optional_gripper_effort(
            sim_app,
            steps,
            counter,
            dc,
            gripper_dofs,
            effort_value,
        )
        observed = _read_positions(dc, gripper_dofs)
    else:
        if len(target_positions) != len(gripper_dofs):
            raise ValueError(f"Expected {len(gripper_dofs)} gripper targets, got {len(target_positions)}")
        targets = [float(value) for value in target_positions]
        _send_position_targets(dc, gripper_dofs, targets)
        effort_result = _run_updates_with_optional_gripper_effort(
            sim_app,
            steps,
            counter,
            dc,
            gripper_dofs,
            effort_value,
        )
        observed = _read_positions(dc, gripper_dofs)
    condition_met = bool(skipped or targets)
    _append_phase(
        phase_log,
        phase=phase_name,
        start_step=start_step,
        end_step=counter["step"],
        commanded_targets=None if skipped else _named_positions(gripper_dofs, targets),
        ee_position=_body_pose_position(dc, end_effector_body),
        gripper_values=_named_positions(gripper_dofs, observed),
        condition_met=condition_met,
        details={
            "skipped": skipped,
            "official_gripper_open_width": OFFICIAL_GRIPPER_OPEN_WIDTH,
            "official_gripper_close_width": OFFICIAL_GRIPPER_CLOSE_WIDTH,
            "target_semantics": "official same-sign finger joint positions",
            "gripper_effort": effort_result,
        },
    )
    print(f"phase={phase_name} condition_met={condition_met} skipped={skipped}")
    return condition_met


def _inside_bin(center: np.ndarray, bin_bbox: dict[str, list[float]], wall_thickness: float, floor_top_z: float) -> bool:
    min_v = bin_bbox["min"]
    max_v = bin_bbox["max"]
    return bool(
        min_v[0] + wall_thickness <= center[0] <= max_v[0] - wall_thickness
        and min_v[1] + wall_thickness <= center[1] <= max_v[1] - wall_thickness
        and floor_top_z <= center[2] <= max_v[2] + 0.12
    )


def _settle_and_measure(
    stage: Any,
    target_path: str,
    sim_app: Any,
    settle_steps: int,
    counter: dict[str, int],
) -> tuple[dict[str, Any], float]:
    samples: list[np.ndarray] = []
    sample_start = max(0, settle_steps - 30)
    for step in range(settle_steps):
        sim_app.update()
        counter["step"] += 1
        if step >= sample_start:
            samples.append(_center_from_bbox(_bbox(stage, target_path)))

    final_state = _bbox_state(stage, target_path)
    jitter = math.inf
    finite_samples = [sample for sample in samples if np.isfinite(sample).all()]
    if finite_samples:
        mean = np.mean(np.stack(finite_samples, axis=0), axis=0)
        jitter = max(float(np.linalg.norm(sample - mean)) for sample in finite_samples)
    return final_state, jitter


def _fail(reason: str, message: str) -> None:
    raise RunFailure(reason, message)


def _phase_from_failure_reason(reason: str) -> str:
    mapping = {
        "target_outside_front_workspace": "approach",
        "pre_grasp_unreachable": "approach",
        "descend_failed": "grasp_window",
        "gripper_command_failed": "grasp_window",
        "continuous_motion_failed": "approach",
        "object_not_lifted": "post_close_verify",
        "dropped_during_lift": "lift",
        "dropped_during_transport": "carry",
        "release_failed": "place_window",
        "object_outside_bin": "place_window",
        "object_unstable_after_settle": "retreat",
    }
    return mapping.get(reason, "unknown")


def _retained_with_gripper(
    *,
    object_center: np.ndarray,
    ee_position: np.ndarray,
    initial_center: np.ndarray,
    min_lift_delta: float,
    distance_tolerance: float,
) -> bool:
    return bool(
        object_center[2] >= initial_center[2] + min_lift_delta
        and _distance(ee_position, object_center) <= distance_tolerance
    )


def _failure_kind(result: dict[str, Any], reason: str) -> str:
    if reason in {"pre_grasp_unreachable", "descend_failed"}:
        return "miss_grasp"
    if reason == "object_not_lifted":
        return "close_but_no_lift" if result.get("close_event_fired") else "miss_grasp"
    if reason in {"dropped_during_lift", "dropped_during_transport"}:
        return "lift_then_slip_during_carry"
    if reason == "object_outside_bin":
        return "release_timing_failure" if result.get("object_retained_at_preplace") else "near_bin_place_failure"
    if reason == "object_unstable_after_settle":
        return "near_bin_place_failure"
    if reason == "release_failed":
        return "release_timing_failure"
    return "runtime_or_safety_failure"


def _attempt_pick_place_target(
    *,
    attempt_number: int,
    target_index: int,
    target_path: str,
    stage: Any,
    cfg: dict[str, Any],
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    front_workspace_x: tuple[float, float],
    front_workspace_y: tuple[float, float],
    front_workspace_z: tuple[float, float],
    table_top_z: float,
    robot_base_position: list[float],
    bin_bbox: dict[str, list[float]],
    bin_collider: dict[str, Any],
    marker_paths: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempt_number": attempt_number,
        "target_index": target_index,
        "selected_object_id": target_path,
        "target_prim": target_path,
        "success": False,
        "failure_phase": None,
        "failure_reason": None,
        "first_failing_phase": None,
        "object_lifted": False,
        "object_retained_after_lift": False,
        "close_event_fired": False,
        "open_event_fired": False,
        "lift_height_threshold_achieved": False,
        "object_retained_at_carry_midpoint": None,
        "object_retained_at_carry_midpoint_source": "not_evaluated",
        "object_retained_at_prebin": None,
        "object_retained_at_preplace": None,
        "release_timing_failure_suspected": False,
        "object_transported": False,
        "final_inside_bin": False,
        "object_stable": False,
        "hard_stop": False,
    }
    hard_stop_reasons = {
        "target_outside_front_workspace",
        "pre_grasp_unreachable",
        "descend_failed",
        "gripper_command_failed",
        "continuous_motion_failed",
        "release_failed",
    }

    target_prim = stage.GetPrimAtPath(target_path)
    target_refs = _reference_paths(target_prim) if target_prim and target_prim.IsValid() else []
    category_from_refs = _category_from_reference(target_refs)
    num_parts_per_class = int(cfg["part"].get("num_parts", 2))
    category_from_order = "part_a" if target_index < num_parts_per_class else "part_b"
    category_for_log = category_from_refs if category_from_refs != "unknown" else category_from_order
    category_inference_method = "reference_path" if category_from_refs != "unknown" else "scene_builder_creation_order"
    initial_state = _bbox_state(stage, target_path)
    initial_center = _center_from_bbox(initial_state["bbox"])
    pre_grasp_pose = {
        "position": [
            initial_state["bbox"]["center"][0],
            initial_state["bbox"]["center"][1],
            initial_state["bbox"]["max"][2] + args.pre_grasp_clearance,
        ],
        "orientation": "fixed_downward",
        "orientation_search": False,
    }
    descend_pose = {
        "position": [
            initial_state["bbox"]["center"][0],
            initial_state["bbox"]["center"][1],
            max(
                initial_state["bbox"]["max"][2] + args.descend_clearance + args.grasp_depth_offset,
                table_top_z + args.min_ee_table_clearance,
            ),
        ],
        "orientation": "fixed_downward",
        "orientation_search": False,
    }
    bin_drop_pose = {
        "position": [
            bin_bbox["center"][0],
            bin_bbox["center"][1],
            bin_bbox["max"][2] + args.safe_drop_height,
        ],
        "orientation": "fixed_downward",
        "orientation_search": False,
    }
    target_workspace_checks = _front_workspace_checks(
        initial_center,
        front_workspace_x,
        front_workspace_y,
        front_workspace_z,
        table_top_z,
    )
    pre_grasp_workspace_checks = _front_workspace_checks(
        np.array(pre_grasp_pose["position"], dtype=float),
        front_workspace_x,
        front_workspace_y,
        front_workspace_z,
        table_top_z,
    )
    pre_grasp_geometry = _pregrasp_geometry_summary(
        target_index=target_index,
        target_path=target_path,
        object_center=initial_center,
        bbox_top_z=float(initial_state["bbox"]["max"][2]),
        pre_grasp_position=np.array(pre_grasp_pose["position"], dtype=float),
        robot_base_position=robot_base_position,
        table_top_z=table_top_z,
    )
    _print_pregrasp_geometry(pre_grasp_geometry)
    marker_paths.extend(
        [
            _debug_marker(stage, f"/World/DebugTask1Target_{target_index}", initial_state["bbox"]["center"], 0.025, (1.0, 0.2, 0.1)),
            _debug_marker(stage, f"/World/DebugTask1PreGraspFront_{target_index}", pre_grasp_pose["position"], 0.025, (0.2, 0.6, 1.0)),
        ]
    )
    result.update(
        {
            "referenced_usd_paths": target_refs,
            "category_from_reference": category_from_refs,
            "category_from_scene_builder_order": category_from_order,
            "inferred_category": category_for_log,
            "category_inference_method": category_inference_method,
            "initial_pose": initial_state,
            "pre_grasp_pose": pre_grasp_pose,
            "pre_grasp_geometry": pre_grasp_geometry,
            "descend_pose": descend_pose,
            "bin_drop_pose": bin_drop_pose,
            "target_workspace_checks": target_workspace_checks,
            "pre_grasp_workspace_checks": pre_grasp_workspace_checks,
        }
    )

    try:
        if not target_workspace_checks["front_workspace_ok"] or not pre_grasp_workspace_checks["front_workspace_ok"]:
            _fail(
                "target_outside_front_workspace",
                "Selected target or explicit pre-grasp pose is outside the conservative front tabletop workspace",
            )

        pre_grasp_target_np = np.array(pre_grasp_pose["position"], dtype=float)
        grasp_depth_target_np = np.array(descend_pose["position"], dtype=float)
        lift_target_np = grasp_depth_target_np.copy()
        lift_target_np[2] = max(
            grasp_depth_target_np[2] + args.pre_grasp_clearance,
            initial_center[2] + args.pre_grasp_clearance + args.min_lift_delta,
            table_top_z + args.min_ee_table_clearance + 0.08,
        )
        prebin_target_np = np.array(bin_drop_pose["position"], dtype=float)
        place_depth_target_np = prebin_target_np.copy()
        place_depth_target_np[2] = max(
            float(bin_bbox["max"][2]) + min(args.safe_drop_height * 0.35, 0.04) + args.place_depth_offset,
            table_top_z + args.min_ee_table_clearance,
        )
        retreat_target_np = prebin_target_np.copy()
        place_pose = {
            "position": place_depth_target_np.tolist(),
            "orientation": "fixed_downward",
            "orientation_search": False,
        }
        result["place_pose"] = place_pose

        continuous_plan = _build_continuous_cycle_plan(
            pre_grasp_target=pre_grasp_target_np,
            grasp_depth_target=grasp_depth_target_np,
            lift_clearance_target=lift_target_np,
            prebin_target=prebin_target_np,
            place_depth_target=place_depth_target_np,
            retreat_target=retreat_target_np,
            args=args,
        )
        result["continuous_motion_plan"] = [
            {
                "name": segment.name,
                "phase_tag": segment.phase_tag,
                "waypoint_type": segment.waypoint_type,
                "speed_profile": segment.speed_profile,
                "blend_radius": segment.blend_radius,
                "event_marker": segment.event_marker,
                "target_position": segment.target_position.tolist(),
            }
            for segment in continuous_plan
        ]
        try:
            cycle_metrics = _execute_continuous_motion_cycle(
                target_index=target_index,
                target_path=target_path,
                stage=stage,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                gripper_dofs=gripper_dofs,
                end_effector_body=end_effector_body,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                segments=continuous_plan,
            )
        except RuntimeError as exc:
            if "Gripper" in str(exc) or "gripper" in str(exc):
                _fail("gripper_command_failed", str(exc))
            _fail("continuous_motion_failed", str(exc))
        result["cycle_time_steps"] = cycle_metrics["cycle_time_steps"]
        result["micro_stop_frames"] = cycle_metrics["micro_stop_frames"]
        result["micro_stop_samples"] = cycle_metrics["micro_stop_samples"]
        result["close_event_fired"] = bool(cycle_metrics.get("close_event_fired"))
        result["open_event_fired"] = bool(cycle_metrics.get("open_event_fired"))
        result["continuous_cycle_metrics"] = cycle_metrics

        segment_by_name = {item["name"]: item for item in cycle_metrics["segment_summaries"]}
        phase_by_name = {entry["phase"]: entry for entry in phase_log if entry.get("details", {}).get("target_index") == target_index}
        ee_pre = np.array(segment_by_name["continuous_pregrasp"]["actual_position"], dtype=float)
        pre_distance = _distance(ee_pre, pre_grasp_target_np)
        pre_safety = _ee_front_safety_checks(ee_pre, front_workspace_x, front_workspace_y, table_top_z, args.min_ee_table_clearance)
        if "continuous_pregrasp" in phase_by_name:
            phase_by_name["continuous_pregrasp"]["details"]["ee_to_pre_grasp_target_distance"] = pre_distance
            phase_by_name["continuous_pregrasp"]["details"]["pre_grasp_front_safety_checks"] = pre_safety
            phase_by_name["continuous_pregrasp"]["condition_met"] = bool(
                pre_safety["front_safety_ok"] and pre_distance <= args.pre_grasp_ee_tolerance
            )
        if not pre_safety["front_safety_ok"]:
            _fail("pre_grasp_unreachable", "Pre-grasp end-effector target left the front tabletop workspace")
        if pre_distance > args.pre_grasp_ee_tolerance:
            _fail("pre_grasp_unreachable", f"Pre-grasp end effector remained {pre_distance:.3f} m from explicit pre-grasp pose")

        ee_descend = np.array(segment_by_name["continuous_grasp_depth"]["actual_position"], dtype=float)
        descend_error = float(segment_by_name["continuous_grasp_depth"]["position_error"])
        after_descend = cycle_metrics["snapshots"].get("before_close_gripper") or cycle_metrics["snapshots"]["after_continuous_grasp_depth"]
        object_distance_before_close = _distance(ee_descend, _center_from_bbox(after_descend["bbox"]))
        descend_safety = _ee_front_safety_checks(ee_descend, front_workspace_x, front_workspace_y, table_top_z, 0.0)
        if "continuous_grasp_depth" in phase_by_name:
            phase_by_name["continuous_grasp_depth"]["details"]["target_object_distance_before_close"] = object_distance_before_close
            phase_by_name["continuous_grasp_depth"]["details"]["ee_to_explicit_descend_target_distance"] = descend_error
            phase_by_name["continuous_grasp_depth"]["details"]["descend_front_safety_checks"] = descend_safety
            phase_by_name["continuous_grasp_depth"]["condition_met"] = bool(
                descend_safety["front_safety_ok"] and object_distance_before_close <= args.descend_object_tolerance
            )
        if not descend_safety["front_safety_ok"]:
            _fail("descend_failed", "Descend front target left the front workspace or moved under the table")
        if object_distance_before_close > args.descend_object_tolerance:
            _fail("descend_failed", f"End effector remained {object_distance_before_close:.3f} m from target object before close")

        validation_before = cycle_metrics["snapshots"].get("before_close_gripper") or after_descend
        validation_before_center = _center_from_bbox(validation_before["bbox"])
        ee_validation = np.array(segment_by_name["continuous_lift_clearance"]["actual_position"], dtype=float)
        validation_error = float(segment_by_name["continuous_lift_clearance"]["position_error"])
        validation_after = cycle_metrics["snapshots"]["after_continuous_lift_clearance"]
        validation_after_center = _center_from_bbox(validation_after["bbox"])
        validation_delta = validation_after_center - validation_before_center
        ee_to_object_after_validation = _distance(ee_validation, validation_after_center)
        validation_safety = _ee_front_safety_checks(ee_validation, front_workspace_x, front_workspace_y, table_top_z, args.min_ee_table_clearance)
        object_lifted = bool(
            validation_delta[2] >= args.min_lift_delta
            and ee_to_object_after_validation <= args.descend_object_tolerance
            and validation_safety["front_safety_ok"]
        )
        lift_height_threshold_achieved = bool(validation_delta[2] >= args.min_lift_delta)
        result["object_lifted"] = object_lifted
        result["lift_height_threshold_achieved"] = lift_height_threshold_achieved
        if not result["close_event_fired"]:
            _fail("gripper_command_failed", "Continuous close_gripper event marker did not fire before lift validation")
        if "continuous_lift_clearance" in phase_by_name:
            phase_by_name["continuous_lift_clearance"]["details"].update(
                {
                    "object_pose_before_validation": validation_before,
                    "object_pose_after_validation": validation_after,
                    "object_delta_during_validation_m": validation_delta.tolist(),
                    "ee_to_object_after_validation": ee_to_object_after_validation,
                    "ee_to_validation_target_distance": validation_error,
                    "grasp_validation_front_safety_checks": validation_safety,
                    "post_close_verify_phase_tag": "post_close_verify",
                    "lift_phase_tag": "lift",
                    "object_lifted": object_lifted,
                    "lift_height_threshold_achieved": lift_height_threshold_achieved,
                    "gripper_effort_active_during_validation": True,
                    "gripper_effort_value": args.gripper_hold_effort,
                }
            )
            phase_by_name["continuous_lift_clearance"]["condition_met"] = bool(object_lifted)
        if not object_lifted:
            _fail("object_not_lifted", "Object did not move upward with the gripper during mandatory grasp validation")

        ee_lift = ee_validation
        lift_error = validation_error
        after_lift = validation_after
        lift_center = _center_from_bbox(after_lift["bbox"])
        lift_delta = lift_center - validation_before_center
        ee_to_object_after_lift = _distance(ee_lift, lift_center)
        lift_safety = _ee_front_safety_checks(ee_lift, front_workspace_x, front_workspace_y, table_top_z, args.min_ee_table_clearance)
        object_retained_after_lift = bool(
            lift_center[2] >= initial_center[2] + args.min_lift_delta
            and ee_to_object_after_lift <= args.descend_object_tolerance
            and lift_safety["front_safety_ok"]
        )
        result["object_retained_after_lift"] = object_retained_after_lift
        if "continuous_lift_clearance" in phase_by_name:
            phase_by_name["continuous_lift_clearance"]["details"].update(
                {
                    "ee_to_lift_target_distance": lift_error,
                    "lift_front_safety_checks": lift_safety,
                    "object_delta_during_lift_m": lift_delta.tolist(),
                    "ee_to_object_after_lift": ee_to_object_after_lift,
                    "object_retained_after_lift": object_retained_after_lift,
                    "gripper_effort_active_during_lift": True,
                    "gripper_effort_value": args.gripper_hold_effort,
                }
            )
            phase_by_name["continuous_lift_clearance"]["condition_met"] = bool(object_retained_after_lift)
        if not lift_safety["front_safety_ok"]:
            _fail("dropped_during_lift", "Lift front target left the front workspace or moved too close to the table")
        if lift_center[2] < initial_center[2] + args.min_lift_delta:
            _fail("dropped_during_lift", "Object was not above initial height after lift")
        if ee_to_object_after_lift > args.descend_object_tolerance:
            _fail("dropped_during_lift", "Object did not remain near gripper after lift")

        bin_target_np = prebin_target_np
        ee_bin = np.array(segment_by_name["continuous_prebin"]["actual_position"], dtype=float)
        bin_error = float(segment_by_name["continuous_prebin"]["position_error"])
        after_transport = cycle_metrics["snapshots"]["after_continuous_prebin"]
        transport_center = _center_from_bbox(after_transport["bbox"])
        prebin_retained = _retained_with_gripper(
            object_center=transport_center,
            ee_position=ee_bin,
            initial_center=initial_center,
            min_lift_delta=args.min_lift_delta,
            distance_tolerance=args.descend_object_tolerance,
        )
        result["object_retained_at_prebin"] = prebin_retained
        result["object_retained_at_carry_midpoint"] = bool(object_retained_after_lift and prebin_retained)
        result["object_retained_at_carry_midpoint_source"] = "inferred_from_lift_and_prebin_retention"
        transport_distance = _distance(transport_center, initial_center)
        distance_to_bin_initial = _distance(initial_center, np.array(bin_bbox["center"], dtype=float))
        distance_to_bin_after = _distance(transport_center, np.array(bin_bbox["center"], dtype=float))
        bin_safety = _ee_front_safety_checks(ee_bin, front_workspace_x, front_workspace_y, table_top_z, args.min_ee_table_clearance)
        above_bin_wall = bool(transport_center[2] >= float(bin_bbox["max"][2]) - 0.02)
        object_transported = bool(
            transport_distance >= args.min_transport_distance
            and distance_to_bin_after < distance_to_bin_initial
            and bin_safety["front_safety_ok"]
            and above_bin_wall
        )
        result["object_transported"] = object_transported
        result["transport_distance_m"] = transport_distance
        if "continuous_prebin" in phase_by_name:
            phase_by_name["continuous_prebin"]["details"].update(
                {
                    "explicit_bin_drop_pose": bin_drop_pose,
                    "ee_to_explicit_bin_drop_target_distance": _distance(ee_bin, bin_target_np),
                    "bin_target_position_error": bin_error,
                    "object_transport_distance_from_initial": transport_distance,
                    "object_distance_to_bin_initial": distance_to_bin_initial,
                    "object_distance_to_bin_after_transport": distance_to_bin_after,
                    "move_to_bin_front_safety_checks": bin_safety,
                    "object_above_bin_wall_or_near_drop_height": above_bin_wall,
                    "object_transported": object_transported,
                    "object_retained_at_carry_midpoint": result["object_retained_at_carry_midpoint"],
                    "object_retained_at_carry_midpoint_source": result["object_retained_at_carry_midpoint_source"],
                    "object_retained_at_prebin": prebin_retained,
                }
            )
            phase_by_name["continuous_prebin"]["condition_met"] = bool(object_transported)
        if not object_transported:
            _fail("dropped_during_transport", "Object did not move the required minimum distance toward the destination bin")

        preplace_state = cycle_metrics["snapshots"].get("before_open_gripper") or cycle_metrics["snapshots"]["after_continuous_place_depth"]
        preplace_center = _center_from_bbox(preplace_state["bbox"])
        ee_preplace = np.array(segment_by_name["continuous_place_depth"]["actual_position"], dtype=float)
        preplace_retained = _retained_with_gripper(
            object_center=preplace_center,
            ee_position=ee_preplace,
            initial_center=initial_center,
            min_lift_delta=args.min_lift_delta,
            distance_tolerance=args.descend_object_tolerance,
        )
        result["object_retained_at_preplace"] = preplace_retained
        if "continuous_place_depth" in phase_by_name:
            phase_by_name["continuous_place_depth"]["details"].update(
                {
                    "object_pose_before_release": preplace_state,
                    "ee_to_object_before_release": _distance(ee_preplace, preplace_center),
                    "object_retained_at_preplace": preplace_retained,
                    "open_event_fired": result["open_event_fired"],
                }
            )

        settle_start = counter["step"]
        final_state, final_jitter = _settle_and_measure(stage, target_path, sim_app, args.settle_steps, counter)
        final_center = _center_from_bbox(final_state["bbox"])
        final_inside_bin = _inside_bin(final_center, bin_bbox, float(bin_collider["wall_thickness"]), float(bin_collider["floor_top_z"]))
        object_stable = bool(final_jitter <= args.stable_jitter)
        result["final_inside_bin"] = final_inside_bin
        result["object_stable"] = object_stable
        _append_phase(
            phase_log,
            phase="settle",
            start_step=settle_start,
            end_step=counter["step"],
            commanded_targets=None,
            ee_position=_body_pose_position(dc, end_effector_body),
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=bool(final_inside_bin and object_stable),
            details={
                "target_index": target_index,
                "final_target_pose": final_state,
                "final_jitter_m": final_jitter,
                "stable_jitter_threshold_m": args.stable_jitter,
                "final_inside_bin": final_inside_bin,
                "object_stable": object_stable,
            },
        )
        if not final_inside_bin:
            result["release_timing_failure_suspected"] = bool(result["open_event_fired"] and result["object_retained_at_preplace"])
            _fail("object_outside_bin", "Target object final pose is outside the diagnostic bin volume")
        if not object_stable:
            _fail("object_unstable_after_settle", "Target object did not settle stably after release")

        result["success"] = True
        result["failure_phase"] = None
        result["diagnostic_failure_kind"] = "success"
        return result
    except RunFailure as exc:
        failure_phase = _phase_from_failure_reason(exc.reason)
        result["failure_reason"] = exc.reason
        result["failure_phase"] = failure_phase
        result["first_failing_phase"] = failure_phase
        result["diagnostic_failure_kind"] = _failure_kind(result, exc.reason)
        result["error"] = str(exc)
        result["hard_stop"] = exc.reason in hard_stop_reasons
        try:
            _command_gripper_phase(
                "recovery_open_after_failed_attempt",
                dc,
                gripper_dofs,
                end_effector_body,
                [OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs),
                sim_app,
                _smooth_pause_steps(args, "release"),
                counter,
                phase_log,
                effort_value=0.0,
            )
        except Exception as recovery_exc:
            result["hard_stop"] = True
            result["recovery_error"] = str(recovery_exc)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--asset-root")
    parser.add_argument("--prim-path", default=OFFICIAL_ROBOT_PRIM_PATH)
    parser.add_argument("--end-effector-body")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
    parser.add_argument("--pause-steps", type=int, default=DEFAULT_PAUSE_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-suffix")
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA, help="Deprecated; official fixed gripper widths are used.")
    parser.add_argument("--gripper-hold-effort", type=float, default=DEFAULT_GRIPPER_HOLD_EFFORT)
    parser.add_argument("--pre-grasp-clearance", type=float, default=DEFAULT_PRE_GRASP_CLEARANCE)
    parser.add_argument("--safe-drop-height", type=float, default=DEFAULT_SAFE_DROP_HEIGHT)
    parser.add_argument("--stable-jitter", type=float, default=DEFAULT_STABLE_JITTER)
    parser.add_argument("--min-lift-delta", type=float, default=DEFAULT_MIN_LIFT_DELTA)
    parser.add_argument("--min-transport-distance", type=float, default=DEFAULT_MIN_TRANSPORT_DISTANCE)
    parser.add_argument("--pre-grasp-ee-tolerance", type=float, default=DEFAULT_PRE_GRASP_EE_TOLERANCE)
    parser.add_argument("--descend-object-tolerance", type=float, default=DEFAULT_DESCEND_OBJECT_TOLERANCE)
    parser.add_argument("--joint-tolerance", type=float, default=DEFAULT_JOINT_TOLERANCE)
    parser.add_argument("--max-local-joint-adjustment", type=float, default=DEFAULT_MAX_LOCAL_JOINT_ADJUSTMENT)
    parser.add_argument("--front-workspace-x", nargs=2, type=float, default=list(DEFAULT_FRONT_WORKSPACE_X), metavar=("MIN", "MAX"))
    parser.add_argument("--front-workspace-y", nargs=2, type=float, default=list(DEFAULT_FRONT_WORKSPACE_Y), metavar=("MIN", "MAX"))
    parser.add_argument("--front-workspace-z", nargs=2, type=float, default=list(DEFAULT_FRONT_WORKSPACE_Z), metavar=("MIN", "MAX"))
    parser.add_argument("--min-ee-table-clearance", type=float, default=DEFAULT_MIN_EE_TABLE_CLEARANCE)
    parser.add_argument("--ik-steps", type=int, default=DEFAULT_IK_STEPS)
    parser.add_argument("--ik-settle-steps", type=int, default=DEFAULT_IK_SETTLE_STEPS)
    parser.add_argument("--ik-position-eps", type=float, default=DEFAULT_POSITION_EPS)
    parser.add_argument("--ik-damping", type=float, default=DEFAULT_DAMPING)
    parser.add_argument("--ik-max-step", type=float, default=DEFAULT_IK_MAX_STEP)
    parser.add_argument("--ik-posture-gain", type=float, default=DEFAULT_POSTURE_GAIN)
    parser.add_argument("--ik-stop-tolerance", type=float, default=DEFAULT_STOP_TOLERANCE)
    parser.add_argument("--ik-hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument("--smooth-motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-ik-hold-steps", type=int, default=1)
    parser.add_argument("--smooth-non-contact-pause-steps", type=int, default=1)
    parser.add_argument("--continuous-soft-tolerance", type=float, default=DEFAULT_CONTINUOUS_SOFT_TOLERANCE)
    parser.add_argument("--continuous-blend-radius", type=float, default=DEFAULT_CONTINUOUS_BLEND_RADIUS)
    parser.add_argument("--micro-stop-speed-threshold", type=float, default=DEFAULT_MICRO_STOP_SPEED_THRESHOLD)
    tuning_group = parser.add_argument_group(
        "controlled tuning knobs",
        "Sweep one family at a time: grasp depth, contact dwell, carry stabilization, place depth, release timing, then soft tolerance.",
    )
    tuning_group.add_argument("--descend-clearance", type=float, default=DEFAULT_DESCEND_CLEARANCE, help="Base grasp-depth clearance above the object bbox top.")
    tuning_group.add_argument("--grasp-depth-offset", type=float, default=DEFAULT_GRASP_DEPTH_TUNING_OFFSET, help="Active first-pass grasp-depth sweep knob; negative values insert slightly deeper.")
    tuning_group.add_argument("--continuous-contact-dwell-steps", type=int, default=DEFAULT_CONTINUOUS_CONTACT_DWELL_STEPS, help="Contact dwell knob for close/lift/place windows; leave unchanged unless close succeeds but lift is inconsistent.")
    tuning_group.add_argument("--carry-stabilization-steps", type=int, default=DEFAULT_CARRY_STABILIZATION_STEPS, help="Neutral by default; use only if lift succeeds but object_retained_at_prebin fails.")
    tuning_group.add_argument("--place-depth-offset", type=float, default=DEFAULT_PLACE_DEPTH_OFFSET, help="Neutral by default; use only if pre-place retention is true but final placement fails.")
    tuning_group.add_argument("--release-timing-dwell-steps", type=int, default=DEFAULT_RELEASE_TIMING_DWELL_STEPS, help="Neutral by default; use only for suspected release timing failures.")
    parser.add_argument("--scene-only", action="store_true", help="Build and log the randomized scene, then exit before robot manipulation.")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--skip-gripper-close", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    if args.init_steps < 1 or args.phase_steps < 1 or args.pause_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --phase-steps, --pause-steps, and --settle-steps must be positive")
    if args.target_index is not None and args.target_index < 0:
        raise RuntimeError("--target-index must be non-negative")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be positive")
    if args.pre_grasp_clearance <= 0.0 or args.safe_drop_height <= 0.0:
        raise RuntimeError("--pre-grasp-clearance and --safe-drop-height must be positive")
    if args.stable_jitter <= 0.0 or args.min_lift_delta <= 0.0 or args.min_transport_distance <= 0.0:
        raise RuntimeError("--stable-jitter, --min-lift-delta, and --min-transport-distance must be positive")
    if args.joint_tolerance <= 0.0 or args.max_local_joint_adjustment < 0.0:
        raise RuntimeError("--joint-tolerance must be positive and --max-local-joint-adjustment must be non-negative")
    if args.min_ee_table_clearance < 0.0:
        raise RuntimeError("--min-ee-table-clearance must be non-negative")
    if args.ik_steps < 1 or args.ik_settle_steps < 1 or args.ik_hold_steps < 1:
        raise RuntimeError("--ik-steps, --ik-settle-steps, and --ik-hold-steps must be positive")
    if args.smooth_ik_hold_steps < 1 or args.smooth_non_contact_pause_steps < 0:
        raise RuntimeError("--smooth-ik-hold-steps must be positive and --smooth-non-contact-pause-steps must be non-negative")
    if args.continuous_soft_tolerance <= 0.0 or args.continuous_blend_radius < 0.0:
        raise RuntimeError("--continuous-soft-tolerance must be positive and --continuous-blend-radius must be non-negative")
    if args.continuous_contact_dwell_steps < 0:
        raise RuntimeError("--continuous-contact-dwell-steps must be non-negative")
    if args.carry_stabilization_steps < 0 or args.release_timing_dwell_steps < 0:
        raise RuntimeError("--carry-stabilization-steps and --release-timing-dwell-steps must be non-negative")
    if args.micro_stop_speed_threshold < 0.0:
        raise RuntimeError("--micro-stop-speed-threshold must be non-negative")
    if args.ik_position_eps <= 0.0 or args.ik_damping <= 0.0 or args.ik_max_step <= 0.0 or args.ik_stop_tolerance <= 0.0:
        raise RuntimeError("--ik-position-eps, --ik-damping, --ik-max-step, and --ik-stop-tolerance must be positive")
    if args.ik_posture_gain < 0.0:
        raise RuntimeError("--ik-posture-gain must be non-negative")
    if args.gripper_hold_effort < 0.0:
        raise RuntimeError("--gripper-hold-effort must be non-negative")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")
    front_workspace_x = _parse_bounds(args.front_workspace_x, "--front-workspace-x")
    front_workspace_y = _parse_bounds(args.front_workspace_y, "--front-workspace-y")
    front_workspace_z = _parse_bounds(args.front_workspace_z, "--front-workspace-z")

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    timestamp_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_was_provided = args.seed is not None
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)
    cli_args = vars(args).copy()
    sys.argv = [sys.argv[0]]

    paths: dict[str, Path] = {}
    payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp_utc": timestamp_utc,
            "timestamp_compact": timestamp_compact,
            "script_name": SCRIPT_NAME,
            "cli_args": cli_args,
            "seed_provided": seed_was_provided,
            "actual_seed": args.seed,
            "random_scene_active": True,
            "active_tuning_knobs": _tuning_knob_summary(args),
        },
        "scene": {},
        "target": {},
        "bin": {},
        "robot": {},
        "task_space_targets": {},
        "motion_policy": {
            "official_startup_pose": {},
            "object_reach_policy": "target_conditioned_finite_difference_dls_position_control",
            "ik_steps": args.ik_steps,
            "ik_settle_steps": args.ik_settle_steps,
            "ik_position_eps": args.ik_position_eps,
            "ik_damping": args.ik_damping,
            "ik_max_step": args.ik_max_step,
            "ik_posture_gain": args.ik_posture_gain,
            "ik_stop_tolerance": args.ik_stop_tolerance,
            "ik_hold_steps": args.ik_hold_steps,
            "gripper_policy": "official_same_sign_position_targets",
            "gripper_effort_policy": "dynamic_control_set_dof_effort_sustained_during_close_validation_lift",
            "gripper_hold_effort": args.gripper_hold_effort,
            "official_gripper_open_width": OFFICIAL_GRIPPER_OPEN_WIDTH,
            "official_gripper_close_width": OFFICIAL_GRIPPER_CLOSE_WIDTH,
            "descend_clearance": args.descend_clearance,
            "grasp_depth_offset": args.grasp_depth_offset,
            "tuning_knobs": _tuning_knob_summary(args),
            "smooth_motion": bool(args.smooth_motion),
            "smooth_non_contact_pause_steps": args.smooth_non_contact_pause_steps,
            "smooth_ik_hold_steps": args.smooth_ik_hold_steps,
            "continuous_cycle_policy": "queued_soft_and_hard_waypoints_with_event_markers",
            "continuous_soft_tolerance": args.continuous_soft_tolerance,
            "continuous_blend_radius": args.continuous_blend_radius,
            "continuous_contact_dwell_steps": args.continuous_contact_dwell_steps,
            "micro_stop_speed_threshold": args.micro_stop_speed_threshold,
        },
        "front_workspace": {
            "x_limits": list(front_workspace_x),
            "y_limits": list(front_workspace_y),
            "z_limits": list(front_workspace_z),
            "min_ee_table_clearance": args.min_ee_table_clearance,
            "policy": "front_tabletop_only_no_under_table_no_side_origin",
        },
        "phase_order": [
            "apply_official_startup_pose",
            "open_gripper_initial",
            "continuous_pregrasp",
            "continuous_grasp_depth",
            "continuous_lift_clearance",
            "continuous_prebin",
            "continuous_place_depth",
            "continuous_retreat",
            "settle",
        ],
        "phase_log": [],
        "object_trace": {},
        "multi_object": {
            "enabled": args.target_index is None,
            "attempts": [],
            "continuation_rule": "continue_after_manipulation_failure_unless_scene_or_robot_safety_fails",
        },
        "result_flags": {
            "object_lifted": False,
            "object_retained_after_lift": False,
            "object_transported": False,
            "final_inside_bin": False,
            "object_stable": False,
        },
        "final_status": "fail",
        "failure_reason": "runtime_error",
    }
    sim_app = None
    timeline = None
    log_paths: list[str] = []
    tuning_knobs = _tuning_knob_summary(args)
    print(
        "active_tuning_knob="
        f"{tuning_knobs['active_knob_family']} "
        f"targeted_failure_pattern={tuning_knobs['targeted_failure_pattern']} "
        f"grasp_effective_clearance={tuning_knobs['grasp_depth']['effective_clearance']}"
    )

    try:
        paths = _validate_environment()
        baseline_root = _as_path(args.baseline_root, paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
        asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
        config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
        box_usd = asset_root / DEFAULT_BOX_RELATIVE
        payload["run_metadata"].update(
            {
                "repo_path": str(paths["HRC_REPO"]),
                "yaml_path": str(config_path),
                "root_path_override": str(asset_root),
                "gui_enabled": bool(args.no_headless or args.gui),
            }
        )

        if config_path.name != "Part_Sorting.yaml":
            _fail("scene_build_failed", f"Wrong Task 1 config; expected Part_Sorting.yaml, got {config_path}")
        if not config_path.exists():
            _fail("scene_build_failed", f"Task 1 config missing: {config_path}")
        if not asset_root.exists():
            _fail("scene_build_failed", f"Asset root missing: {asset_root}")
        if not box_usd.exists():
            _fail("scene_build_failed", f"Diagnostic bin visual USD missing: {box_usd}")

        random.seed(args.seed)
        np.random.seed(args.seed)
        SimulationApp = _load_simulation_app()
        sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
        counter = {"step": 0}

        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        original_root_path = cfg.get("root_path")
        cfg["root_path"] = str(asset_root)
        apply_scatter_config(cfg)

        import omni.replicator.core as rep  # type: ignore

        if hasattr(rep, "set_global_seed"):
            rep.set_global_seed(args.seed)

        stage = _create_minimal_scene()
        from pxr import UsdGeom  # type: ignore

        UsdGeom.Xform.Define(stage, "/Root")
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        scene.build_table()
        scene.build_parts()

        bin_position = [float(value) for value in cfg["box"]["box_position"][0]]
        bin_scale = [float(value) for value in cfg["box"]["box_scale"][0]]
        _add_reference(stage, "/World/DiagnosticBinVisual", box_usd)
        _set_xform(stage, "/World/DiagnosticBinVisual", bin_position, scale=bin_scale)
        _run_updates(sim_app, 5, counter)
        bin_visual_bbox = _bbox(stage, "/World/DiagnosticBinVisual")
        removed_bin_visual_physics = _disable_physics_under(stage, "/World/DiagnosticBinVisual")
        bin_collider = _add_static_bin_colliders(stage, bin_visual_bbox)
        bin_bbox = _bbox(stage, "/World/DiagnosticBinVisual")

        robot_cfg = cfg.get("robot", {})
        configured_robot_position = [float(value) for value in robot_cfg.get("robot_position", [0.0, 0.0, 0.0])]
        configured_robot_rotation = [float(value) for value in robot_cfg.get("robot_rotation", [0.0, 0.0, 0.0])]
        robot_usd = asset_root / str(robot_cfg.get("robot_usd", ""))
        if not robot_usd.exists():
            _fail("scene_build_failed", f"Official robot USD from Part_Sorting.yaml is missing: {robot_usd}")
        scene.build_robot()
        actual_robot_container_path = getattr(scene, "robot_prim_path", None)

        rep.orchestrator.step()
        _run_updates(sim_app, args.init_steps, counter)

        table_path = "/Replicator/Ref_Xform"
        part_paths = list(getattr(scene, "parts_prim_paths", []))
        if not part_paths:
            _fail("no_target_parts_found", "SceneBuilder did not expose any Task 1 part prim paths")
        attempt_indices = list(range(len(part_paths))) if args.target_index is None else [args.target_index]
        if any(index >= len(part_paths) for index in attempt_indices):
            _fail("target_selection_failed", f"--target-index {args.target_index} out of range for {len(part_paths)} parts")

        target_path = part_paths[attempt_indices[0]]
        target_prim = stage.GetPrimAtPath(target_path)
        target_refs = _reference_paths(target_prim) if target_prim and target_prim.IsValid() else []
        category_from_refs = _category_from_reference(target_refs)
        num_parts_per_class = int(cfg["part"].get("num_parts", 2))
        category_from_order = "part_a" if attempt_indices[0] < num_parts_per_class else "part_b"
        category_for_log = category_from_refs if category_from_refs != "unknown" else category_from_order
        category_inference_method = "reference_path" if category_from_refs != "unknown" else "scene_builder_creation_order"
        initial_state = _bbox_state(stage, target_path)
        initial_center = _center_from_bbox(initial_state["bbox"])
        table_bbox = _bbox(stage, table_path)
        table_top_z = float(table_bbox["max"][2])
        pre_grasp_pose = {
            "position": [
                initial_state["bbox"]["center"][0],
                initial_state["bbox"]["center"][1],
                initial_state["bbox"]["max"][2] + args.pre_grasp_clearance,
            ],
            "orientation": "fixed_downward",
            "orientation_search": False,
        }
        target_workspace_checks = _front_workspace_checks(
            initial_center,
            front_workspace_x,
            front_workspace_y,
            front_workspace_z,
            table_top_z,
        )
        pre_grasp_workspace_checks = _front_workspace_checks(
            np.array(pre_grasp_pose["position"], dtype=float),
            front_workspace_x,
            front_workspace_y,
            front_workspace_z,
            table_top_z,
        )
        pre_grasp_geometry = _pregrasp_geometry_summary(
            target_index=attempt_indices[0],
            target_path=target_path,
            object_center=initial_center,
            bbox_top_z=float(initial_state["bbox"]["max"][2]),
            pre_grasp_position=np.array(pre_grasp_pose["position"], dtype=float),
            robot_base_position=configured_robot_position,
            table_top_z=table_top_z,
        )
        _print_pregrasp_geometry(pre_grasp_geometry)
        marker_paths = [
            _debug_marker(stage, "/World/DebugTask1Target", initial_state["bbox"]["center"], 0.025, (1.0, 0.2, 0.1)),
            _debug_marker(stage, "/World/DebugTask1PreGraspFront", pre_grasp_pose["position"], 0.025, (0.2, 0.6, 1.0)),
            _debug_marker(stage, "/World/DebugTask1BinCenter", bin_bbox["center"], 0.03, (0.2, 1.0, 0.2)),
        ]
        descend_pose = {
            "position": [
                initial_state["bbox"]["center"][0],
                initial_state["bbox"]["center"][1],
                max(
                    initial_state["bbox"]["max"][2] + args.descend_clearance + args.grasp_depth_offset,
                    table_top_z + args.min_ee_table_clearance,
                ),
            ],
            "orientation": "fixed_downward",
            "orientation_search": False,
        }
        bin_drop_pose = {
            "position": [
                bin_bbox["center"][0],
                bin_bbox["center"][1],
                bin_bbox["max"][2] + args.safe_drop_height,
            ],
            "orientation": "fixed_downward",
            "orientation_search": False,
        }

        payload["scene"] = {
            "baseline_root": str(baseline_root),
            "config_path": str(config_path),
            "original_config_root_path": original_root_path,
            "overridden_root_path": str(asset_root),
            "scene_builder_methods": ["build_table", "build_parts"],
            "official_box_pipeline_used_for_destination_physics": False,
            "spawned_part_prim_list": part_paths,
            "table_prim": table_path,
            "table": {
                "configured_usd": cfg["table"]["table_usd"],
                "bbox": table_bbox,
                "physics": _physics_summary(stage, table_path),
            },
            "debug_marker_paths": marker_paths,
        }
        payload["bin"] = {
            "destination_bin_visual_prim": "/World/DiagnosticBinVisual",
            "destination_bin_collider_prims": bin_collider["collider_paths"],
            "official_visual_usd": str(box_usd),
            "removed_visual_physics": removed_bin_visual_physics,
            "configured_position": bin_position,
            "configured_scale": bin_scale,
            "bounds": bin_bbox,
            "floor_top_z": bin_collider["floor_top_z"],
            "wall_thickness": bin_collider["wall_thickness"],
        }
        payload["target"] = {
            "prim_path": target_path,
            "target_index": attempt_indices[0],
            "referenced_usd_paths": target_refs,
            "category_from_reference": category_from_refs,
            "category_from_scene_builder_order": category_from_order,
            "inferred_category": category_for_log,
            "category_inference_method": category_inference_method,
            "initial_pose": initial_state,
        }
        payload["task_space_targets"] = {
            "pre_grasp": pre_grasp_pose,
            "pre_grasp_geometry": pre_grasp_geometry,
            "descend": descend_pose,
            "bin_drop": bin_drop_pose,
            "continuous_cycle": "per-object plan logged inside multi_object.attempts[].continuous_motion_plan",
        }
        payload["multi_object"]["attempt_order"] = attempt_indices
        payload["scene"]["initial_part_poses"] = {
            path: _bbox_state(stage, path)
            for path in part_paths
        }
        payload["front_workspace"]["target_workspace_checks"] = target_workspace_checks
        payload["front_workspace"]["pre_grasp_workspace_checks"] = pre_grasp_workspace_checks
        payload["object_trace"]["initial"] = initial_state
        if args.scene_only:
            payload["final_status"] = "pass"
            payload["failure_reason"] = None
            print(f"status=pass scene_only=true seed={args.seed} part_paths={part_paths}")
            return 0
        if not target_workspace_checks["front_workspace_ok"] or not pre_grasp_workspace_checks["front_workspace_ok"]:
            _fail(
                "target_outside_front_workspace",
                "Selected target or explicit pre-grasp pose is outside the conservative front tabletop workspace",
            )

        articulation_roots: list[str] = []
        joint_names: list[str] = []
        for _ in range(args.init_steps):
            sim_app.update()
            counter["step"] += 1
            articulation_roots = _find_articulation_roots(stage, args.prim_path)
            joint_names = _find_joint_names(stage, args.prim_path)
            if articulation_roots and joint_names:
                break
        if not articulation_roots:
            _fail("scene_build_failed", "Walker S2 loaded, but no articulation root was detected")

        articulation_path = articulation_roots[0]
        timeline = _start_timeline()
        _run_updates(sim_app, args.pause_steps, counter)

        dc, articulation = _acquire_articulation(articulation_path)
        dof_observation = _read_dof_observation(dc, articulation)
        arm_dofs = _select_right_arm_dofs(dc, articulation, args.max_arm_dofs)
        gripper_dofs = _select_right_gripper_dofs(dc, articulation)
        official_startup_joint_map = _load_official_startup_joint_map(
            baseline_root,
            args.prim_path,
            asset_root / "s2.urdf",
        )
        payload["motion_policy"]["official_startup_pose"] = official_startup_joint_map
        startup_dofs, missing_official_startup_optional_dofs = _select_dofs_by_target_names(
            dc,
            articulation,
            official_startup_joint_map,
            OFFICIAL_STARTUP_ARM_JOINT_NAMES,
        )
        end_effector_body, end_effector_name, end_effector_path = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
        )
        official_startup_targets = _targets_from_map(startup_dofs, official_startup_joint_map)

        payload["robot"] = {
            "usd_path": str(robot_usd),
            "official_robot_container_prim_path": actual_robot_container_path,
            "prim_path": args.prim_path,
            "configured_position_from_yaml": configured_robot_position,
            "configured_rotation_xyz_deg_from_yaml": configured_robot_rotation,
            "config_robot_pose_applied_by": "official_SceneBuilder.build_robot_and__apply_robot_pose",
            "position_applied": configured_robot_position,
            "rotation_xyz_deg_applied": configured_robot_rotation,
            "articulation_path": articulation_path,
            "joint_count": len(joint_names),
            "right_arm_dof_names": [name for _, _, name in arm_dofs],
            "right_gripper_dof_indices": [index for index, _, _ in gripper_dofs],
            "right_gripper_dof_names": [name for _, _, name in gripper_dofs],
            "right_gripper_hold_effort": args.gripper_hold_effort,
            "official_startup_dof_names": [name for _, _, name in startup_dofs],
            "missing_optional_official_startup_dofs": missing_official_startup_optional_dofs,
            "official_startup_source": "lerobot.common.robot_devices.robots.isaac_sim_robot_interface.IsaacSimRobotInterface._joint_value_map",
            "end_effector_name": end_effector_name,
            "end_effector_path": end_effector_path,
            "dof_observation_sample": dof_observation[:12],
        }

        phase_log: list[dict[str, Any]] = payload["phase_log"]
        startup_start_step = counter["step"]
        _set_joint_positions_and_targets(dc, startup_dofs, official_startup_targets)
        _run_updates(sim_app, args.pause_steps, counter)
        observed_startup = _current_positions(dc, startup_dofs)
        ee_startup = _body_pose_position(dc, end_effector_body)
        startup_max_error = max(abs(float(obs - target)) for obs, target in zip(observed_startup, official_startup_targets))
        startup_ok = bool(startup_max_error <= args.joint_tolerance)
        _append_phase(
            phase_log,
            phase="apply_official_startup_pose",
            start_step=startup_start_step,
            end_step=counter["step"],
            commanded_targets=_named_positions(startup_dofs, official_startup_targets),
            ee_position=ee_startup,
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=startup_ok,
            details={
                "official_startup_joint_values": official_startup_joint_map,
                "missing_optional_official_startup_dofs": missing_official_startup_optional_dofs,
                "observed_joint_values": _named_positions(startup_dofs, observed_startup),
                "max_joint_error": startup_max_error,
                "joint_tolerance": args.joint_tolerance,
            },
        )
        print(f"phase=apply_official_startup_pose condition_met={startup_ok} ee={ee_startup.tolist()}")
        if not startup_ok:
            _fail("official_startup_pose_failed", "Official startup joint pose was not reached within tolerance")

        open_ok = _command_gripper_phase(
            "open_gripper_initial",
            dc,
            gripper_dofs,
            end_effector_body,
            [OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs),
            sim_app,
            _smooth_pause_steps(args, "open_gripper_initial"),
            counter,
            phase_log,
            effort_value=0.0,
        )
        if not open_ok:
            _fail("gripper_command_failed", "Right gripper open command failed before approach")

        attempt_results: list[dict[str, Any]] = []
        for attempt_number, target_attempt_index in enumerate(attempt_indices, start=1):
            target_attempt_path = part_paths[target_attempt_index]
            print(
                f"multi_object_attempt={attempt_number}/{len(attempt_indices)} "
                f"target_index={target_attempt_index} target_prim={target_attempt_path}"
            )
            result = _attempt_pick_place_target(
                attempt_number=attempt_number,
                target_index=target_attempt_index,
                target_path=target_attempt_path,
                stage=stage,
                cfg=cfg,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                gripper_dofs=gripper_dofs,
                end_effector_body=end_effector_body,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                front_workspace_x=front_workspace_x,
                front_workspace_y=front_workspace_y,
                front_workspace_z=front_workspace_z,
                table_top_z=table_top_z,
                robot_base_position=configured_robot_position,
                bin_bbox=bin_bbox,
                bin_collider=bin_collider,
                marker_paths=marker_paths,
            )
            attempt_results.append(result)
            payload["multi_object"]["attempts"] = attempt_results
            print(
                f"multi_object_result target_index={target_attempt_index} "
                f"success={result['success']} first_failing_phase={result.get('first_failing_phase')}"
            )
            if result.get("hard_stop"):
                payload["multi_object"]["stop_condition"] = "hard_stop_after_failed_attempt"
                break

        success_count = sum(1 for result in attempt_results if result.get("success"))
        attempted_count = len(attempt_results)
        failed_results = [result for result in attempt_results if not result.get("success")]
        payload["multi_object"]["attempted_count"] = attempted_count
        payload["multi_object"]["success_count"] = success_count
        payload["multi_object"]["failure_count"] = len(failed_results)
        payload["multi_object"]["continuous_motion_metrics"] = {
            "total_cycle_time_steps": sum(int(result.get("cycle_time_steps", 0)) for result in attempt_results),
            "total_micro_stop_frames": sum(int(result.get("micro_stop_frames", 0)) for result in attempt_results),
            "total_micro_stop_samples": sum(int(result.get("micro_stop_samples", 0)) for result in attempt_results),
            "per_object": [
                {
                    "selected_object_id": result.get("selected_object_id") or result.get("target_prim"),
                    "target_index": result.get("target_index"),
                    "success": result.get("success"),
                    "failure_phase": result.get("failure_phase"),
                    "failure_reason": result.get("failure_reason"),
                    "diagnostic_failure_kind": result.get("diagnostic_failure_kind"),
                    "cycle_time_steps": result.get("cycle_time_steps"),
                    "micro_stop_frames": result.get("micro_stop_frames"),
                    "micro_stop_samples": result.get("micro_stop_samples"),
                    "close_event_fired": result.get("close_event_fired"),
                    "lift_height_threshold_achieved": result.get("lift_height_threshold_achieved"),
                    "object_retained_at_carry_midpoint": result.get("object_retained_at_carry_midpoint"),
                    "object_retained_at_prebin": result.get("object_retained_at_prebin"),
                    "object_retained_at_preplace": result.get("object_retained_at_preplace"),
                }
                for result in attempt_results
            ],
        }
        payload["multi_object"].setdefault(
            "stop_condition",
            "all_requested_objects_attempted" if attempted_count == len(attempt_indices) else "stopped_before_all_requested_objects",
        )
        if attempt_results:
            last_result = attempt_results[-1]
            payload["result_flags"]["object_lifted"] = any(bool(result.get("object_lifted")) for result in attempt_results)
            payload["result_flags"]["object_retained_after_lift"] = any(bool(result.get("object_retained_after_lift")) for result in attempt_results)
            payload["result_flags"]["object_transported"] = any(bool(result.get("object_transported")) for result in attempt_results)
            payload["result_flags"]["final_inside_bin"] = bool(last_result.get("final_inside_bin"))
            payload["result_flags"]["object_stable"] = bool(last_result.get("object_stable"))

        if args.target_index is not None:
            if not attempt_results or not attempt_results[0].get("success"):
                failure = attempt_results[0] if attempt_results else {"failure_reason": "unknown", "error": "No attempt result was produced"}
                _fail(str(failure.get("failure_reason") or "single_object_failed"), str(failure.get("error") or "Single-object attempt failed"))
            payload["final_status"] = "pass"
            payload["failure_reason"] = None
        else:
            all_succeeded = bool(attempt_results) and success_count == len(attempt_indices)
            payload["final_status"] = "pass" if all_succeeded else "fail"
            payload["failure_reason"] = None if all_succeeded else "multi_object_partial_failure"

        payload["scene"]["debug_marker_paths"] = marker_paths
        print(
            f"status={payload['final_status']} attempted={attempted_count} "
            f"succeeded={success_count} failure_reason={payload['failure_reason']}"
        )

        if (args.no_headless or args.gui) and args.hold_open:
            _hold_gui_open(sim_app)
        return 0 if payload["final_status"] == "pass" else 1

    except RunFailure as exc:
        payload["final_status"] = "fail"
        payload["failure_reason"] = exc.reason
        payload["error"] = str(exc)
        print(f"status=fail failure_reason={exc.reason} error={exc}", file=sys.stderr)
    except Exception as exc:
        payload["final_status"] = "fail"
        payload["failure_reason"] = "runtime_error"
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        print(f"status=fail failure_reason=runtime_error error={exc}", file=sys.stderr)
    finally:
        if timeline is not None:
            try:
                timeline.stop()
            except Exception:
                pass
        if paths.get("LOG_ROOT") is not None:
            try:
                log_paths = _write_logs(paths["LOG_ROOT"], payload, args.log_suffix)
                payload["log_paths"] = log_paths
                print(f"log_paths={log_paths}")
            except Exception as log_exc:
                print(f"failed_to_write_log={log_exc}", file=sys.stderr)
        if sim_app is not None:
            sim_app.close()

    return 0 if payload["final_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
