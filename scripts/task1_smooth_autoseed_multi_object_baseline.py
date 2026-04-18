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
    DEFAULT_IK_STEPS,
    DEFAULT_POSITION_EPS,
    DEFAULT_POSTURE_GAIN,
    DEFAULT_SETTLE_STEPS as DEFAULT_IK_SETTLE_STEPS,
    DEFAULT_STOP_TOLERANCE,
    _body_pose_position,
    _create_debug_marker,
    _identify_end_effector_body,
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
DEFAULT_PAUSE_STEPS = 0
DEFAULT_SETTLE_STEPS = 240
DEFAULT_STARTUP_TARGET_TRACK_STEPS = 5
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
DEFAULT_PRE_GRASP_EE_TOLERANCE = 0.3
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
DEFAULT_CONTINUOUS_CONTACT_DWELL_STEPS = 2
DEFAULT_CARRY_STABILIZATION_STEPS = 0
DEFAULT_PLACE_DEPTH_OFFSET = 0.0
DEFAULT_RELEASE_TIMING_DWELL_STEPS = 0
DEFAULT_MICRO_STOP_SPEED_THRESHOLD = 0.0005
DEFAULT_GRASP_CONTACT_OFFSET_X = -0.04
DEFAULT_GRASP_CONTACT_OFFSET_Y = 0.0
DEFAULT_PIVOT_TO_PINCH_DISTANCE = 0.32
DEFAULT_PIVOT_ANCHOR_HEIGHT_OFFSET = 0.03
DEFAULT_PIVOT_ANCHOR_FORWARD_OFFSET = 0.0
DEFAULT_PIVOT_ANCHOR_LATERAL_OFFSET = 0.0
DEFAULT_PIVOT_ARC_CONTACT_TOLERANCE = 0.08
DEFAULT_PIVOT_ARC_MAX_STEPS = 12
DEFAULT_PIVOT_ARC_CLOSE_DISTANCE = 0.08
DEFAULT_PIVOT_ARC_FRAME_STEP_UPDATES = 1
DEFAULT_PIVOT_ARC_STREAM_SAMPLES = 80
DEFAULT_STREAM_SAMPLES = 80
DEFAULT_STREAM_FRAME_STEP_UPDATES = 1
OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"
LEFT_ARM_TOKENS = ("l_shoulder", "l_elbow", "l_wrist", "left_shoulder", "left_elbow", "left_wrist")
RIGHT_ARM_TOKENS = ("r_shoulder", "r_elbow", "r_wrist", "right_shoulder", "right_elbow", "right_wrist")
LEFT_GRIPPER_TOKENS = ("l_finger", "left_finger", "l_thumb", "left_thumb", "l_gripper", "left_gripper")
RIGHT_GRIPPER_TOKENS = ("r_finger", "right_finger", "r_thumb", "right_thumb", "r_gripper", "right_gripper")
LEFT_END_EFFECTOR_TOKENS = ("l_hand", "left_hand", "l_palm", "left_palm", "l_wrist", "left_wrist")
END_EFFECTOR_EXCLUDE_TOKENS = ("finger", "thumb", "sensor", "camera")

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
    control_body: Any | None = None
    control_body_name: str | None = None
    control_body_path: str | None = None
    control_dofs: list[tuple[int, Any, str]] | None = None
    locked_dofs: list[tuple[int, Any, str]] | None = None
    locked_targets: dict[str, float] | None = None
    pinch_metric_pivot_body: Any | None = None


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _center_from_bbox(box: dict[str, list[float]]) -> np.ndarray:
    return np.array(box["center"], dtype=float)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _quintic_blend(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return (6.0 * t**5) - (15.0 * t**4) + (10.0 * t**3)


def _yaw_rad_from_rotation_xyz_deg(rotation_xyz_deg: list[float] | tuple[float, ...] | None) -> float:
    if not rotation_xyz_deg or len(rotation_xyz_deg) < 3:
        return 0.0
    return math.radians(float(rotation_xyz_deg[2]))


def _robot_base_xy_axes(robot_base_yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([math.cos(robot_base_yaw_rad), math.sin(robot_base_yaw_rad)], dtype=float)
    left = np.array([-math.sin(robot_base_yaw_rad), math.cos(robot_base_yaw_rad)], dtype=float)
    return forward, left


def _arm_mirror_sign(active_arm: str) -> float:
    return 1.0 if active_arm == "left" else -1.0


def _base_frame_motion_components(
    *,
    cartesian_delta: np.ndarray,
    active_arm: str,
    robot_base_yaw_rad: float,
    backward_guard_m: float = 0.0,
) -> dict[str, float]:
    forward_axis, left_axis = _robot_base_xy_axes(robot_base_yaw_rad)
    delta = np.array(cartesian_delta, dtype=float)
    raw_forward = float(np.dot(delta[:2], forward_axis))
    raw_lateral = float(np.dot(delta[:2], left_axis))
    side_sign = _arm_mirror_sign(active_arm)
    outward_lateral = raw_lateral * side_sign

    # Normal task targets should stay in front of the torso. If a local target
    # requests rearward motion, do not let the heuristic turn that into a large
    # behind-the-back shoulder/wrist command.
    guarded_forward = raw_forward
    if raw_forward < -abs(float(backward_guard_m)):
        guarded_forward = 0.0

    # Crossing the torso is sometimes necessary near the bin, but it should not
    # dominate the mirrored arm joints. Attenuate inward lateral requests.
    guarded_lateral = raw_lateral
    if outward_lateral < 0.0:
        guarded_lateral *= 0.35
        outward_lateral = guarded_lateral * side_sign

    return {
        "signed_forward": guarded_forward,
        "raw_signed_forward": raw_forward,
        "signed_lateral": guarded_lateral,
        "raw_signed_lateral": raw_lateral,
        "arm_outward_lateral": outward_lateral,
        "vertical": float(delta[2]),
        "backward_component_clamped": float(min(raw_forward, 0.0)) if guarded_forward == 0.0 else 0.0,
    }


def _select_arm_dofs_for_side(dc: Any, articulation: Any, active_arm: str, max_dofs: int) -> list[tuple[int, Any, str]]:
    tokens = RIGHT_ARM_TOKENS if active_arm == "right" else LEFT_ARM_TOKENS
    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        if any(token in name.lower() for token in tokens):
            selected.append((index, dof, name))
        if len(selected) >= max_dofs:
            break

    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
        raise RuntimeError(f"No {active_arm}-arm DOFs matched tokens={tokens}; available_dof_names={all_names}")
    return selected


def _select_gripper_dofs_for_side(dc: Any, articulation: Any, active_arm: str) -> list[tuple[int, Any, str]]:
    if active_arm == "right":
        return _select_right_gripper_dofs(dc, articulation)

    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        if any(token in name.lower() for token in LEFT_GRIPPER_TOKENS):
            selected.append((index, dof, name))

    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
        raise RuntimeError(f"No left gripper/finger DOFs matched tokens={LEFT_GRIPPER_TOKENS}; available_dof_names={all_names}")
    return selected


def _identify_end_effector_body_for_side(
    dc: Any,
    articulation: Any,
    active_arm: str,
    requested_body: str | None,
) -> tuple[Any, str, str]:
    if active_arm == "right":
        return _identify_end_effector_body(dc, articulation, requested_body)

    bodies: list[tuple[int, Any, str, str]] = []
    for index in range(dc.get_articulation_body_count(articulation)):
        body = dc.get_articulation_body(articulation, index)
        bodies.append((index, body, str(dc.get_rigid_body_name(body)), str(dc.get_rigid_body_path(body))))

    if requested_body:
        requested_lower = requested_body.lower()
        for _, body, name, path in bodies:
            if requested_body == name or requested_body == path or requested_lower in path.lower():
                return body, name, path

    candidates: list[tuple[int, Any, str, str]] = []
    for index, body, name, path in bodies:
        lower = f"{name} {path}".lower()
        if any(token in lower for token in END_EFFECTOR_EXCLUDE_TOKENS):
            continue
        if any(token in lower for token in LEFT_END_EFFECTOR_TOKENS):
            candidates.append((index, body, name, path))

    if not candidates:
        raise RuntimeError(
            "Could not identify a left-arm end-effector body. "
            f"Tokens={LEFT_END_EFFECTOR_TOKENS}; available body paths={[path for _, _, _, path in bodies]}"
        )

    _, body, name, path = candidates[-1]
    return body, name, path


def _find_body_for_side(dc: Any, articulation: Any, active_arm: str, link_token: str) -> tuple[Any, str, str]:
    side_prefix = "R_" if active_arm == "right" else "L_"
    expected = f"{side_prefix}{link_token}"
    bodies: list[tuple[int, Any, str, str]] = []
    for index in range(dc.get_articulation_body_count(articulation)):
        body = dc.get_articulation_body(articulation, index)
        bodies.append((index, body, str(dc.get_rigid_body_name(body)), str(dc.get_rigid_body_path(body))))

    expected_lower = expected.lower()
    for _, body, name, path in bodies:
        if name.lower() == expected_lower or path.lower().endswith(f"/{expected_lower}"):
            return body, name, path
    for _, body, name, path in bodies:
        if expected_lower in f"{name} {path}".lower():
            return body, name, path

    raise RuntimeError(
        f"Could not find {active_arm} body for token={link_token}; expected={expected}; "
        f"available_body_paths={[path for _, _, _, path in bodies]}"
    )


def _select_dofs_by_names(
    selected_dofs: list[tuple[int, Any, str]],
    names: set[str],
) -> list[tuple[int, Any, str]]:
    return [item for item in selected_dofs if item[2] in names]


def _capture_locked_joint_targets(dc: Any, dofs: list[tuple[int, Any, str]]) -> dict[str, float]:
    return {name: float(dc.get_dof_position(dof)) for _, dof, name in dofs}


def _apply_locked_joint_targets(
    dc: Any,
    locked_dofs: list[tuple[int, Any, str]],
    locked_targets: dict[str, float],
) -> None:
    targets = [float(locked_targets[name]) for _, _, name in locked_dofs]
    _send_position_targets(dc, locked_dofs, targets)


def _compute_pivot_drift(reference_pivot_position: np.ndarray, current_pivot_position: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(current_pivot_position, dtype=float) - np.array(reference_pivot_position, dtype=float)))


def _summarize_pivot_drift(drifts: list[float]) -> dict[str, Any]:
    if not drifts:
        return {
            "pivot_drift_norm_max": 0.0,
            "pivot_drift_norm_mean": 0.0,
            "pivot_drift_norm_final": 0.0,
            "pivot_drift_sample_count": 0,
        }
    drift_array = np.array(drifts, dtype=float)
    return {
        "pivot_drift_norm_max": float(np.max(drift_array)),
        "pivot_drift_norm_mean": float(np.mean(drift_array)),
        "pivot_drift_norm_final": float(drift_array[-1]),
        "pivot_drift_sample_count": int(len(drifts)),
    }


def _summarize_scalar_samples(samples: list[float], prefix: str) -> dict[str, Any]:
    if not samples:
        return {
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_final": 0.0,
            f"{prefix}_sample_count": 0,
        }
    sample_array = np.array(samples, dtype=float)
    return {
        f"{prefix}_max": float(np.max(sample_array)),
        f"{prefix}_mean": float(np.mean(sample_array)),
        f"{prefix}_final": float(sample_array[-1]),
        f"{prefix}_sample_count": int(len(samples)),
    }


def _locked_joint_error(
    dc: Any,
    locked_dofs: list[tuple[int, Any, str]],
    locked_targets: dict[str, float],
) -> dict[str, Any]:
    by_joint = {
        name: abs(float(dc.get_dof_position(dof)) - float(locked_targets[name]))
        for _, dof, name in locked_dofs
    }
    values = list(by_joint.values())
    return {
        "by_joint": by_joint,
        "max": max(values) if values else 0.0,
        "mean": float(np.mean(np.array(values, dtype=float))) if values else 0.0,
    }


def _arc_micro_stop_estimate(
    trace: list[dict[str, Any]],
    frame_span_per_trace_sample: int,
    threshold: float,
) -> dict[str, Any]:
    previous: np.ndarray | None = None
    micro_stop_frames = 0
    micro_stop_samples = 0
    for sample in trace:
        actual = np.array(sample["actual"], dtype=float)
        if previous is not None:
            speed = _distance(previous, actual) / float(max(frame_span_per_trace_sample, 1))
            if speed <= threshold:
                micro_stop_samples += 1
                micro_stop_frames += max(frame_span_per_trace_sample, 1)
        previous = actual
    return {
        "micro_stop_frames": int(micro_stop_frames),
        "micro_stop_samples": int(micro_stop_samples),
    }


def _estimate_unlocked_stream_endpoint(
    *,
    dc: Any,
    articulation: Any,
    selected_dofs: list[tuple[int, Any, str]],
    body: Any,
    start_positions: np.ndarray,
    start_body_position: np.ndarray,
    target_position: np.ndarray,
    sim_app: Any,
    counter: dict[str, int],
    eps: float,
    damping: float,
    max_step: float,
    sample_count: int,
) -> np.ndarray:
    jacobian = np.zeros((3, len(selected_dofs)), dtype=float)
    for column in range(len(selected_dofs)):
        trial = start_positions.copy()
        trial[column] += eps
        _send_position_targets(dc, selected_dofs, [float(value) for value in trial])
        sim_app.update()
        counter["step"] += 1
        dc.wake_up_articulation(articulation)
        moved_position = _body_pose_position(dc, body)
        jacobian[:, column] = (moved_position - start_body_position) / eps

    _send_position_targets(dc, selected_dofs, [float(value) for value in start_positions])
    sim_app.update()
    counter["step"] += 1
    dc.wake_up_articulation(articulation)
    error_vector = target_position - start_body_position
    lhs = jacobian @ jacobian.T + (damping**2) * np.eye(3)
    delta = jacobian.T @ np.linalg.solve(lhs, error_vector)
    delta_norm = float(np.linalg.norm(delta))
    max_total = min(max_step * float(max(int(sample_count), 1)), 0.25)
    if delta_norm > max_total:
        delta *= max_total / delta_norm
    return start_positions + delta


def _heuristic_lower_chain_arc_endpoint(
    *,
    lower_dofs: list[tuple[int, Any, str]],
    start_positions: np.ndarray,
    start_body_position: np.ndarray,
    target_position: np.ndarray,
    active_arm: str,
    robot_base_yaw_rad: float,
    max_step: float,
    sample_count: int,
) -> np.ndarray:
    motion = _base_frame_motion_components(
        cartesian_delta=target_position - start_body_position,
        active_arm=active_arm,
        robot_base_yaw_rad=robot_base_yaw_rad,
    )
    forward = max(float(motion["signed_forward"]), 0.0)
    lateral = float(motion["signed_lateral"])
    outward_lateral = float(motion["arm_outward_lateral"])
    vertical = float(motion["vertical"])
    mirror_sign = _arm_mirror_sign(active_arm)
    max_total = min(max_step * float(max(int(sample_count), 1)), 0.55)
    endpoint = start_positions.copy()
    for index, (_, _, name) in enumerate(lower_dofs):
        lower_name = name.lower()
        if "elbow_yaw" in lower_name:
            delta = 0.45 * mirror_sign * outward_lateral
        elif "wrist_pitch" in lower_name:
            delta = mirror_sign * (0.95 * forward + 0.55 * vertical)
        elif "wrist_roll" in lower_name:
            delta = 0.18 * lateral
        else:
            delta = 0.0
        endpoint[index] += float(np.clip(delta, -max_total, max_total))
    return endpoint


def _heuristic_upper_chain_anchor_endpoint(
    *,
    selected_dofs: list[tuple[int, Any, str]],
    start_positions: np.ndarray,
    start_body_position: np.ndarray,
    target_position: np.ndarray,
    active_arm: str,
    robot_base_yaw_rad: float,
    max_step: float,
    sample_count: int,
) -> np.ndarray:
    motion = _base_frame_motion_components(
        cartesian_delta=target_position - start_body_position,
        active_arm=active_arm,
        robot_base_yaw_rad=robot_base_yaw_rad,
    )
    forward = max(float(motion["signed_forward"]), 0.0)
    lateral = float(motion["signed_lateral"])
    outward_lateral = float(motion["arm_outward_lateral"])
    vertical = float(motion["vertical"])
    mirror_sign = _arm_mirror_sign(active_arm)
    max_total = max_step * float(max(int(sample_count), 1))
    endpoint = start_positions.copy()
    for index, (_, _, name) in enumerate(selected_dofs):
        lower_name = name.lower()
        if "shoulder_pitch" in lower_name:
            delta = mirror_sign * (0.85 * forward + 0.30 * vertical)
        elif "shoulder_roll" in lower_name:
            delta = 0.50 * outward_lateral + 0.18 * vertical
        elif "shoulder_yaw" in lower_name:
            delta = 0.36 * mirror_sign * outward_lateral
        elif "elbow_roll" in lower_name:
            delta = -0.65 * forward + 0.30 * vertical
        else:
            delta = 0.0
        endpoint[index] += float(np.clip(delta, -max_total, max_total))
    return endpoint


def _heuristic_main_cycle_endpoint_delta(
    *,
    selected_dofs: list[tuple[int, Any, str]],
    cartesian_delta: np.ndarray,
    active_arm: str,
    robot_base_yaw_rad: float,
    max_step: float,
    sample_count: int,
) -> np.ndarray:
    motion = _base_frame_motion_components(
        cartesian_delta=cartesian_delta,
        active_arm=active_arm,
        robot_base_yaw_rad=robot_base_yaw_rad,
    )
    forward = max(float(motion["signed_forward"]), 0.0)
    lateral = float(motion["signed_lateral"])
    outward_lateral = float(motion["arm_outward_lateral"])
    vertical = float(motion["vertical"])
    mirror_sign = _arm_mirror_sign(active_arm)
    max_total = min(max_step * float(max(int(sample_count), 1)), 0.35)
    delta = np.zeros(len(selected_dofs), dtype=float)
    for index, (_, _, name) in enumerate(selected_dofs):
        lower_name = name.lower()
        if "shoulder_pitch" in lower_name:
            value = mirror_sign * (0.42 * forward + 0.16 * vertical)
        elif "shoulder_roll" in lower_name:
            value = 0.24 * outward_lateral + 0.08 * vertical
        elif "shoulder_yaw" in lower_name:
            value = 0.20 * mirror_sign * outward_lateral
        elif "elbow_roll" in lower_name:
            value = -0.30 * forward + 0.12 * vertical
        elif "elbow_yaw" in lower_name:
            value = 0.14 * mirror_sign * outward_lateral
        elif "wrist_pitch" in lower_name:
            value = mirror_sign * (0.14 * forward + 0.20 * vertical)
        elif "wrist_roll" in lower_name:
            value = 0.06 * lateral
        else:
            value = 0.0
        delta[index] = float(np.clip(value, -max_total, max_total))
    return delta


def _precompute_main_cycle_joint_waypoints(
    *,
    selected_dofs: list[tuple[int, Any, str]],
    start_positions: np.ndarray,
    start_body_position: np.ndarray,
    segments: list[MotionSegment],
    active_arm: str,
    robot_base_yaw_rad: float,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    waypoints = [start_positions.copy()]
    for segment in segments:
        sample_count = int(segment.details.get("streaming_sample_count", segment.max_ik_steps))
        absolute_cartesian_delta = np.array(segment.target_position, dtype=float) - np.array(start_body_position, dtype=float)
        endpoint = start_positions + _heuristic_main_cycle_endpoint_delta(
            selected_dofs=selected_dofs,
            cartesian_delta=absolute_cartesian_delta,
            active_arm=active_arm,
            robot_base_yaw_rad=robot_base_yaw_rad,
            max_step=args.ik_max_step,
            sample_count=sample_count,
        )
        waypoints.append(endpoint.copy())
    return waypoints


def _joint_waypoint_tangents(waypoints: list[np.ndarray], cumulative_samples: list[float]) -> list[np.ndarray]:
    if len(waypoints) <= 1:
        return [np.zeros_like(waypoints[0])] if waypoints else []

    tangents: list[np.ndarray] = []
    for index, waypoint in enumerate(waypoints):
        if index == 0:
            dt = max(float(cumulative_samples[1] - cumulative_samples[0]), 1.0)
            tangents.append((waypoints[1] - waypoint) / dt)
            continue
        if index == len(waypoints) - 1:
            dt = max(float(cumulative_samples[index] - cumulative_samples[index - 1]), 1.0)
            tangents.append((waypoint - waypoints[index - 1]) / dt)
            continue

        dt_prev = max(float(cumulative_samples[index] - cumulative_samples[index - 1]), 1.0)
        dt_next = max(float(cumulative_samples[index + 1] - cumulative_samples[index]), 1.0)
        prev_slope = (waypoint - waypoints[index - 1]) / dt_prev
        next_slope = (waypoints[index + 1] - waypoint) / dt_next
        same_direction = (prev_slope * next_slope) > 0.0
        tangent = np.zeros_like(waypoint)
        tangent[same_direction] = np.sign(prev_slope[same_direction]) * np.minimum(
            np.abs(prev_slope[same_direction]),
            np.abs(next_slope[same_direction]),
        )
        tangents.append(tangent)
    return tangents


def _sample_cubic_hermite_joint_chain(
    *,
    waypoints: list[np.ndarray],
    tangents: list[np.ndarray],
    cumulative_samples: list[float],
    progress: float,
) -> tuple[int, float, np.ndarray]:
    if len(waypoints) == 1:
        return 0, 0.0, waypoints[0].copy()

    segment_index = _segment_index_for_progress(cumulative_samples, progress)
    segment_start = float(cumulative_samples[segment_index])
    segment_end = float(cumulative_samples[segment_index + 1])
    segment_span = max(segment_end - segment_start, 1.0)
    u = float(np.clip((progress - segment_start) / segment_span, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    p0 = waypoints[segment_index]
    p1 = waypoints[segment_index + 1]
    m0 = tangents[segment_index]
    m1 = tangents[segment_index + 1]
    target = (
        (2.0 * u3 - 3.0 * u2 + 1.0) * p0
        + (u3 - 2.0 * u2 + u) * segment_span * m0
        + (-2.0 * u3 + 3.0 * u2) * p1
        + (u3 - u2) * segment_span * m1
    )
    return segment_index, u, target


def _execute_streaming_body_segment(
    *,
    target_index: int,
    dc: Any,
    articulation: Any,
    sim_app: Any,
    counter: dict[str, int],
    selected_dofs: list[tuple[int, Any, str]],
    body: Any,
    body_name: str,
    body_path: str,
    target_position: np.ndarray,
    sample_count: int,
    frame_step_updates: int,
    endpoint_strategy: str,
    args: argparse.Namespace,
    phase_name: str,
    phase_tag: str,
    waypoint_type: str,
    speed_profile: str,
    gripper_dofs: list[tuple[int, Any, str]],
    phase_log: list[dict[str, Any]],
    extra_details: dict[str, Any],
    gripper_effort_value: float | None = None,
    locked_dofs: list[tuple[int, Any, str]] | None = None,
    locked_targets: dict[str, float] | None = None,
    append_phase: bool = True,
    stop_tolerance: float | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    start_positions = np.array(_current_positions(dc, selected_dofs), dtype=float)
    start_body_position = _body_pose_position(dc, body)
    samples = max(int(sample_count), 1)
    frame_updates = max(int(frame_step_updates), 1)
    if endpoint_strategy == "hold_current":
        end_positions = start_positions.copy()
    elif endpoint_strategy == "upper_anchor_heuristic":
        active_arm = str(extra_details.get("active_arm", "right"))
        robot_base_yaw_rad = float(extra_details.get("robot_base_yaw_rad", 0.0))
        end_positions = _heuristic_upper_chain_anchor_endpoint(
            selected_dofs=selected_dofs,
            start_positions=start_positions,
            start_body_position=start_body_position,
            target_position=target_position,
            active_arm=active_arm,
            robot_base_yaw_rad=robot_base_yaw_rad,
            max_step=args.ik_max_step,
            sample_count=samples,
        )
    elif endpoint_strategy in {"main_cycle_heuristic_endpoint", "global_precomputed_joint_waypoint"}:
        active_arm = str(extra_details.get("active_arm", "right"))
        robot_base_yaw_rad = float(extra_details.get("robot_base_yaw_rad", 0.0))
        end_positions = start_positions + _heuristic_main_cycle_endpoint_delta(
            selected_dofs=selected_dofs,
            cartesian_delta=target_position - start_body_position,
            active_arm=active_arm,
            robot_base_yaw_rad=robot_base_yaw_rad,
            max_step=args.ik_max_step,
            sample_count=samples,
        )
    elif endpoint_strategy == "single_dls_endpoint":
        end_positions = _estimate_unlocked_stream_endpoint(
            dc=dc,
            articulation=articulation,
            selected_dofs=selected_dofs,
            body=body,
            start_positions=start_positions,
            start_body_position=start_body_position,
            target_position=target_position,
            sim_app=sim_app,
            counter=counter,
            eps=args.ik_position_eps,
            damping=args.ik_damping,
            max_step=args.ik_max_step,
            sample_count=samples,
        )
    else:
        raise RuntimeError(f"Unknown streaming endpoint strategy: {endpoint_strategy}")

    if gripper_effort_value is not None:
        effort_before = _apply_gripper_effort(dc, gripper_dofs, gripper_effort_value)
        if not effort_before["supported"]:
            raise RuntimeError(f"Gripper effort command failed before {phase_name}: {effort_before}")
    else:
        effort_before = None

    trace: list[dict[str, Any]] = []
    for sample_index in range(samples):
        t = (sample_index + 1) / float(samples)
        alpha = _quintic_blend(t)
        target_joints = start_positions + alpha * (end_positions - start_positions)
        _send_position_targets(dc, selected_dofs, [float(value) for value in target_joints])
        for _ in range(frame_updates):
            if locked_dofs is not None and locked_targets is not None:
                _apply_locked_joint_targets(dc, locked_dofs, locked_targets)
            if gripper_effort_value is not None:
                effort_during = _apply_gripper_effort(dc, gripper_dofs, gripper_effort_value)
                if not effort_during["supported"]:
                    raise RuntimeError(f"Gripper effort command failed during {phase_name}: {effort_during}")
            sim_app.update()
            counter["step"] += 1
        dc.wake_up_articulation(articulation)
        actual_position = _body_pose_position(dc, body)
        trace.append(
            {
                "phase": phase_name,
                "sample_index": int(sample_index),
                "target_joint_positions": _named_positions(selected_dofs, target_joints),
                "target": target_position.tolist(),
                "actual": actual_position.tolist(),
                "error_norm": float(np.linalg.norm(target_position - actual_position)),
            }
        )

    actual_position = _body_pose_position(dc, body)
    position_error = float(np.linalg.norm(target_position - actual_position))
    micro_stop_estimate = _arc_micro_stop_estimate(
        trace=trace,
        frame_span_per_trace_sample=frame_updates,
        threshold=args.micro_stop_speed_threshold,
    )
    effective_stop_tolerance = float(stop_tolerance if stop_tolerance is not None else args.ik_stop_tolerance)
    condition_met = bool(position_error <= effective_stop_tolerance)
    details = {
        "target_index": target_index,
        "continuous_motion": True,
        "phase_tag": phase_tag,
        "waypoint_type": waypoint_type,
        "speed_profile": speed_profile,
        "streaming_controller": True,
        "streaming_endpoint_strategy": endpoint_strategy,
        "finite_difference_jacobian_calls_during_stream": 0 if endpoint_strategy != "single_dls_endpoint" else 1,
        "stream_sample_count": samples,
        "frame_step_updates": frame_updates,
        "interpolation_profile": "quintic_minimum_jerk",
        "control_body_name": body_name,
        "control_body_path": body_path,
        "control_dof_names": [name for _, _, name in selected_dofs],
        "target_position": target_position.tolist(),
        "position_error": position_error,
        "stop_tolerance": effective_stop_tolerance,
        "ik_step_count": 0,
        "ik_trace": trace,
        "micro_stop_frames": micro_stop_estimate["micro_stop_frames"],
        "micro_stop_samples": micro_stop_estimate["micro_stop_samples"],
        "gripper_effort_before_segment": effort_before,
        "locked_target_reapplication_active": bool(locked_dofs is not None and locked_targets is not None),
    }
    details.update(extra_details)
    if append_phase:
        _append_phase(
            phase_log,
            phase=phase_name,
            start_step=start_step,
            end_step=counter["step"],
            commanded_targets=_named_positions(selected_dofs, _current_positions(dc, selected_dofs)),
            ee_position=actual_position,
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=condition_met,
            details=details,
        )
    return {
        "actual_position": actual_position,
        "position_error": position_error,
        "trace": trace,
        "condition_met": condition_met,
        "details": details,
        "micro_stop_estimate": micro_stop_estimate,
    }


def _interpolate_arc_waypoints(
    start_pos: np.ndarray,
    mid_pos: np.ndarray,
    end_pos: np.ndarray,
    num_steps: int,
) -> list[np.ndarray]:
    count = max(int(num_steps), 1)
    start = np.array(start_pos, dtype=float)
    mid = np.array(mid_pos, dtype=float)
    end = np.array(end_pos, dtype=float)
    waypoints: list[np.ndarray] = []
    for index in range(1, count + 1):
        t = index / float(count)
        waypoint = ((1.0 - t) ** 2) * start + 2.0 * (1.0 - t) * t * mid + (t**2) * end
        waypoints.append(waypoint)
    return waypoints


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


def _pregrasp_with_pullback(
    pre_grasp_position: np.ndarray,
    robot_base_position: list[float],
    pullback_m: float,
    max_bias_m: float,
) -> tuple[np.ndarray, float, float]:
    biased = np.array(pre_grasp_position, dtype=float).copy()
    robot_base = np.array(robot_base_position, dtype=float)
    vx = float(robot_base[0] - biased[0])
    vy = float(robot_base[1] - biased[1])
    dist = math.sqrt(vx * vx + vy * vy) + 1e-9
    applied = min(max(float(pullback_m), 0.0), max(float(max_bias_m), 0.0))
    scale = applied / dist
    biased[0] += scale * vx
    biased[1] += scale * vy
    return biased, applied, dist


def _build_pregrasp_candidates(
    *,
    pregrasp_before_bias: np.ndarray,
    pregrasp_after_bias: np.ndarray,
    robot_base_position: list[float],
    active_arm: str,
    args: argparse.Namespace,
    front_workspace_z: tuple[float, float],
) -> list[dict[str, Any]]:
    candidate_specs = [
        ("center", float(np.linalg.norm(pregrasp_after_bias[:2] - pregrasp_before_bias[:2])), 0.0),
        ("pullback_2cm", 0.02, 0.0),
        ("pullback_4cm", 0.04, 0.0),
        ("center_z_plus_2cm", float(np.linalg.norm(pregrasp_after_bias[:2] - pregrasp_before_bias[:2])), 0.02),
        ("pullback_2cm_z_plus_2cm", 0.02, 0.02),
        ("pullback_4cm_z_minus_2cm", 0.04, -0.02),
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for label, pullback_m, z_offset in candidate_specs:
        candidate_position, applied, _ = _pregrasp_with_pullback(
            pregrasp_before_bias,
            robot_base_position,
            pullback_m,
            args.pregrasp_max_bias_m,
        )
        candidate_position[2] = float(np.clip(pregrasp_before_bias[2] + z_offset, *front_workspace_z))
        key = tuple(round(float(value), 6) for value in candidate_position)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_index": len(candidates),
                "label": label,
                "active_arm": active_arm,
                "pregrasp_before_bias": pregrasp_before_bias.tolist(),
                "pregrasp_after_bias": pregrasp_after_bias.tolist(),
                "position": candidate_position,
                "pullback_applied": float(applied),
                "base_to_target_xy": {
                    "dx": float(candidate_position[0] - robot_base_position[0]),
                    "dy": float(candidate_position[1] - robot_base_position[1]),
                    "distance": float(np.linalg.norm(candidate_position[:2] - np.array(robot_base_position[:2], dtype=float))),
                },
                "z_offset": float(z_offset),
                "z_clamped": bool(abs(float(candidate_position[2] - (pregrasp_before_bias[2] + z_offset))) > 1e-9),
            }
        )
        if len(candidates) >= 6:
            break
    return candidates


def _select_pregrasp_candidate(
    *,
    target_index: int,
    active_arm: str,
    fallback_triggered: bool,
    candidates: list[dict[str, Any]],
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for candidate in candidates:
        start_step = counter["step"]
        target_position = np.array(candidate["position"], dtype=float)
        stream_result = _execute_streaming_body_segment(
            target_index=target_index,
            dc=dc,
            articulation=articulation,
            sim_app=sim_app,
            counter=counter,
            selected_dofs=arm_dofs,
            body=end_effector_body,
            body_name=f"{active_arm}_candidate_end_effector",
            body_path=f"{active_arm}_candidate_end_effector",
            target_position=target_position,
            sample_count=args.stream_samples,
            frame_step_updates=args.stream_frame_step_updates,
            endpoint_strategy="single_dls_endpoint",
            args=args,
            phase_name=f"pregrasp_candidate_{active_arm}_{candidate['candidate_index']}",
            phase_tag="approach",
            waypoint_type="soft",
            speed_profile="streaming_pregrasp_candidate",
            gripper_dofs=gripper_dofs,
            phase_log=phase_log,
            extra_details={
                "pregrasp_candidate_streaming": True,
                "streaming_endpoint_strategy": "single_dls_endpoint",
                "streaming_sample_count": int(args.stream_samples),
                "interpolation_profile": "quintic_minimum_jerk",
            },
            append_phase=False,
            stop_tolerance=args.continuous_soft_tolerance,
        )
        actual_position = stream_result["actual_position"]
        pre_distance = float(stream_result["position_error"])
        trace = stream_result["trace"]
        error_vector = target_position - actual_position
        candidate_log = {
            "target_index": target_index,
            "active_arm": active_arm,
            "fallback_triggered": bool(fallback_triggered),
            "candidate_index": int(candidate["candidate_index"]),
            "candidate_label": candidate["label"],
            "pregrasp_before_bias": candidate["pregrasp_before_bias"],
            "pregrasp_after_bias": candidate["pregrasp_after_bias"],
            "target_position": target_position.tolist(),
            "actual_position": actual_position.tolist(),
            "pullback_applied": float(candidate["pullback_applied"]),
            "base_to_target_xy": candidate["base_to_target_xy"],
            "error_vector": {
                "dx": float(error_vector[0]),
                "dy": float(error_vector[1]),
                "dz": float(error_vector[2]),
            },
            "error_norm": float(pre_distance),
            "pre_grasp_ee_tolerance": args.pre_grasp_ee_tolerance,
            "z_clamped": bool(candidate["z_clamped"]),
            "ik_step_count": len(trace),
            "ik_trace": trace,
            "streaming_controller": True,
            "no_blocking_ik": True,
            "stream_sample_count": int(args.stream_samples),
            "frame_step_updates": int(args.stream_frame_step_updates),
            "interpolation_profile": "quintic_minimum_jerk",
        }
        _append_phase(
            phase_log,
            phase="pregrasp_candidate_selection",
            start_step=start_step,
            end_step=counter["step"],
            commanded_targets=_named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
            ee_position=actual_position,
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=bool(pre_distance <= args.pre_grasp_ee_tolerance),
            details=candidate_log,
        )
        print(f"pregrasp_candidate={json.dumps(candidate_log, sort_keys=True)}")
        if pre_distance <= args.pre_grasp_ee_tolerance:
            selected = dict(candidate)
            selected["actual_position"] = actual_position.tolist()
            selected["pre_distance"] = float(pre_distance)
            selected["error_vector"] = candidate_log["error_vector"]
            selected["fallback_triggered"] = bool(fallback_triggered)
            return selected
    return None


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
    locked_dofs: list[tuple[int, Any, str]] | None = None,
    locked_targets: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    last_effort_result: dict[str, Any] | None = None
    for _ in range(steps):
        if locked_dofs is not None and locked_targets is not None:
            _apply_locked_joint_targets(dc, locked_dofs, locked_targets)
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
    _send_position_targets(dc, selected_dofs, [float(target) for target in target_positions])


def _seed_joint_positions_for_initialization(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    target_positions: list[float],
) -> dict[str, Any]:
    """Seed the official startup pose once before manipulation streaming begins."""
    targets = [float(target) for target in target_positions]
    applied: dict[str, float] = {}
    errors: list[str] = []
    if not hasattr(dc, "set_dof_position"):
        errors.append("dynamic_control does not expose set_dof_position")
    else:
        for (index, dof, name), target in zip(selected_dofs, targets):
            try:
                dc.set_dof_position(dof, target)
                applied[name] = target
            except Exception as exc:  # pragma: no cover - Isaac runtime API detail.
                errors.append(f"{index}:{name}: {exc}")

    _send_position_targets(dc, selected_dofs, targets)
    return {
        "supported": not errors and len(applied) == len(selected_dofs),
        "method": "dynamic_control.set_dof_position_initialization_only_then_set_dof_position_target",
        "normal_motion_runtime_uses_set_dof_position": False,
        "dof_indices": [index for index, _, _ in selected_dofs],
        "dof_names": [name for _, _, name in selected_dofs],
        "applied_positions": applied,
        "errors": errors,
    }


def _articulation_acquire_candidates(detected_path: str, robot_prim_path: str) -> list[str]:
    candidates: list[str] = []

    def _add(path: str) -> None:
        if path and path not in candidates:
            candidates.append(path)

    _add(detected_path)
    _add(robot_prim_path)
    current = detected_path.rstrip("/")
    while "/" in current:
        current = current.rsplit("/", 1)[0]
        if not current:
            break
        _add(current)
        if current == robot_prim_path:
            break
    return candidates


def _acquire_articulation_with_fallback(
    detected_path: str,
    robot_prim_path: str,
) -> tuple[Any, Any, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for candidate_path in _articulation_acquire_candidates(detected_path, robot_prim_path):
        try:
            dc, articulation = _acquire_articulation(candidate_path)
            return dc, articulation, {
                "detected_articulation_path": detected_path,
                "acquired_articulation_path": candidate_path,
                "candidate_paths": _articulation_acquire_candidates(detected_path, robot_prim_path),
                "attempts": attempts + [{"path": candidate_path, "success": True}],
            }
        except Exception as exc:
            attempts.append({"path": candidate_path, "success": False, "error": str(exc)})
    raise RuntimeError(
        "dynamic_control could not acquire the Walker S2 articulation from any candidate path: "
        f"{json.dumps(attempts, sort_keys=True)}"
    )


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
        "experimental_pivot_arc_grasp": {
            "enabled": bool(args.experimental_pivot_arc_grasp),
            "pivot_to_pinch_distance_m": args.pivot_to_pinch_distance_m,
            "pivot_anchor_height_offset_m": args.pivot_anchor_height_offset_m,
            "pivot_anchor_forward_offset_m": args.pivot_anchor_forward_offset_m,
            "pivot_anchor_lateral_offset_m": args.pivot_anchor_lateral_offset_m,
            "pivot_arc_contact_tolerance_m": args.pivot_arc_contact_tolerance_m,
            "pivot_arc_max_steps": args.pivot_arc_max_steps,
            "pivot_arc_close_distance_m": args.pivot_arc_close_distance_m,
            "pivot_arc_frame_step_updates": args.pivot_arc_frame_step_updates,
            "pivot_arc_stream_samples": args.pivot_arc_stream_samples,
            "stream_samples": args.stream_samples,
            "stream_frame_step_updates": args.stream_frame_step_updates,
        },
    }


def _build_continuous_cycle_plan(
    *,
    active_arm: str,
    robot_base_rotation: list[float],
    pre_grasp_target: np.ndarray,
    grasp_align_target: np.ndarray | None,
    grasp_depth_target: np.ndarray,
    lift_clearance_target: np.ndarray,
    prebin_target: np.ndarray,
    place_depth_target: np.ndarray,
    retreat_target: np.ndarray,
    args: argparse.Namespace,
) -> list[MotionSegment]:
    soft_tolerance = args.continuous_soft_tolerance if args.smooth_motion else args.ik_stop_tolerance
    soft_samples = max(int(args.stream_samples), 1)
    hard_samples = max(int(args.stream_samples), 1)
    main_endpoint_strategy = "global_precomputed_joint_waypoint"
    robot_base_yaw_rad = _yaw_rad_from_rotation_xyz_deg(robot_base_rotation)

    def _stream_details(legacy_phase: str, endpoint_strategy: str, sample_count: int) -> dict[str, Any]:
        return {
            "legacy_phase": legacy_phase,
            "streaming_segment": True,
            "streaming_endpoint_strategy": endpoint_strategy,
            "streaming_sample_count": int(sample_count),
            "interpolation_profile": "global_quintic_timewarp_cubic_hermite_joint_chain",
            "endpoint_rebuilds_before_segment": 0,
            "finite_difference_jacobian_calls_during_stream": 0,
            "active_arm": active_arm,
            "robot_base_yaw_rad": float(robot_base_yaw_rad),
        }

    segments = [
        MotionSegment(
            name="continuous_pregrasp",
            phase_tag="approach",
            target_position=pre_grasp_target,
            waypoint_type="soft",
            speed_profile="streaming_transit",
            max_ik_steps=soft_samples,
            stop_tolerance=soft_tolerance,
            hold_steps=0,
            blend_radius=args.continuous_blend_radius,
            details=_stream_details("move_to_pre_grasp_front", main_endpoint_strategy, soft_samples),
        ),
    ]
    if grasp_align_target is not None:
        segments.append(
            MotionSegment(
                name="continuous_grasp_align",
                phase_tag="grasp_window",
                target_position=grasp_align_target,
                waypoint_type="soft",
                speed_profile="streaming_pre_contact_alignment",
                max_ik_steps=hard_samples,
                stop_tolerance=soft_tolerance,
                hold_steps=0,
                blend_radius=args.continuous_blend_radius,
                details=_stream_details("pre_contact_alignment", main_endpoint_strategy, hard_samples),
            )
        )
    segments.extend(
        [
            MotionSegment(
                name="continuous_grasp_depth",
                phase_tag="grasp_window",
                target_position=grasp_depth_target,
                waypoint_type="hard",
                speed_profile="streaming_contact_approach",
                max_ik_steps=hard_samples,
                stop_tolerance=args.ik_stop_tolerance,
                hold_steps=0,
                blend_radius=0.0,
                event_marker="close_gripper",
                contact_window=True,
                details=_stream_details("descend_front", main_endpoint_strategy, hard_samples),
            ),
            MotionSegment(
                name="continuous_lift_clearance",
                phase_tag="lift",
                target_position=lift_clearance_target,
                waypoint_type="hard",
                speed_profile="streaming_lift",
                max_ik_steps=hard_samples,
                stop_tolerance=args.ik_stop_tolerance,
                hold_steps=0,
                blend_radius=0.0,
                contact_window=True,
                gripper_effort=args.gripper_hold_effort,
                details=_stream_details("grasp_validation_and_lift_front", main_endpoint_strategy, hard_samples),
            ),
            MotionSegment(
                name="continuous_prebin",
                phase_tag="carry",
                target_position=prebin_target,
                waypoint_type="soft",
                speed_profile="streaming_carry",
                max_ik_steps=soft_samples,
                stop_tolerance=soft_tolerance,
                hold_steps=0,
                blend_radius=args.continuous_blend_radius,
                gripper_effort=args.gripper_hold_effort,
                details=_stream_details("move_to_bin_front", main_endpoint_strategy, soft_samples),
            ),
            MotionSegment(
                name="continuous_place_depth",
                phase_tag="place_window",
                target_position=place_depth_target,
                waypoint_type="hard",
                speed_profile="streaming_place_release",
                max_ik_steps=hard_samples,
                stop_tolerance=args.ik_stop_tolerance,
                hold_steps=0,
                blend_radius=0.0,
                event_marker="open_gripper",
                contact_window=True,
                gripper_effort=args.gripper_hold_effort,
                details=_stream_details("release", main_endpoint_strategy, hard_samples),
            ),
            MotionSegment(
                name="continuous_retreat",
                phase_tag="retreat",
                target_position=retreat_target,
                waypoint_type="soft",
                speed_profile="streaming_retreat",
                max_ik_steps=soft_samples,
                stop_tolerance=soft_tolerance,
                hold_steps=0,
                blend_radius=args.continuous_blend_radius,
                details=_stream_details("retreat", main_endpoint_strategy, soft_samples),
            ),
        ]
    )
    return segments


def _grasp_contact_geometry(
    *,
    selected_pregrasp_target: np.ndarray,
    object_center: np.ndarray,
    bbox_top_z: float,
    table_top_z: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contact_target = np.array(selected_pregrasp_target, dtype=float).copy()
    contact_target[0] += float(args.grasp_contact_offset_x)
    contact_target[1] += float(args.grasp_contact_offset_y)
    contact_target[2] = max(
        float(bbox_top_z) + args.descend_clearance + args.grasp_depth_offset,
        table_top_z + args.min_ee_table_clearance,
    )
    align_target = np.array(contact_target, dtype=float).copy()
    align_target[2] = float(selected_pregrasp_target[2])
    xy_delta = object_center[:2] - contact_target[:2]
    vertical_only = bool(np.allclose(align_target[:2], contact_target[:2], atol=1e-9))
    return {
        "selected_pregrasp_target": selected_pregrasp_target.tolist(),
        "grasp_contact_target": contact_target.tolist(),
        "pre_contact_alignment_target": align_target.tolist(),
        "object_center": object_center.tolist(),
        "align_target_xy": align_target[:2].tolist(),
        "descend_target_xy": contact_target[:2].tolist(),
        "vertical_only_descend": vertical_only,
        "xy_delta_contact_to_object": {
            "dx": float(xy_delta[0]),
            "dy": float(xy_delta[1]),
            "distance": float(np.linalg.norm(xy_delta)),
        },
        "grasp_contact_offset": {
            "x": float(args.grasp_contact_offset_x),
            "y": float(args.grasp_contact_offset_y),
        },
        "final_descend_vertical_only": vertical_only,
    }


def _unit_xy_from_base_to_target(robot_base_position: list[float], object_center: np.ndarray) -> np.ndarray:
    delta_xy = object_center[:2] - np.array(robot_base_position[:2], dtype=float)
    norm = float(np.linalg.norm(delta_xy))
    if norm <= 1.0e-9:
        return np.array([1.0, 0.0], dtype=float)
    return delta_xy / norm


def _estimate_pinch_center_from_pivot(
    *,
    pivot_position: np.ndarray,
    wrist_position: np.ndarray,
    object_center: np.ndarray,
    pivot_to_pinch_distance: float,
) -> dict[str, Any]:
    approach = object_center - pivot_position
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm <= 1.0e-9:
        approach_unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        approach_unit = approach / approach_norm
    pivot_to_wrist = float(np.linalg.norm(wrist_position - pivot_position))
    wrist_to_pinch = max(float(pivot_to_pinch_distance) - pivot_to_wrist, 0.0)
    pinch_center = wrist_position + approach_unit * wrist_to_pinch
    return {
        "approach_unit": approach_unit.tolist(),
        "pivot_to_wrist_distance": pivot_to_wrist,
        "wrist_to_pinch_distance_estimated": wrist_to_pinch,
        "pinch_center_estimated_position": pinch_center.tolist(),
        "pinch_center_to_object_distance": float(np.linalg.norm(pinch_center - object_center)),
        "wrist_to_object_distance": float(np.linalg.norm(wrist_position - object_center)),
    }


def _experimental_pivot_arc_plan(
    *,
    target_index: int,
    active_arm: str,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    end_effector_name: str,
    end_effector_path: str,
    pivot_body: Any,
    pivot_name: str,
    pivot_path: str,
    object_center: np.ndarray,
    bbox_top_z: float,
    table_top_z: float,
    robot_base_position: list[float],
    robot_base_rotation: list[float],
    pre_grasp_target: np.ndarray,
    lift_clearance_target: np.ndarray,
    prebin_target: np.ndarray,
    place_depth_target: np.ndarray,
    retreat_target: np.ndarray,
    args: argparse.Namespace,
    front_workspace_z: tuple[float, float],
) -> tuple[list[MotionSegment], dict[str, Any]]:
    upper_names = {
        f"{'R' if active_arm == 'right' else 'L'}_shoulder_pitch_joint",
        f"{'R' if active_arm == 'right' else 'L'}_shoulder_roll_joint",
        f"{'R' if active_arm == 'right' else 'L'}_shoulder_yaw_joint",
        f"{'R' if active_arm == 'right' else 'L'}_elbow_roll_joint",
    }
    lower_names = {
        f"{'R' if active_arm == 'right' else 'L'}_elbow_yaw_joint",
        f"{'R' if active_arm == 'right' else 'L'}_wrist_pitch_joint",
        f"{'R' if active_arm == 'right' else 'L'}_wrist_roll_joint",
    }
    upper_dofs = _select_dofs_by_names(arm_dofs, upper_names)
    lower_dofs = _select_dofs_by_names(arm_dofs, lower_names)
    if len(upper_dofs) != 4 or len(lower_dofs) != 3:
        raise RuntimeError(
            f"Experimental pivot arc missing expected DOFs: "
            f"upper={[name for _, _, name in upper_dofs]} lower={[name for _, _, name in lower_dofs]}"
        )

    robot_base_yaw_rad = _yaw_rad_from_rotation_xyz_deg(robot_base_rotation)
    approach_xy = _unit_xy_from_base_to_target(robot_base_position, object_center)
    lateral_xy = np.array([-approach_xy[1], approach_xy[0]], dtype=float)
    contact_object = np.array(object_center, dtype=float).copy()
    contact_object[2] = max(
        float(bbox_top_z) + args.descend_clearance + args.grasp_depth_offset,
        table_top_z + args.min_ee_table_clearance,
    )
    contact_object[2] = float(np.clip(contact_object[2], *front_workspace_z))
    pivot_anchor = contact_object.copy()
    pivot_anchor[:2] -= approach_xy * (float(args.pivot_to_pinch_distance_m) + float(args.pivot_anchor_forward_offset_m))
    pivot_anchor[:2] += lateral_xy * float(args.pivot_anchor_lateral_offset_m)
    pivot_anchor[2] = float(np.clip(contact_object[2] + float(args.pivot_anchor_height_offset_m), *front_workspace_z))

    # Approximate the remaining wrist-to-pinch distance after the wrist-pitch pivot.
    wrist_roll_offset_from_pivot = 0.076
    wrist_to_pinch = max(float(args.pivot_to_pinch_distance_m) - wrist_roll_offset_from_pivot, 0.03)
    contact_wrist_target = contact_object.copy()
    contact_wrist_target[:2] -= approach_xy * wrist_to_pinch
    contact_wrist_target[2] = contact_object[2]
    arc_mid_target = contact_wrist_target.copy()
    arc_mid_target[:2] -= approach_xy * 0.03
    arc_mid_target[2] = float(np.clip(contact_object[2] + 0.025, *front_workspace_z))

    details = {
        "enabled": True,
        "fallback_to_baseline_used": False,
        "control_pivot_frame_name": pivot_name,
        "control_pivot_frame_path": pivot_path,
        "pinch_center_reference_frame_name": "estimated_pinch_center_from_wrist_roll_plus_pivot_line_offset",
        "pinch_center_reference_wrist_frame_name": end_effector_name,
        "pinch_center_reference_wrist_frame_path": end_effector_path,
        "target_object_center": object_center.tolist(),
        "target_object_contact_point": contact_object.tolist(),
        "pivot_anchor_target": pivot_anchor.tolist(),
        "pivot_to_object_distance": float(np.linalg.norm(pivot_anchor - contact_object)),
        "pivot_to_pinch_distance_used": float(args.pivot_to_pinch_distance_m),
        "wrist_to_pinch_distance_estimated_for_targets": float(wrist_to_pinch),
        "stage_a_succeeded": None,
        "stage_b_started": False,
        "close_condition_satisfied_before_close": False,
        "final_local_approach_shape": "arc-like",
        "object_under_wrist_or_near_finger_gap": "not_evaluated",
        "upper_chain_dof_names": [name for _, _, name in upper_dofs],
        "lower_chain_dof_names": [name for _, _, name in lower_dofs],
        "approach_xy_unit_from_base_to_object": approach_xy.tolist(),
        "active_arm": active_arm,
        "robot_base_yaw_rad": float(robot_base_yaw_rad),
        "estimated_defaults": {
            "pivot_to_pinch_distance_m": float(args.pivot_to_pinch_distance_m),
            "wrist_roll_offset_from_pivot_m": wrist_roll_offset_from_pivot,
        },
    }
    common_close_details = {
        "experimental_pivot_arc": True,
        "pivot_arc_conditional_close": True,
        "pivot_arc_object_center": contact_object.tolist(),
        "pivot_arc_close_distance_m": float(args.pivot_arc_close_distance_m),
        "pivot_to_pinch_distance_used": float(args.pivot_to_pinch_distance_m),
    }
    contact_hold_steps = args.continuous_contact_dwell_steps if args.smooth_motion else args.ik_hold_steps
    soft_tolerance = args.continuous_soft_tolerance if args.smooth_motion else args.ik_stop_tolerance
    max_arc_steps = max(int(args.pivot_arc_stream_samples), int(args.pivot_arc_max_steps), 1)
    stream_samples = max(int(args.stream_samples), 1)
    segments = [
        MotionSegment(
            name="continuous_pregrasp",
            phase_tag="approach",
            target_position=pre_grasp_target,
            waypoint_type="soft",
            speed_profile="experimental_streaming_pregrasp",
            max_ik_steps=max_arc_steps,
            stop_tolerance=soft_tolerance,
            hold_steps=0,
            blend_radius=args.continuous_blend_radius,
            details={
                "legacy_phase": "move_to_pre_grasp_front",
                "experimental_pivot_arc": True,
                "experimental_streaming_segment": True,
                "streaming_endpoint_strategy": "hold_current",
                "streaming_sample_count": max_arc_steps,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
        ),
        MotionSegment(
            name="experimental_pivot_anchor",
            phase_tag="grasp_window",
            target_position=pivot_anchor,
            waypoint_type="soft",
            speed_profile="experimental_streaming_pivot_anchor",
            max_ik_steps=max_arc_steps,
            stop_tolerance=args.pivot_arc_contact_tolerance_m,
            hold_steps=0,
            blend_radius=args.continuous_blend_radius,
            details={
                "experimental_pivot_arc": True,
                "experimental_streaming_segment": True,
                "streaming_endpoint_strategy": "upper_anchor_heuristic",
                "streaming_sample_count": max_arc_steps,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
                "stage": "A_pivot_positioning",
                **details,
            },
            control_body=pivot_body,
            control_body_name=pivot_name,
            control_body_path=pivot_path,
            control_dofs=upper_dofs,
        ),
        MotionSegment(
            name="experimental_locked_lower_chain_arc",
            phase_tag="grasp_window",
            target_position=contact_wrist_target,
            waypoint_type="hard",
            speed_profile="locked_pivot_lower_chain_arc",
            max_ik_steps=max_arc_steps,
            stop_tolerance=args.pivot_arc_contact_tolerance_m,
            hold_steps=0,
            blend_radius=0.0,
            event_marker="close_gripper",
            contact_window=True,
            details={
                **common_close_details,
                "stage": "B_locked_lower_chain_arc",
                "locked_lower_chain_arc": True,
                "pivot_anchor_target": pivot_anchor.tolist(),
                "arc_mid_target": arc_mid_target.tolist(),
                "arc_contact_target": contact_wrist_target.tolist(),
                "arc_waypoint_count": max_arc_steps,
                "contact_dwell_steps_after_close": contact_hold_steps,
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
            control_body=end_effector_body,
            control_body_name=end_effector_name,
            control_body_path=end_effector_path,
            control_dofs=lower_dofs,
            pinch_metric_pivot_body=pivot_body,
        ),
        MotionSegment(
            name="continuous_lift_clearance",
            phase_tag="lift",
            target_position=lift_clearance_target,
            waypoint_type="hard",
            speed_profile="streaming_lift",
            max_ik_steps=stream_samples,
            stop_tolerance=args.ik_stop_tolerance,
            hold_steps=0,
            blend_radius=0.0,
            contact_window=True,
            gripper_effort=args.gripper_hold_effort,
            details={
                "legacy_phase": "grasp_validation_and_lift_front",
                "experimental_pivot_arc": True,
                "streaming_segment": True,
                "streaming_endpoint_strategy": "main_cycle_heuristic_endpoint",
                "streaming_sample_count": stream_samples,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
        ),
        MotionSegment(
            name="continuous_prebin",
            phase_tag="carry",
            target_position=prebin_target,
            waypoint_type="soft",
            speed_profile="streaming_carry",
            max_ik_steps=stream_samples,
            stop_tolerance=soft_tolerance,
            hold_steps=0,
            blend_radius=args.continuous_blend_radius,
            gripper_effort=args.gripper_hold_effort,
            details={
                "legacy_phase": "move_to_bin_front",
                "experimental_pivot_arc": True,
                "streaming_segment": True,
                "streaming_endpoint_strategy": "main_cycle_heuristic_endpoint",
                "streaming_sample_count": stream_samples,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
        ),
        MotionSegment(
            name="continuous_place_depth",
            phase_tag="place_window",
            target_position=place_depth_target,
            waypoint_type="hard",
            speed_profile="streaming_place_release",
            max_ik_steps=stream_samples,
            stop_tolerance=args.ik_stop_tolerance,
            hold_steps=0,
            blend_radius=0.0,
            event_marker="open_gripper",
            contact_window=True,
            gripper_effort=args.gripper_hold_effort,
            details={
                "legacy_phase": "release",
                "experimental_pivot_arc": True,
                "streaming_segment": True,
                "streaming_endpoint_strategy": "main_cycle_heuristic_endpoint",
                "streaming_sample_count": stream_samples,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
        ),
        MotionSegment(
            name="continuous_retreat",
            phase_tag="retreat",
            target_position=retreat_target,
            waypoint_type="soft",
            speed_profile="streaming_retreat",
            max_ik_steps=stream_samples,
            stop_tolerance=soft_tolerance,
            hold_steps=0,
            blend_radius=args.continuous_blend_radius,
            details={
                "legacy_phase": "retreat",
                "experimental_pivot_arc": True,
                "streaming_segment": True,
                "streaming_endpoint_strategy": "main_cycle_heuristic_endpoint",
                "streaming_sample_count": stream_samples,
                "interpolation_profile": "quintic_minimum_jerk",
                "active_arm": active_arm,
                "robot_base_yaw_rad": float(robot_base_yaw_rad),
            },
        ),
    ]
    return segments, details


def _trigger_motion_event(
    *,
    event_marker: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    locked_dofs: list[tuple[int, Any, str]] | None = None,
    locked_targets: dict[str, float] | None = None,
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
                locked_dofs,
                locked_targets,
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
            locked_dofs,
            locked_targets,
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
                locked_dofs,
                locked_targets,
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
            locked_dofs,
            locked_targets,
        )
        return {
            "event_marker": event_marker,
            "skipped": False,
            "dwell_steps": args.continuous_contact_dwell_steps,
            "commanded_targets": _named_positions(gripper_dofs, open_targets),
            "gripper_effort": effort_result,
        }, None

    raise RuntimeError(f"Unknown motion event marker: {event_marker}")


def _run_locked_updates(
    *,
    dc: Any,
    sim_app: Any,
    counter: dict[str, int],
    locked_dofs: list[tuple[int, Any, str]],
    locked_targets: dict[str, float],
    steps: int,
) -> None:
    for _ in range(max(int(steps), 1)):
        _apply_locked_joint_targets(dc, locked_dofs, locked_targets)
        sim_app.update()
        counter["step"] += 1


def _apply_lower_chain_positions_with_lock(
    *,
    dc: Any,
    lower_dofs: list[tuple[int, Any, str]],
    positions: np.ndarray,
    sim_app: Any,
    counter: dict[str, int],
    locked_dofs: list[tuple[int, Any, str]],
    locked_targets: dict[str, float],
    settle_steps: int,
) -> None:
    _send_position_targets(dc, lower_dofs, [float(value) for value in positions])
    _run_locked_updates(
        dc=dc,
        sim_app=sim_app,
        counter=counter,
        locked_dofs=locked_dofs,
        locked_targets=locked_targets,
        steps=settle_steps,
    )


def _estimate_position_jacobian_with_lock(
    *,
    dc: Any,
    articulation: Any,
    lower_dofs: list[tuple[int, Any, str]],
    lower_body: Any,
    base_positions: np.ndarray,
    base_body_position: np.ndarray,
    sim_app: Any,
    counter: dict[str, int],
    locked_dofs: list[tuple[int, Any, str]],
    locked_targets: dict[str, float],
    eps: float,
    settle_steps: int,
) -> np.ndarray:
    jacobian = np.zeros((3, len(lower_dofs)), dtype=float)
    for column in range(len(lower_dofs)):
        trial = base_positions.copy()
        trial[column] += eps
        _apply_lower_chain_positions_with_lock(
            dc=dc,
            lower_dofs=lower_dofs,
            positions=trial,
            sim_app=sim_app,
            counter=counter,
            locked_dofs=locked_dofs,
            locked_targets=locked_targets,
            settle_steps=settle_steps,
        )
        dc.wake_up_articulation(articulation)
        moved_position = _body_pose_position(dc, lower_body)
        jacobian[:, column] = (moved_position - base_body_position) / eps

    _apply_lower_chain_positions_with_lock(
        dc=dc,
        lower_dofs=lower_dofs,
        positions=base_positions,
        sim_app=sim_app,
        counter=counter,
        locked_dofs=locked_dofs,
        locked_targets=locked_targets,
        settle_steps=settle_steps,
    )
    dc.wake_up_articulation(articulation)
    return jacobian


def _execute_locked_lower_chain_arc(
    *,
    target_index: int,
    target_path: str,
    stage: Any,
    dc: Any,
    articulation: Any,
    sim_app: Any,
    counter: dict[str, int],
    lower_dofs: list[tuple[int, Any, str]],
    lower_body: Any,
    lower_body_name: str,
    lower_body_path: str,
    pivot_body: Any,
    upper_locked_dofs: list[tuple[int, Any, str]],
    upper_locked_targets: dict[str, float],
    pivot_reference_position: np.ndarray,
    arc_waypoints: list[np.ndarray],
    ik_steps_per_waypoint: int,
    ik_settle_steps: int,
    ik_position_eps: float,
    ik_damping: float,
    ik_max_step: float,
    ik_posture_gain: float,
    stop_tolerance: float,
    frame_step_updates: int,
    phase_label_prefix: str,
    phase_log: list[dict[str, Any]],
    gripper_dofs: list[tuple[int, Any, str]],
    args: argparse.Namespace,
    object_center: np.ndarray,
    pivot_to_pinch_distance: float,
    close_distance: float,
    active_arm: str,
    robot_base_yaw_rad: float,
) -> dict[str, Any]:
    start_step = counter["step"]
    start_positions = np.array(_current_positions(dc, lower_dofs), dtype=float)
    trace: list[dict[str, Any]] = []
    drift_samples: list[float] = []
    drift_per_waypoint: list[dict[str, Any]] = []
    lock_active_per_step: list[bool] = []
    upper_joint_error_per_step: list[dict[str, Any]] = []
    upper_joint_error_max_samples: list[float] = []
    snapshots: dict[str, Any] = {}
    event_details = None
    active_effort: float | None = None
    close_event_fired = False
    close_condition_satisfied = False
    close_condition_metric: dict[str, Any] | None = None
    pivot_position_during_arc_start = _body_pose_position(dc, pivot_body)
    pivot_position_during_arc_end = pivot_position_during_arc_start.copy()
    target_position = arc_waypoints[-1] if arc_waypoints else _body_pose_position(dc, lower_body)
    start_body_position = _body_pose_position(dc, lower_body)
    frame_updates = max(int(frame_step_updates), 1)

    end_positions = _heuristic_lower_chain_arc_endpoint(
        lower_dofs=lower_dofs,
        start_positions=start_positions,
        start_body_position=start_body_position,
        target_position=target_position,
        active_arm=active_arm,
        robot_base_yaw_rad=robot_base_yaw_rad,
        max_step=ik_max_step,
        sample_count=len(arc_waypoints),
    )

    def _record_stream_state(sample_index: int, target: np.ndarray, actual_position: np.ndarray, error_norm: float) -> None:
        nonlocal pivot_position_during_arc_end
        pivot_position = _body_pose_position(dc, pivot_body)
        drift = _compute_pivot_drift(pivot_reference_position, pivot_position)
        joint_error = _locked_joint_error(dc, upper_locked_dofs, upper_locked_targets)
        pivot_position_during_arc_end = pivot_position.copy()
        drift_samples.append(drift)
        lock_active_per_step.append(True)
        upper_joint_error_max_samples.append(float(joint_error["max"]))
        upper_joint_error_per_step.append(
            {
                "sample_index": int(sample_index),
                "max": float(joint_error["max"]),
                "mean": float(joint_error["mean"]),
                "by_joint": joint_error["by_joint"],
            }
        )
        trace.append(
            {
                "phase": phase_label_prefix,
                "sample_index": int(sample_index),
                "target": target.tolist(),
                "actual": actual_position.tolist(),
                "error_norm": float(error_norm),
                "pivot_position": pivot_position.tolist(),
                "pivot_drift_norm": drift,
                "upper_chain_lock_active": True,
                "upper_chain_joint_error_max": float(joint_error["max"]),
            }
        )

    for waypoint_index, waypoint in enumerate(arc_waypoints):
        waypoint_drift_before = len(drift_samples)
        t = (waypoint_index + 1) / float(max(len(arc_waypoints), 1))
        alpha = _quintic_blend(t)
        lower_target = start_positions + alpha * (end_positions - start_positions)
        _send_position_targets(dc, lower_dofs, [float(value) for value in lower_target])
        for _ in range(frame_updates):
            _apply_locked_joint_targets(dc, upper_locked_dofs, upper_locked_targets)
            sim_app.update()
            counter["step"] += 1
            dc.wake_up_articulation(articulation)
            current_position = _body_pose_position(dc, lower_body)
            current_error = float(np.linalg.norm(waypoint - current_position))
            _record_stream_state(waypoint_index, waypoint, current_position, current_error)
            close_condition_metric = _estimate_pinch_center_from_pivot(
                pivot_position=_body_pose_position(dc, pivot_body),
                wrist_position=current_position,
                object_center=object_center,
                pivot_to_pinch_distance=pivot_to_pinch_distance,
            )
            close_condition_satisfied = bool(
                close_condition_metric["pinch_center_to_object_distance"] <= float(close_distance)
            )
            if close_condition_satisfied and not close_event_fired:
                snapshots["before_close_gripper"] = _bbox_state(stage, target_path)
                event_details, active_effort = _trigger_motion_event(
                    event_marker="close_gripper",
                    dc=dc,
                    gripper_dofs=gripper_dofs,
                    sim_app=sim_app,
                    args=args,
                    counter=counter,
                    locked_dofs=upper_locked_dofs,
                    locked_targets=upper_locked_targets,
                )
                _apply_locked_joint_targets(dc, upper_locked_dofs, upper_locked_targets)
                snapshots["after_close_gripper"] = _bbox_state(stage, target_path)
                close_event_fired = True
                break
        waypoint_drifts = drift_samples[waypoint_drift_before:]
        drift_per_waypoint.append(
            {
                "waypoint_index": int(waypoint_index),
                "target": waypoint.tolist(),
                **_summarize_pivot_drift(waypoint_drifts),
            }
        )
        if close_event_fired:
            break

    actual_position = _body_pose_position(dc, lower_body)
    position_error = float(np.linalg.norm(target_position - actual_position))
    if close_condition_metric is None:
        close_condition_metric = _estimate_pinch_center_from_pivot(
            pivot_position=_body_pose_position(dc, pivot_body),
            wrist_position=actual_position,
            object_center=object_center,
            pivot_to_pinch_distance=pivot_to_pinch_distance,
        )
        close_condition_satisfied = bool(
            close_condition_metric["pinch_center_to_object_distance"] <= float(close_distance)
        )
    drift_summary = _summarize_pivot_drift(drift_samples)
    upper_error_summary = _summarize_scalar_samples(upper_joint_error_max_samples, "upper_chain_joint_error")
    micro_stop_estimate = _arc_micro_stop_estimate(
        trace=trace,
        frame_span_per_trace_sample=frame_updates,
        threshold=args.micro_stop_speed_threshold,
    )
    final_pivot_position = _body_pose_position(dc, pivot_body)
    object_alignment = (
        "near_finger_gap"
        if close_condition_metric["pinch_center_to_object_distance"] < close_condition_metric["wrist_to_object_distance"]
        else "under_wrist_or_palm_projection"
    )
    condition_met = bool(close_event_fired or position_error <= stop_tolerance)
    details = {
        "target_index": target_index,
        "continuous_motion": True,
        "phase_tag": "grasp_window",
        "waypoint_type": "hard",
        "speed_profile": "locked_pivot_lower_chain_arc",
        "streaming_controller": True,
        "arc_runtime_controller": "streaming_lower_joint_interpolation",
        "finite_difference_jacobian_calls_during_arc_stream": 0,
        "arc_endpoint_strategy": "heuristic_lower_joint_offsets_no_runtime_jacobian",
        "interpolation_profile": "quintic_minimum_jerk",
        "control_body_name": lower_body_name,
        "control_body_path": lower_body_path,
        "control_dof_names": [name for _, _, name in lower_dofs],
        "locked_dof_names": [name for _, _, name in upper_locked_dofs],
        "upper_chain_locked_targets": dict(upper_locked_targets),
        "upper_chain_lock_active": True,
        "upper_chain_lock_active_during_arc": True,
        "upper_chain_lock_active_per_step": lock_active_per_step,
        "upper_chain_joint_error_per_step": upper_joint_error_per_step,
        "upper_chain_joint_error_max_during_arc": upper_error_summary["upper_chain_joint_error_max"],
        "upper_chain_joint_error_mean_during_arc": upper_error_summary["upper_chain_joint_error_mean"],
        "upper_chain_joint_error_final_during_arc": upper_error_summary["upper_chain_joint_error_final"],
        "upper_chain_joint_error_sample_count": upper_error_summary["upper_chain_joint_error_sample_count"],
        "pivot_reference_position": pivot_reference_position.tolist(),
        "pivot_position_after_anchor": pivot_reference_position.tolist(),
        "pivot_position_during_arc_start": pivot_position_during_arc_start.tolist(),
        "pivot_position_during_arc_end": final_pivot_position.tolist(),
        "pivot_drift_per_waypoint": drift_per_waypoint,
        **drift_summary,
        "arc_waypoints": [waypoint.tolist() for waypoint in arc_waypoints],
        "lower_chain_start_targets": _named_positions(lower_dofs, start_positions),
        "lower_chain_stream_endpoint_targets": _named_positions(lower_dofs, end_positions),
        "pivot_arc_frame_step_updates": frame_updates,
        "configured_stream_sample_count": len(arc_waypoints),
        "target_position": target_position.tolist(),
        "position_error": position_error,
        "ik_step_count": 0,
        "stream_sample_count": len(trace),
        "ik_trace": trace,
        "micro_stop_frames": micro_stop_estimate["micro_stop_frames"],
        "micro_stop_samples": micro_stop_estimate["micro_stop_samples"],
        "stop_tolerance": stop_tolerance,
        "hold_steps": 0,
        "event": event_details,
        "stage": "B_locked_lower_chain_arc",
        "experimental_pivot_arc": True,
        "stage_b_started": True,
        "close_condition_metric": close_condition_metric,
        "close_condition_satisfied_before_close": bool(close_condition_satisfied and close_event_fired),
        "close_event_fired": close_event_fired,
        "final_local_approach_shape": "arc-like_locked_lower_chain",
        "object_under_wrist_or_near_finger_gap": object_alignment,
    }
    _append_phase(
        phase_log,
        phase=phase_label_prefix,
        start_step=start_step,
        end_step=counter["step"],
        commanded_targets=_named_positions(lower_dofs, _current_positions(dc, lower_dofs)),
        ee_position=actual_position,
        gripper_values=_gripper_values(dc, gripper_dofs),
        condition_met=condition_met,
        details=details,
    )
    return {
        "actual_position": actual_position,
        "position_error": position_error,
        "trace": trace,
        "event": event_details,
        "active_effort": active_effort,
        "close_event_fired": close_event_fired,
        "snapshots": snapshots,
        "condition_met": condition_met,
        "experimental_details": details,
        "pivot_drift_summary": drift_summary,
        "micro_stop_estimate": micro_stop_estimate,
    }


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


def _can_execute_as_global_precomputed_cycle(segments: list[MotionSegment]) -> bool:
    if not segments:
        return False
    return all(
        segment.control_body is None
        and segment.control_dofs is None
        and not segment.details.get("locked_lower_chain_arc")
        and str(segment.details.get("streaming_endpoint_strategy")) == "global_precomputed_joint_waypoint"
        for segment in segments
    )


def _segment_stream_sample_count(segment: MotionSegment) -> int:
    return max(int(segment.details.get("streaming_sample_count", segment.max_ik_steps)), 1)


def _segment_index_for_progress(cumulative_samples: list[float], progress: float) -> int:
    last_segment = len(cumulative_samples) - 2
    for index in range(last_segment + 1):
        if progress <= cumulative_samples[index + 1]:
            return index
    return last_segment


def _execute_global_precomputed_motion_cycle(
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
    start_positions = np.array(_current_positions(dc, arm_dofs), dtype=float)
    start_body_position = _body_pose_position(dc, end_effector_body)
    sample_counts = [_segment_stream_sample_count(segment) for segment in segments]
    total_samples = max(sum(sample_counts), 1)
    cumulative_samples = [0.0]
    for sample_count in sample_counts:
        cumulative_samples.append(cumulative_samples[-1] + float(sample_count))
    active_arm = str(segments[0].details.get("active_arm", "right"))
    robot_base_yaw_rad = float(segments[0].details.get("robot_base_yaw_rad", 0.0))
    joint_waypoints = _precompute_main_cycle_joint_waypoints(
        selected_dofs=arm_dofs,
        start_positions=start_positions,
        start_body_position=start_body_position,
        segments=segments,
        active_arm=active_arm,
        robot_base_yaw_rad=robot_base_yaw_rad,
        args=args,
    )
    joint_waypoint_tangents = _joint_waypoint_tangents(joint_waypoints, cumulative_samples)

    metrics: dict[str, Any] = {
        "target_index": target_index,
        "cycle_start_step": cycle_start_step,
        "segment_count": len(segments),
        "execution_style": "global_absolute_joint_waypoint_cubic_hermite_stream_no_per_segment_endpoint_rebuild",
        "streaming_controller_active": True,
        "endpoint_rebuilds_during_cycle": 0,
        "finite_difference_jacobian_calls_during_cycle": 0,
        "stream_samples": int(args.stream_samples),
        "stream_frame_step_updates": int(args.stream_frame_step_updates),
        "pivot_arc_stream_samples": int(args.pivot_arc_stream_samples),
        "pivot_arc_frame_step_updates": int(args.pivot_arc_frame_step_updates),
        "interpolation_profile": "global_quintic_timewarp_cubic_hermite_joint_chain",
        "waypoint_generation": "absolute_target_relative_to_cycle_start_no_cumulative_heuristic_drift",
        "directional_heuristic_frame": "robot_base_xy_signed_forward_lateral_with_backward_guard",
        "active_arm": active_arm,
        "robot_base_yaw_rad": float(robot_base_yaw_rad),
        "micro_stop_frames": 0,
        "micro_stop_samples": 0,
        "micro_stop_speed_threshold_m_per_frame": args.micro_stop_speed_threshold,
        "phase_tags": ["approach", "grasp_window", "post_close_verify", "lift", "carry", "place_window", "retreat"],
        "tuning_knobs": _tuning_knob_summary(args),
        "close_event_fired": False,
        "open_event_fired": False,
        "snapshots": {},
        "segment_summaries": [],
        "global_stream": {
            "total_samples": int(total_samples),
            "segment_sample_counts": {segment.name: count for segment, count in zip(segments, sample_counts)},
            "joint_waypoint_count": len(joint_waypoints),
            "joint_trajectory_interpolator": "cubic_hermite_minmod_tangents",
            "slope_continuity_across_waypoints": True,
            "joint_waypoint_tangent_norm_max": float(
                max((np.linalg.norm(tangent) for tangent in joint_waypoint_tangents), default=0.0)
            ),
            "target_updates_per_sim_update": 1,
        },
    }

    active_effort: float | None = None
    segment_traces: list[list[dict[str, Any]]] = [[] for _ in segments]
    segment_start_steps: dict[int, int] = {}
    completed_segments: set[int] = set()
    started_segments: set[int] = set()
    frame_updates = max(int(args.stream_frame_step_updates), 1)

    def _start_segment(segment_index: int) -> None:
        nonlocal active_effort
        if segment_index in started_segments:
            return
        segment = segments[segment_index]
        started_segments.add(segment_index)
        segment_start_steps[segment_index] = counter["step"]
        metrics["snapshots"][f"before_{segment.name}"] = _bbox_state(stage, target_path)
        if segment.gripper_effort is not None:
            active_effort = segment.gripper_effort

    def _complete_segment(segment_index: int) -> None:
        nonlocal active_effort
        if segment_index in completed_segments:
            return
        _start_segment(segment_index)
        segment = segments[segment_index]
        actual_position = _body_pose_position(dc, end_effector_body)
        position_error = float(np.linalg.norm(segment.target_position - actual_position))
        trace = segment_traces[segment_index]
        micro_stop_estimate = _arc_micro_stop_estimate(
            trace=trace,
            frame_span_per_trace_sample=frame_updates,
            threshold=args.micro_stop_speed_threshold,
        )
        details = {
            **segment.details,
            "target_index": target_index,
            "continuous_motion": True,
            "phase_tag": segment.phase_tag,
            "waypoint_type": segment.waypoint_type,
            "speed_profile": segment.speed_profile,
            "streaming_controller": True,
            "streaming_endpoint_strategy": "global_precomputed_joint_waypoint",
            "single_global_stream_cycle": True,
            "endpoint_rebuilds_before_segment": 0,
            "finite_difference_jacobian_calls_during_stream": 0,
            "stream_sample_count": sample_counts[segment_index],
            "frame_step_updates": frame_updates,
            "interpolation_profile": "global_quintic_timewarp_cubic_hermite_joint_chain",
            "joint_trajectory_interpolator": "cubic_hermite_minmod_tangents",
            "waypoint_generation": "absolute_target_relative_to_cycle_start_no_cumulative_heuristic_drift",
            "control_body_name": "default_end_effector",
            "control_body_path": "default_end_effector",
            "control_dof_names": [name for _, _, name in arm_dofs],
            "target_position": segment.target_position.tolist(),
            "position_error": position_error,
            "stop_tolerance": segment.stop_tolerance,
            "ik_step_count": 0,
            "ik_trace": trace,
            "micro_stop_frames": micro_stop_estimate["micro_stop_frames"],
            "micro_stop_samples": micro_stop_estimate["micro_stop_samples"],
            "joint_waypoint_start": _named_positions(arm_dofs, joint_waypoints[segment_index]),
            "joint_waypoint_end": _named_positions(arm_dofs, joint_waypoints[segment_index + 1]),
        }
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
            details["event"] = event_details
            if segment.event_marker == "close_gripper":
                metrics["close_event_fired"] = True
            if segment.event_marker == "open_gripper":
                metrics["open_event_fired"] = True
            metrics["snapshots"][f"after_{segment.event_marker}"] = _bbox_state(stage, target_path)
        if active_effort is not None:
            effort_after = _apply_gripper_effort(dc, gripper_dofs, active_effort)
            if not effort_after["supported"]:
                raise RuntimeError(f"Gripper effort command failed after {segment.name}: {effort_after}")
        else:
            effort_after = None
        details["gripper_effort_after_segment"] = effort_after
        details["carry_stabilization_steps"] = args.carry_stabilization_steps if segment.phase_tag == "carry" else 0
        details["carry_stabilization_effort"] = None
        if segment.phase_tag == "carry" and args.carry_stabilization_steps > 0:
            details["carry_stabilization_effort"] = _run_updates_with_optional_gripper_effort(
                sim_app,
                args.carry_stabilization_steps,
                counter,
                dc,
                gripper_dofs,
                active_effort,
            )
            metrics["snapshots"]["after_carry_stabilization"] = _bbox_state(stage, target_path)
        condition_met = bool(position_error <= segment.stop_tolerance)
        _append_phase(
            phase_log,
            phase=segment.name,
            start_step=segment_start_steps.get(segment_index, cycle_start_step),
            end_step=counter["step"],
            commanded_targets=_named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
            ee_position=actual_position,
            gripper_values=_gripper_values(dc, gripper_dofs),
            condition_met=condition_met,
            details=details,
        )
        metrics["micro_stop_frames"] += int(micro_stop_estimate["micro_stop_frames"])
        metrics["micro_stop_samples"] += int(micro_stop_estimate["micro_stop_samples"])
        metrics["snapshots"][f"after_{segment.name}"] = _bbox_state(stage, target_path)
        metrics["segment_summaries"].append(
            {
                "name": segment.name,
                "phase_tag": segment.phase_tag,
                "waypoint_type": segment.waypoint_type,
                "speed_profile": segment.speed_profile,
                "control_body_name": "default_end_effector",
                "control_body_path": "default_end_effector",
                "control_dof_names": [name for _, _, name in arm_dofs],
                "blend_radius": segment.blend_radius,
                "target_position": segment.target_position.tolist(),
                "actual_position": actual_position.tolist(),
                "position_error": position_error,
                "condition_met": condition_met,
                "contact_window": segment.contact_window,
                "event_marker": segment.event_marker,
                "micro_stop_estimate": micro_stop_estimate,
                "streaming_details": details,
            }
        )
        print(
            f"phase={segment.name} streaming_controller=True no_blocking_ik=True "
            f"global_stream=True stream_samples={sample_counts[segment_index]} "
            f"frame_step_updates={frame_updates} position_error={position_error} event={segment.event_marker}"
        )
        completed_segments.add(segment_index)

    next_boundary_index = 0
    _start_segment(0)
    for sample_index in range(total_samples):
        t = (sample_index + 1) / float(total_samples)
        progress = _quintic_blend(t) * float(total_samples)
        segment_index, local_alpha, target_joints = _sample_cubic_hermite_joint_chain(
            waypoints=joint_waypoints,
            tangents=joint_waypoint_tangents,
            cumulative_samples=cumulative_samples,
            progress=progress,
        )
        _start_segment(segment_index)
        _send_position_targets(dc, arm_dofs, [float(value) for value in target_joints])
        for _ in range(frame_updates):
            if active_effort is not None:
                effort_during = _apply_gripper_effort(dc, gripper_dofs, active_effort)
                if not effort_during["supported"]:
                    raise RuntimeError(f"Gripper effort command failed during global stream: {effort_during}")
            sim_app.update()
            counter["step"] += 1
        dc.wake_up_articulation(articulation)
        actual_position = _body_pose_position(dc, end_effector_body)
        segment_traces[segment_index].append(
            {
                "phase": segments[segment_index].name,
                "sample_index": int(sample_index),
                "target_joint_positions": _named_positions(arm_dofs, target_joints),
                "target": segments[segment_index].target_position.tolist(),
                "actual": actual_position.tolist(),
                "error_norm": float(np.linalg.norm(segments[segment_index].target_position - actual_position)),
                "global_progress_sample": float(progress),
                "local_spline_alpha": float(local_alpha),
            }
        )
        while (
            next_boundary_index < len(segments)
            and progress >= cumulative_samples[next_boundary_index + 1] - 1.0e-9
        ):
            _complete_segment(next_boundary_index)
            next_boundary_index += 1
            if next_boundary_index < len(segments):
                _start_segment(next_boundary_index)

    for segment_index in range(len(segments)):
        _complete_segment(segment_index)

    metrics["cycle_end_step"] = counter["step"]
    metrics["cycle_time_steps"] = counter["step"] - cycle_start_step
    return metrics


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
    if _can_execute_as_global_precomputed_cycle(segments):
        return _execute_global_precomputed_motion_cycle(
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
            segments=segments,
        )

    cycle_start_step = counter["step"]
    metrics: dict[str, Any] = {
        "target_index": target_index,
        "cycle_start_step": cycle_start_step,
        "segment_count": len(segments),
        "execution_style": "streaming_targets_no_blocking_ik_for_main_smooth_path",
        "streaming_controller_active": True,
        "stream_samples": int(args.stream_samples),
        "stream_frame_step_updates": int(args.stream_frame_step_updates),
        "pivot_arc_stream_samples": int(args.pivot_arc_stream_samples),
        "pivot_arc_frame_step_updates": int(args.pivot_arc_frame_step_updates),
        "interpolation_profile": "quintic_minimum_jerk",
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
    experimental_lock_state: dict[str, Any] | None = None

    for segment in segments:
        start_step = counter["step"]
        segment_body = segment.control_body if segment.control_body is not None else end_effector_body
        segment_dofs = segment.control_dofs if segment.control_dofs is not None else arm_dofs
        segment_body_name = segment.control_body_name or "default_end_effector"
        segment_body_path = segment.control_body_path or "default_end_effector"
        start_ee = _body_pose_position(dc, segment_body)

        if not segment.details.get("locked_lower_chain_arc"):
            if segment.gripper_effort is not None:
                active_effort = segment.gripper_effort
            metrics["snapshots"][f"before_{segment.name}"] = _bbox_state(stage, target_path)
            stream_sample_count = int(segment.details.get("streaming_sample_count", segment.max_ik_steps))
            stream_result = _execute_streaming_body_segment(
                target_index=target_index,
                dc=dc,
                articulation=articulation,
                sim_app=sim_app,
                counter=counter,
                selected_dofs=segment_dofs,
                body=segment_body,
                body_name=segment_body_name,
                body_path=segment_body_path,
                target_position=segment.target_position,
                sample_count=stream_sample_count,
                frame_step_updates=args.stream_frame_step_updates,
                endpoint_strategy=str(segment.details.get("streaming_endpoint_strategy", "main_cycle_heuristic_endpoint")),
                args=args,
                phase_name=segment.name,
                phase_tag=segment.phase_tag,
                waypoint_type=segment.waypoint_type,
                speed_profile=segment.speed_profile,
                gripper_dofs=gripper_dofs,
                phase_log=phase_log,
                extra_details=segment.details,
                gripper_effort_value=active_effort,
                stop_tolerance=segment.stop_tolerance,
            )
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
                stream_result["details"]["event"] = event_details
            if active_effort is not None:
                effort_after = _apply_gripper_effort(dc, gripper_dofs, active_effort)
                if not effort_after["supported"]:
                    raise RuntimeError(f"Gripper effort command failed after {segment.name}: {effort_after}")
            else:
                effort_after = None
            stream_result["details"]["gripper_effort_after_segment"] = effort_after
            if segment.phase_tag == "carry" and args.carry_stabilization_steps > 0:
                carry_stabilization_effort = _run_updates_with_optional_gripper_effort(
                    sim_app,
                    args.carry_stabilization_steps,
                    counter,
                    dc,
                    gripper_dofs,
                    active_effort,
                )
                metrics["snapshots"]["after_carry_stabilization"] = _bbox_state(stage, target_path)
            else:
                carry_stabilization_effort = None
            stream_result["details"]["carry_stabilization_steps"] = args.carry_stabilization_steps if segment.phase_tag == "carry" else 0
            stream_result["details"]["carry_stabilization_effort"] = carry_stabilization_effort
            metrics["micro_stop_frames"] += int(stream_result["micro_stop_estimate"]["micro_stop_frames"])
            metrics["micro_stop_samples"] += int(stream_result["micro_stop_estimate"]["micro_stop_samples"])
            metrics["snapshots"][f"after_{segment.name}"] = _bbox_state(stage, target_path)
            if segment.name == "experimental_pivot_anchor" and segment.details.get("experimental_pivot_arc"):
                upper_locked_targets = _capture_locked_joint_targets(dc, segment_dofs)
                pivot_reference_position = _body_pose_position(dc, segment_body)
                experimental_lock_state = {
                    "upper_locked_dofs": segment_dofs,
                    "upper_locked_targets": upper_locked_targets,
                    "pivot_reference_position": pivot_reference_position,
                    "pivot_position_after_anchor": pivot_reference_position,
                }
                stream_result["details"]["upper_chain_locked_targets"] = dict(upper_locked_targets)
                stream_result["details"]["upper_chain_lock_captured"] = True
                stream_result["details"]["upper_chain_lock_active_during_anchor"] = False
                stream_result["details"]["upper_chain_lock_active_during_arc"] = True
                stream_result["details"]["pivot_reference_position"] = pivot_reference_position.tolist()
                stream_result["details"]["pivot_position_after_anchor"] = pivot_reference_position.tolist()
            segment_summary = {
                "name": segment.name,
                "phase_tag": segment.phase_tag,
                "waypoint_type": segment.waypoint_type,
                "speed_profile": segment.speed_profile,
                "control_body_name": segment_body_name,
                "control_body_path": segment_body_path,
                "control_dof_names": [name for _, _, name in segment_dofs],
                "blend_radius": segment.blend_radius,
                "target_position": segment.target_position.tolist(),
                "actual_position": stream_result["actual_position"].tolist(),
                "position_error": stream_result["position_error"],
                "condition_met": bool(stream_result["condition_met"]),
                "contact_window": segment.contact_window,
                "event_marker": segment.event_marker,
                "micro_stop_estimate": stream_result["micro_stop_estimate"],
                "streaming_details": stream_result["details"],
            }
            if segment.details.get("experimental_pivot_arc"):
                segment_summary["experimental_details"] = stream_result["details"]
            metrics["segment_summaries"].append(segment_summary)
            print(
                f"phase={segment.name} streaming_controller=True "
                f"no_blocking_ik=True stream_samples={stream_sample_count} "
                f"frame_step_updates={args.stream_frame_step_updates} "
                f"position_error={stream_result['position_error']} event={segment.event_marker}"
            )
            continue

        if segment.details.get("locked_lower_chain_arc"):
            if experimental_lock_state is None:
                raise RuntimeError("Experimental lower-chain arc requested before upper-chain lock capture")
            pivot_body = segment.pinch_metric_pivot_body
            if pivot_body is None:
                raise RuntimeError("Experimental lower-chain arc missing pivot body for lock/drift monitoring")
            arc_waypoints = _interpolate_arc_waypoints(
                start_ee,
                np.array(segment.details["arc_mid_target"], dtype=float),
                np.array(segment.details["arc_contact_target"], dtype=float),
                int(segment.details.get("arc_waypoint_count", segment.max_ik_steps)),
            )
            segment.locked_dofs = experimental_lock_state["upper_locked_dofs"]
            segment.locked_targets = experimental_lock_state["upper_locked_targets"]
            metrics["snapshots"][f"before_{segment.name}"] = _bbox_state(stage, target_path)
            arc_result = _execute_locked_lower_chain_arc(
                target_index=target_index,
                target_path=target_path,
                stage=stage,
                dc=dc,
                articulation=articulation,
                sim_app=sim_app,
                counter=counter,
                lower_dofs=segment_dofs,
                lower_body=segment_body,
                lower_body_name=segment_body_name,
                lower_body_path=segment_body_path,
                pivot_body=pivot_body,
                upper_locked_dofs=experimental_lock_state["upper_locked_dofs"],
                upper_locked_targets=experimental_lock_state["upper_locked_targets"],
                pivot_reference_position=experimental_lock_state["pivot_reference_position"],
                arc_waypoints=arc_waypoints,
                ik_steps_per_waypoint=1,
                ik_settle_steps=args.ik_settle_steps,
                ik_position_eps=args.ik_position_eps,
                ik_damping=args.ik_damping,
                ik_max_step=args.ik_max_step,
                ik_posture_gain=args.ik_posture_gain,
                stop_tolerance=segment.stop_tolerance,
                frame_step_updates=args.pivot_arc_frame_step_updates,
                phase_label_prefix=segment.name,
                phase_log=phase_log,
                gripper_dofs=gripper_dofs,
                args=args,
                object_center=np.array(segment.details["pivot_arc_object_center"], dtype=float),
                pivot_to_pinch_distance=float(segment.details["pivot_to_pinch_distance_used"]),
                close_distance=float(segment.details["pivot_arc_close_distance_m"]),
                active_arm=str(segment.details.get("active_arm", "right")),
                robot_base_yaw_rad=float(segment.details.get("robot_base_yaw_rad", 0.0)),
            )
            if arc_result.get("active_effort") is not None:
                active_effort = arc_result["active_effort"]
            metrics["close_event_fired"] = bool(metrics["close_event_fired"] or arc_result["close_event_fired"])
            metrics["micro_stop_frames"] += int(arc_result["micro_stop_estimate"]["micro_stop_frames"])
            metrics["micro_stop_samples"] += int(arc_result["micro_stop_estimate"]["micro_stop_samples"])
            metrics["snapshots"].update(arc_result.get("snapshots", {}))
            metrics["snapshots"][f"after_{segment.name}"] = _bbox_state(stage, target_path)
            segment_summary = {
                "name": segment.name,
                "phase_tag": segment.phase_tag,
                "waypoint_type": segment.waypoint_type,
                "speed_profile": segment.speed_profile,
                "control_body_name": segment_body_name,
                "control_body_path": segment_body_path,
                "control_dof_names": [name for _, _, name in segment_dofs],
                "locked_dof_names": [name for _, _, name in experimental_lock_state["upper_locked_dofs"]],
                "blend_radius": segment.blend_radius,
                "target_position": segment.target_position.tolist(),
                "actual_position": arc_result["actual_position"].tolist(),
                "position_error": arc_result["position_error"],
                "condition_met": bool(arc_result["condition_met"]),
                "contact_window": segment.contact_window,
                "event_marker": segment.event_marker,
                "micro_stop_estimate": arc_result["micro_stop_estimate"],
                "experimental_details": arc_result["experimental_details"],
            }
            metrics["segment_summaries"].append(segment_summary)
            print(
                f"phase={segment.name} streaming_controller=True no_blocking_ik=True "
                f"stream_samples={int(segment.details.get('arc_waypoint_count', segment.max_ik_steps))} "
                f"frame_step_updates={args.pivot_arc_frame_step_updates} waypoint_type={segment.waypoint_type} "
                f"position_error={arc_result['position_error']} event={segment.event_marker}"
            )
            continue

    metrics["cycle_end_step"] = counter["step"]
    metrics["cycle_time_steps"] = counter["step"] - cycle_start_step
    return metrics


def _smooth_pause_steps(args: argparse.Namespace, phase_name: str) -> int:
    if not args.smooth_motion:
        return args.pause_steps
    return args.smooth_non_contact_pause_steps


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
    arm_bundles: dict[str, dict[str, Any]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    front_workspace_x: tuple[float, float],
    front_workspace_y: tuple[float, float],
    front_workspace_z: tuple[float, float],
    table_top_z: float,
    robot_base_position: list[float],
    robot_base_rotation: list[float],
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
    robot_base_np = np.array(robot_base_position, dtype=float)
    active_arm = "right" if float(initial_center[1]) >= float(robot_base_np[1]) else "left"
    fallback_arm = "left" if active_arm == "right" else "right"
    pregrasp_before_bias = np.array(
        [
            initial_state["bbox"]["center"][0],
            initial_state["bbox"]["center"][1],
            initial_state["bbox"]["max"][2] + args.pre_grasp_clearance,
        ],
        dtype=float,
    )
    pregrasp_before_bias[2] = float(np.clip(pregrasp_before_bias[2], *front_workspace_z))
    pregrasp_after_bias, initial_pullback_applied, initial_base_to_target_xy = _pregrasp_with_pullback(
        pregrasp_before_bias,
        robot_base_position,
        args.pregrasp_pullback_m,
        args.pregrasp_max_bias_m,
    )
    pregrasp_after_bias[2] = float(np.clip(pregrasp_after_bias[2], *front_workspace_z))
    pre_grasp_pose = {
        "position": pregrasp_after_bias.tolist(),
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
    pregrasp_candidates_by_arm: dict[str, list[dict[str, Any]]] = {
        arm_name: _build_pregrasp_candidates(
            pregrasp_before_bias=pregrasp_before_bias,
            pregrasp_after_bias=pregrasp_after_bias,
            robot_base_position=robot_base_position,
            active_arm=arm_name,
            args=args,
            front_workspace_z=front_workspace_z,
        )
        for arm_name in (active_arm, fallback_arm)
    }
    pregrasp_selection_log = {
        "active_arm": active_arm,
        "fallback_arm": fallback_arm,
        "fallback_triggered": False,
        "pregrasp_before_bias": pregrasp_before_bias.tolist(),
        "pregrasp_after_bias": pregrasp_after_bias.tolist(),
        "pullback_applied": float(initial_pullback_applied),
        "base_to_target_xy": {
            "dx": float(pregrasp_after_bias[0] - robot_base_np[0]),
            "dy": float(pregrasp_after_bias[1] - robot_base_np[1]),
            "distance_before_bias": float(initial_base_to_target_xy),
            "distance_after_bias": float(np.linalg.norm(pregrasp_after_bias[:2] - robot_base_np[:2])),
        },
        "candidate_count_by_arm": {arm_name: len(candidates) for arm_name, candidates in pregrasp_candidates_by_arm.items()},
        "selected_candidate": None,
    }
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
            "active_arm": active_arm,
            "pregrasp_selection": pregrasp_selection_log,
            "initial_pose": initial_state,
            "pre_grasp_pose": pre_grasp_pose,
            "pre_grasp_geometry": pre_grasp_geometry,
            "descend_pose": descend_pose,
            "bin_drop_pose": bin_drop_pose,
            "target_workspace_checks": target_workspace_checks,
            "pre_grasp_workspace_checks": pre_grasp_workspace_checks,
        }
    )
    arm_dofs = arm_bundles[active_arm]["arm_dofs"]
    gripper_dofs = arm_bundles[active_arm]["gripper_dofs"]
    end_effector_body = arm_bundles[active_arm]["end_effector_body"]

    try:
        if not target_workspace_checks["front_workspace_ok"]:
            _fail(
                "target_outside_front_workspace",
                "Selected target is outside the conservative front tabletop workspace",
            )

        selected_arm = active_arm
        selected_candidate = _select_pregrasp_candidate(
            target_index=target_index,
            active_arm=active_arm,
            fallback_triggered=False,
            candidates=pregrasp_candidates_by_arm[active_arm],
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_bundles[active_arm]["arm_dofs"],
            gripper_dofs=arm_bundles[active_arm]["gripper_dofs"],
            end_effector_body=arm_bundles[active_arm]["end_effector_body"],
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        if selected_candidate is None:
            pregrasp_selection_log["fallback_triggered"] = True
            selected_arm = fallback_arm
            selected_candidate = _select_pregrasp_candidate(
                target_index=target_index,
                active_arm=fallback_arm,
                fallback_triggered=True,
                candidates=pregrasp_candidates_by_arm[fallback_arm],
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_bundles[fallback_arm]["arm_dofs"],
                gripper_dofs=arm_bundles[fallback_arm]["gripper_dofs"],
                end_effector_body=arm_bundles[fallback_arm]["end_effector_body"],
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
            )
        if selected_candidate is None:
            _fail("pre_grasp_unreachable", "No deterministic pre-grasp candidate reached tolerance for either arm")

        pregrasp_selection_log["selected_arm"] = selected_arm
        pregrasp_selection_log["selected_candidate"] = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in selected_candidate.items()
            if key != "position"
        }
        pregrasp_selection_log["selected_candidate"]["position"] = np.array(selected_candidate["position"], dtype=float).tolist()
        result["selected_arm"] = selected_arm
        result["pregrasp_selection"] = pregrasp_selection_log
        arm_dofs = arm_bundles[selected_arm]["arm_dofs"]
        gripper_dofs = arm_bundles[selected_arm]["gripper_dofs"]
        end_effector_body = arm_bundles[selected_arm]["end_effector_body"]
        pre_grasp_target_np = np.array(selected_candidate["position"], dtype=float)
        pre_grasp_pose["position"] = pre_grasp_target_np.tolist()
        pre_grasp_workspace_checks = _front_workspace_checks(
            pre_grasp_target_np,
            front_workspace_x,
            front_workspace_y,
            front_workspace_z,
            table_top_z,
        )
        result["pre_grasp_pose"] = pre_grasp_pose
        result["pre_grasp_workspace_checks"] = pre_grasp_workspace_checks
        grasp_contact_geometry = _grasp_contact_geometry(
            selected_pregrasp_target=pre_grasp_target_np,
            object_center=initial_center,
            bbox_top_z=float(initial_state["bbox"]["max"][2]),
            table_top_z=table_top_z,
            args=args,
        )
        grasp_align_target_np = np.array(grasp_contact_geometry["pre_contact_alignment_target"], dtype=float)
        grasp_depth_target_np = np.array(grasp_contact_geometry["grasp_contact_target"], dtype=float)
        descend_pose["position"] = grasp_depth_target_np.tolist()
        descend_pose["selected_pregrasp_coupled_xy"] = True
        descend_pose["final_descend_vertical_only"] = grasp_contact_geometry["final_descend_vertical_only"]
        result["descend_pose"] = descend_pose
        result["grasp_contact_geometry"] = grasp_contact_geometry
        marker_paths.extend(
            [
                _debug_marker(stage, f"/World/DebugTask1SelectedPreGrasp_{target_index}", pre_grasp_target_np.tolist(), 0.025, (0.0, 0.7, 1.0)),
                _debug_marker(stage, f"/World/DebugTask1GraspContact_{target_index}", grasp_depth_target_np.tolist(), 0.025, (1.0, 0.9, 0.1)),
            ]
        )
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

        experimental_pivot_arc_log: dict[str, Any] = {
            "enabled": bool(args.experimental_pivot_arc_grasp),
            "fallback_to_baseline_used": False,
            "fallback_reason": None,
        }
        if args.experimental_pivot_arc_grasp:
            try:
                continuous_plan, experimental_details = _experimental_pivot_arc_plan(
                    target_index=target_index,
                    active_arm=selected_arm,
                    arm_dofs=arm_dofs,
                    end_effector_body=end_effector_body,
                    end_effector_name=arm_bundles[selected_arm]["end_effector_name"],
                    end_effector_path=arm_bundles[selected_arm]["end_effector_path"],
                    pivot_body=arm_bundles[selected_arm]["pivot_body"],
                    pivot_name=arm_bundles[selected_arm]["pivot_name"],
                    pivot_path=arm_bundles[selected_arm]["pivot_path"],
                    object_center=initial_center,
                    bbox_top_z=float(initial_state["bbox"]["max"][2]),
                    table_top_z=table_top_z,
                    robot_base_position=robot_base_position,
                    robot_base_rotation=robot_base_rotation,
                    pre_grasp_target=pre_grasp_target_np,
                    lift_clearance_target=lift_target_np,
                    prebin_target=prebin_target_np,
                    place_depth_target=place_depth_target_np,
                    retreat_target=retreat_target_np,
                    args=args,
                    front_workspace_z=front_workspace_z,
                )
                experimental_pivot_arc_log.update(experimental_details)
                marker_paths.extend(
                    [
                        _debug_marker(stage, f"/World/DebugTask1PivotAnchor_{target_index}", experimental_details["pivot_anchor_target"], 0.025, (0.9, 0.2, 1.0)),
                        _debug_marker(stage, f"/World/DebugTask1PivotContactPoint_{target_index}", experimental_details["target_object_contact_point"], 0.02, (0.2, 1.0, 0.3)),
                    ]
                )
            except Exception as exc:
                experimental_pivot_arc_log["fallback_to_baseline_used"] = True
                experimental_pivot_arc_log["fallback_reason"] = str(exc)
                continuous_plan = _build_continuous_cycle_plan(
                    active_arm=selected_arm,
                    robot_base_rotation=robot_base_rotation,
                    pre_grasp_target=pre_grasp_target_np,
                    grasp_align_target=grasp_align_target_np,
                    grasp_depth_target=grasp_depth_target_np,
                    lift_clearance_target=lift_target_np,
                    prebin_target=prebin_target_np,
                    place_depth_target=place_depth_target_np,
                    retreat_target=retreat_target_np,
                    args=args,
                )
        else:
            continuous_plan = _build_continuous_cycle_plan(
                active_arm=selected_arm,
                robot_base_rotation=robot_base_rotation,
                pre_grasp_target=pre_grasp_target_np,
                grasp_align_target=grasp_align_target_np,
                grasp_depth_target=grasp_depth_target_np,
                lift_clearance_target=lift_target_np,
                prebin_target=prebin_target_np,
                place_depth_target=place_depth_target_np,
                retreat_target=retreat_target_np,
                args=args,
            )
        result["experimental_pivot_arc_grasp"] = experimental_pivot_arc_log
        for segment in continuous_plan:
            if segment.name in {"continuous_grasp_align", "continuous_grasp_depth"}:
                segment.details.update(
                    {
                        "selected_pregrasp_target": grasp_contact_geometry["selected_pregrasp_target"],
                        "grasp_contact_target": grasp_contact_geometry["grasp_contact_target"],
                        "object_center": grasp_contact_geometry["object_center"],
                        "align_target_xy": grasp_contact_geometry["align_target_xy"],
                        "descend_target_xy": grasp_contact_geometry["descend_target_xy"],
                        "vertical_only_descend": grasp_contact_geometry["vertical_only_descend"],
                        "xy_delta_contact_to_object": grasp_contact_geometry["xy_delta_contact_to_object"],
                        "final_descend_vertical_only": grasp_contact_geometry["final_descend_vertical_only"],
                        "close_trigger_attached_to_final_vertical_descend": segment.name == "continuous_grasp_depth",
                    }
                )
        result["continuous_motion_plan"] = [
            {
                "name": segment.name,
                "phase_tag": segment.phase_tag,
                "waypoint_type": segment.waypoint_type,
                "speed_profile": segment.speed_profile,
                "blend_radius": segment.blend_radius,
                "event_marker": segment.event_marker,
                "control_body_name": segment.control_body_name,
                "control_body_path": segment.control_body_path,
                "control_dof_names": [name for _, _, name in segment.control_dofs] if segment.control_dofs else [name for _, _, name in arm_dofs],
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
        if args.experimental_pivot_arc_grasp and not experimental_pivot_arc_log.get("fallback_to_baseline_used"):
            segment_by_name_for_experiment = {item["name"]: item for item in cycle_metrics["segment_summaries"]}
            anchor_summary = segment_by_name_for_experiment.get("experimental_pivot_anchor", {})
            contact_summary = segment_by_name_for_experiment.get("experimental_locked_lower_chain_arc", {})
            close_metric = (contact_summary.get("experimental_details") or {}).get("close_condition_metric")
            experimental_pivot_arc_log["stage_a_succeeded"] = bool(anchor_summary.get("condition_met"))
            experimental_pivot_arc_log["stage_b_started"] = "experimental_locked_lower_chain_arc" in segment_by_name_for_experiment
            experimental_pivot_arc_log["close_condition_satisfied_before_close"] = bool(
                (contact_summary.get("experimental_details") or {}).get("close_condition_satisfied_before_close")
            )
            experimental_details = contact_summary.get("experimental_details") or {}
            for key in (
                "upper_chain_locked_targets",
                "upper_chain_lock_active",
                "upper_chain_lock_active_during_arc",
                "upper_chain_joint_error_max_during_arc",
                "upper_chain_joint_error_mean_during_arc",
                "upper_chain_joint_error_final_during_arc",
                "pivot_reference_position",
                "pivot_position_after_anchor",
                "pivot_position_during_arc_start",
                "pivot_position_during_arc_end",
                "pivot_drift_per_waypoint",
                "pivot_drift_norm_max",
                "pivot_drift_norm_mean",
                "pivot_drift_norm_final",
                "micro_stop_frames",
                "micro_stop_samples",
            ):
                if key in experimental_details:
                    experimental_pivot_arc_log[key] = experimental_details[key]
            if close_metric:
                experimental_pivot_arc_log["pinch_center_estimated_position"] = close_metric.get("pinch_center_estimated_position")
                experimental_pivot_arc_log["pinch_center_to_object_distance"] = close_metric.get("pinch_center_to_object_distance")
                wrist_dist = close_metric.get("wrist_to_object_distance")
                pinch_dist = close_metric.get("pinch_center_to_object_distance")
                if isinstance(wrist_dist, (int, float)) and isinstance(pinch_dist, (int, float)):
                    experimental_pivot_arc_log["object_under_wrist_or_near_finger_gap"] = (
                        "near_finger_gap" if pinch_dist < wrist_dist else "under_wrist_or_palm_projection"
                    )
            experimental_pivot_arc_log["final_local_approach_shape"] = (
                experimental_details.get("final_local_approach_shape") or "arc-like_locked_lower_chain"
            )
            result["experimental_pivot_arc_grasp"] = experimental_pivot_arc_log

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

        if "continuous_grasp_depth" in segment_by_name:
            descend_segment_name = "continuous_grasp_depth"
        elif "experimental_locked_lower_chain_arc" in segment_by_name:
            descend_segment_name = "experimental_locked_lower_chain_arc"
        else:
            descend_segment_name = "experimental_pinch_contact"
        ee_descend = np.array(segment_by_name[descend_segment_name]["actual_position"], dtype=float)
        descend_error = float(segment_by_name[descend_segment_name]["position_error"])
        after_descend_key = f"after_{descend_segment_name}"
        after_descend = cycle_metrics["snapshots"].get("before_close_gripper") or cycle_metrics["snapshots"][after_descend_key]
        experimental_close_metric = (segment_by_name[descend_segment_name].get("experimental_details") or {}).get("close_condition_metric")
        if experimental_close_metric and isinstance(experimental_close_metric.get("pinch_center_to_object_distance"), (int, float)):
            object_distance_before_close = float(experimental_close_metric["pinch_center_to_object_distance"])
        else:
            object_distance_before_close = _distance(ee_descend, _center_from_bbox(after_descend["bbox"]))
        descend_safety = _ee_front_safety_checks(ee_descend, front_workspace_x, front_workspace_y, table_top_z, 0.0)
        if "continuous_grasp_align" in phase_by_name and "continuous_grasp_align" in segment_by_name:
            align_summary = segment_by_name["continuous_grasp_align"]
            phase_by_name["continuous_grasp_align"]["details"]["ee_to_grasp_alignment_target_distance"] = float(
                align_summary["position_error"]
            )
            phase_by_name["continuous_grasp_align"]["details"]["close_trigger_position"] = None
        if descend_segment_name in phase_by_name:
            phase_by_name[descend_segment_name]["details"]["target_object_distance_before_close"] = object_distance_before_close
            phase_by_name[descend_segment_name]["details"]["ee_to_explicit_descend_target_distance"] = descend_error
            phase_by_name[descend_segment_name]["details"]["descend_front_safety_checks"] = descend_safety
            phase_by_name[descend_segment_name]["details"]["close_trigger_position"] = ee_descend.tolist()
            phase_by_name[descend_segment_name]["condition_met"] = bool(
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
    parser.add_argument("--startup-target-track-steps", type=int, default=DEFAULT_STARTUP_TARGET_TRACK_STEPS)
    parser.add_argument("--strict-startup-pose", action="store_true")
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-suffix")
    parser.add_argument("--gripper-delta", type=float, default=DEFAULT_GRIPPER_DELTA, help="Deprecated; official fixed gripper widths are used.")
    parser.add_argument("--gripper-hold-effort", type=float, default=DEFAULT_GRIPPER_HOLD_EFFORT)
    parser.add_argument("--pre-grasp-clearance", type=float, default=DEFAULT_PRE_GRASP_CLEARANCE)
    parser.add_argument("--pregrasp-pullback-m", type=float, default=0.0)
    parser.add_argument("--pregrasp-max-bias-m", type=float, default=0.06)
    parser.add_argument("--grasp-contact-offset-x", type=float, default=DEFAULT_GRASP_CONTACT_OFFSET_X)
    parser.add_argument("--grasp-contact-offset-y", type=float, default=DEFAULT_GRASP_CONTACT_OFFSET_Y)
    parser.add_argument("--experimental-pivot-arc-grasp", action="store_true")
    parser.add_argument("--pivot-to-pinch-distance-m", type=float, default=DEFAULT_PIVOT_TO_PINCH_DISTANCE)
    parser.add_argument("--pivot-anchor-height-offset-m", type=float, default=DEFAULT_PIVOT_ANCHOR_HEIGHT_OFFSET)
    parser.add_argument("--pivot-anchor-forward-offset-m", type=float, default=DEFAULT_PIVOT_ANCHOR_FORWARD_OFFSET)
    parser.add_argument("--pivot-anchor-lateral-offset-m", type=float, default=DEFAULT_PIVOT_ANCHOR_LATERAL_OFFSET)
    parser.add_argument("--pivot-arc-contact-tolerance-m", type=float, default=DEFAULT_PIVOT_ARC_CONTACT_TOLERANCE)
    parser.add_argument("--pivot-arc-max-steps", type=int, default=DEFAULT_PIVOT_ARC_MAX_STEPS)
    parser.add_argument("--pivot-arc-close-distance-m", type=float, default=DEFAULT_PIVOT_ARC_CLOSE_DISTANCE)
    parser.add_argument("--pivot-arc-frame-step-updates", type=int, default=DEFAULT_PIVOT_ARC_FRAME_STEP_UPDATES)
    parser.add_argument("--pivot-arc-stream-samples", type=int, default=DEFAULT_PIVOT_ARC_STREAM_SAMPLES)
    parser.add_argument("--stream-samples", type=int, default=DEFAULT_STREAM_SAMPLES)
    parser.add_argument("--stream-frame-step-updates", type=int, default=DEFAULT_STREAM_FRAME_STEP_UPDATES)
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
    parser.add_argument("--ik-hold-steps", type=int, default=0)
    parser.add_argument("--smooth-motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-ik-hold-steps", type=int, default=0)
    parser.add_argument("--smooth-non-contact-pause-steps", type=int, default=0)
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

    if args.init_steps < 1 or args.phase_steps < 1 or args.pause_steps < 0 or args.settle_steps < 1:
        raise RuntimeError("--init-steps, --phase-steps, and --settle-steps must be positive; --pause-steps must be non-negative")
    if args.startup_target_track_steps < 1:
        raise RuntimeError("--startup-target-track-steps must be positive")
    if args.target_index is not None and args.target_index < 0:
        raise RuntimeError("--target-index must be non-negative")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be positive")
    if args.pre_grasp_clearance <= 0.0 or args.safe_drop_height <= 0.0:
        raise RuntimeError("--pre-grasp-clearance and --safe-drop-height must be positive")
    if args.pregrasp_pullback_m < 0.0 or args.pregrasp_max_bias_m < 0.0:
        raise RuntimeError("--pregrasp-pullback-m and --pregrasp-max-bias-m must be non-negative")
    if args.pivot_to_pinch_distance_m <= 0.0:
        raise RuntimeError("--pivot-to-pinch-distance-m must be positive")
    if args.pivot_arc_contact_tolerance_m <= 0.0 or args.pivot_arc_close_distance_m <= 0.0:
        raise RuntimeError("--pivot-arc-contact-tolerance-m and --pivot-arc-close-distance-m must be positive")
    if args.pivot_arc_max_steps < 1:
        raise RuntimeError("--pivot-arc-max-steps must be positive")
    if args.pivot_arc_frame_step_updates < 1:
        raise RuntimeError("--pivot-arc-frame-step-updates must be positive")
    if args.pivot_arc_stream_samples < 1:
        raise RuntimeError("--pivot-arc-stream-samples must be positive")
    if args.stream_samples < 1 or args.stream_frame_step_updates < 1:
        raise RuntimeError("--stream-samples and --stream-frame-step-updates must be positive")
    if args.stable_jitter <= 0.0 or args.min_lift_delta <= 0.0 or args.min_transport_distance <= 0.0:
        raise RuntimeError("--stable-jitter, --min-lift-delta, and --min-transport-distance must be positive")
    if args.joint_tolerance <= 0.0 or args.max_local_joint_adjustment < 0.0:
        raise RuntimeError("--joint-tolerance must be positive and --max-local-joint-adjustment must be non-negative")
    if args.min_ee_table_clearance < 0.0:
        raise RuntimeError("--min-ee-table-clearance must be non-negative")
    if args.ik_steps < 1 or args.ik_settle_steps < 1 or args.ik_hold_steps < 0:
        raise RuntimeError("--ik-steps and --ik-settle-steps must be positive; --ik-hold-steps must be non-negative")
    if args.smooth_ik_hold_steps < 0 or args.smooth_non_contact_pause_steps < 0:
        raise RuntimeError("--smooth-ik-hold-steps and --smooth-non-contact-pause-steps must be non-negative")
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
            "startup_target_track_steps": args.startup_target_track_steps,
            "strict_startup_pose": bool(args.strict_startup_pose),
            "gripper_policy": "official_same_sign_position_targets",
            "gripper_effort_policy": "dynamic_control_set_dof_effort_sustained_during_close_validation_lift",
            "gripper_hold_effort": args.gripper_hold_effort,
            "official_gripper_open_width": OFFICIAL_GRIPPER_OPEN_WIDTH,
            "official_gripper_close_width": OFFICIAL_GRIPPER_CLOSE_WIDTH,
            "descend_clearance": args.descend_clearance,
            "grasp_depth_offset": args.grasp_depth_offset,
            "pregrasp_pullback_m": args.pregrasp_pullback_m,
            "pregrasp_max_bias_m": args.pregrasp_max_bias_m,
            "pregrasp_candidate_policy": "arm_aware_greedy_center_pullback2cm_pullback4cm_z_offsets_max_6_with_one_fallback_arm",
            "grasp_contact_offset_x": args.grasp_contact_offset_x,
            "grasp_contact_offset_y": args.grasp_contact_offset_y,
            "grasp_contact_policy": "selected_pregrasp_xy_plus_contact_offset_then_vertical_descend",
            "tuning_knobs": _tuning_knob_summary(args),
            "smooth_motion": bool(args.smooth_motion),
            "smooth_non_contact_pause_steps": args.smooth_non_contact_pause_steps,
            "smooth_ik_hold_steps": args.smooth_ik_hold_steps,
            "continuous_cycle_policy": "global_absolute_joint_waypoint_cubic_hermite_stream_no_per_segment_endpoint_rebuild",
            "streaming_controller_active": True,
            "ordinary_cycle_endpoint_rebuilds": 0,
            "ordinary_cycle_finite_difference_jacobian_calls": 0,
            "stream_samples": args.stream_samples,
            "stream_frame_step_updates": args.stream_frame_step_updates,
            "pivot_arc_stream_samples": args.pivot_arc_stream_samples,
            "pivot_arc_frame_step_updates": args.pivot_arc_frame_step_updates,
            "interpolation_profile": "global_quintic_timewarp_cubic_hermite_joint_chain",
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
            "open_gripper_initial_right",
            "open_gripper_initial_left",
            "pregrasp_candidate_selection",
            "continuous_pregrasp",
            "continuous_grasp_align",
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
        articulation_startup_updates = max(args.pause_steps, 5)
        _run_updates(sim_app, articulation_startup_updates, counter)

        dc, articulation, articulation_acquire_log = _acquire_articulation_with_fallback(articulation_path, args.prim_path)
        dof_observation = _read_dof_observation(dc, articulation)
        arm_bundles: dict[str, dict[str, Any]] = {}
        for arm_name in ("right", "left"):
            end_effector_body_for_arm, end_effector_name_for_arm, end_effector_path_for_arm = _identify_end_effector_body_for_side(
                dc,
                articulation,
                arm_name,
                args.end_effector_body,
            )
            pivot_body_for_arm, pivot_name_for_arm, pivot_path_for_arm = _find_body_for_side(
                dc,
                articulation,
                arm_name,
                "wrist_pitch_link",
            )
            arm_bundles[arm_name] = {
                "arm_dofs": _select_arm_dofs_for_side(dc, articulation, arm_name, args.max_arm_dofs),
                "gripper_dofs": _select_gripper_dofs_for_side(dc, articulation, arm_name),
                "end_effector_body": end_effector_body_for_arm,
                "end_effector_name": end_effector_name_for_arm,
                "end_effector_path": end_effector_path_for_arm,
                "pivot_body": pivot_body_for_arm,
                "pivot_name": pivot_name_for_arm,
                "pivot_path": pivot_path_for_arm,
            }
        arm_dofs = arm_bundles["right"]["arm_dofs"]
        gripper_dofs = arm_bundles["right"]["gripper_dofs"]
        end_effector_body = arm_bundles["right"]["end_effector_body"]
        end_effector_name = arm_bundles["right"]["end_effector_name"]
        end_effector_path = arm_bundles["right"]["end_effector_path"]
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
            "arm_dof_names_by_arm": {arm_name: [name for _, _, name in bundle["arm_dofs"]] for arm_name, bundle in arm_bundles.items()},
            "experimental_pivot_frame_by_arm": {arm_name: bundle["pivot_name"] for arm_name, bundle in arm_bundles.items()},
            "experimental_pivot_frame_path_by_arm": {arm_name: bundle["pivot_path"] for arm_name, bundle in arm_bundles.items()},
            "gripper_dof_indices_by_arm": {arm_name: [index for index, _, _ in bundle["gripper_dofs"]] for arm_name, bundle in arm_bundles.items()},
            "gripper_dof_names_by_arm": {arm_name: [name for _, _, name in bundle["gripper_dofs"]] for arm_name, bundle in arm_bundles.items()},
            "gripper_hold_effort": args.gripper_hold_effort,
            "official_startup_dof_names": [name for _, _, name in startup_dofs],
            "missing_optional_official_startup_dofs": missing_official_startup_optional_dofs,
            "official_startup_source": "lerobot.common.robot_devices.robots.isaac_sim_robot_interface.IsaacSimRobotInterface._joint_value_map",
            "official_startup_baseline_source": "Ubtech_sim/source/RobotArticulation.py uses the same _joint_value_map for initialization",
            "startup_joint_seed_policy": "initialization_only_direct_joint_seed_from_official_baseline_then_target_streaming",
            "end_effector_by_arm": {
                arm_name: {
                    "name": bundle["end_effector_name"],
                    "path": bundle["end_effector_path"],
                }
                for arm_name, bundle in arm_bundles.items()
            },
            "end_effector_name": end_effector_name,
            "end_effector_path": end_effector_path,
            "dof_observation_sample": dof_observation[:12],
            "articulation_acquire": articulation_acquire_log,
            "articulation_startup_updates": articulation_startup_updates,
        }

        phase_log: list[dict[str, Any]] = payload["phase_log"]
        startup_start_step = counter["step"]
        startup_seed_result = _seed_joint_positions_for_initialization(dc, startup_dofs, official_startup_targets)
        startup_sync_steps = args.startup_target_track_steps
        _run_updates(sim_app, startup_sync_steps, counter)
        observed_startup = _current_positions(dc, startup_dofs)
        ee_startup = _body_pose_position(dc, end_effector_body)
        startup_max_error = max(abs(float(obs - target)) for obs, target in zip(observed_startup, official_startup_targets))
        startup_ok = bool(startup_seed_result["supported"] and startup_max_error <= args.joint_tolerance)
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
                "initialization_seed_result": startup_seed_result,
                "post_seed_sync_steps": startup_sync_steps,
                "strict_startup_pose": bool(args.strict_startup_pose),
                "startup_pose_best_effort": not bool(args.strict_startup_pose),
            },
        )
        print(
            "phase=apply_official_startup_pose "
            f"condition_met={startup_ok} "
            f"initialization_seed_supported={startup_seed_result['supported']} "
            f"max_joint_error={startup_max_error} ee={ee_startup.tolist()}"
        )
        if not startup_ok and args.strict_startup_pose:
            _fail("official_startup_pose_failed", "Official startup joint pose was not reached within tolerance")
        if not startup_ok:
            print(
                "phase=apply_official_startup_pose warning=initialization_startup_pose_not_within_tolerance "
                f"max_joint_error={startup_max_error} continuing_best_effort=True"
            )

        for arm_name, bundle in arm_bundles.items():
            open_gripper_dofs = bundle["gripper_dofs"]
            open_ok = _command_gripper_phase(
                f"open_gripper_initial_{arm_name}",
                dc,
                open_gripper_dofs,
                bundle["end_effector_body"],
                [OFFICIAL_GRIPPER_OPEN_WIDTH] * len(open_gripper_dofs),
                sim_app,
                _smooth_pause_steps(args, "open_gripper_initial"),
                counter,
                phase_log,
                effort_value=0.0,
            )
            if not open_ok:
                _fail("gripper_command_failed", f"{arm_name} gripper open command failed before approach")

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
                arm_bundles=arm_bundles,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                front_workspace_x=front_workspace_x,
                front_workspace_y=front_workspace_y,
                front_workspace_z=front_workspace_z,
                table_top_z=table_top_z,
                robot_base_position=configured_robot_position,
                robot_base_rotation=configured_robot_rotation,
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
