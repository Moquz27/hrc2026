#!/usr/bin/env python3
"""Task 1 DualArmIK no-gate diagnostic duplicate for Walker S2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from load_walker_s2 import DEFAULT_INIT_STEPS, _create_minimal_scene, _find_joint_names, _load_simulation_app, _validate_environment  # type: ignore
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


SCRIPT_NAME = "task1_dualarmik_phase_nogate.py"
LOG_STEM = "task1_dualarmik_phase_nogate"

OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"
OFFICIAL_GRIPPER_OPEN_WIDTH = -0.0215
OFFICIAL_GRIPPER_CLOSE_WIDTH = 0.01
DEFAULT_GRIPPER_HOLD_EFFORT = 100.0

TCP_OFFSET_FALLBACK_X = 0.155
TCP_OFFSET_EPS = 1.0e-4
OFFICIAL_TORSO_COMPENSATION_MATRIX = np.array(
    [
        [9.99999e-01, -1.11400e-03, 1.16200e-03, -9.64000e-04],
        [-2.00000e-05, 7.13609e-01, 7.00544e-01, -9.59927e-01],
        [-1.61000e-03, -7.00544e-01, 7.13608e-01, 6.56540e-01],
        [0.00000e00, 0.00000e00, 0.00000e00, 1.00000e00],
    ],
    dtype=float,
)


def _preset_euler_xyz_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _preset_rot_to_euler_xyz(rot: np.ndarray) -> list[float]:
    r = np.array(rot, dtype=float)
    if abs(float(r[2, 0])) < 1.0 - 1.0e-9:
        pitch = math.asin(-float(r[2, 0]))
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        pitch = math.pi / 2.0 if float(r[2, 0]) <= -1.0 else -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return [float(roll), float(pitch), float(yaw)]


def _preset_with_local_ab_axis_roll(rpy_values: list[float], axial_roll_rad: float) -> list[float]:
    base_rot = _preset_euler_xyz_to_rot(float(rpy_values[0]), float(rpy_values[1]), float(rpy_values[2]))
    local_ab_roll = _preset_euler_xyz_to_rot(0.0, 0.0, float(axial_roll_rad))
    return _preset_rot_to_euler_xyz(base_rot @ local_ab_roll)


def _build_world_y_approach_presets(base_presets: list[tuple[str, list[float]]]) -> list[tuple[str, list[float]]]:
    presets: list[tuple[str, list[float]]] = []
    for label, rpy_values in base_presets:
        for variant_label, axial_roll_rad in WORLD_Y_APPROACH_AB_ROLL_VARIANTS:
            if abs(float(axial_roll_rad)) <= 1.0e-12:
                rolled_rpy = [float(value) for value in rpy_values]
            else:
                rolled_rpy = _preset_with_local_ab_axis_roll(rpy_values, float(axial_roll_rad))
            presets.append((f"{label}_{variant_label}", rolled_rpy))
    return presets


RIGHT_Z_APPROACH_PRESETS = [
    ("right_z_approach_straight", [0.0, math.pi, math.pi]),
    ("right_z_approach_tilted_forward", [0.0, 2.80, math.pi]),
]

LEFT_Z_APPROACH_PRESETS = [
    ("left_z_approach_straight", [0.0, math.pi, math.pi]),
    ("left_z_approach_tilted_forward", [0.0, 2.80, math.pi]),
]

# First-test calibration presets. Current FAR diagnostics showed the older
# world_y_approach family produced approach_axis_world near world +X, so this
# family applies a yaw quarter-turn to move the local +Z/AB axis toward base
# +/-X, which should map to world +/-Y for the observed robot/world transform.
RIGHT_WORLD_Y_APPROACH_BASE_PRESETS = [
    ("right_world_y_approach_pos_y_yaw_plus_quarter", [0.5 * math.pi, 0.0, 0.5 * math.pi]),
    ("right_world_y_approach_neg_y_yaw_minus_quarter", [0.5 * math.pi, 0.0, -0.5 * math.pi]),
    ("right_world_y_approach_pos_y_roll_neg_yaw_minus_quarter", [-0.5 * math.pi, 0.0, -0.5 * math.pi]),
    ("right_world_y_approach_neg_y_roll_neg_yaw_plus_quarter", [-0.5 * math.pi, 0.0, 0.5 * math.pi]),
]

LEFT_WORLD_Y_APPROACH_BASE_PRESETS = [
    ("left_world_y_approach_pos_y_yaw_plus_quarter", [0.5 * math.pi, 0.0, 0.5 * math.pi]),
    ("left_world_y_approach_neg_y_yaw_minus_quarter", [0.5 * math.pi, 0.0, -0.5 * math.pi]),
    ("left_world_y_approach_pos_y_roll_neg_yaw_minus_quarter", [-0.5 * math.pi, 0.0, -0.5 * math.pi]),
    ("left_world_y_approach_neg_y_roll_neg_yaw_plus_quarter", [-0.5 * math.pi, 0.0, 0.5 * math.pi]),
]

WORLD_Y_APPROACH_AB_ROLL_VARIANTS = [
    ("ab_roll_plus_quarter_palm_down_test", 0.5 * math.pi),
    ("ab_roll_minus_quarter_palm_down_test", -0.5 * math.pi),
    ("ab_roll_none_reference", 0.0),
]

RIGHT_WORLD_Y_APPROACH_PRESETS = _build_world_y_approach_presets(RIGHT_WORLD_Y_APPROACH_BASE_PRESETS)
LEFT_WORLD_Y_APPROACH_PRESETS = _build_world_y_approach_presets(LEFT_WORLD_Y_APPROACH_BASE_PRESETS)

DEFAULT_FAR_THRESHOLD = 0.42
DEFAULT_NEAR_BODY_THRESHOLD = 0.28
DEFAULT_FAR_LOW_SIDE_CLEARANCE = 0.002
DEFAULT_FAR_POINT_B_GAP_ABOVE_SUPPORT = 0.002
DEFAULT_FAR_LOW_SIDE_GAP_ABOVE_SUPPORT = DEFAULT_FAR_POINT_B_GAP_ABOVE_SUPPORT
DEFAULT_FAR_XY_ALIGN_CLEARANCE_ABOVE_OBJECT = 0.035
DEFAULT_FAR_POINT_B_FORWARD_EXTENSION = 0.012
DEFAULT_FAR_POINT_A_EXTRA_HEIGHT_CLEARANCE = 0.018
DEFAULT_FAR_AB_DOWNWARD_SLANT_DEG = 8.0
DEFAULT_FAR_OUTBOARD_TRANSITION_OFFSET = 0.12
DEFAULT_FAR_OUTBOARD_TRANSITION_CLEARANCE = 0.08
DEFAULT_FAR_NULL_WEIGHT = 0.08
DEFAULT_FAR_OUTBOARD_SHOULDER_ROLL_BIAS = 0.35
DEFAULT_PRE_CLOSE_POINT_B_TOLERANCE = 0.005
DEFAULT_LIVE_COORDINATE_TRANSFORM = True
DEFAULT_FAR_CANDIDATE_POSITION_TOLERANCE = 0.07
DEFAULT_FAR_CANDIDATE_ROTATION_TOLERANCE = 0.28
DEFAULT_VERTICAL_POINT_B_GAP_ABOVE_SUPPORT = 0.001
DEFAULT_VERTICAL_CLOSE_POINT_B_TOLERANCE = 0.005
DEFAULT_VERTICAL_XY_REFERENCE_LINK = "finger_midpoint"
DEFAULT_VERTICAL_XY_REFERENCE_TOLERANCE = 0.008
DEFAULT_VERTICAL_ARM_LATERAL_BIAS_CORRECTION = 0.04
DEFAULT_NOGATE_VERTICAL_CONTINUOUS_IK_DESCEND = True
DEFAULT_NOGATE_VERTICAL_IK_REFRESH_PERIOD = 10
DEFAULT_NOGATE_VERTICAL_DESCEND_STEP_Z = 0.010
DEFAULT_NOGATE_CLOSE_DLS_ENABLE = True
DEFAULT_NOGATE_CLOSE_DLS_SWITCH_DISTANCE = 0.04
DEFAULT_NOGATE_CLOSE_DLS_MAX_ITERS = 18
DEFAULT_NOGATE_CLOSE_DLS_EPS = 0.006
DEFAULT_NOGATE_CLOSE_DLS_DAMPING = 0.05
DEFAULT_NOGATE_CLOSE_DLS_MAX_STEP = 0.012
DEFAULT_NOGATE_CLOSE_DLS_MAX_ABS_JOINT_STEP = 0.006
DEFAULT_NOGATE_CLOSE_DLS_BLEND = 0.45
DEFAULT_NOGATE_CLOSE_DLS_SETTLE_STEPS = 1
DEFAULT_NOGATE_CLOSE_DLS_HOLD_STEPS = 0
DEFAULT_NOGATE_CLOSE_DLS_STOP_TOLERANCE = 0.004
DEFAULT_NOGATE_CLOSE_DLS_POSTURE_GAIN = 0.015
DEFAULT_NOGATE_TOUCH_MAX_TICKS = 360
DEFAULT_NOGATE_TOUCH_FIX_TICKS = 30
DEFAULT_NOGATE_TOUCH_STEP_Z = 0.002
DEFAULT_NOGATE_TOUCH_GAP_ABOVE_TABLE = 0.001
DEFAULT_NOGATE_TOUCH_XY_TOLERANCE = 0.008
DEFAULT_NOGATE_TOUCH_OBJECT_EXPAND = 0.012
DEFAULT_NOGATE_TOUCH_OBJECT_MOTION_THRESHOLD = 0.002
DEFAULT_NOGATE_TOUCH_STALL_TOLERANCE = 0.0005
DEFAULT_NOGATE_POST_TOUCH_REPOSITION_TICKS = 120
DEFAULT_NOGATE_FINGER_CAPTURE_TOLERANCE = 0.025
DEFAULT_NOGATE_FINGER_CAPTURE_SEGMENT_MARGIN = 0.15
DEFAULT_NOGATE_REQUIRE_TABLE_TOUCH_BEFORE_CLOSE = True
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_ENABLE = True
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_HEIGHT = 0.010
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_TICKS = 90
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_BLEND = 0.12
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_MAX_STEP_NORM = 0.015
DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_MAX_ABS_JOINT_STEP = 0.008
VERTICAL_FINGER_MIDPOINT_REFERENCE_ALIASES = {
    "finger_midpoint",
    "finger_pair_midpoint",
    "finger1_finger2_midpoint",
    "finger1_link_finger2_link_midpoint",
    "finger1_link+finger2_link",
}

DEFAULT_SETTLE_STEPS = 240
DEFAULT_GRIPPER_STEPS = 24
DEFAULT_SERVO_MAX_TICKS = 180
DEFAULT_SERVO_CARRY_TICKS = 240
DEFAULT_SERVO_BLEND = 0.35
DEFAULT_SERVO_MAX_STEP_NORM = 0.035
DEFAULT_SERVO_MAX_ABS_JOINT_STEP = 0.025
DEFAULT_IK_MAX_ITER = 90
DEFAULT_IK_POS_TOL = 0.004
DEFAULT_IK_ROT_TOL = 0.08
DEFAULT_IK_DAMPING = 1.0e-4
DEFAULT_IK_DQ_MAX = 0.35
DEFAULT_IK_NULL_WEIGHT = 0.0
DEFAULT_IK_REFRESH_ENABLE = True
DEFAULT_IK_REFRESH_PERIOD = 12
DEFAULT_IK_REFRESH_DRIFT_THRESHOLD = 0.0
DEFAULT_CANDIDATE_IK_MAX_ITER = 120
DEFAULT_CANDIDATE_IK_POS_TOL = 0.03
DEFAULT_CANDIDATE_IK_ROT_TOL = 0.18
EE_FRAME_DELTA_TRANSLATION_MATCH_TOL = 0.02
EE_FRAME_DELTA_ROTATION_MATCH_TOL = 0.05
EE_FRAME_OFFSET_MIN_TRANSLATION = 0.03

DEFAULT_PREGRASP_TOLERANCE = 0.045
DEFAULT_ALIGN_TOLERANCE = 0.04
DEFAULT_DESCEND_TOLERANCE = 0.03
DEFAULT_LIFT_TOLERANCE = 0.055
DEFAULT_CARRY_TOLERANCE = 0.08
DEFAULT_PLACE_TOLERANCE = 0.065
DEFAULT_RETREAT_TOLERANCE = 0.08
DEFAULT_ROT_TOLERANCE = 0.15

DEFAULT_PREGRASP_CLEARANCE = 0.20  
DEFAULT_PREGRASP_STANDOFF = 0.12
DEFAULT_ALIGN_CLEARANCE = 0.10
DEFAULT_DESCEND_CLEARANCE = 0.005
DEFAULT_MICRO_LIFT_PROBE = 0.018
DEFAULT_MICRO_LIFT_MIN_DELTA = 0.010
DEFAULT_LIFT_HEIGHT = 0.17
DEFAULT_SAFE_DROP_HEIGHT = 0.12
DEFAULT_PLACE_CLEARANCE = 0.055
DEFAULT_RETREAT_LIFT = 0.16
DEFAULT_MIN_TRANSPORT_DISTANCE = 0.08
DEFAULT_STABLE_JITTER = 0.01
DEFAULT_MIN_EE_TABLE_CLEARANCE = 0.025

DEFAULT_WORKSPACE_X = (0.25, 1.60)
DEFAULT_WORKSPACE_Y = (-0.80, 0.85)
DEFAULT_WORKSPACE_Z = (0.50, 1.35)

RIGHT_GRIPPER_TOKENS = ("r_finger", "right_finger", "r_thumb", "right_thumb", "r_gripper", "right_gripper")
LEFT_GRIPPER_TOKENS = ("l_finger", "left_finger", "l_thumb", "left_thumb", "l_gripper", "left_gripper")
RIGHT_EE_TOKENS = ("r_sixforce_link", "r_hand", "right_hand", "r_palm", "right_palm", "r_wrist", "right_wrist")
LEFT_EE_TOKENS = ("l_sixforce_link", "l_hand", "left_hand", "l_palm", "left_palm", "l_wrist", "left_wrist")
EE_EXCLUDE_TOKENS = ("finger", "thumb", "camera")

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

PHASE_ORDER = [
    "startup_official_pose",
    "open_gripper_initial",
    "select_target",
    "plan_grasp_geometry",
    "select_pregrasp_candidate",
    "servo_pregrasp",
    "servo_align",
    "servo_descend",
    "far_outboard_transition",
    "far_prepare_low_side_approach",
    "far_align_B_over_object_xy",
    "far_lower_B_world_z",
    "mid_align_AB_vertical_over_object",
    "mid_pre_descend_AB_vertical",
    "mid_descend_world_z_keep_AB_vertical",
    "close_gripper",
    "micro_lift_probe",
    "far_lift",
    "mid_lift",
    "servo_lift",
    "motion_policy_stop_after_lift",
    "servo_carry",
    "servo_place",
    "open_gripper",
    "servo_retreat",
    "settle_and_score",
]


class RunFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ServoSpec:
    name: str
    target_pose_base: np.ndarray
    pos_tolerance: float
    rot_tolerance: float
    max_ticks: int
    gripper_effort: float | None = None


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


NO_GATE_MODE = True
NO_GATE_FATAL_REASONS = {
    "scene_build_failed",
    "no_target_parts_found",
    "target_index_out_of_range",
}
NO_GATE_BYPASSED_FAILURES: list[dict[str, str]] = []


def _fail(reason: str, message: str) -> None:
    if NO_GATE_MODE and reason not in NO_GATE_FATAL_REASONS:
        NO_GATE_BYPASSED_FAILURES.append({"reason": str(reason), "message": str(message)})
        print(f"nogate_bypass reason={reason} message={message}")
        return
    raise RunFailure(reason, message)


def _add_reference(stage: Any, prim_path: str, usd_path: Path) -> Any:
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.GetReferences().AddReference(str(usd_path)):
        raise RuntimeError(f"Could not add reference {usd_path} at {prim_path}")
    return prim


def _set_xform(
    stage: Any,
    prim_path: str,
    position: list[float],
    rotation_xyz_deg: list[float] | None = None,
    scale: list[float] | None = None,
) -> None:
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


def _valid_prim_path(stage: Any, prim_path: str | None) -> bool:
    if not prim_path:
        return False
    prim = stage.GetPrimAtPath(str(prim_path))
    return bool(prim and prim.IsValid())


def _prim_has_articulation_api(prim: Any) -> bool:
    from pxr import UsdPhysics  # type: ignore

    try:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
    except Exception:
        pass
    try:
        applied_schemas = set(prim.GetAppliedSchemas())
    except Exception:
        applied_schemas = set()
    if "PhysicsArticulationRootAPI" in applied_schemas:
        return True
    try:
        from pxr import PhysxSchema  # type: ignore

        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            return True
    except Exception:
        pass
    return "PhysxArticulationAPI" in applied_schemas


def _find_articulation_roots_anywhere(stage: Any) -> list[str]:
    from pxr import Usd  # type: ignore

    roots: list[str] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if _prim_has_articulation_api(prim):
            roots.append(str(prim.GetPath()))
    return roots


def _choose_robot_prim_path(
    stage: Any,
    scene_robot_prim_path: str | None,
    detected_articulation_roots: list[str],
    requested_prim_path: str | None,
) -> dict[str, Any]:
    candidates = [
        ("scene.robot_prim_path", scene_robot_prim_path, False),
        ("detected_articulation_root", detected_articulation_roots[0] if detected_articulation_roots else None, False),
        ("requested_prim_path", requested_prim_path, True),
        ("hardcoded_official_fallback", OFFICIAL_ROBOT_PRIM_PATH, True),
    ]
    attempts: list[dict[str, Any]] = []
    for source, path, fallback_used in candidates:
        valid = _valid_prim_path(stage, path)
        attempts.append({"source": source, "path": path, "valid": valid, "fallback_used": fallback_used})
        if valid:
            return {
                "chosen_robot_prim_path": str(path),
                "chosen_source": source,
                "fallback_used": bool(fallback_used),
                "scene_robot_prim_path": scene_robot_prim_path,
                "detected_articulation_roots": detected_articulation_roots,
                "requested_prim_path": requested_prim_path,
                "hardcoded_official_fallback": OFFICIAL_ROBOT_PRIM_PATH,
                "attempts": attempts,
            }
    raise RuntimeError(f"Could not choose a valid robot prim path: {json.dumps(attempts, sort_keys=True)}")


def _articulation_acquire_candidates(detected_path: str, robot_prim_path: str) -> list[str]:
    candidates: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in candidates:
            candidates.append(path)

    add(detected_path)
    add(robot_prim_path)
    current = detected_path.rstrip("/")
    while "/" in current:
        current = current.rsplit("/", 1)[0]
        if not current:
            break
        add(current)
        if current == robot_prim_path:
            break
    return candidates


def _acquire_articulation_with_fallback(detected_path: str, robot_prim_path: str) -> tuple[Any, Any, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    candidates = _articulation_acquire_candidates(detected_path, robot_prim_path)
    for candidate in candidates:
        try:
            dc, articulation = _acquire_articulation(candidate)
            return dc, articulation, {
                "detected_articulation_path": detected_path,
                "acquired_articulation_path": candidate,
                "candidate_paths": candidates,
                "attempts": attempts + [{"path": candidate, "success": True}],
            }
        except Exception as exc:
            attempts.append({"path": candidate, "success": False, "error": str(exc)})
    raise RuntimeError(f"dynamic_control could not acquire Walker S2 articulation: {json.dumps(attempts, sort_keys=True)}")


def _vector3(value: Any) -> np.ndarray:
    try:
        return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)
    except AttributeError:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def _body_pose_position(dc: Any, body: Any) -> np.ndarray:
    return _vector3(dc.get_rigid_body_pose(body).p)


def _body_pose_orientation(dc: Any, body: Any) -> dict[str, float] | None:
    try:
        q = dc.get_rigid_body_pose(body).r
        return {"w": float(q.w), "x": float(q.x), "y": float(q.y), "z": float(q.z)}
    except Exception:
        return None


def _world_se3_from_prim(stage: Any, prim_path: str) -> tuple[Any, dict[str, Any]]:
    from pxr import UsdGeom  # type: ignore
    import pinocchio as pin  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim for frame alignment: {prim_path}")
    tf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    pos = np.array(tf.ExtractTranslation(), dtype=float)
    rot_gf = tf.ExtractRotationMatrix()
    rot = np.array([[rot_gf[i][j] for j in range(3)] for i in range(3)], dtype=float).T
    return pin.SE3(rot, pos), {"prim_path": prim_path, "world_position": pos.tolist(), "world_rotation": rot.tolist()}


def _pin_frame_se3(ik_solver: Any, frame_name: str) -> Any:
    import pinocchio as pin  # type: ignore

    frame_id = ik_solver.model.getFrameId(frame_name)
    if frame_id >= len(ik_solver.model.frames):
        raise RuntimeError(f"Pinocchio frame not found: {frame_name}")
    pin.forwardKinematics(ik_solver.model, ik_solver.data, ik_solver.q)
    pin.updateFramePlacements(ik_solver.model, ik_solver.data)
    return ik_solver.data.oMf[frame_id].copy()


def _se3_from_matrix(matrix: np.ndarray) -> Any:
    import pinocchio as pin  # type: ignore

    mat = np.array(matrix, dtype=float)
    return pin.SE3(mat[:3, :3], mat[:3, 3])


def _matrix_from_se3(se3: Any) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.array(se3.rotation, dtype=float)
    mat[:3, 3] = np.array(se3.translation, dtype=float)
    return mat


def _rotation_delta_rad(delta_se3: Any) -> float:
    import pinocchio as pin  # type: ignore

    return float(np.linalg.norm(pin.log(delta_se3).vector[3:]))


def _fk_ee_world_se3(coord_transform: Any, ik_solver: Any, arm_side: str) -> tuple[Any, Any]:
    import pinocchio as pin  # type: ignore

    fk_base = ik_solver.get_ee_pose(arm_side)
    fk_world_rot = np.array(coord_transform.robot_world_R, dtype=float) @ np.array(fk_base.rotation, dtype=float)
    fk_world_pos = np.array(coord_transform.robot_to_world(np.array(fk_base.translation, dtype=float)), dtype=float)
    return pin.SE3(fk_world_rot, fk_world_pos), fk_base


def _ee_compensation_se3_from_map(ee_compensation_by_arm: dict[str, Any] | None, arm_side: str) -> Any | None:
    if not ee_compensation_by_arm:
        return None
    raw = ee_compensation_by_arm.get(arm_side) or ee_compensation_by_arm.get("common")
    if raw is None:
        return None
    return _se3_from_matrix(np.array(raw, dtype=float))


def _ee_compensation_se3(args: argparse.Namespace | None, arm_side: str) -> Any | None:
    if args is None or not bool(getattr(args, "ee_frame_compensation_active", False)):
        return None
    return _ee_compensation_se3_from_map(getattr(args, "ee_frame_compensation_by_arm", None), arm_side)


def _ik_target_pose_se3(ik_solver: Any, arm_side: str, target_pose_base: np.ndarray, args: argparse.Namespace | None) -> Any:
    target_se3 = ik_solver.xyzrpy_to_se3(np.array(target_pose_base, dtype=float))
    compensation = _ee_compensation_se3(args, arm_side)
    if compensation is not None:
        return target_se3 * compensation.inverse()
    return target_se3


def _ee_pose_base_from_ik_state(ik_solver: Any, arm_side: str, args: argparse.Namespace | None = None) -> np.ndarray:
    se3 = ik_solver.get_ee_pose(arm_side)
    compensation = _ee_compensation_se3(args, arm_side)
    if compensation is not None:
        se3 = se3 * compensation
    return np.array(ik_solver.se3_to_xyzrpy(se3), dtype=float)


def _coordinate_transform_from_anchor(
    CoordinateTransform: Any,
    ik_solver: Any,
    stage: Any,
    prim_path: str,
    frame_name: str,
    compensation_matrix: np.ndarray | None = None,
    compensation_mode: str = "none",
) -> tuple[Any, dict[str, Any]]:
    isaac_anchor, isaac_log = _world_se3_from_prim(stage, prim_path)
    pin_anchor = _pin_frame_se3(ik_solver, frame_name)
    compensated_anchor = isaac_anchor
    if compensation_matrix is not None:
        compensation = _se3_from_matrix(compensation_matrix)
        if compensation_mode == "post":
            compensated_anchor = isaac_anchor * compensation
        elif compensation_mode == "post_inverse":
            compensated_anchor = isaac_anchor * compensation.inverse()
        else:
            raise RuntimeError(f"Unsupported compensation mode: {compensation_mode}")
    base_world = compensated_anchor * pin_anchor.inverse()
    coord_transform = CoordinateTransform(np.array(base_world.translation), np.array(base_world.rotation))
    return coord_transform, {
        "anchor_prim_path": prim_path,
        "pin_frame_name": frame_name,
        "compensation_mode": compensation_mode,
        "compensation_matrix": None if compensation_matrix is None else np.array(compensation_matrix, dtype=float).tolist(),
        "isaac_anchor": isaac_log,
        "pin_anchor_translation": np.array(pin_anchor.translation, dtype=float).tolist(),
        "pin_anchor_rotation": np.array(pin_anchor.rotation, dtype=float).tolist(),
        "coordinate_transform_robot_world_pos": np.array(coord_transform.robot_world_pos, dtype=float).tolist(),
        "coordinate_transform_robot_world_R": np.array(coord_transform.robot_world_R, dtype=float).tolist(),
    }




def _refresh_coordinate_transform_from_selection(
    *,
    CoordinateTransform: Any,
    ik_solver: Any,
    stage: Any,
    coord_transform: Any,
    torso_prim_path: str,
    coordinate_alignment_selection: dict[str, Any],
) -> dict[str, Any]:
    selected_label = str(coordinate_alignment_selection.get("selected_label", ""))
    selected_candidate = None
    for candidate in coordinate_alignment_selection.get("candidates", []):
        if candidate.get("selected") or str(candidate.get("label")) == selected_label:
            selected_candidate = candidate
            break
    build_log = selected_candidate.get("build_log", {}) if isinstance(selected_candidate, dict) else {}
    compensation_mode = str(build_log.get("compensation_mode", "none"))
    compensation_matrix = build_log.get("compensation_matrix")
    if compensation_mode == "official_raw_from_torso_link":
        compensation_mode = "none"
        compensation_matrix = None
    refreshed, refresh_log = _coordinate_transform_from_anchor(
        CoordinateTransform,
        ik_solver,
        stage,
        torso_prim_path,
        "torso_link",
        compensation_matrix=None if compensation_matrix is None else np.array(compensation_matrix, dtype=float),
        compensation_mode=compensation_mode,
    )
    old_pos = np.array(coord_transform.robot_world_pos, dtype=float)
    old_rot = np.array(coord_transform.robot_world_R, dtype=float)
    coord_transform.robot_world_pos = np.array(refreshed.robot_world_pos, dtype=float)
    coord_transform.robot_world_R = np.array(refreshed.robot_world_R, dtype=float)
    coord_transform.robot_world_R_inv = np.array(refreshed.robot_world_R_inv, dtype=float)
    pos_delta = coord_transform.robot_world_pos - old_pos
    rot_delta_norm = float(np.linalg.norm(coord_transform.robot_world_R - old_rot))
    return {
        "selected_alignment_label": selected_label,
        "refresh_build_log": refresh_log,
        "position_delta_world_m": pos_delta.tolist(),
        "position_delta_norm_m": float(np.linalg.norm(pos_delta)),
        "rotation_matrix_delta_frobenius": rot_delta_norm,
        "coordinate_transform_robot_world_pos": coord_transform.robot_world_pos.tolist(),
        "coordinate_transform_robot_world_R": coord_transform.robot_world_R.tolist(),
    }

def _resolve_link_prim_path(stage: Any, robot_root_path: str, link_name: str) -> tuple[str, list[dict[str, Any]]]:
    from pxr import Usd  # type: ignore

    candidates = [
        f"{robot_root_path.rstrip('/')}/Ref/{link_name}",
        f"{robot_root_path.rstrip('/')}/{link_name}",
        f"/Root/Ref_Xform/Ref/{link_name}",
    ]
    attempts: list[dict[str, Any]] = []
    for path in candidates:
        valid = _valid_prim_path(stage, path)
        attempts.append({"path": path, "valid": valid, "source": "candidate"})
        if valid:
            return path, attempts
    suffix = f"/{link_name}"
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if path.endswith(suffix):
            attempts.append({"path": path, "valid": True, "source": "stage_suffix_scan"})
            return path, attempts
    raise RuntimeError(f"Could not resolve link prim path for {link_name}: {attempts}")


def _verify_ee_alignment_dynamic(
    stage: Any,
    coord_transform: Any,
    ik_solver: Any,
    robot_root_path: str,
    ee_compensation_by_arm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for side, link_name in (("left", "L_sixforce_link"), ("right", "R_sixforce_link")):
        link_path, attempts = _resolve_link_prim_path(stage, robot_root_path, link_name)
        isaac_world, _ = _world_se3_from_prim(stage, link_path)
        fk_world_se3, fk_base_se3 = _fk_ee_world_se3(coord_transform, ik_solver, side)
        compensation = _ee_compensation_se3_from_map(ee_compensation_by_arm, side)
        if compensation is not None:
            fk_world_se3 = fk_world_se3 * compensation
        fk_base = np.array(fk_base_se3.translation, dtype=float)
        fk_world = np.array(fk_world_se3.translation, dtype=float)
        isaac_pos = np.array(isaac_world.translation, dtype=float)
        diff = isaac_pos - fk_world
        results.append(
            {
                "arm": side,
                "link_name": link_name,
                "link_prim_path": link_path,
                "link_path_attempts": attempts,
                "fk_position_base": fk_base.tolist(),
                "fk_position_world": fk_world.tolist(),
                "isaac_position_world": isaac_pos.tolist(),
                "diff_vector_world": diff.tolist(),
                "diff_norm_m": float(np.linalg.norm(diff)),
                "fk_rotation_world": np.array(fk_world_se3.rotation, dtype=float).tolist(),
                "isaac_rotation_world": np.array(isaac_world.rotation, dtype=float).tolist(),
                "rotation_diff_rad": _rotation_delta_rad(fk_world_se3.inverse() * isaac_world),
                "ee_frame_compensation_applied": bool(compensation is not None),
            }
        )
    diff_norms = [float(result["diff_norm_m"]) for result in results]
    rot_norms = [float(result["rotation_diff_rad"]) for result in results]
    return {
        "per_arm": results,
        "max_diff_m": max(diff_norms) if diff_norms else math.inf,
        "mean_diff_m": float(np.mean(diff_norms)) if diff_norms else math.inf,
        "max_rotation_diff_rad": max(rot_norms) if rot_norms else math.inf,
        "mean_rotation_diff_rad": float(np.mean(rot_norms)) if rot_norms else math.inf,
        "ee_frame_compensation_applied": bool(ee_compensation_by_arm),
    }


def _compute_ee_frame_delta_diagnostics(stage: Any, coord_transform: Any, ik_solver: Any, robot_root_path: str) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    per_arm: list[dict[str, Any]] = []
    for side, link_name in (("left", "L_sixforce_link"), ("right", "R_sixforce_link")):
        link_path, attempts = _resolve_link_prim_path(stage, robot_root_path, link_name)
        isaac_world, isaac_log = _world_se3_from_prim(stage, link_path)
        fk_world, fk_base = _fk_ee_world_se3(coord_transform, ik_solver, side)
        delta = fk_world.inverse() * isaac_world
        deltas[side] = delta
        world_diff = np.array(isaac_world.translation, dtype=float) - np.array(fk_world.translation, dtype=float)
        per_arm.append(
            {
                "arm": side,
                "link_name": link_name,
                "link_prim_path": link_path,
                "link_path_attempts": attempts,
                "dualarmik_fk_frame_name": link_name,
                "dualarmik_fk_position_base": np.array(fk_base.translation, dtype=float).tolist(),
                "dualarmik_fk_rotation_base": np.array(fk_base.rotation, dtype=float).tolist(),
                "fk_ee_world_position": np.array(fk_world.translation, dtype=float).tolist(),
                "fk_ee_world_rotation": np.array(fk_world.rotation, dtype=float).tolist(),
                "isaac_ee_world_position": np.array(isaac_world.translation, dtype=float).tolist(),
                "isaac_ee_world_rotation": np.array(isaac_world.rotation, dtype=float).tolist(),
                "isaac_prim_world_log": isaac_log,
                "world_position_diff_vector": world_diff.tolist(),
                "world_position_diff_norm_m": float(np.linalg.norm(world_diff)),
                "ee_delta_fk_to_isaac_translation": np.array(delta.translation, dtype=float).tolist(),
                "ee_delta_fk_to_isaac_translation_norm_m": float(np.linalg.norm(np.array(delta.translation, dtype=float))),
                "ee_delta_fk_to_isaac_rotation": np.array(delta.rotation, dtype=float).tolist(),
                "ee_delta_fk_to_isaac_rotation_angle_rad": _rotation_delta_rad(delta),
                "ee_delta_fk_to_isaac_matrix": _matrix_from_se3(delta).tolist(),
            }
        )

    comparison: dict[str, Any] = {
        "left_right_delta_translation_difference_norm_m": math.inf,
        "left_right_delta_rotation_difference_rad": math.inf,
        "left_right_world_diff_difference_norm_m": math.inf,
        "left_right_deltas_approximately_equal": False,
        "root_cause_classification": "insufficient_ee_delta_data",
        "ee_frame_compensation_supported": False,
    }
    if "left" in deltas and "right" in deltas and len(per_arm) == 2:
        left_delta = deltas["left"]
        right_delta = deltas["right"]
        delta_difference = left_delta.inverse() * right_delta
        delta_translation_difference = float(np.linalg.norm(np.array(delta_difference.translation, dtype=float)))
        delta_rotation_difference = _rotation_delta_rad(delta_difference)
        world_diff_difference = float(
            np.linalg.norm(
                np.array(per_arm[0]["world_position_diff_vector"], dtype=float)
                - np.array(per_arm[1]["world_position_diff_vector"], dtype=float)
            )
        )
        mean_delta_translation_norm = float(
            np.mean([float(item["ee_delta_fk_to_isaac_translation_norm_m"]) for item in per_arm])
        )
        approximately_equal = bool(
            delta_translation_difference <= EE_FRAME_DELTA_TRANSLATION_MATCH_TOL
            and delta_rotation_difference <= EE_FRAME_DELTA_ROTATION_MATCH_TOL
        )
        if approximately_equal and mean_delta_translation_norm >= EE_FRAME_OFFSET_MIN_TRANSLATION:
            classification = "fixed_ee_frame_offset_mismatch"
        elif approximately_equal:
            classification = "no_significant_ee_frame_offset"
        elif world_diff_difference <= EE_FRAME_DELTA_TRANSLATION_MATCH_TOL:
            classification = "remaining_torso_root_transform_mismatch"
        else:
            classification = "urdf_usd_revision_or_link_frame_mismatch"
        comparison = {
            "left_right_delta_translation_difference_norm_m": delta_translation_difference,
            "left_right_delta_rotation_difference_rad": delta_rotation_difference,
            "left_right_world_diff_difference_norm_m": world_diff_difference,
            "left_right_deltas_approximately_equal": approximately_equal,
            "mean_delta_translation_norm_m": mean_delta_translation_norm,
            "translation_match_tolerance_m": EE_FRAME_DELTA_TRANSLATION_MATCH_TOL,
            "rotation_match_tolerance_rad": EE_FRAME_DELTA_ROTATION_MATCH_TOL,
            "minimum_offset_to_compensate_m": EE_FRAME_OFFSET_MIN_TRANSLATION,
            "root_cause_classification": classification,
            "ee_frame_compensation_supported": bool(classification == "fixed_ee_frame_offset_mismatch"),
        }
    return {"per_arm": per_arm, "comparison": comparison}


def _configure_ee_frame_compensation(args: argparse.Namespace, ee_delta_diagnostics: dict[str, Any]) -> dict[str, Any]:
    comparison = ee_delta_diagnostics.get("comparison", {})
    supported = bool(comparison.get("ee_frame_compensation_supported"))
    compensation_by_arm: dict[str, Any] = {}
    if supported:
        for item in ee_delta_diagnostics.get("per_arm", []):
            arm = str(item.get("arm"))
            matrix = item.get("ee_delta_fk_to_isaac_matrix")
            if arm in ("left", "right") and matrix is not None:
                compensation_by_arm[arm] = matrix
    active = bool(supported and set(compensation_by_arm) == {"left", "right"})
    args.ee_frame_compensation_active = active
    args.ee_frame_compensation_by_arm = compensation_by_arm if active else {}
    return {
        "active": active,
        "reason": "near_constant_fk_to_isaac_ee_delta" if active else "ee_delta_not_near_constant",
        "root_cause_classification": comparison.get("root_cause_classification"),
        "compensation_by_arm": args.ee_frame_compensation_by_arm,
        "target_construction_rule": "pinocchio_target = desired_isaac_ee_target * inverse(ee_delta_fk_to_isaac)",
        "current_pose_reporting_rule": "reported_current_ee = pinocchio_fk_ee * ee_delta_fk_to_isaac",
    }


def _select_coordinate_transform_with_alignment(
    CoordinateTransform: Any,
    ik_solver: Any,
    stage: Any,
    torso_prim_path: str,
    chosen_robot_prim_path: str,
) -> tuple[Any, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def add_candidate(label: str, builder: Callable[[], tuple[Any, dict[str, Any]]]) -> None:
        try:
            transform, build_log = builder()
            alignment = _verify_ee_alignment_dynamic(stage, transform, ik_solver, chosen_robot_prim_path)
            candidates.append(
                {
                    "label": label,
                    "success": True,
                    "transform": transform,
                    "build_log": build_log,
                    "alignment": alignment,
                    "score_max_diff_m": float(alignment["max_diff_m"]),
                    "score_mean_diff_m": float(alignment["mean_diff_m"]),
                }
            )
        except Exception as exc:
            candidates.append({"label": label, "success": False, "error": str(exc)})

    add_candidate(
        "official_coordinate_utils_from_torso_link_uncompensated",
        lambda: (
            CoordinateTransform.from_torso_link(ik_solver=ik_solver, torso_prim_path=torso_prim_path),
            {"anchor_prim_path": torso_prim_path, "pin_frame_name": "torso_link", "compensation_mode": "official_raw_from_torso_link"},
        ),
    )
    add_candidate(
        "torso_link_with_official_compensation_post",
        lambda: _coordinate_transform_from_anchor(
            CoordinateTransform,
            ik_solver,
            stage,
            torso_prim_path,
            "torso_link",
            compensation_matrix=OFFICIAL_TORSO_COMPENSATION_MATRIX,
            compensation_mode="post",
        ),
    )
    add_candidate(
        "torso_link_with_official_compensation_post_inverse",
        lambda: _coordinate_transform_from_anchor(
            CoordinateTransform,
            ik_solver,
            stage,
            torso_prim_path,
            "torso_link",
            compensation_matrix=OFFICIAL_TORSO_COMPENSATION_MATRIX,
            compensation_mode="post_inverse",
        ),
    )

    valid_candidates = [candidate for candidate in candidates if candidate.get("success")]
    if not valid_candidates:
        raise RuntimeError(f"No coordinate transform candidate could be built: {candidates}")
    selected = min(valid_candidates, key=lambda candidate: (candidate["score_max_diff_m"], candidate["score_mean_diff_m"]))
    selected_transform = selected["transform"]
    serializable_candidates = []
    for candidate in candidates:
        item = {key: value for key, value in candidate.items() if key != "transform"}
        item["selected"] = bool(candidate is selected)
        serializable_candidates.append(item)
    return selected_transform, {
        "selected_label": selected["label"],
        "selected_score_max_diff_m": float(selected["score_max_diff_m"]),
        "selected_score_mean_diff_m": float(selected["score_mean_diff_m"]),
        "selected_alignment": selected["alignment"],
        "candidates": serializable_candidates,
    }


def _list_articulation_bodies(dc: Any, articulation: Any) -> list[tuple[int, Any, str, str]]:
    bodies: list[tuple[int, Any, str, str]] = []
    for index in range(dc.get_articulation_body_count(articulation)):
        body = dc.get_articulation_body(articulation, index)
        bodies.append((index, body, str(dc.get_rigid_body_name(body)), str(dc.get_rigid_body_path(body))))
    return bodies


def _identify_end_effector_body(
    dc: Any,
    articulation: Any,
    requested_body: str | None,
    arm_side: str,
) -> tuple[Any, str, str, str]:
    bodies = _list_articulation_bodies(dc, articulation)
    if requested_body:
        requested_lower = requested_body.lower()
        for _, body, name, path in bodies:
            if requested_body == name or requested_body == path or requested_lower in path.lower():
                return body, name, path, "requested"
        raise RuntimeError(f"Requested end-effector body not found: {requested_body}")

    preferred = "R_sixforce_link" if arm_side == "right" else "L_sixforce_link"
    preferred_lower = preferred.lower()
    for _, body, name, path in bodies:
        if preferred_lower in f"{name} {path}".lower():
            return body, name, path, "official_dualarmik_sixforce"

    tokens = RIGHT_EE_TOKENS if arm_side == "right" else LEFT_EE_TOKENS
    candidates: list[tuple[int, Any, str, str]] = []
    for index, body, name, path in bodies:
        lower = f"{name} {path}".lower()
        if any(token in lower for token in EE_EXCLUDE_TOKENS):
            continue
        if any(token in lower for token in tokens):
            candidates.append((index, body, name, path))
    if not candidates:
        raise RuntimeError(f"Could not identify {arm_side} end-effector body; available={[path for _, _, _, path in bodies]}")
    _, body, name, path = candidates[-1]
    return body, name, path, "fallback_local_ee"


def _arm_side_match_score(name_or_path: str, arm_side: str) -> int:
    lower = str(name_or_path).lower()
    if arm_side == "right":
        strong = ("/r_", "/right", "r_", "right")
        weak = ("/r", "_r")
    else:
        strong = ("/l_", "/left", "l_", "left")
        weak = ("/l", "_l")
    if any(token in lower for token in strong):
        return 2
    if any(token in lower for token in weak):
        return 1
    return 0


def _prim_bbox_center_world(stage: Any, prim_path: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        box = _bbox(stage, prim_path)
        center = np.array(box.get("center"), dtype=float)
    except Exception as exc:
        return None, {
            "bbox_center_available": False,
            "bbox_center_source": "usd_bbox",
            "bbox_center_error": repr(exc),
            "prim_path": prim_path,
        }
    if center.shape != (3,) or not np.isfinite(center).all():
        return None, {
            "bbox_center_available": False,
            "bbox_center_source": "usd_bbox",
            "bbox_center_error": "non_finite_or_wrong_shape_bbox_center",
            "prim_path": prim_path,
            "bbox": box,
        }
    return center, {
        "bbox_center_available": True,
        "bbox_center_source": "usd_bbox",
        "prim_path": prim_path,
        "bbox": box,
        "bbox_center_world": center.tolist(),
    }


def _resolve_named_body_or_prim_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    name_token: str,
    arm_side: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    token = str(name_token or "").strip().lower()
    if not token:
        return None, {"enabled": False, "reason": "empty_reference_link_token"}

    body_matches: list[dict[str, Any]] = []
    for index, body, name, path in _list_articulation_bodies(dc, articulation):
        combined = f"{name} {path}"
        if token in combined.lower():
            body_matches.append(
                {
                    "index": index,
                    "body": body,
                    "name": name,
                    "path": path,
                    "arm_side_match_score": _arm_side_match_score(combined, arm_side),
                    "source": "dynamic_control_body",
                }
            )
    if body_matches:
        selected = max(body_matches, key=lambda item: (int(item["arm_side_match_score"]), -int(item["index"])))
        body_pos = _body_pose_position(dc, selected["body"])
        bbox_pos, bbox_log = _prim_bbox_center_world(stage, str(selected["path"]))
        use_bbox_center = bbox_pos is not None
        pos = bbox_pos if use_bbox_center else body_pos
        return pos, {
            "enabled": True,
            "resolved": True,
            "source": "dynamic_control_body_usd_bbox_center" if use_bbox_center else selected["source"],
            "position_semantics": "usd_bbox_center_of_selected_reference_prim" if use_bbox_center else "dynamic_control_rigid_body_pose_translation",
            "requested_name_token": name_token,
            "selected_name": selected["name"],
            "selected_path": selected["path"],
            "selected_arm_side_match_score": selected["arm_side_match_score"],
            "candidate_count": len(body_matches),
            "candidates": [
                {key: value for key, value in item.items() if key != "body"}
                for item in body_matches
            ],
            "dynamic_control_body_world_position": body_pos.tolist(),
            "bbox_center_log": bbox_log,
            "world_position": pos.tolist(),
        }

    from pxr import Usd  # type: ignore

    prim_matches: list[dict[str, Any]] = []
    root = str(robot_root_path).rstrip("/")
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if root and not path.startswith(root):
            continue
        if token in path.lower():
            prim_matches.append(
                {
                    "path": path,
                    "arm_side_match_score": _arm_side_match_score(path, arm_side),
                    "source": "usd_prim",
                }
            )
    if prim_matches:
        selected = max(prim_matches, key=lambda item: (int(item["arm_side_match_score"]), -len(str(item["path"]))))
        se3, prim_log = _world_se3_from_prim(stage, str(selected["path"]))
        prim_pos = np.array(se3.translation, dtype=float)
        bbox_pos, bbox_log = _prim_bbox_center_world(stage, str(selected["path"]))
        use_bbox_center = bbox_pos is not None
        pos = bbox_pos if use_bbox_center else prim_pos
        return pos, {
            "enabled": True,
            "resolved": True,
            "source": "usd_prim_bbox_center" if use_bbox_center else selected["source"],
            "position_semantics": "usd_bbox_center_of_selected_reference_prim" if use_bbox_center else "usd_prim_transform_translation",
            "requested_name_token": name_token,
            "selected_path": selected["path"],
            "selected_arm_side_match_score": selected["arm_side_match_score"],
            "candidate_count": len(prim_matches),
            "candidates": prim_matches,
            "world_position": pos.tolist(),
            "prim_world_log": prim_log,
            "bbox_center_log": bbox_log,
        }

    return None, {
        "enabled": True,
        "resolved": False,
        "requested_name_token": name_token,
        "source": None,
        "reason": "no_dynamic_control_body_or_usd_prim_matched_token",
        "fallback": "vertical point_B XY remains active",
    }


def _is_finger_midpoint_vertical_reference(name_token: str) -> bool:
    normalized = str(name_token or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in VERTICAL_FINGER_MIDPOINT_REFERENCE_ALIASES


def _resolve_finger_midpoint_reference_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    requested_name_token: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prefix = "R" if arm_side == "right" else "L"
    component_specs = [
        ("finger1_link", f"{prefix}_finger1_link", "finger1_link"),
        ("finger2_link", f"{prefix}_finger2_link", "finger2_link"),
    ]
    positions: list[np.ndarray] = []
    component_logs: list[dict[str, Any]] = []
    missing_roles: list[str] = []

    for role, preferred_token, fallback_token in component_specs:
        selected_position: np.ndarray | None = None
        selected_log: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for token in (preferred_token, fallback_token):
            pos, log = _resolve_named_body_or_prim_position(
                stage=stage,
                dc=dc,
                articulation=articulation,
                robot_root_path=robot_root_path,
                name_token=token,
                arm_side=arm_side,
            )
            log = {
                **log,
                "component_role": role,
                "component_name_token": token,
            }
            attempts.append(log)
            if pos is not None:
                selected_position = np.array(pos, dtype=float)
                selected_log = log
                break
        if selected_position is None or selected_log is None:
            missing_roles.append(role)
            component_logs.append(
                {
                    "component_role": role,
                    "resolved": False,
                    "attempts": attempts,
                }
            )
            continue
        positions.append(selected_position)
        component_logs.append(
            {
                "component_role": role,
                "resolved": True,
                "selected_world_position": selected_position.tolist(),
                "selected_log": selected_log,
                "attempts": attempts,
            }
        )

    if missing_roles or len(positions) != 2:
        return None, {
            "enabled": True,
            "resolved": False,
            "source": None,
            "requested_name_token": requested_name_token,
            "reference_mode": "finger1_finger2_midpoint",
            "arm_side": arm_side,
            "missing_component_roles": missing_roles,
            "component_reference_logs": component_logs,
            "reason": "finger_midpoint_reference_requires_both_finger1_link_and_finger2_link",
            "fallback": "vertical point_B XY remains active",
        }

    midpoint = 0.5 * (positions[0] + positions[1])
    return midpoint, {
        "enabled": True,
        "resolved": True,
        "source": "finger_link_pair_midpoint",
        "position_semantics": "midpoint_of_finger1_link_and_finger2_link_reference_positions",
        "requested_name_token": requested_name_token,
        "reference_mode": "finger1_finger2_midpoint",
        "arm_side": arm_side,
        "component_reference_logs": component_logs,
        "component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "finger1_world_position": positions[0].tolist(),
        "finger2_world_position": positions[1].tolist(),
        "world_position": midpoint.tolist(),
    }


def _resolve_real_grasp_center_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    center_world, log = _resolve_finger_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token="real_grasp_center_finger_midpoint",
    )
    return center_world, {
        **log,
        "grasp_center_definition": "midpoint_between_active_finger1_link_and_finger2_link_reference_positions",
        "close_critical_reference": bool(center_world is not None),
        "fallback": None if center_world is not None else "close-critical evaluation falls back to point_B proxy",
    }


def _resolve_current_vertical_xy_reference_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    reference_log: dict[str, Any] | None,
    coord_transform: Any,
    current_pose_base: np.ndarray,
    reference_offset_local: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    log = reference_log or {}
    requested_reference = str(log.get("requested_name_token") or DEFAULT_VERTICAL_XY_REFERENCE_LINK)
    if _is_finger_midpoint_vertical_reference(requested_reference):
        live_world, live_log = _resolve_finger_midpoint_reference_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            requested_name_token=requested_reference,
        )
    else:
        live_world, live_log = _resolve_named_body_or_prim_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            name_token=requested_reference,
            arm_side=arm_side,
        )

    if live_world is not None:
        return np.array(live_world, dtype=float), {
            **live_log,
            "runtime_source": "live_vertical_xy_reference_query",
            "runtime_fallback_used": False,
        }

    fallback_world = _point_world_from_pose(coord_transform, current_pose_base, reference_offset_local)
    return fallback_world, {
        **live_log,
        "runtime_source": "ee_pose_plus_initial_reference_offset_fallback",
        "runtime_fallback_used": True,
        "fallback_reference_world": fallback_world.tolist(),
    }


def _resolve_vertical_xy_reference_offset(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    coord_transform: Any,
    ik_solver: Any,
    arm_side: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    requested_reference = str(args.vertical_xy_reference_link)
    if _is_finger_midpoint_vertical_reference(requested_reference):
        reference_world, reference_log = _resolve_finger_midpoint_reference_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            requested_name_token=requested_reference,
        )
    else:
        reference_world, reference_log = _resolve_named_body_or_prim_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            name_token=requested_reference,
            arm_side=arm_side,
        )
    if reference_world is None:
        return None, {
            **reference_log,
            "vertical_xy_reference_active": False,
            "reference_semantics": "vertical XY reference requested but unresolved; fallback keeps point_B XY tracking",
        }

    ee_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    ee_world = _pose_position_world(coord_transform, ee_pose)
    ee_rot_world = _pose_rotation_world(coord_transform, ee_pose)
    offset_local = ee_rot_world.T @ (np.array(reference_world, dtype=float) - ee_world)
    return offset_local, {
        **reference_log,
        "vertical_xy_reference_active": True,
        "reference_semantics": "resolved vertical XY reference position is converted to a local EE-frame offset and used for vertical object-XY alignment",
        "ee_pose_base_used_for_offset": ee_pose.tolist(),
        "ee_world_position_used_for_offset": ee_world.tolist(),
        "reference_world_position_used_for_offset": np.array(reference_world, dtype=float).tolist(),
        "reference_offset_local_from_ee": offset_local.tolist(),
    }


def _current_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> np.ndarray:
    return np.array([float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs], dtype=float)


def _named_positions(selected_dofs: list[tuple[int, Any, str]], positions: np.ndarray | list[float]) -> dict[str, float]:
    return {name: float(position) for (_, _, name), position in zip(selected_dofs, positions)}


def _read_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> list[float]:
    return [float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs]


def _targets_from_map(selected_dofs: list[tuple[int, Any, str]], target_by_name: dict[str, float]) -> list[float]:
    missing = [name for _, _, name in selected_dofs if name not in target_by_name]
    if missing:
        raise RuntimeError(f"Missing target values for DOFs: {missing}")
    return [float(target_by_name[name]) for _, _, name in selected_dofs]


def _all_joint_state_for_ik(dc: Any, articulation: Any) -> tuple[list[str], list[float]]:
    names: list[str] = []
    positions: list[float] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        names.append(str(dc.get_dof_name(dof)))
        positions.append(float(dc.get_dof_position(dof)))
    return names, positions


def _select_dofs_in_name_order(dc: Any, articulation: Any, ordered_names: list[str]) -> list[tuple[int, Any, str]]:
    by_name: dict[str, tuple[int, Any, str]] = {}
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        by_name[name] = (index, dof, name)
    missing = [name for name in ordered_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing required arm DOFs for DualArmIK order: {missing}")
    return [by_name[name] for name in ordered_names]


def _select_gripper_dofs(dc: Any, articulation: Any, arm_side: str) -> list[tuple[int, Any, str]]:
    include = RIGHT_GRIPPER_TOKENS if arm_side == "right" else LEFT_GRIPPER_TOKENS
    exclude = LEFT_GRIPPER_TOKENS if arm_side == "right" else RIGHT_GRIPPER_TOKENS
    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        lower = name.lower()
        if any(token in lower for token in exclude):
            continue
        if any(token in lower for token in include):
            selected.append((index, dof, name))
    if not selected:
        all_names = [str(dc.get_dof_name(dc.get_articulation_dof(articulation, index))) for index in range(dc.get_articulation_dof_count(articulation))]
        raise RuntimeError(f"No {arm_side} gripper DOFs matched tokens={include}; available_dof_names={all_names}")
    return selected


def _select_dofs_by_target_names(
    dc: Any,
    articulation: Any,
    target_by_name: dict[str, float],
    required_names: set[str],
) -> tuple[list[tuple[int, Any, str]], list[str]]:
    selected: list[tuple[int, Any, str]] = []
    found: set[str] = set()
    missing_optional: list[str] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        if name in target_by_name:
            selected.append((index, dof, name))
            found.add(name)
    missing_required = sorted(required_names - found)
    if missing_required:
        raise RuntimeError(f"Missing required official startup DOFs: {missing_required}")
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
    interface = module.IsaacSimRobotInterface(prim_path=prim_path, name=OFFICIAL_ROBOT_NAME, world=None, urdf_path=str(urdf_path))
    joint_map = dict(getattr(interface, "_joint_value_map", {}))
    missing = sorted(OFFICIAL_STARTUP_ARM_JOINT_NAMES - set(joint_map))
    if missing:
        raise RuntimeError(f"Official startup joint map missing expected arm joints: {missing}")
    return {name: float(value) for name, value in joint_map.items()}


def _load_official_ik_classes(baseline_root: Path) -> tuple[Any, Any]:
    ubtech_path = baseline_root / "Ubtech_sim"
    if not ubtech_path.exists():
        raise RuntimeError(f"Official Ubtech_sim directory missing: {ubtech_path}")
    if str(ubtech_path) not in sys.path:
        sys.path.insert(0, str(ubtech_path))
    from source.DualArmIK import DualArmIK  # type: ignore
    from source.coordinate_utils import CoordinateTransform  # type: ignore

    return DualArmIK, CoordinateTransform


def _seed_joint_positions_for_initialization(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    target_positions: list[float] | np.ndarray,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(target_positions) != len(selected_dofs):
        errors.append(f"target length {len(target_positions)} does not match DOF length {len(selected_dofs)}")
    for index, (_, dof, name), target_value in zip(range(len(selected_dofs)), selected_dofs, target_positions):
        try:
            if hasattr(dc, "set_dof_position"):
                dc.set_dof_position(dof, float(target_value))
            if hasattr(dc, "set_dof_position_target"):
                dc.set_dof_position_target(dof, float(target_value))
        except Exception as exc:
            errors.append(f"{index}:{name}: {exc}")
    return {
        "supported": not errors,
        "method": "initialization_only_set_dof_position_and_target",
        "dof_names": [name for _, _, name in selected_dofs],
        "errors": errors,
    }


def _apply_gripper_effort(dc: Any, gripper_dofs: list[tuple[int, Any, str]], effort_value: float) -> dict[str, Any]:
    if not hasattr(dc, "set_dof_effort"):
        return {
            "supported": False,
            "method": None,
            "effort_value": float(effort_value),
            "dof_names": [name for _, _, name in gripper_dofs],
            "errors": ["dynamic_control does not expose set_dof_effort"],
        }
    applied: dict[str, float] = {}
    errors: list[str] = []
    for index, dof, name in gripper_dofs:
        try:
            dc.set_dof_effort(dof, float(effort_value))
            applied[name] = float(effort_value)
        except Exception as exc:
            errors.append(f"{index}:{name}: {exc}")
    return {
        "supported": not errors and len(applied) == len(gripper_dofs),
        "method": "dynamic_control.set_dof_effort",
        "effort_value": float(effort_value),
        "dof_names": [name for _, _, name in gripper_dofs],
        "applied_efforts": applied,
        "errors": errors,
    }


def _run_updates(
    sim_app: Any,
    steps: int,
    counter: dict[str, int],
    dc: Any | None = None,
    gripper_dofs: list[tuple[int, Any, str]] | None = None,
    gripper_effort: float | None = None,
) -> dict[str, Any] | None:
    last_effort_result: dict[str, Any] | None = None
    for _ in range(max(0, int(steps))):
        if dc is not None and gripper_dofs is not None and gripper_effort is not None:
            last_effort_result = _apply_gripper_effort(dc, gripper_dofs, gripper_effort)
            if not last_effort_result["supported"]:
                raise RuntimeError(f"Gripper effort command failed: {last_effort_result}")
        sim_app.update()
        counter["step"] += 1
    return last_effort_result


def _gripper_values(dc: Any, gripper_dofs: list[tuple[int, Any, str]]) -> dict[str, float]:
    return _named_positions(gripper_dofs, _read_positions(dc, gripper_dofs))


def _append_phase(
    phase_log: list[dict[str, Any]],
    *,
    phase: str,
    start_step: int,
    end_step: int,
    condition_met: bool,
    details: dict[str, Any],
) -> None:
    phase_log.append(
        {
            "phase": phase,
            "start_step": int(start_step),
            "end_step": int(end_step),
            "condition_met": bool(condition_met),
            "details": details,
        }
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_logs(log_root: Path, payload: dict[str, Any], log_suffix: str | None) -> list[str]:
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = payload["run_metadata"].get("timestamp_compact") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{log_suffix}" if log_suffix else ""
    rolling = log_root / f"{LOG_STEM}.log"
    per_run = log_root / f"{LOG_STEM}_{timestamp}{suffix}.log"
    payload["log_paths"] = [str(rolling), str(per_run)]
    safe_payload = _json_safe(payload)
    text = "\n".join(
        (
            f"status={safe_payload.get('final_status', 'fail')}",
            f"failure_reason={safe_payload.get('failure_reason')}",
            f"timestamp_utc={safe_payload['run_metadata'].get('timestamp_utc')}",
            f"script_name={SCRIPT_NAME}",
            f"selected_target_prim={safe_payload.get('target', {}).get('prim_path')}",
            f"chosen_arm={safe_payload.get('robot', {}).get('chosen_arm')}",
            f"ee_frame={safe_payload.get('robot', {}).get('end_effector_path')}",
            f"object_lifted={safe_payload['result_flags'].get('object_lifted')}",
            f"object_transported={safe_payload['result_flags'].get('object_transported')}",
            f"final_inside_bin={safe_payload['result_flags'].get('final_inside_bin')}",
            f"payload={json.dumps(safe_payload, indent=2, sort_keys=True)}",
        )
    ) + "\n"
    rolling.write_text(text, encoding="utf-8")
    per_run.write_text(text, encoding="utf-8")
    return payload["log_paths"]


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _bbox_state(stage: Any, prim_path: str) -> dict[str, Any]:
    box = _bbox(stage, prim_path)
    return {
        "bbox": box,
        "center": box["center"],
        "finite": _finite(box["min"] + box["max"] + box["center"]),
    }


def _center_from_bbox(box: dict[str, list[float]]) -> np.ndarray:
    return np.array(box["center"], dtype=float)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _workspace_check(position: np.ndarray, x_limits: tuple[float, float], y_limits: tuple[float, float], z_limits: tuple[float, float]) -> dict[str, Any]:
    point = np.array(position, dtype=float)
    return {
        "position": point.tolist(),
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "z_limits": list(z_limits),
        "x_ok": bool(x_limits[0] <= point[0] <= x_limits[1]),
        "y_ok": bool(y_limits[0] <= point[1] <= y_limits[1]),
        "z_ok": bool(z_limits[0] <= point[2] <= z_limits[1]),
        "workspace_ok": bool(x_limits[0] <= point[0] <= x_limits[1] and y_limits[0] <= point[1] <= y_limits[1] and z_limits[0] <= point[2] <= z_limits[1]),
    }


def _inside_bin(center: np.ndarray, bin_bbox: dict[str, list[float]], wall_thickness: float, floor_top_z: float) -> bool:
    min_v = bin_bbox["min"]
    max_v = bin_bbox["max"]
    return bool(
        min_v[0] + wall_thickness <= center[0] <= max_v[0] - wall_thickness
        and min_v[1] + wall_thickness <= center[1] <= max_v[1] - wall_thickness
        and floor_top_z <= center[2] <= max_v[2] + 0.12
    )


def _settle_and_measure(stage: Any, target_path: str, sim_app: Any, settle_steps: int, counter: dict[str, int]) -> tuple[dict[str, Any], float]:
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


def _create_debug_marker(stage: Any, path: str, position: np.ndarray | list[float], radius: float, color: tuple[float, float, float]) -> str:
    from pxr import Gf, UsdGeom  # type: ignore

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    point = np.array(position, dtype=float)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return path


def _compute_robot_base_target_components(object_world: np.ndarray, robot_base_position: np.ndarray, robot_base_yaw_rad: float) -> dict[str, Any]:
    delta_world = np.array(object_world, dtype=float) - np.array(robot_base_position, dtype=float)
    c = math.cos(robot_base_yaw_rad)
    s = math.sin(robot_base_yaw_rad)
    forward = c * delta_world[0] + s * delta_world[1]
    lateral = -s * delta_world[0] + c * delta_world[1]
    vertical = delta_world[2]
    return {
        "object_world": np.array(object_world, dtype=float).tolist(),
        "delta_world": delta_world.tolist(),
        "forward_base": float(forward),
        "lateral_base": float(lateral),
        "vertical_base": float(vertical),
        "object_base": [float(forward), float(lateral), float(vertical)],
    }


def _choose_arm_side(args: argparse.Namespace, target_components: dict[str, Any]) -> str:
    if args.arm in ("left", "right"):
        return str(args.arm)
    return "left" if float(target_components["lateral_base"]) > 0.0 else "right"


def _category_from_target(stage: Any, target_path: str, target_index: int, num_parts_per_class: int) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(target_path)
    refs = _reference_paths(prim) if prim and prim.IsValid() else []
    category_from_refs = _category_from_reference(refs)
    category_from_order = "part_a" if target_index < num_parts_per_class else "part_b"
    return {
        "referenced_usd_paths": refs,
        "category_from_reference": category_from_refs,
        "category_from_scene_builder_order": category_from_order,
        "inferred_category": category_from_refs if category_from_refs != "unknown" else category_from_order,
        "category_inference_method": "reference_path" if category_from_refs != "unknown" else "scene_builder_creation_order",
    }


def _target_nearness_sort_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
    if not bool(record.get("valid", False)):
        return (math.inf, math.inf, math.inf, int(record.get("target_index", 0)))
    return (
        float(record["forward_base"]),
        abs(float(record["lateral_base"])),
        float(record["robot_to_object_distance_world_m"]),
        int(record["target_index"]),
    )


def _build_target_candidate_records(
    *,
    stage: Any,
    part_paths: list[str],
    num_parts_per_class: int,
    robot_base_position: np.ndarray,
    robot_base_yaw_rad: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target_index, target_path in enumerate(part_paths):
        try:
            state = _bbox_state(stage, target_path)
            center = _center_from_bbox(state["bbox"])
            target_components = _compute_robot_base_target_components(center, robot_base_position, robot_base_yaw_rad)
            delta_world = np.array(target_components["delta_world"], dtype=float)
            finite = bool(state["finite"] and np.isfinite(delta_world).all())
            category = _category_from_target(stage, target_path, target_index, num_parts_per_class)
            record = {
                "target_index": int(target_index),
                "prim_path": target_path,
                "valid": finite,
                "invalid_reason": None if finite else "non_finite_bbox_or_target_components",
                "initial_state": state,
                "center_world": center.tolist(),
                "robot_base_target_components": target_components,
                "forward_base": float(target_components["forward_base"]),
                "lateral_base": float(target_components["lateral_base"]),
                "abs_lateral_base": abs(float(target_components["lateral_base"])),
                "robot_to_object_distance_world_m": float(np.linalg.norm(delta_world)),
                "robot_to_object_horizontal_distance_world_m": float(np.linalg.norm(delta_world[:2])),
                "nearness_metric_name": "forward_base",
                "nearness_metric_value": float(target_components["forward_base"]),
                "category": category,
            }
        except Exception as exc:
            record = {
                "target_index": int(target_index),
                "prim_path": target_path,
                "valid": False,
                "invalid_reason": f"target_record_failed: {exc}",
                "initial_state": None,
                "center_world": None,
                "robot_base_target_components": None,
                "forward_base": math.inf,
                "lateral_base": math.inf,
                "abs_lateral_base": math.inf,
                "robot_to_object_distance_world_m": math.inf,
                "robot_to_object_horizontal_distance_world_m": math.inf,
                "nearness_metric_name": "forward_base",
                "nearness_metric_value": math.inf,
                "category": {},
            }
        records.append(record)
    return records


def _select_target_record(
    *,
    stage: Any,
    part_paths: list[str],
    requested_target_index: int,
    selection_policy: str,
    num_parts_per_class: int,
    robot_base_position: np.ndarray,
    robot_base_yaw_rad: float,
) -> dict[str, Any]:
    records = _build_target_candidate_records(
        stage=stage,
        part_paths=part_paths,
        num_parts_per_class=num_parts_per_class,
        robot_base_position=robot_base_position,
        robot_base_yaw_rad=robot_base_yaw_rad,
    )
    if selection_policy == "index":
        selected = records[int(requested_target_index)]
        selection_reason = "explicit_target_index_policy"
    elif selection_policy == "nearest":
        valid_records = [record for record in records if bool(record.get("valid", False))]
        if not valid_records:
            raise RuntimeError("No finite target object records are available for nearest target selection")
        selected = min(valid_records, key=_target_nearness_sort_key)
        selection_reason = "nearest_valid_object_by_smallest_forward_base"
    else:
        raise RuntimeError(f"Unsupported target selection policy: {selection_policy}")

    if not bool(selected.get("valid", False)):
        raise RuntimeError(f"Selected target record is invalid: {selected.get('invalid_reason')}")

    ranked_records = sorted(records, key=_target_nearness_sort_key)
    return {
        "selection_policy": selection_policy,
        "selection_reason": selection_reason,
        "requested_target_index": int(requested_target_index),
        "selected_target_index": int(selected["target_index"]),
        "selected_prim_path": selected["prim_path"],
        "selected_nearness_metric_name": selected["nearness_metric_name"],
        "selected_nearness_metric_value": float(selected["nearness_metric_value"]),
        "selected_forward_base": float(selected["forward_base"]),
        "selected_lateral_base": float(selected["lateral_base"]),
        "selected_robot_to_object_distance_world_m": float(selected["robot_to_object_distance_world_m"]),
        "selected_robot_to_object_horizontal_distance_world_m": float(selected["robot_to_object_horizontal_distance_world_m"]),
        "ranking_rule": "smallest forward_base, then smallest abs(lateral_base), then smallest Euclidean world distance",
        "candidate_records": records,
        "candidate_records_ranked_by_nearness": [
            {
                "target_index": int(record["target_index"]),
                "prim_path": record["prim_path"],
                "valid": bool(record.get("valid", False)),
                "forward_base": float(record["forward_base"]),
                "lateral_base": float(record["lateral_base"]),
                "robot_to_object_distance_world_m": float(record["robot_to_object_distance_world_m"]),
                "nearness_rank_key": list(_target_nearness_sort_key(record)),
            }
            for record in ranked_records
        ],
        "selected_record": selected,
    }


def _resolve_tcp_offset(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.tcp_offset is not None:
        raw: Any = args.tcp_offset
        source = "cli"
    else:
        raw = cfg.get("grasp", {}).get("tcp_offset", [0.0, 0.0, 0.0])
        source = "yaml"
    if isinstance(raw, dict):
        raw = [raw.get("x", 0.0), raw.get("y", 0.0), raw.get("z", 0.0)]
    try:
        tcp = np.array(list(raw), dtype=float)
    except Exception:
        tcp = np.zeros(3, dtype=float)
    fallback = np.array([float(args.tcp_fallback_x), 0.0, 0.0], dtype=float)
    fallback_used = bool(tcp.shape != (3,) or float(np.linalg.norm(tcp)) < TCP_OFFSET_EPS)
    if fallback_used:
        tcp = fallback.copy()
    return tcp, {
        "source": source,
        "raw_value": raw,
        "fallback_used": fallback_used,
        "fallback_value": fallback.tolist(),
        "value": tcp.tolist(),
        "zero_norm_threshold": TCP_OFFSET_EPS,
        "tuning_hint": "--tcp-fallback-x 0.17",
    }


def _euler_xyz_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _rot_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    r = np.array(rot, dtype=float)
    if abs(float(r[2, 0])) < 1.0 - 1.0e-9:
        pitch = math.asin(-float(r[2, 0]))
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        pitch = math.pi / 2.0 if float(r[2, 0]) <= -1.0 else -math.pi / 2.0
        roll = 0.0
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return np.array([roll, pitch, yaw], dtype=float)


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.array(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 1.0e-9:
        v = np.array(fallback, dtype=float)
        norm = float(np.linalg.norm(v))
    return v / max(norm, 1.0e-9)

def _approach_axis_from_mode(rot: np.ndarray, mode: str) -> np.ndarray:
    if mode == "neg_z":
        return -rot[:, 2]
    if mode == "pos_z":
        return rot[:, 2]
    if mode == "pos_x":
        return rot[:, 0]
    if mode == "neg_x":
        return -rot[:, 0]
    raise RuntimeError(f"Unsupported approach axis mode: {mode}")

def _base_down_vector(coord_transform: Any) -> np.ndarray:
    world_down = np.array([0.0, 0.0, -1.0], dtype=float)
    return _normalize(coord_transform.robot_world_R_inv @ world_down, np.array([0.0, 0.0, -1.0], dtype=float))


def _classify_target_region(forward_base: float, args: argparse.Namespace) -> str:
    if float(forward_base) >= float(args.far_threshold):
        return "far"
    if float(forward_base) < float(args.near_body_threshold):
        return "near_body"
    return "mid"


def _approach_family_order_for_region(region: str) -> list[str]:
    if region == "far":
        return ["world_y_approach"]
    if region == "near_body":
        return ["z_approach", "world_y_approach"]
    if region == "mid":
        return ["z_approach"]
    raise RuntimeError(f"Unsupported target region: {region}")


def _raw_orientation_presets_by_arm_and_family() -> dict[str, dict[str, list[tuple[str, list[float]]]]]:
    return {
        "right": {
            "z_approach": list(RIGHT_Z_APPROACH_PRESETS),
            "world_y_approach": list(RIGHT_WORLD_Y_APPROACH_PRESETS),
        },
        "left": {
            "z_approach": list(LEFT_Z_APPROACH_PRESETS),
            "world_y_approach": list(LEFT_WORLD_Y_APPROACH_PRESETS),
        },
    }


def _debug_fixed_rpy_for_arm(args: argparse.Namespace, arm_side: str) -> tuple[str, list[float]] | None:
    if arm_side == "right" and args.debug_fixed_rpy_right is not None:
        return "right_debug_fixed_rpy", [float(value) for value in args.debug_fixed_rpy_right]
    if arm_side == "left" and args.debug_fixed_rpy_left is not None:
        return "left_debug_fixed_rpy", [float(value) for value in args.debug_fixed_rpy_left]
    return None


def _world_y_axis_diagnostics(coord_transform: Any | None, approach_axis_base: np.ndarray) -> dict[str, Any]:
    if coord_transform is None:
        return {
            "approach_axis_world": None,
            "dot_with_world_pos_y": None,
            "dot_with_world_neg_y": None,
            "world_axis_diagnostics_available": False,
        }
    axis_world = _normalize(
        np.array(coord_transform.robot_world_R, dtype=float) @ np.array(approach_axis_base, dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )
    world_pos_y = np.array([0.0, 1.0, 0.0], dtype=float)
    world_neg_y = np.array([0.0, -1.0, 0.0], dtype=float)
    return {
        "approach_axis_world": axis_world.tolist(),
        "dot_with_world_pos_y": float(np.dot(axis_world, world_pos_y)),
        "dot_with_world_neg_y": float(np.dot(axis_world, world_neg_y)),
        "world_axis_diagnostics_available": True,
    }


def _preset_axial_roll_metadata(label: str) -> dict[str, Any]:
    for variant_label, axial_roll_rad in WORLD_Y_APPROACH_AB_ROLL_VARIANTS:
        if variant_label in label:
            return {
                "preset_axial_roll_variant_label": variant_label,
                "preset_axial_roll_about_ab_rad": float(axial_roll_rad),
            }
    return {
        "preset_axial_roll_variant_label": None,
        "preset_axial_roll_about_ab_rad": None,
    }


def _orientation_preset_record(
    index: int,
    label: str,
    rpy_values: list[float],
    family_name: str,
    *,
    coord_transform: Any | None = None,
    debug_override: bool = False,
) -> dict[str, Any]:
    rpy = np.array(rpy_values, dtype=float)
    rot = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    approach_axis_base = _normalize(rot[:, 2], np.array([0.0, 0.0, -1.0], dtype=float))
    axis_diagnostics = _world_y_axis_diagnostics(coord_transform, approach_axis_base)
    return {
        "preset_index": int(index),
        "preset_label": label,
        "preset_family": family_name,
        "rpy": rpy.tolist(),
        "rotation_matrix": rot.tolist(),
        "approach_axis_base": approach_axis_base.tolist(),
        "AB_axis_base": approach_axis_base.tolist(),
        "AB_axis_world": axis_diagnostics["approach_axis_world"],
        "up_axis_base": (-rot[:, 2]).tolist(),
        "z_axis_base": rot[:, 2].tolist(),
        "debug_override": bool(debug_override),
        **_preset_axial_roll_metadata(label),
        **axis_diagnostics,
    }


def _orientation_presets_by_arm_and_family(
    args: argparse.Namespace,
    coord_transform: Any | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw_by_arm = _raw_orientation_presets_by_arm_and_family()
    presets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for arm_side, raw_by_family in raw_by_arm.items():
        presets[arm_side] = {}
        for family_name, raw_presets in raw_by_family.items():
            presets[arm_side][family_name] = [
                _orientation_preset_record(index, label, rpy_values, family_name, coord_transform=coord_transform)
                for index, (label, rpy_values) in enumerate(raw_presets)
            ]
    return presets


def _region_filtered_orientation_presets(
    arm_side: str,
    region: str,
    args: argparse.Namespace,
    coord_transform: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_arm = _raw_orientation_presets_by_arm_and_family()
    if arm_side not in raw_by_arm:
        raise RuntimeError(f"Unsupported arm side for orientation presets: {arm_side}")

    family_order = _approach_family_order_for_region(region)
    debug_fixed_rpy = _debug_fixed_rpy_for_arm(args, arm_side)
    merged: list[tuple[str, list[float], str, bool]] = []
    for family_index, family_name in enumerate(family_order):
        if family_name not in raw_by_arm[arm_side]:
            raise RuntimeError(f"Unsupported approach family for {arm_side}: {family_name}")
        if family_index == 0 and debug_fixed_rpy is not None:
            label, rpy_values = debug_fixed_rpy
            merged.append((label, rpy_values, family_name, True))
        for label, rpy_values in raw_by_arm[arm_side][family_name]:
            merged.append((label, rpy_values, family_name, False))

    presets = [
        _orientation_preset_record(
            index,
            label,
            rpy_values,
            family_name,
            coord_transform=coord_transform,
            debug_override=debug_override,
        )
        for index, (label, rpy_values, family_name, debug_override) in enumerate(merged)
    ]
    return presets, {
        "arm_side": arm_side,
        "target_region": region,
        "approach_family_order": family_order,
        "preset_labels": [preset["preset_label"] for preset in presets],
        "preset_families": [preset["preset_family"] for preset in presets],
        "preset_world_y_axis_diagnostics": [
            {
                "preset_label": preset["preset_label"],
                "preset_family": preset["preset_family"],
                "approach_axis_base": preset["approach_axis_base"],
                "approach_axis_world": preset["approach_axis_world"],
                "AB_axis_world": preset.get("AB_axis_world"),
                "preset_axial_roll_variant_label": preset.get("preset_axial_roll_variant_label"),
                "preset_axial_roll_about_ab_rad": preset.get("preset_axial_roll_about_ab_rad"),
                "dot_with_world_pos_y": preset["dot_with_world_pos_y"],
                "dot_with_world_neg_y": preset["dot_with_world_neg_y"],
            }
            for preset in presets
        ],
        "debug_fixed_rpy_override_active": bool(debug_fixed_rpy is not None),
        "world_y_approach_calibration_note": "world_y_approach RPY values are first-test candidates; axial roll variants are generated by rotating around the local AB/+Z axis and still require GUI verification",
    }


def _fixed_downward_rpy_by_arm(
    ik_solver: Any,
    coord_transform: Any,
    work_area_world: np.ndarray,
    debug_fixed_rpy_by_arm: dict[str, list[float] | None] | None = None,
) -> dict[str, Any]:
    base_down = _base_down_vector(coord_transform)
    work_area_base = np.array(coord_transform.world_to_robot(np.array(work_area_world, dtype=float)), dtype=float)
    debug_fixed_rpy_by_arm = debug_fixed_rpy_by_arm or {}
    result: dict[str, Any] = {
        "source": "official_grasp_planner_z_down_convention_with_fixed_task_area_x_axis",
        "work_area_world": np.array(work_area_world, dtype=float).tolist(),
        "work_area_base": work_area_base.tolist(),
        "base_down": base_down.tolist(),
        "debug_fixed_rpy_override_active": any(value is not None for value in debug_fixed_rpy_by_arm.values()),
        "by_arm": {},
    }
    for side in ("left", "right"):
        ee_se3 = ik_solver.get_ee_pose(side)
        ee_pos = np.array(ee_se3.translation, dtype=float)
        override = debug_fixed_rpy_by_arm.get(side)
        if override is not None:
            rpy = np.array(override, dtype=float)
            rot = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
            result["by_arm"][side] = {
                "source": "cli_debug_fixed_rpy_override",
                "rpy": rpy.tolist(),
                "rotation_matrix": rot.tolist(),
                "x_axis_base": rot[:, 0].tolist(),
                "y_axis_base": rot[:, 1].tolist(),
                "z_axis_base": rot[:, 2].tolist(),
                "startup_ee_position_base": ee_pos.tolist(),
                "x_axis_seed_base": None,
                "debug_override_used": True,
            }
            continue
        x_seed = work_area_base - ee_pos
        x_axis = x_seed - float(np.dot(x_seed, base_down)) * base_down
        if float(np.linalg.norm(x_axis)) <= 1.0e-6:
            x_seed = np.array(ee_se3.rotation, dtype=float)[:, 0]
            x_axis = x_seed - float(np.dot(x_seed, base_down)) * base_down
        x_axis = _normalize(x_axis, np.array([1.0, 0.0, 0.0], dtype=float))
        y_axis = _normalize(np.cross(base_down, x_axis), np.array([0.0, 1.0, 0.0], dtype=float))
        x_axis = _normalize(np.cross(y_axis, base_down), x_axis)
        rot = np.column_stack([x_axis, y_axis, base_down])
        rpy = _rot_to_euler_xyz(rot)
        result["by_arm"][side] = {
            "source": "derived_fixed_downward_from_task_area",
            "rpy": rpy.tolist(),
            "rotation_matrix": rot.tolist(),
            "x_axis_base": x_axis.tolist(),
            "y_axis_base": y_axis.tolist(),
            "z_axis_base": base_down.tolist(),
            "startup_ee_position_base": ee_pos.tolist(),
            "x_axis_seed_base": x_seed.tolist(),
            "debug_override_used": False,
        }
    return result


def _pose_contact_base(
    contact_world: np.ndarray,
    coord_transform: Any,
    rpy: np.ndarray,
    tcp_offset_local: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    rot = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    contact_base = np.array(coord_transform.world_to_robot(np.array(contact_world, dtype=float)), dtype=float)
    contact_base[0] += float(args.contact_base_forward_bias)
    contact_base[1] += float(args.contact_base_lateral_bias)

    tcp_offset_base = rot @ np.array(tcp_offset_local, dtype=float)

    if args.target_mode == "contact_axis":
        ee_origin_base = contact_base.copy()
        tcp_compensation_active = False
    else:
        ee_origin_base = contact_base - tcp_offset_base
        tcp_compensation_active = True

    pose = np.array(
        [ee_origin_base[0], ee_origin_base[1], ee_origin_base[2], rpy[0], rpy[1], rpy[2]],
        dtype=float,
    )
    return pose, {
        "contact_world": np.array(contact_world, dtype=float).tolist(),
        "contact_base": contact_base.tolist(),
        "tcp_offset_base": tcp_offset_base.tolist(),
        "ee_origin_base": ee_origin_base.tolist(),
        "target_mode": args.target_mode,
        "tcp_compensation_active": tcp_compensation_active,
    }


def _pose_position_world(coord_transform: Any, pose_base: np.ndarray) -> np.ndarray:
    return np.array(coord_transform.robot_to_world(np.array(pose_base[:3], dtype=float)), dtype=float)


def _pose_rotation_base(pose_base: np.ndarray) -> np.ndarray:
    pose = np.array(pose_base, dtype=float)
    return _euler_xyz_to_rot(float(pose[3]), float(pose[4]), float(pose[5]))


def _pose_rotation_world(coord_transform: Any, pose_base: np.ndarray) -> np.ndarray:
    return np.array(coord_transform.robot_world_R, dtype=float) @ _pose_rotation_base(pose_base)


def _resolve_point_b_offset_local(tcp_offset_local: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.point_b_offset_local is not None:
        offset = np.array(args.point_b_offset_local, dtype=float)
        source = "cli_point_b_offset_local"
    else:
        tcp = np.array(tcp_offset_local, dtype=float)
        length = max(float(np.linalg.norm(tcp)), float(args.tcp_fallback_x))
        offset = np.array([0.0, 0.0, length], dtype=float)
        source = "inferred_local_z_from_tcp_length"
    return offset, {
        "point_a_definition": "physical EE/sixforce origin used by DualArmIK",
        "point_b_definition": "finger/fingertip proxy at point_a + R_world @ point_b_offset_local",
        "point_b_offset_local": offset.tolist(),
        "source": source,
        "tcp_offset_local_reference": np.array(tcp_offset_local, dtype=float).tolist(),
        "inference_note": "exact finger mesh frame is not resolved here; local +Z is used because current top-down presets make that axis vertical in world",
    }


def _point_world_from_pose(coord_transform: Any, pose_base: np.ndarray, offset_local: np.ndarray) -> np.ndarray:
    point_a_world = _pose_position_world(coord_transform, pose_base)
    rot_world = _pose_rotation_world(coord_transform, pose_base)
    return point_a_world + rot_world @ np.array(offset_local, dtype=float)


def _point_b_world_from_pose(coord_transform: Any, pose_base: np.ndarray, point_b_offset_local: np.ndarray) -> np.ndarray:
    return _point_world_from_pose(coord_transform, pose_base, point_b_offset_local)


def _point_b_target_for_xy_reference(
    point_b_world: np.ndarray,
    xy_reference_target_world: np.ndarray,
    coord_transform: Any,
    rpy: np.ndarray,
    point_b_offset_local: np.ndarray,
    xy_reference_offset_local: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.array(point_b_world, dtype=float).copy()
    desired_xy = np.array(xy_reference_target_world[:2], dtype=float)
    if xy_reference_offset_local is None:
        return target, {
            "vertical_xy_reference_active": False,
            "vertical_xy_reference_target_xy_world": desired_xy.tolist(),
            "vertical_xy_reference_note": "fallback_point_B_xy_used_no_reference_offset",
        }

    rot_base = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    rot_world = np.array(coord_transform.robot_world_R, dtype=float) @ rot_base
    delta_world = rot_world @ (np.array(xy_reference_offset_local, dtype=float) - np.array(point_b_offset_local, dtype=float))
    target[0] = float(desired_xy[0] - delta_world[0])
    target[1] = float(desired_xy[1] - delta_world[1])
    return target, {
        "vertical_xy_reference_active": True,
        "vertical_xy_reference_target_xy_world": desired_xy.tolist(),
        "vertical_point_B_target_xy_adjusted_for_reference": True,
        "vertical_point_B_to_xy_reference_delta_world": delta_world.tolist(),
        "vertical_point_B_xy_shift_world": (target[:2] - np.array(point_b_world[:2], dtype=float)).tolist(),
        "vertical_xy_reference_semantics": "point_B Z is controlled at the contact mark while the vertical XY reference is aligned over the object",
    }


def _pose_for_point_b_world(
    point_b_world: np.ndarray,
    coord_transform: Any,
    rpy: np.ndarray,
    point_b_offset_local: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rot_base = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    point_b_base = np.array(coord_transform.world_to_robot(np.array(point_b_world, dtype=float)), dtype=float)
    point_b_offset_base = rot_base @ np.array(point_b_offset_local, dtype=float)
    point_a_base = point_b_base - point_b_offset_base
    pose = np.array([point_a_base[0], point_a_base[1], point_a_base[2], rpy[0], rpy[1], rpy[2]], dtype=float)
    return pose, {
        "point_b_target_world": np.array(point_b_world, dtype=float).tolist(),
        "point_b_target_base": point_b_base.tolist(),
        "point_b_offset_local": np.array(point_b_offset_local, dtype=float).tolist(),
        "point_b_offset_base": point_b_offset_base.tolist(),
        "point_a_target_base": point_a_base.tolist(),
        "target_semantics": "point_B_proxy_driven",
    }


def _ab_pose_semantics(
    coord_transform: Any,
    pose_base: np.ndarray,
    point_b_offset_local: np.ndarray,
) -> dict[str, Any]:
    point_a_world = _pose_position_world(coord_transform, pose_base)
    point_b_world = _point_b_world_from_pose(coord_transform, pose_base, point_b_offset_local)
    ab_vector_world = point_b_world - point_a_world
    length = float(np.linalg.norm(ab_vector_world))
    ab_axis_world = _normalize(ab_vector_world, np.array([0.0, 0.0, -1.0], dtype=float))
    table_up = np.array([0.0, 0.0, 1.0], dtype=float)
    ab_dot_world_z = float(np.dot(ab_axis_world, table_up))
    ab_horizontal_axis_norm = float(np.linalg.norm(ab_axis_world - ab_dot_world_z * table_up))
    ab_downward_slant_deg = float(math.degrees(math.atan2(-ab_dot_world_z, max(ab_horizontal_axis_norm, 1.0e-9))))
    world_pos_y = np.array([0.0, 1.0, 0.0], dtype=float)
    world_neg_y = np.array([0.0, -1.0, 0.0], dtype=float)
    return {
        "point_A_world": point_a_world.tolist(),
        "point_B_world": point_b_world.tolist(),
        "AB_vector_world": ab_vector_world.tolist(),
        "AB_axis_world": ab_axis_world.tolist(),
        "AB_length_m": length,
        "AB_table_parallel_error_abs_dot_z": abs(ab_dot_world_z),
        "AB_vertical_alignment_abs_dot_z": abs(ab_dot_world_z),
        "AB_dot_with_world_z": ab_dot_world_z,
        "AB_downward_slant_deg": ab_downward_slant_deg,
        "AB_dot_with_world_pos_y": float(np.dot(ab_axis_world, world_pos_y)),
        "AB_dot_with_world_neg_y": float(np.dot(ab_axis_world, world_neg_y)),
    }


def _target_world_from_pose_key(
    geometry: dict[str, Any],
    key: str,
    coord_transform: Any,
    point_b_offset_local: np.ndarray,
) -> dict[str, Any]:
    return _ab_pose_semantics(coord_transform, np.array(geometry[key], dtype=float), point_b_offset_local)


def _compute_contact_z_world(
    bbox_top_z: float,
    table_top_z: float,
    descend_clearance: float,
    grasp_depth_offset: float,
    min_ee_table_clearance: float,
) -> float:
    return max(
        float(bbox_top_z) + float(descend_clearance) + float(grasp_depth_offset),
        float(table_top_z) + float(min_ee_table_clearance),
    )


def _plan_grasp_geometry(
    *,
    object_state: dict[str, Any],
    table_top_z: float,
    bin_bbox: dict[str, list[float]],
    bin_floor_top_z: float,
    coord_transform: Any,
    arm_side: str,
    downward_rpy_by_arm: dict[str, Any],
    tcp_offset_local: np.ndarray,
    args: argparse.Namespace,
    target_region: str = "mid",
    motion_family: str = "z_approach",
    point_b_offset_local: np.ndarray | None = None,
    vertical_xy_reference_offset_local: np.ndarray | None = None,
    vertical_xy_reference_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bbox = object_state["bbox"]
    object_world = _center_from_bbox(bbox)
    contact_world = object_world.copy()
    contact_world[1] += float(args.contact_world_y_bias)
    contact_world[2] = _compute_contact_z_world(
        bbox_top_z=float(bbox["max"][2]),
        table_top_z=table_top_z,
        descend_clearance=args.descend_clearance,
        grasp_depth_offset=args.grasp_depth_offset,
        min_ee_table_clearance=args.min_ee_table_clearance,
    )
    object_support_z_world = max(float(table_top_z), float(bbox["min"][2]))

    rpy = np.array(downward_rpy_by_arm["by_arm"][arm_side]["rpy"], dtype=float)
    base_point_b_offset = np.array(point_b_offset_local if point_b_offset_local is not None else tcp_offset_local, dtype=float)
    point_b_offset = base_point_b_offset.copy()
    rot = _euler_xyz_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    x_axis_base = rot[:, 0]
    y_axis_base = rot[:, 1]
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    base_up = _normalize(np.array(coord_transform.robot_world_R_inv, dtype=float) @ world_up, np.array([0.0, 0.0, 1.0], dtype=float))
    base_ab_vector_world = np.array(coord_transform.robot_world_R, dtype=float) @ (rot @ base_point_b_offset)
    base_ab_horizontal_world = base_ab_vector_world - float(np.dot(base_ab_vector_world, world_up)) * world_up
    far_ab_base_horizontal_span_m = float(np.linalg.norm(base_ab_horizontal_world))
    far_point_a_min_extra_height_clearance = 0.0
    far_point_a_extra_height_clearance = 0.0
    far_ab_requested_downward_slant_deg: float | None = None
    far_ab_slant_height_clearance_m = 0.0
    far_point_b_offset_local_adjustment = np.zeros(3, dtype=float)
    if motion_family == "world_y_approach":
        far_point_a_min_extra_height_clearance = float(args.far_point_a_extra_height_clearance)
        far_ab_requested_downward_slant_deg = float(args.far_ab_downward_slant_deg)
        far_ab_slant_height_clearance_m = far_ab_base_horizontal_span_m * math.tan(math.radians(far_ab_requested_downward_slant_deg))
        far_point_a_extra_height_clearance = max(far_point_a_min_extra_height_clearance, far_ab_slant_height_clearance_m)
        if far_point_a_extra_height_clearance > 0.0:
            desired_b_drop_base = np.array(coord_transform.robot_world_R_inv, dtype=float) @ (-world_up * far_point_a_extra_height_clearance)
            far_point_b_offset_local_adjustment = rot.T @ desired_b_drop_base
            point_b_offset = point_b_offset + far_point_b_offset_local_adjustment
    ab_axis_base = _normalize(rot @ point_b_offset, np.array([0.0, 0.0, -1.0], dtype=float))
    ab_axis_world = _normalize(np.array(coord_transform.robot_world_R, dtype=float) @ ab_axis_base, np.array([0.0, 0.0, -1.0], dtype=float))
    ab_dot_world_z = float(np.dot(ab_axis_world, world_up))
    ab_horizontal_axis_norm = float(np.linalg.norm(ab_axis_world - ab_dot_world_z * world_up))
    far_ab_downward_slant_deg = float(math.degrees(math.atan2(-ab_dot_world_z, max(ab_horizontal_axis_norm, 1.0e-9))))
    robot_belly_forward_world = _normalize(
        np.array(coord_transform.robot_world_R, dtype=float) @ np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    far_point_b_forward_extension = 0.0
    far_reach_axis_world = None
    far_xy_align_b_world = None
    legacy_side_contact_b_world = None
    far_outboard_transition_b_world = None
    far_outboard_axis_world = None

    if motion_family == "world_y_approach":
        motion_policy = "far_low_side_B_driven"
        far_contact_z_world = object_support_z_world + float(args.far_point_b_gap_above_support)
        far_reach_axis_world = _normalize(
            ab_axis_world - float(np.dot(ab_axis_world, world_up)) * world_up,
            np.array([0.0, 1.0, 0.0], dtype=float),
        )
        far_point_b_forward_extension = float(args.far_point_b_forward_extension)
        far_xy_align_z_world = max(
            far_contact_z_world + float(args.far_low_side_clearance),
            float(bbox["max"][2]) + float(args.far_xy_align_clearance_above_object),
        )
        far_xy_align_b_world = np.array(contact_world, dtype=float).copy()
        far_xy_align_b_world[2] = far_xy_align_z_world
        contact_b_world = far_xy_align_b_world.copy()
        contact_b_world[2] = far_contact_z_world
        legacy_side_contact_b_world = contact_b_world + far_reach_axis_world * far_point_b_forward_extension
        low_side_prepare_b_world = legacy_side_contact_b_world - far_reach_axis_world * float(args.pregrasp_standoff)
        low_side_prepare_b_world[2] = far_xy_align_b_world[2]
        side_sign = -1.0 if arm_side == "right" else 1.0
        far_outboard_axis_world = _normalize(
            np.array(coord_transform.robot_world_R, dtype=float) @ np.array([0.0, side_sign, 0.0], dtype=float),
            np.array([side_sign, 0.0, 0.0], dtype=float),
        )
        far_outboard_transition_b_world = (
            low_side_prepare_b_world
            + far_outboard_axis_world * float(args.far_outboard_transition_offset)
            + world_up * float(args.far_outboard_transition_clearance)
        )
        pregrasp_b_world = low_side_prepare_b_world.copy()
        align_b_world = far_xy_align_b_world.copy()
        policy_details = {
            "far_contact_sequence_policy": "outboard_transition_then_low_side_prepare_then_xy_align_then_world_z_descend",
            "far_outboard_transition_B_world": far_outboard_transition_b_world.tolist(),
            "far_outboard_axis_world": far_outboard_axis_world.tolist(),
            "far_outboard_transition_offset_m": float(args.far_outboard_transition_offset),
            "far_outboard_transition_clearance_m": float(args.far_outboard_transition_clearance),
            "far_low_side_prepare_B_world": low_side_prepare_b_world.tolist(),
            "far_xy_align_B_world": far_xy_align_b_world.tolist(),
            "far_descend_B_world": contact_b_world.tolist(),
            "far_B_target_world": contact_b_world.tolist(),
            "far_legacy_side_contact_B_world": legacy_side_contact_b_world.tolist(),
            "far_reach_axis_world": far_reach_axis_world.tolist(),
            "far_point_b_forward_extension_m": far_point_b_forward_extension,
            "far_point_a_extra_height_clearance_m": far_point_a_extra_height_clearance,
            "far_point_a_min_extra_height_clearance_m": far_point_a_min_extra_height_clearance,
            "far_ab_requested_downward_slant_deg": far_ab_requested_downward_slant_deg,
            "far_ab_downward_slant_deg": far_ab_downward_slant_deg,
            "far_ab_slant_height_clearance_m": far_ab_slant_height_clearance_m,
            "far_ab_base_horizontal_span_m": far_ab_base_horizontal_span_m,
            "far_ab_target_slant_range_deg": [5.0, 10.0],
            "far_point_b_offset_local_adjustment": far_point_b_offset_local_adjustment.tolist(),
            "far_low_side_support_z_world": float(object_support_z_world),
            "far_point_b_gap_above_support_m": float(args.far_point_b_gap_above_support),
            "far_low_side_gap_above_support_m": float(args.far_point_b_gap_above_support),
            "far_xy_align_clearance_above_object_m": float(args.far_xy_align_clearance_above_object),
            "far_xy_align_clearance_reference": "max(descend_z + far_low_side_clearance, object_bbox_max_z + far_xy_align_clearance_above_object)",
            "far_xy_align_z_world": float(far_xy_align_b_world[2]),
            "far_descend_z_world": float(far_contact_z_world),
            "far_low_side_contact_z_world": float(far_contact_z_world),
            "far_low_side_clearance_m": float(args.far_low_side_clearance),
            "topdown_reference_contact_z_world": float(contact_world[2]),
            "AB_parallel_to_table_score_abs_dot_z": abs(float(np.dot(ab_axis_world, world_up))),
            "AB_perpendicular_to_robot_belly_abs_dot": abs(float(np.dot(ab_axis_world, robot_belly_forward_world))),
            "strict_AB_parallel_during_final_reach": False,
            "side_push_avoidance": "point_B aligns over object world XY at prepare height before final world-Z descent",
        }
    else:
        motion_policy = "mid_vertical_Z_descend" if target_region == "mid" else "near_body_vertical_Z_descend"
        vertical_contact_z_world = object_support_z_world + float(args.vertical_point_b_gap_above_support)
        vertical_uncorrected_reference_world = np.array([contact_world[0], contact_world[1], vertical_contact_z_world], dtype=float)
        vertical_lateral_correction_base_y = (
            float(args.vertical_arm_lateral_bias_correction)
            if arm_side == "left"
            else -float(args.vertical_arm_lateral_bias_correction)
        )
        vertical_lateral_correction_base = np.array([0.0, vertical_lateral_correction_base_y, 0.0], dtype=float)
        vertical_lateral_correction_world = np.array(coord_transform.robot_world_R, dtype=float) @ vertical_lateral_correction_base
        vertical_xy_reference_world = vertical_uncorrected_reference_world.copy()
        vertical_xy_reference_world[:2] = vertical_xy_reference_world[:2] + vertical_lateral_correction_world[:2]
        raw_contact_b_world = vertical_xy_reference_world.copy()
        contact_b_world, vertical_xy_reference_details = _point_b_target_for_xy_reference(
            raw_contact_b_world,
            vertical_xy_reference_world,
            coord_transform,
            rpy,
            point_b_offset,
            vertical_xy_reference_offset_local,
        )
        pregrasp_b_world = contact_b_world + world_up * float(args.pregrasp_clearance)
        align_b_world = contact_b_world + world_up * float(args.align_clearance)
        low_side_prepare_b_world = None
        policy_details = {
            "vertical_contact_sequence_policy": "vertical_xy_reference_then_point_B_descends_to_support_gap_before_close",
            "vertical_object_world_xy_target": vertical_xy_reference_world[:2].tolist(),
            "vertical_uncorrected_object_world_xy_target": vertical_uncorrected_reference_world[:2].tolist(),
            "vertical_arm_lateral_bias_correction_m": float(args.vertical_arm_lateral_bias_correction),
            "vertical_arm_lateral_bias_correction_base_y_m": float(vertical_lateral_correction_base_y),
            "vertical_arm_lateral_bias_correction_base_vector": vertical_lateral_correction_base.tolist(),
            "vertical_arm_lateral_bias_correction_world": vertical_lateral_correction_world.tolist(),
            "vertical_arm_lateral_bias_correction_rule": "right arm shifts -baseY; left arm shifts +baseY, opposing observed inward vertical grasp bias",
            "vertical_raw_point_B_contact_mark_before_xy_reference": raw_contact_b_world.tolist(),
            "vertical_contact_mark_B_world": contact_b_world.tolist(),
            "vertical_descend_target_B_world": contact_b_world.tolist(),
            "vertical_xy_reference_link_log": vertical_xy_reference_log,
            **vertical_xy_reference_details,
            "vertical_support_z_world": float(object_support_z_world),
            "vertical_point_b_gap_above_support_m": float(args.vertical_point_b_gap_above_support),
            "vertical_close_point_b_tolerance_m": float(args.vertical_close_point_b_tolerance),
            "vertical_descend_axis_world": world_up.tolist(),
            "topdown_reference_contact_z_world": float(contact_world[2]),
            "mid_object_world_xy_target": vertical_xy_reference_world[:2].tolist(),
            "mid_uncorrected_object_world_xy_target": vertical_uncorrected_reference_world[:2].tolist(),
            "mid_descend_target_B_world": contact_b_world.tolist(),
            "mid_AB_vertical_alignment_abs_dot_z": abs(float(np.dot(ab_axis_world, world_up))),
            "mid_descend_axis_world": world_up.tolist(),
            "strict_AB_vertical_during_descend": True,
            "close_after_point_B_contact_gate": True,
        }

    pregrasp_pose, pregrasp_details = _pose_for_point_b_world(pregrasp_b_world, coord_transform, rpy, point_b_offset)
    align_pose, align_details = _pose_for_point_b_world(align_b_world, coord_transform, rpy, point_b_offset)
    contact_pose, contact_details = _pose_for_point_b_world(contact_b_world, coord_transform, rpy, point_b_offset)
    far_outboard_transition_pose = None
    far_outboard_transition_details = None
    if far_outboard_transition_b_world is not None:
        far_outboard_transition_pose, far_outboard_transition_details = _pose_for_point_b_world(
            far_outboard_transition_b_world, coord_transform, rpy, point_b_offset
        )
    micro_lift_b_world = contact_b_world + world_up * float(args.micro_lift_probe)
    lift_b_world = contact_b_world + world_up * float(args.lift_height)
    micro_lift_pose, micro_lift_details = _pose_for_point_b_world(micro_lift_b_world, coord_transform, rpy, point_b_offset)
    lift_pose, lift_details = _pose_for_point_b_world(lift_b_world, coord_transform, rpy, point_b_offset)

    bin_center = np.array(bin_bbox["center"], dtype=float)
    carry_contact_world = np.array([bin_center[0], bin_center[1], float(bin_bbox["max"][2]) + float(args.safe_drop_height)], dtype=float)
    carry_pose, carry_details = _pose_for_point_b_world(carry_contact_world, coord_transform, rpy, point_b_offset)
    place_contact_world = np.array(
        [
            bin_center[0],
            bin_center[1],
            max(float(bin_floor_top_z) + 0.12, float(bin_bbox["max"][2]) + float(args.place_clearance)),
        ],
        dtype=float,
    )
    place_pose, place_details = _pose_for_point_b_world(place_contact_world, coord_transform, rpy, point_b_offset)
    retreat_pose = place_pose.copy()
    retreat_pose[:3] = retreat_pose[:3] + base_up * float(args.retreat_lift)

    return {
        "arm_side": arm_side,
        "target_region": target_region,
        "motion_family": motion_family,
        "motion_policy": motion_policy,
        "object_center_world": object_world.tolist(),
        "bbox_top_z": float(bbox["max"][2]),
        "object_support_z_world": float(object_support_z_world),
        "topdown_reference_contact_z_world": float(contact_world[2]),
        "contact_z_world": float(contact_b_world[2]),
        "fixed_downward_rpy_base": rpy.tolist(),
        "fixed_downward_rotation_base": rot.tolist(),
        "x_axis_base": x_axis_base.tolist(),
        "y_axis_base": y_axis_base.tolist(),
        "base_up": base_up.tolist(),
        "up_axis_base": base_up.tolist(),
        "construction_policy": f"AB_point_B_proxy_{motion_policy}_with_{args.target_mode}_runtime_mode",
        "point_A_proxy_definition": "DualArmIK physical EE/sixforce origin",
        "point_B_proxy_definition": "point A plus selected local point_b_offset, used as fingertip/contact proxy",
        "base_point_b_offset_local": base_point_b_offset.tolist(),
        "point_b_offset_local": point_b_offset.tolist(),
        "far_point_b_offset_local_adjustment": far_point_b_offset_local_adjustment.tolist(),
        "far_point_a_extra_height_clearance_m": float(far_point_a_extra_height_clearance),
        "far_point_a_min_extra_height_clearance_m": float(far_point_a_min_extra_height_clearance),
        "far_ab_requested_downward_slant_deg": far_ab_requested_downward_slant_deg,
        "far_ab_downward_slant_deg": float(far_ab_downward_slant_deg),
        "far_ab_slant_height_clearance_m": float(far_ab_slant_height_clearance_m),
        "far_ab_base_horizontal_span_m": float(far_ab_base_horizontal_span_m),
        "far_point_b_forward_extension_m": float(far_point_b_forward_extension),
        "far_reach_axis_world": None if far_reach_axis_world is None else far_reach_axis_world.tolist(),
        "far_contact_sequence_policy": "outboard_transition_then_low_side_prepare_then_xy_align_then_world_z_descend" if motion_family == "world_y_approach" else None,
        "far_outboard_transition_B_world": None if far_outboard_transition_b_world is None else far_outboard_transition_b_world.tolist(),
        "far_outboard_transition_pose_base": None if far_outboard_transition_pose is None else far_outboard_transition_pose.tolist(),
        "far_outboard_transition_details": far_outboard_transition_details,
        "far_outboard_axis_world": None if far_outboard_axis_world is None else far_outboard_axis_world.tolist(),
        "far_outboard_transition_offset_m": float(args.far_outboard_transition_offset) if motion_family == "world_y_approach" else None,
        "far_outboard_transition_clearance_m": float(args.far_outboard_transition_clearance) if motion_family == "world_y_approach" else None,
        "far_xy_align_B_world": None if far_xy_align_b_world is None else far_xy_align_b_world.tolist(),
        "far_descend_B_world": contact_b_world.tolist() if motion_family == "world_y_approach" else None,
        "far_legacy_side_contact_B_world": None if legacy_side_contact_b_world is None else legacy_side_contact_b_world.tolist(),
        "far_point_b_gap_above_support_m": float(args.far_point_b_gap_above_support) if motion_family == "world_y_approach" else None,
        "far_xy_align_clearance_above_object_m": float(args.far_xy_align_clearance_above_object) if motion_family == "world_y_approach" else None,
        "vertical_contact_sequence_policy": "vertical_xy_reference_then_point_B_descends_to_support_gap_before_close" if motion_family != "world_y_approach" else None,
        "vertical_contact_mark_B_world": contact_b_world.tolist() if motion_family != "world_y_approach" else None,
        "vertical_uncorrected_object_world_xy_target": vertical_uncorrected_reference_world[:2].tolist() if motion_family != "world_y_approach" else None,
        "vertical_arm_lateral_bias_correction_m": float(args.vertical_arm_lateral_bias_correction) if motion_family != "world_y_approach" else None,
        "vertical_arm_lateral_bias_correction_base_y_m": float(vertical_lateral_correction_base_y) if motion_family != "world_y_approach" else None,
        "vertical_arm_lateral_bias_correction_base_vector": vertical_lateral_correction_base.tolist() if motion_family != "world_y_approach" else None,
        "vertical_arm_lateral_bias_correction_world": vertical_lateral_correction_world.tolist() if motion_family != "world_y_approach" else None,
        "vertical_arm_lateral_bias_correction_rule": "right arm shifts -baseY; left arm shifts +baseY, opposing observed inward vertical grasp bias" if motion_family != "world_y_approach" else None,
        "vertical_raw_point_B_contact_mark_before_xy_reference": raw_contact_b_world.tolist() if motion_family != "world_y_approach" else None,
        "vertical_xy_reference_link_log": vertical_xy_reference_log if motion_family != "world_y_approach" else None,
        "vertical_xy_reference_mode": None if motion_family == "world_y_approach" or vertical_xy_reference_log is None else vertical_xy_reference_log.get("reference_mode", "single_reference_link"),
        "vertical_xy_reference_source": None if motion_family == "world_y_approach" or vertical_xy_reference_log is None else vertical_xy_reference_log.get("source"),
        "vertical_xy_reference_world_position_used_for_offset": None if motion_family == "world_y_approach" or vertical_xy_reference_log is None else vertical_xy_reference_log.get("reference_world_position_used_for_offset", vertical_xy_reference_log.get("world_position")),
        "vertical_xy_reference_component_logs": None if motion_family == "world_y_approach" or vertical_xy_reference_log is None else vertical_xy_reference_log.get("component_reference_logs"),
        "vertical_xy_reference_offset_local": None if motion_family == "world_y_approach" or vertical_xy_reference_offset_local is None else np.array(vertical_xy_reference_offset_local, dtype=float).tolist(),
        "vertical_xy_reference_active": bool(motion_family != "world_y_approach" and vertical_xy_reference_offset_local is not None),
        "vertical_xy_reference_tolerance_m": float(args.vertical_xy_reference_tolerance) if motion_family != "world_y_approach" else None,
        "vertical_xy_reference_target_xy_world": vertical_xy_reference_world[:2].tolist() if motion_family != "world_y_approach" else None,
        "vertical_point_b_gap_above_support_m": float(args.vertical_point_b_gap_above_support) if motion_family != "world_y_approach" else None,
        "vertical_close_point_b_tolerance_m": float(args.vertical_close_point_b_tolerance) if motion_family != "world_y_approach" else None,
        "tcp_offset_local": np.array(tcp_offset_local, dtype=float).tolist(),
        "robot_belly_forward_world": robot_belly_forward_world.tolist(),
        "AB_axis_base": ab_axis_base.tolist(),
        "AB_axis_world": ab_axis_world.tolist(),
        "AB_dot_with_world_z": ab_dot_world_z,
        "AB_downward_slant_deg": float(far_ab_downward_slant_deg),
        "AB_dot_with_world_pos_y": float(np.dot(ab_axis_world, np.array([0.0, 1.0, 0.0], dtype=float))),
        "AB_dot_with_world_neg_y": float(np.dot(ab_axis_world, np.array([0.0, -1.0, 0.0], dtype=float))),
        "AB_dot_with_robot_belly_forward": float(np.dot(ab_axis_world, robot_belly_forward_world)),
        "pregrasp_contact_world": pregrasp_b_world.tolist(),
        "align_contact_world": align_b_world.tolist(),
        "contact_point_world": contact_b_world.tolist(),
        "pregrasp_point_B_world": pregrasp_b_world.tolist(),
        "align_point_B_world": align_b_world.tolist(),
        "contact_point_B_world": contact_b_world.tolist(),
        "micro_lift_point_B_world": micro_lift_b_world.tolist(),
        "lift_point_B_world": lift_b_world.tolist(),
        "far_low_side_prepare_B_world": None if low_side_prepare_b_world is None else low_side_prepare_b_world.tolist(),
        "far_xy_align_pose_base": align_pose.tolist() if motion_family == "world_y_approach" else None,
        "far_descend_pose_base": contact_pose.tolist() if motion_family == "world_y_approach" else None,
        "pregrasp_pose_base": pregrasp_pose.tolist(),
        "align_pose_base": align_pose.tolist(),
        "contact_pose_base": contact_pose.tolist(),
        "micro_lift_pose_base": micro_lift_pose.tolist(),
        "lift_pose_base": lift_pose.tolist(),
        "carry_pose_base": carry_pose.tolist(),
        "place_pose_base": place_pose.tolist(),
        "retreat_pose_base": retreat_pose.tolist(),
        "pregrasp_ee_origin_world": _pose_position_world(coord_transform, pregrasp_pose).tolist(),
        "align_ee_origin_world": _pose_position_world(coord_transform, align_pose).tolist(),
        "contact_ee_origin_world": _pose_position_world(coord_transform, contact_pose).tolist(),
        "micro_lift_ee_origin_world": _pose_position_world(coord_transform, micro_lift_pose).tolist(),
        "lift_ee_origin_world": _pose_position_world(coord_transform, lift_pose).tolist(),
        "carry_ee_origin_world": _pose_position_world(coord_transform, carry_pose).tolist(),
        "place_ee_origin_world": _pose_position_world(coord_transform, place_pose).tolist(),
        "retreat_ee_origin_world": _pose_position_world(coord_transform, retreat_pose).tolist(),
        "pregrasp_AB_semantics": _ab_pose_semantics(coord_transform, pregrasp_pose, point_b_offset),
        "align_AB_semantics": _ab_pose_semantics(coord_transform, align_pose, point_b_offset),
        "contact_AB_semantics": _ab_pose_semantics(coord_transform, contact_pose, point_b_offset),
        "lift_AB_semantics": _ab_pose_semantics(coord_transform, lift_pose, point_b_offset),
        "motion_policy_details": policy_details,
        "contact_details": contact_details,
        "pregrasp_details": pregrasp_details,
        "align_details": align_details,
        "micro_lift_details": micro_lift_details,
        "lift_details": lift_details,
        "carry_contact_world": carry_contact_world.tolist(),
        "place_contact_world": place_contact_world.tolist(),
        "carry_details": carry_details,
        "place_details": place_details,
        "approach_axis_mode": args.approach_axis_mode,
        "approach_axis_base": ab_axis_base.tolist(),
    }


def _plan_grasp_geometry_for_preset(
    *,
    object_state: dict[str, Any],
    table_top_z: float,
    bin_bbox: dict[str, list[float]],
    bin_floor_top_z: float,
    coord_transform: Any,
    arm_side: str,
    preset: dict[str, Any],
    tcp_offset_local: np.ndarray,
    args: argparse.Namespace,
    target_region: str,
    point_b_offset_local: np.ndarray,
    vertical_xy_reference_offset_local: np.ndarray | None = None,
    vertical_xy_reference_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geometry = _plan_grasp_geometry(
        object_state=object_state,
        table_top_z=table_top_z,
        bin_bbox=bin_bbox,
        bin_floor_top_z=bin_floor_top_z,
        coord_transform=coord_transform,
        arm_side=arm_side,
        downward_rpy_by_arm={"by_arm": {arm_side: {"rpy": preset["rpy"]}}},
        tcp_offset_local=tcp_offset_local,
        args=args,
        target_region=target_region,
        motion_family=str(preset.get("preset_family", "z_approach")),
        point_b_offset_local=point_b_offset_local,
        vertical_xy_reference_offset_local=vertical_xy_reference_offset_local,
        vertical_xy_reference_log=vertical_xy_reference_log,
    )
    geometry["orientation_source"] = "fixed_orientation_preset_library"
    geometry["orientation_preset"] = preset
    return geometry


def _pregrasp_candidates(geometry: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    base = np.array(geometry["pregrasp_pose_base"], dtype=float)
    base_up = np.array(geometry["base_up"], dtype=float)

    variants = [
        ("nominal", 0.0),
        ("slightly_higher", float(args.candidate_higher_offset)),
    ]

    candidates: list[dict[str, Any]] = []
    for index, (label, higher) in enumerate(variants):
        pose = base.copy()
        pose[:3] = pose[:3] + base_up * higher
        candidates.append(
            {
                "candidate_index": index,
                "candidate_label": label,
                "pose_base": pose.tolist(),
                "higher_offset_m": float(higher),
            }
        )
    return candidates


def _sync_ik_from_dc(ik_solver: Any, dc: Any, articulation: Any) -> tuple[list[str], list[float]]:
    names, positions = _all_joint_state_for_ik(dc, articulation)
    ik_solver.sync_joint_positions(names, positions)
    return names, positions


def _current_ee_pose_base(
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    arm_side: str,
    args: argparse.Namespace | None = None,
) -> np.ndarray:
    _sync_ik_from_dc(ik_solver, dc, articulation)
    return _ee_pose_base_from_ik_state(ik_solver, arm_side, args=args)

def _ee_pose_base_for_arm_solution(
    ik_solver: Any,
    arm_side: str,
    arm_solution_q: np.ndarray,
    args: argparse.Namespace | None = None,
) -> np.ndarray:
    q_saved = np.array(ik_solver.q, dtype=float).copy()
    q_trial = q_saved.copy()

    arm_joint_names = list(
        ik_solver.RIGHT_ARM_JOINTS if arm_side == "right" else ik_solver.LEFT_ARM_JOINTS
    )

    name_to_q_index = {}
    for joint_name in arm_joint_names:
        joint_id = ik_solver.model.getJointId(joint_name)
        q_index = int(ik_solver.model.joints[joint_id].idx_q)
        name_to_q_index[joint_name] = q_index

    try:
        for joint_name, value in zip(arm_joint_names, arm_solution_q):
            q_trial[name_to_q_index[joint_name]] = float(value)
        ik_solver.q = q_trial
        return _ee_pose_base_from_ik_state(ik_solver, arm_side, args=args)
    finally:
        ik_solver.q = q_saved

def _pose_error(ik_solver: Any, current_pose_base: np.ndarray, target_pose_base: np.ndarray) -> tuple[float, float]:
    try:
        import pinocchio as pin  # type: ignore

        current = ik_solver.xyzrpy_to_se3(current_pose_base)
        target = ik_solver.xyzrpy_to_se3(target_pose_base)
        error = pin.log(current.actInv(target)).vector
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    except Exception:
        pos_err = float(np.linalg.norm(np.array(target_pose_base[:3], dtype=float) - np.array(current_pose_base[:3], dtype=float)))
        rot_delta = np.array(target_pose_base[3:], dtype=float) - np.array(current_pose_base[3:], dtype=float)
        rot_delta = (rot_delta + math.pi) % (2.0 * math.pi) - math.pi
        return pos_err, float(np.linalg.norm(rot_delta))


def _limit_joint_delta(
    q_current: np.ndarray,
    q_target: np.ndarray,
    blend: float,
    max_step_norm: float,
    max_abs_step: float,
) -> np.ndarray:
    dq = (np.array(q_target, dtype=float) - np.array(q_current, dtype=float)) * float(blend)
    if max_abs_step > 0.0:
        dq = np.clip(dq, -float(max_abs_step), float(max_abs_step))
    norm = float(np.linalg.norm(dq))
    if norm > float(max_step_norm):
        dq *= float(max_step_norm) / max(norm, 1.0e-9)
    return np.array(q_current, dtype=float) + dq


def _ik_kwargs(args: argparse.Namespace, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    kwargs = {
        "max_iter": int(args.ik_max_iter),
        "pos_tol": float(args.ik_pos_tol),
        "rot_tol": float(args.ik_rot_tol),
        "damping": float(args.ik_damping),
        "dq_max": float(args.ik_dq_max),
        "pos_weight": 1.0,
        "rot_weight": float(args.ik_rot_weight),
        "null_weight": float(args.ik_null_weight),
    }
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _solve_single_arm_pose(
    ik_solver: Any,
    arm_side: str,
    target_pose_base: np.ndarray,
    args: argparse.Namespace,
    ik_overrides: dict[str, Any] | None = None,
    update_fail_count: bool = True,
) -> tuple[np.ndarray, bool]:
    target_se3 = _ik_target_pose_se3(ik_solver, arm_side, np.array(target_pose_base, dtype=float), args)
    q_sol, ok = ik_solver.solve_ik_single_arm(target_se3, arm_side, **_ik_kwargs(args, ik_overrides))
    if update_fail_count and hasattr(ik_solver, "_update_fail_count"):
        ik_solver._update_fail_count(arm_side, bool(ok))
    return np.array(q_sol, dtype=float), bool(ok)


def _candidate_ik_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_iter": int(args.candidate_ik_max_iter),
        "pos_tol": float(args.candidate_ik_pos_tol),
        "rot_tol": float(args.candidate_ik_rot_tol),
        "null_weight": 0.0,
    }


def _finite_vector(value: Any, expected_size: int | None = None) -> bool:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return False
    if expected_size is not None and arr.size != expected_size:
        return False
    return bool(arr.size > 0 and np.isfinite(arr).all())


def _candidate_failure_summary(candidate_results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "catastrophic_no_solution_count": sum(1 for result in candidate_results if result["candidate_classification"] == "catastrophic_no_solution"),
        "solved_but_not_internal_ok_count": sum(1 for result in candidate_results if result["candidate_classification"] == "solved_but_not_internal_ok"),
        "candidate_error_exceeded_tolerance_count": sum(1 for result in candidate_results if result["candidate_classification"] == "candidate_error_exceeded_tolerance"),
        "valid_candidate_count": sum(1 for result in candidate_results if result["candidate_classification"] == "valid_candidate"),
        "candidate_count": len(candidate_results),
    }


def _candidate_acceptance_tolerances(target_region: str, args: argparse.Namespace) -> tuple[float, float, dict[str, Any]]:
    far_candidate_gate_active = target_region == "far"
    if far_candidate_gate_active:
        pos_tol = float(args.far_candidate_position_tolerance)
        rot_tol = float(args.far_candidate_rotation_tolerance)
        pos_source = "far_candidate_position_tolerance"
        rot_source = "far_candidate_rotation_tolerance"
    else:
        pos_tol = float(args.pregrasp_tolerance)
        rot_tol = float(args.rot_tolerance)
        pos_source = "pregrasp_tolerance"
        rot_source = "rot_tolerance"

    if args.debug_pregrasp_pos_tol is not None:
        pos_tol = float(args.debug_pregrasp_pos_tol)
        pos_source = "debug_pregrasp_pos_tol"
    if args.debug_pregrasp_rot_tol is not None:
        rot_tol = float(args.debug_pregrasp_rot_tol)
        rot_source = "debug_pregrasp_rot_tol"

    return pos_tol, rot_tol, {
        "target_region": target_region,
        "far_candidate_gate_active": far_candidate_gate_active,
        "position_tolerance_source": pos_source,
        "rotation_tolerance_source": rot_source,
        "position_tolerance_m": pos_tol,
        "rotation_tolerance_rad": rot_tol,
        "mid_behavior_unchanged": bool(not far_candidate_gate_active),
        "forced_acceptance_active": False,
    }


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def _candidate_gate_metrics(
    position_error: float,
    rotation_error: float,
    position_tolerance: float,
    rotation_tolerance: float,
) -> dict[str, Any]:
    pos_err = _finite_float_or_none(position_error)
    rot_err = _finite_float_or_none(rotation_error)
    if pos_err is None or rot_err is None:
        return {
            "position_error_margin_to_candidate_gate_m": None,
            "rotation_error_margin_to_candidate_gate_rad": None,
            "position_error_excess_over_candidate_gate_m": None,
            "rotation_error_excess_over_candidate_gate_rad": None,
            "combined_gate_error_norm": None,
            "combined_gate_excess_norm": None,
        }

    pos_margin = float(position_tolerance) - pos_err
    rot_margin = float(rotation_tolerance) - rot_err
    pos_ratio = pos_err / max(float(position_tolerance), 1.0e-9)
    rot_ratio = rot_err / max(float(rotation_tolerance), 1.0e-9)
    pos_excess_ratio = max(pos_ratio - 1.0, 0.0)
    rot_excess_ratio = max(rot_ratio - 1.0, 0.0)
    return {
        "position_error_margin_to_candidate_gate_m": pos_margin,
        "rotation_error_margin_to_candidate_gate_rad": rot_margin,
        "position_error_excess_over_candidate_gate_m": max(pos_err - float(position_tolerance), 0.0),
        "rotation_error_excess_over_candidate_gate_rad": max(rot_err - float(rotation_tolerance), 0.0),
        "combined_gate_error_norm": float(math.sqrt(pos_ratio * pos_ratio + rot_ratio * rot_ratio)),
        "combined_gate_excess_norm": float(math.sqrt(pos_excess_ratio * pos_excess_ratio + rot_excess_ratio * rot_excess_ratio)),
    }


def _candidate_diagnostic_view(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    keys = (
        "candidate_index",
        "candidate_label",
        "preset_index",
        "preset_label",
        "preset_family",
        "success",
        "failure_reason",
        "candidate_classification",
        "dualarmik_success",
        "dualarmik_internal_ok",
        "q_solution_valid",
        "reached_pose_valid",
        "error_finite",
        "position_error_base_m",
        "rotation_error_rad",
        "candidate_position_tolerance_m",
        "candidate_rotation_tolerance_rad",
        "position_error_margin_to_candidate_gate_m",
        "rotation_error_margin_to_candidate_gate_rad",
        "position_error_excess_over_candidate_gate_m",
        "rotation_error_excess_over_candidate_gate_rad",
        "combined_gate_error_norm",
        "combined_gate_excess_norm",
        "approach_axis_world",
        "AB_axis_world",
        "preset_axial_roll_variant_label",
        "preset_axial_roll_about_ab_rad",
        "far_point_b_forward_extension_m",
        "far_point_a_extra_height_clearance_m",
        "far_point_a_min_extra_height_clearance_m",
        "far_ab_requested_downward_slant_deg",
        "far_ab_downward_slant_deg",
        "far_ab_slant_height_clearance_m",
        "far_ab_base_horizontal_span_m",
        "far_point_b_gap_above_support_m",
        "far_xy_align_clearance_above_object_m",
        "vertical_contact_mark_B_world",
        "vertical_raw_point_B_contact_mark_before_xy_reference",
        "vertical_point_b_gap_above_support_m",
        "vertical_close_point_b_tolerance_m",
        "vertical_xy_reference_active",
        "vertical_xy_reference_mode",
        "vertical_xy_reference_source",
        "vertical_xy_reference_world_position_used_for_offset",
        "vertical_xy_reference_component_logs",
        "vertical_xy_reference_target_xy_world",
        "vertical_uncorrected_object_world_xy_target",
        "vertical_arm_lateral_bias_correction_m",
        "vertical_arm_lateral_bias_correction_base_y_m",
        "vertical_arm_lateral_bias_correction_world",
        "vertical_arm_lateral_bias_correction_rule",
        "vertical_xy_reference_offset_local",
        "far_reach_axis_world",
        "dot_with_world_pos_y",
        "dot_with_world_neg_y",
        "far_low_side_prepare_B_world",
        "far_xy_align_B_world",
        "far_descend_B_world",
        "contact_point_B_world",
        "target_point_A_world",
        "target_point_B_world",
        "target_pose_base",
        "reached_pose_base",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def _best_candidate_by_metric(candidate_results: list[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    finite_candidates = [
        candidate
        for candidate in candidate_results
        if _finite_float_or_none(candidate.get(metric_key)) is not None
    ]
    if not finite_candidates:
        return None
    best = min(finite_candidates, key=lambda candidate: float(candidate[metric_key]))
    return _candidate_diagnostic_view(best)


def _best_candidate_diagnostics(candidate_results: list[dict[str, Any]], target_region: str) -> dict[str, Any]:
    return {
        "target_region": target_region,
        "far_diagnostics_active": bool(target_region == "far"),
        "scope": "flattened_pregrasp_candidate_results",
        "combined_error_metric": "sqrt((position_error / candidate_position_tolerance)^2 + (rotation_error / candidate_rotation_tolerance)^2)",
        "best_position_error_candidate": _best_candidate_by_metric(candidate_results, "position_error_base_m"),
        "best_rotation_error_candidate": _best_candidate_by_metric(candidate_results, "rotation_error_rad"),
        "best_combined_error_candidate": _best_candidate_by_metric(candidate_results, "combined_gate_error_norm"),
    }


def _classify_candidate_result(
    *,
    q_solution_valid: bool,
    reached_pose_valid: bool,
    error_finite: bool,
    dualarmik_internal_ok: bool,
    position_error: float,
    rotation_error: float,
    position_tolerance: float,
    rotation_tolerance: float,
) -> str:
    if not q_solution_valid or not reached_pose_valid or not error_finite:
        return "catastrophic_no_solution"
    if position_error > position_tolerance or rotation_error > rotation_tolerance:
        return "candidate_error_exceeded_tolerance"
    if not dualarmik_internal_ok:
        return "solved_but_not_internal_ok"
    return "valid_candidate"



def _evaluate_pregrasp_candidates(
    *,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    arm_side: str,
    coord_transform: Any,
    object_state: dict[str, Any],
    table_top_z: float,
    bin_bbox: dict[str, list[float]],
    bin_floor_top_z: float,
    tcp_offset_local: np.ndarray,
    orientation_presets: list[dict[str, Any]],
    target_region: str,
    point_b_offset_local: np.ndarray,
    vertical_xy_reference_offset_local: np.ndarray | None,
    vertical_xy_reference_log: dict[str, Any] | None,
    forward_base: float,
    approach_family_order: list[str],
    args: argparse.Namespace,
    phase_log: list[dict[str, Any]],
    counter: dict[str, int],
) -> dict[str, Any]:
    start_step = counter["step"]
    _sync_ik_from_dc(ik_solver, dc, articulation)
    reference_q = ik_solver.q.copy()
    reference_pose = _ee_pose_base_from_ik_state(ik_solver, arm_side, args=args)
    candidate_pos_tol, candidate_rot_tol, candidate_tolerance_policy = _candidate_acceptance_tolerances(target_region, args)
    preset_results: list[dict[str, Any]] = []
    flat_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    fallback_selection_records: list[dict[str, Any]] = []
    candidate_ik_settings = _candidate_ik_overrides(args)

    for preset in orientation_presets:
        preset_geometry = _plan_grasp_geometry_for_preset(
            object_state=object_state,
            table_top_z=table_top_z,
            bin_bbox=bin_bbox,
            bin_floor_top_z=bin_floor_top_z,
            coord_transform=coord_transform,
            arm_side=arm_side,
            preset=preset,
            tcp_offset_local=tcp_offset_local,
            args=args,
            target_region=target_region,
            point_b_offset_local=point_b_offset_local,
            vertical_xy_reference_offset_local=vertical_xy_reference_offset_local,
            vertical_xy_reference_log=vertical_xy_reference_log,
        )
        candidate_point_b_offset = np.array(preset_geometry.get("point_b_offset_local", point_b_offset_local), dtype=float)
        candidate_results: list[dict[str, Any]] = []
        for candidate in _pregrasp_candidates(preset_geometry, args):
            ik_solver.q = reference_q.copy()
            target_pose = np.array(candidate["pose_base"], dtype=float)
            target_pose_for_ik = np.array(ik_solver.se3_to_xyzrpy(_ik_target_pose_se3(ik_solver, arm_side, target_pose, args)), dtype=float)
            q_sol, ok = _solve_single_arm_pose(
                ik_solver,
                arm_side,
                target_pose,
                args,
                ik_overrides=candidate_ik_settings,
                update_fail_count=False,
            )
            reached_pose = _ee_pose_base_for_arm_solution(ik_solver, arm_side, q_sol, args=args)
            pos_err, rot_err = _pose_error(ik_solver, reached_pose, target_pose)
            q_solution_valid = _finite_vector(q_sol)
            reached_pose_valid = _finite_vector(reached_pose, expected_size=6)
            error_finite = bool(math.isfinite(float(pos_err)) and math.isfinite(float(rot_err)))
            candidate_classification = _classify_candidate_result(
                q_solution_valid=q_solution_valid,
                reached_pose_valid=reached_pose_valid,
                error_finite=error_finite,
                dualarmik_internal_ok=bool(ok),
                position_error=pos_err,
                rotation_error=rot_err,
                position_tolerance=candidate_pos_tol,
                rotation_tolerance=candidate_rot_tol,
            )

            valid = (
                q_solution_valid
                and reached_pose_valid
                and error_finite
                and pos_err <= candidate_pos_tol
                and rot_err <= candidate_rot_tol
            )

            if valid:
                candidate_classification = "valid_candidate"
                candidate_failure_reason = None
            else:
                candidate_failure_reason = candidate_classification
            gate_metrics = _candidate_gate_metrics(pos_err, rot_err, candidate_pos_tol, candidate_rot_tol)
            result = {
                **candidate,
                "success": valid,
                "failure_reason": candidate_failure_reason,
                "candidate_classification": candidate_classification,
                "preset_label": preset["preset_label"],
                "preset_index": int(preset["preset_index"]),
                "preset_family": preset.get("preset_family"),
                "preset_rpy": preset["rpy"],
                "approach_axis_base": preset.get("approach_axis_base"),
                "approach_axis_world": preset.get("approach_axis_world"),
                "AB_axis_base": preset_geometry.get("AB_axis_base"),
                "AB_axis_world": preset_geometry.get("AB_axis_world"),
                "preset_axial_roll_variant_label": preset.get("preset_axial_roll_variant_label"),
                "preset_axial_roll_about_ab_rad": preset.get("preset_axial_roll_about_ab_rad"),
                "far_point_b_forward_extension_m": preset_geometry.get("far_point_b_forward_extension_m"),
                "far_point_a_extra_height_clearance_m": preset_geometry.get("far_point_a_extra_height_clearance_m"),
                "far_point_a_min_extra_height_clearance_m": preset_geometry.get("far_point_a_min_extra_height_clearance_m"),
                "far_ab_requested_downward_slant_deg": preset_geometry.get("far_ab_requested_downward_slant_deg"),
                "far_ab_downward_slant_deg": preset_geometry.get("far_ab_downward_slant_deg"),
                "far_ab_slant_height_clearance_m": preset_geometry.get("far_ab_slant_height_clearance_m"),
                "far_ab_base_horizontal_span_m": preset_geometry.get("far_ab_base_horizontal_span_m"),
                "far_point_b_gap_above_support_m": preset_geometry.get("far_point_b_gap_above_support_m"),
                "far_xy_align_clearance_above_object_m": preset_geometry.get("far_xy_align_clearance_above_object_m"),
                "vertical_contact_mark_B_world": preset_geometry.get("vertical_contact_mark_B_world"),
                "vertical_point_b_gap_above_support_m": preset_geometry.get("vertical_point_b_gap_above_support_m"),
                "vertical_close_point_b_tolerance_m": preset_geometry.get("vertical_close_point_b_tolerance_m"),
                "vertical_xy_reference_active": preset_geometry.get("vertical_xy_reference_active"),
                "vertical_xy_reference_mode": preset_geometry.get("vertical_xy_reference_mode"),
                "vertical_xy_reference_source": preset_geometry.get("vertical_xy_reference_source"),
                "vertical_xy_reference_world_position_used_for_offset": preset_geometry.get("vertical_xy_reference_world_position_used_for_offset"),
                "vertical_xy_reference_component_logs": preset_geometry.get("vertical_xy_reference_component_logs"),
                "vertical_xy_reference_target_xy_world": preset_geometry.get("vertical_xy_reference_target_xy_world"),
                "vertical_uncorrected_object_world_xy_target": preset_geometry.get("vertical_uncorrected_object_world_xy_target"),
                "vertical_arm_lateral_bias_correction_m": preset_geometry.get("vertical_arm_lateral_bias_correction_m"),
                "vertical_arm_lateral_bias_correction_base_y_m": preset_geometry.get("vertical_arm_lateral_bias_correction_base_y_m"),
                "vertical_arm_lateral_bias_correction_world": preset_geometry.get("vertical_arm_lateral_bias_correction_world"),
                "vertical_arm_lateral_bias_correction_rule": preset_geometry.get("vertical_arm_lateral_bias_correction_rule"),
                "vertical_xy_reference_offset_local": preset_geometry.get("vertical_xy_reference_offset_local"),
                "vertical_raw_point_B_contact_mark_before_xy_reference": preset_geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference"),
                "far_reach_axis_world": preset_geometry.get("far_reach_axis_world"),
                "dot_with_world_pos_y": preset.get("dot_with_world_pos_y"),
                "dot_with_world_neg_y": preset.get("dot_with_world_neg_y"),
                "far_low_side_prepare_B_world": preset_geometry.get("far_low_side_prepare_B_world"),
                "far_xy_align_B_world": preset_geometry.get("far_xy_align_B_world"),
                "far_descend_B_world": preset_geometry.get("far_descend_B_world"),
                "contact_point_B_world": preset_geometry.get("contact_point_B_world"),
                "reference_pose_base": reference_pose.tolist(),
                "target_pose_base": target_pose.tolist(),
                "target_pose_for_dualarmik_base": target_pose_for_ik.tolist(),
                "ee_frame_compensation_active": bool(getattr(args, "ee_frame_compensation_active", False)),
                "dualarmik_success": bool(ok),
                "dualarmik_internal_ok": bool(ok),
                "q_solution_valid": q_solution_valid,
                "reached_pose_valid": reached_pose_valid,
                "error_finite": error_finite,
                "position_error_base_m": pos_err,
                "rotation_error_rad": rot_err,
                "candidate_position_tolerance_m": candidate_pos_tol,
                "candidate_rotation_tolerance_rad": candidate_rot_tol,
                "candidate_tolerance_policy": candidate_tolerance_policy,
                **gate_metrics,
                "candidate_ik_settings": candidate_ik_settings,
                "debug_pregrasp_gate_active": bool(args.debug_pregrasp_pos_tol is not None or args.debug_pregrasp_rot_tol is not None),
                "solution_joint_positions": q_sol.tolist(),
                "reached_pose_base": reached_pose.tolist(),
                "target_point_A_world": _pose_position_world(coord_transform, target_pose).tolist(),
                "target_point_B_world": _point_b_world_from_pose(coord_transform, target_pose, candidate_point_b_offset).tolist(),
                "target_ee_origin_world": _pose_position_world(coord_transform, target_pose).tolist(),
                "accepted_for_pregrasp_selection": bool(valid),
                "accepted_despite_internal_not_ok": bool(valid and not ok),
            }
            selection_record = {
                **result,
                "geometry": preset_geometry,
                "selected_orientation_preset": preset,
            }
            fallback_selection_records.append(selection_record)
            candidate_results.append(result)
            flat_results.append(result)
            if valid and selected is None:
                selected = {
                    **selection_record,
                    "strict_candidate_success": True,
                    "selection_mode": "strict_valid_candidate",
                }
        preset_summary = _candidate_failure_summary(candidate_results)
        preset_results.append(
            {
                "preset_label": preset["preset_label"],
                "preset_index": int(preset["preset_index"]),
                "preset_family": preset.get("preset_family"),
                "preset_rpy": preset["rpy"],
                "preset_rotation_matrix": preset["rotation_matrix"],
                "approach_axis_base": preset["approach_axis_base"],
                "approach_axis_world": preset.get("approach_axis_world"),
                "AB_axis_world": preset_geometry.get("AB_axis_world"),
                "preset_axial_roll_variant_label": preset.get("preset_axial_roll_variant_label"),
                "preset_axial_roll_about_ab_rad": preset.get("preset_axial_roll_about_ab_rad"),
                "far_point_b_forward_extension_m": preset_geometry.get("far_point_b_forward_extension_m"),
                "far_point_a_extra_height_clearance_m": preset_geometry.get("far_point_a_extra_height_clearance_m"),
                "far_point_a_min_extra_height_clearance_m": preset_geometry.get("far_point_a_min_extra_height_clearance_m"),
                "far_ab_requested_downward_slant_deg": preset_geometry.get("far_ab_requested_downward_slant_deg"),
                "far_ab_downward_slant_deg": preset_geometry.get("far_ab_downward_slant_deg"),
                "far_ab_slant_height_clearance_m": preset_geometry.get("far_ab_slant_height_clearance_m"),
                "far_ab_base_horizontal_span_m": preset_geometry.get("far_ab_base_horizontal_span_m"),
                "far_point_b_gap_above_support_m": preset_geometry.get("far_point_b_gap_above_support_m"),
                "far_xy_align_clearance_above_object_m": preset_geometry.get("far_xy_align_clearance_above_object_m"),
                "vertical_contact_mark_B_world": preset_geometry.get("vertical_contact_mark_B_world"),
                "vertical_point_b_gap_above_support_m": preset_geometry.get("vertical_point_b_gap_above_support_m"),
                "vertical_close_point_b_tolerance_m": preset_geometry.get("vertical_close_point_b_tolerance_m"),
                "vertical_xy_reference_active": preset_geometry.get("vertical_xy_reference_active"),
                "vertical_xy_reference_mode": preset_geometry.get("vertical_xy_reference_mode"),
                "vertical_xy_reference_source": preset_geometry.get("vertical_xy_reference_source"),
                "vertical_xy_reference_world_position_used_for_offset": preset_geometry.get("vertical_xy_reference_world_position_used_for_offset"),
                "vertical_xy_reference_component_logs": preset_geometry.get("vertical_xy_reference_component_logs"),
                "vertical_xy_reference_target_xy_world": preset_geometry.get("vertical_xy_reference_target_xy_world"),
                "vertical_uncorrected_object_world_xy_target": preset_geometry.get("vertical_uncorrected_object_world_xy_target"),
                "vertical_arm_lateral_bias_correction_m": preset_geometry.get("vertical_arm_lateral_bias_correction_m"),
                "vertical_arm_lateral_bias_correction_base_y_m": preset_geometry.get("vertical_arm_lateral_bias_correction_base_y_m"),
                "vertical_arm_lateral_bias_correction_world": preset_geometry.get("vertical_arm_lateral_bias_correction_world"),
                "vertical_arm_lateral_bias_correction_rule": preset_geometry.get("vertical_arm_lateral_bias_correction_rule"),
                "vertical_xy_reference_offset_local": preset_geometry.get("vertical_xy_reference_offset_local"),
                "vertical_raw_point_B_contact_mark_before_xy_reference": preset_geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference"),
                "far_reach_axis_world": preset_geometry.get("far_reach_axis_world"),
                "dot_with_world_pos_y": preset.get("dot_with_world_pos_y"),
                "dot_with_world_neg_y": preset.get("dot_with_world_neg_y"),
                "up_axis_base": preset["up_axis_base"],
                "geometry": preset_geometry,
                "all_candidates_catastrophic_no_solution": bool(candidate_results and preset_summary["catastrophic_no_solution_count"] == len(candidate_results)),
                "has_reachable_candidates": bool(
                    preset_summary["solved_but_not_internal_ok_count"]
                    + preset_summary["candidate_error_exceeded_tolerance_count"]
                    + preset_summary["valid_candidate_count"]
                ),
                "candidate_failure_summary": preset_summary,
                "candidate_results": candidate_results,
            }
        )

    ik_solver.q = reference_q.copy()
    if selected is None and NO_GATE_MODE and fallback_selection_records:
        def _nogate_candidate_score(record: dict[str, Any]) -> tuple[float, float, float, int]:
            combined = _finite_float_or_none(record.get("combined_gate_error_norm"))
            pos_err = _finite_float_or_none(record.get("position_error_base_m"))
            rot_err = _finite_float_or_none(record.get("rotation_error_rad"))
            return (
                math.inf if combined is None else float(combined),
                math.inf if pos_err is None else float(pos_err),
                math.inf if rot_err is None else float(rot_err),
                int(record.get("candidate_index", 0)),
            )

        fallback = min(fallback_selection_records, key=_nogate_candidate_score)
        selected = {
            **fallback,
            "strict_candidate_success": False,
            "selection_mode": "nogate_best_available_candidate_forced_acceptance",
            "nogate_forced_acceptance": True,
            "nogate_original_candidate_classification": fallback.get("candidate_classification"),
        }
    success = selected is not None
    summary = _candidate_failure_summary(flat_results)
    if success:
        failure_reason = None
    elif not flat_results:
        failure_reason = "no_pregrasp_candidates_generated"
    elif summary["catastrophic_no_solution_count"] == len(flat_results):
        failure_reason = "all_candidates_catastrophic_no_solution"
    elif (
        summary["solved_but_not_internal_ok_count"]
        + summary["candidate_error_exceeded_tolerance_count"]
        + summary["valid_candidate_count"]
        > 0
    ):
        failure_reason = "reachable_candidates_failed_strict_acceptance"
    else:
        failure_reason = "mixed_candidate_reachability_failure"
    best_candidate_diagnostics = _best_candidate_diagnostics(flat_results, target_region)
    details = {
        "target_position": None if selected is None else selected["pose_base"],
        "start_ee_pose_base": reference_pose.tolist(),
        "start_ee_position_world": _pose_position_world(coord_transform, reference_pose).tolist(),
        "final_ee_pose_base": reference_pose.tolist(),
        "best_ee_pose_base": None if selected is None else selected["reached_pose_base"],
        "final_error": None if selected is None else selected["position_error_base_m"],
        "best_error": None if selected is None else selected["position_error_base_m"],
        "iteration_count": len(flat_results),
        "failure_reason": failure_reason,
        "candidate_policy": "region_filtered_approach_family_order_region_specific_strict_first_valid_candidate",
        "target_region": target_region,
        "forward_base": float(forward_base),
        "approach_family_order": list(approach_family_order),
        "candidate_validation_position_tolerance_m": candidate_pos_tol,
        "candidate_validation_rotation_tolerance_rad": candidate_rot_tol,
        "candidate_tolerance_policy": candidate_tolerance_policy,
        "far_candidate_position_tolerance_m": float(args.far_candidate_position_tolerance),
        "far_candidate_rotation_tolerance_rad": float(args.far_candidate_rotation_tolerance),
        "candidate_ik_settings": candidate_ik_settings,
        "point_b_offset_local": np.array(point_b_offset_local, dtype=float).tolist(),
        "selection_mode": None if selected is None else selected["selection_mode"],
        "no_gate_mode": bool(NO_GATE_MODE),
        "no_gate_candidate_forced_acceptance": bool(selected.get("nogate_forced_acceptance", False)) if selected is not None else False,
        "no_gate_original_candidate_classification": None if selected is None else selected.get("nogate_original_candidate_classification"),
        "debug_pregrasp_gate_active": bool(args.debug_pregrasp_pos_tol is not None or args.debug_pregrasp_rot_tol is not None),
        "orientation_preset_count": int(len(orientation_presets)),
        "orientation_preset_results": preset_results,
        "candidate_failure_summary": summary,
        "best_position_error_candidate": best_candidate_diagnostics["best_position_error_candidate"],
        "best_rotation_error_candidate": best_candidate_diagnostics["best_rotation_error_candidate"],
        "best_combined_error_candidate": best_candidate_diagnostics["best_combined_error_candidate"],
        "best_candidate_diagnostics": best_candidate_diagnostics,
        "candidate_results": flat_results,
        "selected_candidate": selected,
        "selected_orientation_preset": None if selected is None else selected["selected_orientation_preset"],
        "selected_approach_family": None if selected is None else selected["selected_orientation_preset"].get("preset_family"),
        "selected_grasp_family": None if selected is None else selected["selected_orientation_preset"].get("preset_family"),
        "selected_orientation_preset_label": None if selected is None else selected["selected_orientation_preset"]["preset_label"],
        "selected_orientation_preset_approach_axis_world": None if selected is None else selected["selected_orientation_preset"].get("approach_axis_world"),
        "selected_orientation_preset_AB_axis_world": None if selected is None else selected["selected_orientation_preset"].get("AB_axis_world"),
        "selected_orientation_preset_axial_roll_variant_label": None if selected is None else selected["selected_orientation_preset"].get("preset_axial_roll_variant_label"),
        "selected_orientation_preset_axial_roll_about_ab_rad": None if selected is None else selected["selected_orientation_preset"].get("preset_axial_roll_about_ab_rad"),
        "selected_orientation_preset_dot_with_world_pos_y": None if selected is None else selected["selected_orientation_preset"].get("dot_with_world_pos_y"),
        "selected_orientation_preset_dot_with_world_neg_y": None if selected is None else selected["selected_orientation_preset"].get("dot_with_world_neg_y"),
        "selected_orientation_preset_rpy": None if selected is None else selected["selected_orientation_preset"]["rpy"],
    }
    _append_phase(
        phase_log,
        phase="select_pregrasp_candidate",
        start_step=start_step,
        end_step=counter["step"],
        condition_met=success,
        details=details,
    )
    if not success:
        _fail("pregrasp_candidate_failed", f"No pregrasp candidate passed official DualArmIK validation: {failure_reason}")
    return selected


def _execute_dualarmik_servo_phase(
    spec: ServoSpec,
    *,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    arm_side: str,
    coord_transform: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    target_pose_fn: Callable[[], np.ndarray] | None = None,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    ik_overrides: dict[str, Any] | None = None,
    ik_refresh_enable_override: bool | None = None,
    ik_refresh_period_override: int | None = None,
    completion_condition_fn: Callable[[], bool] | None = None,
    early_stop_condition_fn: Callable[[], bool] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    point_b_offset_for_log = getattr(args, "point_b_offset_local_resolved", None)
    point_b_offset_for_log = None if point_b_offset_for_log is None else np.array(point_b_offset_for_log, dtype=float)
    point_b_metric_active = bool(point_b_offset_for_log is not None)
    coord_refresh_samples: list[dict[str, Any]] = []
    coord_refresh_failures = 0
    target_pose_eval_count = 0

    ik_refresh_active = bool(
        getattr(args, "ik_refresh_enable", DEFAULT_IK_REFRESH_ENABLE)
        if ik_refresh_enable_override is None
        else ik_refresh_enable_override
    )
    ik_refresh_period_source = (
        getattr(args, "ik_refresh_period", DEFAULT_IK_REFRESH_PERIOD)
        if ik_refresh_period_override is None
        else ik_refresh_period_override
    )
    ik_refresh_period = max(1, int(ik_refresh_period_source))
    ik_refresh_drift_threshold = float(getattr(args, "ik_refresh_drift_threshold", DEFAULT_IK_REFRESH_DRIFT_THRESHOLD))
    drift_refresh_active = bool(ik_refresh_active and ik_refresh_drift_threshold > 0.0 and target_pose_fn is not None)
    ik_refresh_events: list[dict[str, Any]] = []
    target_drift_checks = 0
    target_drift_samples: list[dict[str, Any]] = []
    completion_condition_error: str | None = None
    early_stop_condition_error: str | None = None
    early_stop_triggered = False
    early_stop_tick: int | None = None
    q_goal_update_count = 0
    q_goal: np.ndarray | None = None
    q_goal_target_pose: np.ndarray | None = None
    q_goal_ik_ok: bool | None = None
    last_ik_refresh_tick: int | None = None
    latest_ik_refresh_reason: str | None = None

    def evaluate_target_pose() -> np.ndarray:
        nonlocal target_pose_eval_count
        target_pose_eval_count += 1
        if target_pose_fn is not None:
            return np.array(target_pose_fn(), dtype=float)
        return np.array(spec.target_pose_base, dtype=float)

    def maybe_refresh_coord_transform(tick: int, reason: str, *, force_sample: bool = False) -> None:
        nonlocal coord_refresh_failures
        if coord_transform_refresh_fn is None:
            return
        try:
            refresh_log = coord_transform_refresh_fn()
        except Exception as exc:
            coord_refresh_failures += 1
            if force_sample or tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
                coord_refresh_samples.append({"tick": tick, "reason": reason, "error": repr(exc)})
            return
        if force_sample or tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
            coord_refresh_samples.append({"tick": tick, "reason": reason, **refresh_log})

    def position_metric_errors(current_pose_base: np.ndarray, target_pose_base: np.ndarray) -> tuple[float, float, float | None]:
        ee_pos_err, _ = _pose_error(ik_solver, current_pose_base, target_pose_base)
        if point_b_offset_for_log is None:
            return ee_pos_err, ee_pos_err, None
        current_b = _point_b_world_from_pose(coord_transform, current_pose_base, point_b_offset_for_log)
        target_b = _point_b_world_from_pose(coord_transform, target_pose_base, point_b_offset_for_log)
        point_b_err = float(np.linalg.norm(current_b - target_b))
        return point_b_err, ee_pos_err, point_b_err

    def target_pose_drift(reference_pose_base: np.ndarray, live_pose_base: np.ndarray) -> dict[str, Any]:
        _, rot_drift = _pose_error(ik_solver, reference_pose_base, live_pose_base)
        if point_b_offset_for_log is None:
            ref_point = np.array(reference_pose_base[:3], dtype=float)
            live_point = np.array(live_pose_base[:3], dtype=float)
            metric = "ee_origin_base_position"
        else:
            ref_point = _point_b_world_from_pose(coord_transform, reference_pose_base, point_b_offset_for_log)
            live_point = _point_b_world_from_pose(coord_transform, live_pose_base, point_b_offset_for_log)
            metric = "point_B_world_position"
        delta = live_point - ref_point
        return {
            "target_position_drift_m": float(np.linalg.norm(delta)),
            "target_rotation_drift_rad": float(rot_drift),
            "target_drift_metric": metric,
            "target_drift_vector": delta.tolist(),
        }

    def refresh_ik_goal(
        tick: int,
        reason: str,
        *,
        refresh_coord: bool,
        drift_metrics: dict[str, Any] | None = None,
    ) -> np.ndarray:
        nonlocal q_goal, q_goal_target_pose, q_goal_ik_ok, last_ik_refresh_tick, latest_ik_refresh_reason
        nonlocal ik_failures, q_goal_update_count
        if refresh_coord:
            maybe_refresh_coord_transform(tick, f"ik_refresh_{reason}_before_target", force_sample=True)
        target_pose = evaluate_target_pose()
        q_solution, ik_ok = _solve_single_arm_pose(ik_solver, arm_side, target_pose, args, ik_overrides=ik_overrides)
        if not ik_ok:
            ik_failures += 1
        q_goal = np.array(q_solution, dtype=float)
        q_goal_target_pose = np.array(target_pose, dtype=float)
        q_goal_ik_ok = bool(ik_ok)
        last_ik_refresh_tick = int(tick)
        latest_ik_refresh_reason = str(reason)
        q_goal_update_count += 1
        event = {
            "tick": int(tick),
            "reason": str(reason),
            "dualarmik_success": bool(ik_ok),
            "q_goal_update_index": int(q_goal_update_count),
            "target_pose_base": target_pose.tolist(),
            "target_pose_for_dualarmik_base": np.array(
                ik_solver.se3_to_xyzrpy(_ik_target_pose_se3(ik_solver, arm_side, target_pose, args)),
                dtype=float,
            ).tolist(),
            "q_goal_joint_targets": _named_positions(arm_dofs, q_goal),
        }
        if drift_metrics is not None:
            event.update(drift_metrics)
        ik_refresh_events.append(event)
        return target_pose

    maybe_refresh_coord_transform(0, "phase_start", force_sample=True)
    initial_target = evaluate_target_pose()
    start_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    start_pos_err, start_ee_pos_err, start_point_b_err = position_metric_errors(start_pose, initial_target)
    _, start_rot_err = _pose_error(ik_solver, start_pose, initial_target)
    best_pose = start_pose.copy()
    best_pos_error = start_pos_err
    best_ee_pos_error = start_ee_pos_err
    best_point_b_error = start_point_b_err
    best_rot_error = start_rot_err
    best_joint_positions = _current_positions(dc, arm_dofs)
    trace: list[dict[str, Any]] = []
    ik_failures = 0
    failure_reason: str | None = None
    target_workspace = _workspace_check(_pose_position_world(coord_transform, initial_target), args.workspace_x, args.workspace_y, args.workspace_z)
    workspace_violation = not bool(target_workspace["workspace_ok"])
    final_target = initial_target.copy()

    if ik_refresh_active:
        refresh_ik_goal(0, "initial", refresh_coord=False)

    for tick in range(1, int(spec.max_ticks) + 1):
        if ik_refresh_active:
            refresh_reason: str | None = None
            drift_metrics: dict[str, Any] | None = None
            if q_goal is None or q_goal_target_pose is None or last_ik_refresh_tick is None:
                refresh_reason = "initial_missing_goal"
            elif tick - int(last_ik_refresh_tick) >= ik_refresh_period:
                refresh_reason = "period"
            elif drift_refresh_active:
                live_target = evaluate_target_pose()
                target_drift_checks += 1
                drift_metrics = target_pose_drift(q_goal_target_pose, live_target)
                if tick == 1 or tick % int(args.trace_interval) == 0:
                    target_drift_samples.append({"tick": int(tick), **drift_metrics})
                if float(drift_metrics["target_position_drift_m"]) >= ik_refresh_drift_threshold:
                    refresh_reason = "drift_threshold"

            if refresh_reason is not None:
                target_pose = refresh_ik_goal(
                    tick,
                    refresh_reason,
                    refresh_coord=True,
                    drift_metrics=drift_metrics,
                )
            else:
                target_pose = np.array(q_goal_target_pose, dtype=float)
        else:
            maybe_refresh_coord_transform(tick, "servo_tick_before_target")
            target_pose = evaluate_target_pose()
            q_goal, q_goal_ik_ok = _solve_single_arm_pose(ik_solver, arm_side, target_pose, args, ik_overrides=ik_overrides)
            q_goal = np.array(q_goal, dtype=float)
            if not q_goal_ik_ok:
                ik_failures += 1
            q_goal_target_pose = np.array(target_pose, dtype=float)
            last_ik_refresh_tick = int(tick)
            latest_ik_refresh_reason = "legacy_per_tick_solve"
            q_goal_update_count += 1

        final_target = np.array(target_pose, dtype=float)
        current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        pos_err, ee_pos_err, point_b_err = position_metric_errors(current_pose, target_pose)
        _, rot_err = _pose_error(ik_solver, current_pose, target_pose)
        if pos_err < best_pos_error:
            best_pose = current_pose.copy()
            best_pos_error = pos_err
            best_ee_pos_error = ee_pos_err
            best_point_b_error = point_b_err
            best_rot_error = rot_err
            best_joint_positions = _current_positions(dc, arm_dofs)
        if early_stop_condition_fn is not None:
            try:
                if bool(early_stop_condition_fn()):
                    early_stop_triggered = True
                    early_stop_tick = int(tick)
                    break
            except Exception as exc:
                early_stop_condition_error = repr(exc)
        completion_condition_met = True
        if completion_condition_fn is not None:
            try:
                completion_condition_met = bool(completion_condition_fn())
            except Exception as exc:
                completion_condition_met = False
                completion_condition_error = repr(exc)
        if pos_err <= float(spec.pos_tolerance) and rot_err <= float(spec.rot_tolerance) and completion_condition_met:
            break

        q_current = _current_positions(dc, arm_dofs)
        if q_goal is None:
            q_goal, q_goal_ik_ok = _solve_single_arm_pose(ik_solver, arm_side, target_pose, args, ik_overrides=ik_overrides)
            q_goal = np.array(q_goal, dtype=float)
            if not q_goal_ik_ok:
                ik_failures += 1
        q_command = _limit_joint_delta(
            q_current,
            q_goal,
            blend=float(args.servo_blend),
            max_step_norm=float(args.servo_max_step_norm),
            max_abs_step=float(args.servo_max_abs_joint_step),
        )
        _send_position_targets(dc, arm_dofs, [float(value) for value in q_command])
        if spec.gripper_effort is not None:
            effort = _apply_gripper_effort(dc, gripper_dofs, float(spec.gripper_effort))
            if not effort["supported"]:
                raise RuntimeError(f"Gripper effort command failed during {spec.name}: {effort}")
        sim_app.update()
        counter["step"] += 1
        dc.wake_up_articulation(articulation)

        if tick == 1 or tick % int(args.trace_interval) == 0 or tick == int(spec.max_ticks):
            updated_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
            updated_pos_err, updated_ee_pos_err, updated_point_b_err = position_metric_errors(updated_pose, target_pose)
            _, updated_rot_err = _pose_error(ik_solver, updated_pose, target_pose)
            workspace_status = _workspace_check(_pose_position_world(coord_transform, updated_pose), args.workspace_x, args.workspace_y, args.workspace_z)
            workspace_violation = bool(workspace_violation or not workspace_status["workspace_ok"])
            trace_item = {
                "tick": tick,
                "target_pose_base": target_pose.tolist(),
                "target_pose_for_dualarmik_base": np.array(
                    ik_solver.se3_to_xyzrpy(_ik_target_pose_se3(ik_solver, arm_side, target_pose, args)),
                    dtype=float,
                ).tolist(),
                "ee_pose_base": updated_pose.tolist(),
                "target_ee_origin_world": _pose_position_world(coord_transform, target_pose).tolist(),
                "ee_origin_world": _pose_position_world(coord_transform, updated_pose).tolist(),
                "position_error_m": updated_pos_err,
                "position_error_metric": "point_B_world" if point_b_metric_active else "ee_pose_base",
                "ee_position_error_base_m": updated_ee_pos_err,
                "point_B_position_error_world_m": updated_point_b_err,
                "rotation_error_rad": updated_rot_err,
                "dualarmik_success": None if q_goal_ik_ok is None else bool(q_goal_ik_ok),
                "commanded_joint_targets": _named_positions(arm_dofs, q_command),
                "workspace_ok": workspace_status["workspace_ok"],
                "ik_refresh_mode_active": bool(ik_refresh_active),
                "latest_ik_refresh_tick": last_ik_refresh_tick,
                "latest_ik_refresh_reason": latest_ik_refresh_reason,
                "ik_goal_age_ticks": None if last_ik_refresh_tick is None else int(tick - int(last_ik_refresh_tick)),
                "ik_refreshed_this_tick": bool(ik_refresh_events and int(ik_refresh_events[-1]["tick"]) == int(tick)),
                "q_goal_update_count": int(q_goal_update_count),
            }
            if point_b_offset_for_log is not None:
                trace_item["target_AB_semantics"] = _ab_pose_semantics(coord_transform, target_pose, point_b_offset_for_log)
                trace_item["current_AB_semantics"] = _ab_pose_semantics(coord_transform, updated_pose, point_b_offset_for_log)
            trace.append(trace_item)

    maybe_refresh_coord_transform(int(spec.max_ticks), "phase_end_before_final_error", force_sample=True)
    final_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    final_pos_error, final_ee_pos_error, final_point_b_error = position_metric_errors(final_pose, final_target)
    _, final_rot_error = _pose_error(ik_solver, final_pose, final_target)
    if best_pos_error < final_pos_error and best_pos_error <= float(spec.pos_tolerance):
        _send_position_targets(dc, arm_dofs, [float(value) for value in best_joint_positions])
        _run_updates(sim_app, args.ik_settle_steps, counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=spec.gripper_effort)
        maybe_refresh_coord_transform(int(spec.max_ticks), "after_best_pose_restore", force_sample=True)
        final_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        final_pos_error, final_ee_pos_error, final_point_b_error = position_metric_errors(final_pose, final_target)
        _, final_rot_error = _pose_error(ik_solver, final_pose, final_target)

    completion_condition_met = True
    if completion_condition_fn is not None:
        try:
            completion_condition_met = bool(completion_condition_fn())
        except Exception as exc:
            completion_condition_met = False
            completion_condition_error = repr(exc)
    success = bool(
        final_pos_error <= float(spec.pos_tolerance)
        and final_rot_error <= float(spec.rot_tolerance)
        and completion_condition_met
    )
    if not success:
        failure_reason = "tolerance_not_met"
    if not completion_condition_met and not success:
        failure_reason = "completion_condition_not_met"
    if workspace_violation and not success:
        failure_reason = failure_reason or "workspace_violation"

    effective_ik_parameters = _ik_kwargs(args, ik_overrides)
    details: dict[str, Any] = {
        "target_pose_base": final_target.tolist(),
        "target_pose_for_dualarmik_base": np.array(
            ik_solver.se3_to_xyzrpy(_ik_target_pose_se3(ik_solver, arm_side, final_target, args)),
            dtype=float,
        ).tolist(),
        "target_ee_origin_world": _pose_position_world(coord_transform, final_target).tolist(),
        "start_ee_pose_base": start_pose.tolist(),
        "start_ee_origin_world": _pose_position_world(coord_transform, start_pose).tolist(),
        "final_ee_pose_base": final_pose.tolist(),
        "final_ee_origin_world": _pose_position_world(coord_transform, final_pose).tolist(),
        "best_ee_pose_base": best_pose.tolist(),
        "best_ee_origin_world": _pose_position_world(coord_transform, best_pose).tolist(),
        "final_error": final_pos_error,
        "best_error": best_pos_error,
        "position_error_metric": "point_B_world" if point_b_metric_active else "ee_pose_base",
        "point_B_metric_active": point_b_metric_active,
        "final_ee_position_error_base_m": final_ee_pos_error,
        "best_ee_position_error_base_m": best_ee_pos_error,
        "final_point_B_position_error_world_m": final_point_b_error,
        "best_point_B_position_error_world_m": best_point_b_error,
        "final_rotation_error_rad": final_rot_error,
        "best_rotation_error_rad": best_rot_error,
        "position_tolerance_m": float(spec.pos_tolerance),
        "rotation_tolerance_rad": float(spec.rot_tolerance),
        "iteration_count": counter["step"] - start_step,
        "failure_reason": None if success else failure_reason,
        "workspace_violation": bool(workspace_violation),
        "target_workspace_check": target_workspace,
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
        "ee_frame_compensation_active": bool(getattr(args, "ee_frame_compensation_active", False)),
        "final_joint_positions": _named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
        "dualarmik_parameters": {
            "max_iter": int(effective_ik_parameters["max_iter"]),
            "pos_tol": float(effective_ik_parameters["pos_tol"]),
            "rot_tol": float(effective_ik_parameters["rot_tol"]),
            "damping": float(effective_ik_parameters["damping"]),
            "dq_max": float(effective_ik_parameters["dq_max"]),
            "pos_weight": float(effective_ik_parameters.get("pos_weight", 1.0)),
            "rot_weight": float(effective_ik_parameters["rot_weight"]),
            "null_weight": float(effective_ik_parameters["null_weight"]),
            "base_null_weight": float(args.ik_null_weight),
            "servo_blend": float(args.servo_blend),
            "servo_max_step_norm": float(args.servo_max_step_norm),
            "servo_max_abs_joint_step": float(args.servo_max_abs_joint_step),
            "ik_failures": int(ik_failures),
            "ik_overrides": ik_overrides or {},
        },
        "periodic_ik_refresh_active": bool(ik_refresh_active),
        "ik_refresh_period_ticks": int(ik_refresh_period),
        "ik_refresh_drift_threshold_m": float(ik_refresh_drift_threshold),
        "ik_refresh_drift_check_active": bool(drift_refresh_active),
        "ik_refresh_count": int(len(ik_refresh_events)),
        "ik_refresh_ticks": [int(event["tick"]) for event in ik_refresh_events],
        "ik_refresh_reasons": [str(event["reason"]) for event in ik_refresh_events],
        "ik_refresh_events": ik_refresh_events,
        "target_pose_evaluation_count": int(target_pose_eval_count),
        "target_drift_check_count": int(target_drift_checks),
        "target_drift_samples": target_drift_samples,
        "q_goal_update_count": int(q_goal_update_count),
        "ik_execution_cadence": "periodic_q_goal_refresh_with_smooth_joint_tracking" if ik_refresh_active else "legacy_solve_ik_every_servo_tick",
        "ik_refresh_enable_override": ik_refresh_enable_override,
        "ik_refresh_period_override": ik_refresh_period_override,
        "completion_condition_required": bool(completion_condition_fn is not None),
        "completion_condition_met": bool(completion_condition_met),
        "completion_condition_error": completion_condition_error,
        "early_stop_condition_required": bool(early_stop_condition_fn is not None),
        "early_stop_triggered": bool(early_stop_triggered),
        "early_stop_tick": early_stop_tick,
        "early_stop_condition_error": early_stop_condition_error,
        "live_coordinate_transform_refresh_active": bool(coord_transform_refresh_fn is not None),
        "live_coordinate_transform_refresh_failures": int(coord_refresh_failures),
        "live_coordinate_transform_refresh_samples": coord_refresh_samples,
        "trace": trace,
    }
    if point_b_offset_for_log is not None:
        details["point_b_offset_local"] = point_b_offset_for_log.tolist()
        details["target_AB_semantics"] = _ab_pose_semantics(coord_transform, final_target, point_b_offset_for_log)
        details["start_AB_semantics"] = _ab_pose_semantics(coord_transform, start_pose, point_b_offset_for_log)
        details["final_AB_semantics"] = _ab_pose_semantics(coord_transform, final_pose, point_b_offset_for_log)
        details["best_AB_semantics"] = _ab_pose_semantics(coord_transform, best_pose, point_b_offset_for_log)
    if spec.gripper_effort is not None:
        details["gripper_effort_active"] = True
        details["gripper_effort_value"] = float(spec.gripper_effort)
    if extra_details:
        details.update(extra_details)
    _append_phase(
        phase_log,
        phase=spec.name,
        start_step=start_step,
        end_step=counter["step"],
        condition_met=success,
        details=details,
    )
    print(f"phase={spec.name} condition_met={success} final_pos_error={final_pos_error:.4f} final_rot_error={final_rot_error:.4f}")
    return details

def _command_gripper_phase(
    phase_name: str,
    *,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    target_positions: list[float],
    sim_app: Any,
    steps: int,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    skipped: bool = False,
    effort_value: float | None = None,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> bool:
    start_step = counter["step"]
    coord_transform_refresh_log = None
    if coord_transform_refresh_fn is not None:
        try:
            coord_transform_refresh_log = coord_transform_refresh_fn()
        except Exception as exc:
            coord_transform_refresh_log = {"error": repr(exc)}
    if skipped:
        effort_result = _run_updates(sim_app, steps, counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=effort_value)
        observed = _read_positions(dc, gripper_dofs)
        command_supported = True
    else:
        if len(target_positions) != len(gripper_dofs):
            raise ValueError(f"Expected {len(gripper_dofs)} gripper targets, got {len(target_positions)}")
        _send_position_targets(dc, gripper_dofs, [float(value) for value in target_positions])
        effort_result = _run_updates(sim_app, steps, counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=effort_value)
        observed = _read_positions(dc, gripper_dofs)
        command_supported = bool(target_positions)

    ee_position = _body_pose_position(dc, end_effector_body)
    details: dict[str, Any] = {
        "target_position": None,
        "final_ee_position": ee_position.tolist(),
        "final_ee_orientation_quat": _body_pose_orientation(dc, end_effector_body),
        "iteration_count": int(steps),
        "failure_reason": None if command_supported else "gripper_command_empty",
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
        "skipped": bool(skipped),
        "commanded_gripper_targets": None if skipped else _named_positions(gripper_dofs, target_positions),
        "observed_gripper_joint_values": _named_positions(gripper_dofs, observed),
        "official_gripper_open_width": OFFICIAL_GRIPPER_OPEN_WIDTH,
        "official_gripper_close_width": OFFICIAL_GRIPPER_CLOSE_WIDTH,
        "target_semantics": "official same-sign finger joint positions",
        "gripper_effort": effort_result,
        "coordinate_transform_refresh_before_gripper": coord_transform_refresh_log,
    }
    if extra_details:
        details.update(extra_details)
    _append_phase(
        phase_log,
        phase=phase_name,
        start_step=start_step,
        end_step=counter["step"],
        condition_met=command_supported,
        details=details,
    )
    print(f"phase={phase_name} condition_met={command_supported} skipped={skipped}")
    return command_supported


def _resolve_torso_prim_path(stage: Any, chosen_robot_prim_path: str, articulation_path: str) -> tuple[str, list[dict[str, Any]]]:
    candidates = [
        f"{chosen_robot_prim_path.rstrip('/')}/torso_link",
        f"{chosen_robot_prim_path.rstrip('/')}/Ref/torso_link",
        f"{articulation_path.rstrip('/')}/torso_link",
        "/Root/Ref_Xform/Ref/torso_link",
    ]
    attempts: list[dict[str, Any]] = []
    for path in candidates:
        valid = _valid_prim_path(stage, path)
        attempts.append({"path": path, "valid": valid})
        if valid:
            return path, attempts
    raise RuntimeError(f"torso_link prim not found: {json.dumps(attempts, sort_keys=True)}")


def _work_area_world_from_cfg(cfg: dict[str, Any], table_bbox: dict[str, list[float]]) -> np.ndarray:
    scatter = cfg.get("grasp", {}).get("scatter_area", {})
    center = scatter.get("center") if isinstance(scatter, dict) else None
    if center is not None and len(center) >= 3:
        return np.array([float(center[0]), float(center[1]), float(center[2])], dtype=float)
    return np.array(table_bbox["center"], dtype=float)


def _pre_close_gate(
    *,
    stage: Any,
    target_path: str,
    geometry: dict[str, Any],
    coord_transform: Any,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    tcp_offset_local: np.ndarray,
    end_effector_name: str,
    end_effector_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = _bbox_state(stage, target_path)
    current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    rot = _euler_xyz_to_rot(float(current_pose[3]), float(current_pose[4]), float(current_pose[5]))
    tcp_base = current_pose[:3] + rot @ np.array(tcp_offset_local, dtype=float)
    tcp_world = np.array(coord_transform.robot_to_world(tcp_base), dtype=float)
    point_b_offset = np.array(geometry.get("point_b_offset_local", tcp_offset_local), dtype=float)
    point_b_world = _point_b_world_from_pose(coord_transform, current_pose, point_b_offset)
    target_point_b_world = np.array(geometry.get("contact_point_B_world", point_b_world), dtype=float)
    point_b_error = float(np.linalg.norm(point_b_world - target_point_b_world))
    real_grasp_center_world, real_grasp_center_log = _resolve_real_grasp_center_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
    )
    real_grasp_center_error = None
    proxy_to_real_delta_world = None
    proxy_to_real_delta_local = None
    proxy_to_real_delta_norm = None
    close_critical_world = point_b_world
    close_critical_error = point_b_error
    close_critical_metric = "point_B_proxy_world_fallback"
    if real_grasp_center_world is not None:
        real_grasp_center_world = np.array(real_grasp_center_world, dtype=float)
        proxy_to_real_delta_world = real_grasp_center_world - point_b_world
        proxy_to_real_delta_norm = float(np.linalg.norm(proxy_to_real_delta_world))
        proxy_to_real_delta_local = _pose_rotation_world(coord_transform, current_pose).T @ proxy_to_real_delta_world
        real_grasp_center_error = float(np.linalg.norm(real_grasp_center_world - target_point_b_world))
        close_critical_world = real_grasp_center_world
        close_critical_error = real_grasp_center_error
        close_critical_metric = "real_grasp_center_world"
    vertical_xy_reference_offset = geometry.get("vertical_xy_reference_offset_local")
    vertical_xy_reference_world = None
    vertical_xy_reference_error = None
    vertical_xy_reference_runtime_log = None
    if vertical_xy_reference_offset is not None and geometry.get("vertical_xy_reference_target_xy_world") is not None:
        vertical_xy_reference_world, vertical_xy_reference_runtime_log = _resolve_current_vertical_xy_reference_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            reference_log=geometry.get("vertical_xy_reference_link_log"),
            coord_transform=coord_transform,
            current_pose_base=current_pose,
            reference_offset_local=np.array(vertical_xy_reference_offset, dtype=float),
        )
        target_xy = np.array(geometry["vertical_xy_reference_target_xy_world"], dtype=float)
        vertical_xy_reference_error = float(np.linalg.norm(vertical_xy_reference_world[:2] - target_xy[:2]))
    return {
        "object_center_world": _center_from_bbox(state["bbox"]).tolist(),
        "object_bbox": state,
        "chosen_contact_point_world": geometry["contact_point_world"],
        "chosen_ee_pose_base": geometry["contact_pose_base"],
        "chosen_ee_origin_world": geometry["contact_ee_origin_world"],
        "actual_ee_pose_base_before_close": current_pose.tolist(),
        "actual_tcp_world_before_close": tcp_world.tolist(),
        "point_B_proxy_world": point_b_world.tolist(),
        "actual_point_B_world_before_close": point_b_world.tolist(),
        "target_point_A_world_before_close": geometry.get("contact_AB_semantics", {}).get("point_A_world"),
        "target_point_B_world_before_close": target_point_b_world.tolist(),
        "point_B_error_before_close_m": point_b_error,
        "real_grasp_center_world": None if real_grasp_center_world is None else real_grasp_center_world.tolist(),
        "real_grasp_center_log": real_grasp_center_log,
        "real_grasp_center_source": real_grasp_center_log.get("source"),
        "real_grasp_center_component_positions_world": real_grasp_center_log.get("component_positions_world"),
        "real_grasp_center_error_before_close_m": real_grasp_center_error,
        "proxy_to_real_grasp_center_delta_world": None if proxy_to_real_delta_world is None else proxy_to_real_delta_world.tolist(),
        "proxy_to_real_grasp_center_delta_world_norm_m": proxy_to_real_delta_norm,
        "proxy_to_real_grasp_center_delta_local": None if proxy_to_real_delta_local is None else proxy_to_real_delta_local.tolist(),
        "close_critical_metric": close_critical_metric,
        "close_critical_eval_world": close_critical_world.tolist(),
        "close_critical_target_world": target_point_b_world.tolist(),
        "close_critical_error_before_close_m": close_critical_error,
        "close_critical_uses_real_grasp_center": bool(real_grasp_center_world is not None),
        "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
        "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
        "far_descend_B_world": geometry.get("far_descend_B_world"),
        "far_contact_sequence_policy": geometry.get("far_contact_sequence_policy"),
        "far_point_b_forward_extension_m": geometry.get("far_point_b_forward_extension_m"),
        "far_point_a_extra_height_clearance_m": geometry.get("far_point_a_extra_height_clearance_m"),
        "far_point_a_min_extra_height_clearance_m": geometry.get("far_point_a_min_extra_height_clearance_m"),
        "far_ab_requested_downward_slant_deg": geometry.get("far_ab_requested_downward_slant_deg"),
        "far_ab_downward_slant_deg": geometry.get("far_ab_downward_slant_deg"),
        "far_ab_slant_height_clearance_m": geometry.get("far_ab_slant_height_clearance_m"),
        "far_ab_base_horizontal_span_m": geometry.get("far_ab_base_horizontal_span_m"),
        "far_point_b_gap_above_support_m": geometry.get("far_point_b_gap_above_support_m"),
        "far_xy_align_clearance_above_object_m": geometry.get("far_xy_align_clearance_above_object_m"),
        "vertical_contact_mark_B_world": geometry.get("vertical_contact_mark_B_world"),
        "vertical_raw_point_B_contact_mark_before_xy_reference": geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference"),
        "vertical_point_b_gap_above_support_m": geometry.get("vertical_point_b_gap_above_support_m"),
        "vertical_close_point_b_tolerance_m": geometry.get("vertical_close_point_b_tolerance_m"),
        "vertical_xy_reference_active": geometry.get("vertical_xy_reference_active"),
        "vertical_xy_reference_mode": geometry.get("vertical_xy_reference_mode"),
        "vertical_xy_reference_source": geometry.get("vertical_xy_reference_source"),
        "vertical_xy_reference_world_position_used_for_offset": geometry.get("vertical_xy_reference_world_position_used_for_offset"),
        "vertical_xy_reference_component_logs": geometry.get("vertical_xy_reference_component_logs"),
        "vertical_xy_reference_target_xy_world": geometry.get("vertical_xy_reference_target_xy_world"),
        "vertical_xy_reference_offset_local": geometry.get("vertical_xy_reference_offset_local"),
        "actual_vertical_xy_reference_world_before_close": None if vertical_xy_reference_world is None else vertical_xy_reference_world.tolist(),
        "actual_vertical_xy_reference_runtime_source_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("runtime_source"),
        "actual_vertical_xy_reference_runtime_fallback_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("runtime_fallback_used"),
        "actual_finger_midpoint_component_positions_world_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("component_positions_world"),
        "vertical_xy_reference_error_before_close_m": vertical_xy_reference_error,
        "vertical_xy_reference_tolerance_m": geometry.get("vertical_xy_reference_tolerance_m"),
        "vertical_contact_sequence_policy": geometry.get("vertical_contact_sequence_policy"),
        "vertical_actual_point_B_gap_above_support_m": None
        if geometry.get("object_support_z_world") is None
        else float(point_b_world[2] - float(geometry.get("object_support_z_world"))),
        "far_reach_axis_world": geometry.get("far_reach_axis_world"),
        "tcp_offset_local_used": np.array(tcp_offset_local, dtype=float).tolist(),
        "point_b_offset_local_used": point_b_offset.tolist(),
        "tcp_offset_base_for_target": geometry["contact_details"].get("point_b_offset_base"),
        "end_effector_name": end_effector_name,
        "end_effector_path": end_effector_path,
        "ee_frame_compensation_active": bool(getattr(args, "ee_frame_compensation_active", False)),
    }


def _execute_nogate_close_dls_phase(
    *,
    stage: Any,
    target_path: str,
    geometry: dict[str, Any],
    coord_transform: Any,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    arm_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    target_metric_world: np.ndarray,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    switch_log: dict[str, Any] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    enabled = bool(NO_GATE_MODE and getattr(args, "nogate_close_dls_enable", True))
    if not enabled or skip_reason is not None:
        details = {
            "enabled": bool(enabled),
            "skipped": True,
            "skip_reason": skip_reason or "disabled_or_not_no_gate_mode",
            "no_gate_mode": bool(NO_GATE_MODE),
            "source_algorithm": "task1_single_target_random_scene_baseline measured finite-difference DLS",
            "switch_log": switch_log,
        }
        _append_phase(
            phase_log,
            phase="nogate_near_contact_measured_dls_finish",
            start_step=start_step,
            end_step=counter["step"],
            condition_met=True,
            details=details,
        )
        return details

    point_b_offset = np.array(geometry.get("point_b_offset_local", geometry.get("tcp_offset_local", [0.0, 0.0, 0.0])), dtype=float)
    if point_b_offset.shape != (3,) or not np.isfinite(point_b_offset).all():
        point_b_offset = np.array(getattr(args, "point_b_offset_local_resolved", [0.0, 0.0, float(args.tcp_fallback_x)]), dtype=float)

    max_iters = int(args.nogate_close_dls_max_iters)
    settle_steps = int(args.nogate_close_dls_settle_steps)
    hold_steps = int(args.nogate_close_dls_hold_steps)
    eps = float(args.nogate_close_dls_eps)
    damping = float(args.nogate_close_dls_damping)
    max_step = float(args.nogate_close_dls_max_step)
    max_abs_step = float(args.nogate_close_dls_max_abs_joint_step)
    blend = float(args.nogate_close_dls_blend)
    stop_tolerance = float(args.nogate_close_dls_stop_tolerance)
    posture_gain = float(args.nogate_close_dls_posture_gain)
    target_world = np.array(target_metric_world, dtype=float)
    trace: list[dict[str, Any]] = []
    jacobian_trace: list[dict[str, Any]] = []
    coord_refresh_samples: list[dict[str, Any]] = []
    coord_refresh_failures = 0
    solve_failures = 0

    def maybe_refresh_coord(tick: int, reason: str) -> None:
        nonlocal coord_refresh_failures
        if coord_transform_refresh_fn is None:
            return
        try:
            refresh_log = coord_transform_refresh_fn()
        except Exception as exc:
            coord_refresh_failures += 1
            if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
                coord_refresh_samples.append({"tick": int(tick), "reason": reason, "error": repr(exc)})
            return
        if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
            coord_refresh_samples.append({"tick": int(tick), "reason": reason, **refresh_log})

    def apply_arm_targets(positions: np.ndarray, steps: int) -> None:
        _send_position_targets(dc, arm_dofs, [float(value) for value in positions])
        _run_updates(sim_app, max(0, int(steps)), counter)
        dc.wake_up_articulation(articulation)

    def read_close_metric() -> tuple[np.ndarray, np.ndarray, str, dict[str, Any], np.ndarray]:
        current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        point_b_world = _point_b_world_from_pose(coord_transform, current_pose, point_b_offset)
        real_center_world, real_log = _resolve_real_grasp_center_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
        )
        if real_center_world is not None:
            real_center = np.array(real_center_world, dtype=float)
            return current_pose, real_center, "real_grasp_center_world", real_log, real_center - point_b_world
        return current_pose, point_b_world.copy(), "point_B_proxy_world_fallback", real_log, np.zeros(3, dtype=float)

    def estimate_metric_jacobian(base_positions: np.ndarray, base_metric_world: np.ndarray, iter_index: int) -> np.ndarray:
        jacobian = np.zeros((3, len(arm_dofs)), dtype=float)
        for column in range(len(arm_dofs)):
            trial = np.array(base_positions, dtype=float).copy()
            trial[column] += eps
            apply_arm_targets(trial, settle_steps)
            _, moved_metric, _, _, _ = read_close_metric()
            jacobian[:, column] = (moved_metric - base_metric_world) / eps
        apply_arm_targets(base_positions, settle_steps)
        if iter_index == 1 or iter_index % int(args.trace_interval) == 0:
            column_norms = [float(np.linalg.norm(jacobian[:, idx])) for idx in range(jacobian.shape[1])]
            jacobian_trace.append(
                {
                    "iter": int(iter_index),
                    "eps_m_or_rad": eps,
                    "settle_steps_per_trial": settle_steps,
                    "column_norms": column_norms,
                    "jacobian_frobenius_norm": float(np.linalg.norm(jacobian)),
                }
            )
        return jacobian

    maybe_refresh_coord(0, "nogate_close_dls_start")
    positions = _current_positions(dc, arm_dofs)
    posture = positions.copy()
    _, start_metric, start_metric_name, start_metric_log, start_proxy_delta = read_close_metric()
    best_metric = start_metric.copy()
    best_error = float(np.linalg.norm(target_world - start_metric))
    best_joint_positions = positions.copy()
    final_metric = start_metric.copy()
    final_metric_name = start_metric_name
    final_metric_log = start_metric_log
    final_proxy_delta = start_proxy_delta
    failure_reason = None

    for iter_index in range(1, max_iters + 1):
        maybe_refresh_coord(iter_index, "nogate_close_dls_iter")
        positions = _current_positions(dc, arm_dofs)
        _, current_metric, current_metric_name, current_metric_log, current_proxy_delta = read_close_metric()
        error_vector = target_world - current_metric
        position_error = float(np.linalg.norm(error_vector))
        if position_error < best_error:
            best_error = position_error
            best_metric = current_metric.copy()
            best_joint_positions = positions.copy()
        if position_error <= stop_tolerance:
            final_metric = current_metric.copy()
            final_metric_name = current_metric_name
            final_metric_log = current_metric_log
            final_proxy_delta = current_proxy_delta
            break

        jacobian = estimate_metric_jacobian(positions, current_metric, iter_index)
        lhs = jacobian @ jacobian.T + (damping**2) * np.eye(3)
        try:
            delta = jacobian.T @ np.linalg.solve(lhs, error_vector)
        except Exception:
            solve_failures += 1
            delta = jacobian.T @ np.linalg.pinv(lhs) @ error_vector
        delta += posture_gain * (posture - positions)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > max_step:
            delta *= max_step / max(delta_norm, 1.0e-9)
        raw_goal = positions + delta
        q_current = _current_positions(dc, arm_dofs)
        q_command = _limit_joint_delta(
            q_current,
            raw_goal,
            blend=blend,
            max_step_norm=max_step,
            max_abs_step=max_abs_step,
        )
        apply_arm_targets(q_command, settle_steps)
        _, updated_metric, updated_metric_name, updated_metric_log, updated_proxy_delta = read_close_metric()
        updated_error = float(np.linalg.norm(target_world - updated_metric))
        if updated_error < best_error:
            best_error = updated_error
            best_metric = updated_metric.copy()
            best_joint_positions = _current_positions(dc, arm_dofs).copy()
        final_metric = updated_metric.copy()
        final_metric_name = updated_metric_name
        final_metric_log = updated_metric_log
        final_proxy_delta = updated_proxy_delta
        if iter_index == 1 or iter_index % int(args.trace_interval) == 0 or updated_error <= stop_tolerance or iter_index == max_iters:
            trace.append(
                {
                    "iter": int(iter_index),
                    "target_close_metric_world": target_world.tolist(),
                    "close_metric_world_before": current_metric.tolist(),
                    "close_metric_world_after": updated_metric.tolist(),
                    "metric_name_before": current_metric_name,
                    "metric_name_after": updated_metric_name,
                    "position_error_before_m": position_error,
                    "position_error_after_m": updated_error,
                    "raw_delta_norm": delta_norm,
                    "limited_delta_norm": float(np.linalg.norm(q_command - q_current)),
                    "commanded_joint_targets": _named_positions(arm_dofs, q_command),
                    "proxy_to_real_delta_world_after": updated_proxy_delta.tolist(),
                }
            )
        if updated_error <= stop_tolerance:
            break

    if best_error < float(np.linalg.norm(target_world - final_metric)):
        apply_arm_targets(best_joint_positions, hold_steps)
        _, final_metric, final_metric_name, final_metric_log, final_proxy_delta = read_close_metric()
    elif hold_steps > 0:
        apply_arm_targets(_current_positions(dc, arm_dofs), hold_steps)
        _, final_metric, final_metric_name, final_metric_log, final_proxy_delta = read_close_metric()

    final_error = float(np.linalg.norm(target_world - final_metric))
    success = bool(final_error <= stop_tolerance)
    if not success:
        failure_reason = "dls_stop_tolerance_not_met"
    details = {
        "enabled": True,
        "skipped": False,
        "no_gate_mode": bool(NO_GATE_MODE),
        "source_algorithm": "task1_single_target_random_scene_baseline measured finite-difference DLS position controller",
        "reduced_stop_go_policy": "near-contact only, settle_steps=1 by default, hold_steps=0 by default, small blended joint steps",
        "switch_log": switch_log,
        "target_path": target_path,
        "target_close_metric_world": target_world.tolist(),
        "start_close_metric_world": start_metric.tolist(),
        "start_close_metric_name": start_metric_name,
        "start_real_grasp_center_log": start_metric_log,
        "start_proxy_to_real_delta_world": start_proxy_delta.tolist(),
        "final_close_metric_world": final_metric.tolist(),
        "final_close_metric_name": final_metric_name,
        "final_real_grasp_center_log": final_metric_log,
        "final_proxy_to_real_delta_world": final_proxy_delta.tolist(),
        "final_error_m": final_error,
        "best_error_m": best_error,
        "best_close_metric_world": best_metric.tolist(),
        "condition_met": success,
        "failure_reason": failure_reason,
        "measured_finite_difference_dls_active": True,
        "max_iters": max_iters,
        "eps": eps,
        "damping": damping,
        "max_step_norm": max_step,
        "max_abs_joint_step": max_abs_step,
        "blend": blend,
        "settle_steps_per_trial_and_command": settle_steps,
        "hold_steps": hold_steps,
        "stop_tolerance_m": stop_tolerance,
        "posture_gain": posture_gain,
        "solve_failures": int(solve_failures),
        "live_coordinate_transform_refresh_failures": int(coord_refresh_failures),
        "live_coordinate_transform_refresh_samples": coord_refresh_samples,
        "jacobian_trace": jacobian_trace,
        "trace": trace,
        "observed_joint_values": _named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
    }
    _append_phase(
        phase_log,
        phase="nogate_near_contact_measured_dls_finish",
        start_step=start_step,
        end_step=counter["step"],
        condition_met=success,
        details=details,
    )
    print(
        "phase=nogate_near_contact_measured_dls_finish "
        f"condition_met={success} final_error={final_error:.4f} "
        f"best_error={best_error:.4f} metric={final_metric_name}"
    )
    return details


def _execute_nogate_preclose_touch_probe(
    *,
    stage: Any,
    target_path: str,
    geometry: dict[str, Any],
    coord_transform: Any,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    arm_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    table_top_z: float,
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    ik_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    enabled = bool(NO_GATE_MODE and getattr(args, "nogate_preclose_touch_enable", True))
    if not enabled:
        details = {
            "enabled": False,
            "reason": "disabled_or_not_no_gate_mode",
            "no_gate_mode": bool(NO_GATE_MODE),
            "safe_to_close": True,
        }
        _append_phase(
            phase_log,
            phase="nogate_preclose_mesh_touch_probe",
            start_step=start_step,
            end_step=counter["step"],
            condition_met=True,
            details=details,
        )
        return details

    point_b_offset = np.array(geometry.get("point_b_offset_local", geometry.get("tcp_offset_local", [0.0, 0.0, 0.0])), dtype=float)
    if point_b_offset.shape != (3,) or not np.isfinite(point_b_offset).all():
        point_b_offset = np.array(getattr(args, "point_b_offset_local_resolved", [0.0, 0.0, float(args.tcp_fallback_x)]), dtype=float)
    locked_rpy = np.array(geometry.get("fixed_downward_rpy_base", geometry.get("contact_pose_base", [0, 0, 0, 0, 0, 0])[3:]), dtype=float)
    contact_point_b_world = np.array(geometry.get("contact_point_B_world"), dtype=float)
    target_xy = np.array(contact_point_b_world[:2], dtype=float)
    touch_gap = float(args.nogate_touch_gap_above_table)
    z_step = float(args.nogate_touch_step_z)
    xy_tolerance = float(args.nogate_touch_xy_tolerance)
    object_expand = float(args.nogate_touch_object_expand)
    object_motion_threshold = float(args.nogate_touch_object_motion_threshold)
    stall_tolerance = float(args.nogate_touch_stall_tolerance)
    capture_tolerance = float(args.nogate_finger_capture_tolerance)
    capture_segment_margin = float(args.nogate_finger_capture_segment_margin)
    table_touch_z = float(table_top_z) + touch_gap
    require_table_touch_before_close = bool(getattr(args, "nogate_require_table_touch_before_close", True))
    trace: list[dict[str, Any]] = []
    coord_refresh_samples: list[dict[str, Any]] = []
    ik_failures = 0
    coord_refresh_failures = 0
    fix_attempts = 0
    post_touch_reposition_attempts = 0
    latest_ik_ok = False
    latest_ik_reason = "not_attempted"
    confirmed_touch_detected = False
    table_touch_confirmed = False
    unconfirmed_stop_detected = False
    touch_detected = False
    touch_reason: str | None = None
    touch_tick: int | None = None
    stop_reason: str | None = None
    stop_tick: int | None = None
    final_fix_performed = False
    post_touch_reposition_performed = False

    initial_object_state = _bbox_state(stage, target_path)
    initial_object_center = _center_from_bbox(initial_object_state["bbox"])

    def maybe_refresh_coord(tick: int, reason: str) -> None:
        nonlocal coord_refresh_failures
        if coord_transform_refresh_fn is None:
            return
        try:
            refresh_log = coord_transform_refresh_fn()
        except Exception as exc:
            coord_refresh_failures += 1
            if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
                coord_refresh_samples.append({"tick": int(tick), "reason": reason, "error": repr(exc)})
            return
        if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
            coord_refresh_samples.append({"tick": int(tick), "reason": reason, **refresh_log})

    def read_close_metric() -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any], np.ndarray]:
        current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        point_b_world = _point_b_world_from_pose(coord_transform, current_pose, point_b_offset)
        real_center_world, real_log = _resolve_real_grasp_center_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
        )
        if real_center_world is not None:
            real_center = np.array(real_center_world, dtype=float)
            return current_pose, point_b_world, real_center, "real_grasp_center_world", real_log, real_center - point_b_world
        return current_pose, point_b_world, point_b_world.copy(), "point_B_proxy_world_fallback", real_log, np.zeros(3, dtype=float)

    def bbox_touch_estimate(center_world: np.ndarray, bbox: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        bbox_min = np.array(bbox["min"], dtype=float)
        bbox_max = np.array(bbox["max"], dtype=float)
        xy_inside = bool(
            bbox_min[0] - object_expand <= center_world[0] <= bbox_max[0] + object_expand
            and bbox_min[1] - object_expand <= center_world[1] <= bbox_max[1] + object_expand
        )
        z_overlap = bool(bbox_min[2] - object_expand <= center_world[2] <= bbox_max[2] + object_expand)
        z_near_top = bool(center_world[2] <= bbox_max[2] + object_expand)
        touched = bool(xy_inside and (z_overlap or z_near_top))
        return touched, {
            "target_object_bbox_min_world": bbox_min.tolist(),
            "target_object_bbox_max_world": bbox_max.tolist(),
            "target_object_xy_inside_expanded_bbox": xy_inside,
            "target_object_z_overlap_or_near_top": bool(z_overlap or z_near_top),
            "target_object_bbox_expand_m": object_expand,
        }

    def object_between_fingers_status() -> dict[str, Any]:
        object_state = _bbox_state(stage, target_path)
        bbox_min = np.array(object_state["bbox"]["min"], dtype=float)
        bbox_max = np.array(object_state["bbox"]["max"], dtype=float)
        object_center = _center_from_bbox(object_state["bbox"])
        half_extent = 0.5 * np.maximum(bbox_max - bbox_min, 0.0)
        object_radius = float(np.linalg.norm(half_extent))
        _, grasp_log = _resolve_real_grasp_center_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
        )
        component_positions = grasp_log.get("component_positions_world")
        if not component_positions or len(component_positions) != 2:
            return {
                "resolved": False,
                "object_between_fingers": False,
                "reason": "finger1_finger2_positions_unresolved",
                "target_path": target_path,
                "target_object_center_world": object_center.tolist(),
                "target_object_bbox": object_state,
                "real_grasp_center_log": grasp_log,
                "finger_capture_tolerance_m": capture_tolerance,
                "finger_capture_segment_margin": capture_segment_margin,
            }
        finger1 = np.array(component_positions[0], dtype=float)
        finger2 = np.array(component_positions[1], dtype=float)
        segment = finger2 - finger1
        segment_len = float(np.linalg.norm(segment))
        if segment_len <= 1.0e-9:
            return {
                "resolved": False,
                "object_between_fingers": False,
                "reason": "finger_segment_degenerate",
                "target_path": target_path,
                "target_object_center_world": object_center.tolist(),
                "target_object_bbox": object_state,
                "finger1_world_position": finger1.tolist(),
                "finger2_world_position": finger2.tolist(),
                "finger_segment_length_m": segment_len,
                "real_grasp_center_log": grasp_log,
                "finger_capture_tolerance_m": capture_tolerance,
                "finger_capture_segment_margin": capture_segment_margin,
            }
        t_raw = float(np.dot(object_center - finger1, segment) / max(segment_len * segment_len, 1.0e-12))
        t_clamped = float(np.clip(t_raw, 0.0, 1.0))
        closest = finger1 + t_clamped * segment
        perpendicular_distance = float(np.linalg.norm(object_center - closest))
        distance_threshold = float(capture_tolerance + object_radius)
        between_along_segment = bool(-capture_segment_margin <= t_raw <= 1.0 + capture_segment_margin)
        close_to_segment = bool(perpendicular_distance <= distance_threshold)
        return {
            "resolved": True,
            "object_between_fingers": bool(between_along_segment and close_to_segment),
            "reason": "object_center_projected_between_finger1_finger2_segment" if between_along_segment and close_to_segment else "object_center_not_between_finger1_finger2_segment",
            "target_path": target_path,
            "target_object_center_world": object_center.tolist(),
            "target_object_bbox": object_state,
            "target_object_half_extent_world_m": half_extent.tolist(),
            "target_object_radius_estimate_m": object_radius,
            "finger1_world_position": finger1.tolist(),
            "finger2_world_position": finger2.tolist(),
            "finger_segment_length_m": segment_len,
            "object_projection_fraction_between_fingers": t_raw,
            "object_projection_fraction_clamped": t_clamped,
            "closest_point_on_finger_segment_world": closest.tolist(),
            "object_distance_to_finger_segment_m": perpendicular_distance,
            "finger_capture_distance_threshold_m": distance_threshold,
            "finger_capture_tolerance_m": capture_tolerance,
            "finger_capture_segment_margin": capture_segment_margin,
            "between_along_finger_segment": between_along_segment,
            "close_to_finger_segment": close_to_segment,
            "real_grasp_center_log": grasp_log,
        }

    def command_desired_center(tick: int, desired_center_world: np.ndarray, command_role: str) -> dict[str, Any]:
        nonlocal ik_failures, latest_ik_ok, latest_ik_reason
        current_pose, point_b_world, center_world, metric_name, metric_log, center_minus_point_b = read_close_metric()
        desired_point_b_world = np.array(desired_center_world, dtype=float) - center_minus_point_b
        target_pose, target_details = _pose_for_point_b_world(desired_point_b_world, coord_transform, locked_rpy, point_b_offset)
        q_goal, ik_ok = _solve_single_arm_pose(ik_solver, arm_side, target_pose, args, ik_overrides=ik_overrides)
        latest_ik_ok = bool(ik_ok)
        latest_ik_reason = "dualarmik_success" if ik_ok else "dualarmik_failed"
        if not ik_ok:
            ik_failures += 1
        q_current = _current_positions(dc, arm_dofs)
        q_command = _limit_joint_delta(
            q_current,
            q_goal,
            blend=float(args.servo_blend),
            max_step_norm=float(args.servo_max_step_norm),
            max_abs_step=float(args.servo_max_abs_joint_step),
        )
        _send_position_targets(dc, arm_dofs, [float(value) for value in q_command])
        sim_app.update()
        counter["step"] += 1
        dc.wake_up_articulation(articulation)
        after_pose, after_point_b, after_center, after_metric_name, after_metric_log, after_delta = read_close_metric()
        return {
            "tick": int(tick),
            "command_role": command_role,
            "desired_close_metric_world": np.array(desired_center_world, dtype=float).tolist(),
            "desired_point_B_world": desired_point_b_world.tolist(),
            "target_pose_base": target_pose.tolist(),
            "target_details": target_details,
            "dualarmik_success": bool(ik_ok),
            "metric_name_before": metric_name,
            "close_metric_world_before": center_world.tolist(),
            "point_B_proxy_world_before": point_b_world.tolist(),
            "proxy_to_real_delta_world_before": center_minus_point_b.tolist(),
            "metric_name_after": after_metric_name,
            "close_metric_world_after": after_center.tolist(),
            "point_B_proxy_world_after": after_point_b.tolist(),
            "proxy_to_real_delta_world_after": after_delta.tolist(),
            "real_grasp_center_log_after": after_metric_log,
            "commanded_joint_targets": _named_positions(arm_dofs, q_command),
            "close_metric_xy_error_to_target_m": float(np.linalg.norm(after_center[:2] - target_xy)),
            "close_metric_z_gap_above_table_m": float(after_center[2] - float(table_top_z)),
        }

    maybe_refresh_coord(0, "nogate_touch_probe_start")
    _, _, initial_center_metric, initial_metric_name, initial_metric_log, initial_delta = read_close_metric()
    previous_center_z = float(initial_center_metric[2])
    last_command_log: dict[str, Any] | None = None

    for tick in range(1, int(args.nogate_touch_max_ticks) + 1):
        maybe_refresh_coord(tick, "nogate_touch_probe_tick")
        _, _, current_center, _, _, _ = read_close_metric()
        desired_center = current_center.copy()
        desired_center[:2] = target_xy
        desired_center[2] = max(table_touch_z, float(current_center[2]) - z_step)
        command_log = command_desired_center(tick, desired_center, "force_mesh_down_to_table_or_selected_object")
        last_command_log = command_log
        after_center = np.array(command_log["close_metric_world_after"], dtype=float)
        object_state = _bbox_state(stage, target_path)
        object_center = _center_from_bbox(object_state["bbox"])
        object_delta = object_center - initial_object_center
        target_object_touch, object_touch_log = bbox_touch_estimate(after_center, object_state["bbox"])
        table_touch = bool(after_center[2] <= table_touch_z + stall_tolerance)
        object_motion_contact = bool(float(np.linalg.norm(object_delta)) >= object_motion_threshold)
        downward_progress = previous_center_z - float(after_center[2])
        possible_contact_stall = bool(
            tick >= 4
            and desired_center[2] < previous_center_z - 0.5 * z_step
            and downward_progress <= stall_tolerance
        )
        sample = {
            **command_log,
            "target_xy_world": target_xy.tolist(),
            "table_touch_z_world": float(table_touch_z),
            "target_object_touch_estimate": target_object_touch,
            "table_touch_estimate": table_touch,
            "target_object_motion_contact_estimate": object_motion_contact,
            "possible_contact_or_ik_stall": possible_contact_stall,
            "possible_contact_or_ik_stall_is_confirmed_touch": False,
            "object_center_delta_during_touch_probe_m": object_delta.tolist(),
            "object_center_delta_norm_during_touch_probe_m": float(np.linalg.norm(object_delta)),
            **object_touch_log,
        }
        if tick == 1 or tick % int(args.trace_interval) == 0 or target_object_touch or table_touch or object_motion_contact or possible_contact_stall:
            trace.append(sample)
        previous_center_z = float(after_center[2])
        if target_object_touch:
            confirmed_touch_detected = True
            touch_detected = True
            touch_reason = "selected_target_object_bbox_overlap"
            stop_reason = touch_reason
            touch_tick = tick
            stop_tick = tick
            break
        if table_touch:
            confirmed_touch_detected = True
            table_touch_confirmed = True
            touch_detected = True
            touch_reason = "table_height_reached"
            stop_reason = touch_reason
            touch_tick = tick
            stop_tick = tick
            break
        if object_motion_contact:
            confirmed_touch_detected = True
            touch_detected = True
            touch_reason = "selected_target_object_motion_detected"
            stop_reason = touch_reason
            touch_tick = tick
            stop_tick = tick
            break
        if possible_contact_stall:
            unconfirmed_stop_detected = True
            if stop_reason is None:
                stop_reason = "possible_contact_or_ik_stall_unconfirmed_continued_lowering"
                stop_tick = tick

    if stop_reason is None:
        stop_reason = "max_touch_ticks_reached_without_confirmed_touch"
        stop_tick = int(args.nogate_touch_max_ticks)

    final_capture_before_reposition = object_between_fingers_status()
    post_touch_reposition_performed = bool(int(args.nogate_post_touch_reposition_ticks) > 0)
    for reposition_index in range(1, int(args.nogate_post_touch_reposition_ticks) + 1):
        capture_status = object_between_fingers_status()
        if bool(capture_status.get("object_between_fingers")) and latest_ik_ok and confirmed_touch_detected and post_touch_reposition_attempts > 0 and (table_touch_confirmed or not require_table_touch_before_close):
            break
        maybe_refresh_coord(reposition_index, "nogate_post_touch_reposition_before_close")
        _, _, current_center, _, _, _ = read_close_metric()
        object_center = np.array(capture_status.get("target_object_center_world", contact_point_b_world), dtype=float)
        desired_center = current_center.copy()
        desired_center[:2] = object_center[:2]
        if confirmed_touch_detected and (table_touch_confirmed or not require_table_touch_before_close):
            desired_center[2] = max(table_touch_z, float(current_center[2]))
        else:
            desired_center[2] = max(table_touch_z, float(current_center[2]) - z_step)
        command_log = command_desired_center(
            reposition_index,
            desired_center,
            "post_touch_recalculate_ik_move_object_between_fingers_before_close",
        )
        post_touch_reposition_attempts += 1
        last_command_log = command_log
        after_center_for_touch = np.array(command_log["close_metric_world_after"], dtype=float)
        if after_center_for_touch[2] <= table_touch_z + stall_tolerance:
            table_touch_confirmed = True
            confirmed_touch_detected = True
            if touch_reason is None:
                touch_reason = "table_height_reached_after_post_touch_reposition"
        capture_after = object_between_fingers_status()
        command_log["table_touch_confirmed_after_command"] = bool(table_touch_confirmed)
        command_log["object_between_fingers_after_command"] = capture_after
        if (
            reposition_index == 1
            or reposition_index % int(args.trace_interval) == 0
            or bool(capture_after.get("object_between_fingers"))
            or not bool(command_log["dualarmik_success"])
        ):
            trace.append(command_log)
        if bool(command_log["dualarmik_success"]) and bool(capture_after.get("object_between_fingers")) and confirmed_touch_detected and (table_touch_confirmed or not require_table_touch_before_close):
            break

    _, _, final_center_before_fix, _, _, _ = read_close_metric()
    final_xy_error_before_fix = float(np.linalg.norm(final_center_before_fix[:2] - target_xy))
    final_capture_before_fix = object_between_fingers_status()
    if (not latest_ik_ok or final_xy_error_before_fix > xy_tolerance) and not bool(final_capture_before_fix.get("object_between_fingers")):
        final_fix_performed = True
        for fix_index in range(1, int(args.nogate_touch_fix_ticks) + 1):
            maybe_refresh_coord(fix_index, "nogate_touch_probe_xy_fix")
            _, _, fix_center, _, _, _ = read_close_metric()
            object_center = np.array(final_capture_before_fix.get("target_object_center_world", contact_point_b_world), dtype=float)
            desired_fix_center = fix_center.copy()
            desired_fix_center[:2] = object_center[:2]
            desired_fix_center[2] = max(table_touch_z, float(fix_center[2])) if confirmed_touch_detected and (table_touch_confirmed or not require_table_touch_before_close) else max(table_touch_z, float(fix_center[2]) - z_step)
            command_log = command_desired_center(fix_index, desired_fix_center, "final_ik_position_fix_before_close_check")
            fix_attempts += 1
            after_center_for_touch = np.array(command_log["close_metric_world_after"], dtype=float)
            if after_center_for_touch[2] <= table_touch_z + stall_tolerance:
                table_touch_confirmed = True
                confirmed_touch_detected = True
                if touch_reason is None:
                    touch_reason = "table_height_reached_after_final_ik_fix"
            capture_after = object_between_fingers_status()
            command_log["table_touch_confirmed_after_command"] = bool(table_touch_confirmed)
            command_log["object_between_fingers_after_command"] = capture_after
            if fix_index == 1 or fix_index % int(args.trace_interval) == 0 or bool(capture_after.get("object_between_fingers")):
                trace.append(command_log)
            final_capture_before_fix = capture_after
            if bool(command_log["dualarmik_success"]) and bool(capture_after.get("object_between_fingers")):
                break

    _, _, final_center, final_metric_name, final_metric_log, final_delta = read_close_metric()
    final_xy_error = float(np.linalg.norm(final_center[:2] - target_xy))
    final_z_gap_above_table = float(final_center[2] - float(table_top_z))
    final_finger_capture_status = object_between_fingers_status()
    table_touch_confirmed = bool(table_touch_confirmed or final_z_gap_above_table <= touch_gap + max(stall_tolerance, 1.0e-6))
    table_or_target_touch_confirmed = bool(confirmed_touch_detected)
    touch_requirement_met = bool(table_touch_confirmed if require_table_touch_before_close else table_or_target_touch_confirmed)
    object_between_fingers = bool(final_finger_capture_status.get("object_between_fingers"))
    safe_to_close = bool(touch_requirement_met and latest_ik_ok and object_between_fingers)
    ik_right_before_close = safe_to_close
    close_block_reason = None
    if not touch_requirement_met:
        close_block_reason = "no_confirmed_table_touch" if require_table_touch_before_close else "no_confirmed_table_or_selected_object_touch"
    elif not latest_ik_ok:
        close_block_reason = "latest_dualarmik_reposition_failed"
    elif not object_between_fingers:
        close_block_reason = "selected_object_not_between_finger1_finger2"

    details = {
        "enabled": True,
        "no_gate_mode": bool(NO_GATE_MODE),
        "policy": "force_mesh_to_table_or_selected_object_then_recompute_ik_and_close_only_if_object_between_fingers",
        "mesh_metric_priority": "real_grasp_center_world_then_point_B_proxy_world",
        "target_path": target_path,
        "target_point_B_world": contact_point_b_world.tolist(),
        "target_xy_world": target_xy.tolist(),
        "table_top_z_world": float(table_top_z),
        "table_touch_z_world": float(table_touch_z),
        "confirmed_touch_detected": bool(confirmed_touch_detected),
        "table_touch_confirmed": bool(table_touch_confirmed),
        "require_table_touch_before_close": bool(require_table_touch_before_close),
        "touch_requirement_met": bool(touch_requirement_met),
        "touch_detected": bool(touch_detected),
        "touch_reason": touch_reason,
        "touch_tick": touch_tick,
        "unconfirmed_stop_detected": bool(unconfirmed_stop_detected),
        "stop_reason": stop_reason,
        "stop_tick": stop_tick,
        "safe_to_close": safe_to_close,
        "close_block_reason": close_block_reason,
        "ik_right_before_close": ik_right_before_close,
        "latest_ik_success": bool(latest_ik_ok),
        "latest_ik_reason": latest_ik_reason,
        "ik_failure_count": int(ik_failures),
        "post_touch_reposition_performed": bool(post_touch_reposition_performed),
        "post_touch_reposition_attempts": int(post_touch_reposition_attempts),
        "fix_position_attempts": int(fix_attempts),
        "final_fix_performed": bool(final_fix_performed),
        "object_between_fingers_before_reposition": final_capture_before_reposition,
        "object_between_fingers_before_close": final_finger_capture_status,
        "initial_close_metric_world": initial_center_metric.tolist(),
        "initial_close_metric_name": initial_metric_name,
        "initial_real_grasp_center_log": initial_metric_log,
        "initial_proxy_to_real_delta_world": initial_delta.tolist(),
        "final_close_metric_world": final_center.tolist(),
        "final_close_metric_name": final_metric_name,
        "final_real_grasp_center_log": final_metric_log,
        "final_proxy_to_real_delta_world": final_delta.tolist(),
        "final_xy_error_to_target_m": final_xy_error,
        "final_z_gap_above_table_m": final_z_gap_above_table,
        "xy_tolerance_m": xy_tolerance,
        "z_step_m": z_step,
        "max_ticks": int(args.nogate_touch_max_ticks),
        "fix_ticks": int(args.nogate_touch_fix_ticks),
        "post_touch_reposition_ticks": int(args.nogate_post_touch_reposition_ticks),
        "object_bbox_expand_m": object_expand,
        "object_motion_threshold_m": object_motion_threshold,
        "stall_tolerance_m": stall_tolerance,
        "finger_capture_tolerance_m": capture_tolerance,
        "finger_capture_segment_margin": capture_segment_margin,
        "last_command_log": last_command_log,
        "live_coordinate_transform_refresh_failures": int(coord_refresh_failures),
        "live_coordinate_transform_refresh_samples": coord_refresh_samples,
        "trace": trace,
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
    }
    _append_phase(
        phase_log,
        phase="nogate_preclose_mesh_touch_probe",
        start_step=start_step,
        end_step=counter["step"],
        condition_met=bool(safe_to_close),
        details=details,
    )
    print(
        "phase=nogate_preclose_mesh_touch_probe "
        f"confirmed_touch={confirmed_touch_detected} stop_reason={stop_reason} "
        f"safe_to_close={safe_to_close} close_block_reason={close_block_reason} "
        f"final_xy_error={final_xy_error:.4f} final_z_gap={final_z_gap_above_table:.4f}"
    )
    return details


def _execute_nogate_post_close_slow_lift_hold(
    *,
    stage: Any,
    target_path: str,
    geometry: dict[str, Any],
    coord_transform: Any,
    ik_solver: Any,
    dc: Any,
    articulation: Any,
    arm_side: str,
    arm_dofs: list[tuple[int, Any, str]],
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    ik_overrides: dict[str, Any] | None = None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    enabled = bool(NO_GATE_MODE and getattr(args, "nogate_post_close_slow_lift_enable", True))
    if not enabled or skipped:
        details = {
            "enabled": bool(enabled),
            "skipped": True,
            "skip_reason": skip_reason or ("disabled" if not enabled else "close_gripper_was_skipped"),
            "no_gate_mode": bool(NO_GATE_MODE),
        }
        _append_phase(
            phase_log,
            phase="nogate_post_close_slow_lift_hold_grip",
            start_step=start_step,
            end_step=counter["step"],
            condition_met=True,
            details=details,
        )
        return details

    height = float(args.nogate_post_close_slow_lift_height)
    max_ticks = int(args.nogate_post_close_slow_lift_ticks)
    blend = float(args.nogate_post_close_slow_lift_blend)
    max_step_norm = float(args.nogate_post_close_slow_lift_max_step_norm)
    max_abs_step = float(args.nogate_post_close_slow_lift_max_abs_joint_step)
    gripper_effort = float(args.gripper_hold_effort)
    start_object_state = _bbox_state(stage, target_path)
    start_object_center = _center_from_bbox(start_object_state["bbox"])
    coord_refresh_samples: list[dict[str, Any]] = []
    coord_refresh_failures = 0

    def maybe_refresh_coord(tick: int, reason: str) -> None:
        nonlocal coord_refresh_failures
        if coord_transform_refresh_fn is None:
            return
        try:
            refresh_log = coord_transform_refresh_fn()
        except Exception as exc:
            coord_refresh_failures += 1
            if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
                coord_refresh_samples.append({"tick": int(tick), "reason": reason, "error": repr(exc)})
            return
        if tick == 0 or tick == 1 or tick % int(args.trace_interval) == 0:
            coord_refresh_samples.append({"tick": int(tick), "reason": reason, **refresh_log})

    maybe_refresh_coord(0, "nogate_post_close_slow_lift_start")
    start_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    start_world = _pose_position_world(coord_transform, start_pose)
    target_world = start_world + np.array([0.0, 0.0, height], dtype=float)
    target_pose = np.array(start_pose, dtype=float)
    target_pose[:3] = np.array(coord_transform.world_to_robot(target_world), dtype=float)
    q_goal, ik_ok = _solve_single_arm_pose(ik_solver, arm_side, target_pose, args, ik_overrides=ik_overrides)
    q_goal = np.array(q_goal, dtype=float)
    point_b_offset = np.array(geometry.get("point_b_offset_local", geometry.get("tcp_offset_local", [0.0, 0.0, 0.0])), dtype=float)
    point_b_metric_active = bool(point_b_offset.shape == (3,) and np.isfinite(point_b_offset).all())
    start_point_b_world = None
    target_point_b_world = None
    if point_b_metric_active:
        start_point_b_world = _point_b_world_from_pose(coord_transform, start_pose, point_b_offset)
        target_point_b_world = start_point_b_world + np.array([0.0, 0.0, height], dtype=float)
    trace: list[dict[str, Any]] = []
    last_effort_result: dict[str, Any] | None = None
    final_pose = start_pose.copy()
    final_error = math.inf
    final_rot_error = math.inf
    success = False
    failure_reason = None if ik_ok else "dualarmik_failed_initial_slow_lift_target"

    if ik_ok:
        for tick in range(1, max_ticks + 1):
            maybe_refresh_coord(tick, "nogate_post_close_slow_lift_tick")
            _send_position_targets(dc, gripper_dofs, [OFFICIAL_GRIPPER_CLOSE_WIDTH] * len(gripper_dofs))
            last_effort_result = _apply_gripper_effort(dc, gripper_dofs, gripper_effort)
            q_current = _current_positions(dc, arm_dofs)
            q_command = _limit_joint_delta(
                q_current,
                q_goal,
                blend=blend,
                max_step_norm=max_step_norm,
                max_abs_step=max_abs_step,
            )
            _send_position_targets(dc, arm_dofs, [float(value) for value in q_command])
            sim_app.update()
            counter["step"] += 1
            dc.wake_up_articulation(articulation)
            final_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
            final_error, final_rot_error = _pose_error(ik_solver, final_pose, target_pose)
            if tick == 1 or tick % int(args.trace_interval) == 0 or tick == max_ticks:
                sample: dict[str, Any] = {
                    "tick": int(tick),
                    "target_pose_base": target_pose.tolist(),
                    "current_pose_base": final_pose.tolist(),
                    "target_ee_origin_world": target_world.tolist(),
                    "current_ee_origin_world": _pose_position_world(coord_transform, final_pose).tolist(),
                    "position_error_m": float(final_error),
                    "rotation_error_rad": float(final_rot_error),
                    "commanded_joint_targets": _named_positions(arm_dofs, q_command),
                    "gripper_close_targets_reissued": True,
                    "gripper_effort_result": last_effort_result,
                }
                if point_b_metric_active:
                    sample["target_point_B_world"] = target_point_b_world.tolist()
                    sample["current_point_B_world"] = _point_b_world_from_pose(coord_transform, final_pose, point_b_offset).tolist()
                trace.append(sample)
            if final_error <= float(args.lift_tolerance) and final_rot_error <= float(args.rot_tolerance):
                success = True
                break
        if not success and failure_reason is None:
            failure_reason = "slow_lift_target_not_reached_within_ticks"

    final_object_state = _bbox_state(stage, target_path)
    final_object_center = _center_from_bbox(final_object_state["bbox"])
    object_delta = final_object_center - start_object_center
    details = {
        "enabled": True,
        "skipped": False,
        "policy": "after_close_slow_world_z_lift_with_closed_targets_and_continuous_gripper_effort",
        "height_m": height,
        "max_ticks": max_ticks,
        "blend": blend,
        "max_step_norm": max_step_norm,
        "max_abs_joint_step": max_abs_step,
        "gripper_hold_effort": gripper_effort,
        "gripper_close_targets_reissued_each_tick": bool(ik_ok),
        "gripper_effort_applied_each_tick": bool(ik_ok),
        "dualarmik_success_initial": bool(ik_ok),
        "target_pose_base": target_pose.tolist(),
        "start_pose_base": start_pose.tolist(),
        "final_pose_base": final_pose.tolist(),
        "target_ee_origin_world": target_world.tolist(),
        "start_ee_origin_world": start_world.tolist(),
        "final_ee_origin_world": _pose_position_world(coord_transform, final_pose).tolist(),
        "final_error_m": None if not math.isfinite(final_error) else float(final_error),
        "final_rotation_error_rad": None if not math.isfinite(final_rot_error) else float(final_rot_error),
        "position_tolerance_m": float(args.lift_tolerance),
        "rotation_tolerance_rad": float(args.rot_tolerance),
        "failure_reason": failure_reason,
        "start_object_pose": start_object_state,
        "final_object_pose": final_object_state,
        "object_delta_during_slow_lift_m": object_delta.tolist(),
        "object_delta_z_during_slow_lift_m": float(object_delta[2]),
        "last_gripper_effort_result": last_effort_result,
        "live_coordinate_transform_refresh_failures": int(coord_refresh_failures),
        "live_coordinate_transform_refresh_samples": coord_refresh_samples,
        "trace": trace,
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
    }
    if point_b_metric_active:
        details["point_b_offset_local"] = point_b_offset.tolist()
        details["start_point_B_world"] = start_point_b_world.tolist()
        details["target_point_B_world"] = target_point_b_world.tolist()
        details["final_point_B_world"] = _point_b_world_from_pose(coord_transform, final_pose, point_b_offset).tolist()
    _append_phase(
        phase_log,
        phase="nogate_post_close_slow_lift_hold_grip",
        start_step=start_step,
        end_step=counter["step"],
        condition_met=bool(success),
        details=details,
    )
    print(
        "phase=nogate_post_close_slow_lift_hold_grip "
        f"condition_met={success} final_error={details['final_error_m']} "
        f"object_delta_z={float(object_delta[2]):.4f}"
    )
    return details

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-mode",
        choices=("tcp_compensated", "contact_axis"),
        default="tcp_compensated",
    )
    parser.add_argument(
        "--approach-axis-mode",
        choices=("neg_z", "pos_z", "pos_x", "neg_x"),
        default="neg_z",
    )
    parser.add_argument("--far-threshold", "--horizontal-far-threshold", dest="far_threshold", type=float, default=DEFAULT_FAR_THRESHOLD)
    parser.add_argument("--near-body-threshold", type=float, default=DEFAULT_NEAR_BODY_THRESHOLD)
    parser.add_argument("--baseline-root")
    parser.add_argument("--contact-world-y-bias", type=float, default=0.0)
    parser.add_argument("--contact-base-forward-bias", type=float, default=0.0)
    parser.add_argument("--contact-base-lateral-bias", type=float, default=0.0)
    parser.add_argument("--asset-root")
    parser.add_argument("--prim-path")
    parser.add_argument("--end-effector-body")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-selection-policy", choices=("nearest", "index"), default="nearest")
    parser.add_argument("--arm", choices=("auto", "right", "left"), default="auto")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--gripper-steps", type=int, default=DEFAULT_GRIPPER_STEPS)
    parser.add_argument("--servo-max-ticks", type=int, default=DEFAULT_SERVO_MAX_TICKS)
    parser.add_argument("--servo-carry-ticks", type=int, default=DEFAULT_SERVO_CARRY_TICKS)
    parser.add_argument("--servo-blend", type=float, default=DEFAULT_SERVO_BLEND)
    parser.add_argument("--servo-max-step-norm", type=float, default=DEFAULT_SERVO_MAX_STEP_NORM)
    parser.add_argument("--servo-max-abs-joint-step", type=float, default=DEFAULT_SERVO_MAX_ABS_JOINT_STEP)
    parser.add_argument("--ik-max-iter", type=int, default=DEFAULT_IK_MAX_ITER)
    parser.add_argument("--ik-pos-tol", type=float, default=DEFAULT_IK_POS_TOL)
    parser.add_argument("--ik-rot-tol", type=float, default=DEFAULT_IK_ROT_TOL)
    parser.add_argument("--ik-damping", type=float, default=DEFAULT_IK_DAMPING)
    parser.add_argument("--ik-dq-max", type=float, default=DEFAULT_IK_DQ_MAX)
    parser.add_argument("--ik-rot-weight", type=float, default=None)
    parser.add_argument("--ik-null-weight", type=float, default=DEFAULT_IK_NULL_WEIGHT)
    parser.add_argument("--ik-refresh-enable", action=argparse.BooleanOptionalAction, default=DEFAULT_IK_REFRESH_ENABLE)
    parser.add_argument("--ik-refresh-period", type=int, default=DEFAULT_IK_REFRESH_PERIOD)
    parser.add_argument("--ik-refresh-drift-threshold", type=float, default=DEFAULT_IK_REFRESH_DRIFT_THRESHOLD)
    parser.add_argument("--candidate-ik-max-iter", type=int, default=DEFAULT_CANDIDATE_IK_MAX_ITER)
    parser.add_argument("--candidate-ik-pos-tol", type=float, default=DEFAULT_CANDIDATE_IK_POS_TOL)
    parser.add_argument("--candidate-ik-rot-tol", type=float, default=DEFAULT_CANDIDATE_IK_ROT_TOL)
    parser.add_argument("--far-candidate-position-tolerance", type=float, default=DEFAULT_FAR_CANDIDATE_POSITION_TOLERANCE)
    parser.add_argument("--far-candidate-rotation-tolerance", type=float, default=DEFAULT_FAR_CANDIDATE_ROTATION_TOLERANCE)
    parser.add_argument("--ik-settle-steps", type=int, default=4)
    parser.add_argument("--pregrasp-tolerance", type=float, default=DEFAULT_PREGRASP_TOLERANCE)
    parser.add_argument("--align-tolerance", type=float, default=DEFAULT_ALIGN_TOLERANCE)
    parser.add_argument("--descend-tolerance", type=float, default=DEFAULT_DESCEND_TOLERANCE)
    parser.add_argument("--lift-tolerance", type=float, default=DEFAULT_LIFT_TOLERANCE)
    parser.add_argument("--carry-tolerance", type=float, default=DEFAULT_CARRY_TOLERANCE)
    parser.add_argument("--place-tolerance", type=float, default=DEFAULT_PLACE_TOLERANCE)
    parser.add_argument("--retreat-tolerance", type=float, default=DEFAULT_RETREAT_TOLERANCE)
    parser.add_argument("--rot-tolerance", type=float, default=DEFAULT_ROT_TOLERANCE)
    parser.add_argument("--pregrasp-clearance", type=float, default=DEFAULT_PREGRASP_CLEARANCE)
    parser.add_argument("--pregrasp-standoff", type=float, default=DEFAULT_PREGRASP_STANDOFF)
    parser.add_argument("--far-low-side-clearance", type=float, default=DEFAULT_FAR_LOW_SIDE_CLEARANCE)
    parser.add_argument(
        "--far-point-b-gap-above-support",
        "--far-low-side-gap-above-support",
        dest="far_point_b_gap_above_support",
        type=float,
        default=DEFAULT_FAR_POINT_B_GAP_ABOVE_SUPPORT,
    )
    parser.add_argument("--far-xy-align-clearance-above-object", type=float, default=DEFAULT_FAR_XY_ALIGN_CLEARANCE_ABOVE_OBJECT)
    parser.add_argument("--far-point-b-forward-extension", type=float, default=DEFAULT_FAR_POINT_B_FORWARD_EXTENSION)
    parser.add_argument("--far-point-a-extra-height-clearance", type=float, default=DEFAULT_FAR_POINT_A_EXTRA_HEIGHT_CLEARANCE)
    parser.add_argument("--far-ab-downward-slant-deg", type=float, default=DEFAULT_FAR_AB_DOWNWARD_SLANT_DEG)
    parser.add_argument("--far-outboard-transition-offset", type=float, default=DEFAULT_FAR_OUTBOARD_TRANSITION_OFFSET)
    parser.add_argument("--far-outboard-transition-clearance", type=float, default=DEFAULT_FAR_OUTBOARD_TRANSITION_CLEARANCE)
    parser.add_argument("--far-null-weight", type=float, default=DEFAULT_FAR_NULL_WEIGHT)
    parser.add_argument("--far-outboard-shoulder-roll-bias", type=float, default=DEFAULT_FAR_OUTBOARD_SHOULDER_ROLL_BIAS)
    parser.add_argument("--pre-close-point-b-tolerance", type=float, default=DEFAULT_PRE_CLOSE_POINT_B_TOLERANCE)
    parser.add_argument("--nogate-preclose-touch-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nogate-touch-max-ticks", type=int, default=DEFAULT_NOGATE_TOUCH_MAX_TICKS)
    parser.add_argument("--nogate-touch-fix-ticks", type=int, default=DEFAULT_NOGATE_TOUCH_FIX_TICKS)
    parser.add_argument("--nogate-touch-step-z", type=float, default=DEFAULT_NOGATE_TOUCH_STEP_Z)
    parser.add_argument("--nogate-touch-gap-above-table", type=float, default=DEFAULT_NOGATE_TOUCH_GAP_ABOVE_TABLE)
    parser.add_argument("--nogate-touch-xy-tolerance", type=float, default=DEFAULT_NOGATE_TOUCH_XY_TOLERANCE)
    parser.add_argument("--nogate-touch-object-expand", type=float, default=DEFAULT_NOGATE_TOUCH_OBJECT_EXPAND)
    parser.add_argument("--nogate-touch-object-motion-threshold", type=float, default=DEFAULT_NOGATE_TOUCH_OBJECT_MOTION_THRESHOLD)
    parser.add_argument("--nogate-touch-stall-tolerance", type=float, default=DEFAULT_NOGATE_TOUCH_STALL_TOLERANCE)
    parser.add_argument("--nogate-post-touch-reposition-ticks", type=int, default=DEFAULT_NOGATE_POST_TOUCH_REPOSITION_TICKS)
    parser.add_argument("--nogate-finger-capture-tolerance", type=float, default=DEFAULT_NOGATE_FINGER_CAPTURE_TOLERANCE)
    parser.add_argument("--nogate-finger-capture-segment-margin", type=float, default=DEFAULT_NOGATE_FINGER_CAPTURE_SEGMENT_MARGIN)
    parser.add_argument("--nogate-require-table-touch-before-close", action=argparse.BooleanOptionalAction, default=DEFAULT_NOGATE_REQUIRE_TABLE_TOUCH_BEFORE_CLOSE)
    parser.add_argument("--nogate-post-close-slow-lift-enable", action=argparse.BooleanOptionalAction, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_ENABLE)
    parser.add_argument("--nogate-post-close-slow-lift-height", type=float, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_HEIGHT)
    parser.add_argument("--nogate-post-close-slow-lift-ticks", type=int, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_TICKS)
    parser.add_argument("--nogate-post-close-slow-lift-blend", type=float, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_BLEND)
    parser.add_argument("--nogate-post-close-slow-lift-max-step-norm", type=float, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_MAX_STEP_NORM)
    parser.add_argument("--nogate-post-close-slow-lift-max-abs-joint-step", type=float, default=DEFAULT_NOGATE_POST_CLOSE_SLOW_LIFT_MAX_ABS_JOINT_STEP)
    parser.add_argument("--no-live-coordinate-transform", action="store_true")
    parser.add_argument("--vertical-point-b-gap-above-support", type=float, default=DEFAULT_VERTICAL_POINT_B_GAP_ABOVE_SUPPORT)
    parser.add_argument("--vertical-close-point-b-tolerance", type=float, default=DEFAULT_VERTICAL_CLOSE_POINT_B_TOLERANCE)
    parser.add_argument("--vertical-xy-reference-link", default=DEFAULT_VERTICAL_XY_REFERENCE_LINK)
    parser.add_argument("--vertical-xy-reference-tolerance", type=float, default=DEFAULT_VERTICAL_XY_REFERENCE_TOLERANCE)
    parser.add_argument("--vertical-arm-lateral-bias-correction", type=float, default=DEFAULT_VERTICAL_ARM_LATERAL_BIAS_CORRECTION)
    parser.add_argument("--nogate-vertical-continuous-ik-descend", action=argparse.BooleanOptionalAction, default=DEFAULT_NOGATE_VERTICAL_CONTINUOUS_IK_DESCEND)
    parser.add_argument("--nogate-vertical-ik-refresh-period", type=int, default=DEFAULT_NOGATE_VERTICAL_IK_REFRESH_PERIOD)
    parser.add_argument("--nogate-vertical-descend-step-z", type=float, default=DEFAULT_NOGATE_VERTICAL_DESCEND_STEP_Z)
    parser.add_argument("--nogate-close-dls-enable", action=argparse.BooleanOptionalAction, default=DEFAULT_NOGATE_CLOSE_DLS_ENABLE)
    parser.add_argument("--nogate-close-dls-switch-distance", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_SWITCH_DISTANCE)
    parser.add_argument("--nogate-close-dls-max-iters", type=int, default=DEFAULT_NOGATE_CLOSE_DLS_MAX_ITERS)
    parser.add_argument("--nogate-close-dls-eps", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_EPS)
    parser.add_argument("--nogate-close-dls-damping", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_DAMPING)
    parser.add_argument("--nogate-close-dls-max-step", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_MAX_STEP)
    parser.add_argument("--nogate-close-dls-max-abs-joint-step", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_MAX_ABS_JOINT_STEP)
    parser.add_argument("--nogate-close-dls-blend", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_BLEND)
    parser.add_argument("--nogate-close-dls-settle-steps", type=int, default=DEFAULT_NOGATE_CLOSE_DLS_SETTLE_STEPS)
    parser.add_argument("--nogate-close-dls-hold-steps", type=int, default=DEFAULT_NOGATE_CLOSE_DLS_HOLD_STEPS)
    parser.add_argument("--nogate-close-dls-stop-tolerance", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_STOP_TOLERANCE)
    parser.add_argument("--nogate-close-dls-posture-gain", type=float, default=DEFAULT_NOGATE_CLOSE_DLS_POSTURE_GAIN)
    parser.add_argument("--align-clearance", type=float, default=DEFAULT_ALIGN_CLEARANCE)
    parser.add_argument("--descend-clearance", type=float, default=DEFAULT_DESCEND_CLEARANCE)
    parser.add_argument("--grasp-depth-offset", type=float, default=0.0)
    parser.add_argument("--micro-lift-probe", type=float, default=DEFAULT_MICRO_LIFT_PROBE)
    parser.add_argument("--micro-lift-min-delta", type=float, default=DEFAULT_MICRO_LIFT_MIN_DELTA)
    parser.add_argument("--lift-height", type=float, default=DEFAULT_LIFT_HEIGHT)
    parser.add_argument("--safe-drop-height", type=float, default=DEFAULT_SAFE_DROP_HEIGHT)
    parser.add_argument("--place-clearance", type=float, default=DEFAULT_PLACE_CLEARANCE)
    parser.add_argument("--retreat-lift", type=float, default=DEFAULT_RETREAT_LIFT)
    parser.add_argument("--min-transport-distance", type=float, default=DEFAULT_MIN_TRANSPORT_DISTANCE)
    parser.add_argument("--stable-jitter", type=float, default=DEFAULT_STABLE_JITTER)
    parser.add_argument("--min-ee-table-clearance", type=float, default=DEFAULT_MIN_EE_TABLE_CLEARANCE)
    parser.add_argument("--candidate-higher-offset", type=float, default=0.03)
    parser.add_argument("--candidate-farther-offset", type=float, default=0.03)
    parser.add_argument("--debug-pregrasp-pos-tol", type=float)
    parser.add_argument("--debug-pregrasp-rot-tol", type=float)
    parser.add_argument("--debug-fixed-rpy-right", type=float, nargs=3)
    parser.add_argument("--debug-fixed-rpy-left", type=float, nargs=3)
    parser.add_argument("--point-b-offset-local", type=float, nargs=3)
    parser.add_argument("--tcp-offset", type=float, nargs=3)
    parser.add_argument("--tcp-fallback-x", type=float, default=TCP_OFFSET_FALLBACK_X)
    parser.add_argument("--workspace-x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X)
    parser.add_argument("--workspace-y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y)
    parser.add_argument("--workspace-z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z)
    parser.add_argument("--gripper-hold-effort", type=float, default=DEFAULT_GRIPPER_HOLD_EFFORT)
    parser.add_argument("--joint-tolerance", type=float, default=0.06)
    parser.add_argument("--trace-interval", type=int, default=10)
    parser.add_argument("--skip-gripper-close", action="store_true")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--continue-after-lift", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--log-suffix")
    args = parser.parse_args()

    if args.target_index < 0:
        raise RuntimeError("--target-index must be non-negative")
    if args.servo_max_ticks < 1 or args.servo_carry_ticks < 1 or args.ik_max_iter < 1:
        raise RuntimeError("servo and IK iteration counts must be positive")
    if args.trace_interval < 1:
        raise RuntimeError("--trace-interval must be positive")
    if args.ik_refresh_period < 1 or not math.isfinite(float(args.ik_refresh_drift_threshold)) or args.ik_refresh_drift_threshold < 0.0:
        raise RuntimeError("--ik-refresh-period must be positive and --ik-refresh-drift-threshold must be finite non-negative")
    if not math.isfinite(float(args.far_threshold)) or not math.isfinite(float(args.near_body_threshold)):
        raise RuntimeError("region thresholds must be finite")
    if float(args.near_body_threshold) >= float(args.far_threshold):
        raise RuntimeError("--near-body-threshold must be less than --far-threshold")
    if args.candidate_ik_max_iter < 1:
        raise RuntimeError("--candidate-ik-max-iter must be positive")
    if args.candidate_ik_pos_tol <= 0.0 or args.candidate_ik_rot_tol <= 0.0:
        raise RuntimeError("--candidate-ik-pos-tol and --candidate-ik-rot-tol must be positive")
    if (
        not math.isfinite(float(args.far_candidate_position_tolerance))
        or not math.isfinite(float(args.far_candidate_rotation_tolerance))
        or args.far_candidate_position_tolerance <= 0.0
        or args.far_candidate_rotation_tolerance <= 0.0
    ):
        raise RuntimeError("--far-candidate-position-tolerance and --far-candidate-rotation-tolerance must be finite positive values")
    if args.debug_pregrasp_pos_tol is not None and args.debug_pregrasp_pos_tol <= 0.0:
        raise RuntimeError("--debug-pregrasp-pos-tol must be positive when provided")
    if args.debug_pregrasp_rot_tol is not None and args.debug_pregrasp_rot_tol <= 0.0:
        raise RuntimeError("--debug-pregrasp-rot-tol must be positive when provided")
    if (
        not math.isfinite(float(args.far_low_side_clearance))
        or not math.isfinite(float(args.far_point_b_gap_above_support))
        or not math.isfinite(float(args.far_xy_align_clearance_above_object))
        or not math.isfinite(float(args.far_point_b_forward_extension))
        or not math.isfinite(float(args.far_point_a_extra_height_clearance))
        or not math.isfinite(float(args.far_ab_downward_slant_deg))
        or not math.isfinite(float(args.far_outboard_transition_offset))
        or not math.isfinite(float(args.far_outboard_transition_clearance))
        or not math.isfinite(float(args.far_null_weight))
        or not math.isfinite(float(args.far_outboard_shoulder_roll_bias))
        or args.far_low_side_clearance < 0.0
        or args.far_point_b_gap_above_support < 0.0
        or args.far_xy_align_clearance_above_object < 0.0
        or args.far_point_b_forward_extension < 0.0
        or args.far_point_a_extra_height_clearance < 0.0
        or args.far_ab_downward_slant_deg < 0.0
        or args.far_ab_downward_slant_deg >= 80.0
        or args.far_outboard_transition_offset < 0.0
        or args.far_outboard_transition_clearance < 0.0
        or args.far_null_weight < 0.0
        or args.far_outboard_shoulder_roll_bias < 0.0
    ):
        raise RuntimeError("FAR geometry/null-space knobs must be finite non-negative values, with --far-ab-downward-slant-deg in [0, 80)")
    if (
        not math.isfinite(float(args.vertical_point_b_gap_above_support))
        or not math.isfinite(float(args.vertical_close_point_b_tolerance))
        or not math.isfinite(float(args.vertical_xy_reference_tolerance))
        or not math.isfinite(float(args.vertical_arm_lateral_bias_correction))
        or not math.isfinite(float(args.pre_close_point_b_tolerance))
        or args.vertical_point_b_gap_above_support < 0.0
        or args.vertical_arm_lateral_bias_correction < 0.0
        or args.vertical_close_point_b_tolerance <= 0.0
        or args.vertical_xy_reference_tolerance <= 0.0
        or args.pre_close_point_b_tolerance <= 0.0
    ):
        raise RuntimeError("vertical/pre-close contact knobs must be finite values, with non-negative gaps/corrections and positive tolerances")
    if (
        args.nogate_touch_max_ticks < 1
        or args.nogate_touch_fix_ticks < 0
        or args.nogate_vertical_ik_refresh_period < 1
        or not math.isfinite(float(args.nogate_vertical_descend_step_z))
        or not math.isfinite(float(args.nogate_touch_step_z))
        or not math.isfinite(float(args.nogate_touch_gap_above_table))
        or not math.isfinite(float(args.nogate_touch_xy_tolerance))
        or not math.isfinite(float(args.nogate_touch_object_expand))
        or not math.isfinite(float(args.nogate_touch_object_motion_threshold))
        or not math.isfinite(float(args.nogate_touch_stall_tolerance))
        or not math.isfinite(float(args.nogate_finger_capture_tolerance))
        or not math.isfinite(float(args.nogate_finger_capture_segment_margin))
        or args.nogate_post_touch_reposition_ticks < 0
        or args.nogate_vertical_descend_step_z <= 0.0
        or args.nogate_touch_step_z <= 0.0
        or args.nogate_touch_gap_above_table < 0.0
        or args.nogate_touch_xy_tolerance <= 0.0
        or args.nogate_touch_object_expand < 0.0
        or args.nogate_touch_object_motion_threshold < 0.0
        or args.nogate_touch_stall_tolerance < 0.0
        or args.nogate_finger_capture_tolerance <= 0.0
        or args.nogate_finger_capture_segment_margin < 0.0
    ):
        raise RuntimeError("no-gate pre-close/vertical touch knobs must be finite, with positive tick/step/tolerance values")
    if (
        args.nogate_close_dls_max_iters < 1
        or args.nogate_close_dls_settle_steps < 1
        or args.nogate_close_dls_hold_steps < 0
        or not math.isfinite(float(args.nogate_close_dls_switch_distance))
        or not math.isfinite(float(args.nogate_close_dls_eps))
        or not math.isfinite(float(args.nogate_close_dls_damping))
        or not math.isfinite(float(args.nogate_close_dls_max_step))
        or not math.isfinite(float(args.nogate_close_dls_max_abs_joint_step))
        or not math.isfinite(float(args.nogate_close_dls_blend))
        or not math.isfinite(float(args.nogate_close_dls_stop_tolerance))
        or not math.isfinite(float(args.nogate_close_dls_posture_gain))
        or args.nogate_close_dls_switch_distance <= 0.0
        or args.nogate_close_dls_eps <= 0.0
        or args.nogate_close_dls_damping <= 0.0
        or args.nogate_close_dls_max_step <= 0.0
        or args.nogate_close_dls_max_abs_joint_step <= 0.0
        or args.nogate_close_dls_blend <= 0.0
        or args.nogate_close_dls_blend > 1.0
        or args.nogate_close_dls_stop_tolerance <= 0.0
        or args.nogate_close_dls_posture_gain < 0.0
    ):
        raise RuntimeError(
            "no-gate near-contact DLS knobs must be finite, with positive distances/steps/tolerances, settle>=1, hold>=0, and blend in (0, 1]"
        )
    if (
        args.nogate_post_close_slow_lift_ticks < 1
        or not math.isfinite(float(args.nogate_post_close_slow_lift_height))
        or not math.isfinite(float(args.nogate_post_close_slow_lift_blend))
        or not math.isfinite(float(args.nogate_post_close_slow_lift_max_step_norm))
        or not math.isfinite(float(args.nogate_post_close_slow_lift_max_abs_joint_step))
        or args.nogate_post_close_slow_lift_height < 0.0
        or args.nogate_post_close_slow_lift_blend <= 0.0
        or args.nogate_post_close_slow_lift_blend > 1.0
        or args.nogate_post_close_slow_lift_max_step_norm <= 0.0
        or args.nogate_post_close_slow_lift_max_abs_joint_step <= 0.0
    ):
        raise RuntimeError("no-gate post-close slow-lift knobs must be finite, with positive ticks/blend/step values")
    if args.point_b_offset_local is not None and not np.isfinite(np.array(args.point_b_offset_local, dtype=float)).all():
        raise RuntimeError("--point-b-offset-local must contain finite values")
    if args.prim_path is not None and not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")
    args.workspace_x = tuple(float(value) for value in args.workspace_x)
    args.workspace_y = tuple(float(value) for value in args.workspace_y)
    args.workspace_z = tuple(float(value) for value in args.workspace_z)
    args.live_coordinate_transform = not bool(args.no_live_coordinate_transform)
    args.ee_frame_compensation_active = False
    args.ee_frame_compensation_by_arm = {}
    args.no_gate_mode = bool(NO_GATE_MODE)
    args.stop_after_lift = not bool(args.continue_after_lift)

    sys.argv = [sys.argv[0]]
    timestamp_utc = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp_utc": timestamp_utc.isoformat(),
            "timestamp_compact": timestamp_utc.strftime("%Y%m%dT%H%M%SZ"),
            "script_name": SCRIPT_NAME,
            "seed": int(args.seed),
            "gui_enabled": bool(args.no_headless or args.gui),
        },
        "motion_policy": {
            "architecture": "phase_based_official_dualarmik_6d_servo",
            "target_mode": args.target_mode,
            "old_reach_stack_imported": False,
            "measured_finite_difference_dls_active": False,
            "measured_jacobian_estimation_active": False,
            "position_only_control_active": False,
            "pinocchio_6d_solve_active": True,
            "orientation_control_active": True,
            "approach_family_selection_active": True,
            "target_selection_policy": str(args.target_selection_policy),
            "target_selection_nearness_metric": "forward_base",
            "approach_family_region_axis": "robot_base_forward_base",
            "region_thresholds": {
                "near_body_threshold": float(args.near_body_threshold),
                "far_threshold": float(args.far_threshold),
            },
            "region_to_approach_family_order": {
                "far": ["world_y_approach"],
                "mid": ["z_approach"],
                "near_body": ["z_approach", "world_y_approach"],
            },
            "far_candidate_acceptance_tolerances": {
                "position_tolerance_m": float(args.far_candidate_position_tolerance),
                "rotation_tolerance_rad": float(args.far_candidate_rotation_tolerance),
                "scope": "pregrasp_candidate_validation_only",
            },
            "far_low_side_height_policy": {
                "prepare_clearance_m": float(args.far_low_side_clearance),
                "point_b_gap_above_support_m": float(args.far_point_b_gap_above_support),
                "xy_align_clearance_above_object_m": float(args.far_xy_align_clearance_above_object),
                "point_b_forward_extension_m": float(args.far_point_b_forward_extension),
                "point_a_extra_height_clearance_m": float(args.far_point_a_extra_height_clearance),
                "ab_downward_slant_deg": float(args.far_ab_downward_slant_deg),
                "outboard_transition_offset_m": float(args.far_outboard_transition_offset),
                "outboard_transition_clearance_m": float(args.far_outboard_transition_clearance),
                "far_null_weight": float(args.far_null_weight),
                "far_outboard_shoulder_roll_bias_rad": float(args.far_outboard_shoulder_roll_bias),
            },
            "live_coordinate_transform_refresh": {
                "enabled": bool(args.live_coordinate_transform),
                "rule": "recompute base/world transform from live torso_link at servo phase start, periodic IK refreshes, and final error checks when periodic IK refresh is enabled",
            },
            "semi_closed_loop_ik_refresh": {
                "enabled": bool(args.ik_refresh_enable),
                "refresh_period_ticks": int(args.ik_refresh_period),
                "drift_threshold_m": float(args.ik_refresh_drift_threshold),
                "drift_refresh_enabled": bool(args.ik_refresh_enable and args.ik_refresh_drift_threshold > 0.0),
                "execution_rule": "solve DualArmIK periodically into q_goal, then track q_goal smoothly with the existing blend and step limiter between refreshes",
            },
            "mandatory_pre_close_gate": {
                "point_B_tolerance_m": float(args.pre_close_point_b_tolerance),
                "scope": "all motion policies before close_gripper",
                "disabled_by_no_gate_duplicate": bool(NO_GATE_MODE),
            },
            "nogate_near_contact_dls_handoff": {
                "enabled": bool(args.nogate_close_dls_enable),
                "switch_distance_m": float(args.nogate_close_dls_switch_distance),
                "source_algorithm": "task1_single_target_random_scene_baseline measured finite-difference DLS position controller",
                "scope": "no-gate vertical near-contact finish after live gripper metric reaches switch distance",
                "reduced_stop_go_policy": {
                    "max_iters": int(args.nogate_close_dls_max_iters),
                    "settle_steps_per_trial_and_command": int(args.nogate_close_dls_settle_steps),
                    "hold_steps": int(args.nogate_close_dls_hold_steps),
                    "max_step_norm": float(args.nogate_close_dls_max_step),
                    "max_abs_joint_step": float(args.nogate_close_dls_max_abs_joint_step),
                    "blend": float(args.nogate_close_dls_blend),
                    "stop_tolerance_m": float(args.nogate_close_dls_stop_tolerance),
                },
            },
            "nogate_preclose_touch_probe": {
                "enabled": bool(args.nogate_preclose_touch_enable),
                "max_ticks": int(args.nogate_touch_max_ticks),
                "fix_ticks": int(args.nogate_touch_fix_ticks),
                "step_z_m": float(args.nogate_touch_step_z),
                "gap_above_table_m": float(args.nogate_touch_gap_above_table),
                "xy_tolerance_m": float(args.nogate_touch_xy_tolerance),
                "object_bbox_expand_m": float(args.nogate_touch_object_expand),
                "object_motion_threshold_m": float(args.nogate_touch_object_motion_threshold),
                "stall_tolerance_m": float(args.nogate_touch_stall_tolerance),
                "post_touch_reposition_ticks": int(args.nogate_post_touch_reposition_ticks),
                "finger_capture_tolerance_m": float(args.nogate_finger_capture_tolerance),
                "finger_capture_segment_margin": float(args.nogate_finger_capture_segment_margin),
                "require_table_touch_before_close": bool(args.nogate_require_table_touch_before_close),
                "metric_priority": "real_grasp_center_world_then_point_B_proxy_world",
                "close_rule": "by default do not close unless table touch happened, IK is still valid, and the selected object center is between finger1_link and finger2_link; --no-nogate-require-table-touch-before-close allows selected-object touch as a diagnostic substitute",
            },
            "nogate_post_close_slow_lift_hold_grip": {
                "enabled": bool(args.nogate_post_close_slow_lift_enable),
                "height_m": float(args.nogate_post_close_slow_lift_height),
                "ticks": int(args.nogate_post_close_slow_lift_ticks),
                "blend": float(args.nogate_post_close_slow_lift_blend),
                "max_step_norm": float(args.nogate_post_close_slow_lift_max_step_norm),
                "max_abs_joint_step": float(args.nogate_post_close_slow_lift_max_abs_joint_step),
                "gripper_hold_effort": float(args.gripper_hold_effort),
                "rule": "after a non-skipped close_gripper phase, slowly lift in world Z while reissuing closed finger targets and applying gripper effort every tick",
            },
            "no_gate_duplicate": {
                "enabled": bool(NO_GATE_MODE),
                "fatal_reasons_still_enabled": sorted(NO_GATE_FATAL_REASONS),
                "behavior": "motion/contact/lift/carry/place/scoring gates log and continue; missing scene/assets/targets still fail",
            },
            "AB_motion_semantics_active": True,
            "far_motion_policy": "low_side_prepare_then_xy_align_then_world_z_descend",
            "mid_motion_policy": "vertical_AB_world_Z_descend",
            "vertical_contact_policy": {
                "point_b_gap_above_support_m": float(args.vertical_point_b_gap_above_support),
                "close_point_b_tolerance_m": float(args.vertical_close_point_b_tolerance),
                "xy_reference_link": str(args.vertical_xy_reference_link),
                "xy_reference_tolerance_m": float(args.vertical_xy_reference_tolerance),
                "arm_lateral_bias_correction_m": float(args.vertical_arm_lateral_bias_correction),
                "nogate_continuous_ik_descend": bool(args.nogate_vertical_continuous_ik_descend),
                "nogate_descend_step_z_m": float(args.nogate_vertical_descend_step_z),
                "nogate_vertical_ik_refresh_period_ticks": int(args.nogate_vertical_ik_refresh_period) if bool(args.nogate_vertical_continuous_ik_descend) else int(args.ik_refresh_period),
                "nogate_vertical_effective_z_step_per_frame_m": float(args.nogate_vertical_descend_step_z) / float(max(1, int(args.nogate_vertical_ik_refresh_period))),
                "arm_lateral_bias_correction_rule": "right arm -baseY, left arm +baseY for vertical-only XY target correction",
                "close_after_point_B_contact_gate": True,
            },
            "stop_after_lift": bool(args.stop_after_lift),
            "phase_order": PHASE_ORDER,
        },
        "phase_log": [],
        "result_flags": {
            "object_lifted": False,
            "object_transported": False,
            "final_inside_bin": False,
            "object_stable": False,
        },
        "object_trace": {},
        "final_status": "fail",
        "failure_reason": "runtime_error",
        "no_gate_mode": {
            "enabled": bool(NO_GATE_MODE),
            "fatal_reasons_still_enabled": sorted(NO_GATE_FATAL_REASONS),
        },
        "no_gate_bypassed_failures": NO_GATE_BYPASSED_FAILURES,
    }
    sim_app = None
    timeline = None

    try:
        paths = _validate_environment()
        baseline_root = _as_path(args.baseline_root, paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
        asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
        config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
        box_usd = asset_root / DEFAULT_BOX_RELATIVE
        urdf_path = asset_root / "s2.urdf"
        log_root = Path(paths["LOG_ROOT"]).resolve()
        payload["run_metadata"].update(
            {
                "repo_path": str(paths["HRC_REPO"]),
                "baseline_root": str(baseline_root),
                "asset_root": str(asset_root),
                "yaml_path": str(config_path),
                "urdf_path": str(urdf_path),
                "log_root": str(log_root),
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
        if not urdf_path.exists():
            _fail("scene_build_failed", f"Official Walker S2 URDF missing: {urdf_path}")

        random.seed(args.seed)
        np.random.seed(args.seed)
        SimulationApp = _load_simulation_app()
        sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
        counter = {"step": 0}

        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        if args.ik_rot_weight is None:
            args.ik_rot_weight = float(cfg.get("grasp", {}).get("ik_rot_weight", 1.0))
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
        scene_robot_prim_path = getattr(scene, "robot_prim_path", None)

        rep.orchestrator.step()
        _run_updates(sim_app, args.init_steps, counter)
        detected_articulation_roots = _find_articulation_roots_anywhere(stage)
        robot_path_selection = _choose_robot_prim_path(stage, scene_robot_prim_path, detected_articulation_roots, args.prim_path)
        chosen_robot_prim_path = robot_path_selection["chosen_robot_prim_path"]
        articulation_path = detected_articulation_roots[0] if detected_articulation_roots else chosen_robot_prim_path
        payload["robot_path_selection"] = robot_path_selection

        part_paths = list(getattr(scene, "parts_prim_paths", []))
        if not part_paths:
            _fail("no_target_parts_found", "SceneBuilder did not expose any Task 1 part prim paths")
        if args.target_index >= len(part_paths):
            _fail("target_index_out_of_range", f"--target-index {args.target_index} out of range for {len(part_paths)} parts")

        table_path = "/Replicator/Ref_Xform"
        table_bbox = _bbox(stage, table_path)
        table_top_z = float(table_bbox["max"][2])
        robot_base_position = np.array(configured_robot_position, dtype=float)
        robot_base_yaw_rad = math.radians(float(configured_robot_rotation[2]) if len(configured_robot_rotation) >= 3 else 0.0)
        target_selection = _select_target_record(
            stage=stage,
            part_paths=part_paths,
            requested_target_index=int(args.target_index),
            selection_policy=str(args.target_selection_policy),
            num_parts_per_class=int(cfg["part"].get("num_parts", 2)),
            robot_base_position=robot_base_position,
            robot_base_yaw_rad=robot_base_yaw_rad,
        )
        selected_target_record = target_selection["selected_record"]
        target_path = str(target_selection["selected_prim_path"])
        selected_target_index = int(target_selection["selected_target_index"])
        target_category = dict(selected_target_record["category"])
        initial_state = selected_target_record["initial_state"]
        initial_center = np.array(selected_target_record["center_world"], dtype=float)
        target_components = dict(selected_target_record["robot_base_target_components"])
        chosen_arm = _choose_arm_side(args, target_components)
        forward_base = float(target_components["forward_base"])
        target_region = _classify_target_region(forward_base, args)
        approach_family_order = _approach_family_order_for_region(target_region)
        payload["scene"] = {
            "config_path": str(config_path),
            "original_config_root_path": original_root_path,
            "overridden_root_path": str(asset_root),
            "scene_builder_methods": ["build_table", "build_parts", "build_robot"],
            "official_box_pipeline_used_for_destination_physics": False,
            "spawned_part_prim_list": part_paths,
            "table_prim": table_path,
            "table": {"bbox": table_bbox, "physics": _physics_summary(stage, table_path)},
            "debug_marker_paths": [],
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
            "target_index": selected_target_index,
            "requested_target_index": int(args.target_index),
            "target_selection_policy": str(args.target_selection_policy),
            "target_selection_reason": target_selection["selection_reason"],
            "robot_to_object_nearness_metric_name": target_selection["selected_nearness_metric_name"],
            "robot_to_object_nearness_metric_value": target_selection["selected_nearness_metric_value"],
            "initial_pose": initial_state,
            "robot_base_target_components": target_components,
            "forward_base": forward_base,
            "target_region": target_region,
            "approach_family_order": approach_family_order,
            "region_thresholds": {
                "near_body_threshold": float(args.near_body_threshold),
                "far_threshold": float(args.far_threshold),
            },
            **target_category,
        }
        payload["target_selection"] = target_selection
        payload["object_trace"]["initial"] = initial_state

        phase_log: list[dict[str, Any]] = payload["phase_log"]
        select_start = counter["step"]
        _append_phase(
            phase_log,
            phase="select_target",
            start_step=select_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "target_position": initial_center.tolist(),
                "failure_reason": None,
                "selected_target": payload["target"],
                "target_selection": target_selection,
                "chosen_arm_preliminary": chosen_arm,
                "forward_base": forward_base,
                "target_region": target_region,
                "approach_family_order": approach_family_order,
            },
        )
        print(
            f"phase=select_target target={target_path} center={initial_center.tolist()} "
            f"selection_policy={args.target_selection_policy} selected_index={selected_target_index} "
            f"chosen_arm={chosen_arm} target_region={target_region} forward_base={forward_base:.3f}"
        )

        joint_names = _find_joint_names(stage, chosen_robot_prim_path)
        if not detected_articulation_roots and not _valid_prim_path(stage, articulation_path):
            _fail("scene_build_failed", "Walker S2 loaded, but no articulation root was detected")

        timeline = _start_timeline()
        _run_updates(sim_app, 5, counter)
        dc, articulation, articulation_acquire = _acquire_articulation_with_fallback(articulation_path, chosen_robot_prim_path)
        dof_observation = _read_dof_observation(dc, articulation)

        DualArmIK, CoordinateTransform = _load_official_ik_classes(baseline_root)
        arm_joint_names = list(DualArmIK.RIGHT_ARM_JOINTS if chosen_arm == "right" else DualArmIK.LEFT_ARM_JOINTS)
        arm_dofs = _select_dofs_in_name_order(dc, articulation, arm_joint_names)
        gripper_dofs = _select_gripper_dofs(dc, articulation, chosen_arm)
        end_effector_body, end_effector_name, end_effector_path, end_effector_policy = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
            chosen_arm,
        )

        official_startup_joint_map = _load_official_startup_joint_map(baseline_root, chosen_robot_prim_path, urdf_path)
        startup_dofs, missing_official_startup_optional_dofs = _select_dofs_by_target_names(
            dc,
            articulation,
            official_startup_joint_map,
            OFFICIAL_STARTUP_ARM_JOINT_NAMES,
        )
        official_startup_targets = _targets_from_map(startup_dofs, official_startup_joint_map)

        payload["robot"] = {
            "usd_path": str(robot_usd),
            "scene_robot_prim_path": scene_robot_prim_path,
            "detected_articulation_roots": detected_articulation_roots,
            "chosen_robot_prim_path": chosen_robot_prim_path,
            "robot_path_selection": robot_path_selection,
            "articulation_path": articulation_path,
            "articulation_acquire": articulation_acquire,
            "joint_count": len(joint_names),
            "chosen_arm": chosen_arm,
            "arm_dof_names": [name for _, _, name in arm_dofs],
            "gripper_dof_names": [name for _, _, name in gripper_dofs],
            "official_startup_dof_names": [name for _, _, name in startup_dofs],
            "missing_optional_official_startup_dofs": missing_official_startup_optional_dofs,
            "official_startup_source": "lerobot.common.robot_devices.robots.isaac_sim_robot_interface.IsaacSimRobotInterface._joint_value_map",
            "official_startup_baseline_source": "Ubtech_sim/source/RobotArticulation.py uses the same _joint_value_map for initialization",
            "dualarmik_class": "Ubtech_sim/source/DualArmIK.py::DualArmIK",
            "coordinate_transform_class": "Ubtech_sim/source/coordinate_utils.py::CoordinateTransform",
            "dualarmik_arm_joint_source": f"DualArmIK.{'RIGHT' if chosen_arm == 'right' else 'LEFT'}_ARM_JOINTS",
            "dualarmik_ee_frame_source": f"DualArmIK.{'RIGHT' if chosen_arm == 'right' else 'LEFT'}_EE_FRAME",
            "end_effector_name": end_effector_name,
            "end_effector_path": end_effector_path,
            "end_effector_policy": end_effector_policy,
            "end_effector_preference": f"{'R' if chosen_arm == 'right' else 'L'}_sixforce_link",
            "dof_observation_sample": dof_observation[:12],
        }

        startup_start = counter["step"]
        startup_seed = _seed_joint_positions_for_initialization(dc, startup_dofs, official_startup_targets)
        _run_updates(sim_app, 12, counter)
        observed_startup = _current_positions(dc, startup_dofs)
        ee_startup = _body_pose_position(dc, end_effector_body)
        startup_max_error = max(abs(float(obs - target)) for obs, target in zip(observed_startup, official_startup_targets))
        startup_ok = bool(startup_seed["supported"] and startup_max_error <= args.joint_tolerance)
        _append_phase(
            phase_log,
            phase="startup_official_pose",
            start_step=startup_start,
            end_step=counter["step"],
            condition_met=startup_ok,
            details={
                "target_position": None,
                "final_ee_position": ee_startup.tolist(),
                "final_error": startup_max_error,
                "failure_reason": None if startup_ok else "official_startup_pose_failed",
                "chosen_ee_frame_name": end_effector_name,
                "chosen_ee_frame_path": end_effector_path,
                "chosen_ee_frame_policy": end_effector_policy,
                "official_startup_joint_values": official_startup_joint_map,
                "observed_joint_values": _named_positions(startup_dofs, observed_startup),
                "max_joint_error": startup_max_error,
                "joint_tolerance": args.joint_tolerance,
                "initialization_seed_result": startup_seed,
            },
        )
        if not startup_ok:
            _fail("official_startup_pose_failed", "Official startup joint pose was not reached within tolerance")

        ik_solver = DualArmIK(str(urdf_path))
        all_names, all_positions = _all_joint_state_for_ik(dc, articulation)
        ik_solver.sync_joint_positions(all_names, all_positions)
        ik_solver.save_initial_q()
        left_neutral_outboard = [official_startup_joint_map.get(joint, 0.0) for joint in DualArmIK.LEFT_ARM_JOINTS]
        right_neutral_outboard = [official_startup_joint_map.get(joint, 0.0) for joint in DualArmIK.RIGHT_ARM_JOINTS]
        far_outboard_bias = float(args.far_outboard_shoulder_roll_bias)
        try:
            left_neutral_outboard[list(DualArmIK.LEFT_ARM_JOINTS).index("L_shoulder_roll_joint")] += far_outboard_bias
            right_neutral_outboard[list(DualArmIK.RIGHT_ARM_JOINTS).index("R_shoulder_roll_joint")] -= far_outboard_bias
        except ValueError:
            pass
        ik_solver.set_neutral_config(left_neutral_outboard, right_neutral_outboard)
        payload["robot"]["far_nullspace_neutral_outboard"] = {
            "source": "official_startup_pose_with_arm_side_shoulder_roll_outboard_bias",
            "far_null_weight": float(args.far_null_weight),
            "far_outboard_shoulder_roll_bias_rad": far_outboard_bias,
            "left_arm_joint_names": list(DualArmIK.LEFT_ARM_JOINTS),
            "left_neutral_outboard": left_neutral_outboard,
            "right_arm_joint_names": list(DualArmIK.RIGHT_ARM_JOINTS),
            "right_neutral_outboard": right_neutral_outboard,
            "scope": "FAR servo phases only via ik_overrides",
        }
        torso_prim_path, torso_path_attempts = _resolve_torso_prim_path(stage, chosen_robot_prim_path, articulation_acquire["acquired_articulation_path"])
        coord_transform, coordinate_alignment_selection = _select_coordinate_transform_with_alignment(
            CoordinateTransform,
            ik_solver,
            stage,
            torso_prim_path,
            chosen_robot_prim_path,
        )
        print(
            "frame_alignment "
            f"selected={coordinate_alignment_selection['selected_label']} "
            f"max_diff_m={coordinate_alignment_selection['selected_score_max_diff_m']:.4f} "
            f"mean_diff_m={coordinate_alignment_selection['selected_score_mean_diff_m']:.4f}"
        )
        try:
            ee_alignment = coord_transform.verify_ee_alignment(ik_solver)
        except Exception as exc:
            ee_alignment = [{"error": str(exc)}]
        try:
            ee_alignment_raw_dynamic = _verify_ee_alignment_dynamic(stage, coord_transform, ik_solver, chosen_robot_prim_path)
        except Exception as exc:
            ee_alignment_raw_dynamic = {"error": str(exc)}
        try:
            ee_frame_delta_diagnostics = _compute_ee_frame_delta_diagnostics(stage, coord_transform, ik_solver, chosen_robot_prim_path)
        except Exception as exc:
            ee_frame_delta_diagnostics = {"error": str(exc), "comparison": {"root_cause_classification": "ee_delta_diagnostics_failed"}}
        ee_frame_compensation = _configure_ee_frame_compensation(args, ee_frame_delta_diagnostics)

        coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None
        if bool(args.live_coordinate_transform):
            def coord_transform_refresh_fn() -> dict[str, Any]:
                _sync_ik_from_dc(ik_solver, dc, articulation)
                return _refresh_coordinate_transform_from_selection(
                    CoordinateTransform=CoordinateTransform,
                    ik_solver=ik_solver,
                    stage=stage,
                    coord_transform=coord_transform,
                    torso_prim_path=torso_prim_path,
                    coordinate_alignment_selection=coordinate_alignment_selection,
                )

        try:
            ee_alignment_after_fix = _verify_ee_alignment_dynamic(
                stage,
                coord_transform,
                ik_solver,
                chosen_robot_prim_path,
                ee_compensation_by_arm=getattr(args, "ee_frame_compensation_by_arm", {}) if getattr(args, "ee_frame_compensation_active", False) else None,
            )
        except Exception as exc:
            ee_alignment_after_fix = {"error": str(exc)}
        print(
            "ee_frame_delta "
            f"classification={ee_frame_delta_diagnostics.get('comparison', {}).get('root_cause_classification')} "
            f"compensation_active={ee_frame_compensation['active']} "
            f"post_max_diff_m={float(ee_alignment_after_fix.get('max_diff_m', math.inf)):.4f}"
        )
        startup_frame_alignment = {
            "torso_prim_path_used": torso_prim_path,
            "coordinate_transform_selected_label": coordinate_alignment_selection["selected_label"],
            "coordinate_transform_world_origin": np.array(coord_transform.robot_world_pos, dtype=float).tolist(),
            "coordinate_transform_world_rotation": np.array(coord_transform.robot_world_R, dtype=float).tolist(),
            "fk_vs_isaac_ee_alignment_raw": ee_alignment_raw_dynamic,
            "ee_frame_delta_diagnostics": ee_frame_delta_diagnostics,
            "ee_frame_compensation": ee_frame_compensation,
            "fk_vs_isaac_ee_alignment_after_sync": ee_alignment_after_fix,
        }
        for phase in phase_log:
            if phase["phase"] == "startup_official_pose":
                phase["details"]["frame_alignment_after_startup_sync"] = startup_frame_alignment
                break

        payload["robot"].update(
            {
                "torso_prim_path": torso_prim_path,
                "torso_prim_path_used": torso_prim_path,
                "torso_prim_path_attempts": torso_path_attempts,
                "coordinate_transform_robot_world_pos": np.array(coord_transform.robot_world_pos, dtype=float).tolist(),
                "coordinate_transform_robot_world_R": np.array(coord_transform.robot_world_R, dtype=float).tolist(),
                "coordinate_transform_world_origin": np.array(coord_transform.robot_world_pos, dtype=float).tolist(),
                "coordinate_transform_world_rotation": np.array(coord_transform.robot_world_R, dtype=float).tolist(),
                "coordinate_transform_alignment_selection": coordinate_alignment_selection,
                "ee_alignment_check": ee_alignment,
                "ee_alignment_check_raw_dynamic": ee_alignment_raw_dynamic,
                "ee_frame_delta_diagnostics": ee_frame_delta_diagnostics,
                "ee_frame_compensation": ee_frame_compensation,
                "ee_alignment_check_after_fix": ee_alignment_after_fix,
                "ee_alignment_diagnostics_after_fix": ee_alignment_after_fix,
                "startup_frame_alignment": startup_frame_alignment,
                "live_coordinate_transform_refresh_enabled": bool(args.live_coordinate_transform),
                "live_coordinate_transform_refresh_rule": "recompute base/world transform at servo phase start, periodic IK refreshes, and final error checks when periodic IK refresh is enabled",
                "semi_closed_loop_ik_refresh_enabled": bool(args.ik_refresh_enable),
                "ik_refresh_period_ticks": int(args.ik_refresh_period),
                "ik_refresh_drift_threshold_m": float(args.ik_refresh_drift_threshold),
            }
        )

        open_ok = _command_gripper_phase(
            "open_gripper_initial",
            dc=dc,
            gripper_dofs=gripper_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            target_positions=[OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs),
            sim_app=sim_app,
            steps=args.gripper_steps,
            counter=counter,
            phase_log=phase_log,
            effort_value=0.0,
        )
        if not open_ok:
            _fail("gripper_command_failed", f"{chosen_arm} gripper open command failed before approach")

        tcp_offset, tcp_log = _resolve_tcp_offset(cfg, args)
        point_b_offset, point_b_log = _resolve_point_b_offset_local(tcp_offset, args)
        args.point_b_offset_local_resolved = point_b_offset.tolist()
        vertical_xy_reference_offset, vertical_xy_reference_log = _resolve_vertical_xy_reference_offset(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=chosen_robot_prim_path,
            coord_transform=coord_transform,
            ik_solver=ik_solver,
            arm_side=chosen_arm,
            args=args,
        )
        work_area_world = _work_area_world_from_cfg(cfg, table_bbox)
        _sync_ik_from_dc(ik_solver, dc, articulation)
        derived_downward_rpy_by_arm = _fixed_downward_rpy_by_arm(
            ik_solver,
            coord_transform,
            work_area_world,
            debug_fixed_rpy_by_arm={
                "right": args.debug_fixed_rpy_right,
                "left": args.debug_fixed_rpy_left,
            },
        )
        orientation_presets, preset_selection_log = _region_filtered_orientation_presets(
            chosen_arm,
            target_region,
            args,
            coord_transform,
        )
        preset_selection_log = {
            **preset_selection_log,
            "forward_base": forward_base,
            "region_thresholds": {
                "near_body_threshold": float(args.near_body_threshold),
                "far_threshold": float(args.far_threshold),
            },
        }

        derived_geometry_reference = _plan_grasp_geometry(
            object_state=initial_state,
            table_top_z=table_top_z,
            bin_bbox=bin_bbox,
            bin_floor_top_z=float(bin_collider["floor_top_z"]),
            coord_transform=coord_transform,
            arm_side=chosen_arm,
            downward_rpy_by_arm=derived_downward_rpy_by_arm,
            tcp_offset_local=tcp_offset,
            args=args,
            target_region=target_region,
            motion_family=approach_family_order[0],
            point_b_offset_local=point_b_offset,
            vertical_xy_reference_offset_local=vertical_xy_reference_offset,
            vertical_xy_reference_log=vertical_xy_reference_log,
        )
        orientation_preset_library = _orientation_presets_by_arm_and_family(args, coord_transform=coord_transform)
        payload["tcp_offset"] = tcp_log
        payload["point_ab_semantics"] = point_b_log
        payload["vertical_xy_reference"] = vertical_xy_reference_log
        payload["fixed_downward_orientation_reference"] = derived_downward_rpy_by_arm
        payload["orientation_preset_library"] = orientation_preset_library
        payload["approach_region_selection"] = preset_selection_log
        payload["derived_grasp_geometry_reference"] = derived_geometry_reference
        plan_start = counter["step"]
        _append_phase(
            phase_log,
            phase="plan_grasp_geometry",
            start_step=plan_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "target_pose_base": derived_geometry_reference["contact_pose_base"],
                "target_ee_origin_world": derived_geometry_reference["contact_ee_origin_world"],
                "failure_reason": None,
                "geometry_reference": derived_geometry_reference,
                "tcp_offset": tcp_log,
                "point_ab_semantics": point_b_log,
                "vertical_xy_reference": vertical_xy_reference_log,
                "fixed_downward_orientation_reference": derived_downward_rpy_by_arm,
                "orientation_preset_library": orientation_preset_library,
                "approach_region_selection": preset_selection_log,
                "selected_approach_filtered_presets": orientation_presets,
                "candidate_geometry_policy": "region_filtered_approach_family_preset_library_for_candidate_solving",
            },
        )
        print(
            "phase=plan_grasp_geometry "
            f"reference_contact={derived_geometry_reference['contact_point_world']} "
            f"reference_ee_pose_base={derived_geometry_reference['contact_pose_base']} "
            f"target_region={target_region} "
            f"approach_family_order={approach_family_order} "
            f"preset_count={len(orientation_presets)}"
        )

        selected_candidate = _evaluate_pregrasp_candidates(
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            object_state=initial_state,
            table_top_z=table_top_z,
            bin_bbox=bin_bbox,
            bin_floor_top_z=float(bin_collider["floor_top_z"]),
            tcp_offset_local=tcp_offset,
            orientation_presets=orientation_presets,
            target_region=target_region,
            point_b_offset_local=point_b_offset,
            vertical_xy_reference_offset_local=vertical_xy_reference_offset,
            vertical_xy_reference_log=vertical_xy_reference_log,
            forward_base=forward_base,
            approach_family_order=approach_family_order,
            args=args,
            phase_log=phase_log,
            counter=counter,
        )
        geometry = selected_candidate["geometry"]
        args.point_b_offset_local_resolved = list(geometry.get("point_b_offset_local", point_b_offset.tolist()))
        selected_orientation_preset = selected_candidate["selected_orientation_preset"]
        payload["selected_approach_family"] = selected_orientation_preset.get("preset_family")
        payload["selected_grasp_family"] = selected_orientation_preset.get("preset_family")
        payload["target_region"] = target_region
        payload["forward_base"] = forward_base
        payload["approach_family_order"] = approach_family_order
        payload["grasp_geometry"] = geometry
        payload["selected_motion_policy"] = geometry.get("motion_policy")
        payload["selected_point_A_proxy"] = geometry.get("point_A_proxy_definition")
        payload["selected_point_B_proxy"] = geometry.get("point_B_proxy_definition")
        payload["selected_contact_point_B_world"] = geometry.get("contact_point_B_world")
        payload["selected_contact_point_A_world"] = geometry.get("contact_AB_semantics", {}).get("point_A_world")
        payload["selected_far_outboard_transition_B_world"] = geometry.get("far_outboard_transition_B_world")
        payload["selected_far_outboard_axis_world"] = geometry.get("far_outboard_axis_world")
        payload["selected_far_outboard_transition_offset_m"] = geometry.get("far_outboard_transition_offset_m")
        payload["selected_far_outboard_transition_clearance_m"] = geometry.get("far_outboard_transition_clearance_m")
        payload["selected_far_low_side_prepare_B_world"] = geometry.get("far_low_side_prepare_B_world")
        payload["selected_far_xy_align_B_world"] = geometry.get("far_xy_align_B_world")
        payload["selected_far_descend_B_world"] = geometry.get("far_descend_B_world")
        payload["selected_far_contact_sequence_policy"] = geometry.get("far_contact_sequence_policy")
        payload["selected_far_point_b_forward_extension_m"] = geometry.get("far_point_b_forward_extension_m")
        payload["selected_far_point_a_extra_height_clearance_m"] = geometry.get("far_point_a_extra_height_clearance_m")
        payload["selected_far_point_a_min_extra_height_clearance_m"] = geometry.get("far_point_a_min_extra_height_clearance_m")
        payload["selected_far_ab_requested_downward_slant_deg"] = geometry.get("far_ab_requested_downward_slant_deg")
        payload["selected_far_ab_downward_slant_deg"] = geometry.get("far_ab_downward_slant_deg")
        payload["selected_far_ab_slant_height_clearance_m"] = geometry.get("far_ab_slant_height_clearance_m")
        payload["selected_far_ab_base_horizontal_span_m"] = geometry.get("far_ab_base_horizontal_span_m")
        payload["selected_far_point_b_gap_above_support_m"] = geometry.get("far_point_b_gap_above_support_m")
        payload["selected_far_xy_align_clearance_above_object_m"] = geometry.get("far_xy_align_clearance_above_object_m")
        payload["selected_far_reach_axis_world"] = geometry.get("far_reach_axis_world")
        payload["selected_vertical_contact_sequence_policy"] = geometry.get("vertical_contact_sequence_policy")
        payload["selected_vertical_contact_mark_B_world"] = geometry.get("vertical_contact_mark_B_world")
        payload["selected_vertical_point_b_gap_above_support_m"] = geometry.get("vertical_point_b_gap_above_support_m")
        payload["selected_vertical_close_point_b_tolerance_m"] = geometry.get("vertical_close_point_b_tolerance_m")
        payload["selected_vertical_xy_reference_active"] = geometry.get("vertical_xy_reference_active")
        payload["selected_vertical_xy_reference_mode"] = geometry.get("vertical_xy_reference_mode")
        payload["selected_vertical_xy_reference_source"] = geometry.get("vertical_xy_reference_source")
        payload["selected_vertical_xy_reference_target_xy_world"] = geometry.get("vertical_xy_reference_target_xy_world")
        payload["selected_vertical_uncorrected_object_world_xy_target"] = geometry.get("vertical_uncorrected_object_world_xy_target")
        payload["selected_vertical_arm_lateral_bias_correction_m"] = geometry.get("vertical_arm_lateral_bias_correction_m")
        payload["selected_vertical_arm_lateral_bias_correction_base_y_m"] = geometry.get("vertical_arm_lateral_bias_correction_base_y_m")
        payload["selected_vertical_arm_lateral_bias_correction_world"] = geometry.get("vertical_arm_lateral_bias_correction_world")
        payload["selected_vertical_arm_lateral_bias_correction_rule"] = geometry.get("vertical_arm_lateral_bias_correction_rule")
        payload["selected_vertical_xy_reference_world_position_used_for_offset"] = geometry.get("vertical_xy_reference_world_position_used_for_offset")
        payload["selected_vertical_xy_reference_component_logs"] = geometry.get("vertical_xy_reference_component_logs")
        payload["selected_vertical_xy_reference_offset_local"] = geometry.get("vertical_xy_reference_offset_local")
        payload["selected_vertical_raw_point_B_contact_mark_before_xy_reference"] = geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference")
        payload["selected_contact_AB_semantics"] = geometry.get("contact_AB_semantics")
        payload["selected_orientation_preset"] = selected_orientation_preset
        payload["selected_orientation_preset_label"] = selected_orientation_preset["preset_label"]
        payload["selected_orientation_preset_rpy"] = selected_orientation_preset["rpy"]
        payload["selected_orientation_preset_approach_axis_world"] = selected_orientation_preset.get("approach_axis_world")
        payload["selected_orientation_preset_AB_axis_world"] = selected_orientation_preset.get("AB_axis_world")
        payload["selected_orientation_preset_axial_roll_variant_label"] = selected_orientation_preset.get("preset_axial_roll_variant_label")
        payload["selected_orientation_preset_axial_roll_about_ab_rad"] = selected_orientation_preset.get("preset_axial_roll_about_ab_rad")
        payload["selected_orientation_preset_dot_with_world_pos_y"] = selected_orientation_preset.get("dot_with_world_pos_y")
        payload["selected_orientation_preset_dot_with_world_neg_y"] = selected_orientation_preset.get("dot_with_world_neg_y")
        motion_policy = str(geometry.get("motion_policy", "mid_vertical_Z_descend"))
        far_motion_policy = motion_policy == "far_low_side_B_driven"
        vertical_prefix = "mid" if target_region == "mid" else "near_body"
        pregrasp_phase_name = "far_prepare_low_side_approach" if far_motion_policy else f"{vertical_prefix}_align_AB_vertical_over_object"
        contact_phase_name = "far_align_B_over_object_xy" if far_motion_policy else f"{vertical_prefix}_pre_descend_AB_vertical"
        descend_phase_name = "far_lower_B_world_z" if far_motion_policy else f"{vertical_prefix}_descend_world_z_keep_AB_vertical"

        debug_markers = [
            _create_debug_marker(stage, "/World/DebugDualArmIKTarget", initial_center, 0.025, (1.0, 0.2, 0.1)),
            _create_debug_marker(stage, "/World/DebugDualArmIKPregraspB", geometry["pregrasp_point_B_world"], 0.025, (0.2, 0.6, 1.0)),
            _create_debug_marker(stage, "/World/DebugDualArmIKContactB", geometry["contact_point_B_world"], 0.022, (1.0, 0.8, 0.1)),
            _create_debug_marker(stage, "/World/DebugDualArmIKContactA", geometry["contact_AB_semantics"]["point_A_world"], 0.018, (0.9, 0.2, 0.9)),
            _create_debug_marker(stage, "/World/DebugDualArmIKBin", bin_bbox["center"], 0.03, (0.2, 1.0, 0.2)),
        ]
        if geometry.get("far_outboard_transition_B_world") is not None:
            debug_markers.append(
                _create_debug_marker(stage, "/World/DebugDualArmIKFarOutboardTransitionB", geometry["far_outboard_transition_B_world"], 0.024, (0.9, 0.4, 0.1))
            )
        if geometry.get("far_low_side_prepare_B_world") is not None:
            debug_markers.append(
                _create_debug_marker(stage, "/World/DebugDualArmIKFarLowSidePrepareB", geometry["far_low_side_prepare_B_world"], 0.024, (0.1, 0.9, 0.9))
            )
        if geometry.get("far_xy_align_B_world") is not None:
            debug_markers.append(
                _create_debug_marker(stage, "/World/DebugDualArmIKFarXYAlignB", geometry["far_xy_align_B_world"], 0.022, (0.2, 0.9, 0.4))
            )
        payload["scene"]["debug_marker_paths"] = debug_markers

        far_ik_overrides = {"null_weight": float(args.far_null_weight)} if far_motion_policy else None
        if far_motion_policy and geometry.get("far_outboard_transition_pose_base") is not None:
            far_outboard_result = _execute_dualarmik_servo_phase(
                ServoSpec(
                    "far_outboard_transition",
                    np.array(geometry["far_outboard_transition_pose_base"], dtype=float),
                    args.pregrasp_tolerance,
                    args.rot_tolerance,
                    args.servo_max_ticks,
                ),
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                arm_side=chosen_arm,
                coord_transform=coord_transform,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                ik_overrides=far_ik_overrides,
                extra_details={
                    "motion_policy": motion_policy,
                    "far_contact_sequence_policy": geometry.get("far_contact_sequence_policy"),
                    "waypoint_role": "lateral_outboard_clearance_before_low_side_prepare",
                    "target_point_B_world": geometry.get("far_outboard_transition_B_world"),
                    "far_outboard_transition_B_world": geometry.get("far_outboard_transition_B_world"),
                    "far_outboard_axis_world": geometry.get("far_outboard_axis_world"),
                    "far_outboard_transition_offset_m": geometry.get("far_outboard_transition_offset_m"),
                    "far_outboard_transition_clearance_m": geometry.get("far_outboard_transition_clearance_m"),
                    "far_null_weight": float(args.far_null_weight),
                    "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                    "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                },
            )
            if far_outboard_result["final_error"] > args.pregrasp_tolerance:
                _fail("far_outboard_transition_failed", "far_outboard_transition did not reach the lateral clearance waypoint")

        pregrasp_result = _execute_dualarmik_servo_phase(
            ServoSpec(pregrasp_phase_name, np.array(selected_candidate["pose_base"], dtype=float), args.pregrasp_tolerance, args.rot_tolerance, args.servo_max_ticks),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            ik_overrides=far_ik_overrides,
            extra_details={
                "selected_pregrasp_candidate": selected_candidate,
                "motion_policy": motion_policy,
                "point_A_proxy": geometry["point_A_proxy_definition"],
                "point_B_proxy": geometry["point_B_proxy_definition"],
                "target_point_B_world": geometry["pregrasp_point_B_world"],
                "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
                "far_descend_B_world": geometry.get("far_descend_B_world"),
                "contact_point_B_world": geometry.get("contact_point_B_world"),
                "contact_point_A_world": geometry.get("contact_AB_semantics", {}).get("point_A_world"),
                "far_point_b_forward_extension_m": geometry.get("far_point_b_forward_extension_m"),
                "far_point_a_extra_height_clearance_m": geometry.get("far_point_a_extra_height_clearance_m"),
                "far_point_a_min_extra_height_clearance_m": geometry.get("far_point_a_min_extra_height_clearance_m"),
                "far_ab_requested_downward_slant_deg": geometry.get("far_ab_requested_downward_slant_deg"),
                "far_ab_downward_slant_deg": geometry.get("far_ab_downward_slant_deg"),
                "far_ab_slant_height_clearance_m": geometry.get("far_ab_slant_height_clearance_m"),
                "far_ab_base_horizontal_span_m": geometry.get("far_ab_base_horizontal_span_m"),
                "far_point_b_gap_above_support_m": geometry.get("far_point_b_gap_above_support_m"),
                "far_xy_align_clearance_above_object_m": geometry.get("far_xy_align_clearance_above_object_m"),
                "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                "AB_axis_world": geometry.get("AB_axis_world"),
                "selected_orientation_preset_axial_roll_variant_label": selected_orientation_preset.get("preset_axial_roll_variant_label"),
                "selected_orientation_preset_axial_roll_about_ab_rad": selected_orientation_preset.get("preset_axial_roll_about_ab_rad"),
            },
        )
        if pregrasp_result["final_error"] > args.pregrasp_tolerance:
            _fail("pregrasp_failed", f"{pregrasp_phase_name} did not reach selected DualArmIK 6D target")

        if far_motion_policy:
            far_xy_align_result = _execute_dualarmik_servo_phase(
                ServoSpec("far_align_B_over_object_xy", np.array(geometry["align_pose_base"], dtype=float), args.align_tolerance, args.rot_tolerance, args.servo_max_ticks),
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                arm_side=chosen_arm,
                coord_transform=coord_transform,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                ik_overrides=far_ik_overrides,
                extra_details={
                    "motion_policy": motion_policy,
                    "far_contact_sequence_policy": geometry.get("far_contact_sequence_policy"),
                    "point_A_proxy": geometry["point_A_proxy_definition"],
                    "point_B_proxy": geometry["point_B_proxy_definition"],
                    "target_point_B_world": geometry["far_xy_align_B_world"],
                    "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                    "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
                    "far_descend_B_world": geometry.get("far_descend_B_world"),
                    "contact_point_B_world": geometry.get("contact_point_B_world"),
                    "contact_point_A_world": geometry.get("contact_AB_semantics", {}).get("point_A_world"),
                    "far_point_b_forward_extension_m": geometry.get("far_point_b_forward_extension_m"),
                    "far_point_a_extra_height_clearance_m": geometry.get("far_point_a_extra_height_clearance_m"),
                    "far_point_a_min_extra_height_clearance_m": geometry.get("far_point_a_min_extra_height_clearance_m"),
                    "far_ab_requested_downward_slant_deg": geometry.get("far_ab_requested_downward_slant_deg"),
                    "far_ab_downward_slant_deg": geometry.get("far_ab_downward_slant_deg"),
                    "far_ab_slant_height_clearance_m": geometry.get("far_ab_slant_height_clearance_m"),
                    "far_ab_base_horizontal_span_m": geometry.get("far_ab_base_horizontal_span_m"),
                    "far_point_b_gap_above_support_m": geometry.get("far_point_b_gap_above_support_m"),
                    "far_xy_align_clearance_above_object_m": geometry.get("far_xy_align_clearance_above_object_m"),
                    "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                    "AB_axis_world": geometry.get("AB_axis_world"),
                    "selected_orientation_preset_axial_roll_variant_label": selected_orientation_preset.get("preset_axial_roll_variant_label"),
                    "selected_orientation_preset_axial_roll_about_ab_rad": selected_orientation_preset.get("preset_axial_roll_about_ab_rad"),
                    "side_push_avoidance": "XY alignment happens at low-side prepare height before final world-Z lowering",
                },
            )
            if far_xy_align_result["final_error"] > args.align_tolerance:
                _fail("align_failed", "far_align_B_over_object_xy did not reach DualArmIK 6D target")
            payload["object_trace"]["after_far_xy_align"] = _bbox_state(stage, target_path)

            far_descend_locked_rpy = np.array(selected_orientation_preset["rpy"], dtype=float)
            selected_point_b_offset = np.array(geometry.get("point_b_offset_local", point_b_offset), dtype=float)
            far_descend_B_world = np.array(geometry["contact_point_B_world"], dtype=float)

            def far_world_z_lower_fn():
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_B_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
                next_B_world = far_descend_B_world.copy()
                next_B_world[2] = max(float(far_descend_B_world[2]), float(curr_B_world[2]) - 0.002)
                target_pose, _ = _pose_for_point_b_world(next_B_world, coord_transform, far_descend_locked_rpy, selected_point_b_offset)
                return target_pose

            descend_result = _execute_dualarmik_servo_phase(
                ServoSpec("far_lower_B_world_z", np.array(geometry["contact_pose_base"], dtype=float), args.descend_tolerance, args.rot_tolerance, args.servo_max_ticks * 2),
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                arm_side=chosen_arm,
                coord_transform=coord_transform,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                ik_overrides=far_ik_overrides,
                target_pose_fn=far_world_z_lower_fn,
                extra_details={
                    "motion_policy": motion_policy,
                    "far_contact_sequence_policy": geometry.get("far_contact_sequence_policy"),
                    "descend_locked_rpy_source": "selected_orientation_preset",
                    "selected_approach_family": selected_orientation_preset.get("preset_family"),
                    "selected_grasp_family": selected_orientation_preset.get("preset_family"),
                    "selected_orientation_preset_label": selected_orientation_preset["preset_label"],
                    "selected_orientation_preset_rpy": selected_orientation_preset["rpy"],
                    "selected_orientation_preset_approach_axis_world": selected_orientation_preset.get("approach_axis_world"),
                    "target_point_B_world": geometry["contact_point_B_world"],
                    "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                    "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
                    "far_descend_B_world": geometry.get("far_descend_B_world"),
                    "contact_point_B_world": geometry.get("contact_point_B_world"),
                    "contact_point_A_world": geometry.get("contact_AB_semantics", {}).get("point_A_world"),
                    "far_point_b_gap_above_support_m": geometry.get("far_point_b_gap_above_support_m"),
                    "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                    "AB_axis_world": geometry.get("AB_axis_world"),
                    "world_z_lowering": True,
                    "side_push_avoidance": "final contact moves only in world Z after XY alignment",
                },
            )
            payload["object_trace"]["after_far_world_z_lower"] = _bbox_state(stage, target_path)
            contact_gate_phase_index = -1
            if descend_result["final_error"] > args.descend_tolerance:
                _fail("descend_failed", "far_lower_B_world_z did not reach DualArmIK 6D target")

        else:
            final_contact_pose = np.array(geometry["contact_pose_base"], dtype=float)
            descend_locked_rpy = np.array(selected_orientation_preset["rpy"], dtype=float)
            vertical_descend_pos_tolerance = min(float(args.descend_tolerance), float(args.vertical_close_point_b_tolerance))
            selected_point_b_offset = np.array(geometry.get("point_b_offset_local", point_b_offset), dtype=float)
            vertical_ref_offset_raw = geometry.get("vertical_xy_reference_offset_local")
            vertical_ref_offset = None if vertical_ref_offset_raw is None else np.array(vertical_ref_offset_raw, dtype=float)
            vertical_ref_target_xy_raw = geometry.get("vertical_xy_reference_target_xy_world")
            vertical_ref_target_xy = None if vertical_ref_target_xy_raw is None else np.array(vertical_ref_target_xy_raw, dtype=float)
            vertical_ref_log = geometry.get("vertical_xy_reference_link_log")
            vertical_contact_b_world = np.array(geometry["contact_point_B_world"], dtype=float)
            vertical_xy_feedback_samples: list[dict[str, Any]] = []
            vertical_xy_feedback_call_count = 0
            vertical_descend_step_z = float(args.nogate_vertical_descend_step_z)
            vertical_descend_refresh_period = max(1, int(args.nogate_vertical_ik_refresh_period))
            vertical_continuous_ik_descend = bool(args.nogate_vertical_continuous_ik_descend)
            vertical_descend_servo_tolerance = (
                min(float(vertical_descend_pos_tolerance), max(0.0005, vertical_descend_step_z * 0.5))
                if vertical_continuous_ik_descend
                else float(vertical_descend_pos_tolerance)
            )
            vertical_descend_state: dict[str, Any] = {
                "last_commanded_z": None,
                "start_point_B_z_world": None,
                "final_contact_z_world": float(vertical_contact_b_world[2]),
                "ik_refresh_period_ticks": int(vertical_descend_refresh_period),
                "effective_z_step_per_frame_m": float(vertical_descend_step_z) / float(vertical_descend_refresh_period),
                "final_z_reached_by_command": False,
                "close_dls_handoff_enabled": bool(NO_GATE_MODE and args.nogate_close_dls_enable),
                "close_dls_switch_distance_m": float(args.nogate_close_dls_switch_distance),
                "close_dls_handoff_requested": False,
                "close_dls_handoff_reason": None,
                "close_dls_handoff_source": None,
                "close_dls_handoff_metric_name": None,
                "close_dls_handoff_metric_world": None,
                "close_dls_handoff_object_center_world": None,
                "close_dls_handoff_contact_mark_world": vertical_contact_b_world.tolist(),
                "close_dls_handoff_distance_to_object_center_m": None,
                "close_dls_handoff_distance_to_object_bbox_m": None,
                "close_dls_handoff_distance_to_contact_mark_m": None,
            }

            def update_close_dls_handoff_state(close_metric_world: np.ndarray, metric_name: str, source: str) -> bool:
                if not bool(vertical_descend_state["close_dls_handoff_enabled"]):
                    return False
                if bool(vertical_descend_state["close_dls_handoff_requested"]):
                    return True
                close_metric = np.array(close_metric_world, dtype=float)
                object_state = _bbox_state(stage, target_path)
                object_center = _center_from_bbox(object_state["bbox"])
                bbox_min = np.array(object_state["bbox"]["min"], dtype=float)
                bbox_max = np.array(object_state["bbox"]["max"], dtype=float)
                closest_bbox_point = np.minimum(np.maximum(close_metric, bbox_min), bbox_max)
                distance_to_object = float(np.linalg.norm(close_metric - object_center))
                distance_to_object_bbox = float(np.linalg.norm(close_metric - closest_bbox_point))
                distance_to_contact = float(np.linalg.norm(close_metric - vertical_contact_b_world))
                switch_distance = float(vertical_descend_state["close_dls_switch_distance_m"])
                trigger_reason = None
                if distance_to_object_bbox <= switch_distance:
                    trigger_reason = "gripper_metric_within_switch_distance_of_object_bbox"
                elif distance_to_contact <= switch_distance:
                    trigger_reason = "gripper_metric_within_switch_distance_of_contact_mark"
                if trigger_reason is not None:
                    vertical_descend_state["close_dls_handoff_requested"] = True
                    vertical_descend_state["close_dls_handoff_reason"] = trigger_reason
                    vertical_descend_state["close_dls_handoff_source"] = source
                    vertical_descend_state["close_dls_handoff_metric_name"] = metric_name
                    vertical_descend_state["close_dls_handoff_metric_world"] = close_metric.tolist()
                    vertical_descend_state["close_dls_handoff_object_center_world"] = object_center.tolist()
                    vertical_descend_state["close_dls_handoff_distance_to_object_center_m"] = distance_to_object
                    vertical_descend_state["close_dls_handoff_distance_to_object_bbox_m"] = distance_to_object_bbox
                    vertical_descend_state["close_dls_handoff_closest_object_bbox_point_world"] = closest_bbox_point.tolist()
                    vertical_descend_state["close_dls_handoff_distance_to_contact_mark_m"] = distance_to_contact
                    vertical_descend_state["close_dls_handoff_object_bbox"] = object_state
                    return True
                vertical_descend_state["latest_close_dls_distance_to_object_center_m"] = distance_to_object
                vertical_descend_state["latest_close_dls_distance_to_object_bbox_m"] = distance_to_object_bbox
                vertical_descend_state["latest_close_dls_closest_object_bbox_point_world"] = closest_bbox_point.tolist()
                vertical_descend_state["latest_close_dls_distance_to_contact_mark_m"] = distance_to_contact
                vertical_descend_state["latest_close_dls_metric_name"] = metric_name
                vertical_descend_state["latest_close_dls_metric_world"] = close_metric.tolist()
                vertical_descend_state["latest_close_dls_object_center_world"] = object_center.tolist()
                return False

            def vertical_close_dls_early_stop_condition() -> bool:
                if bool(vertical_descend_state["close_dls_handoff_requested"]):
                    return True
                if not bool(vertical_descend_state["close_dls_handoff_enabled"]):
                    return False
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_b_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
                real_center_world, real_center_log = _resolve_real_grasp_center_world(
                    stage=stage,
                    dc=dc,
                    articulation=articulation,
                    robot_root_path=chosen_robot_prim_path,
                    arm_side=chosen_arm,
                )
                if real_center_world is not None:
                    return update_close_dls_handoff_state(np.array(real_center_world, dtype=float), "real_grasp_center_world", "servo_tick_live_grasp_center")
                vertical_descend_state["latest_close_dls_real_grasp_center_fallback"] = real_center_log
                return update_close_dls_handoff_state(curr_b_world, "point_B_proxy_world_fallback", "servo_tick_point_B_fallback")

            def vertical_xy_locked_descend_target_fn():
                nonlocal vertical_xy_feedback_call_count
                vertical_xy_feedback_call_count += 1
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_b_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
                vertical_descend_state["target_pose_update_count"] = int(vertical_xy_feedback_call_count)
                curr_ref_world = None
                curr_ref_runtime_log: dict[str, Any] = {
                    "runtime_source": "vertical_point_B_fallback_no_xy_reference",
                    "runtime_fallback_used": True,
                }
                xy_error = np.zeros(2, dtype=float)
                if vertical_ref_offset is not None and vertical_ref_target_xy is not None:
                    curr_ref_world, curr_ref_runtime_log = _resolve_current_vertical_xy_reference_world(
                        stage=stage,
                        dc=dc,
                        articulation=articulation,
                        robot_root_path=chosen_robot_prim_path,
                        arm_side=chosen_arm,
                        reference_log=vertical_ref_log if isinstance(vertical_ref_log, dict) else None,
                        coord_transform=coord_transform,
                        current_pose_base=curr_pose,
                        reference_offset_local=vertical_ref_offset,
                    )
                    xy_error = vertical_ref_target_xy[:2] - curr_ref_world[:2]
                close_dls_metric_world = curr_ref_world if curr_ref_world is not None else curr_b_world
                close_dls_metric_name = (
                    str(curr_ref_runtime_log.get("reference_mode") or geometry.get("vertical_xy_reference_mode") or "vertical_xy_reference_world")
                    if curr_ref_world is not None
                    else "point_B_proxy_world_fallback"
                )
                update_close_dls_handoff_state(close_dls_metric_world, close_dls_metric_name, "vertical_descend_target_refresh")
                target_b_world = vertical_contact_b_world.copy()
                target_b_world[:2] = vertical_contact_b_world[:2] + xy_error
                final_z = float(vertical_contact_b_world[2])
                if vertical_continuous_ik_descend:
                    if vertical_descend_state["start_point_B_z_world"] is None:
                        vertical_descend_state["start_point_B_z_world"] = float(curr_b_world[2])
                    previous_z = vertical_descend_state["last_commanded_z"]
                    if previous_z is None:
                        next_z = float(curr_b_world[2]) - vertical_descend_step_z
                    else:
                        next_z = float(previous_z) - vertical_descend_step_z
                    commanded_z = max(final_z, next_z)
                    vertical_descend_state["last_commanded_z"] = float(commanded_z)
                    vertical_descend_state["final_z_reached_by_command"] = bool(commanded_z <= final_z + 1.0e-9)
                    target_b_world[2] = float(commanded_z)
                else:
                    target_b_world[2] = final_z
                    vertical_descend_state["last_commanded_z"] = final_z
                    vertical_descend_state["final_z_reached_by_command"] = True
                if vertical_xy_feedback_call_count == 1 or vertical_xy_feedback_call_count % int(args.trace_interval) == 0:
                    sample = {
                        "call_index": vertical_xy_feedback_call_count,
                        "current_point_B_world": curr_b_world.tolist(),
                        "current_vertical_xy_reference_world": None if curr_ref_world is None else curr_ref_world.tolist(),
                        "vertical_xy_reference_target_xy_world": None if vertical_ref_target_xy is None else vertical_ref_target_xy[:2].tolist(),
                        "vertical_xy_reference_error_xy_m": xy_error.tolist(),
                        "vertical_xy_reference_error_norm_m": float(np.linalg.norm(xy_error)),
                        "vertical_xy_reference_feedback_rule": "nominal_contact_B_xy_plus_live_reference_error_with_world_Z_ramp",
                        "current_vertical_xy_reference_runtime_source": curr_ref_runtime_log.get("runtime_source"),
                        "current_vertical_xy_reference_runtime_fallback_used": curr_ref_runtime_log.get("runtime_fallback_used"),
                        "current_vertical_xy_reference_mode": curr_ref_runtime_log.get("reference_mode", geometry.get("vertical_xy_reference_mode")),
                        "current_finger_midpoint_component_positions_world": curr_ref_runtime_log.get("component_positions_world"),
                        "nominal_contact_point_B_world": vertical_contact_b_world.tolist(),
                        "commanded_target_point_B_world": target_b_world.tolist(),
                        "commanded_target_z_world": float(target_b_world[2]),
                        "final_contact_z_world": final_z,
                        "z_remaining_to_contact_m": float(max(0.0, target_b_world[2] - final_z)),
                        "z_step_per_ik_refresh_m": vertical_descend_step_z,
                        "ik_refresh_period_ticks": int(vertical_descend_refresh_period),
                        "effective_z_step_per_frame_m": float(vertical_descend_step_z) / float(vertical_descend_refresh_period),
                        "world_z_only_descent_target": True,
                        "continuous_ik_refresh_expected": vertical_continuous_ik_descend,
                        "final_z_reached_by_command": bool(vertical_descend_state["final_z_reached_by_command"]),
                        "close_dls_handoff_enabled": bool(vertical_descend_state["close_dls_handoff_enabled"]),
                        "close_dls_handoff_requested": bool(vertical_descend_state["close_dls_handoff_requested"]),
                        "close_dls_handoff_reason": vertical_descend_state.get("close_dls_handoff_reason"),
                        "close_dls_switch_distance_m": float(vertical_descend_state["close_dls_switch_distance_m"]),
                        "close_dls_distance_to_object_center_m": vertical_descend_state.get("latest_close_dls_distance_to_object_center_m"),
                        "close_dls_distance_to_object_bbox_m": vertical_descend_state.get("latest_close_dls_distance_to_object_bbox_m"),
                        "close_dls_distance_to_contact_mark_m": vertical_descend_state.get("latest_close_dls_distance_to_contact_mark_m"),
                    }
                    vertical_xy_feedback_samples.append(sample)
                target_pose, _ = _pose_for_point_b_world(target_b_world, coord_transform, descend_locked_rpy, selected_point_b_offset)
                return target_pose

            descend_result = _execute_dualarmik_servo_phase(
                ServoSpec(descend_phase_name, final_contact_pose, vertical_descend_servo_tolerance, args.rot_tolerance, args.servo_max_ticks * 2),
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                arm_dofs=arm_dofs,
                arm_side=chosen_arm,
                coord_transform=coord_transform,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                target_pose_fn=vertical_xy_locked_descend_target_fn,
                ik_refresh_enable_override=True if vertical_continuous_ik_descend else None,
                ik_refresh_period_override=vertical_descend_refresh_period if vertical_continuous_ik_descend else None,
                completion_condition_fn=(
                    (lambda: bool(vertical_descend_state["final_z_reached_by_command"]))
                    if vertical_continuous_ik_descend
                    else None
                ),
                early_stop_condition_fn=vertical_close_dls_early_stop_condition if bool(vertical_descend_state["close_dls_handoff_enabled"]) else None,
                extra_details={
                    "motion_policy": motion_policy,
                    "descend_locked_rpy_source": "selected_orientation_preset",
                    "selected_approach_family": selected_orientation_preset.get("preset_family"),
                    "selected_grasp_family": selected_orientation_preset.get("preset_family"),
                    "selected_orientation_preset_label": selected_orientation_preset["preset_label"],
                    "selected_orientation_preset_rpy": selected_orientation_preset["rpy"],
                    "selected_orientation_preset_approach_axis_world": selected_orientation_preset.get("approach_axis_world"),
                    "selected_orientation_preset_dot_with_world_pos_y": selected_orientation_preset.get("dot_with_world_pos_y"),
                    "selected_orientation_preset_dot_with_world_neg_y": selected_orientation_preset.get("dot_with_world_neg_y"),
                    "target_point_B_world": geometry["contact_point_B_world"],
                    "vertical_contact_sequence_policy": geometry.get("vertical_contact_sequence_policy"),
                    "vertical_contact_mark_B_world": geometry.get("vertical_contact_mark_B_world"),
                    "vertical_raw_point_B_contact_mark_before_xy_reference": geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference"),
                    "vertical_xy_reference_active": geometry.get("vertical_xy_reference_active"),
                    "vertical_xy_reference_mode": geometry.get("vertical_xy_reference_mode"),
                    "vertical_xy_reference_source": geometry.get("vertical_xy_reference_source"),
                    "vertical_xy_reference_world_position_used_for_offset": geometry.get("vertical_xy_reference_world_position_used_for_offset"),
                    "vertical_xy_reference_component_logs": geometry.get("vertical_xy_reference_component_logs"),
                    "vertical_xy_reference_target_xy_world": geometry.get("vertical_xy_reference_target_xy_world"),
                    "vertical_uncorrected_object_world_xy_target": geometry.get("vertical_uncorrected_object_world_xy_target"),
                    "vertical_arm_lateral_bias_correction_m": geometry.get("vertical_arm_lateral_bias_correction_m"),
                    "vertical_arm_lateral_bias_correction_base_y_m": geometry.get("vertical_arm_lateral_bias_correction_base_y_m"),
                    "vertical_arm_lateral_bias_correction_world": geometry.get("vertical_arm_lateral_bias_correction_world"),
                    "vertical_arm_lateral_bias_correction_rule": geometry.get("vertical_arm_lateral_bias_correction_rule"),
                    "vertical_xy_reference_offset_local": geometry.get("vertical_xy_reference_offset_local"),
                    "vertical_xy_reference_tolerance_m": geometry.get("vertical_xy_reference_tolerance_m"),
                    "vertical_point_b_gap_above_support_m": geometry.get("vertical_point_b_gap_above_support_m"),
                    "vertical_close_point_b_tolerance_m": geometry.get("vertical_close_point_b_tolerance_m"),
                    "vertical_descend_position_tolerance_m": vertical_descend_pos_tolerance,
                    "vertical_descend_servo_tolerance_m": vertical_descend_servo_tolerance,
                    "nogate_vertical_continuous_ik_descend": vertical_continuous_ik_descend,
                    "nogate_vertical_descend_step_z_m": vertical_descend_step_z,
                    "nogate_vertical_ik_refresh_period_ticks": vertical_descend_refresh_period if vertical_continuous_ik_descend else int(args.ik_refresh_period),
                    "nogate_vertical_effective_z_step_per_frame_m": float(vertical_descend_step_z) / float(vertical_descend_refresh_period),
                    "vertical_descend_state": vertical_descend_state,
                    "vertical_descend_target_policy": "no_gate_continuous_ik_world_z_ramp_with_live_xy_reference_lock",
                    "vertical_xy_reference_feedback_active": bool(vertical_ref_offset is not None and vertical_ref_target_xy is not None),
                    "vertical_xy_reference_feedback_rule": "nominal_contact_B_xy_plus_live_reference_error_with_world_Z_ramp",
                    "vertical_xy_reference_feedback_samples": vertical_xy_feedback_samples,
                    "close_after_point_B_contact_gate": True,
                    "strict_AB_vertical_during_descend": True,
                },
            )
            after_descend = _bbox_state(stage, target_path)
            payload["object_trace"]["after_descend"] = after_descend
            contact_gate_phase_index = -1
            if descend_result["final_error"] > vertical_descend_pos_tolerance:
                _fail("descend_failed", f"{descend_phase_name} did not bring point B to the vertical contact mark")

            close_dls_skip_reason = None if bool(vertical_descend_state.get("close_dls_handoff_requested")) else "switch_distance_not_reached_before_vertical_descend_finished"
            nogate_close_dls_result = _execute_nogate_close_dls_phase(
                stage=stage,
                target_path=target_path,
                geometry=geometry,
                coord_transform=coord_transform,
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                robot_root_path=chosen_robot_prim_path,
                arm_side=chosen_arm,
                arm_dofs=arm_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                target_metric_world=vertical_contact_b_world,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                switch_log=dict(vertical_descend_state),
                skip_reason=close_dls_skip_reason,
            )
            descend_result["nogate_near_contact_measured_dls_finish"] = nogate_close_dls_result
            payload["object_trace"]["after_nogate_near_contact_dls"] = _bbox_state(stage, target_path)

        pre_close_gate = _pre_close_gate(
            stage=stage,
            target_path=target_path,
            geometry=geometry,
            coord_transform=coord_transform,
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            robot_root_path=chosen_robot_prim_path,
            arm_side=chosen_arm,
            tcp_offset_local=tcp_offset,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            args=args,
        )
        if far_motion_policy:
            far_point_b_error = float(pre_close_gate.get("point_B_error_before_close_m", float("inf")))
            far_close_critical_error = float(pre_close_gate.get("close_critical_error_before_close_m", far_point_b_error))
            far_close_tolerance = float(args.pre_close_point_b_tolerance)
            far_close_gate = {
                "gate_name": "far_real_grasp_center_reached_contact_mark_before_close",
                "condition_met": bool(far_close_critical_error <= far_close_tolerance),
                "close_critical_metric": pre_close_gate.get("close_critical_metric"),
                "close_critical_error_before_close_m": far_close_critical_error,
                "close_critical_tolerance_m": far_close_tolerance,
                "close_critical_eval_world": pre_close_gate.get("close_critical_eval_world"),
                "close_critical_target_world": pre_close_gate.get("close_critical_target_world"),
                "close_critical_uses_real_grasp_center": pre_close_gate.get("close_critical_uses_real_grasp_center"),
                "point_B_error_before_close_m": far_point_b_error,
                "point_B_tolerance_m": far_close_tolerance,
                "point_B_proxy_world": pre_close_gate.get("point_B_proxy_world"),
                "real_grasp_center_world": pre_close_gate.get("real_grasp_center_world"),
                "real_grasp_center_error_before_close_m": pre_close_gate.get("real_grasp_center_error_before_close_m"),
                "proxy_to_real_grasp_center_delta_world": pre_close_gate.get("proxy_to_real_grasp_center_delta_world"),
                "proxy_to_real_grasp_center_delta_world_norm_m": pre_close_gate.get("proxy_to_real_grasp_center_delta_world_norm_m"),
                "proxy_to_real_grasp_center_delta_local": pre_close_gate.get("proxy_to_real_grasp_center_delta_local"),
                "target_point_B_world_before_close": pre_close_gate.get("target_point_B_world_before_close"),
                "actual_point_B_world_before_close": pre_close_gate.get("actual_point_B_world_before_close"),
                "far_contact_sequence_policy": geometry.get("far_contact_sequence_policy"),
                "far_outboard_transition_B_world": geometry.get("far_outboard_transition_B_world"),
                "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
                "far_descend_B_world": geometry.get("far_descend_B_world"),
                "mandatory_pre_close_gate": True,
            }
            pre_close_gate["far_close_point_B_gate"] = far_close_gate
        if not far_motion_policy:
            vertical_point_b_error = float(pre_close_gate.get("point_B_error_before_close_m", float("inf")))
            vertical_close_critical_error = float(pre_close_gate.get("close_critical_error_before_close_m", vertical_point_b_error))
            vertical_close_tolerance = float(args.vertical_close_point_b_tolerance)
            vertical_xy_reference_error = pre_close_gate.get("vertical_xy_reference_error_before_close_m")
            vertical_xy_reference_active = bool(pre_close_gate.get("vertical_xy_reference_active"))
            vertical_xy_condition = (
                True
                if not vertical_xy_reference_active or vertical_xy_reference_error is None
                else bool(float(vertical_xy_reference_error) <= float(args.vertical_xy_reference_tolerance))
            )
            vertical_close_gate = {
                "gate_name": "vertical_real_grasp_center_and_xy_reference_reached_contact_mark_before_close",
                "condition_met": bool(vertical_close_critical_error <= vertical_close_tolerance and vertical_xy_condition),
                "close_critical_condition_met": bool(vertical_close_critical_error <= vertical_close_tolerance),
                "point_B_condition_met": bool(vertical_point_b_error <= vertical_close_tolerance),
                "vertical_xy_reference_condition_met": bool(vertical_xy_condition),
                "close_critical_metric": pre_close_gate.get("close_critical_metric"),
                "close_critical_error_before_close_m": vertical_close_critical_error,
                "close_critical_tolerance_m": vertical_close_tolerance,
                "close_critical_eval_world": pre_close_gate.get("close_critical_eval_world"),
                "close_critical_target_world": pre_close_gate.get("close_critical_target_world"),
                "close_critical_uses_real_grasp_center": pre_close_gate.get("close_critical_uses_real_grasp_center"),
                "point_B_error_before_close_m": vertical_point_b_error,
                "point_B_tolerance_m": vertical_close_tolerance,
                "point_B_proxy_world": pre_close_gate.get("point_B_proxy_world"),
                "real_grasp_center_world": pre_close_gate.get("real_grasp_center_world"),
                "real_grasp_center_error_before_close_m": pre_close_gate.get("real_grasp_center_error_before_close_m"),
                "proxy_to_real_grasp_center_delta_world": pre_close_gate.get("proxy_to_real_grasp_center_delta_world"),
                "proxy_to_real_grasp_center_delta_world_norm_m": pre_close_gate.get("proxy_to_real_grasp_center_delta_world_norm_m"),
                "proxy_to_real_grasp_center_delta_local": pre_close_gate.get("proxy_to_real_grasp_center_delta_local"),
                "vertical_xy_reference_error_before_close_m": vertical_xy_reference_error,
                "vertical_xy_reference_tolerance_m": float(args.vertical_xy_reference_tolerance),
                "target_point_B_world_before_close": pre_close_gate.get("target_point_B_world_before_close"),
                "actual_point_B_world_before_close": pre_close_gate.get("actual_point_B_world_before_close"),
                "vertical_contact_mark_B_world": geometry.get("vertical_contact_mark_B_world"),
                "vertical_raw_point_B_contact_mark_before_xy_reference": geometry.get("vertical_raw_point_B_contact_mark_before_xy_reference"),
                "vertical_xy_reference_active": vertical_xy_reference_active,
                "vertical_xy_reference_mode": geometry.get("vertical_xy_reference_mode"),
                "vertical_xy_reference_source": geometry.get("vertical_xy_reference_source"),
                "vertical_xy_reference_target_xy_world": geometry.get("vertical_xy_reference_target_xy_world"),
                "vertical_uncorrected_object_world_xy_target": geometry.get("vertical_uncorrected_object_world_xy_target"),
                "vertical_arm_lateral_bias_correction_m": geometry.get("vertical_arm_lateral_bias_correction_m"),
                "vertical_arm_lateral_bias_correction_base_y_m": geometry.get("vertical_arm_lateral_bias_correction_base_y_m"),
                "vertical_arm_lateral_bias_correction_world": geometry.get("vertical_arm_lateral_bias_correction_world"),
                "actual_vertical_xy_reference_world_before_close": pre_close_gate.get("actual_vertical_xy_reference_world_before_close"),
                "vertical_point_b_gap_above_support_m": geometry.get("vertical_point_b_gap_above_support_m"),
                "actual_point_B_gap_above_support_m": pre_close_gate.get("vertical_actual_point_B_gap_above_support_m"),
            }
            pre_close_gate["vertical_close_point_B_gate"] = vertical_close_gate
        phase_log[contact_gate_phase_index]["details"]["pre_close_gate"] = pre_close_gate
        if far_motion_policy and not pre_close_gate["far_close_point_B_gate"]["condition_met"]:
            _fail(
                "far_contact_not_reached_before_close",
                "FAR grasp close skipped because the close-critical grasp center did not reach the contact mark within the mandatory pre-close gate",
            )
        if not far_motion_policy and not pre_close_gate["vertical_close_point_B_gate"]["condition_met"]:
            _fail(
                "vertical_contact_not_reached_before_close",
                "Vertical grasp close skipped because the close-critical grasp center and/or vertical XY reference did not reach the contact mark",
            )

        nogate_touch_probe_result = None
        close_blocked_by_nogate_touch_probe = False
        if NO_GATE_MODE and bool(args.nogate_preclose_touch_enable):
            pre_touch_pre_close_gate = pre_close_gate
            touch_probe_result = _execute_nogate_preclose_touch_probe(
                stage=stage,
                target_path=target_path,
                geometry=geometry,
                coord_transform=coord_transform,
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                robot_root_path=chosen_robot_prim_path,
                arm_side=chosen_arm,
                arm_dofs=arm_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                table_top_z=table_top_z,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                ik_overrides=far_ik_overrides,
            )
            post_touch_pre_close_gate = _pre_close_gate(
                stage=stage,
                target_path=target_path,
                geometry=geometry,
                coord_transform=coord_transform,
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                robot_root_path=chosen_robot_prim_path,
                arm_side=chosen_arm,
                tcp_offset_local=tcp_offset,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                args=args,
            )
            post_touch_pre_close_gate["before_nogate_preclose_touch_probe"] = pre_touch_pre_close_gate
            post_touch_pre_close_gate["nogate_preclose_touch_probe"] = touch_probe_result
            pre_close_gate = post_touch_pre_close_gate
            nogate_touch_probe_result = touch_probe_result
            close_blocked_by_nogate_touch_probe = not bool(touch_probe_result.get("safe_to_close", False))
            if close_blocked_by_nogate_touch_probe:
                _fail(
                    "nogate_touch_probe_not_safe_to_close",
                    f"No-gate close skipped because mesh touch/reposition probe was not safe to close: {touch_probe_result.get('close_block_reason')}",
                )
            payload["object_trace"]["after_nogate_preclose_touch_probe"] = _bbox_state(stage, target_path)

        close_skipped = bool(args.skip_gripper_close or close_blocked_by_nogate_touch_probe)
        close_ok = _command_gripper_phase(
            "close_gripper",
            dc=dc,
            gripper_dofs=gripper_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            target_positions=[OFFICIAL_GRIPPER_CLOSE_WIDTH] * len(gripper_dofs),
            sim_app=sim_app,
            steps=args.gripper_steps,
            counter=counter,
            phase_log=phase_log,
            skipped=close_skipped,
            effort_value=None if close_blocked_by_nogate_touch_probe else args.gripper_hold_effort,
            extra_details={
                "pre_close_gate": pre_close_gate,
                "nogate_touch_probe_result": nogate_touch_probe_result,
                "nogate_close_blocked_by_touch_probe": bool(close_blocked_by_nogate_touch_probe),
                "nogate_close_skip_reason": None if not close_blocked_by_nogate_touch_probe else "selected object was not confirmed between finger1_link and finger2_link after table/object touch and IK reposition",
            },
        )
        payload["object_trace"]["after_close"] = _bbox_state(stage, target_path)
        if not close_ok:
            _fail("close_gripper_failed", f"{chosen_arm} gripper close command failed")

        slow_lift_result = None
        if NO_GATE_MODE and bool(args.nogate_post_close_slow_lift_enable):
            slow_lift_result = _execute_nogate_post_close_slow_lift_hold(
                stage=stage,
                target_path=target_path,
                geometry=geometry,
                coord_transform=coord_transform,
                ik_solver=ik_solver,
                dc=dc,
                articulation=articulation,
                arm_side=chosen_arm,
                arm_dofs=arm_dofs,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                end_effector_name=end_effector_name,
                end_effector_path=end_effector_path,
                end_effector_policy=end_effector_policy,
                coord_transform_refresh_fn=coord_transform_refresh_fn,
                ik_overrides=far_ik_overrides,
                skipped=close_skipped,
                skip_reason=None if not close_skipped else "close_gripper_was_skipped_or_blocked_by_touch_probe",
            )
            payload["object_trace"]["after_nogate_post_close_slow_lift"] = _bbox_state(stage, target_path)

        probe_before = _bbox_state(stage, target_path)
        probe_before_center = _center_from_bbox(probe_before["bbox"])
        micro_lift_result = _execute_dualarmik_servo_phase(
            ServoSpec("micro_lift_probe", np.array(geometry["micro_lift_pose_base"], dtype=float), args.lift_tolerance, args.rot_tolerance, args.servo_max_ticks, gripper_effort=args.gripper_hold_effort),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            extra_details={"object_pose_before_micro_lift_probe": probe_before},
        )
        probe_after = _bbox_state(stage, target_path)
        probe_after_center = _center_from_bbox(probe_after["bbox"])
        probe_delta = probe_after_center - probe_before_center
        probe_success = bool(probe_delta[2] > float(args.micro_lift_min_delta))
        payload["object_trace"]["after_micro_lift_probe"] = probe_after
        phase_log[-1]["condition_met"] = probe_success
        phase_log[-1]["details"].update(
            {
                "object_pose_after_micro_lift_probe": probe_after,
                "object_delta_during_micro_lift_probe_m": probe_delta.tolist(),
                "micro_lift_probe_success": probe_success,
                "micro_lift_probe_min_delta_m": float(args.micro_lift_min_delta),
                "servo_result_condition_met": micro_lift_result["final_error"] <= args.lift_tolerance,
            }
        )
        if not probe_success:
            _fail("object_not_lifted", "Object did not move upward during the 15-20 mm micro-lift probe")

        lift_before = _bbox_state(stage, target_path)
        lift_before_center = _center_from_bbox(lift_before["bbox"])
        lift_phase_name = "far_lift" if far_motion_policy else f"{vertical_prefix}_lift"
        lift_result = _execute_dualarmik_servo_phase(
            ServoSpec(lift_phase_name, np.array(geometry["lift_pose_base"], dtype=float), args.lift_tolerance, args.rot_tolerance, args.servo_max_ticks, gripper_effort=args.gripper_hold_effort),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            extra_details={
                "object_pose_before_lift": lift_before,
                "motion_policy": motion_policy,
                "target_point_B_world": geometry["lift_point_B_world"],
            },
        )
        lift_after = _bbox_state(stage, target_path)
        lift_after_center = _center_from_bbox(lift_after["bbox"])
        lift_delta = lift_after_center - lift_before_center
        object_lifted = bool(lift_after_center[2] >= initial_center[2] + args.micro_lift_min_delta and lift_result["final_error"] <= args.lift_tolerance)
        payload["object_trace"]["after_lift"] = lift_after
        payload["result_flags"]["object_lifted"] = object_lifted
        phase_log[-1]["condition_met"] = object_lifted
        phase_log[-1]["details"].update(
            {
                "object_pose_after_lift": lift_after,
                "object_delta_during_lift_m": lift_delta.tolist(),
                "object_lifted": object_lifted,
            }
        )
        if not object_lifted:
            _fail("dropped_during_lift", "Object was not retained after full lift")

        if args.stop_after_lift:
            stop_start = counter["step"]
            _append_phase(
                phase_log,
                phase="motion_policy_stop_after_lift",
                start_step=stop_start,
                end_step=counter["step"],
                condition_met=True,
                details={
                    "target_position": None,
                    "failure_reason": None,
                    "motion_policy": motion_policy,
                    "stop_after_lift": True,
                    "object_pose_after_lift": lift_after,
                    "object_lifted": object_lifted,
                    "carry_place_skipped": True,
                    "reason": "AB motion policy validation stops after grasp and lift",
                },
            )
            payload["motion_policy_completion"] = {
                "stop_after_lift": True,
                "carry_place_skipped": True,
                "motion_policy": motion_policy,
            }
            payload["final_status"] = "pass"
            payload["failure_reason"] = None
            print(f"status=pass no_gate_mode={NO_GATE_MODE} object_lifted={str(object_lifted).lower()} stop_after_lift=true carry_place_skipped=true")
            if (args.no_headless or args.gui) and args.hold_open:
                _hold_gui_open(sim_app)
            return 0

        carry_result = _execute_dualarmik_servo_phase(
            ServoSpec("servo_carry", np.array(geometry["carry_pose_base"], dtype=float), args.carry_tolerance, args.rot_tolerance, args.servo_carry_ticks, gripper_effort=args.gripper_hold_effort),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
        )
        after_carry = _bbox_state(stage, target_path)
        payload["object_trace"]["after_carry"] = after_carry
        carry_center = _center_from_bbox(after_carry["bbox"])
        distance_to_bin_initial = _distance(initial_center, np.array(bin_bbox["center"], dtype=float))
        distance_to_bin_after = _distance(carry_center, np.array(bin_bbox["center"], dtype=float))
        object_transported = bool(
            _distance(carry_center, initial_center) >= args.min_transport_distance
            and distance_to_bin_after < distance_to_bin_initial
            and carry_result["final_error"] <= args.carry_tolerance
        )
        payload["result_flags"]["object_transported"] = object_transported
        phase_log[-1]["condition_met"] = object_transported
        phase_log[-1]["details"].update(
            {
                "object_pose_after_carry": after_carry,
                "object_distance_to_bin_initial": distance_to_bin_initial,
                "object_distance_to_bin_after_carry": distance_to_bin_after,
                "object_transported": object_transported,
            }
        )
        if carry_result["final_error"] > args.carry_tolerance:
            _fail("carry_failed", "Arm/EE failed to reach carry target")
        if not object_transported:
            _fail("dropped_during_transport", "Object did not move the required minimum distance toward the destination bin")

        place_result = _execute_dualarmik_servo_phase(
            ServoSpec("servo_place", np.array(geometry["place_pose_base"], dtype=float), args.place_tolerance, args.rot_tolerance, args.servo_carry_ticks, gripper_effort=args.gripper_hold_effort),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
        )
        if place_result["final_error"] > args.place_tolerance:
            _fail("place_failed", "Arm/EE failed to reach place target")

        release_ok = _command_gripper_phase(
            "open_gripper",
            dc=dc,
            gripper_dofs=gripper_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            target_positions=[OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs),
            sim_app=sim_app,
            steps=args.gripper_steps,
            counter=counter,
            phase_log=phase_log,
            skipped=args.skip_release,
            effort_value=0.0,
        )
        payload["object_trace"]["after_release"] = _bbox_state(stage, target_path)
        if not release_ok:
            _fail("release_failed", f"{chosen_arm} gripper release command failed near place target")

        retreat_result = _execute_dualarmik_servo_phase(
            ServoSpec("servo_retreat", np.array(geometry["retreat_pose_base"], dtype=float), args.retreat_tolerance, args.rot_tolerance, args.servo_max_ticks),
            ik_solver=ik_solver,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            arm_side=chosen_arm,
            coord_transform=coord_transform,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
        )
        if retreat_result["final_error"] > args.retreat_tolerance:
            _fail("retreat_failed", "Retreat phase failed to reach DualArmIK 6D target")

        settle_start = counter["step"]
        final_state, final_jitter = _settle_and_measure(stage, target_path, sim_app, args.settle_steps, counter)
        final_center = _center_from_bbox(final_state["bbox"])
        final_inside_bin = _inside_bin(final_center, bin_bbox, float(bin_collider["wall_thickness"]), float(bin_collider["floor_top_z"]))
        object_stable = bool(final_jitter <= args.stable_jitter)
        payload["object_trace"]["final_after_settle"] = final_state
        payload["result_flags"]["final_inside_bin"] = final_inside_bin
        payload["result_flags"]["object_stable"] = object_stable
        _append_phase(
            phase_log,
            phase="settle_and_score",
            start_step=settle_start,
            end_step=counter["step"],
            condition_met=bool(final_inside_bin and object_stable),
            details={
                "target_position": None,
                "final_ee_position": _body_pose_position(dc, end_effector_body).tolist(),
                "iteration_count": args.settle_steps,
                "failure_reason": None if final_inside_bin and object_stable else "object_outside_bin_or_unstable",
                "chosen_ee_frame_name": end_effector_name,
                "chosen_ee_frame_path": end_effector_path,
                "chosen_ee_frame_policy": end_effector_policy,
                "final_target_pose": final_state,
                "final_jitter_m": final_jitter,
                "stable_jitter_threshold_m": args.stable_jitter,
                "final_inside_bin": final_inside_bin,
                "object_stable": object_stable,
            },
        )
        if not final_inside_bin:
            _fail("object_outside_bin", "Target object final pose is outside the diagnostic bin volume")
        if not object_stable:
            _fail("object_unstable_after_settle", "Target object did not settle stably after release")

        payload["final_status"] = "pass"
        payload["failure_reason"] = None
        print(
            f"status=pass no_gate_mode={NO_GATE_MODE} "
            f"object_lifted={str(object_lifted).lower()} "
            f"object_transported={str(object_transported).lower()} "
            f"final_inside_bin={str(final_inside_bin).lower()} "
            f"object_stable={str(object_stable).lower()}"
        )

        if (args.no_headless or args.gui) and args.hold_open:
            _hold_gui_open(sim_app)

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
        try:
            paths = _validate_environment()
            log_paths = _write_logs(Path(paths["LOG_ROOT"]).resolve(), payload, args.log_suffix)
            for path in log_paths:
                print(f"log={path}")
        except Exception as log_exc:
            print(f"warning=failed_to_write_log error={log_exc}", file=sys.stderr)
        if timeline is not None:
            try:
                timeline.stop()
            except Exception:
                pass
        if sim_app is not None:
            try:
                sim_app.close()
            except Exception:
                pass
    return 0 if payload.get("final_status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
