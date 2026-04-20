#!/usr/bin/env python3
"""Task 1 Phase 2 hybrid geometric contact-hardening for Walker S2.

This script is copied from scripts/task1_hybrid_geometric_phase1.py and keeps
the validated table-frame, scene_state object_info, finite candidate generation,
DualArmIK backend, and phase-machine execution. Phase 2 adds cheap geometric
grasp filtering plus local contact/descent hardening; it does not add Thinker,
YOLO, or a new planner backend.
"""

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


SCRIPT_NAME = "task1_hybrid_geometric_phase2.py"
LOG_STEM = "task1_hybrid_geometric_phase2"
SOURCE_BASELINE_SCRIPT = "scripts/task1_hybrid_geometric_phase1.py"

TABLE_UNIT_M = 0.035
HYBRID_PHASE1_PRESET_YAW_DEG = (-30.0, -10.0, 10.0, 30.0)
HYBRID_PHASE1_SCORE_REACH_WEIGHT = 1.0
HYBRID_PHASE1_SCORE_SIDE_PENALTY = 0.15
HYBRID_PHASE1_SCORE_YAW_WEIGHT = 0.01
HYBRID_PHASE1_SCORE_WIDTH_PENALTY = 10.0
HYBRID_PHASE1_BASIC_WIDTH_MIN_M = 0.010
HYBRID_PHASE1_BASIC_WIDTH_MAX_M = 0.090
HYBRID_PHASE1_BASIC_SYMMETRY_TOL_M = 0.040

PHASE2_ALIGNMENT_ERROR_MAX_RAD = 0.75
PHASE2_SYMMETRY_ERROR_MAX_M = 0.030
PHASE2_CONTACT_ASYMMETRY_MAX_M = 0.030
PHASE2_TABLE_CLEARANCE_MIN_M = -0.010
PHASE2_HORIZONTAL_TABLE_CLEARANCE_MIN_M = -0.005
PHASE2_WIDTH_MIN_M = HYBRID_PHASE1_BASIC_WIDTH_MIN_M
PHASE2_WIDTH_MAX_M = HYBRID_PHASE1_BASIC_WIDTH_MAX_M
PHASE2_GEOMETRIC_VIOLATION_WEIGHT = 100.0
PHASE2_ALLOW_LEAST_BAD_CANDIDATE = False
PHASE2_CLOSE_REAL_CENTER_TOLERANCE_M = 0.030
PHASE2_CLOSE_ORIENTATION_TOLERANCE_RAD = 0.35
PHASE2_CATASTROPHIC_ORIENTATION_ERROR_MAX_RAD = 0.85
PHASE2_SOFT_ORIENTATION_WARNING_MAX_RAD = PHASE2_CLOSE_ORIENTATION_TOLERANCE_RAD
PHASE2_CLOSE_XY_DRIFT_MAX_M = 0.018
PHASE2_RUNTIME_COMMIT_FALLBACK_TIP_MID_ERROR_MAX_M = 0.08
PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_Z_PROGRESS_MAX_M = 0.010
PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_XY_DRIFT_MAX_M = PHASE2_CLOSE_XY_DRIFT_MAX_M
PHASE2_RUNTIME_COMMIT_FALLBACK_MIN_SAMPLES = 3
PHASE2_CATASTROPHIC_TABLE_CLEARANCE_MIN_M = -0.010
PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MAX_M = 0.004
PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MIN_M = -0.020
PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M = 0.010
PHASE2_VERTICAL_FALLBACK_MIN_DESCENT_SAMPLES = 2
PHASE2_VERTICAL_FALLBACK_ORIENTATION_TOLERANCE_RAD = 0.65
PHASE2_VERTICAL_FALLBACK_ALLOW_FAR_POLICY = True
PHASE2_VERTICAL_TIP_TABLE_Z_CLOSE_THRESHOLD_M = 0.0005
PHASE2_DESCENT_XY_STEP_M = 0.01
PHASE2_DESCENT_Z_STEP_M = 0.003
PHASE2_DESCENT_YAW_STEP_RAD = 0.04
PHASE2_DESCENT_MAX_TICKS = 120
PHASE2_HORIZONTAL_DESCENT_XY_TRIGGER_M = 0.06
PHASE2_MIN_TARGET_GAP_M = 0.004
PHASE2_TARGET_GAP_MARGIN_M = 0.006
PHASE2_CLOSE_STAGE_A_FRACTION = 0.65
PHASE2_RETENTION_STEPS = 18
PHASE2_SHORT_LIFT_HEIGHT_M = 0.018
PHASE2_SHORT_LIFT_MIN_DELTA_M = 0.006
PHASE2_SHORT_LIFT_TICKS = 80
PHASE2_MAX_RETRIES = 1
PHASE2_FINGERTIP_DISTAL_PROXY_FACE_EPS_M = 0.002
PHASE2_FINGERTIP_PROXY_MAX_OFFSET_M = 0.16

OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"
OFFICIAL_GRIPPER_OPEN_WIDTH = -0.0215
OFFICIAL_GRIPPER_CLOSE_WIDTH = 0.01
DEFAULT_GRIPPER_HOLD_EFFORT = 100.0
DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH = "/World/DebugProxyMiddlePoint"
DEBUG_PROXY_MIDDLE_POINT_MARKER_RADIUS_M = 0.02
DEBUG_PROXY_MIDDLE_POINT_MARKER_COLOR = (1.0, 0.12, 0.00)
DEBUG_OBJECT_GRASP_CENTER_MARKER_PATH = "/World/DebugObjectGraspCenter"
DEBUG_OBJECT_GRASP_CENTER_MARKER_RADIUS_M = 0.013
DEBUG_OBJECT_GRASP_CENTER_MARKER_COLOR = (0.0, 1.0, 0.2)
DEBUG_PREGRASP_TARGET_MARKER_PATH = "/World/DebugPregraspTarget"
DEBUG_PREGRASP_TARGET_MARKER_RADIUS_M = 0.010
DEBUG_PREGRASP_TARGET_MARKER_COLOR = (0.2, 0.75, 1.0)
DEBUG_TIP1_MARKER_PATH = "/World/DebugTip1World"
DEBUG_TIP2_MARKER_PATH = "/World/DebugTip2World"
DEBUG_TIP_MID_MARKER_PATH = "/World/DebugTipMidWorld"
DEBUG_OBJECT_CENTER_MARKER_PATH = "/World/DebugObjectCenter"
DEBUG_RUNTIME_OBJECT_GRASP_CENTER_MARKER_PATH = "/World/DebugRuntimeObjectGraspCenter"
DEBUG_REAL_GRASP_CENTER_MARKER_PATH = "/World/DebugRealGraspCenter"
DEBUG_CONTACT_POINT_MARKER_PATH = "/World/DebugContactPointWorld"
DEBUG_CONTACT_POINT_B_MARKER_PATH = "/World/DebugContactPointBWorld"
DEBUG_TIP_MARKER_RADIUS_M = 0.010
DEBUG_TIP1_MARKER_COLOR = (0.0, 0.45, 1.0)
DEBUG_TIP2_MARKER_COLOR = (0.0, 0.95, 1.0)
DEBUG_TIP_MID_MARKER_COLOR = (1.0, 0.85, 0.0)
DEBUG_OBJECT_CENTER_MARKER_COLOR = (1.0, 1.0, 1.0)
DEBUG_RUNTIME_OBJECT_GRASP_CENTER_MARKER_COLOR = (0.1, 1.0, 0.25)
DEBUG_REAL_GRASP_CENTER_MARKER_COLOR = (1.0, 0.15, 0.15)
DEBUG_CONTACT_POINT_MARKER_COLOR = (0.8, 0.2, 1.0)
DEBUG_CONTACT_POINT_B_MARKER_COLOR = (1.0, 0.55, 0.05)

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
DEFAULT_HORIZONTAL_DESCENT_XY_TRIGGER_TOLERANCE = PHASE2_HORIZONTAL_DESCENT_XY_TRIGGER_M
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
VERTICAL_FINGER_MIDPOINT_REFERENCE_ALIASES = {
    "finger_midpoint",
    "fingertip_midpoint",
    "fingertip_end_midpoint",
    "finger_tip_midpoint",
    "tip_midpoint",
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
    "resolve_table_frame",
    "select_target",
    "build_scene_state_object_info",
    "hybrid_phase1_select_candidate",
    "phase2_estimate_object_grasp_frame",
    "phase2_geometric_filter_select_candidate",
    "plan_grasp_geometry",
    "select_pregrasp_candidate",
    "servo_pregrasp",
    "servo_align",
    "servo_descend",
    "far_outboard_transition",
    "far_prepare_low_side_approach",
    "far_align_B_over_object_xy",
    "far_lower_B_world_z",
    "phase2_far_final_descent_local_ik",
    "mid_align_AB_vertical_over_object",
    "mid_pre_descend_AB_vertical",
    "mid_descend_world_z_keep_AB_vertical",
    "phase2_vertical_final_descent_local_ik",
    "phase2_gap_close_stage_a",
    "phase2_retention_hold_stage_b",
    "phase2_short_lift_verify",
    "phase2_recover_and_retry",
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


class ServoEarlyStop(RuntimeError):
    def __init__(self, reason: str, details: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


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


def _fail(reason: str, message: str) -> None:
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
        "tip1_world_position": positions[0].tolist(),
        "tip2_world_position": positions[1].tolist(),
        "finger1_world_position": positions[0].tolist(),
        "finger2_world_position": positions[1].tolist(),
        "world_position": midpoint.tolist(),
        "fingertip_midpoint_world": midpoint.tolist(),
    }


def _resolve_named_transform_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    name_tokens: list[str],
    arm_side: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    tokens = [str(token or "").strip().lower() for token in name_tokens if str(token or "").strip()]
    if not tokens:
        return None, {"resolved": False, "reason": "empty_transform_position_tokens"}

    body_matches: list[dict[str, Any]] = []
    for index, body, name, path in _list_articulation_bodies(dc, articulation):
        combined = f"{name} {path}".lower()
        for token in tokens:
            if token in combined:
                body_matches.append(
                    {
                        "index": index,
                        "body": body,
                        "name": name,
                        "path": path,
                        "matched_token": token,
                        "arm_side_match_score": _arm_side_match_score(combined, arm_side),
                        "source": "dynamic_control_body_transform",
                    }
                )
    if body_matches:
        selected = max(body_matches, key=lambda item: (int(item["arm_side_match_score"]), len(str(item["matched_token"])), -int(item["index"])))
        body_pos = _body_pose_position(dc, selected["body"])
        return body_pos, {
            "resolved": True,
            "source": selected["source"],
            "position_semantics": "dynamic_control_rigid_body_pose_translation_for_explicit_fingertip_frame",
            "requested_name_tokens": name_tokens,
            "matched_token": selected["matched_token"],
            "selected_name": selected["name"],
            "selected_path": selected["path"],
            "selected_arm_side_match_score": selected["arm_side_match_score"],
            "candidate_count": len(body_matches),
            "candidates": [{key: value for key, value in item.items() if key != "body"} for item in body_matches],
            "world_position": body_pos.tolist(),
        }

    from pxr import Usd  # type: ignore

    prim_matches: list[dict[str, Any]] = []
    root = str(robot_root_path).rstrip("/")
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if root and not path.startswith(root):
            continue
        lower = path.lower()
        for token in tokens:
            if token in lower:
                prim_matches.append(
                    {
                        "path": path,
                        "matched_token": token,
                        "arm_side_match_score": _arm_side_match_score(path, arm_side),
                        "source": "usd_prim_transform",
                    }
                )
    if prim_matches:
        selected = max(prim_matches, key=lambda item: (int(item["arm_side_match_score"]), len(str(item["matched_token"])), -len(str(item["path"]))))
        se3, prim_log = _world_se3_from_prim(stage, str(selected["path"]))
        prim_pos = np.array(se3.translation, dtype=float)
        return prim_pos, {
            "resolved": True,
            "source": selected["source"],
            "position_semantics": "usd_prim_transform_translation_for_explicit_fingertip_frame",
            "requested_name_tokens": name_tokens,
            "matched_token": selected["matched_token"],
            "selected_path": selected["path"],
            "selected_arm_side_match_score": selected["arm_side_match_score"],
            "candidate_count": len(prim_matches),
            "candidates": prim_matches,
            "world_position": prim_pos.tolist(),
            "prim_world_log": prim_log,
        }

    return None, {
        "resolved": False,
        "source": None,
        "requested_name_tokens": name_tokens,
        "reason": "no_dynamic_control_body_or_usd_prim_matched_explicit_fingertip_tokens",
    }


def _resolve_active_palm_reference_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prefix = "R" if arm_side == "right" else "L"
    tokens = [
        f"{prefix}_pgc_base_link",
        "pgc_base_link",
        f"{prefix}_sixforce_link",
        "sixforce_link",
        f"{prefix}_wrist_roll_link",
        "wrist_roll_link",
    ]
    for token in tokens:
        pos, log = _resolve_named_body_or_prim_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            name_token=token,
            arm_side=arm_side,
        )
        if pos is not None:
            return np.array(pos, dtype=float), {
                **log,
                "palm_reference_token": token,
                "palm_reference_semantics": "proximal hand/palm reference used to choose distal fingertip end",
            }
    return None, {
        "resolved": False,
        "source": None,
        "requested_tokens": tokens,
        "reason": "no_palm_or_wrist_reference_resolved_for_distal_tip_proxy",
    }


def _resolve_actual_fingertip_frame_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    finger_role: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prefix = "R" if arm_side == "right" else "L"
    compact_role = finger_role.replace("_link", "")
    tokens = [
        f"{prefix}_{compact_role}_tip",
        f"{prefix}_{compact_role}_tip_link",
        f"{prefix}_{compact_role}_fingertip",
        f"{prefix}_{compact_role}_finger_tip",
        f"{prefix}_{compact_role}_distal",
        f"{prefix}_{compact_role}_end",
        f"{prefix}_{compact_role}_contact",
        f"{compact_role}_tip",
        f"{compact_role}_fingertip",
        f"{compact_role}_distal",
        f"{compact_role}_end",
    ]
    pos, log = _resolve_named_transform_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        name_tokens=tokens,
        arm_side=arm_side,
    )
    if pos is None:
        return None, {
            **log,
            "finger_role": finger_role,
            "resolution_layer": "actual_fingertip_frame",
        }
    return np.array(pos, dtype=float), {
        **log,
        "finger_role": finger_role,
        "resolution_layer": "actual_fingertip_frame",
        "fingertip_reference_source": "actual_fingertip_frame",
    }


def _resolve_actual_fingertip_pair_midpoint_reference_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    requested_name_token: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    positions: list[np.ndarray] = []
    component_logs: list[dict[str, Any]] = []
    missing_roles: list[str] = []
    for role in ("finger1_link", "finger2_link"):
        pos, log = _resolve_actual_fingertip_frame_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            finger_role=role,
        )
        if pos is None:
            missing_roles.append(role)
            component_logs.append({"component_role": role, "resolved": False, "tip_resolution_log": log})
            continue
        pos = np.array(pos, dtype=float)
        positions.append(pos)
        component_logs.append({"component_role": role, "resolved": True, "selected_world_position": pos.tolist(), "tip_resolution_log": log})

    if missing_roles or len(positions) != 2:
        return None, {
            "enabled": True,
            "resolved": False,
            "source": None,
            "requested_name_token": requested_name_token,
            "reference_mode": "actual_fingertip_frame_pair_midpoint",
            "arm_side": arm_side,
            "missing_component_roles": missing_roles,
            "component_reference_logs": component_logs,
            "reason": "actual_fingertip_pair_requires_two_resolved_fingertip_frames",
            "fallback": "stable_finger_link_pair_midpoint",
        }

    midpoint = 0.5 * (positions[0] + positions[1])
    return midpoint, {
        "enabled": True,
        "resolved": True,
        "source": "actual_fingertip_frame_pair_midpoint",
        "fingertip_reference_source": "actual_fingertip_frame_pair",
        "fingertip_reference_source_used": "actual_fingertip_frame_pair",
        "fingertip_component_reference_sources": ["actual_fingertip_frame", "actual_fingertip_frame"],
        "position_semantics": "midpoint_of_two_actual_fingertip_frames",
        "requested_name_token": requested_name_token,
        "reference_mode": "actual_fingertip_frame_pair_midpoint",
        "arm_side": arm_side,
        "component_reference_logs": component_logs,
        "component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "fingertip_component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "finger1_tip_world_position": positions[0].tolist(),
        "finger2_tip_world_position": positions[1].tolist(),
        "fingertip_midpoint_world": midpoint.tolist(),
        "world_position": midpoint.tolist(),
        "fallback_used": False,
        "fallback_status": "actual_fingertip_frame_pair_used",
    }


def _resolve_finger_link_distal_tip_proxy_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    finger_role: str,
    palm_world: np.ndarray | None,
    palm_log: dict[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prefix = "R" if arm_side == "right" else "L"
    preferred_token = f"{prefix}_{finger_role}"
    fallback_token = finger_role
    attempts: list[dict[str, Any]] = []
    link_pos: np.ndarray | None = None
    link_log: dict[str, Any] | None = None
    for token in (preferred_token, fallback_token):
        pos, log = _resolve_named_body_or_prim_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            name_token=token,
            arm_side=arm_side,
        )
        log = {**log, "component_role": finger_role, "component_name_token": token}
        attempts.append(log)
        if pos is not None:
            link_pos = np.array(pos, dtype=float)
            link_log = log
            break
    if link_pos is None or link_log is None:
        return None, {
            "resolved": False,
            "source": None,
            "finger_role": finger_role,
            "resolution_layer": "finger_link_distal_tip_proxy",
            "attempts": attempts,
            "reason": "finger_link_unresolved_for_tip_proxy",
        }

    selected_path = link_log.get("selected_path")
    if not selected_path:
        return None, {
            "resolved": False,
            "source": None,
            "finger_role": finger_role,
            "resolution_layer": "finger_link_distal_tip_proxy",
            "link_reference_log": link_log,
            "reason": "finger_link_selected_path_missing_for_tip_proxy",
        }

    try:
        bbox = _bbox(stage, str(selected_path))
        corners = _bbox_corners_from_bbox(bbox)
        bbox_center = np.array(bbox["center"], dtype=float)
        link_se3, link_world_log = _world_se3_from_prim(stage, str(selected_path))
        link_origin = np.array(link_se3.translation, dtype=float)
        link_rot = np.array(link_se3.rotation, dtype=float)
    except Exception as exc:
        return None, {
            "resolved": False,
            "source": None,
            "finger_role": finger_role,
            "resolution_layer": "finger_link_distal_tip_proxy",
            "link_reference_log": link_log,
            "reason": "finger_link_bbox_or_transform_unavailable",
            "error": repr(exc),
        }

    palm = np.array(palm_world, dtype=float) if palm_world is not None else None
    if palm is not None and np.isfinite(palm).all():
        distal_axis = _normalize(bbox_center - palm, np.array([0.0, 0.0, -1.0], dtype=float))
        distal_axis_source = "palm_reference_to_finger_bbox_center"
    else:
        extents = np.abs(np.array(bbox["max"], dtype=float) - np.array(bbox["min"], dtype=float))
        largest_axis_index = int(np.argmax(extents))
        fallback_axis = np.eye(3, dtype=float)[largest_axis_index]
        distal_axis = fallback_axis
        distal_axis_source = "largest_world_bbox_axis_fallback_no_palm_reference"

    projections = corners @ distal_axis
    max_projection = float(np.max(projections))
    center_projection = float(np.dot(bbox_center, distal_axis))
    tip_world = bbox_center + distal_axis * (max_projection - center_projection)
    local_offset = link_rot.T @ (tip_world - link_origin)
    local_offset_norm = float(np.linalg.norm(local_offset))
    if local_offset_norm > PHASE2_FINGERTIP_PROXY_MAX_OFFSET_M:
        return None, {
            "resolved": False,
            "source": None,
            "finger_role": finger_role,
            "resolution_layer": "finger_link_distal_tip_proxy",
            "link_reference_log": link_log,
            "link_world_log": link_world_log,
            "computed_local_offset": local_offset.tolist(),
            "computed_local_offset_norm_m": local_offset_norm,
            "max_allowed_local_offset_norm_m": PHASE2_FINGERTIP_PROXY_MAX_OFFSET_M,
            "reason": "computed_fingertip_proxy_offset_too_large",
        }
    calibrated_tip_world = link_origin + link_rot @ local_offset
    return calibrated_tip_world, {
        "resolved": True,
        "source": "finger_link_bbox_distal_face_local_offset_proxy",
        "fingertip_reference_source": "calibrated_finger_link_tip_proxy",
        "position_semantics": "distal fingertip proxy from finger-link bbox face farthest from palm reference",
        "finger_role": finger_role,
        "resolution_layer": "finger_link_distal_tip_proxy",
        "arm_side": arm_side,
        "selected_link_path": str(selected_path),
        "link_reference_log": link_log,
        "link_world_log": link_world_log,
        "palm_reference_log": palm_log,
        "palm_reference_world": None if palm is None else palm.tolist(),
        "finger_link_bbox": bbox,
        "finger_link_bbox_center_world": bbox_center.tolist(),
        "distal_axis_world": distal_axis.tolist(),
        "distal_axis_source": distal_axis_source,
        "max_distal_projection": max_projection,
        "center_distal_projection": center_projection,
        "calibrated_local_tip_offset_from_link_frame": local_offset.tolist(),
        "calibrated_local_tip_offset_norm_m": local_offset_norm,
        "world_position": calibrated_tip_world.tolist(),
        "attempts": attempts,
    }


def _resolve_single_fingertip_end_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    finger_role: str,
    palm_world: np.ndarray | None,
    palm_log: dict[str, Any] | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    actual_pos, actual_log = _resolve_actual_fingertip_frame_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        finger_role=finger_role,
    )
    if actual_pos is not None:
        return actual_pos, {
            **actual_log,
            "fallback_used": False,
            "fallback_status": "actual_fingertip_frame_used",
        }

    proxy_pos, proxy_log = _resolve_finger_link_distal_tip_proxy_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        finger_role=finger_role,
        palm_world=palm_world,
        palm_log=palm_log,
    )
    if proxy_pos is not None:
        return proxy_pos, {
            **proxy_log,
            "actual_fingertip_frame_attempt": actual_log,
            "fallback_used": True,
            "fallback_status": "actual_tip_frame_missing_used_calibrated_link_tip_proxy",
        }
    return None, {
        "resolved": False,
        "source": None,
        "finger_role": finger_role,
        "actual_fingertip_frame_attempt": actual_log,
        "proxy_attempt": proxy_log,
        "fallback_used": True,
        "fallback_status": "actual_tip_frame_and_calibrated_link_tip_proxy_failed",
    }


def _resolve_calibrated_distal_proxy_pair_midpoint_reference_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    requested_name_token: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    palm_world, palm_log = _resolve_active_palm_reference_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
    )
    positions: list[np.ndarray] = []
    component_logs: list[dict[str, Any]] = []
    missing_roles: list[str] = []
    for role in ("finger1_link", "finger2_link"):
        pos, log = _resolve_finger_link_distal_tip_proxy_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            finger_role=role,
            palm_world=palm_world,
            palm_log=palm_log,
        )
        if pos is None:
            missing_roles.append(role)
            component_logs.append({"component_role": role, "resolved": False, "tip_resolution_log": log})
            continue
        pos = np.array(pos, dtype=float)
        positions.append(pos)
        component_logs.append({"component_role": role, "resolved": True, "selected_world_position": pos.tolist(), "tip_resolution_log": log})

    if missing_roles or len(positions) != 2:
        return None, {
            "enabled": True,
            "resolved": False,
            "source": None,
            "requested_name_token": requested_name_token,
            "reference_mode": "calibrated_distal_proxy_pair_midpoint",
            "arm_side": arm_side,
            "missing_component_roles": missing_roles,
            "component_reference_logs": component_logs,
            "palm_reference_log": palm_log,
            "reason": "calibrated_distal_proxy_pair_requires_two_resolved_distal_proxies",
            "fallback": "explicit_proxy_only",
        }

    midpoint = 0.5 * (positions[0] + positions[1])
    return midpoint, {
        "enabled": True,
        "resolved": True,
        "source": "calibrated_finger_link_tip_proxy_pair_midpoint",
        "fingertip_reference_source": "calibrated_finger_link_tip_proxy_pair",
        "fingertip_reference_source_used": "calibrated_finger_link_tip_proxy_pair",
        "fingertip_component_reference_sources": ["calibrated_finger_link_tip_proxy", "calibrated_finger_link_tip_proxy"],
        "position_semantics": "midpoint_of_two_calibrated_distal_fingertip_proxy_references",
        "requested_name_token": requested_name_token,
        "reference_mode": "calibrated_distal_proxy_pair_midpoint",
        "arm_side": arm_side,
        "component_reference_logs": component_logs,
        "component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "fingertip_component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "finger1_tip_world_position": positions[0].tolist(),
        "finger2_tip_world_position": positions[1].tolist(),
        "fingertip_midpoint_world": midpoint.tolist(),
        "world_position": midpoint.tolist(),
        "fallback_used": True,
        "fallback_status": "calibrated_distal_proxy_pair_used_after_actual_and_stable_link_midpoint_failed",
        "palm_reference_log": palm_log,
    }


def _resolve_fingertip_midpoint_reference_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    requested_name_token: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    palm_world, palm_log = _resolve_active_palm_reference_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
    )
    positions: list[np.ndarray] = []
    component_logs: list[dict[str, Any]] = []
    missing_roles: list[str] = []
    for role in ("finger1_link", "finger2_link"):
        pos, log = _resolve_single_fingertip_end_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            finger_role=role,
            palm_world=palm_world,
            palm_log=palm_log,
        )
        if pos is None:
            missing_roles.append(role)
            component_logs.append({"component_role": role, "resolved": False, "tip_resolution_log": log})
            continue
        pos = np.array(pos, dtype=float)
        positions.append(pos)
        component_logs.append({"component_role": role, "resolved": True, "selected_world_position": pos.tolist(), "tip_resolution_log": log})

    legacy_midpoint, legacy_log = _resolve_finger_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token=f"{requested_name_token}_legacy_link_midpoint_for_delta",
    )
    if missing_roles or len(positions) != 2:
        return None, {
            "enabled": True,
            "resolved": False,
            "source": None,
            "requested_name_token": requested_name_token,
            "reference_mode": "fingertip_end_midpoint",
            "arm_side": arm_side,
            "missing_component_roles": missing_roles,
            "component_reference_logs": component_logs,
            "palm_reference_log": palm_log,
            "legacy_link_midpoint_log": legacy_log,
            "reason": "fingertip_midpoint_requires_two_resolved_fingertip_end_references",
            "fallback": "callers may fall back to legacy finger-link midpoint or point_B proxy",
        }

    midpoint = 0.5 * (positions[0] + positions[1])
    legacy_delta = None if legacy_midpoint is None else midpoint - np.array(legacy_midpoint, dtype=float)
    low_level_sources = [str(item.get("tip_resolution_log", {}).get("source")) for item in component_logs]
    component_reference_sources = [
        str(item.get("tip_resolution_log", {}).get("fingertip_reference_source", "unknown"))
        for item in component_logs
    ]
    if all(source == "actual_fingertip_frame" for source in component_reference_sources):
        fingertip_reference_source_used = "actual_fingertip_frame_pair"
        source = "actual_fingertip_frame_pair_midpoint"
    elif all(source == "calibrated_finger_link_tip_proxy" for source in component_reference_sources):
        fingertip_reference_source_used = "calibrated_finger_link_tip_proxy_pair"
        source = "calibrated_finger_link_tip_proxy_pair_midpoint"
    else:
        fingertip_reference_source_used = "mixed_actual_and_calibrated_finger_link_tip_proxy_pair"
        source = "mixed_fingertip_end_pair_midpoint"
    return midpoint, {
        "enabled": True,
        "resolved": True,
        "source": source,
        "fingertip_reference_source": fingertip_reference_source_used,
        "fingertip_reference_source_used": fingertip_reference_source_used,
        "fingertip_component_reference_sources": component_reference_sources,
        "fingertip_low_level_sources": low_level_sources,
        "position_semantics": "midpoint_of_two_fingertip_end_references_not_finger_link_centers",
        "requested_name_token": requested_name_token,
        "reference_mode": "fingertip_end_midpoint",
        "arm_side": arm_side,
        "component_reference_logs": component_logs,
        "component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "fingertip_component_positions_world": [positions[0].tolist(), positions[1].tolist()],
        "finger1_tip_world_position": positions[0].tolist(),
        "finger2_tip_world_position": positions[1].tolist(),
        "fingertip_midpoint_world": midpoint.tolist(),
        "world_position": midpoint.tolist(),
        "fallback_used": any(bool(item.get("tip_resolution_log", {}).get("fallback_used", False)) for item in component_logs),
        "fallback_status": "actual_tip_frame_if_available_else_calibrated_link_tip_proxy",
        "palm_reference_log": palm_log,
        "legacy_link_midpoint_world": None if legacy_midpoint is None else np.array(legacy_midpoint, dtype=float).tolist(),
        "legacy_link_midpoint_log": legacy_log,
        "legacy_link_midpoint_to_fingertip_delta_world": None if legacy_delta is None else legacy_delta.tolist(),
        "legacy_link_midpoint_to_fingertip_delta_norm_m": None if legacy_delta is None else float(np.linalg.norm(legacy_delta)),
        "legacy_link_midpoint_to_fingertip_midpoint_delta_world": None if legacy_delta is None else legacy_delta.tolist(),
        "legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m": None if legacy_delta is None else float(np.linalg.norm(legacy_delta)),
    }


def _finite_world_vector_or_none(value: Any, *, size: int = 3) -> np.ndarray | None:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < size:
        return None
    arr = arr[:size]
    if not np.isfinite(arr).all():
        return None
    return arr


def _first_finite_vector_from_mapping(mapping: dict[str, Any] | None, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(mapping, dict):
        return None, None
    for key in keys:
        arr = _finite_world_vector_or_none(mapping.get(key))
        if arr is not None:
            return arr, key
    return None, None


def _compute_runtime_two_finger_metrics(
    *,
    reference_log: dict[str, Any] | None,
    object_grasp_frame: dict[str, Any] | None,
    object_center_world: np.ndarray | list[float] | None = None,
    fallback_midpoint_world: np.ndarray | list[float] | None = None,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    # Runtime truth is fingertip-centric.
    # point_B is compatibility-only and must not be treated as final grasp truth.
    reference_log = reference_log if isinstance(reference_log, dict) else {}
    object_grasp_frame = object_grasp_frame if isinstance(object_grasp_frame, dict) else {}
    component_positions = reference_log.get(
        "fingertip_component_positions_world",
        reference_log.get("component_positions_world"),
    )

    tip1_world = None
    tip2_world = None
    if isinstance(component_positions, list) and len(component_positions) >= 2:
        tip1_world = _finite_world_vector_or_none(component_positions[0])
        tip2_world = _finite_world_vector_or_none(component_positions[1])

    two_tip_geometry_available = bool(tip1_world is not None and tip2_world is not None)
    tip_mid_world = None
    tip_axis_world = None
    tip_span_m = None
    if two_tip_geometry_available:
        tip_mid_world = 0.5 * (tip1_world + tip2_world)
        tip_delta = tip2_world - tip1_world
        tip_span_m = float(np.linalg.norm(tip_delta))
        if tip_span_m > 1e-9:
            tip_axis_world = tip_delta / tip_span_m
    else:
        fallback_mid = _finite_world_vector_or_none(fallback_midpoint_world)
        if fallback_mid is not None:
            tip_mid_world = fallback_mid

    object_grasp_center, object_center_source = _first_finite_vector_from_mapping(
        object_grasp_frame,
        [
            "object_grasp_center_world",
            "grasp_center_world",
            "center_world",
            "target_center_world",
            "object_center_world",
        ],
    )
    object_center = _finite_world_vector_or_none(object_center_world)
    if object_center is None:
        object_center, object_center_world_source = _first_finite_vector_from_mapping(
            object_grasp_frame,
            ["object_center_world", "object_bbox_center_world"],
        )
    else:
        object_center_world_source = "runtime_object_bbox_center_world"

    object_grasp_axis, object_axis_source = _first_finite_vector_from_mapping(
        object_grasp_frame,
        [
            "tip_axis_world",
            "closing_axis_world",
            "hand_closing_axis_world",
            "grasp_axis_world",
            "minor_axis_world",
            "width_axis_world",
        ],
    )
    if object_grasp_axis is not None:
        axis_norm = float(np.linalg.norm(object_grasp_axis))
        object_grasp_axis = None if axis_norm <= 1e-9 else object_grasp_axis / axis_norm

    tip_mid_error = None
    tip_mid_xy_error = None
    tip_mid_z_error = None
    tip_symmetry_error = None
    if tip_mid_world is not None and object_grasp_center is not None:
        tip_delta_to_grasp = tip_mid_world - object_grasp_center
        tip_mid_error = float(np.linalg.norm(tip_delta_to_grasp))
        tip_mid_xy_error = float(np.linalg.norm(tip_delta_to_grasp[:2]))
        tip_mid_z_error = float(abs(tip_delta_to_grasp[2]))
    if two_tip_geometry_available and object_grasp_center is not None:
        tip1_to_grasp = float(np.linalg.norm(tip1_world - object_grasp_center))
        tip2_to_grasp = float(np.linalg.norm(tip2_world - object_grasp_center))
        tip_symmetry_error = float(abs(tip1_to_grasp - tip2_to_grasp))
    else:
        tip1_to_grasp = None
        tip2_to_grasp = None

    if two_tip_geometry_available and object_center is not None:
        tip1_to_object = float(np.linalg.norm(tip1_world - object_center))
        tip2_to_object = float(np.linalg.norm(tip2_world - object_center))
    else:
        tip1_to_object = None
        tip2_to_object = None

    tip_axis_alignment_error = None
    if tip_axis_world is not None and object_grasp_axis is not None:
        tip_axis_alignment_error = _angle_between_axes_unsigned(tip_axis_world, object_grasp_axis)

    tip_z_asymmetry = None
    if two_tip_geometry_available:
        tip_z_asymmetry = float(abs(float(tip1_world[2]) - float(tip2_world[2])))

    close_critical_reference_trusted = bool(reference_log.get("close_critical_reference", False))
    primary_truth_available = bool(
        two_tip_geometry_available
        and tip_mid_world is not None
        and close_critical_reference_trusted
    )
    return {
        "metric_schema": "contact_centric_two_finger_runtime_metrics_v1",
        "primary_runtime_truth": primary_truth_available,
        "close_critical_reference_trusted": close_critical_reference_trusted,
        "two_tip_geometry_available": two_tip_geometry_available,
        "tip1_world": None if tip1_world is None else tip1_world.tolist(),
        "tip2_world": None if tip2_world is None else tip2_world.tolist(),
        "tip_mid_world": None if tip_mid_world is None else tip_mid_world.tolist(),
        "tip_axis_world": None if tip_axis_world is None else tip_axis_world.tolist(),
        "tip_span_m": tip_span_m,
        "object_center_world": None if object_center is None else object_center.tolist(),
        "object_center_world_source": object_center_world_source,
        "object_grasp_center_world": None if object_grasp_center is None else object_grasp_center.tolist(),
        "object_grasp_center_source": object_center_source,
        "object_grasp_axis_world": None if object_grasp_axis is None else object_grasp_axis.tolist(),
        "object_grasp_axis_source": object_axis_source,
        "tip_mid_error_to_object_grasp_center_m": tip_mid_error,
        "tip_mid_xy_error_m": tip_mid_xy_error,
        "tip_mid_z_error_m": tip_mid_z_error,
        "tip1_to_object_grasp_center_distance_m": tip1_to_grasp,
        "tip2_to_object_grasp_center_distance_m": tip2_to_grasp,
        "tip1_to_object_center_distance_m": tip1_to_object,
        "tip2_to_object_center_distance_m": tip2_to_object,
        "tip_axis_alignment_error_rad": tip_axis_alignment_error,
        "tip_symmetry_error_m": tip_symmetry_error,
        "tip_z_asymmetry_m": tip_z_asymmetry,
        "runtime_reference_source": reference_log.get(
            "fingertip_reference_source_used",
            reference_log.get("fingertip_reference_source", reference_log.get("source")),
        ),
        "runtime_reference_resolution_source": reference_log.get("source"),
        "runtime_reference_fallback_used": bool(reference_log.get("fallback_used", False)),
        "fallback_midpoint_source": fallback_source,
        "point_B_is_compatibility_only": True,
    }


def _pose_for_contact_reference_world(
    contact_reference_world: np.ndarray | list[float],
    coord_transform: Any,
    rpy_base: np.ndarray | list[float],
    contact_reference_offset_local: np.ndarray | list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    pose, pose_log = _pose_for_point_b_world(
        contact_reference_world,
        coord_transform,
        rpy_base,
        contact_reference_offset_local,
    )
    log = {
        **pose_log,
        "legacy_pose_builder_target_semantics": pose_log.get("target_semantics"),
        "target_semantics": "contact_reference_world_driven",
        "pose_construction_semantics": "contact_reference_converted_through_local_compatibility_offset",
        "contact_reference_world": np.array(contact_reference_world, dtype=float).tolist(),
        "contact_reference_offset_local": np.array(contact_reference_offset_local, dtype=float).tolist(),
        "point_b_target_world_is_contact_reference_alias": True,
        "point_B_final_truth_source": False,
        "compatibility_conversion_only": True,
    }
    return pose, log


def _resolve_vertical_finger_midpoint_reference_position(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    requested_name_token: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    actual_world, actual_log = _resolve_actual_fingertip_pair_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token=f"{requested_name_token}_actual_fingertip_pair",
    )
    link_world, link_log = _resolve_finger_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token=f"{requested_name_token}_stable_link_pair",
    )
    if actual_world is not None:
        return np.array(actual_world, dtype=float), {
            **actual_log,
            "requested_name_token": requested_name_token,
            "source": "actual_fingertip_frame_pair_midpoint",
            "reference_mode": "vertical_actual_fingertip_or_stable_link_midpoint",
            "vertical_xy_reference_truth_source": "actual_fingertip_frame_pair_midpoint",
            "vertical_xy_reference_proxy_policy": "point_B_compatibility_only_calibrated_distal_proxy_not_used",
            "stable_link_midpoint_attempt": link_log,
            "close_critical_reference": True,
            "fallback_used": False,
        }
    if link_world is not None:
        return np.array(link_world, dtype=float), {
            **link_log,
            "requested_name_token": requested_name_token,
            "source": "stable_finger_link_pair_midpoint",
            "fingertip_reference_source": "stable_finger_link_pair_midpoint",
            "fingertip_reference_source_used": "stable_finger_link_pair_midpoint",
            "reference_mode": "vertical_actual_fingertip_or_stable_link_midpoint",
            "vertical_xy_reference_truth_source": "stable_finger_link_pair_midpoint",
            "vertical_xy_reference_proxy_policy": "point_B_compatibility_only_calibrated_distal_proxy_not_used",
            "actual_fingertip_pair_attempt": actual_log,
            "close_critical_reference": True,
            "fallback_used": True,
            "fallback_source": "stable_finger_link_pair_midpoint",
            "fallback_status": "actual_fingertip_pair_missing_used_stable_finger_link_pair_midpoint",
        }
    return None, {
        "enabled": True,
        "resolved": False,
        "source": None,
        "requested_name_token": requested_name_token,
        "reference_mode": "vertical_actual_fingertip_or_stable_link_midpoint",
        "arm_side": arm_side,
        "actual_fingertip_pair_attempt": actual_log,
        "stable_link_midpoint_attempt": link_log,
        "vertical_xy_reference_proxy_policy": "calibrated_distal_proxy_is_diagnostic_only_not_vertical_xy_truth",
        "reason": "vertical_xy_reference_requires_actual_fingertip_pair_or_stable_finger_link_pair_midpoint",
        "fallback": "point_B XY compatibility remains active",
    }


def _upsert_two_finger_runtime_debug_markers(
    *,
    stage: Any,
    metrics: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    marker_specs = [
        ("tip1_world", DEBUG_TIP1_MARKER_PATH, DEBUG_TIP_MARKER_RADIUS_M, DEBUG_TIP1_MARKER_COLOR),
        ("tip2_world", DEBUG_TIP2_MARKER_PATH, DEBUG_TIP_MARKER_RADIUS_M, DEBUG_TIP2_MARKER_COLOR),
        ("tip_mid_world", DEBUG_TIP_MID_MARKER_PATH, DEBUG_TIP_MARKER_RADIUS_M, DEBUG_TIP_MID_MARKER_COLOR),
        ("object_center_world", DEBUG_OBJECT_CENTER_MARKER_PATH, DEBUG_TIP_MARKER_RADIUS_M, DEBUG_OBJECT_CENTER_MARKER_COLOR),
        (
            "object_grasp_center_world",
            DEBUG_RUNTIME_OBJECT_GRASP_CENTER_MARKER_PATH,
            DEBUG_TIP_MARKER_RADIUS_M,
            DEBUG_RUNTIME_OBJECT_GRASP_CENTER_MARKER_COLOR,
        ),
    ]
    marker_log: dict[str, Any] = {
        "enabled": bool(enabled),
        "marker_schema": "contact_centric_two_finger_runtime_markers_v1",
        "markers": {},
    }
    if not enabled:
        return marker_log
    for key, path, radius, color in marker_specs:
        pos = _finite_world_vector_or_none(metrics.get(key))
        if pos is None:
            marker_log["markers"][key] = {"path": path, "updated": False, "reason": "position_unavailable"}
            continue
        try:
            _upsert_debug_marker(stage=stage, path=path, position=pos, radius=radius, color=color)
            marker_log["markers"][key] = {"path": path, "updated": True, "world_position": pos.tolist()}
        except Exception as exc:
            marker_log["markers"][key] = {"path": path, "updated": False, "error": repr(exc)}
    return marker_log


def _reference_comparison_payload(
    *,
    actual_log: dict[str, Any] | None,
    link_log: dict[str, Any] | None,
    calibrated_log: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "comparison_schema": "actual_vs_link_midpoint_vs_calibrated_proxy_v1",
        "actual_fingertip_pair": actual_log or {},
        "stable_finger_link_midpoint": link_log or {},
        "calibrated_distal_proxy_pair": calibrated_log or {},
        "truth_order": [
            "actual_fingertip_frame_pair",
            "stable_finger_link_pair_midpoint",
            "calibrated_distal_proxy_pair_diagnostic_only",
            "explicit_fallback_proxy",
        ],
        "calibrated_proxy_close_authority": "diagnostic_only_unless_promoted_after_runtime_validation",
        "bad_proxy_must_not_silently_dominate": True,
    }


def _resolve_finger_link_midpoint_diagnostic_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    midpoint_world, midpoint_log = _resolve_finger_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token="phase2_debug_finger_link_midpoint_bypass",
    )
    if midpoint_world is None:
        return None, {
            "enabled": True,
            "resolved": False,
            "source": None,
            "diagnostic_mode": True,
            "reference_mode": "finger_link_midpoint_bypass",
            "requested_name_token": "phase2_debug_finger_link_midpoint_bypass",
            "reference_mode_diagnostic": "diagnostic_finger_link_midpoint",
            "reason": midpoint_log.get("reason"),
            "fallback": "fingertip_end_midpoint_or_legacy_path",
            "attempt_log": midpoint_log,
        }
    return np.array(midpoint_world, dtype=float), {
        **midpoint_log,
        "source": midpoint_log.get("source", "finger_link_pair_midpoint"),
        "reference_mode": "finger_link_midpoint_bypass",
        "diagnostic_mode": True,
        "fingertip_proxy_bypass": "active",
    }


def _resolve_real_grasp_center_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    fallback_world: np.ndarray | list[float] | None = None,
    fallback_source: str = "point_B_proxy",
    diagnostic_finger_link_midpoint_bypass: bool = False,
    include_diagnostic_comparison: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    def _compare_with_diagnostic_link_midpoint(
        primary_world: np.ndarray | list[float] | None,
        primary_log: dict[str, Any],
    ) -> dict[str, Any]:
        if not include_diagnostic_comparison:
            return {
                "diagnostic_compare_standard_midpoint_enabled": False,
            }
        midpoint_world, midpoint_log = _resolve_finger_link_midpoint_diagnostic_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
        )
        if midpoint_world is None:
            return {
                "diagnostic_compare_standard_midpoint_enabled": True,
                "diagnostic_standard_midpoint_world": None,
                "diagnostic_standard_midpoint_source": midpoint_log.get("source"),
                "diagnostic_midpoint_compare_reason": midpoint_log.get("reason"),
                "diagnostic_midpoint_delta_world": None,
                "diagnostic_midpoint_delta_norm_m": None,
            }
        midpoint_arr = np.array(midpoint_world, dtype=float)
        primary_arr = np.array(primary_world, dtype=float) if primary_world is not None and np.isfinite(np.array(primary_world, dtype=float)).all() else None
        if primary_arr is None:
            return {
                "diagnostic_compare_standard_midpoint_enabled": True,
                "diagnostic_standard_midpoint_world": midpoint_arr.tolist(),
                "diagnostic_standard_midpoint_source": midpoint_log.get("source"),
                "diagnostic_midpoint_delta_world": None,
                "diagnostic_midpoint_delta_norm_m": None,
            }
        delta = primary_arr - midpoint_arr
        return {
            "diagnostic_compare_standard_midpoint_enabled": True,
            "diagnostic_standard_midpoint_world": midpoint_arr.tolist(),
            "diagnostic_standard_midpoint_source": midpoint_log.get("source"),
            "diagnostic_midpoint_delta_world": delta.tolist(),
            "diagnostic_midpoint_delta_norm_m": float(np.linalg.norm(delta)),
        }

    actual_center_world, actual_log = _resolve_actual_fingertip_pair_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token="real_grasp_center_actual_fingertip_frame_midpoint",
    )
    link_midpoint_world, link_midpoint_log = _resolve_finger_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token="stable_real_grasp_center_finger_link_pair_midpoint",
    )
    calibrated_center_world, calibrated_log = _resolve_calibrated_distal_proxy_pair_midpoint_reference_position(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        requested_name_token="calibrated_real_grasp_center_distal_proxy_pair_midpoint",
    )
    comparison_payload = _reference_comparison_payload(
        actual_log=actual_log,
        link_log=link_midpoint_log,
        calibrated_log=calibrated_log,
    )
    calibrated_vs_link_midpoint_offset_m = None
    calibrated_vs_link_midpoint_warning = None
    if calibrated_center_world is not None and link_midpoint_world is not None:
        calibrated_arr = np.array(calibrated_center_world, dtype=float)
        link_arr = np.array(link_midpoint_world, dtype=float)
        if np.isfinite(calibrated_arr).all() and np.isfinite(link_arr).all():
            calibrated_vs_link_midpoint_offset_m = float(np.linalg.norm(calibrated_arr - link_arr))
            if calibrated_vs_link_midpoint_offset_m < 0.02:
                calibrated_vs_link_midpoint_warning = "Calibrated fingertip proxy offset too small - likely incorrect proxy geometry"

    if actual_center_world is not None:
        comparison_log = _compare_with_diagnostic_link_midpoint(
            primary_world=actual_center_world,
            primary_log=actual_log,
        )
        return actual_center_world, {
            **actual_log,
            "grasp_center_definition": "midpoint_between_actual_fingertip_frames",
            "resolution_priority": [
                "actual_fingertip_frames_if_available",
                "stable_finger_link_pair_midpoint",
                "calibrated_finger_link_tip_proxy_from_distal_bbox_face_diagnostic_only",
                "explicit_fallback_proxy",
            ],
            "fingertip_source": "actual",
            "calibrated_proxy_close_authority": "diagnostic_only",
            "calibrated_vs_link_midpoint_offset_m": calibrated_vs_link_midpoint_offset_m,
            "calibrated_vs_link_midpoint_warning": calibrated_vs_link_midpoint_warning,
            "fallback_used": False,
            "diagnostic_finger_link_midpoint_bypass": bool(diagnostic_finger_link_midpoint_bypass),
            "diagnostic_finger_link_midpoint_bypass_effect": (
                "comparison_only_link_midpoint_no_override"
                if diagnostic_finger_link_midpoint_bypass
                else "inactive"
            ),
            "close_critical_reference": True,
            "diagnostic_two_finger_reference_comparison": comparison_payload,
            **comparison_log,
        }

    if link_midpoint_world is not None:
        fallback_comparison = _compare_with_diagnostic_link_midpoint(
            primary_world=np.array(link_midpoint_world, dtype=float),
            primary_log=link_midpoint_log,
        )
        return link_midpoint_world, {
            **link_midpoint_log,
            "actual_fingertip_pair_resolution_log": actual_log,
            "calibrated_distal_proxy_resolution_log": calibrated_log,
            "source": "stable_finger_link_pair_midpoint",
            "fingertip_reference_source": "stable_finger_link_pair_midpoint",
            "fingertip_reference_source_used": "stable_finger_link_pair_midpoint",
            "fingertip_source": "stable_link_midpoint",
            "grasp_center_definition": "stable midpoint between finger link references after actual fingertip frames failed; calibrated distal proxy remains diagnostic only",
            "resolution_priority": [
                "actual_fingertip_frames_if_available",
                "stable_finger_link_pair_midpoint",
                "calibrated_finger_link_tip_proxy_from_distal_bbox_face_diagnostic_only",
                "explicit_fallback_proxy",
            ],
            "calibrated_proxy_close_authority": "diagnostic_only",
            "calibrated_vs_link_midpoint_offset_m": calibrated_vs_link_midpoint_offset_m,
            "calibrated_vs_link_midpoint_warning": calibrated_vs_link_midpoint_warning,
            "fallback_used": True,
            "fallback_source": "stable_finger_link_pair_midpoint",
            "diagnostic_finger_link_midpoint_bypass": bool(diagnostic_finger_link_midpoint_bypass),
            "diagnostic_finger_link_midpoint_bypass_effect": "comparison_only_no_override",
            "close_critical_reference": True,
            "diagnostic_two_finger_reference_comparison": comparison_payload,
            **fallback_comparison,
        }

    if calibrated_center_world is not None:
        comparison_log = _compare_with_diagnostic_link_midpoint(
            primary_world=calibrated_center_world,
            primary_log=calibrated_log,
        )
        return calibrated_center_world, {
            **calibrated_log,
            "actual_fingertip_pair_resolution_log": actual_log,
            "stable_finger_link_midpoint_resolution_log": link_midpoint_log,
            "grasp_center_definition": "diagnostic midpoint between calibrated distal fingertip proxies after actual and stable link midpoint failed",
            "resolution_priority": [
                "actual_fingertip_frames_if_available",
                "stable_finger_link_pair_midpoint",
                "calibrated_finger_link_tip_proxy_from_distal_bbox_face_diagnostic_only",
                "explicit_fallback_proxy",
            ],
            "fingertip_source": "calibrated_proxy_diagnostic_only",
            "calibrated_proxy_close_authority": "diagnostic_only",
            "calibrated_vs_link_midpoint_offset_m": calibrated_vs_link_midpoint_offset_m,
            "calibrated_vs_link_midpoint_warning": calibrated_vs_link_midpoint_warning,
            "fallback_used": True,
            "fallback_source": "calibrated_distal_proxy_pair_diagnostic_only",
            "diagnostic_finger_link_midpoint_bypass": bool(diagnostic_finger_link_midpoint_bypass),
            "diagnostic_finger_link_midpoint_bypass_effect": "comparison_only_no_override",
            "close_critical_reference": False,
            "close_critical_rejected_reason": "calibrated_distal_proxy_pair_not_runtime_validated",
            "diagnostic_two_finger_reference_comparison": comparison_payload,
            **comparison_log,
        }

    center_world = None
    log = calibrated_log
    if center_world is None and fallback_world is not None:
        fallback = np.array(fallback_world, dtype=float)
        if np.isfinite(fallback).all():
            fallback_comparison = _compare_with_diagnostic_link_midpoint(
                primary_world=fallback,
                primary_log={"source": "fallback_proxy", **(log if isinstance(log, dict) else {})},
            )
            return fallback, {
                **log,
                "source": "fallback_proxy",
                "fallback_source": fallback_source,
                "fallback_used": True,
                "fallback_world": fallback.tolist(),
                "grasp_center_definition": "explicit fallback proxy after actual fingertip, stable finger-link midpoint, and calibrated distal proxy resolution failed",
                "resolution_priority": [
                    "actual_fingertip_frames_if_available",
                    "stable_finger_link_pair_midpoint",
                    "calibrated_finger_link_tip_proxy_from_distal_bbox_face_diagnostic_only",
                    "explicit_fallback_proxy",
                ],
                "fingertip_source": "explicit_fallback_proxy",
                "calibrated_proxy_close_authority": "diagnostic_only",
                "calibrated_vs_link_midpoint_offset_m": calibrated_vs_link_midpoint_offset_m,
                "calibrated_vs_link_midpoint_warning": calibrated_vs_link_midpoint_warning,
                "close_critical_reference": False,
                "diagnostic_two_finger_reference_comparison": comparison_payload,
                **fallback_comparison,
            }
    fallback_fallback_comparison = _compare_with_diagnostic_link_midpoint(
        primary_world=center_world,
        primary_log={"source": log.get("source") if isinstance(log, dict) else None},
    )
    return center_world, {
        **log,
        "grasp_center_definition": "unresolved fingertip-end midpoint",
        "resolution_priority": [
            "actual_fingertip_frames_if_available",
            "stable_finger_link_pair_midpoint",
            "calibrated_finger_link_tip_proxy_from_distal_bbox_face_diagnostic_only",
            "explicit_fallback_proxy",
        ],
        "fingertip_source": None,
        "calibrated_proxy_close_authority": "diagnostic_only",
        "calibrated_vs_link_midpoint_offset_m": calibrated_vs_link_midpoint_offset_m,
        "calibrated_vs_link_midpoint_warning": calibrated_vs_link_midpoint_warning,
        "fallback_used": False,
        "diagnostic_finger_link_midpoint_bypass": bool(diagnostic_finger_link_midpoint_bypass),
        "diagnostic_finger_link_midpoint_bypass_effect": "comparison_only_no_override",
        "close_critical_reference": bool(center_world is not None),
        "fallback": None if center_world is not None else "close-critical evaluation falls back to point_B proxy",
        "diagnostic_two_finger_reference_comparison": comparison_payload,
        **fallback_fallback_comparison,
    }


def resolve_real_grasp_center_world(
    *,
    stage: Any,
    dc: Any,
    articulation: Any,
    robot_root_path: str,
    arm_side: str,
    fallback_world: np.ndarray | list[float] | None = None,
    fallback_source: str = "point_B_proxy",
    diagnostic_finger_link_midpoint_bypass: bool = False,
    include_diagnostic_comparison: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    return _resolve_real_grasp_center_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        fallback_world=fallback_world,
        fallback_source=fallback_source,
        diagnostic_finger_link_midpoint_bypass=diagnostic_finger_link_midpoint_bypass,
        include_diagnostic_comparison=include_diagnostic_comparison,
    )


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
        live_world, live_log = _resolve_vertical_finger_midpoint_reference_position(
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
        reference_world, reference_log = _resolve_vertical_finger_midpoint_reference_position(
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
            f"source_baseline={safe_payload.get('run_metadata', {}).get('source_baseline_script')}",
            f"table_frame_axis_aligned={safe_payload.get('table_frame', {}).get('axis_aligned_with_world_xy')}",
            f"table_unit_m={safe_payload.get('table_frame', {}).get('table_unit_m')}",
            f"perception_source={safe_payload.get('object_info', {}).get('perception_source')}",
            f"hybrid_selected_candidate={safe_payload.get('hybrid_phase1', {}).get('selected_candidate', {}).get('preset_id')}",
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


def _table_units(values_m: np.ndarray | list[float]) -> list[float]:
    values = np.array(values_m, dtype=float)
    return (values / TABLE_UNIT_M).tolist()


def _bbox_corners_from_bbox(box: dict[str, list[float]]) -> np.ndarray:
    min_v = np.array(box["min"], dtype=float)
    max_v = np.array(box["max"], dtype=float)
    return np.array(
        [
            [x, y, z]
            for x in (min_v[0], max_v[0])
            for y in (min_v[1], max_v[1])
            for z in (min_v[2], max_v[2])
        ],
        dtype=float,
    )


def _axis_aligned_with_world_xy(x_axis: np.ndarray, y_axis: np.ndarray) -> bool:
    world_x = np.array([1.0, 0.0, 0.0], dtype=float)
    world_y = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _normalize(np.array(x_axis, dtype=float), world_x)
    y_axis = _normalize(np.array(y_axis, dtype=float), world_y)
    return bool(
        max(abs(float(np.dot(x_axis, world_x))), abs(float(np.dot(x_axis, world_y)))) >= 0.999
        and max(abs(float(np.dot(y_axis, world_x))), abs(float(np.dot(y_axis, world_y)))) >= 0.999
    )


def _inspect_table_xform_axes(stage: Any, table_path: str) -> dict[str, Any]:
    log: dict[str, Any] = {
        "table_path": table_path,
        "xform_inspected": False,
        "usable_horizontal_axes": False,
        "source": "unavailable",
    }
    try:
        from pxr import Usd, UsdGeom  # type: ignore

        prim = stage.GetPrimAtPath(table_path)
        if not prim or not prim.IsValid():
            log["source"] = "missing_table_prim"
            return log
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        mat = np.array([[float(matrix[row][col]) for col in range(4)] for row in range(4)], dtype=float)
        axes = [mat[:3, index] for index in range(3)]
        horizontal_axes = []
        for axis in axes[:2]:
            axis_xy = np.array([float(axis[0]), float(axis[1]), 0.0], dtype=float)
            if np.isfinite(axis_xy).all() and float(np.linalg.norm(axis_xy)) > 1.0e-6:
                horizontal_axes.append(_normalize(axis_xy, np.array([1.0, 0.0, 0.0], dtype=float)))
        log.update(
            {
                "xform_inspected": True,
                "matrix": mat.tolist(),
                "raw_axes_world": [axis.tolist() for axis in axes],
                "horizontal_axes_world": [axis.tolist() for axis in horizontal_axes],
                "usable_horizontal_axes": bool(horizontal_axes),
                "source": "usd_xformable_local_to_world",
            }
        )
        return log
    except Exception as exc:
        log["source"] = "xform_inspection_failed"
        log["error"] = str(exc)
        return log


def build_or_resolve_table_frame(
    *,
    stage: Any,
    table_path: str,
    table_bbox: dict[str, list[float]],
    robot_base_position: np.ndarray,
    robot_base_yaw_rad: float,
) -> dict[str, Any]:
    """Resolve the robot-facing table frame used by the Phase 1 planner."""
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    robot_forward = _normalize(
        np.array([math.cos(robot_base_yaw_rad), math.sin(robot_base_yaw_rad), 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    robot_left = _normalize(
        np.array([-math.sin(robot_base_yaw_rad), math.cos(robot_base_yaw_rad), 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )
    robot_right = -robot_left
    xform_log = _inspect_table_xform_axes(stage, table_path)
    y_seed = robot_forward.copy()
    source = "robot_yaw_and_table_bbox"
    if xform_log.get("usable_horizontal_axes"):
        horizontal_axes = [np.array(axis, dtype=float) for axis in xform_log.get("horizontal_axes_world", [])]
        best = max(horizontal_axes, key=lambda axis: abs(float(np.dot(axis, robot_forward))))
        if abs(float(np.dot(best, robot_forward))) > 0.25:
            y_seed = best if float(np.dot(best, robot_forward)) >= 0.0 else -best
            source = "table_xform_axis_aligned_to_robot_forward"

    x_axis = _normalize(np.cross(y_seed, world_up), robot_right)
    if float(np.dot(x_axis, robot_right)) < 0.0:
        x_axis = -x_axis
    y_axis = _normalize(np.cross(world_up, x_axis), robot_forward)
    if float(np.dot(y_axis, robot_forward)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
    table_center = np.array(table_bbox["center"], dtype=float)
    robot_base = np.array(robot_base_position, dtype=float)
    if float(np.dot(robot_base, y_axis)) > float(np.dot(table_center, y_axis)):
        x_axis = -x_axis
        y_axis = -y_axis

    corners = _bbox_corners_from_bbox(table_bbox)
    x_proj = corners @ x_axis
    y_proj = corners @ y_axis
    x_min = float(np.min(x_proj))
    x_max = float(np.max(x_proj))
    y_min = float(np.min(y_proj))
    y_max = float(np.max(y_proj))
    table_top_z = float(table_bbox["max"][2])
    origin_world = x_axis * x_min + y_axis * y_min + world_up * table_top_z
    axis_aligned = _axis_aligned_with_world_xy(x_axis, y_axis)
    return {
        "frame_name": "task1_table_robot_facing",
        "source": source,
        "table_path": table_path,
        "origin_semantics": "near-left tabletop corner from the robot viewpoint",
        "axis_semantics": {
            "x_table": "near edge, robot-left toward robot-right",
            "y_table": "from robot toward far side of table",
            "z_table": "upward from tabletop surface",
        },
        "origin_world": origin_world.tolist(),
        "x_axis_world": x_axis.tolist(),
        "y_axis_world": y_axis.tolist(),
        "z_axis_world": world_up.tolist(),
        "table_unit_m": TABLE_UNIT_M,
        "x_extent_m": float(x_max - x_min),
        "y_extent_m": float(y_max - y_min),
        "x_extent_unit": float((x_max - x_min) / TABLE_UNIT_M),
        "y_extent_unit": float((y_max - y_min) / TABLE_UNIT_M),
        "table_top_z_world": table_top_z,
        "axis_aligned_with_world_xy": axis_aligned,
        "mapping_mode": "axis_aligned_simplified" if axis_aligned else "explicit_world_table_transform",
        "robot_forward_world": robot_forward.tolist(),
        "robot_left_world": robot_left.tolist(),
        "robot_right_world": robot_right.tolist(),
        "robot_projection_y_axis": float(np.dot(robot_base, y_axis)),
        "table_center_projection_y_axis": float(np.dot(table_center, y_axis)),
        "x_projection_range_world": [x_min, x_max],
        "y_projection_range_world": [y_min, y_max],
        "xform_inspection": xform_log,
    }


def world_to_table(point_world: np.ndarray | list[float], table_frame: dict[str, Any]) -> np.ndarray:
    point = np.array(point_world, dtype=float)
    origin = np.array(table_frame["origin_world"], dtype=float)
    delta = point - origin
    axes = [
        np.array(table_frame["x_axis_world"], dtype=float),
        np.array(table_frame["y_axis_world"], dtype=float),
        np.array(table_frame["z_axis_world"], dtype=float),
    ]
    return np.array([float(np.dot(delta, axis)) for axis in axes], dtype=float)


def table_to_world(point_table_m: np.ndarray | list[float], table_frame: dict[str, Any]) -> np.ndarray:
    point = np.array(point_table_m, dtype=float)
    origin = np.array(table_frame["origin_world"], dtype=float)
    return (
        origin
        + np.array(table_frame["x_axis_world"], dtype=float) * float(point[0])
        + np.array(table_frame["y_axis_world"], dtype=float) * float(point[1])
        + np.array(table_frame["z_axis_world"], dtype=float) * float(point[2])
    )


def _bbox_in_table_frame(box: dict[str, list[float]], table_frame: dict[str, Any]) -> dict[str, Any]:
    corners_world = _bbox_corners_from_bbox(box)
    corners_table = np.array([world_to_table(corner, table_frame) for corner in corners_world], dtype=float)
    min_table = np.min(corners_table, axis=0)
    max_table = np.max(corners_table, axis=0)
    center_table = 0.5 * (min_table + max_table)
    size_table = max_table - min_table
    return {
        "min": min_table.tolist(),
        "max": max_table.tolist(),
        "center": center_table.tolist(),
        "size": size_table.tolist(),
        "min_unit": _table_units(min_table),
        "max_unit": _table_units(max_table),
        "center_unit": _table_units(center_table),
        "size_unit": _table_units(size_table),
    }


def get_object_info_in_table_frame(
    *,
    stage: Any,
    target_path: str,
    target_index: int,
    target_category: dict[str, Any],
    table_frame: dict[str, Any],
) -> dict[str, Any]:
    state = _bbox_state(stage, target_path)
    bbox_world = state["bbox"]
    center_world = _center_from_bbox(bbox_world)
    center_table_m = world_to_table(center_world, table_frame)
    bbox_table = _bbox_in_table_frame(bbox_world, table_frame)
    size_table = np.array(bbox_table["size"], dtype=float)
    xy_sizes = np.sort(np.abs(size_table[:2]))
    class_name = str(target_category.get("inferred_category") or target_category.get("category_from_scene_builder_order") or "unknown")
    return {
        "id": f"target_{int(target_index)}",
        "prim_path": target_path,
        "target_index": int(target_index),
        "class_name": class_name,
        "center_world": center_world.tolist(),
        "center_table_m": center_table_m.tolist(),
        "center_table_unit": _table_units(center_table_m),
        "yaw_table": 0.0,
        "yaw_table_source": "phase1_bbox_axis_aligned_fallback",
        "approx_width_m": float(xy_sizes[0]),
        "approx_length_m": float(xy_sizes[1]),
        "approx_height_m": float(abs(size_table[2])),
        "bbox_world": bbox_world,
        "bbox_table_m": {
            "min": bbox_table["min"],
            "max": bbox_table["max"],
            "center": bbox_table["center"],
            "size": bbox_table["size"],
        },
        "bbox_table_unit": {
            "min": bbox_table["min_unit"],
            "max": bbox_table["max_unit"],
            "center": bbox_table["center_unit"],
            "size": bbox_table["size_unit"],
        },
        "perception_source": "scene_state",
        "perception_confidence": 1.0,
        "category": target_category,
    }


def _candidate_arm_list(requested_arm: str) -> list[str]:
    if requested_arm in ("left", "right"):
        return [requested_arm]
    return ["left", "right"]


def _preferred_arm_from_components(target_components: dict[str, Any]) -> str:
    return "left" if float(target_components.get("lateral_base", 0.0)) > 0.0 else "right"


def _candidate_base_yaw_from_grasp_frame(
    object_info: dict[str, Any],
    object_grasp_frame: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    if isinstance(object_grasp_frame, dict):
        try:
            axis = np.array(object_grasp_frame.get("closing_axis_table", []), dtype=float).reshape(-1)
        except Exception:
            axis = np.array([], dtype=float)
        if axis.size >= 2 and np.isfinite(axis[:2]).all():
            axis_xy = axis[:2]
            axis_norm = float(np.linalg.norm(axis_xy))
            if axis_norm > 1.0e-9:
                normalized_axis = axis_xy / axis_norm
                yaw = float(math.atan2(float(normalized_axis[1]), float(normalized_axis[0])))
                return yaw, {
                    "base_yaw_source": "object_grasp_frame_closing_axis_table",
                    "closing_axis_table": axis[:3].tolist() if axis.size >= 3 else axis.tolist(),
                    "normalized_closing_axis_xy": normalized_axis.tolist(),
                    "base_yaw_rad": yaw,
                    "base_yaw_deg": math.degrees(yaw),
                    "fallback_used": False,
                }

    yaw = float(object_info.get("yaw_table", 0.0))
    return yaw, {
        "base_yaw_source": str(object_info.get("yaw_table_source", "object_info_yaw_table_fallback")),
        "base_yaw_rad": yaw,
        "base_yaw_deg": math.degrees(yaw),
        "fallback_used": True,
        "fallback_reason": "object_grasp_frame_closing_axis_table_unavailable_or_invalid",
    }


def generate_approach_candidates_for_object(
    *,
    object_info: dict[str, Any],
    object_grasp_frame: dict[str, Any] | None,
    table_frame: dict[str, Any],
    target_components: dict[str, Any],
    approach_family_order: list[str],
    requested_arm: str,
    robot_base_position: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    center_table = np.array(object_info["center_table_m"], dtype=float)
    bbox_table = object_info["bbox_table_m"]
    grasp_center_table = center_table.copy()
    grasp_center_world = table_to_world(grasp_center_table, table_frame)
    approach_mode = str(approach_family_order[0]) if approach_family_order else "z_approach"
    preferred_arm = _preferred_arm_from_components(target_components)
    base_hand_yaw, yaw_generation_log = _candidate_base_yaw_from_grasp_frame(object_info, object_grasp_frame)
    candidates: list[dict[str, Any]] = []
    for arm in _candidate_arm_list(requested_arm):
        for yaw_index, yaw_offset_deg in enumerate(HYBRID_PHASE1_PRESET_YAW_DEG):
            hand_yaw = float(base_hand_yaw) + math.radians(float(yaw_offset_deg))
            pregrasp_table = center_table.copy()
            pregrasp_table[2] = max(float(bbox_table["max"][2]) + float(args.pregrasp_clearance), float(args.pregrasp_clearance))
            if approach_mode == "world_y_approach":
                pregrasp_table[1] = max(0.0, float(pregrasp_table[1]) - float(args.pregrasp_standoff))
            pregrasp_world = table_to_world(pregrasp_table, table_frame)
            preset_id = f"phase1_{approach_mode}_{arm}_yaw_{yaw_offset_deg:+.0f}"
            motion_cost = float(np.linalg.norm(pregrasp_world - np.array(robot_base_position, dtype=float)))
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "object_id": object_info["id"],
                    "arm": arm,
                    "preferred_arm": preferred_arm,
                    "preset_id": preset_id,
                    "preset_yaw_offset_deg": float(yaw_offset_deg),
                    "approach_mode": approach_mode,
                    "pregrasp_world": pregrasp_world.tolist(),
                    "pregrasp_table_m": pregrasp_table.tolist(),
                    "pregrasp_table_unit": _table_units(pregrasp_table),
                    "object_grasp_center_world": grasp_center_world.tolist(),
                    "object_grasp_center_table_m": grasp_center_table.tolist(),
                    "object_grasp_center_table_unit": _table_units(grasp_center_table),
                    "hand_yaw": hand_yaw,
                    "hand_yaw_deg": math.degrees(hand_yaw),
                    "hand_yaw_base": float(base_hand_yaw),
                    "hand_yaw_base_deg": math.degrees(float(base_hand_yaw)),
                    "hand_yaw_base_source": yaw_generation_log["base_yaw_source"],
                    "hand_yaw_generation": yaw_generation_log,
                    "hand_pitch": math.pi,
                    "hand_roll": 0.0,
                    "motion_cost": motion_cost,
                    "motion_cost_proxy": "euclidean_robot_base_to_pregrasp_world",
                    "workspace_check": _workspace_check(pregrasp_world, args.workspace_x, args.workspace_y, args.workspace_z),
                    "coarse_width_sanity_range_m": [HYBRID_PHASE1_BASIC_WIDTH_MIN_M, HYBRID_PHASE1_BASIC_WIDTH_MAX_M],
                    "coarse_width_sanity_ok": bool(
                        HYBRID_PHASE1_BASIC_WIDTH_MIN_M <= float(object_info["approx_width_m"]) <= HYBRID_PHASE1_BASIC_WIDTH_MAX_M
                    ),
                    "coarse_symmetry_delta_m": float(abs(float(object_info["approx_length_m"]) - float(object_info["approx_width_m"]))),
                    "coarse_symmetry_tolerance_m": HYBRID_PHASE1_BASIC_SYMMETRY_TOL_M,
                    "coarse_symmetry_sanity_ok": bool(
                        abs(float(object_info["approx_length_m"]) - float(object_info["approx_width_m"])) <= HYBRID_PHASE1_BASIC_SYMMETRY_TOL_M
                    ),
                    "score": None,
                    "valid": None,
                }
            )
    return candidates


def fast_score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    workspace_ok = bool(candidate.get("workspace_check", {}).get("workspace_ok", False))
    width_ok = bool(candidate.get("coarse_width_sanity_ok", False))
    arm_penalty = 0.0 if candidate.get("arm") == candidate.get("preferred_arm") else HYBRID_PHASE1_SCORE_SIDE_PENALTY
    yaw_penalty = abs(float(candidate.get("preset_yaw_offset_deg", 0.0))) * HYBRID_PHASE1_SCORE_YAW_WEIGHT
    width_penalty = 0.0 if width_ok else HYBRID_PHASE1_SCORE_WIDTH_PENALTY
    workspace_penalty = 0.0 if workspace_ok else HYBRID_PHASE1_SCORE_WIDTH_PENALTY
    score = (
        HYBRID_PHASE1_SCORE_REACH_WEIGHT * float(candidate.get("motion_cost", math.inf))
        + arm_penalty
        + yaw_penalty
        + width_penalty
        + workspace_penalty
    )
    scored = dict(candidate)
    scored.update(
        {
            "score": float(score),
            "valid": bool(workspace_ok and width_ok and math.isfinite(score)),
            "score_terms": {
                "motion_cost": float(candidate.get("motion_cost", math.inf)),
                "arm_side_penalty": arm_penalty,
                "yaw_penalty": yaw_penalty,
                "width_penalty": width_penalty,
                "workspace_penalty": workspace_penalty,
            },
            "invalid_reason": None
            if workspace_ok and width_ok
            else ",".join(
                reason
                for reason, active in (
                    ("pregrasp_outside_workspace", not workspace_ok),
                    ("coarse_width_sanity_failed", not width_ok),
                )
                if active
            ),
        }
    )
    return scored


def select_best_candidate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    scored = [fast_score_candidate(candidate) for candidate in candidates]
    arm_order = {"right": 0, "left": 1}
    ranked = sorted(
        scored,
        key=lambda item: (
            not bool(item.get("valid", False)),
            float(item.get("score", math.inf)),
            str(item.get("approach_mode", "")),
            int(arm_order.get(str(item.get("arm")), 99)),
            int(item.get("candidate_index", 0)),
        ),
    )
    valid = [candidate for candidate in ranked if bool(candidate.get("valid", False))]
    selected = valid[0] if valid else None
    return selected, {
        "selection_policy": "phase1_deterministic_score_sort",
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid),
        "score_weights": {
            "reach": HYBRID_PHASE1_SCORE_REACH_WEIGHT,
            "arm_side_penalty": HYBRID_PHASE1_SCORE_SIDE_PENALTY,
            "yaw": HYBRID_PHASE1_SCORE_YAW_WEIGHT,
            "width_penalty": HYBRID_PHASE1_SCORE_WIDTH_PENALTY,
        },
        "candidate_scores": scored,
        "ranked_candidate_ids": [candidate.get("preset_id") for candidate in ranked],
        "selected_candidate": selected,
        "failure_reason": None if selected is not None else "no_valid_phase1_candidate",
    }


def _clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.array(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return vec
    return vec * (float(max_norm) / norm)


def _angle_between_axes_unsigned(a: np.ndarray, b: np.ndarray) -> float:
    axis_a = _normalize(np.array(a, dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))
    axis_b = _normalize(np.array(b, dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))
    return float(math.acos(max(-1.0, min(1.0, abs(float(np.dot(axis_a, axis_b)))))))


def _table_axis_to_world(axis_table: np.ndarray, table_frame: dict[str, Any]) -> np.ndarray:
    axis = np.array(axis_table, dtype=float)
    world_axis = (
        np.array(table_frame["x_axis_world"], dtype=float) * float(axis[0])
        + np.array(table_frame["y_axis_world"], dtype=float) * float(axis[1])
        + np.array(table_frame["z_axis_world"], dtype=float) * float(axis[2])
    )
    return _normalize(world_axis, np.array([1.0, 0.0, 0.0], dtype=float))


def estimate_object_grasp_frame(object_info: dict[str, Any], table_frame: dict[str, Any]) -> dict[str, Any]:
    """Estimate a cheap table-local grasp frame from scene_state bbox geometry."""
    bbox_table = object_info.get("bbox_table_m", {})
    center_table = np.array(object_info["center_table_m"], dtype=float)
    center_world = np.array(object_info["center_world"], dtype=float)
    size = np.abs(np.array(bbox_table.get("size", [object_info["approx_width_m"], object_info["approx_length_m"], object_info["approx_height_m"]]), dtype=float))
    if size.shape[0] < 3 or not np.isfinite(size).all():
        size = np.array([object_info["approx_width_m"], object_info["approx_length_m"], object_info["approx_height_m"]], dtype=float)

    if float(size[0]) <= float(size[1]):
        closing_axis_table = np.array([1.0, 0.0, 0.0], dtype=float)
        lateral_axis_table = np.array([0.0, 1.0, 0.0], dtype=float)
        width_on_closing_axis_m = float(size[0])
        length_on_lateral_axis_m = float(size[1])
        axis_source = "bbox_table_x_short_axis"
    else:
        closing_axis_table = np.array([0.0, 1.0, 0.0], dtype=float)
        lateral_axis_table = np.array([1.0, 0.0, 0.0], dtype=float)
        width_on_closing_axis_m = float(size[1])
        length_on_lateral_axis_m = float(size[0])
        axis_source = "bbox_table_y_short_axis"

    vertical_axis_table = np.array([0.0, 0.0, 1.0], dtype=float)
    closing_axis_world = _table_axis_to_world(closing_axis_table, table_frame)
    lateral_axis_world = _table_axis_to_world(lateral_axis_table, table_frame)
    vertical_axis_world = _table_axis_to_world(vertical_axis_table, table_frame)
    half_width = 0.5 * width_on_closing_axis_m
    bbox_min = np.array(bbox_table.get("min", center_table - 0.5 * size), dtype=float)
    bbox_max = np.array(bbox_table.get("max", center_table + 0.5 * size), dtype=float)
    bottom_z = float(bbox_min[2])
    top_z = float(bbox_max[2])
    return {
        "source": "scene_state_bbox_table_frame",
        "axis_source": axis_source,
        "object_id": object_info.get("id"),
        "class_name": object_info.get("class_name"),
        "grasp_center_world": center_world.tolist(),
        "grasp_center_table_m": center_table.tolist(),
        "grasp_center_table_unit": _table_units(center_table),
        "closing_axis_table": closing_axis_table.tolist(),
        "lateral_axis_table": lateral_axis_table.tolist(),
        "vertical_axis_table": vertical_axis_table.tolist(),
        "closing_axis_world": closing_axis_world.tolist(),
        "lateral_axis_world": lateral_axis_world.tolist(),
        "vertical_axis_world": vertical_axis_world.tolist(),
        "width_on_closing_axis_m": width_on_closing_axis_m,
        "length_on_lateral_axis_m": length_on_lateral_axis_m,
        "height_m": float(size[2]),
        "bbox_table_m": bbox_table,
        "bbox_table_unit": object_info.get("bbox_table_unit"),
        "object_bottom_table_z_m": bottom_z,
        "object_top_table_z_m": top_z,
        "nominal_contact_points_world": [
            (center_world - closing_axis_world * half_width).tolist(),
            (center_world + closing_axis_world * half_width).tolist(),
        ],
        "notes": [
            "Phase 2 uses scene_state bbox axes only; OBB/yaw refinement is deferred.",
            "Closing axis is the shorter horizontal table-frame bbox axis.",
        ],
    }


def _candidate_hand_closing_axis_table(candidate: dict[str, Any]) -> np.ndarray:
    yaw = float(candidate.get("hand_yaw", 0.0))
    return _normalize(np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float), np.array([1.0, 0.0, 0.0], dtype=float))


def predict_early_contact_asymmetry(candidate: dict[str, Any], object_grasp_frame: dict[str, Any]) -> dict[str, Any]:
    hand_axis_table = _candidate_hand_closing_axis_table(candidate)
    closing_axis_table = np.array(object_grasp_frame["closing_axis_table"], dtype=float)
    lateral_axis_table = np.array(object_grasp_frame["lateral_axis_table"], dtype=float)
    candidate_center = np.array(candidate.get("object_grasp_center_table_m", object_grasp_frame["grasp_center_table_m"]), dtype=float)
    object_center = np.array(object_grasp_frame["grasp_center_table_m"], dtype=float)
    center_delta = candidate_center - object_center
    alignment_error = _angle_between_axes_unsigned(hand_axis_table, closing_axis_table)
    lateral_offset = abs(float(np.dot(center_delta, lateral_axis_table)))
    vertical_offset = abs(float(center_delta[2]))
    width = float(object_grasp_frame["width_on_closing_axis_m"])
    skew_component = 0.5 * width * abs(math.sin(alignment_error))
    predicted_contact_asymmetry = float(lateral_offset + skew_component)
    return {
        "hand_closing_axis_table": hand_axis_table.tolist(),
        "object_closing_axis_table": closing_axis_table.tolist(),
        "alignment_error_rad": alignment_error,
        "alignment_error_deg": math.degrees(alignment_error),
        "center_delta_table_m": center_delta.tolist(),
        "lateral_offset_m": lateral_offset,
        "vertical_offset_m": vertical_offset,
        "skew_component_m": skew_component,
        "predicted_contact_asymmetry_m": predicted_contact_asymmetry,
        "prediction_source": "bbox_short_axis_vs_candidate_hand_yaw",
    }


def estimate_table_clearance_margin(candidate: dict[str, Any], object_grasp_frame: dict[str, Any]) -> dict[str, Any]:
    candidate_center = np.array(candidate.get("object_grasp_center_table_m", object_grasp_frame["grasp_center_table_m"]), dtype=float)
    bottom_z = float(object_grasp_frame.get("object_bottom_table_z_m", 0.0))
    height = float(object_grasp_frame.get("height_m", 0.0))
    center_based_margin = float(candidate_center[2] - 0.5 * height)
    margin = min(bottom_z, center_based_margin)
    return {
        "table_clearance_margin_m": margin,
        "object_bottom_table_z_m": bottom_z,
        "center_based_bottom_margin_m": center_based_margin,
        "candidate_grasp_center_table_z_m": float(candidate_center[2]),
        "estimated_height_m": height,
    }


def compute_target_gap(object_grasp_frame: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    width = float(object_grasp_frame.get("width_on_closing_axis_m", math.inf))
    margin = float(getattr(args, "phase2_target_gap_margin", PHASE2_TARGET_GAP_MARGIN_M)) if args is not None else PHASE2_TARGET_GAP_MARGIN_M
    min_gap = float(getattr(args, "phase2_min_target_gap", PHASE2_MIN_TARGET_GAP_M)) if args is not None else PHASE2_MIN_TARGET_GAP_M
    target_gap = max(min_gap, width - 2.0 * margin)
    stage_a_fraction = float(getattr(args, "phase2_close_stage_a_fraction", PHASE2_CLOSE_STAGE_A_FRACTION)) if args is not None else PHASE2_CLOSE_STAGE_A_FRACTION
    stage_a_joint_target = OFFICIAL_GRIPPER_OPEN_WIDTH + (OFFICIAL_GRIPPER_CLOSE_WIDTH - OFFICIAL_GRIPPER_OPEN_WIDTH) * stage_a_fraction
    return {
        "estimated_width_on_closing_axis_m": width,
        "target_gap_m": float(target_gap),
        "target_gap_margin_m": margin,
        "min_target_gap_m": min_gap,
        "stage_a_fraction": stage_a_fraction,
        "stage_a_joint_target": float(stage_a_joint_target),
        "stage_b_joint_target": float(OFFICIAL_GRIPPER_CLOSE_WIDTH),
        "joint_target_semantics": "official same-sign finger joint positions; meter gap is logged as geometry advisory",
    }


def fast_geometric_grasp_filter(candidate: dict[str, Any], object_grasp_frame: dict[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    asymmetry = predict_early_contact_asymmetry(candidate, object_grasp_frame)
    clearance = estimate_table_clearance_margin(candidate, object_grasp_frame)
    width = float(object_grasp_frame["width_on_closing_axis_m"])
    alignment_threshold = float(getattr(args, "phase2_alignment_error_max", PHASE2_ALIGNMENT_ERROR_MAX_RAD)) if args is not None else PHASE2_ALIGNMENT_ERROR_MAX_RAD
    symmetry_threshold = float(getattr(args, "phase2_symmetry_error_max", PHASE2_SYMMETRY_ERROR_MAX_M)) if args is not None else PHASE2_SYMMETRY_ERROR_MAX_M
    asymmetry_threshold = float(getattr(args, "phase2_contact_asymmetry_max", PHASE2_CONTACT_ASYMMETRY_MAX_M)) if args is not None else PHASE2_CONTACT_ASYMMETRY_MAX_M
    clearance_threshold = float(getattr(args, "phase2_table_clearance_min", PHASE2_TABLE_CLEARANCE_MIN_M)) if args is not None else PHASE2_TABLE_CLEARANCE_MIN_M
    width_min = float(getattr(args, "phase2_width_min", PHASE2_WIDTH_MIN_M)) if args is not None else PHASE2_WIDTH_MIN_M
    width_max = float(getattr(args, "phase2_width_max", PHASE2_WIDTH_MAX_M)) if args is not None else PHASE2_WIDTH_MAX_M
    symmetry_error = float(asymmetry["lateral_offset_m"])
    pass_flags = {
        "alignment_pass": bool(float(asymmetry["alignment_error_rad"]) <= alignment_threshold),
        "width_pass": bool(width_min <= width <= width_max),
        "lateral_symmetry_pass": bool(symmetry_error <= symmetry_threshold),
        "table_clearance_pass": bool(float(clearance["table_clearance_margin_m"]) >= clearance_threshold),
        "early_contact_asymmetry_pass": bool(float(asymmetry["predicted_contact_asymmetry_m"]) <= asymmetry_threshold),
    }
    mandatory_pass = bool(all(pass_flags.values()))
    violation_score = (
        max(0.0, float(asymmetry["alignment_error_rad"]) - alignment_threshold)
        + max(0.0, symmetry_error - symmetry_threshold)
        + max(0.0, float(asymmetry["predicted_contact_asymmetry_m"]) - asymmetry_threshold)
        + max(0.0, clearance_threshold - float(clearance["table_clearance_margin_m"]))
        + max(0.0, width_min - width)
        + max(0.0, width - width_max)
    )
    filtered = dict(candidate)
    filter_log = {
        "filter_name": "phase2_fast_vector_geometric_filter",
        "mandatory_pass": mandatory_pass,
        "pass_flags": pass_flags,
        "alignment_error": float(asymmetry["alignment_error_rad"]),
        "alignment_error_rad": float(asymmetry["alignment_error_rad"]),
        "alignment_error_deg": float(asymmetry["alignment_error_deg"]),
        "symmetry_error": symmetry_error,
        "symmetry_error_m": symmetry_error,
        "predicted_contact_asymmetry": float(asymmetry["predicted_contact_asymmetry_m"]),
        "predicted_contact_asymmetry_m": float(asymmetry["predicted_contact_asymmetry_m"]),
        "width_compatibility": {
            "width_on_closing_axis_m": width,
            "min_width_m": width_min,
            "max_width_m": width_max,
            "pass": pass_flags["width_pass"],
        },
        "table_clearance_margin": float(clearance["table_clearance_margin_m"]),
        "table_clearance_margin_m": float(clearance["table_clearance_margin_m"]),
        "thresholds": {
            "alignment_error_max_rad": alignment_threshold,
            "symmetry_error_max_m": symmetry_threshold,
            "contact_asymmetry_max_m": asymmetry_threshold,
            "table_clearance_min_m": clearance_threshold,
            "width_min_m": width_min,
            "width_max_m": width_max,
        },
        "contact_asymmetry_prediction": asymmetry,
        "table_clearance": clearance,
        "violation_score": float(violation_score),
        "filter_scope": "cheap_vector_filter_only_no_force_closure_solver",
    }
    filtered["geometric_filter"] = filter_log
    filtered["phase2_filter_pass"] = mandatory_pass
    filtered["phase2_geometric_score"] = float(filtered.get("score", math.inf)) + PHASE2_GEOMETRIC_VIOLATION_WEIGHT * float(violation_score)
    return filtered


def select_best_phase2_candidate(
    candidates: list[dict[str, Any]],
    object_grasp_frame: dict[str, Any],
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    phase1_selected, phase1_selection = select_best_candidate(candidates)
    scored_by_id = {str(candidate.get("preset_id")): candidate for candidate in phase1_selection["candidate_scores"]}
    arm_order = {"right": 0, "left": 1}
    ranked = sorted(
        phase1_selection["candidate_scores"],
        key=lambda item: (
            not bool(item.get("valid", False)),
            float(item.get("score", math.inf)),
            str(item.get("approach_mode", "")),
            int(arm_order.get(str(item.get("arm")), 99)),
            int(item.get("candidate_index", 0)),
        ),
    )
    filtered_ranked = [fast_geometric_grasp_filter(scored_by_id[str(candidate.get("preset_id"))], object_grasp_frame, args) for candidate in ranked]
    passing = [candidate for candidate in filtered_ranked if bool(candidate.get("valid", False)) and bool(candidate.get("phase2_filter_pass", False))]
    selected = passing[0] if passing else None
    selection_warning = None
    allow_least_bad = bool(getattr(args, "phase2_allow_least_bad_candidate", PHASE2_ALLOW_LEAST_BAD_CANDIDATE))
    if selected is None and allow_least_bad and filtered_ranked:
        valid_or_all = [candidate for candidate in filtered_ranked if bool(candidate.get("valid", False))] or filtered_ranked
        selected = min(
            valid_or_all,
            key=lambda item: (
                float(item.get("geometric_filter", {}).get("violation_score", math.inf)),
                float(item.get("score", math.inf)),
                int(item.get("candidate_index", 0)),
            ),
        )
        selection_warning = "no_candidate_passed_all_phase2_mandatory_checks_least_bad_selected"
        selected["phase2_filter_warning"] = selection_warning
    if selected is None and filtered_ranked and not allow_least_bad:
        selection_warning = "no_candidate_passed_all_phase2_mandatory_checks_least_bad_disabled"
    return selected, {
        "selection_policy": "phase2_score_sort_then_fast_geometric_filter_first_pass_no_least_bad_by_default",
        "least_bad_candidate_allowed": allow_least_bad,
        "phase1_selection": phase1_selection,
        "phase1_selected_candidate": phase1_selected,
        "candidate_scores": phase1_selection["candidate_scores"],
        "candidate_count": len(candidates),
        "valid_candidate_count": int(sum(1 for candidate in filtered_ranked if bool(candidate.get("valid", False)))),
        "geometric_pass_candidate_count": len(passing),
        "filtered_candidates": filtered_ranked,
        "ranked_candidate_ids": [candidate.get("preset_id") for candidate in filtered_ranked],
        "selected_candidate": selected,
        "selection_warning": selection_warning,
        "failure_reason": None
        if selected is not None
        else (
            "no_phase2_candidate_passed_mandatory_geometric_filter"
            if filtered_ranked
            else "no_valid_phase2_candidate"
        ),
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


def _upsert_debug_marker(
    stage: Any,
    path: str,
    position: np.ndarray | list[float],
    radius: float,
    color: tuple[float, float, float],
) -> str:
    from pxr import Gf, UsdGeom  # type: ignore

    point = np.array(position, dtype=float)
    prim = stage.GetPrimAtPath(path)
    if prim is not None and prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        translate_updated = False
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                translate_updated = True
                break
        if not translate_updated:
            xform.AddTranslateOp().Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        return str(path)
    return _create_debug_marker(stage, path, point, radius, color)


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
        object_top_z_world = float(bbox["max"][2])
        far_xy_align_clearance = float(args.far_xy_align_clearance_above_object)
        far_xy_align_z_world = object_top_z_world + far_xy_align_clearance
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
            "object_top_z_world": object_top_z_world,
            "align_clearance_m": far_xy_align_clearance,
            "align_height": float(far_xy_align_b_world[2]),
            "far_xy_align_clearance_reference": "object_top_z + far_xy_align_clearance_above_object",
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
        "point_B_proxy_definition": "legacy point A plus selected local point_b_offset; logged separately from the close-critical fingertip-end midpoint",
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
    counter: dict[str, Any],
    phase_log: list[dict[str, Any]],
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    target_pose_fn: Callable[[], np.ndarray] | None = None,
    per_tick_monitor_fn: Callable[[], None] | None = None,
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    ik_overrides: dict[str, Any] | None = None,
    position_metric_offset_local: np.ndarray | None = None,
    position_metric_label: str | None = None,
    completion_condition_fn: Callable[[], bool] | None = None,
    completion_condition_label: str | None = None,
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    point_b_offset_for_log = getattr(args, "point_b_offset_local_resolved", None)
    point_b_offset_for_log = None if point_b_offset_for_log is None else np.array(point_b_offset_for_log, dtype=float)
    point_b_metric_active = bool(point_b_offset_for_log is not None)
    coord_refresh_samples: list[dict[str, Any]] = []
    coord_refresh_failures = 0
    target_pose_eval_count = 0
    point_b_offset_for_log = getattr(args, "point_b_offset_local_resolved", None)
    point_b_offset_for_log = None if point_b_offset_for_log is None else np.array(point_b_offset_for_log, dtype=float)

    metric_offset_for_stop = None
    if position_metric_offset_local is not None:
        metric_offset_for_stop = np.array(position_metric_offset_local, dtype=float)
    elif point_b_offset_for_log is not None:
        metric_offset_for_stop = point_b_offset_for_log.copy()

    metric_label = position_metric_label or (
        "contact_reference_world" if position_metric_offset_local is not None
        else ("point_B_world" if point_b_offset_for_log is not None else "ee_pose_base")
    )
    ik_refresh_active = bool(getattr(args, "ik_refresh_enable", DEFAULT_IK_REFRESH_ENABLE))
    ik_refresh_period = max(1, int(getattr(args, "ik_refresh_period", DEFAULT_IK_REFRESH_PERIOD)))
    ik_refresh_drift_threshold = float(getattr(args, "ik_refresh_drift_threshold", DEFAULT_IK_REFRESH_DRIFT_THRESHOLD))
    drift_refresh_active = bool(ik_refresh_active and ik_refresh_drift_threshold > 0.0 and target_pose_fn is not None)
    ik_refresh_events: list[dict[str, Any]] = []
    target_drift_checks = 0
    target_drift_samples: list[dict[str, Any]] = []
    q_goal_update_count = 0
    q_goal: np.ndarray | None = None
    q_goal_target_pose: np.ndarray | None = None
    q_goal_ik_ok: bool | None = None
    last_ik_refresh_tick: int | None = None
    latest_ik_refresh_reason: str | None = None
    early_stop_details: dict[str, Any] | None = None
    completion_condition_error: str | None = None
    completion_condition_last_value = True
    completion_condition_blocked_normal_stop_count = 0

    def evaluate_target_pose() -> np.ndarray:
        nonlocal target_pose_eval_count, early_stop_details
        target_pose_eval_count += 1
        if target_pose_fn is not None:
            try:
                return np.array(target_pose_fn(), dtype=float)
            except ServoEarlyStop as exc:
                if early_stop_details is None:
                    early_stop_details = {"reason": exc.reason, **dict(exc.details)}
                try:
                    return np.array(_current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args), dtype=float)
                except Exception:
                    return np.array(spec.target_pose_base, dtype=float)
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

    def position_metric_errors(current_pose_base, target_pose_base):
        ee_pos_err, _ = _pose_error(ik_solver, current_pose_base, target_pose_base)
        if metric_offset_for_stop is None:
            return ee_pos_err, ee_pos_err, None
        current_metric_world = _point_world_from_pose(coord_transform, current_pose_base, metric_offset_for_stop)
        target_metric_world = _point_world_from_pose(coord_transform, target_pose_base, metric_offset_for_stop)
        metric_err = float(np.linalg.norm(current_metric_world - target_metric_world))
        return metric_err, ee_pos_err, metric_err

    def evaluate_completion_condition() -> bool:
        nonlocal completion_condition_error, completion_condition_last_value
        if completion_condition_fn is None:
            completion_condition_last_value = True
            return True
        try:
            completion_condition_last_value = bool(completion_condition_fn())
        except Exception as exc:
            completion_condition_error = repr(exc)
            completion_condition_last_value = False
        return bool(completion_condition_last_value)

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
        if early_stop_details is not None:
            last_ik_refresh_tick = int(tick)
            latest_ik_refresh_reason = f"{reason}_early_stop"
            ik_refresh_events.append(
                {
                    "tick": int(tick),
                    "reason": str(reason),
                    "dualarmik_success": None,
                    "q_goal_update_index": int(q_goal_update_count),
                    "target_pose_base": target_pose.tolist(),
                    "skipped_by_early_stop": True,
                    "early_stop_details": early_stop_details,
                }
            )
            return target_pose
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

    if ik_refresh_active and early_stop_details is None:
        refresh_ik_goal(0, "initial", refresh_coord=False)

    for tick in range(1, int(spec.max_ticks) + 1):
        if early_stop_details is not None:
            break
        if per_tick_monitor_fn is not None:
            try:
                per_tick_monitor_fn()
            except ServoEarlyStop as exc:
                if early_stop_details is None:
                    early_stop_details = {"reason": exc.reason, **dict(exc.details)}
                final_target = np.array(_current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args), dtype=float)
                break
        if ik_refresh_active:
            refresh_reason: str | None = None
            drift_metrics: dict[str, Any] | None = None
            if q_goal is None or q_goal_target_pose is None or last_ik_refresh_tick is None:
                refresh_reason = "initial_missing_goal"
            elif tick - int(last_ik_refresh_tick) >= ik_refresh_period:
                refresh_reason = "period"
            elif drift_refresh_active:
                live_target = evaluate_target_pose()
                if early_stop_details is not None:
                    final_target = np.array(live_target, dtype=float)
                    break
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
                if early_stop_details is not None:
                    final_target = np.array(target_pose, dtype=float)
                    break
            else:
                target_pose = np.array(q_goal_target_pose, dtype=float)
        else:
            maybe_refresh_coord_transform(tick, "servo_tick_before_target")
            target_pose = evaluate_target_pose()
            if early_stop_details is not None:
                final_target = np.array(target_pose, dtype=float)
                break
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
        if pos_err <= float(spec.pos_tolerance) and rot_err <= float(spec.rot_tolerance):
            if evaluate_completion_condition():
                break
            completion_condition_blocked_normal_stop_count += 1

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
                "position_error_metric": metric_label,
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
    if (
        early_stop_details is None
        and completion_condition_fn is None
        and best_pos_error < final_pos_error
        and best_pos_error <= float(spec.pos_tolerance)
    ):
        _send_position_targets(dc, arm_dofs, [float(value) for value in best_joint_positions])
        _run_updates(sim_app, args.ik_settle_steps, counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=spec.gripper_effort)
        maybe_refresh_coord_transform(int(spec.max_ticks), "after_best_pose_restore", force_sample=True)
        final_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        final_pos_error, final_ee_pos_error, final_point_b_error = position_metric_errors(final_pose, final_target)
        _, final_rot_error = _pose_error(ik_solver, final_pose, final_target)

    final_completion_condition_met = True if early_stop_details is not None else evaluate_completion_condition()
    success = bool(
        early_stop_details is not None
        or (
            final_pos_error <= float(spec.pos_tolerance)
            and final_rot_error <= float(spec.rot_tolerance)
            and final_completion_condition_met
        )
    )
    if not success:
        failure_reason = "tolerance_not_met"
        if completion_condition_fn is not None and not final_completion_condition_met:
            failure_reason = "completion_condition_not_met"
        if completion_condition_error is not None:
            failure_reason = "completion_condition_error"
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
        "per_tick_monitor_active": bool(per_tick_monitor_fn is not None),
        "servo_early_stop_triggered": bool(early_stop_details is not None),
        "servo_early_stop_reason": None if early_stop_details is None else early_stop_details.get("reason"),
        "servo_early_stop_details": early_stop_details,
        "completion_condition_required": bool(completion_condition_fn is not None),
        "completion_condition_label": completion_condition_label,
        "completion_condition_met": bool(final_completion_condition_met),
        "completion_condition_last_value": bool(completion_condition_last_value),
        "completion_condition_blocked_normal_stop_count": int(completion_condition_blocked_normal_stop_count),
        "completion_condition_error": completion_condition_error,
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
    object_grasp_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = _bbox_state(stage, target_path)
    object_center_world = _center_from_bbox(state["bbox"])
    current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    rot = _euler_xyz_to_rot(float(current_pose[3]), float(current_pose[4]), float(current_pose[5]))
    tcp_base = current_pose[:3] + rot @ np.array(tcp_offset_local, dtype=float)
    tcp_world = np.array(coord_transform.robot_to_world(tcp_base), dtype=float)
    point_b_offset = np.array(geometry.get("point_b_offset_local", tcp_offset_local), dtype=float)
    point_b_world = _point_b_world_from_pose(coord_transform, current_pose, point_b_offset)
    target_point_b_world = np.array(geometry.get("contact_point_B_world", point_b_world), dtype=float)
    point_b_error = float(np.linalg.norm(point_b_world - target_point_b_world))
    diagnostic_bypass = bool(getattr(args, "phase2_diagnostic_finger_link_midpoint_bypass", False))
    real_grasp_center_world, real_grasp_center_log = _resolve_real_grasp_center_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        diagnostic_finger_link_midpoint_bypass=diagnostic_bypass,
        include_diagnostic_comparison=diagnostic_bypass,
    )
    real_grasp_center_error = None
    proxy_to_real_delta_world = None
    proxy_to_real_delta_local = None
    proxy_to_real_delta_norm = None
    close_critical_world = point_b_world
    close_critical_error = point_b_error
    close_critical_metric = "point_B_proxy_world_fallback"
    close_critical_uses_real_grasp_center = bool(
        real_grasp_center_world is not None
        and real_grasp_center_log.get("close_critical_reference", False)
    )
    if close_critical_uses_real_grasp_center:
        real_grasp_center_world = np.array(real_grasp_center_world, dtype=float)
        proxy_to_real_delta_world = real_grasp_center_world - point_b_world
        proxy_to_real_delta_norm = float(np.linalg.norm(proxy_to_real_delta_world))
        proxy_to_real_delta_local = _pose_rotation_world(coord_transform, current_pose).T @ proxy_to_real_delta_world
        real_grasp_center_error = float(np.linalg.norm(real_grasp_center_world - target_point_b_world))
        close_critical_world = real_grasp_center_world
        close_critical_error = real_grasp_center_error
        close_critical_metric = "real_grasp_center_world"
    elif real_grasp_center_world is not None:
        real_grasp_center_world = np.array(real_grasp_center_world, dtype=float)
    fingertip_reference_source_used = real_grasp_center_log.get(
        "fingertip_reference_source_used",
        real_grasp_center_log.get("fingertip_reference_source", real_grasp_center_log.get("source")),
    )
    fingertip_component_positions = real_grasp_center_log.get(
        "fingertip_component_positions_world",
        real_grasp_center_log.get("component_positions_world"),
    )
    real_grasp_tip1_world = None
    real_grasp_tip2_world = None
    if isinstance(fingertip_component_positions, list) and len(fingertip_component_positions) >= 2:
        real_grasp_tip1_world = fingertip_component_positions[0]
        real_grasp_tip2_world = fingertip_component_positions[1]
    fingertip_midpoint_world = real_grasp_center_log.get("fingertip_midpoint_world")
    distance_tip_mid_to_object = None
    try:
        if (
            fingertip_midpoint_world is not None
            and object_center_world is not None
            and np.isfinite(np.array(fingertip_midpoint_world, dtype=float)).all()
            and np.isfinite(np.array(object_center_world, dtype=float)).all()
        ):
            distance_tip_mid_to_object = float(
                    np.linalg.norm(np.array(fingertip_midpoint_world, dtype=float) - np.array(object_center_world, dtype=float))
            )
    except Exception:
        distance_tip_mid_to_object = None
    runtime_object_grasp_frame = dict(object_grasp_frame or {})
    if not runtime_object_grasp_frame and isinstance(geometry.get("object_grasp_frame"), dict):
        runtime_object_grasp_frame = dict(geometry.get("object_grasp_frame", {}))
    runtime_object_grasp_frame.setdefault("object_grasp_center_world", object_center_world.tolist())
    for axis_key in (
        "tip_axis_world",
        "closing_axis_world",
        "hand_closing_axis_world",
        "grasp_axis_world",
        "minor_axis_world",
        "width_axis_world",
    ):
        if axis_key not in runtime_object_grasp_frame and geometry.get(axis_key) is not None:
            runtime_object_grasp_frame[axis_key] = geometry.get(axis_key)
    close_runtime_metrics = _compute_runtime_two_finger_metrics(
        reference_log=real_grasp_center_log,
        object_grasp_frame=runtime_object_grasp_frame,
        object_center_world=object_center_world,
        fallback_midpoint_world=real_grasp_center_world if close_critical_uses_real_grasp_center else point_b_world,
        fallback_source=close_critical_metric,
    )
    two_finger_marker_log = _upsert_two_finger_runtime_debug_markers(
        stage=stage,
        metrics=close_runtime_metrics,
        enabled=bool(getattr(args, "diagnostic_two_finger_marker_enable", True)),
    )
    close_reference_debug_markers: dict[str, Any] = {
        "enabled": bool(getattr(args, "diagnostic_two_finger_marker_enable", True)),
        "marker_schema": "contact_centric_close_reference_markers_v1",
        "markers": {},
    }
    if close_reference_debug_markers["enabled"]:
        close_marker_specs = [
            (
                "real_grasp_center_world",
                DEBUG_REAL_GRASP_CENTER_MARKER_PATH,
                real_grasp_center_world,
                DEBUG_REAL_GRASP_CENTER_MARKER_COLOR,
            ),
            (
                "contact_point_world",
                DEBUG_CONTACT_POINT_MARKER_PATH,
                geometry.get("contact_point_world"),
                DEBUG_CONTACT_POINT_MARKER_COLOR,
            ),
            (
                "contact_point_B_world",
                DEBUG_CONTACT_POINT_B_MARKER_PATH,
                target_point_b_world,
                DEBUG_CONTACT_POINT_B_MARKER_COLOR,
            ),
        ]
        for marker_key, marker_path, marker_position, marker_color in close_marker_specs:
            marker_pos = _finite_world_vector_or_none(marker_position)
            if marker_pos is None:
                close_reference_debug_markers["markers"][marker_key] = {
                    "path": marker_path,
                    "updated": False,
                    "reason": "position_unavailable",
                }
                continue
            try:
                _upsert_debug_marker(
                    stage=stage,
                    path=marker_path,
                    position=marker_pos,
                    radius=DEBUG_TIP_MARKER_RADIUS_M,
                    color=marker_color,
                )
                close_reference_debug_markers["markers"][marker_key] = {
                    "path": marker_path,
                    "updated": True,
                    "world_position": marker_pos.tolist(),
                }
            except Exception as exc:
                close_reference_debug_markers["markers"][marker_key] = {
                    "path": marker_path,
                    "updated": False,
                    "error": repr(exc),
                }
    legacy_to_tip_delta_world = real_grasp_center_log.get(
        "legacy_link_midpoint_to_fingertip_midpoint_delta_world",
        real_grasp_center_log.get("legacy_link_midpoint_to_fingertip_delta_world"),
    )
    legacy_to_tip_delta_norm = real_grasp_center_log.get(
        "legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m",
        real_grasp_center_log.get("legacy_link_midpoint_to_fingertip_delta_norm_m"),
    )
    point_b_to_tip_delta_world = (
        None
        if real_grasp_center_world is None or real_grasp_center_log.get("reference_mode") != "fingertip_end_midpoint"
        else (real_grasp_center_world - point_b_world)
    )
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
    object_support_z = _finite_float_or_none(geometry.get("object_support_z_world"))
    vertical_point_b_support_gap = None if object_support_z is None else float(point_b_world[2] - object_support_z)
    vertical_real_center_support_gap = (
        None
        if object_support_z is None or real_grasp_center_world is None
        else float(real_grasp_center_world[2] - object_support_z)
    )
    vertical_close_critical_support_gap = (
        None
        if object_support_z is None or close_critical_world is None
        else float(close_critical_world[2] - object_support_z)
    )
    return {
        "object_center_world": object_center_world.tolist(),
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
        "real_grasp_center_close_critical_reference": bool(real_grasp_center_log.get("close_critical_reference", False)),
        "real_grasp_center_close_critical_rejected_reason": real_grasp_center_log.get("close_critical_rejected_reason"),
        "fingertip_reference_source_used": fingertip_reference_source_used,
        "fingertip_reference_fallback_used": bool(real_grasp_center_log.get("fallback_used", False)),
        "fingertip_component_positions_world": fingertip_component_positions,
        "tip1_world": real_grasp_tip1_world,
        "tip2_world": real_grasp_tip2_world,
        "tip_mid_world": close_runtime_metrics.get("tip_mid_world"),
        "tip_axis_world": close_runtime_metrics.get("tip_axis_world"),
        "tip_mid_error_to_object_grasp_center_m": close_runtime_metrics.get("tip_mid_error_to_object_grasp_center_m"),
        "tip_mid_xy_error_m": close_runtime_metrics.get("tip_mid_xy_error_m"),
        "tip_mid_z_error_m": close_runtime_metrics.get("tip_mid_z_error_m"),
        "tip1_to_object_grasp_center_distance_m": close_runtime_metrics.get("tip1_to_object_grasp_center_distance_m"),
        "tip2_to_object_grasp_center_distance_m": close_runtime_metrics.get("tip2_to_object_grasp_center_distance_m"),
        "tip1_to_object_center_distance_m": close_runtime_metrics.get("tip1_to_object_center_distance_m"),
        "tip2_to_object_center_distance_m": close_runtime_metrics.get("tip2_to_object_center_distance_m"),
        "tip_axis_alignment_error_rad": close_runtime_metrics.get("tip_axis_alignment_error_rad"),
        "tip_symmetry_error_m": close_runtime_metrics.get("tip_symmetry_error_m"),
        "tip_z_asymmetry_m": close_runtime_metrics.get("tip_z_asymmetry_m"),
        "close_runtime_metrics": close_runtime_metrics,
        "runtime_two_finger_geometry_primary": bool(close_runtime_metrics.get("primary_runtime_truth", False)),
        "two_finger_runtime_debug_markers": two_finger_marker_log,
        "close_reference_debug_markers": close_reference_debug_markers,
        "distance_tip_mid_to_object_m": distance_tip_mid_to_object,
        "diagnostic_finger_link_midpoint_bypass_requested": diagnostic_bypass,
        "diagnostic_finger_link_midpoint_bypass_compare_enabled": bool(diagnostic_bypass),
        "fingertip_midpoint_world": fingertip_midpoint_world,
        "legacy_link_midpoint_world": real_grasp_center_log.get("legacy_link_midpoint_world"),
        "legacy_link_midpoint_to_fingertip_midpoint_delta_world": legacy_to_tip_delta_world,
        "legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m": legacy_to_tip_delta_norm,
        "point_B_proxy_to_fingertip_midpoint_delta_world": None if point_b_to_tip_delta_world is None else point_b_to_tip_delta_world.tolist(),
        "point_B_proxy_to_fingertip_midpoint_delta_world_norm_m": None if point_b_to_tip_delta_world is None else float(np.linalg.norm(point_b_to_tip_delta_world)),
        "real_grasp_center_component_positions_world": real_grasp_center_log.get("component_positions_world"),
        "real_grasp_center_error_before_close_m": real_grasp_center_error,
        "proxy_to_real_grasp_center_delta_world": None if proxy_to_real_delta_world is None else proxy_to_real_delta_world.tolist(),
        "proxy_to_real_grasp_center_delta_world_norm_m": proxy_to_real_delta_norm,
        "proxy_to_real_grasp_center_delta_local": None if proxy_to_real_delta_local is None else proxy_to_real_delta_local.tolist(),
        "close_critical_metric": close_critical_metric,
        "close_critical_eval_world": close_critical_world.tolist(),
        "close_critical_target_world": target_point_b_world.tolist(),
        "close_critical_error_before_close_m": close_critical_error,
        "close_critical_uses_real_grasp_center": close_critical_uses_real_grasp_center,
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
        "actual_fingertip_midpoint_component_positions_world_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("fingertip_component_positions_world", vertical_xy_reference_runtime_log.get("component_positions_world")),
        "actual_fingertip_midpoint_world_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("fingertip_midpoint_world", vertical_xy_reference_runtime_log.get("world_position")),
        "actual_fingertip_reference_source_before_close": None if vertical_xy_reference_runtime_log is None else vertical_xy_reference_runtime_log.get("fingertip_reference_source_used", vertical_xy_reference_runtime_log.get("source")),
        "vertical_xy_reference_error_before_close_m": vertical_xy_reference_error,
        "vertical_xy_reference_tolerance_m": geometry.get("vertical_xy_reference_tolerance_m"),
        "vertical_contact_sequence_policy": geometry.get("vertical_contact_sequence_policy"),
        "vertical_actual_point_B_gap_above_support_m": vertical_point_b_support_gap,
        "vertical_actual_real_grasp_center_gap_above_support_m": vertical_real_center_support_gap,
        "vertical_actual_close_critical_gap_above_support_m": vertical_close_critical_support_gap,
        "vertical_actual_support_gap_metric_order": [
            "close_critical",
            "real_grasp_center",
            "point_B_proxy",
        ],
        "far_reach_axis_world": geometry.get("far_reach_axis_world"),
        "tcp_offset_local_used": np.array(tcp_offset_local, dtype=float).tolist(),
        "point_b_offset_local_used": point_b_offset.tolist(),
        "tcp_offset_base_for_target": geometry["contact_details"].get("point_b_offset_base"),
        "end_effector_name": end_effector_name,
        "end_effector_path": end_effector_path,
        "ee_frame_compensation_active": bool(getattr(args, "ee_frame_compensation_active", False)),
    }


def execute_pregrasp(servo_spec: ServoSpec, **servo_kwargs: Any) -> dict[str, Any]:
    coord_transform = servo_kwargs.get("coord_transform")
    if coord_transform is None:
        raise RuntimeError("execute_pregrasp requires coord_transform for IK failure instrumentation")

    target_pose_world = _pose_position_world(coord_transform, servo_spec.target_pose_base)
    target_rpy_world = _rot_to_euler_xyz(_pose_rotation_world(coord_transform, servo_spec.target_pose_base))
    target_world_log = {
        "ik_target_position_world": target_pose_world.tolist(),
        "ik_target_rpy": target_rpy_world.tolist(),
        "ik_target_position_base": np.array(servo_spec.target_pose_base[:3], dtype=float).tolist(),
        "ik_target_rpy_base": np.array(servo_spec.target_pose_base[3:6], dtype=float).tolist(),
    }

    primary_result = _execute_dualarmik_servo_phase(servo_spec, **servo_kwargs)
    primary_ok = bool(
        primary_result["final_error"] <= float(servo_spec.pos_tolerance)
        and primary_result["final_rotation_error_rad"] <= float(servo_spec.rot_tolerance)
    )
    primary_result["phase2_component"] = "execute_pregrasp"
    primary_result["pregrasp_primary_ik_target_world"] = target_world_log
    primary_result["ik_target_position_world"] = target_world_log["ik_target_position_world"]
    primary_result["ik_target_rpy"] = target_world_log["ik_target_rpy"]
    primary_result["position_error_norm"] = float(primary_result["final_error"])
    primary_result["orientation_error_rad"] = float(primary_result["final_rotation_error_rad"])
    primary_result["ik_success"] = primary_ok
    primary_result["fallback_attempt"] = None

    if not primary_ok:
        primary_result["ik_pregrasp_fail_reason"] = "IK_PREGRASP_FAIL"
        print("IK_PREGRASP_FAIL: primary pregrasp target not reached; running yaw=0 fallback")
        fallback_pose = np.array(servo_spec.target_pose_base, dtype=float)
        fallback_pose[5] = 0.0
        fallback_target_world = _pose_position_world(coord_transform, fallback_pose)
        fallback_rpy_world = _rot_to_euler_xyz(_pose_rotation_world(coord_transform, fallback_pose))
        fallback_log = {
            "fallback_attempt": True,
            "fallback_attempt_target_position_world": fallback_target_world.tolist(),
            "fallback_attempt_target_rpy_world": fallback_rpy_world.tolist(),
            "fallback_attempt_target_position_base": np.array(fallback_pose[:3], dtype=float).tolist(),
            "fallback_attempt_target_rpy_base": np.array(fallback_pose[3:6], dtype=float).tolist(),
            "fallback_attempt_reason": "retry_pregrasp_with_yaw_zero",
        }
        fallback_result = _execute_dualarmik_servo_phase(
            ServoSpec(
                f"{servo_spec.name}_yaw0_fallback",
                fallback_pose,
                servo_spec.pos_tolerance,
                servo_spec.rot_tolerance,
                servo_spec.max_ticks,
            ),
            **servo_kwargs,
        )
        fallback_ok = bool(
            fallback_result["final_error"] <= float(servo_spec.pos_tolerance)
            and fallback_result["final_rotation_error_rad"] <= float(servo_spec.rot_tolerance)
        )
        fallback_log = {
            **fallback_log,
            "fallback_attempt_ik_target_position_world": fallback_log["fallback_attempt_target_position_world"],
            "fallback_attempt_ik_target_rpy": fallback_log["fallback_attempt_target_rpy_world"],
            "fallback_attempt_position_error_norm": float(fallback_result["final_error"]),
            "fallback_attempt_orientation_error_rad": float(fallback_result["final_rotation_error_rad"]),
            "fallback_attempt_ik_success": fallback_ok,
        }
        primary_result["fallback_attempt"] = fallback_log
        if fallback_ok:
            print("IK_PREGRASP_FAIL: fallback yaw=0 attempt succeeded")
            fallback_result["phase2_component"] = "execute_pregrasp_fallback_yaw0"
            fallback_result["pregrasp_primary_ik_target_world"] = target_world_log
            fallback_result["ik_target_position_world"] = target_world_log["ik_target_position_world"]
            fallback_result["ik_target_rpy"] = target_world_log["ik_target_rpy"]
            fallback_result["position_error_norm"] = float(fallback_result["final_error"])
            fallback_result["orientation_error_rad"] = float(fallback_result["final_rotation_error_rad"])
            fallback_result["ik_success"] = True
            fallback_result["fallback_attempt"] = fallback_log
            fallback_result["fallback_used"] = True
            fallback_result["ik_pregrasp_fail_reason"] = "IK_PREGRASP_FAIL"
            return fallback_result
        print("IK_PREGRASP_FAIL: fallback yaw=0 attempt also failed")
        primary_result["ik_pregrasp_fail_reason"] = (
            "IK_PREGRASP_FAIL: primary and yaw=0 fallback both failed"
        )

    return primary_result


def final_descent_local_ik(
    *,
    phase_name: str,
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
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    locked_target_world: np.ndarray | list[float],
    locked_rpy: np.ndarray | list[float],
    locked_target_pose_base: np.ndarray | list[float],
    point_b_offset_local: np.ndarray | list[float],
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    ik_overrides: dict[str, Any] | None = None,
    object_grasp_frame: dict[str, Any] | None = None,
    selected_candidate_filter: dict[str, Any] | None = None,
    table_frame: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
    """Local closed-loop descent that measures real grasp center each target refresh."""
    locked_target = np.array(locked_target_world, dtype=float)
    locked_rpy_arr = np.array(locked_rpy, dtype=float)
    point_b_offset = np.array(point_b_offset_local, dtype=float)
    diagnostic_bypass = bool(getattr(args, "phase2_diagnostic_finger_link_midpoint_bypass", False))
    object_center_candidate = geometry.get("object_center_world")
    if isinstance(object_center_candidate, (list, tuple)) and len(object_center_candidate) >= 3:
        object_center_world = np.array(object_center_candidate[:3], dtype=float)
    elif isinstance(object_center_candidate, np.ndarray) and object_center_candidate.size >= 3:
        object_center_world = np.array(object_center_candidate[:3], dtype=float)
    else:
        object_center_world = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
    xy_step_max = float(args.phase2_descent_xy_step)
    z_step_max = float(args.phase2_descent_z_step)
    yaw_step_max = float(args.phase2_descent_yaw_step)
    samples: list[dict[str, Any]] = []
    proxy_middle_point_debug_samples: list[dict[str, Any]] = []
    call_count = 0
    previous_commanded_world: np.ndarray | None = None
    last_yaw = float(locked_rpy_arr[2])
    vertical_tip_stop_rule_threshold = float(
        getattr(args, "phase2_vertical_tip_table_z_close_threshold", PHASE2_VERTICAL_TIP_TABLE_Z_CLOSE_THRESHOLD_M)
    )
    vertical_tip_stop_rule_active = bool(phase_name == "phase2_vertical_final_descent_local_ik" and table_frame is not None)
    vertical_tip_stop_samples: list[dict[str, Any]] = []
    vertical_tip_stop_rule_last_sample: dict[str, Any] | None = None
    vertical_tip_reached_table_z0 = False
    vertical_tip_close_stop_reason: str | None = None
    vertical_tip_monitor_call_count = 0
    initial_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
    initial_ee_world = _pose_position_world(coord_transform, initial_pose)
    initial_ee_rot_world = _pose_rotation_world(coord_transform, initial_pose)
    initial_point_b_world = _point_b_world_from_pose(coord_transform, initial_pose, point_b_offset)
    object_support_z = _finite_float_or_none(geometry.get("object_support_z_world"))

    def support_gap_m(world_pos: np.ndarray | list[float] | None) -> float | None:
        if object_support_z is None or world_pos is None:
            return None
        world_arr = np.array(world_pos, dtype=float)
        if world_arr.size < 3 or not math.isfinite(float(world_arr[2])):
            return None
        return float(world_arr[2] - object_support_z)

    z_completion_enforced = bool(phase_name == "phase2_far_final_descent_local_ik")
    z_completion_tolerance = min(float(args.phase2_close_real_center_tolerance), float(args.pre_close_point_b_tolerance))
    phase2_z_descent_state: dict[str, Any] = {
        "completion_rule": "runtime_contact_reference_z_must_reach_locked_target_z_before_phase_success",
        "completion_enforced": z_completion_enforced,
        "target_z_world_m": float(locked_target[2]),
        "z_completion_tolerance_m": z_completion_tolerance,
        "final_z_reached_by_runtime": False,
        "descent_stopped_before_contact_z": True,
    }

    def update_phase2_z_descent_state(
        measured_world: np.ndarray,
        *,
        commanded_world: np.ndarray | None = None,
        source: str | None = None,
    ) -> float:
        z_remaining = float(max(0.0, float(measured_world[2]) - float(locked_target[2])))
        phase2_z_descent_state.update(
            {
                "measured_world": np.array(measured_world, dtype=float).tolist(),
                "measured_world_source": source,
                "current_z_world_m": float(measured_world[2]),
                "target_z_world_m": float(locked_target[2]),
                "z_remaining_to_contact_m": z_remaining,
                "final_z_reached_by_runtime": bool(z_remaining <= z_completion_tolerance),
                "descent_stopped_before_contact_z": bool(z_remaining > z_completion_tolerance),
            }
        )
        if commanded_world is not None:
            commanded_remaining = float(max(0.0, float(commanded_world[2]) - float(locked_target[2])))
            phase2_z_descent_state.update(
                {
                    "commanded_world": np.array(commanded_world, dtype=float).tolist(),
                    "commanded_z_world_m": float(commanded_world[2]),
                    "commanded_z_remaining_to_contact_m": commanded_remaining,
                    "final_z_reached_by_command": bool(commanded_remaining <= z_completion_tolerance),
                }
            )
        return z_remaining

    def resolve_runtime_contact_reference_for_descent() -> dict[str, Any]:
        current_pose = _current_ee_pose_base(ik_solver, dc, articulation, arm_side, args=args)
        real_center_world, real_center_log = _resolve_real_grasp_center_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            diagnostic_finger_link_midpoint_bypass=diagnostic_bypass,
            include_diagnostic_comparison=diagnostic_bypass,
        )
        current_point_b_world = _point_b_world_from_pose(coord_transform, current_pose, point_b_offset)
        trusted_real_center_for_descent = bool(
            real_center_world is not None
            and real_center_log.get("close_critical_reference", False)
        )
        tip_mid_world_for_descent = (
            _finite_world_vector_or_none(real_center_log.get("fingertip_midpoint_world"))
            if trusted_real_center_for_descent
            else None
        )
        if trusted_real_center_for_descent and tip_mid_world_for_descent is None:
            tip_components_for_descent = real_center_log.get(
                "fingertip_component_positions_world",
                real_center_log.get("component_positions_world"),
            )
            if isinstance(tip_components_for_descent, list) and len(tip_components_for_descent) >= 2:
                tip1_for_descent = _finite_world_vector_or_none(tip_components_for_descent[0])
                tip2_for_descent = _finite_world_vector_or_none(tip_components_for_descent[1])
                if tip1_for_descent is not None and tip2_for_descent is not None:
                    tip_mid_world_for_descent = 0.5 * (tip1_for_descent + tip2_for_descent)
        if tip_mid_world_for_descent is not None:
            measured_world = tip_mid_world_for_descent
            control_reference_source = "tip_mid"
        else:
            measured_world = current_point_b_world
            control_reference_source = (
                "point_B_fallback_untrusted_real_grasp_center"
                if real_center_world is not None
                else "point_B_fallback"
            )
        return {
            "current_pose": current_pose,
            "real_center_world": real_center_world,
            "real_center_log": real_center_log,
            "current_point_b_world": current_point_b_world,
            "measured_world": measured_world,
            "control_reference_source": control_reference_source,
        }

    initial_real_center_world, initial_real_center_log = _resolve_real_grasp_center_world(
        stage=stage,
        dc=dc,
        articulation=articulation,
        robot_root_path=robot_root_path,
        arm_side=arm_side,
        fallback_world=initial_point_b_world,
        fallback_source="point_B_proxy_initial_phase2_final_descent",
        diagnostic_finger_link_midpoint_bypass=diagnostic_bypass,
        include_diagnostic_comparison=diagnostic_bypass,
    )
    use_real_contact_offset = bool(
        initial_real_center_world is not None
        and initial_real_center_log.get("source") != "fallback_proxy"
        and initial_real_center_log.get("close_critical_reference", False)
    )
    if use_real_contact_offset:
        contact_control_world = np.array(initial_real_center_world, dtype=float)
        contact_control_offset = initial_ee_rot_world.T @ (contact_control_world - initial_ee_world)
        contact_control_reference_source = str(initial_real_center_log.get("source"))
    else:
        contact_control_world = initial_point_b_world
        contact_control_offset = point_b_offset.copy()
        contact_control_reference_source = "point_B_proxy_fallback"
    point_b_to_contact_control_delta = contact_control_world - initial_point_b_world
    nominal_target_pose_base, nominal_target_pose_log = _pose_for_contact_reference_world(
        locked_target,
        coord_transform,
        locked_rpy_arr,
        contact_control_offset,
    )

    def _update_proxy_middle_point_debug_marker(
        *,
        context: str,
        call_index: int,
    ) -> dict[str, Any] | None:
        marker_path = DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH
        marker_world, marker_log = _resolve_finger_midpoint_reference_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=robot_root_path,
            arm_side=arm_side,
            requested_name_token="phase2_debug_legacy_finger_link_midpoint_runtime",
        )
        if marker_world is None:
            debug_log = {
                "marker_call_index": int(call_index),
                "update_context": context,
                "marker_path": marker_path,
                "marker_resolved": False,
                "marker_source": marker_log.get("source"),
                "marker_reference_mode": marker_log.get("reference_mode"),
                "marker_fallback_used": bool(marker_log.get("fallback_used", False)),
                "marker_reason": marker_log.get("reason"),
                "marker_component_positions_world": None,
                "marker_world_position": None,
                "marker_update_error": "legacy proxy midpoint unresolved during descent",
            }
            proxy_middle_point_debug_samples.append(debug_log)
            return debug_log

        marker_world_arr = np.array(marker_world, dtype=float)
        _upsert_debug_marker(
            stage=stage,
            path=marker_path,
            position=marker_world_arr,
            radius=DEBUG_PROXY_MIDDLE_POINT_MARKER_RADIUS_M,
            color=DEBUG_PROXY_MIDDLE_POINT_MARKER_COLOR,
        )
        debug_log = {
            "marker_call_index": int(call_index),
            "update_context": context,
            "marker_path": marker_path,
            "marker_resolved": True,
            "marker_source": marker_log.get("source"),
            "marker_reference_mode": marker_log.get("reference_mode"),
            "marker_world_position": marker_world_arr.tolist(),
            "marker_component_positions_world": marker_log.get("component_positions_world"),
            "marker_finger1_world_position": marker_log.get("finger1_world_position"),
            "marker_finger2_world_position": marker_log.get("finger2_world_position"),
            "marker_fallback_used": bool(marker_log.get("fallback_used", False)),
            "marker_reference_reason": marker_log.get("reason"),
            "marker_reference_fallback": marker_log.get("fallback"),
        }
        proxy_middle_point_debug_samples.append(debug_log)
        return debug_log

    def monitor_vertical_tip_stop_rule(
        *,
        real_center_world: np.ndarray | list[float] | None = None,
        real_center_log: dict[str, Any] | None = None,
        context: str,
    ) -> dict[str, Any] | None:
        nonlocal vertical_tip_stop_rule_last_sample, vertical_tip_reached_table_z0, vertical_tip_close_stop_reason
        nonlocal vertical_tip_monitor_call_count
        if not vertical_tip_stop_rule_active:
            return None
        vertical_tip_monitor_call_count += 1
        if real_center_log is None:
            real_center_world, real_center_log = _resolve_real_grasp_center_world(
                stage=stage,
                dc=dc,
                articulation=articulation,
                robot_root_path=robot_root_path,
                arm_side=arm_side,
                diagnostic_finger_link_midpoint_bypass=diagnostic_bypass,
                include_diagnostic_comparison=diagnostic_bypass,
            )
        real_center_log = real_center_log or {}
        tip_source = str(
            real_center_log.get(
                "source",
                real_center_log.get(
                    "fingertip_reference_source_used",
                    real_center_log.get("fingertip_reference_source", "unresolved_fingertip_reference"),
                ),
            )
        )
        if real_center_world is not None:
            current_vertical_tip_world = np.array(real_center_world, dtype=float)
            try:
                current_vertical_tip_table = world_to_table(current_vertical_tip_world, table_frame or {})
                current_vertical_tip_table_z = float(current_vertical_tip_table[2])
                vertical_tip_stop_rule_triggered = bool(current_vertical_tip_table_z <= vertical_tip_stop_rule_threshold)
                vertical_tip_error = None
            except Exception as exc:
                current_vertical_tip_table = None
                current_vertical_tip_table_z = None
                vertical_tip_stop_rule_triggered = False
                vertical_tip_error = repr(exc)
        else:
            current_vertical_tip_world = None
            current_vertical_tip_table = None
            current_vertical_tip_table_z = None
            vertical_tip_stop_rule_triggered = False
            vertical_tip_error = "fingertip_reference_unresolved"
            tip_source = "unresolved_fingertip_reference_no_close_rule"
        tip1_table_z = None
        tip2_table_z = None
        tip_mid_table_z = current_vertical_tip_table_z
        tip_mid_minus_object_top_z = None
        tip_mid_minus_support_z = None
        component_positions = real_center_log.get(
            "fingertip_component_positions_world",
            real_center_log.get("component_positions_world"),
        )
        if isinstance(component_positions, list) and len(component_positions) >= 2:
            try:
                tip1_world = _finite_world_vector_or_none(component_positions[0])
                tip2_world = _finite_world_vector_or_none(component_positions[1])
                if tip1_world is not None:
                    tip1_table_z = float(world_to_table(tip1_world, table_frame or {})[2])
                if tip2_world is not None:
                    tip2_table_z = float(world_to_table(tip2_world, table_frame or {})[2])
            except Exception:
                tip1_table_z = None
                tip2_table_z = None
        object_top_z_world = None
        try:
            object_top_z_world = float(_bbox_state(stage, target_path)["bbox"]["max"][2])
        except Exception:
            object_top_z_world = None
        if current_vertical_tip_world is not None:
            if object_top_z_world is not None:
                tip_mid_minus_object_top_z = float(current_vertical_tip_world[2] - object_top_z_world)
            if object_support_z is not None:
                tip_mid_minus_support_z = float(current_vertical_tip_world[2] - object_support_z)
        vertical_tip_log = {
            "monitor_index": int(vertical_tip_monitor_call_count),
            "target_pose_call_index": int(call_count),
            "vertical_tip_stop_rule_context": context,
            "current_vertical_tip_world": None if current_vertical_tip_world is None else current_vertical_tip_world.tolist(),
            "current_vertical_tip_table": None if current_vertical_tip_table is None else current_vertical_tip_table.tolist(),
            "current_vertical_tip_table_z_m": current_vertical_tip_table_z,
            "tip1_table_z_m": tip1_table_z,
            "tip2_table_z_m": tip2_table_z,
            "tip_mid_table_z_m": tip_mid_table_z,
            "object_top_z_world_m": object_top_z_world,
            "tip_mid_minus_object_top_z_m": tip_mid_minus_object_top_z,
            "tip_mid_minus_support_z_m": tip_mid_minus_support_z,
            "vertical_tip_stop_rule_triggered": bool(vertical_tip_stop_rule_triggered),
            "vertical_tip_stop_rule_threshold_m": vertical_tip_stop_rule_threshold,
            "vertical_tip_stop_rule_source": tip_source,
            "vertical_tip_reference_semantics": "active close-critical fingertip-end midpoint in table frame",
            "vertical_tip_fingertip_reference_source_used": real_center_log.get(
                "fingertip_reference_source_used",
                real_center_log.get("fingertip_reference_source"),
            ),
            "vertical_tip_component_positions_world": real_center_log.get(
                "fingertip_component_positions_world",
                real_center_log.get("component_positions_world"),
            ),
            "vertical_tip_resolution_fallback_used": bool(real_center_log.get("fallback_used", False)),
            "vertical_tip_stop_rule_error": vertical_tip_error,
        }
        vertical_tip_stop_samples.append(vertical_tip_log)
        vertical_tip_stop_rule_last_sample = vertical_tip_log
        if vertical_tip_stop_rule_triggered:
            vertical_tip_reached_table_z0 = True
            vertical_tip_close_stop_reason = "tip_table_z_leq_zero"
            raise ServoEarlyStop(
                "vertical_tip_table_z_leq_zero_rule",
                {
                    "vertical_tip_reached_table_z0": True,
                    "vertical_tip_close_stop_reason": "tip_table_z_leq_zero",
                    **vertical_tip_log,
                },
            )
        return vertical_tip_log

    def contact_target_pose_fn() -> np.ndarray:
        nonlocal call_count, previous_commanded_world, last_yaw
        nonlocal vertical_tip_stop_rule_last_sample, vertical_tip_reached_table_z0, vertical_tip_close_stop_reason
        call_count += 1
        reference = resolve_runtime_contact_reference_for_descent()
        real_center_world = reference["real_center_world"]
        real_center_log = reference["real_center_log"]
        current_point_b_world = reference["current_point_b_world"]
        measured_world = reference["measured_world"]
        control_reference_source = reference["control_reference_source"]
        measured_world_source_for_descent = control_reference_source
        descent_object_grasp_frame = dict(object_grasp_frame or {})
        if "object_grasp_center_world" not in descent_object_grasp_frame:
            if np.isfinite(object_center_world).all():
                descent_object_grasp_frame["object_grasp_center_world"] = object_center_world.tolist()
            else:
                descent_object_grasp_frame["object_grasp_center_world"] = locked_target.tolist()
        for axis_key in (
            "tip_axis_world",
            "closing_axis_world",
            "hand_closing_axis_world",
            "grasp_axis_world",
            "minor_axis_world",
            "width_axis_world",
        ):
            if axis_key not in descent_object_grasp_frame and geometry.get(axis_key) is not None:
                descent_object_grasp_frame[axis_key] = geometry.get(axis_key)
        close_runtime_metrics = _compute_runtime_two_finger_metrics(
            reference_log=real_center_log,
            object_grasp_frame=descent_object_grasp_frame,
            object_center_world=object_center_world if np.isfinite(object_center_world).all() else None,
            fallback_midpoint_world=measured_world,
            fallback_source="phase2_final_descent_measured_contact_reference",
        )

        vertical_tip_log = monitor_vertical_tip_stop_rule(
            real_center_world=real_center_world,
            real_center_log=real_center_log,
            context="target_pose_fn",
        )

        delta = locked_target - measured_world
        xy_step = _clamp_norm(delta[:2], xy_step_max)
        z_step = max(-z_step_max, min(0.0, float(delta[2])))
        commanded_world = measured_world.copy()
        commanded_world[:2] += xy_step
        commanded_world[2] += z_step
        if previous_commanded_world is not None and commanded_world[2] > previous_commanded_world[2]:
            commanded_world[2] = previous_commanded_world[2]

        yaw_delta = max(-yaw_step_max, min(yaw_step_max, float(locked_rpy_arr[2] - last_yaw)))
        last_yaw = float(last_yaw + yaw_delta)
        commanded_rpy = locked_rpy_arr.copy()
        commanded_rpy[2] = last_yaw
        commanded_pose, contact_pose_log = _pose_for_contact_reference_world(
            commanded_world,
            coord_transform,
            commanded_rpy,
            contact_control_offset,
        )
        previous_commanded_world = commanded_world.copy()
        commanded_delta_to_target = locked_target - commanded_world
        update_phase2_z_descent_state(
            measured_world,
            commanded_world=commanded_world,
            source=control_reference_source,
        )

        if call_count == 1 or call_count % int(args.trace_interval) == 0:
            trace_sample_proxy_marker_log = _update_proxy_middle_point_debug_marker(
                context="target_pose_fn", call_index=call_count
            )
            two_finger_marker_log = _upsert_two_finger_runtime_debug_markers(
                stage=stage,
                metrics=close_runtime_metrics,
                enabled=bool(getattr(args, "diagnostic_two_finger_marker_enable", True)),
            )
            sample_fingertip_components = real_center_log.get(
                "fingertip_component_positions_world",
                real_center_log.get("component_positions_world"),
            )
            fingertip_delta = None
            if real_center_world is not None:
                fingertip_delta = np.array(real_center_world, dtype=float) - current_point_b_world
            tip1_world = None
            tip2_world = None
            if isinstance(sample_fingertip_components, list) and len(sample_fingertip_components) >= 2:
                tip1_world = sample_fingertip_components[0]
                tip2_world = sample_fingertip_components[1]
            distance_tip_mid_to_object = None
            tip_midpoint_world = real_center_log.get("fingertip_midpoint_world")
            try:
                if (
                    tip_midpoint_world is not None
                    and object_center_world is not None
                    and np.isfinite(np.array(tip_midpoint_world, dtype=float)).all()
                    and np.isfinite(object_center_world).all()
                ):
                    distance_tip_mid_to_object = float(
                        np.linalg.norm(np.array(tip_midpoint_world, dtype=float) - object_center_world)
                    )
            except Exception:
                distance_tip_mid_to_object = None
            sample = {
                    "call_index": int(call_count),
                    "locked_target_grasp_center_world": locked_target.tolist(),
                    "object_center_world": object_center_world.tolist() if object_center_world is not None else None,
                    "measured_real_grasp_center_world": None if real_center_world is None else np.array(real_center_world, dtype=float).tolist(),
                    "real_grasp_center_resolution_source": real_center_log.get("source"),
                    "fingertip_reference_source_used": real_center_log.get(
                        "fingertip_reference_source_used",
                        real_center_log.get("fingertip_reference_source", real_center_log.get("source")),
                    ),
                    "fingertip_component_positions_world": sample_fingertip_components,
                    "tip1_world": tip1_world,
                    "tip2_world": tip2_world,
                    "tip_mid_world": close_runtime_metrics.get("tip_mid_world"),
                    "tip_axis_world": close_runtime_metrics.get("tip_axis_world"),
                    "tip_mid_error_to_object_grasp_center_m": close_runtime_metrics.get("tip_mid_error_to_object_grasp_center_m"),
                    "tip_mid_xy_error_m": close_runtime_metrics.get("tip_mid_xy_error_m"),
                    "tip_mid_z_error_m": close_runtime_metrics.get("tip_mid_z_error_m"),
                    "tip1_to_object_grasp_center_distance_m": close_runtime_metrics.get("tip1_to_object_grasp_center_distance_m"),
                    "tip2_to_object_grasp_center_distance_m": close_runtime_metrics.get("tip2_to_object_grasp_center_distance_m"),
                    "tip1_to_object_center_distance_m": close_runtime_metrics.get("tip1_to_object_center_distance_m"),
                    "tip2_to_object_center_distance_m": close_runtime_metrics.get("tip2_to_object_center_distance_m"),
                    "tip_axis_alignment_error_rad": close_runtime_metrics.get("tip_axis_alignment_error_rad"),
                    "tip_symmetry_error_m": close_runtime_metrics.get("tip_symmetry_error_m"),
                    "tip_z_asymmetry_m": close_runtime_metrics.get("tip_z_asymmetry_m"),
                    "close_runtime_metrics": close_runtime_metrics,
                    "runtime_two_finger_geometry_primary": bool(close_runtime_metrics.get("primary_runtime_truth", False)),
                    "two_finger_runtime_debug_markers": two_finger_marker_log,
                    "distance_tip_mid_to_object_m": distance_tip_mid_to_object,
                    "fingertip_midpoint_world": tip_midpoint_world,
                    "legacy_link_midpoint_world": real_center_log.get("legacy_link_midpoint_world"),
                    "legacy_link_midpoint_to_fingertip_midpoint_delta_world": real_center_log.get(
                        "legacy_link_midpoint_to_fingertip_midpoint_delta_world",
                        real_center_log.get("legacy_link_midpoint_to_fingertip_delta_world"),
                    ),
                    "legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m": real_center_log.get(
                        "legacy_link_midpoint_to_fingertip_midpoint_delta_norm_m",
                        real_center_log.get("legacy_link_midpoint_to_fingertip_delta_norm_m"),
                    ),
                    "real_grasp_center_fallback_used": bool(real_center_log.get("fallback_used", False)),
                    "point_B_proxy_world": current_point_b_world.tolist(),
                    "point_B_proxy_to_fingertip_midpoint_delta_world": None if fingertip_delta is None else fingertip_delta.tolist(),
                    "point_B_proxy_to_fingertip_midpoint_delta_world_norm_m": None if fingertip_delta is None else float(np.linalg.norm(fingertip_delta)),
                    "measured_world_used": measured_world.tolist(),
                    "measured_world_semantics": "tip_mid_if_available_else_point_B_compatibility",
                    "control_reference_source": control_reference_source,
                    "measured_world_source_for_descent": measured_world_source_for_descent,
                    "proxy_middle_point_world": None
                    if trace_sample_proxy_marker_log is None
                    else trace_sample_proxy_marker_log.get("marker_world_position"),
                    "proxy_middle_point_source": None
                    if trace_sample_proxy_marker_log is None
                    else trace_sample_proxy_marker_log.get("marker_source"),
                    "proxy_middle_point_component_positions_world": None
                    if trace_sample_proxy_marker_log is None
                    else trace_sample_proxy_marker_log.get("marker_component_positions_world"),
                    "proxy_middle_point_marker_path": DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH,
                    "proxy_middle_point_fallback_used": None
                    if trace_sample_proxy_marker_log is None
                    else bool(trace_sample_proxy_marker_log.get("marker_fallback_used", False)),
                    "measured_support_gap_m": support_gap_m(measured_world),
                    "delta_to_locked_target_m": delta.tolist(),
                    "current_z_world_m": float(measured_world[2]),
                    "target_z_world_m": float(locked_target[2]),
                    "delta_z_world_m": float(delta[2]),
                    "commanded_z_world_m": float(commanded_world[2]),
                    "commanded_delta_z_world_m": float(commanded_delta_to_target[2]),
                    "measured_tip_mid_world": measured_world.tolist(),
                    "commanded_tip_mid_world": commanded_world.tolist(),
                    "measured_tip_mid_to_target_delta_world": delta.tolist(),
                    "commanded_tip_mid_to_target_delta_world": commanded_delta_to_target.tolist(),
                    "measured_tip_mid_xy_error_m": float(np.linalg.norm(delta[:2])),
                    "measured_tip_mid_z_error_m": float(abs(delta[2])),
                    "commanded_tip_mid_xy_error_m": float(np.linalg.norm(commanded_delta_to_target[:2])),
                    "commanded_tip_mid_z_error_m": float(abs(commanded_delta_to_target[2])),
                    "control_error_vs_following_error_note": "measured fields show runtime contact-reference error; commanded fields show the contact-centric target sent through IK conversion",
                    "xy_step_command_m": xy_step.tolist(),
                    "z_step_command_m": z_step,
                    "commanded_grasp_center_world": commanded_world.tolist(),
                    "commanded_support_gap_m": support_gap_m(commanded_world),
                    "commanded_pose_base": commanded_pose.tolist(),
                    "contact_reference_pose_conversion": contact_pose_log,
                    "contact_control_reference_source": contact_control_reference_source,
                    "control_reference_source": control_reference_source,
                    "measured_world_source_for_descent": measured_world_source_for_descent,
                    "contact_control_offset_local": contact_control_offset.tolist(),
                    "monotonic_z_downward": True,
                    "yaw_step_command_rad": yaw_delta,
                    "locked_yaw_rad": float(locked_rpy_arr[2]),
                    "commanded_yaw_rad": float(commanded_rpy[2]),
                }
            if vertical_tip_log is not None:
                sample.update(vertical_tip_log)
            samples.append(sample)
        return commanded_pose

    def phase2_z_descent_completion_fn() -> bool:
        reference = resolve_runtime_contact_reference_for_descent()
        z_remaining = update_phase2_z_descent_state(
            reference["measured_world"],
            source=reference["control_reference_source"],
        )
        return bool(z_remaining <= z_completion_tolerance)

    result = _execute_dualarmik_servo_phase(
        ServoSpec(
            phase_name,
            nominal_target_pose_base,
            float(args.phase2_close_real_center_tolerance),
            float(args.phase2_close_orientation_tolerance),
            int(args.phase2_final_descent_ticks),
        ),
        ik_solver=ik_solver,
        dc=dc,
        articulation=articulation,
        arm_dofs=arm_dofs,
        arm_side=arm_side,
        coord_transform=coord_transform,
        gripper_dofs=gripper_dofs,
        sim_app=sim_app,
        args=args,
        counter=counter,
        phase_log=phase_log,
        end_effector_name=end_effector_name,
        end_effector_path=end_effector_path,
        end_effector_policy=end_effector_policy,
        target_pose_fn=contact_target_pose_fn,
        position_metric_offset_local=contact_control_offset,
        position_metric_label="contact_reference_world",
        completion_condition_fn=phase2_z_descent_completion_fn if z_completion_enforced else None,
        completion_condition_label=(
            f"{phase_name}_runtime_contact_reference_z_reached_locked_target"
            if z_completion_enforced
            else None
        ),
        per_tick_monitor_fn=(lambda: monitor_vertical_tip_stop_rule(context="per_tick_monitor"))
        if vertical_tip_stop_rule_active
        else None,
        coord_transform_refresh_fn=coord_transform_refresh_fn,
        ik_overrides=ik_overrides,
        extra_details={
            "phase2_component": "final_descent_local_ik",
            "target_lock_frame": {
                "locked_target_grasp_center_world": locked_target.tolist(),
                "locked_rpy": locked_rpy_arr.tolist(),
                "locked_target_pose_base": nominal_target_pose_base.tolist(),
                "nominal_contact_reference_pose_conversion": nominal_target_pose_log,
                "point_b_offset_local": point_b_offset.tolist(),
                "point_B_proxy_world_at_phase_start": initial_point_b_world.tolist(),
                "contact_control_offset_local": contact_control_offset.tolist(),
                "contact_control_reference_source": contact_control_reference_source,
                "contact_control_reference_world_at_phase_start": contact_control_world.tolist(),
                "contact_control_reference_log": initial_real_center_log,
                "point_B_proxy_to_contact_control_reference_delta_world": point_b_to_contact_control_delta.tolist(),
                "point_B_proxy_to_contact_control_reference_delta_world_norm_m": float(np.linalg.norm(point_b_to_contact_control_delta)),
                "z_descent_completion": phase2_z_descent_state,
                "point_B_proxy_semantics": "legacy proxy kept for compatibility and diagnostics only; final descent commands the close-critical contact reference when resolved",
                "contact_centric_command_path": "measure runtime tip-mid/contact reference, clamp contact-reference XY/yaw/Z, then convert contact reference to EE pose for IK",
                "vertical_tip_stop_rule_active": bool(vertical_tip_stop_rule_active),
                "vertical_tip_stop_rule_threshold_m": vertical_tip_stop_rule_threshold,
                "vertical_tip_stop_rule_table_frame_source": None if table_frame is None else table_frame.get("source"),
            },
            "object_grasp_frame": object_grasp_frame,
            "selected_candidate_filter": selected_candidate_filter,
            "local_step_limits": {
                "xy_step_m": xy_step_max,
                "z_step_m": z_step_max,
                "yaw_step_rad": yaw_step_max,
            },
            "control_rule": "contact-centric first: measure fingertip/link midpoint contact reference, clamp XY, clamp yaw, descend monotonically in Z, then convert to EE pose for DualArmIK",
        },
    )
    xy_drift_stable = True
    max_xy_step = 0.0
    recent_xy_drift = 0.0
    if samples:
        max_xy_step = max(float(np.linalg.norm(np.array(sample["xy_step_command_m"], dtype=float))) for sample in samples)
        recent_xy_drift = max(float(np.linalg.norm(np.array(sample["xy_step_command_m"], dtype=float))) for sample in samples[-3:])
        xy_drift_stable = bool(max_xy_step <= float(args.phase2_close_xy_drift_max))
    support_gap_values = [
        float(sample["measured_support_gap_m"])
        for sample in samples
        if _finite_float_or_none(sample.get("measured_support_gap_m")) is not None
    ]
    min_descent_samples = max(2, int(getattr(args, "phase2_vertical_fallback_min_descent_samples", PHASE2_VERTICAL_FALLBACK_MIN_DESCENT_SAMPLES)))
    latest_support_gap = support_gap_values[-1] if support_gap_values else None
    min_measured_support_gap = min(support_gap_values) if support_gap_values else None
    recent_z_gap_change = None
    recent_z_progress = None
    recent_z_motion_abs = None
    stalled_in_z = False
    z_stall_window_sample_count = 0
    if len(support_gap_values) >= min_descent_samples:
        z_stall_window_sample_count = min_descent_samples
        earlier_support_gap = support_gap_values[-min_descent_samples]
        latest_gap = support_gap_values[-1]
        recent_z_gap_change = float(latest_gap - earlier_support_gap)
        recent_z_progress = float(max(earlier_support_gap - latest_gap, 0.0))
        recent_z_motion_abs = float(abs(recent_z_gap_change))
        stalled_in_z = bool(
            recent_z_progress
            <= float(getattr(args, "phase2_vertical_fallback_recent_z_progress_max", PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M))
        )
    phase2_log = {
        "sample_count": len(samples),
        "target_pose_eval_count": int(call_count),
        "samples": samples,
        "proxy_middle_point_marker_path": DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH,
        "proxy_middle_point_marker_update_count": len(proxy_middle_point_debug_samples),
        "proxy_middle_point_marker_last_update": None if not proxy_middle_point_debug_samples else proxy_middle_point_debug_samples[-1],
        "vertical_tip_stop_rule_active": bool(vertical_tip_stop_rule_active),
        "vertical_tip_stop_rule_samples": vertical_tip_stop_samples,
        "vertical_tip_stop_rule_sample_count": len(vertical_tip_stop_samples),
        "vertical_tip_stop_rule_last_sample": vertical_tip_stop_rule_last_sample,
        "vertical_tip_stop_rule_triggered": bool(vertical_tip_reached_table_z0),
        "vertical_tip_stop_rule_threshold_m": vertical_tip_stop_rule_threshold,
        "vertical_tip_stop_rule_source": None
        if vertical_tip_stop_rule_last_sample is None
        else vertical_tip_stop_rule_last_sample.get("vertical_tip_stop_rule_source"),
        "vertical_tip_reached_table_z0": bool(vertical_tip_reached_table_z0),
        "vertical_tip_close_stop_reason": vertical_tip_close_stop_reason,
        "max_xy_step_m": max_xy_step,
        "recent_xy_drift_m": recent_xy_drift,
        "xy_drift_stable_by_logged_samples": xy_drift_stable,
        "xy_drift_threshold_m": float(args.phase2_close_xy_drift_max),
        "latest_measured_support_gap_m": latest_support_gap,
        "min_measured_support_gap_m": min_measured_support_gap,
        "recent_z_gap_change_m": recent_z_gap_change,
        "recent_z_progress_m": recent_z_progress,
        "recent_z_motion_abs_m": recent_z_motion_abs,
        "stalled_in_z": stalled_in_z,
        "stalled_in_z_by_logged_samples": stalled_in_z,
        "z_stall_window_sample_count": z_stall_window_sample_count,
        "z_stall_min_required_samples": min_descent_samples,
        "z_stall_recent_progress_threshold_m": float(
            getattr(args, "phase2_vertical_fallback_recent_z_progress_max", PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M)
        ),
        "z_progress_stall_policy": "near-contact commit treats <= threshold downward support-gap progress as stalled, not as a descent failure",
        "object_support_z_world": object_support_z,
        "monotonic_z_policy": "commanded target z never increases during local descent",
    }
    result["vertical_tip_reached_table_z0"] = bool(vertical_tip_reached_table_z0)
    result["vertical_tip_close_stop_reason"] = vertical_tip_close_stop_reason
    result["vertical_tip_stop_rule_threshold_m"] = vertical_tip_stop_rule_threshold
    result["vertical_tip_stop_rule_source"] = None if vertical_tip_stop_rule_last_sample is None else vertical_tip_stop_rule_last_sample.get("vertical_tip_stop_rule_source")
    result["vertical_tip_stop_rule_last_sample"] = vertical_tip_stop_rule_last_sample
    result["vertical_tip_stop_rule_samples"] = vertical_tip_stop_samples
    result["proxy_middle_point_debug_samples"] = proxy_middle_point_debug_samples
    result["proxy_middle_point_debug_sample_count"] = len(proxy_middle_point_debug_samples)
    result["proxy_middle_point_debug_marker_path"] = DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH
    result["phase2_local_descent"] = phase2_log
    if phase_log and phase_log[-1].get("phase") == phase_name:
        phase_log[-1]["details"]["phase2_local_descent"] = phase2_log
    return result


def evaluate_close_gate(
    *,
    pre_close_gate: dict[str, Any],
    object_grasp_frame: dict[str, Any],
    selected_candidate_filter: dict[str, Any],
    descent_result: dict[str, Any],
    motion_policy: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    close_error = float(pre_close_gate.get("close_critical_error_before_close_m", math.inf))
    close_runtime_metrics = pre_close_gate.get("close_runtime_metrics", {})
    close_runtime_metrics = close_runtime_metrics if isinstance(close_runtime_metrics, dict) else {}
    runtime_tip_mid_error = _finite_float_or_none(close_runtime_metrics.get("tip_mid_error_to_object_grasp_center_m"))
    runtime_tip_mid_is_trusted = bool(close_runtime_metrics.get("primary_runtime_truth", False))
    if runtime_tip_mid_error is not None and runtime_tip_mid_is_trusted:
        close_error = runtime_tip_mid_error
    orientation_error = float(descent_result.get("final_rotation_error_rad", descent_result.get("best_rotation_error_rad", math.inf)))
    geometric_filter = selected_candidate_filter or {}
    candidate_pass_flags = dict(geometric_filter.get("pass_flags", {}))
    lateral_symmetry_error = float(geometric_filter.get("symmetry_error_m", math.inf))
    predicted_asymmetry = float(geometric_filter.get("predicted_contact_asymmetry_m", math.inf))
    table_clearance_margin = float(geometric_filter.get("table_clearance_margin_m", math.inf))
    alignment_error_rad = _finite_float_or_none(
        geometric_filter.get("alignment_error_rad", geometric_filter.get("alignment_error", math.nan))
    )
    if alignment_error_rad is None:
        alignment_error_rad = (
            _finite_float_or_none(candidate_pass_flags.get("alignment_error_rad"))
            if isinstance(candidate_pass_flags, dict)
            else None
        )
    local_descent = descent_result.get("phase2_local_descent", {})
    samples = local_descent.get("samples", []) if isinstance(local_descent, dict) else []
    recent_xy_drift = 0.0
    if samples:
        recent_xy_drift = max(float(np.linalg.norm(np.array(sample.get("xy_step_command_m", [0.0, 0.0]), dtype=float))) for sample in samples[-3:])
    else:
        recent_xy_drift = float(descent_result.get("final_point_B_position_error_world_m") or descent_result.get("final_error", math.inf))
    width_compatibility = geometric_filter.get("width_compatibility") or {}
    width_compatibility_pass = bool(width_compatibility.get("pass", True))
    width_on_closing_axis_m = _finite_float_or_none(width_compatibility.get("width_on_closing_axis_m"))

    # Runtime-first commit gate: prioritize live grasp-center distance over precomputed
    # candidate quality scores. Candidate-stage metrics now contribute warnings only.
    close_commit_zone_tolerance = float(args.phase2_close_real_center_tolerance)
    close_commit_zone_pass = bool(math.isfinite(close_error) and close_error <= close_commit_zone_tolerance)
    close_commit_zone_source = (
        "runtime_two_finger_tip_mid_to_object_grasp_center"
        if runtime_tip_mid_error is not None and runtime_tip_mid_is_trusted
        else pre_close_gate.get("close_critical_metric")
    )
    catastrophic_orientation_tolerance = float(
        getattr(args, "catastrophic_orientation_error_max_rad", PHASE2_CATASTROPHIC_ORIENTATION_ERROR_MAX_RAD)
    )
    soft_orientation_tolerance = float(
        getattr(args, "soft_orientation_warning_max_rad", PHASE2_SOFT_ORIENTATION_WARNING_MAX_RAD)
    )
    catastrophic_table_clearance_min = float(
        getattr(args, "catastrophic_table_clearance_min_m", PHASE2_CATASTROPHIC_TABLE_CLEARANCE_MIN_M)
    )
    nominal_table_clearance_min = float(getattr(args, "phase2_table_clearance_min", PHASE2_TABLE_CLEARANCE_MIN_M))
    if str(motion_policy) == "far_low_side_B_driven":
        table_clearance_min = PHASE2_HORIZONTAL_TABLE_CLEARANCE_MIN_M
        table_clearance_policy = "horizontal_far_low_side_allows_small_negative_margin"
    else:
        table_clearance_min = nominal_table_clearance_min
        table_clearance_policy = "default_phase2_table_clearance_min"
    table_clearance_pass = bool(table_clearance_margin >= table_clearance_min)
    catastrophic_table_clearance_pass = bool(table_clearance_margin >= catastrophic_table_clearance_min)

    hard_blockers = {
        "close_commit_zone_pass": close_commit_zone_pass,
        "table_clearance_pass": table_clearance_pass,
        "catastrophic_table_clearance_pass": catastrophic_table_clearance_pass,
        "width_compatibility_pass": width_compatibility_pass,
        "catastrophic_orientation_pass": bool(orientation_error <= catastrophic_orientation_tolerance),
    }
    soft_warnings = {
        "moderate_orientation_warning_pass": bool(orientation_error <= soft_orientation_tolerance),
        "alignment_pass": bool(candidate_pass_flags.get("alignment_pass", True)),
        "lateral_symmetry_pass": bool(lateral_symmetry_error <= float(args.phase2_symmetry_error_max)),
        "predicted_contact_asymmetry_pass": bool(predicted_asymmetry <= float(args.phase2_contact_asymmetry_max)),
        "recent_xy_drift_stable_pass": bool(recent_xy_drift <= float(args.phase2_close_xy_drift_max)),
    }
    hard_fail_reasons = [name for name, passed in hard_blockers.items() if not passed]
    soft_warning_reasons = [name for name, passed in soft_warnings.items() if not passed]
    pass_breakdown = {
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_warnings,
    }
    return {
        "gate_name": "phase2_multi_condition_close_gate",
        "condition_met": bool(all(hard_blockers.values())),
        "pass_flags": pass_breakdown,
        "hard_blockers": hard_blockers,
        "soft_warning_flags": soft_warnings,
        "hard_fail_reasons": hard_fail_reasons,
        "soft_warning_reasons": soft_warning_reasons,
        "mandatory_filter_pass_flags": candidate_pass_flags,
        "fail_reasons": hard_fail_reasons,
        "hard_condition_passed": bool(all(hard_blockers.values())),
        "soft_warning_present": bool(len(soft_warning_reasons) > 0),
        "commit_zone_error_m": close_error,
        "close_commit_zone_tolerance_m": close_commit_zone_tolerance,
        "close_commit_zone_source": close_commit_zone_source,
        "close_runtime_metrics": close_runtime_metrics,
        "tip_mid_error_to_object_grasp_center_m": runtime_tip_mid_error,
        "runtime_tip_mid_error_trusted_for_close": runtime_tip_mid_is_trusted,
        "tip_mid_xy_error_m": close_runtime_metrics.get("tip_mid_xy_error_m"),
        "tip_mid_z_error_m": close_runtime_metrics.get("tip_mid_z_error_m"),
        "tip_axis_alignment_error_rad": close_runtime_metrics.get("tip_axis_alignment_error_rad"),
        "tip_symmetry_error_m": close_runtime_metrics.get("tip_symmetry_error_m"),
        "tip_z_asymmetry_m": close_runtime_metrics.get("tip_z_asymmetry_m"),
        "runtime_two_finger_geometry_primary": bool(close_runtime_metrics.get("primary_runtime_truth", False)),
        "close_critical_metric": pre_close_gate.get("close_critical_metric"),
        "close_critical_uses_real_grasp_center": pre_close_gate.get("close_critical_uses_real_grasp_center"),
        "real_grasp_center_error_m": close_error,
        "real_grasp_center_tolerance_m": float(args.phase2_close_real_center_tolerance),
        "commit_decision_mode": "runtime_grasp_center_dominant_with_soft_candidate_warning",
        "lateral_symmetry_error_m": lateral_symmetry_error,
        "lateral_symmetry_tolerance_m": float(args.phase2_symmetry_error_max),
        "predicted_contact_asymmetry_m": predicted_asymmetry,
        "predicted_contact_asymmetry_tolerance_m": float(args.phase2_contact_asymmetry_max),
        "table_clearance_margin_m": table_clearance_margin,
        "table_clearance_min_m": table_clearance_min,
        "table_clearance_pass": table_clearance_pass,
        "table_clearance_policy": table_clearance_policy,
        "motion_policy": motion_policy,
        "nominal_table_clearance_min_m": nominal_table_clearance_min,
        "catastrophic_table_clearance_pass": catastrophic_table_clearance_pass,
        "catastrophic_table_clearance_min_m": catastrophic_table_clearance_min,
        "width_compatibility_m": width_on_closing_axis_m,
        "width_compatibility_pass": width_compatibility_pass,
        "orientation_error_rad": orientation_error,
        "orientation_error_tolerance_rad": catastrophic_orientation_tolerance,
        "catastrophic_orientation_error_max_rad": catastrophic_orientation_tolerance,
        "soft_orientation_warning_max_rad": soft_orientation_tolerance,
        "moderate_orientation_warning_pass": bool(orientation_error <= soft_orientation_tolerance),
        "alignment_error_pass": bool(candidate_pass_flags.get("alignment_pass", True)),
        "alignment_tolerance_rad": float(getattr(args, "phase2_alignment_error_max", PHASE2_ALIGNMENT_ERROR_MAX_RAD)),
        "alignment_error_rad": alignment_error_rad,
        "recent_xy_drift_m": recent_xy_drift,
        "recent_xy_drift_tolerance_m": float(args.phase2_close_xy_drift_max),
        "object_grasp_frame": object_grasp_frame,
    }


def evaluate_vertical_support_or_stall_close_fallback(
    *,
    pre_close_gate: dict[str, Any],
    descent_result: dict[str, Any],
    motion_policy: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Conservative vertical-only fallback when support-limited descent stalls."""
    local_descent = descent_result.get("phase2_local_descent", {}) if isinstance(descent_result, dict) else {}
    local_descent = local_descent if isinstance(local_descent, dict) else {}
    samples = local_descent.get("samples", []) if isinstance(local_descent.get("samples", []), list) else []
    recent_xy_drift = _finite_float_or_none(local_descent.get("recent_xy_drift_m"))
    if recent_xy_drift is None:
        recent_xy_drift = 0.0
        if samples:
            recent_xy_drift = max(
                float(np.linalg.norm(np.array(sample.get("xy_step_command_m", [0.0, 0.0]), dtype=float)))
                for sample in samples[-3:]
            )
        else:
            recent_xy_drift = math.inf
    orientation_error = _finite_float_or_none(
        descent_result.get("final_rotation_error_rad", descent_result.get("best_rotation_error_rad", math.inf))
    )
    orientation_error = math.inf if orientation_error is None else orientation_error
    support_gap_sources: list[dict[str, Any]] = []

    def add_support_gap_source(source: str, value: Any, *, eligible_for_gate: bool = True) -> None:
        gap = _finite_float_or_none(value)
        if gap is None:
            return
        support_gap_sources.append(
            {
                "source": source,
                "gap_m": gap,
                "eligible_for_gate": bool(eligible_for_gate),
            }
        )

    add_support_gap_source(
        "pre_close_close_critical_gap",
        pre_close_gate.get("vertical_actual_close_critical_gap_above_support_m"),
    )
    add_support_gap_source(
        "pre_close_real_grasp_center_gap",
        pre_close_gate.get("vertical_actual_real_grasp_center_gap_above_support_m"),
    )
    add_support_gap_source(
        "pre_close_point_B_proxy_gap",
        pre_close_gate.get("vertical_actual_point_B_gap_above_support_m"),
    )
    add_support_gap_source(
        "phase2_descent_latest_measured_support_gap",
        local_descent.get("latest_measured_support_gap_m"),
    )
    add_support_gap_source(
        "phase2_descent_min_measured_support_gap",
        local_descent.get("min_measured_support_gap_m"),
        eligible_for_gate=False,
    )

    support_gap_min = float(getattr(args, "phase2_vertical_fallback_support_gap_min", PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MIN_M))
    support_gap_max = float(getattr(args, "phase2_vertical_fallback_support_gap_max", PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MAX_M))
    support_gap_pass_sources = [
        source
        for source in support_gap_sources
        if source["eligible_for_gate"] and support_gap_min <= float(source["gap_m"]) <= support_gap_max
    ]
    selected_support_gap = None
    if support_gap_pass_sources:
        selected_support_gap = min(support_gap_pass_sources, key=lambda source: abs(float(source["gap_m"])))
    elif support_gap_sources:
        selected_support_gap = min(support_gap_sources, key=lambda source: abs(float(source["gap_m"])))

    recent_z_progress = _finite_float_or_none(local_descent.get("recent_z_progress_m"))
    recent_z_motion_abs = _finite_float_or_none(local_descent.get("recent_z_motion_abs_m"))
    z_stall_window_sample_count = int(local_descent.get("z_stall_window_sample_count", 0) or 0)
    min_samples = max(
        2,
        int(getattr(args, "phase2_vertical_fallback_min_descent_samples", PHASE2_VERTICAL_FALLBACK_MIN_DESCENT_SAMPLES)),
    )
    z_sample_count_pass = bool(z_stall_window_sample_count >= min_samples)
    stalled_in_z = bool(local_descent.get("stalled_in_z", False) and z_sample_count_pass)
    motion_policy_token = str(motion_policy).lower()
    far_motion_policy = motion_policy_token.startswith("far")
    far_stalled_fallback_allowed = bool(
        far_motion_policy
        and getattr(args, "phase2_vertical_fallback_allow_far_policy", PHASE2_VERTICAL_FALLBACK_ALLOW_FAR_POLICY)
    )
    vertical_motion_policy = bool((not far_motion_policy) or far_stalled_fallback_allowed)
    orientation_tolerance = float(
        getattr(args, "phase2_vertical_fallback_orientation_tolerance", PHASE2_VERTICAL_FALLBACK_ORIENTATION_TOLERANCE_RAD)
    )
    conditions = {
        "fallback_enabled": bool(getattr(args, "phase2_vertical_fallback_close_enable", True)),
        "vertical_motion_policy": bool(vertical_motion_policy),
        "far_stalled_fallback_policy_pass": bool((not far_motion_policy) or far_stalled_fallback_allowed),
        "support_gap_small_pass": bool(len(support_gap_pass_sources) > 0),
        "z_stall_sample_count_pass": z_sample_count_pass,
        "stalled_in_z_pass": stalled_in_z,
        "recent_xy_drift_stable_pass": bool(recent_xy_drift <= float(args.phase2_close_xy_drift_max)),
        "orientation_error_pass": bool(orientation_error <= orientation_tolerance),
    }
    fail_reasons = [name for name, passed in conditions.items() if not passed]
    return {
        "gate_name": "vertical_support_or_stall_close_fallback",
        "condition_met": bool(all(conditions.values())),
        "pass_flags": conditions,
        "fail_reasons": fail_reasons,
        "motion_policy": motion_policy,
        "far_motion_policy": bool(far_motion_policy),
        "far_stalled_fallback_allowed": bool(far_stalled_fallback_allowed),
        "support_gap_sources": support_gap_sources,
        "support_gap_pass_sources": support_gap_pass_sources,
        "selected_support_gap_source": None if selected_support_gap is None else selected_support_gap.get("source"),
        "selected_support_gap_m": None if selected_support_gap is None else selected_support_gap.get("gap_m"),
        "support_gap_min_m": support_gap_min,
        "support_gap_max_m": support_gap_max,
        "recent_z_progress_m": recent_z_progress,
        "recent_z_motion_abs_m": recent_z_motion_abs,
        "recent_z_progress_threshold_m": float(
            getattr(args, "phase2_vertical_fallback_recent_z_progress_max", PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M)
        ),
        "z_stall_window_sample_count": z_stall_window_sample_count,
        "z_stall_min_required_samples": min_samples,
        "recent_xy_drift_m": recent_xy_drift,
        "recent_xy_drift_tolerance_m": float(args.phase2_close_xy_drift_max),
        "orientation_error_rad": orientation_error,
        "orientation_error_tolerance_rad": orientation_tolerance,
        "primary_close_orientation_tolerance_rad": float(args.phase2_close_orientation_tolerance),
        "close_policy": "fallback only bypasses close-critical distance for support/stall hover; far policy may use it when support gap is small, Z is stalled, XY drift is bounded, and fallback-local orientation sanity passes",
    }


def evaluate_runtime_commit_fallback(
    *,
    pre_close_gate: dict[str, Any],
    descent_result: dict[str, Any],
    selected_candidate_filter: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generic near-enough stalled close fallback for non-vertical-specific hover cases."""
    close_runtime_metrics = pre_close_gate.get("close_runtime_metrics", {})
    close_runtime_metrics = close_runtime_metrics if isinstance(close_runtime_metrics, dict) else {}
    local_descent = descent_result.get("phase2_local_descent", {}) if isinstance(descent_result, dict) else {}
    local_descent = local_descent if isinstance(local_descent, dict) else {}
    samples = local_descent.get("samples", []) if isinstance(local_descent.get("samples", []), list) else []
    geometric_filter = selected_candidate_filter or {}
    width_compatibility = geometric_filter.get("width_compatibility") or {}
    table_clearance_margin = _finite_float_or_none(geometric_filter.get("table_clearance_margin_m"))
    if table_clearance_margin is None:
        table_clearance_margin = math.inf

    tip_mid_error = _finite_float_or_none(close_runtime_metrics.get("tip_mid_error_to_object_grasp_center_m"))
    runtime_tip_mid_is_trusted = bool(close_runtime_metrics.get("primary_runtime_truth", False))
    if tip_mid_error is None or not runtime_tip_mid_is_trusted:
        tip_mid_error = _finite_float_or_none(pre_close_gate.get("close_critical_error_before_close_m"))
    recent_xy_drift = _finite_float_or_none(local_descent.get("recent_xy_drift_m"))
    if recent_xy_drift is None and samples:
        recent_xy_drift = max(
            float(np.linalg.norm(np.array(sample.get("xy_step_command_m", [0.0, 0.0]), dtype=float)))
            for sample in samples[-3:]
        )
    recent_z_progress = _finite_float_or_none(local_descent.get("recent_z_progress_m"))
    sample_count = int(local_descent.get("sample_count", len(samples)) or 0)
    min_samples = max(
        1,
        int(getattr(args, "runtime_commit_fallback_min_samples", PHASE2_RUNTIME_COMMIT_FALLBACK_MIN_SAMPLES)),
    )
    tip_threshold = float(
        getattr(args, "runtime_commit_fallback_tip_mid_error_max_m", PHASE2_RUNTIME_COMMIT_FALLBACK_TIP_MID_ERROR_MAX_M)
    )
    xy_threshold = float(
        getattr(args, "runtime_commit_fallback_recent_xy_drift_max_m", PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_XY_DRIFT_MAX_M)
    )
    z_threshold = float(
        getattr(args, "runtime_commit_fallback_recent_z_progress_max_m", PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_Z_PROGRESS_MAX_M)
    )
    catastrophic_table_clearance_min = float(
        getattr(args, "catastrophic_table_clearance_min_m", PHASE2_CATASTROPHIC_TABLE_CLEARANCE_MIN_M)
    )
    width_possible = bool(width_compatibility.get("pass", True))
    pass_flags = {
        "fallback_enabled": bool(getattr(args, "runtime_commit_fallback_enable", True)),
        "runtime_tip_mid_near_enough_pass": bool(tip_mid_error is not None and tip_mid_error <= tip_threshold),
        "recent_xy_drift_bounded_pass": bool(recent_xy_drift is not None and recent_xy_drift <= xy_threshold),
        "recent_z_progress_stalled_pass": bool(recent_z_progress is not None and recent_z_progress <= z_threshold),
        "sample_count_pass": bool(sample_count >= min_samples),
        "catastrophic_table_clearance_pass": bool(table_clearance_margin >= catastrophic_table_clearance_min),
        "width_possible_pass": width_possible,
    }
    fail_reasons = [name for name, passed in pass_flags.items() if not passed]
    return {
        "gate_name": "generic_runtime_commit_fallback",
        "condition_met": bool(all(pass_flags.values())),
        "pass_flags": pass_flags,
        "fail_reasons": fail_reasons,
        "tip_mid_error_to_object_grasp_center_m": tip_mid_error,
        "runtime_tip_mid_error_trusted_for_fallback": runtime_tip_mid_is_trusted,
        "tip_mid_error_threshold_m": tip_threshold,
        "recent_xy_drift_m": recent_xy_drift,
        "recent_xy_drift_threshold_m": xy_threshold,
        "recent_z_progress_m": recent_z_progress,
        "recent_z_progress_threshold_m": z_threshold,
        "sample_count": sample_count,
        "min_samples": min_samples,
        "table_clearance_margin_m": table_clearance_margin,
        "catastrophic_table_clearance_min_m": catastrophic_table_clearance_min,
        "table_clearance_policy": "allow mild tabletop penetration during close commit; only penetration beyond catastrophic threshold blocks",
        "width_compatibility": width_compatibility,
        "close_policy": "commit when runtime two-finger contact geometry is near enough, XY drift is bounded, and Z progress has stalled without catastrophic impossibility",
    }


def build_close_debug_summary(
    *,
    pre_close_gate: dict[str, Any],
    phase2_close_gate: dict[str, Any],
    vertical_support_or_stall_fallback_gate: dict[str, Any],
    generic_runtime_commit_fallback_gate: dict[str, Any],
    final_close_decision: dict[str, Any],
    descent_result: dict[str, Any],
) -> dict[str, Any]:
    metrics = pre_close_gate.get("close_runtime_metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}
    orientation_error = None
    if isinstance(descent_result, dict):
        orientation_error = _finite_float_or_none(
            descent_result.get("final_rotation_error_rad", descent_result.get("best_rotation_error_rad"))
        )
    return {
        "summary_name": "close_debug_summary",
        "tip1_world": metrics.get("tip1_world", pre_close_gate.get("tip1_world")),
        "tip2_world": metrics.get("tip2_world", pre_close_gate.get("tip2_world")),
        "tip_mid_world": metrics.get("tip_mid_world", pre_close_gate.get("tip_mid_world")),
        "object_center_world": metrics.get("object_center_world", pre_close_gate.get("object_center_world")),
        "object_grasp_center_world": metrics.get("object_grasp_center_world"),
        "tip_mid_xy_error_m": metrics.get("tip_mid_xy_error_m"),
        "tip_mid_z_error_m": metrics.get("tip_mid_z_error_m"),
        "tip_mid_error_to_object_grasp_center_m": metrics.get("tip_mid_error_to_object_grasp_center_m"),
        "tip_axis_alignment_error_rad": metrics.get("tip_axis_alignment_error_rad"),
        "tip_symmetry_error_m": metrics.get("tip_symmetry_error_m"),
        "tip_z_asymmetry_m": metrics.get("tip_z_asymmetry_m"),
        "orientation_error_rad": orientation_error,
        "table_clearance_margin_m": phase2_close_gate.get("table_clearance_margin_m"),
        "table_clearance_pass": phase2_close_gate.get("table_clearance_pass"),
        "catastrophic_table_clearance_pass": phase2_close_gate.get("catastrophic_table_clearance_pass"),
        "close_commit_zone_pass": phase2_close_gate.get("hard_blockers", {}).get(
            "close_commit_zone_pass",
            phase2_close_gate.get("pass_flags", {}).get("hard_blockers", {}).get("close_commit_zone_pass"),
        ),
        "generic_runtime_commit_fallback_pass": bool(generic_runtime_commit_fallback_gate.get("condition_met", False)),
        "vertical_support_or_stall_fallback_pass": bool(vertical_support_or_stall_fallback_gate.get("condition_met", False)),
        "final_close_decision": final_close_decision,
        "point_B_proxy_world": pre_close_gate.get("point_B_proxy_world"),
        "point_B_is_final_truth": False,
        "runtime_reference_source": metrics.get("runtime_reference_source", pre_close_gate.get("fingertip_reference_source_used")),
    }


def execute_two_stage_close(
    *,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    object_grasp_frame: dict[str, Any],
    pre_close_gate: dict[str, Any],
    coord_transform_refresh_fn: Callable[[], dict[str, Any]] | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    gap_plan = compute_target_gap(object_grasp_frame, args)
    stage_a_ok = _command_gripper_phase(
        "phase2_gap_close_stage_a",
        dc=dc,
        gripper_dofs=gripper_dofs,
        end_effector_body=end_effector_body,
        end_effector_name=end_effector_name,
        end_effector_path=end_effector_path,
        end_effector_policy=end_effector_policy,
        coord_transform_refresh_fn=coord_transform_refresh_fn,
        target_positions=[float(gap_plan["stage_a_joint_target"])] * len(gripper_dofs),
        sim_app=sim_app,
        steps=args.gripper_steps,
        counter=counter,
        phase_log=phase_log,
        skipped=skipped,
        effort_value=args.gripper_hold_effort,
        extra_details={
            "phase2_component": "execute_two_stage_close_stage_a",
            "gap_plan": gap_plan,
            "pre_close_gate": pre_close_gate,
        },
    )
    stage_b_ok = _command_gripper_phase(
        "phase2_retention_hold_stage_b",
        dc=dc,
        gripper_dofs=gripper_dofs,
        end_effector_body=end_effector_body,
        end_effector_name=end_effector_name,
        end_effector_path=end_effector_path,
        end_effector_policy=end_effector_policy,
        coord_transform_refresh_fn=coord_transform_refresh_fn,
        target_positions=[float(gap_plan["stage_b_joint_target"])] * len(gripper_dofs),
        sim_app=sim_app,
        steps=int(args.phase2_retention_steps),
        counter=counter,
        phase_log=phase_log,
        skipped=skipped,
        effort_value=args.gripper_hold_effort,
        extra_details={
            "phase2_component": "execute_two_stage_close_stage_b",
            "gap_plan": gap_plan,
            "retention_status": "hold_effort_commanded" if not skipped else "skipped_by_cli",
            "retention_effort": float(args.gripper_hold_effort),
        },
    )
    return {
        "condition_met": bool(stage_a_ok and stage_b_ok),
        "stage_a_ok": bool(stage_a_ok),
        "stage_b_ok": bool(stage_b_ok),
        "gap_plan": gap_plan,
        "retention_status": "ok" if stage_a_ok and stage_b_ok else "failed",
    }


def verify_short_lift(
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
) -> dict[str, Any]:
    before = _bbox_state(stage, target_path)
    before_center = _center_from_bbox(before["bbox"])
    lift_pose = np.array(geometry["micro_lift_pose_base"], dtype=float)
    result = _execute_dualarmik_servo_phase(
        ServoSpec(
            "phase2_short_lift_verify",
            lift_pose,
            args.lift_tolerance,
            args.rot_tolerance,
            int(args.phase2_short_lift_ticks),
            gripper_effort=args.gripper_hold_effort,
        ),
        ik_solver=ik_solver,
        dc=dc,
        articulation=articulation,
        arm_dofs=arm_dofs,
        arm_side=arm_side,
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
            "phase2_component": "verify_short_lift",
            "object_pose_before_short_lift": before,
            "short_lift_height_m": float(args.phase2_short_lift_height),
        },
    )
    after = _bbox_state(stage, target_path)
    after_center = _center_from_bbox(after["bbox"])
    delta = after_center - before_center
    success = bool(delta[2] >= float(args.phase2_short_lift_min_delta) and result["final_error"] <= args.lift_tolerance)
    if phase_log and phase_log[-1].get("phase") == "phase2_short_lift_verify":
        phase_log[-1]["condition_met"] = success
        phase_log[-1]["details"].update(
            {
                "object_pose_after_short_lift": after,
                "object_delta_during_short_lift_m": delta.tolist(),
                "short_lift_verification_result": success,
                "short_lift_min_delta_m": float(args.phase2_short_lift_min_delta),
                "servo_result_condition_met": result["final_error"] <= args.lift_tolerance,
            }
        )
    return {
        "condition_met": success,
        "before_state": before,
        "after_state": after,
        "object_delta_m": delta.tolist(),
        "short_lift_min_delta_m": float(args.phase2_short_lift_min_delta),
        "servo_result": result,
    }


def recover_and_retry(
    *,
    reason: str,
    dc: Any,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    selected_candidate: dict[str, Any] | None,
    candidate_selection: dict[str, Any] | None,
) -> dict[str, Any]:
    start_step = counter["step"]
    if gripper_dofs:
        _send_position_targets(dc, gripper_dofs, [OFFICIAL_GRIPPER_OPEN_WIDTH] * len(gripper_dofs))
    _run_updates(sim_app, max(1, int(args.gripper_steps // 2)), counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=0.0)
    current_id = None if selected_candidate is None else selected_candidate.get("preset_id")
    filtered = [] if candidate_selection is None else list(candidate_selection.get("filtered_candidates", []))
    remaining = [
        candidate
        for candidate in filtered
        if candidate.get("preset_id") != current_id and bool(candidate.get("valid", False)) and bool(candidate.get("phase2_filter_pass", False))
    ]
    retry_budget = int(args.phase2_max_retries)
    retry_log = {
        "reason": reason,
        "current_candidate": current_id,
        "retry_budget": retry_budget,
        "remaining_passed_candidate_count": len(remaining),
        "next_candidate": remaining[0] if remaining else None,
        "retry_recommended": bool(retry_budget > 0 and remaining),
        "recovery_action": "opened_gripper_and_logged_next_candidate",
        "note": "Phase 2 keeps the existing linear phase machine; retry re-entry is logged deterministically for the next validation pass.",
    }
    _append_phase(
        phase_log,
        phase="phase2_recover_and_retry",
        start_step=start_step,
        end_step=counter["step"],
        condition_met=bool(retry_log["retry_recommended"]),
        details=retry_log,
    )
    return retry_log


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
    parser.add_argument("--horizontal-descent-xy-trigger-tolerance", type=float, default=DEFAULT_HORIZONTAL_DESCENT_XY_TRIGGER_TOLERANCE)
    parser.add_argument("--far-point-b-forward-extension", type=float, default=DEFAULT_FAR_POINT_B_FORWARD_EXTENSION)
    parser.add_argument("--far-point-a-extra-height-clearance", type=float, default=DEFAULT_FAR_POINT_A_EXTRA_HEIGHT_CLEARANCE)
    parser.add_argument("--far-ab-downward-slant-deg", type=float, default=DEFAULT_FAR_AB_DOWNWARD_SLANT_DEG)
    parser.add_argument("--far-outboard-transition-offset", type=float, default=DEFAULT_FAR_OUTBOARD_TRANSITION_OFFSET)
    parser.add_argument("--far-outboard-transition-clearance", type=float, default=DEFAULT_FAR_OUTBOARD_TRANSITION_CLEARANCE)
    parser.add_argument("--far-null-weight", type=float, default=DEFAULT_FAR_NULL_WEIGHT)
    parser.add_argument("--far-outboard-shoulder-roll-bias", type=float, default=DEFAULT_FAR_OUTBOARD_SHOULDER_ROLL_BIAS)
    parser.add_argument("--pre-close-point-b-tolerance", type=float, default=DEFAULT_PRE_CLOSE_POINT_B_TOLERANCE)
    parser.add_argument("--no-live-coordinate-transform", action="store_true")
    parser.add_argument("--vertical-point-b-gap-above-support", type=float, default=DEFAULT_VERTICAL_POINT_B_GAP_ABOVE_SUPPORT)
    parser.add_argument("--vertical-close-point-b-tolerance", type=float, default=DEFAULT_VERTICAL_CLOSE_POINT_B_TOLERANCE)
    parser.add_argument("--vertical-xy-reference-link", default=DEFAULT_VERTICAL_XY_REFERENCE_LINK)
    parser.add_argument("--vertical-xy-reference-tolerance", type=float, default=DEFAULT_VERTICAL_XY_REFERENCE_TOLERANCE)
    parser.add_argument("--vertical-arm-lateral-bias-correction", type=float, default=DEFAULT_VERTICAL_ARM_LATERAL_BIAS_CORRECTION)
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
    parser.add_argument("--phase2-alignment-error-max", type=float, default=PHASE2_ALIGNMENT_ERROR_MAX_RAD)
    parser.add_argument("--phase2-symmetry-error-max", type=float, default=PHASE2_SYMMETRY_ERROR_MAX_M)
    parser.add_argument("--phase2-contact-asymmetry-max", type=float, default=PHASE2_CONTACT_ASYMMETRY_MAX_M)
    parser.add_argument("--phase2-table-clearance-min", type=float, default=PHASE2_TABLE_CLEARANCE_MIN_M)
    parser.add_argument("--phase2-width-min", type=float, default=PHASE2_WIDTH_MIN_M)
    parser.add_argument("--phase2-width-max", type=float, default=PHASE2_WIDTH_MAX_M)
    parser.add_argument(
        "--phase2-allow-least-bad-candidate",
        action=argparse.BooleanOptionalAction,
        default=PHASE2_ALLOW_LEAST_BAD_CANDIDATE,
    )
    parser.add_argument("--phase2-final-descent-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase2-final-descent-ticks", type=int, default=PHASE2_DESCENT_MAX_TICKS)
    parser.add_argument("--phase2-descent-xy-step", type=float, default=PHASE2_DESCENT_XY_STEP_M)
    parser.add_argument("--phase2-descent-z-step", type=float, default=PHASE2_DESCENT_Z_STEP_M)
    parser.add_argument("--phase2-descent-yaw-step", type=float, default=PHASE2_DESCENT_YAW_STEP_RAD)
    parser.add_argument("--phase2-close-real-center-tolerance", type=float, default=PHASE2_CLOSE_REAL_CENTER_TOLERANCE_M)
    parser.add_argument("--phase2-close-orientation-tolerance", type=float, default=PHASE2_CLOSE_ORIENTATION_TOLERANCE_RAD)
    parser.add_argument("--catastrophic-orientation-error-max-rad", type=float, default=PHASE2_CATASTROPHIC_ORIENTATION_ERROR_MAX_RAD)
    parser.add_argument("--soft-orientation-warning-max-rad", type=float, default=PHASE2_SOFT_ORIENTATION_WARNING_MAX_RAD)
    parser.add_argument("--catastrophic-table-clearance-min-m", type=float, default=PHASE2_CATASTROPHIC_TABLE_CLEARANCE_MIN_M)
    parser.add_argument("--runtime-commit-fallback-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-commit-fallback-tip-mid-error-max-m", type=float, default=PHASE2_RUNTIME_COMMIT_FALLBACK_TIP_MID_ERROR_MAX_M)
    parser.add_argument("--runtime-commit-fallback-recent-z-progress-max-m", type=float, default=PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_Z_PROGRESS_MAX_M)
    parser.add_argument("--runtime-commit-fallback-recent-xy-drift-max-m", type=float, default=PHASE2_RUNTIME_COMMIT_FALLBACK_RECENT_XY_DRIFT_MAX_M)
    parser.add_argument("--runtime-commit-fallback-min-samples", type=int, default=PHASE2_RUNTIME_COMMIT_FALLBACK_MIN_SAMPLES)
    parser.add_argument("--close-debug-summary-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-two-finger-marker-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase2-close-xy-drift-max", type=float, default=PHASE2_CLOSE_XY_DRIFT_MAX_M)
    parser.add_argument("--phase2-vertical-fallback-close-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase2-vertical-fallback-support-gap-max", type=float, default=PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MAX_M)
    parser.add_argument("--phase2-vertical-fallback-support-gap-min", type=float, default=PHASE2_VERTICAL_FALLBACK_SUPPORT_GAP_MIN_M)
    parser.add_argument("--phase2-vertical-fallback-recent-z-progress-max", type=float, default=PHASE2_VERTICAL_FALLBACK_RECENT_Z_PROGRESS_MAX_M)
    parser.add_argument("--phase2-vertical-fallback-min-descent-samples", type=int, default=PHASE2_VERTICAL_FALLBACK_MIN_DESCENT_SAMPLES)
    parser.add_argument("--phase2-vertical-fallback-orientation-tolerance", type=float, default=PHASE2_VERTICAL_FALLBACK_ORIENTATION_TOLERANCE_RAD)
    parser.add_argument(
        "--phase2-vertical-fallback-allow-far-policy",
        action=argparse.BooleanOptionalAction,
        default=PHASE2_VERTICAL_FALLBACK_ALLOW_FAR_POLICY,
    )
    parser.add_argument(
        "--phase2-diagnostic-finger-link-midpoint-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--phase2-vertical-tip-table-z-close-threshold", type=float, default=PHASE2_VERTICAL_TIP_TABLE_Z_CLOSE_THRESHOLD_M)
    parser.add_argument("--phase2-min-target-gap", type=float, default=PHASE2_MIN_TARGET_GAP_M)
    parser.add_argument("--phase2-target-gap-margin", type=float, default=PHASE2_TARGET_GAP_MARGIN_M)
    parser.add_argument("--phase2-close-stage-a-fraction", type=float, default=PHASE2_CLOSE_STAGE_A_FRACTION)
    parser.add_argument("--phase2-retention-steps", type=int, default=PHASE2_RETENTION_STEPS)
    parser.add_argument("--phase2-short-lift-height", type=float, default=PHASE2_SHORT_LIFT_HEIGHT_M)
    parser.add_argument("--phase2-short-lift-min-delta", type=float, default=PHASE2_SHORT_LIFT_MIN_DELTA_M)
    parser.add_argument("--phase2-short-lift-ticks", type=int, default=PHASE2_SHORT_LIFT_TICKS)
    parser.add_argument("--phase2-max-retries", type=int, default=PHASE2_MAX_RETRIES)
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
    if args.point_b_offset_local is not None and not np.isfinite(np.array(args.point_b_offset_local, dtype=float)).all():
        raise RuntimeError("--point-b-offset-local must contain finite values")
    if args.prim_path is not None and not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")
    args.workspace_x = tuple(float(value) for value in args.workspace_x)
    args.workspace_y = tuple(float(value) for value in args.workspace_y)
    args.workspace_z = tuple(float(value) for value in args.workspace_z)
    phase2_positive_values = {
        "phase2_alignment_error_max": args.phase2_alignment_error_max,
        "phase2_symmetry_error_max": args.phase2_symmetry_error_max,
        "phase2_contact_asymmetry_max": args.phase2_contact_asymmetry_max,
        "phase2_width_min": args.phase2_width_min,
        "phase2_width_max": args.phase2_width_max,
        "phase2_final_descent_ticks": args.phase2_final_descent_ticks,
        "phase2_descent_xy_step": args.phase2_descent_xy_step,
        "phase2_descent_z_step": args.phase2_descent_z_step,
        "phase2_descent_yaw_step": args.phase2_descent_yaw_step,
        "phase2_close_real_center_tolerance": args.phase2_close_real_center_tolerance,
        "phase2_close_orientation_tolerance": args.phase2_close_orientation_tolerance,
        "phase2_close_xy_drift_max": args.phase2_close_xy_drift_max,
        "phase2_min_target_gap": args.phase2_min_target_gap,
        "phase2_target_gap_margin": args.phase2_target_gap_margin,
        "phase2_close_stage_a_fraction": args.phase2_close_stage_a_fraction,
        "phase2_retention_steps": args.phase2_retention_steps,
        "phase2_short_lift_height": args.phase2_short_lift_height,
        "phase2_short_lift_min_delta": args.phase2_short_lift_min_delta,
        "phase2_short_lift_ticks": args.phase2_short_lift_ticks,
    }
    for name, value in phase2_positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(float(args.phase2_table_clearance_min)):
        raise RuntimeError("--phase2-table-clearance-min must be finite")
    if float(args.phase2_table_clearance_min) < float(args.catastrophic_table_clearance_min_m):
        raise RuntimeError("--phase2-table-clearance-min must be >= --catastrophic-table-clearance-min-m")
    if float(args.phase2_width_min) > float(args.phase2_width_max):
        raise RuntimeError("--phase2-width-min must be <= --phase2-width-max")
    if not (0.0 < float(args.phase2_close_stage_a_fraction) <= 1.0):
        raise RuntimeError("--phase2-close-stage-a-fraction must be in (0, 1]")
    if int(args.phase2_max_retries) < 0:
        raise RuntimeError("--phase2-max-retries must be non-negative")
    args.live_coordinate_transform = not bool(args.no_live_coordinate_transform)
    args.ee_frame_compensation_active = False
    args.ee_frame_compensation_by_arm = {}
    args.stop_after_lift = not bool(args.continue_after_lift)

    sys.argv = [sys.argv[0]]
    timestamp_utc = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp_utc": timestamp_utc.isoformat(),
            "timestamp_compact": timestamp_utc.strftime("%Y%m%dT%H%M%SZ"),
            "script_name": SCRIPT_NAME,
            "source_baseline_script": SOURCE_BASELINE_SCRIPT,
            "plan_phase": "Phase 2 - Geometric planner hardening",
            "seed": int(args.seed),
            "gui_enabled": bool(args.no_headless or args.gui),
        },
        "motion_policy": {
            "architecture": "phase_based_official_dualarmik_6d_servo",
            "phase1_hybrid_skeleton_preserved": True,
            "phase2_geometric_hardening_active": True,
            "phase2_scope": "scene_state_bbox_grasp_frame_fast_filter_local_descent_close_gate_two_stage_close_short_lift_recovery_logging",
            "phase2_deferred": [
                "Thinker_advisor",
                "YOLO_provider",
                "heavy_force_closure_optimizer",
                "full_OBB_or_mesh_contact_model",
                "multi_object_retry_loop_rearchitecture",
            ],
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
                "phase2_real_center_tolerance_m": float(args.phase2_close_real_center_tolerance),
                "scope": "all motion policies before two-stage close",
            },
            "phase2_thresholds": {
                "alignment_error_max_rad": float(args.phase2_alignment_error_max),
                "symmetry_error_max_m": float(args.phase2_symmetry_error_max),
                "contact_asymmetry_max_m": float(args.phase2_contact_asymmetry_max),
                "table_clearance_min_m": float(args.phase2_table_clearance_min),
                "width_min_m": float(args.phase2_width_min),
                "width_max_m": float(args.phase2_width_max),
                "close_real_center_tolerance_m": float(args.phase2_close_real_center_tolerance),
                "close_orientation_tolerance_rad": float(args.phase2_close_orientation_tolerance),
                "close_xy_drift_max_m": float(args.phase2_close_xy_drift_max),
                "descent_xy_step_m": float(args.phase2_descent_xy_step),
                "descent_z_step_m": float(args.phase2_descent_z_step),
                "descent_yaw_step_rad": float(args.phase2_descent_yaw_step),
                "retention_steps": int(args.phase2_retention_steps),
                "max_retries": int(args.phase2_max_retries),
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
        table_frame = build_or_resolve_table_frame(
            stage=stage,
            table_path=table_path,
            table_bbox=table_bbox,
            robot_base_position=robot_base_position,
            robot_base_yaw_rad=robot_base_yaw_rad,
        )
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
        forward_base = float(target_components["forward_base"])
        target_region = _classify_target_region(forward_base, args)
        approach_family_order = _approach_family_order_for_region(target_region)
        object_info = get_object_info_in_table_frame(
            stage=stage,
            target_path=target_path,
            target_index=selected_target_index,
            target_category=target_category,
            table_frame=table_frame,
        )
        object_grasp_frame = estimate_object_grasp_frame(object_info, table_frame)
        hybrid_candidates = generate_approach_candidates_for_object(
            object_info=object_info,
            object_grasp_frame=object_grasp_frame,
            table_frame=table_frame,
            target_components=target_components,
            approach_family_order=approach_family_order,
            requested_arm=str(args.arm),
            robot_base_position=robot_base_position,
            args=args,
        )
        object_grasp_center_world = np.array(object_grasp_frame.get("grasp_center_world", []), dtype=float)
        object_grasp_center_marker_path = None
        if object_grasp_center_world.size == 3 and np.isfinite(object_grasp_center_world).all():
            object_grasp_center_marker_path = _upsert_debug_marker(
                stage=stage,
                path=DEBUG_OBJECT_GRASP_CENTER_MARKER_PATH,
                position=object_grasp_center_world,
                radius=DEBUG_OBJECT_GRASP_CENTER_MARKER_RADIUS_M,
                color=DEBUG_OBJECT_GRASP_CENTER_MARKER_COLOR,
            )
            print(
                "phase=phase2_debug_marker_object_grasp_center "
                f"path={object_grasp_center_marker_path} "
                f"position={object_grasp_center_world.tolist()}"
            )
        selected_hybrid_candidate, hybrid_candidate_selection = select_best_phase2_candidate(hybrid_candidates, object_grasp_frame, args)
        chosen_arm = str(selected_hybrid_candidate["arm"]) if selected_hybrid_candidate is not None else _choose_arm_side(args, target_components)
        if selected_hybrid_candidate is not None:
            selected_approach_mode = str(selected_hybrid_candidate["approach_mode"])
            approach_family_order = [selected_approach_mode] + [
                family for family in approach_family_order if family != selected_approach_mode
            ]
        pregrasp_target_marker_path = None
        pregrasp_target_world = None if selected_hybrid_candidate is None else np.array(selected_hybrid_candidate.get("pregrasp_world", []), dtype=float)
        if pregrasp_target_world is not None and pregrasp_target_world.size == 3 and np.isfinite(pregrasp_target_world).all():
            pregrasp_target_marker_path = _upsert_debug_marker(
                stage=stage,
                path=DEBUG_PREGRASP_TARGET_MARKER_PATH,
                position=pregrasp_target_world,
                radius=DEBUG_PREGRASP_TARGET_MARKER_RADIUS_M,
                color=DEBUG_PREGRASP_TARGET_MARKER_COLOR,
            )
            print(
                "phase=phase2_debug_marker_pregrasp_target "
                f"path={pregrasp_target_marker_path} "
                f"position={pregrasp_target_world.tolist()}"
            )
        payload["scene"] = {
            "config_path": str(config_path),
            "original_config_root_path": original_root_path,
            "overridden_root_path": str(asset_root),
            "scene_builder_methods": ["build_table", "build_parts", "build_robot"],
            "official_box_pipeline_used_for_destination_physics": False,
            "spawned_part_prim_list": part_paths,
            "table_prim": table_path,
            "table": {"bbox": table_bbox, "physics": _physics_summary(stage, table_path), "table_frame": table_frame},
            "debug_marker_paths": [],
        }
        payload["table_frame"] = table_frame
        payload["object_info"] = object_info
        payload["object_grasp_frame"] = object_grasp_frame
        payload["object_grasp_center_debug_marker"] = {
            "marker_path": object_grasp_center_marker_path,
            "marker_position_world": None if object_grasp_center_marker_path is None else object_grasp_center_world.tolist(),
            "marker_created_by": "phase2_estimate_object_grasp_frame",
            "marker_note": "grasp center in world coordinates from scene_state bbox-based grasp frame estimation",
        }
        payload["hybrid_phase1"] = {
            "source_baseline_script": SOURCE_BASELINE_SCRIPT,
            "table_unit_m": TABLE_UNIT_M,
            "perception_source": "scene_state",
            "candidate_generation_scope": "finite_phase1_candidates_preserved_for_phase2_filtering",
            "candidate_count_target_range": [4, 10],
            "generated_candidate_count": len(hybrid_candidates),
            "candidate_generation": hybrid_candidate_selection["phase1_selection"],
            "selected_candidate": selected_hybrid_candidate,
            "deferred_to_later_phases": [
                "Thinker_reranking",
                "YOLO_provider",
                "heavy_force_closure_optimizer",
            ],
        }
        payload["hybrid_phase2"] = {
            "source_phase1_script": SOURCE_BASELINE_SCRIPT,
            "object_grasp_frame": object_grasp_frame,
            "candidate_selection": hybrid_candidate_selection,
            "selected_candidate": selected_hybrid_candidate,
            "geometric_filter": None if selected_hybrid_candidate is None else selected_hybrid_candidate.get("geometric_filter"),
            "thresholds": {
                "alignment_error_max_rad": float(args.phase2_alignment_error_max),
                "symmetry_error_max_m": float(args.phase2_symmetry_error_max),
                "contact_asymmetry_max_m": float(args.phase2_contact_asymmetry_max),
                "table_clearance_min_m": float(args.phase2_table_clearance_min),
                "width_min_m": float(args.phase2_width_min),
                "width_max_m": float(args.phase2_width_max),
            },
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
        table_frame_start = counter["step"]
        _append_phase(
            phase_log,
            phase="resolve_table_frame",
            start_step=table_frame_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "table_frame": table_frame,
                "table_unit_m": TABLE_UNIT_M,
                "axis_aligned_with_world_xy": table_frame["axis_aligned_with_world_xy"],
                "mapping_mode": table_frame["mapping_mode"],
                "failure_reason": None,
            },
        )
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
        object_info_start = counter["step"]
        _append_phase(
            phase_log,
            phase="build_scene_state_object_info",
            start_step=object_info_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "object_info": object_info,
                "perception_source": object_info["perception_source"],
                "center_table_m": object_info["center_table_m"],
                "center_table_unit": object_info["center_table_unit"],
                "bbox_table_m": object_info["bbox_table_m"],
                "bbox_table_unit": object_info["bbox_table_unit"],
                "failure_reason": None,
            },
        )
        grasp_frame_start = counter["step"]
        _append_phase(
            phase_log,
            phase="phase2_estimate_object_grasp_frame",
            start_step=grasp_frame_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "object_grasp_frame": object_grasp_frame,
                "object_grasp_center_table_m": object_grasp_frame["grasp_center_table_m"],
                "object_grasp_center_table_unit": object_grasp_frame["grasp_center_table_unit"],
                "closing_axis_table": object_grasp_frame["closing_axis_table"],
                "lateral_axis_table": object_grasp_frame["lateral_axis_table"],
                "vertical_axis_table": object_grasp_frame["vertical_axis_table"],
                "width_on_closing_axis_m": object_grasp_frame["width_on_closing_axis_m"],
                "failure_reason": None,
            },
        )
        candidate_start = counter["step"]
        _append_phase(
            phase_log,
            phase="hybrid_phase1_select_candidate",
            start_step=candidate_start,
            end_step=counter["step"],
            condition_met=selected_hybrid_candidate is not None,
            details={
                "object_info": object_info,
                "candidate_generation": hybrid_candidate_selection,
                "candidate_list": hybrid_candidate_selection["candidate_scores"],
                "candidate_scores": hybrid_candidate_selection["candidate_scores"],
                "selected_candidate": selected_hybrid_candidate,
                "selected_arm": None if selected_hybrid_candidate is None else selected_hybrid_candidate["arm"],
                "selected_preset_id": None if selected_hybrid_candidate is None else selected_hybrid_candidate["preset_id"],
                "selected_approach_mode": None if selected_hybrid_candidate is None else selected_hybrid_candidate["approach_mode"],
                "pregrasp_table_m": None if selected_hybrid_candidate is None else selected_hybrid_candidate["pregrasp_table_m"],
                "pregrasp_table_unit": None if selected_hybrid_candidate is None else selected_hybrid_candidate["pregrasp_table_unit"],
                "object_grasp_center_table_m": None if selected_hybrid_candidate is None else selected_hybrid_candidate["object_grasp_center_table_m"],
                "object_grasp_center_table_unit": None if selected_hybrid_candidate is None else selected_hybrid_candidate["object_grasp_center_table_unit"],
                "perception_source": object_info["perception_source"],
                "failure_reason": hybrid_candidate_selection["failure_reason"],
            },
        )
        phase2_candidate_start = counter["step"]
        _append_phase(
            phase_log,
            phase="phase2_geometric_filter_select_candidate",
            start_step=phase2_candidate_start,
            end_step=counter["step"],
            condition_met=selected_hybrid_candidate is not None,
            details={
                "object_grasp_frame": object_grasp_frame,
                "selection_policy": hybrid_candidate_selection["selection_policy"],
                "candidate_filter_metrics": hybrid_candidate_selection["filtered_candidates"],
                "geometric_pass_candidate_count": hybrid_candidate_selection["geometric_pass_candidate_count"],
                "selected_candidate": selected_hybrid_candidate,
                "selected_candidate_filter": None if selected_hybrid_candidate is None else selected_hybrid_candidate.get("geometric_filter"),
                "selection_warning": hybrid_candidate_selection["selection_warning"],
                "failure_reason": hybrid_candidate_selection["failure_reason"],
            },
        )
        if selected_hybrid_candidate is None:
            _fail("hybrid_phase2_no_valid_candidate", "No valid Phase 2 hybrid candidate passed scoring/geometric filtering")
        print(
            f"phase=select_target target={target_path} center={initial_center.tolist()} "
            f"selection_policy={args.target_selection_policy} selected_index={selected_target_index} "
            f"chosen_arm={chosen_arm} target_region={target_region} forward_base={forward_base:.3f}"
        )
        print(
            "phase=hybrid_phase1_select_candidate "
            f"table_axis_aligned={table_frame['axis_aligned_with_world_xy']} "
            f"table_unit_m={TABLE_UNIT_M} "
            f"object_table_unit={object_info['center_table_unit']} "
            f"candidate_count={len(hybrid_candidates)} "
            f"selected={selected_hybrid_candidate['preset_id']} "
            f"arm={selected_hybrid_candidate['arm']} "
            f"approach={selected_hybrid_candidate['approach_mode']} "
            f"score={selected_hybrid_candidate['score']:.4f}"
        )
        print(
            "phase=phase2_geometric_filter_select_candidate "
            f"selected={selected_hybrid_candidate['preset_id']} "
            f"phase2_pass={selected_hybrid_candidate.get('phase2_filter_pass')} "
            f"alignment_error_rad={selected_hybrid_candidate.get('geometric_filter', {}).get('alignment_error_rad')} "
            f"symmetry_error_m={selected_hybrid_candidate.get('geometric_filter', {}).get('symmetry_error_m')} "
            f"predicted_contact_asymmetry_m={selected_hybrid_candidate.get('geometric_filter', {}).get('predicted_contact_asymmetry_m')} "
            f"table_clearance_margin_m={selected_hybrid_candidate.get('geometric_filter', {}).get('table_clearance_margin_m')} "
            f"selection_warning={hybrid_candidate_selection.get('selection_warning')}"
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
        proxy_middle_point_world, proxy_middle_point_log = _resolve_finger_midpoint_reference_position(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=chosen_robot_prim_path,
            arm_side=chosen_arm,
            requested_name_token="phase2_debug_legacy_finger_link_midpoint",
        )
        proxy_middle_point_marker_path = None
        if proxy_middle_point_world is not None:
            proxy_middle_point_marker_path = _upsert_debug_marker(
                stage=stage,
                path=DEBUG_PROXY_MIDDLE_POINT_MARKER_PATH,
                position=proxy_middle_point_world,
                radius=DEBUG_PROXY_MIDDLE_POINT_MARKER_RADIUS_M,
                color=DEBUG_PROXY_MIDDLE_POINT_MARKER_COLOR,
            )
            print(
                "phase=phase2_debug_marker_proxy_midpoint "
                f"path={proxy_middle_point_marker_path} "
                f"source={proxy_middle_point_log.get('source')} "
                f"position={np.array(proxy_middle_point_world, dtype=float).tolist()}"
            )

        payload["selected_pregrasp_target_debug_marker"] = {
            "marker_path": pregrasp_target_marker_path,
            "marker_position_world": None if pregrasp_target_world is None else pregrasp_target_world.tolist(),
            "marker_source": "selected_candidate.pregrasp_world",
            "marker_note": "debug marker for selected pregrasp point_B target before execute_pregrasp",
        }

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
        phase1_real_grasp_center_world, phase1_real_grasp_center_log = resolve_real_grasp_center_world(
            stage=stage,
            dc=dc,
            articulation=articulation,
            robot_root_path=chosen_robot_prim_path,
            arm_side=chosen_arm,
            fallback_world=selected_hybrid_candidate["object_grasp_center_world"],
            fallback_source="phase1_selected_candidate_object_grasp_center",
            diagnostic_finger_link_midpoint_bypass=bool(
                getattr(args, "phase2_diagnostic_finger_link_midpoint_bypass", False)
            ),
            include_diagnostic_comparison=bool(
                getattr(args, "phase2_diagnostic_finger_link_midpoint_bypass", False)
            ),
        )
        phase1_real_grasp_center_table_m = None
        phase1_real_grasp_center_table_unit = None
        if phase1_real_grasp_center_world is not None:
            phase1_real_grasp_center_table = world_to_table(phase1_real_grasp_center_world, table_frame)
            phase1_real_grasp_center_table_m = phase1_real_grasp_center_table.tolist()
            phase1_real_grasp_center_table_unit = _table_units(phase1_real_grasp_center_table)
        payload["hybrid_phase1"]["real_grasp_center_resolution"] = {
            "real_grasp_center_world": None if phase1_real_grasp_center_world is None else np.array(phase1_real_grasp_center_world, dtype=float).tolist(),
            "real_grasp_center_table_m": phase1_real_grasp_center_table_m,
            "real_grasp_center_table_unit": phase1_real_grasp_center_table_unit,
            "resolution_log": phase1_real_grasp_center_log,
            "fallback_status": {
                "fallback_used": bool(phase1_real_grasp_center_log.get("fallback_used", False)),
                "fallback_source": phase1_real_grasp_center_log.get("fallback_source"),
                "source": phase1_real_grasp_center_log.get("source"),
            },
        }
        payload["hybrid_phase2"]["real_grasp_center_resolution"] = payload["hybrid_phase1"]["real_grasp_center_resolution"]
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
                "phase1_table_frame": table_frame,
                "phase1_object_info": object_info,
                "phase1_selected_candidate": selected_hybrid_candidate,
                "phase1_real_grasp_center_resolution": payload["hybrid_phase1"].get("real_grasp_center_resolution"),
                "phase2_object_grasp_frame": object_grasp_frame,
                "phase2_selected_candidate_filter": selected_hybrid_candidate.get("geometric_filter"),
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
        payload["selected_hybrid_phase1_candidate"] = selected_hybrid_candidate
        payload["selected_arm"] = selected_hybrid_candidate["arm"]
        payload["selected_preset_id"] = selected_hybrid_candidate["preset_id"]
        payload["selected_approach_mode"] = selected_hybrid_candidate["approach_mode"]
        payload["pregrasp_table_m"] = selected_hybrid_candidate["pregrasp_table_m"]
        payload["pregrasp_table_unit"] = selected_hybrid_candidate["pregrasp_table_unit"]
        payload["object_grasp_center_table_m"] = selected_hybrid_candidate["object_grasp_center_table_m"]
        payload["object_grasp_center_table_unit"] = selected_hybrid_candidate["object_grasp_center_table_unit"]
        motion_policy = str(geometry.get("motion_policy", "mid_vertical_Z_descend"))
        far_motion_policy = motion_policy == "far_low_side_B_driven"
        vertical_prefix = "mid" if target_region == "mid" else "near_body"
        pregrasp_phase_name = "far_prepare_low_side_approach" if far_motion_policy else f"{vertical_prefix}_align_AB_vertical_over_object"
        contact_phase_name = "far_align_B_over_object_xy" if far_motion_policy else f"{vertical_prefix}_pre_descend_AB_vertical"
        descend_phase_name = "far_lower_B_world_z" if far_motion_policy else f"{vertical_prefix}_descend_world_z_keep_AB_vertical"
        phase2_descent_result: dict[str, Any] | None = None

        debug_markers = [
            *(
                [object_grasp_center_marker_path]
                if object_grasp_center_marker_path is not None
                else []
            ),
            _create_debug_marker(stage, "/World/DebugDualArmIKTarget", initial_center, 0.025, (1.0, 0.2, 0.1)),
            _create_debug_marker(stage, "/World/DebugDualArmIKPregraspB", geometry["pregrasp_point_B_world"], 0.025, (0.2, 0.6, 1.0)),
            pregrasp_target_marker_path,
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
        if proxy_middle_point_marker_path is not None:
            debug_markers.append(proxy_middle_point_marker_path)
        payload["scene"]["debug_marker_paths"] = debug_markers
        payload["proxy_middle_point_debug_initial"] = {
            "proxy_middle_point_marker_path": proxy_middle_point_marker_path,
            "legacy_proxy_middle_point_source": None if proxy_middle_point_world is None else proxy_middle_point_log.get("source"),
            "legacy_proxy_middle_point_world": None if proxy_middle_point_world is None else np.array(proxy_middle_point_world, dtype=float).tolist(),
            "legacy_proxy_component_positions_world": None
            if proxy_middle_point_world is None
            else proxy_middle_point_log.get("component_positions_world"),
            "legacy_proxy_reference_mode": proxy_middle_point_log.get("reference_mode"),
            "legacy_proxy_reference_fallback": proxy_middle_point_log.get("fallback"),
            "legacy_proxy_reference_reason": proxy_middle_point_log.get("reason"),
        }

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

        pregrasp_result = execute_pregrasp(
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
        payload["pregrasp_result"] = {
            "phase": pregrasp_phase_name,
            "motion_policy": motion_policy,
            "target_point_B_world": np.array(geometry["pregrasp_point_B_world"], dtype=float).tolist(),
            "ik_target_position_world": pregrasp_result.get("ik_target_position_world"),
            "ik_target_rpy": pregrasp_result.get("ik_target_rpy"),
            "ik_success": bool(pregrasp_result.get("ik_success")),
            "position_error_norm": float(pregrasp_result.get("position_error_norm")),
            "orientation_error_rad": float(pregrasp_result.get("orientation_error_rad")),
            "fallback_attempt": pregrasp_result.get("fallback_attempt"),
        }
        if not bool(pregrasp_result.get("ik_success", False)):
            print(
                "IK_PREGRASP_FAIL: "
                f"target={pregrasp_phase_name} "
                f"ik_target_position_world={pregrasp_result.get('ik_target_position_world')} "
                f"ik_target_rpy={pregrasp_result.get('ik_target_rpy')} "
                f"ik_success={pregrasp_result.get('ik_success')} "
                f"position_error_norm={pregrasp_result.get('position_error_norm')} "
                f"orientation_error_rad={pregrasp_result.get('orientation_error_rad')}"
            )
        if pregrasp_result["final_error"] > args.pregrasp_tolerance:
            _fail("pregrasp_failed", f"{pregrasp_phase_name} did not reach selected DualArmIK 6D target")

        if far_motion_policy:
            selected_point_b_offset = np.array(geometry.get("point_b_offset_local", point_b_offset), dtype=float)
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
                    "object_top_z": geometry.get("object_top_z_world"),
                    "align_clearance": geometry.get("align_clearance_m"),
                    "align_height": geometry.get("align_height"),
                    "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                    "AB_axis_world": geometry.get("AB_axis_world"),
                    "selected_orientation_preset_axial_roll_variant_label": selected_orientation_preset.get("preset_axial_roll_variant_label"),
                    "selected_orientation_preset_axial_roll_about_ab_rad": selected_orientation_preset.get("preset_axial_roll_about_ab_rad"),
                    "side_push_avoidance": "XY alignment happens at low-side prepare height before final world-Z lowering",
                },
            )
            current_align_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
            current_align_b_world = _point_b_world_from_pose(coord_transform, current_align_pose, selected_point_b_offset)
            target_align_b_world = np.array(geometry["far_xy_align_B_world"], dtype=float)
            xy_error_at_descent = float(np.linalg.norm(current_align_b_world[:2] - target_align_b_world[:2]))
            horizontal_descent_trigger_tolerance = float(args.horizontal_descent_xy_trigger_tolerance)
            near_enough_xy = bool(xy_error_at_descent < horizontal_descent_trigger_tolerance)
            descent_trigger_log = {
                "align_height": geometry.get("align_height"),
                "object_top_z": geometry.get("object_top_z_world"),
                "align_clearance": geometry.get("align_clearance_m"),
                "descent_triggered": near_enough_xy,
                "xy_error": xy_error_at_descent,
                "xy_error_at_descent": xy_error_at_descent,
                "xy_error_at_descent_trigger": xy_error_at_descent,
                "horizontal_descent_trigger_tolerance_used": horizontal_descent_trigger_tolerance,
                "xy_descent_trigger_tolerance_m": horizontal_descent_trigger_tolerance,
                "full_align_final_error_m": far_xy_align_result.get("final_error"),
                "full_align_rotation_error_rad": far_xy_align_result.get("final_rotation_error_rad"),
                "descent_trigger_rule": "start descending when horizontal XY error is near enough; do not require perfect full-pose alignment",
            }
            far_xy_align_result["horizontal_descent_trigger"] = descent_trigger_log
            if phase_log and phase_log[-1].get("phase") == "far_align_B_over_object_xy":
                phase_log[-1]["details"]["horizontal_descent_trigger"] = descent_trigger_log
            payload["hybrid_phase2"]["horizontal_descent_trigger"] = descent_trigger_log
            if not near_enough_xy:
                _fail("align_failed", "far_align_B_over_object_xy did not reach loose XY descent trigger")
            payload["object_trace"]["after_far_xy_align"] = _bbox_state(stage, target_path)

            far_descend_locked_rpy = np.array(selected_orientation_preset["rpy"], dtype=float)
            far_descend_B_world = np.array(geometry["contact_point_B_world"], dtype=float)
            horizontal_grasp_expansion_log = None
            closing_axis_world = _finite_world_vector_or_none(object_grasp_frame.get("closing_axis_world"))
            if closing_axis_world is None:
                closing_axis_world = _finite_world_vector_or_none(geometry.get("closing_axis_world"))
            if closing_axis_world is None:
                closing_axis_world = _finite_world_vector_or_none(geometry.get("AB_axis_world"))
            if closing_axis_world is not None:
                closing_axis_norm = float(np.linalg.norm(closing_axis_world))
                if closing_axis_norm > 1e-9:
                    closing_axis_world = closing_axis_world / closing_axis_norm
                    object_width = _finite_float_or_none(object_grasp_frame.get("width_on_closing_axis_m"))
                    if object_width is None:
                        width_filter = selected_hybrid_candidate.get("geometric_filter", {}).get("width_compatibility", {})
                        object_width = _finite_float_or_none(width_filter.get("width_on_closing_axis_m"))
                    object_width = 0.0 if object_width is None else float(object_width)
                    expansion_offset = min(max(object_width * 0.3, 0.01), 0.03)
                    unexpanded_contact_point_B_world = far_descend_B_world.copy()
                    far_descend_B_world = far_descend_B_world + closing_axis_world * expansion_offset
                    expanded_contact_pose_base, expansion_pose_log = _pose_for_point_b_world(
                        far_descend_B_world,
                        coord_transform,
                        far_descend_locked_rpy,
                        selected_point_b_offset,
                    )
                    geometry["unexpanded_contact_point_B_world"] = unexpanded_contact_point_B_world.tolist()
                    geometry["contact_point_B_world"] = far_descend_B_world.tolist()
                    geometry["far_descend_B_world"] = far_descend_B_world.tolist()
                    geometry["contact_pose_base"] = expanded_contact_pose_base.tolist()
                    horizontal_grasp_expansion_log = {
                        "enabled": True,
                        "object_width_m": object_width,
                        "expansion_offset_m": float(expansion_offset),
                        "closing_axis_world": closing_axis_world.tolist(),
                        "unexpanded_contact_point_B_world": unexpanded_contact_point_B_world.tolist(),
                        "expanded_contact_point_B_world": far_descend_B_world.tolist(),
                        "pose_conversion": expansion_pose_log,
                    }
            if horizontal_grasp_expansion_log is None:
                horizontal_grasp_expansion_log = {
                    "enabled": False,
                    "reason": "closing_axis_unavailable",
                }
            geometry["horizontal_grasp_expansion"] = horizontal_grasp_expansion_log

            far_z_completion_tolerance = float(args.pre_close_point_b_tolerance)
            far_z_descent_state: dict[str, Any] = {
                "completion_rule": "point_B_world_z_must_reach_far_descend_B_world_z_before_phase_success",
                "target_z_world_m": float(far_descend_B_world[2]),
                "z_completion_tolerance_m": far_z_completion_tolerance,
                "final_z_reached_by_runtime": False,
                "descent_stopped_before_contact_z": True,
            }

            def update_far_z_descent_state(current_b_world: np.ndarray, commanded_b_world: np.ndarray | None = None) -> float:
                z_remaining = float(max(0.0, float(current_b_world[2]) - float(far_descend_B_world[2])))
                far_z_descent_state.update(
                    {
                        "current_point_B_world": np.array(current_b_world, dtype=float).tolist(),
                        "current_z_world_m": float(current_b_world[2]),
                        "target_z_world_m": float(far_descend_B_world[2]),
                        "z_remaining_to_contact_m": z_remaining,
                        "final_z_reached_by_runtime": bool(z_remaining <= far_z_completion_tolerance),
                        "descent_stopped_before_contact_z": bool(z_remaining > far_z_completion_tolerance),
                    }
                )
                if commanded_b_world is not None:
                    commanded_remaining = float(max(0.0, float(commanded_b_world[2]) - float(far_descend_B_world[2])))
                    far_z_descent_state.update(
                        {
                            "commanded_point_B_world": np.array(commanded_b_world, dtype=float).tolist(),
                            "commanded_z_world_m": float(commanded_b_world[2]),
                            "commanded_z_remaining_to_contact_m": commanded_remaining,
                            "final_z_reached_by_command": bool(commanded_remaining <= far_z_completion_tolerance),
                        }
                    )
                return z_remaining

            def far_world_z_lower_fn():
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_B_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
                next_B_world = far_descend_B_world.copy()
                next_B_world[2] = max(float(far_descend_B_world[2]), float(curr_B_world[2]) - 0.002)
                update_far_z_descent_state(curr_B_world, next_B_world)
                target_pose, _ = _pose_for_point_b_world(next_B_world, coord_transform, far_descend_locked_rpy, selected_point_b_offset)
                return target_pose

            def far_z_descent_completion_fn() -> bool:
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_B_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
                z_remaining = update_far_z_descent_state(curr_B_world)
                return bool(z_remaining <= far_z_completion_tolerance)

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
                completion_condition_fn=far_z_descent_completion_fn,
                completion_condition_label="far_point_B_runtime_z_reached_contact_target",
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
                    "descent_triggered": near_enough_xy,
                    "xy_error": xy_error_at_descent,
                    "xy_error_at_descent_trigger": xy_error_at_descent,
                    "horizontal_descent_trigger_tolerance_used": horizontal_descent_trigger_tolerance,
                    "horizontal_grasp_expansion": horizontal_grasp_expansion_log,
                    "far_low_side_prepare_B_world": geometry.get("far_low_side_prepare_B_world"),
                    "far_xy_align_B_world": geometry.get("far_xy_align_B_world"),
                    "far_descend_B_world": geometry.get("far_descend_B_world"),
                    "contact_point_B_world": geometry.get("contact_point_B_world"),
                    "contact_point_A_world": geometry.get("contact_AB_semantics", {}).get("point_A_world"),
                    "far_point_b_gap_above_support_m": geometry.get("far_point_b_gap_above_support_m"),
                    "far_reach_axis_world": geometry.get("far_reach_axis_world"),
                    "AB_axis_world": geometry.get("AB_axis_world"),
                    "world_z_lowering": True,
                    "z_descent_completion": far_z_descent_state,
                    "side_push_avoidance": "final contact moves only in world Z after XY alignment",
                },
            )
            payload["object_trace"]["after_far_world_z_lower"] = _bbox_state(stage, target_path)
            contact_gate_phase_index = -1
            if bool(args.phase2_final_descent_enable):
                contact_centric_control_subject_log = {
                    "control_subject": "runtime_tip_mid_or_contact_reference",
                    "compatibility_pose_builder": "point_B_offset_conversion_only",
                    "close_subject": "runtime_close_critical_grasp_center",
                }
                geometry["contact_centric_control_subject"] = contact_centric_control_subject_log
                payload["hybrid_phase2"]["contact_centric_control_subject"] = contact_centric_control_subject_log
                phase2_descent_result = final_descent_local_ik(
                    phase_name="phase2_far_final_descent_local_ik",
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
                    gripper_dofs=gripper_dofs,
                    sim_app=sim_app,
                    args=args,
                    counter=counter,
                    phase_log=phase_log,
                    end_effector_name=end_effector_name,
                    end_effector_path=end_effector_path,
                    end_effector_policy=end_effector_policy,
                    locked_target_world=np.array(object_grasp_frame["grasp_center_world"], dtype=float),
                    locked_rpy=far_descend_locked_rpy,
                    locked_target_pose_base=geometry["contact_pose_base"],
                    point_b_offset_local=selected_point_b_offset,
                    coord_transform_refresh_fn=coord_transform_refresh_fn,
                    ik_overrides=far_ik_overrides,
                    object_grasp_frame=object_grasp_frame,
                    selected_candidate_filter=selected_hybrid_candidate.get("geometric_filter"),
                )
                payload["phase2_final_descent"] = phase2_descent_result.get("phase2_local_descent")
                payload["object_trace"]["after_phase2_far_final_descent"] = _bbox_state(stage, target_path)
                contact_gate_phase_index = -1
            if not bool(args.phase2_final_descent_enable) and descend_result["final_error"] > args.descend_tolerance:
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
            vertical_ref_phase_start_world = _finite_world_vector_or_none(
                geometry.get("vertical_xy_reference_world_position_used_for_offset")
            )

            def vertical_xy_locked_descend_target_fn():
                nonlocal vertical_xy_feedback_call_count
                if vertical_ref_offset is None or vertical_ref_target_xy is None:
                    return final_contact_pose
                vertical_xy_feedback_call_count += 1
                curr_pose = _current_ee_pose_base(ik_solver, dc, articulation, chosen_arm, args=args)
                curr_b_world = _point_b_world_from_pose(coord_transform, curr_pose, selected_point_b_offset)
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
                reference_world_delta_from_phase_start = None
                if vertical_ref_phase_start_world is not None and curr_ref_world is not None:
                    reference_world_delta_from_phase_start = np.array(curr_ref_world, dtype=float) - vertical_ref_phase_start_world
                xy_error = vertical_ref_target_xy[:2] - curr_ref_world[:2]
                target_b_world = vertical_contact_b_world.copy()
                target_b_world[:2] = vertical_contact_b_world[:2] + xy_error
                target_b_world[2] = vertical_contact_b_world[2]
                if vertical_xy_feedback_call_count == 1 or vertical_xy_feedback_call_count % int(args.trace_interval) == 0:
                    vertical_xy_feedback_samples.append(
                        {
                            "call_index": vertical_xy_feedback_call_count,
                            "current_point_B_world": curr_b_world.tolist(),
                            "current_vertical_xy_reference_world": curr_ref_world.tolist(),
                            "vertical_xy_reference_target_xy_world": vertical_ref_target_xy[:2].tolist(),
                            "vertical_xy_reference_error_xy_m": xy_error.tolist(),
                            "vertical_xy_reference_error_norm_m": float(np.linalg.norm(xy_error)),
                            "vertical_xy_reference_feedback_rule": "nominal_contact_B_xy_plus_live_reference_error",
                            "reference_world_at_phase_start": None
                            if vertical_ref_phase_start_world is None
                            else vertical_ref_phase_start_world.tolist(),
                            "reference_world_current_tick": None if curr_ref_world is None else np.array(curr_ref_world, dtype=float).tolist(),
                            "reference_world_delta_from_phase_start": None
                            if reference_world_delta_from_phase_start is None
                            else reference_world_delta_from_phase_start.tolist(),
                            "reference_world_delta_from_phase_start_norm_m": None
                            if reference_world_delta_from_phase_start is None
                            else float(np.linalg.norm(reference_world_delta_from_phase_start)),
                            "vertical_xy_reference_staleness_checked": True,
                            "current_vertical_xy_reference_runtime_source": curr_ref_runtime_log.get("runtime_source"),
                            "current_vertical_xy_reference_runtime_fallback_used": curr_ref_runtime_log.get("runtime_fallback_used"),
                            "current_vertical_xy_reference_mode": curr_ref_runtime_log.get("reference_mode", geometry.get("vertical_xy_reference_mode")),
                            "current_finger_midpoint_component_positions_world": curr_ref_runtime_log.get("component_positions_world"),
                            "current_fingertip_midpoint_component_positions_world": curr_ref_runtime_log.get("fingertip_component_positions_world", curr_ref_runtime_log.get("component_positions_world")),
                            "current_fingertip_midpoint_world": curr_ref_runtime_log.get("fingertip_midpoint_world", curr_ref_runtime_log.get("world_position")),
                            "current_fingertip_reference_source": curr_ref_runtime_log.get("fingertip_reference_source_used", curr_ref_runtime_log.get("source")),
                            "nominal_contact_point_B_world": vertical_contact_b_world.tolist(),
                            "commanded_target_point_B_world": target_b_world.tolist(),
                        }
                    )
                target_pose, _ = _pose_for_point_b_world(target_b_world, coord_transform, descend_locked_rpy, selected_point_b_offset)
                return target_pose

            descend_result = _execute_dualarmik_servo_phase(
                ServoSpec(descend_phase_name, final_contact_pose, vertical_descend_pos_tolerance, args.rot_tolerance, args.servo_max_ticks * 2),
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
                    "vertical_descend_target_policy": "continuous_vertical_xy_reference_feedback_to_final_point_B_z",
                    "vertical_xy_reference_feedback_active": bool(vertical_ref_offset is not None and vertical_ref_target_xy is not None),
                    "vertical_xy_reference_feedback_rule": "nominal_contact_B_xy_plus_live_reference_error",
                    "vertical_xy_reference_world_at_phase_start": None
                    if vertical_ref_phase_start_world is None
                    else vertical_ref_phase_start_world.tolist(),
                    "vertical_xy_reference_staleness_diagnostics_enabled": True,
                    "vertical_xy_reference_feedback_samples": vertical_xy_feedback_samples,
                    "close_after_point_B_contact_gate": True,
                    "strict_AB_vertical_during_descend": True,
                },
            )
            after_descend = _bbox_state(stage, target_path)
            payload["object_trace"]["after_descend"] = after_descend
            contact_gate_phase_index = -1
            if bool(args.phase2_final_descent_enable):
                contact_centric_control_subject_log = {
                    "control_subject": "runtime_tip_mid_or_contact_reference",
                    "compatibility_pose_builder": "point_B_offset_conversion_only",
                    "close_subject": "runtime_close_critical_grasp_center",
                }
                geometry["contact_centric_control_subject"] = contact_centric_control_subject_log
                payload["hybrid_phase2"]["contact_centric_control_subject"] = contact_centric_control_subject_log
                phase2_descent_result = final_descent_local_ik(
                    phase_name="phase2_vertical_final_descent_local_ik",
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
                    gripper_dofs=gripper_dofs,
                    sim_app=sim_app,
                    args=args,
                    counter=counter,
                    phase_log=phase_log,
                    end_effector_name=end_effector_name,
                    end_effector_path=end_effector_path,
                    end_effector_policy=end_effector_policy,
                    locked_target_world=geometry["contact_point_B_world"],
                    locked_rpy=descend_locked_rpy,
                    locked_target_pose_base=geometry["contact_pose_base"],
                    point_b_offset_local=selected_point_b_offset,
                    coord_transform_refresh_fn=coord_transform_refresh_fn,
                    object_grasp_frame=object_grasp_frame,
                    selected_candidate_filter=selected_hybrid_candidate.get("geometric_filter"),
                    table_frame=table_frame,
                )
                payload["phase2_final_descent"] = phase2_descent_result.get("phase2_local_descent")
                payload["object_trace"]["after_phase2_vertical_final_descent"] = _bbox_state(stage, target_path)
                contact_gate_phase_index = -1
            if not bool(args.phase2_final_descent_enable) and descend_result["final_error"] > vertical_descend_pos_tolerance:
                _fail("descend_failed", f"{descend_phase_name} did not bring point B to the vertical contact mark")

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
            object_grasp_frame=object_grasp_frame,
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
        vertical_tip_rule_close_allowed = bool(
            (not far_motion_policy)
            and isinstance(phase2_descent_result, dict)
            and phase2_descent_result.get("vertical_tip_reached_table_z0", False)
        )
        vertical_tip_rule_log = {
            "rule_name": "vertical_tip_table_z_leq_zero_rule",
            "condition_met": vertical_tip_rule_close_allowed,
            "vertical_tip_reached_table_z0": bool(vertical_tip_rule_close_allowed),
            "vertical_tip_close_stop_reason": phase2_descent_result.get("vertical_tip_close_stop_reason")
            if isinstance(phase2_descent_result, dict)
            else None,
            "vertical_tip_stop_rule_threshold_m": phase2_descent_result.get("vertical_tip_stop_rule_threshold_m")
            if isinstance(phase2_descent_result, dict)
            else None,
            "vertical_tip_stop_rule_source": phase2_descent_result.get("vertical_tip_stop_rule_source")
            if isinstance(phase2_descent_result, dict)
            else None,
            "vertical_tip_stop_rule_last_sample": phase2_descent_result.get("vertical_tip_stop_rule_last_sample")
            if isinstance(phase2_descent_result, dict)
            else None,
            "decisive_for_vertical_close": False,
            "table_frame_is_source_of_truth": True,
            "rule_scope": "diagnostic_auxiliary_condition_only_no_gate_bypass",
        }
        phase2_close_gate = evaluate_close_gate(
            pre_close_gate=pre_close_gate,
            object_grasp_frame=object_grasp_frame,
            selected_candidate_filter=selected_hybrid_candidate.get("geometric_filter", {}),
            descent_result=phase2_descent_result or descend_result,
            motion_policy=motion_policy,
            args=args,
        )
        vertical_support_or_stall_fallback_gate = evaluate_vertical_support_or_stall_close_fallback(
            pre_close_gate=pre_close_gate,
            descent_result=phase2_descent_result or descend_result,
            motion_policy=motion_policy,
            args=args,
        )
        generic_runtime_commit_fallback_gate = evaluate_runtime_commit_fallback(
            pre_close_gate=pre_close_gate,
            descent_result=phase2_descent_result or descend_result,
            selected_candidate_filter=selected_hybrid_candidate.get("geometric_filter", {}),
            args=args,
        )
        close_allowed_by_primary = bool(phase2_close_gate["condition_met"])
        close_allowed_by_fallback = bool(
            (not close_allowed_by_primary)
            and vertical_support_or_stall_fallback_gate["condition_met"]
        )
        close_allowed_by_generic_runtime_fallback = bool(
            (not close_allowed_by_primary)
            and (not close_allowed_by_fallback)
            and generic_runtime_commit_fallback_gate["condition_met"]
        )
        final_close_decision = {
            "condition_met": bool(close_allowed_by_primary or close_allowed_by_fallback or close_allowed_by_generic_runtime_fallback),
            "allowed_by": "primary_phase2_close_gate"
            if close_allowed_by_primary
            else (
                "vertical_support_or_stall_fallback"
                if close_allowed_by_fallback
                else ("generic_runtime_commit_fallback" if close_allowed_by_generic_runtime_fallback else None)
            ),
            "primary_gate_pass": close_allowed_by_primary,
            "fallback_gate_pass": bool(vertical_support_or_stall_fallback_gate["condition_met"]),
            "fallback_gate_used": close_allowed_by_fallback,
            "generic_runtime_commit_fallback_pass": bool(generic_runtime_commit_fallback_gate["condition_met"]),
            "generic_runtime_commit_fallback_used": close_allowed_by_generic_runtime_fallback,
            "vertical_tip_rule_pass": vertical_tip_rule_close_allowed,
            "vertical_tip_table_z_close_rule": vertical_tip_rule_log,
            "motion_policy": motion_policy,
            "far_motion_policy": bool(far_motion_policy),
            "primary_fail_reasons": phase2_close_gate.get("fail_reasons", []),
            "fallback_fail_reasons": vertical_support_or_stall_fallback_gate.get("fail_reasons", []),
            "generic_runtime_commit_fallback_fail_reasons": generic_runtime_commit_fallback_gate.get("fail_reasons", []),
        }
        pre_close_gate["phase2_multi_condition_close_gate"] = phase2_close_gate
        pre_close_gate["vertical_support_or_stall_close_fallback_gate"] = vertical_support_or_stall_fallback_gate
        pre_close_gate["generic_runtime_commit_fallback_gate"] = generic_runtime_commit_fallback_gate
        pre_close_gate["phase2_final_close_decision"] = final_close_decision
        close_debug_summary = (
            build_close_debug_summary(
                pre_close_gate=pre_close_gate,
                phase2_close_gate=phase2_close_gate,
                vertical_support_or_stall_fallback_gate=vertical_support_or_stall_fallback_gate,
                generic_runtime_commit_fallback_gate=generic_runtime_commit_fallback_gate,
                final_close_decision=final_close_decision,
                descent_result=phase2_descent_result or descend_result,
            )
            if bool(getattr(args, "close_debug_summary_enable", True))
            else {"summary_name": "close_debug_summary", "enabled": False}
        )
        pre_close_gate["close_debug_summary"] = close_debug_summary
        phase_log[contact_gate_phase_index]["details"]["pre_close_gate"] = pre_close_gate
        payload["hybrid_phase2"]["multi_condition_close_gate"] = phase2_close_gate
        payload["hybrid_phase2"]["vertical_support_or_stall_close_fallback_gate"] = vertical_support_or_stall_fallback_gate
        payload["hybrid_phase2"]["generic_runtime_commit_fallback_gate"] = generic_runtime_commit_fallback_gate
        payload["hybrid_phase2"]["final_close_decision"] = final_close_decision
        payload["hybrid_phase2"]["close_debug_summary"] = close_debug_summary
        if not final_close_decision["condition_met"]:
            close_gate_reason = "phase2_close_gate_failed:" + ",".join(phase2_close_gate["fail_reasons"])
            if vertical_support_or_stall_fallback_gate.get("fail_reasons"):
                close_gate_reason += ";vertical_fallback_failed:" + ",".join(vertical_support_or_stall_fallback_gate["fail_reasons"])
            if generic_runtime_commit_fallback_gate.get("fail_reasons"):
                close_gate_reason += ";generic_runtime_fallback_failed:" + ",".join(generic_runtime_commit_fallback_gate["fail_reasons"])
            recovery_log = recover_and_retry(
                reason=close_gate_reason,
                dc=dc,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                selected_candidate=selected_hybrid_candidate,
                candidate_selection=hybrid_candidate_selection,
            )
            payload["hybrid_phase2"]["recovery_after_close_gate"] = recovery_log
            _fail(
                "phase2_close_gate_failed",
                "Phase 2 close decision failed before gripper close: " + close_gate_reason,
            )

        if pre_close_gate.get("close_critical_uses_real_grasp_center") is not True:
            close_critical_reference_gate = {
                "gate_name": "close_critical_real_grasp_center_required",
                "condition_met": False,
                "reason": "No valid close-critical fingertip grasp center; refuse close",
                "close_critical_uses_real_grasp_center": pre_close_gate.get("close_critical_uses_real_grasp_center"),
                "close_critical_metric": pre_close_gate.get("close_critical_metric"),
                "real_grasp_center_world": pre_close_gate.get("real_grasp_center_world"),
                "point_B_proxy_world": pre_close_gate.get("point_B_proxy_world"),
                "explicit_proxy_may_not_authorize_close": True,
            }
            pre_close_gate["close_critical_real_grasp_center_required_gate"] = close_critical_reference_gate
            final_close_decision["close_critical_real_grasp_center_required_gate"] = close_critical_reference_gate
            payload["hybrid_phase2"]["close_critical_real_grasp_center_required_gate"] = close_critical_reference_gate
            if phase_log and -len(phase_log) <= contact_gate_phase_index < len(phase_log):
                phase_log[contact_gate_phase_index]["details"]["pre_close_gate"] = pre_close_gate
            _fail("close_gate_failed", "No valid close-critical fingertip grasp center; refuse close")

        close_result = execute_two_stage_close(
            dc=dc,
            gripper_dofs=gripper_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            object_grasp_frame=object_grasp_frame,
            pre_close_gate=pre_close_gate,
            coord_transform_refresh_fn=coord_transform_refresh_fn,
            skipped=args.skip_gripper_close,
        )
        close_ok = bool(close_result["condition_met"])
        payload["hybrid_phase2"]["two_stage_close"] = close_result
        payload["object_trace"]["after_close"] = _bbox_state(stage, target_path)
        if not close_ok:
            recovery_log = recover_and_retry(
                reason="phase2_two_stage_close_failed",
                dc=dc,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                selected_candidate=selected_hybrid_candidate,
                candidate_selection=hybrid_candidate_selection,
            )
            payload["hybrid_phase2"]["recovery_after_close"] = recovery_log
            _fail("close_gripper_failed", f"{chosen_arm} gripper two-stage close command failed")

        short_lift_result = verify_short_lift(
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
        )
        payload["hybrid_phase2"]["short_lift_verification"] = short_lift_result
        payload["object_trace"]["after_micro_lift_probe"] = short_lift_result["after_state"]
        if not short_lift_result["condition_met"]:
            recovery_log = recover_and_retry(
                reason="phase2_short_lift_verification_failed",
                dc=dc,
                gripper_dofs=gripper_dofs,
                sim_app=sim_app,
                args=args,
                counter=counter,
                phase_log=phase_log,
                selected_candidate=selected_hybrid_candidate,
                candidate_selection=hybrid_candidate_selection,
            )
            payload["hybrid_phase2"]["recovery_after_short_lift"] = recovery_log
            _fail("object_not_lifted", "Object did not follow the gripper during Phase 2 short lift verification")

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
            print("status=pass object_lifted=true stop_after_lift=true carry_place_skipped=true")
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
        print("status=pass object_lifted=true object_transported=true final_inside_bin=true object_stable=true")

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
