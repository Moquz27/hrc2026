#!/usr/bin/env python3
"""Task 1 Cartesian DLS phase baseline for Walker S2.

This script is intentionally narrow: it builds the official Task 1 scene,
loads the official robot startup posture, plans simple absolute Cartesian
targets, and executes each motion phase with a measured Isaac 3D position-only
DLS solver. It does not import or reuse the old heuristic reach stack.
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


SCRIPT_NAME = "task1_cartesian_dls_phase_baseline.py"
LOG_STEM = "task1_cartesian_dls_phase_baseline"

OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"
OFFICIAL_GRIPPER_OPEN_WIDTH = -0.0215
OFFICIAL_GRIPPER_CLOSE_WIDTH = 0.01
DEFAULT_GRIPPER_HOLD_EFFORT = 100.0

DEFAULT_SETTLE_STEPS = 240
DEFAULT_GRIPPER_STEPS = 24
DEFAULT_DLS_MAX_ITERS = 40
DEFAULT_DLS_EPS = 0.01
DEFAULT_DLS_DAMPING = 0.05
DEFAULT_DLS_MAX_JOINT_STEP = 0.035
DEFAULT_DLS_TRACK_STEPS = 2
DEFAULT_DLS_POSTURE_GAIN = 0.015
DEFAULT_DLS_STALL_ITERS = 8
DEFAULT_DLS_STALL_EPS = 1.0e-4

DEFAULT_PREGRASP_TOLERANCE = 0.10
DEFAULT_ALIGN_TOLERANCE = 0.07
DEFAULT_DESCEND_TOLERANCE = 0.045
DEFAULT_LIFT_TOLERANCE = 0.08
DEFAULT_CARRY_TOLERANCE = 0.12
DEFAULT_PLACE_TOLERANCE = 0.09
DEFAULT_RETREAT_TOLERANCE = 0.12

DEFAULT_PREGRASP_CLEARANCE = 0.10
DEFAULT_PREGRASP_STANDOFF = 0.12
DEFAULT_ALIGN_CLEARANCE = 0.055
DEFAULT_DESCEND_CLEARANCE = 0.005
DEFAULT_SAFE_DROP_HEIGHT = 0.12
DEFAULT_PLACE_CLEARANCE = 0.055
DEFAULT_RETREAT_LIFT = 0.16
DEFAULT_MIN_LIFT_DELTA = 0.025
DEFAULT_MIN_TRANSPORT_DISTANCE = 0.08
DEFAULT_STABLE_JITTER = 0.01
DEFAULT_MIN_EE_TABLE_CLEARANCE = 0.025

DEFAULT_WORKSPACE_X = (0.25, 1.60)
DEFAULT_WORKSPACE_Y = (-0.80, 0.85)
DEFAULT_WORKSPACE_Z = (0.50, 1.35)

RIGHT_ARM_TOKENS = ("r_shoulder", "r_elbow", "r_wrist", "right_shoulder", "right_elbow", "right_wrist")
LEFT_ARM_TOKENS = ("l_shoulder", "l_elbow", "l_wrist", "left_shoulder", "left_elbow", "left_wrist")
ARM_EXCLUDE_TOKENS = ("finger", "thumb", "gripper", "hand")
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
    "move_pregrasp",
    "grasp_align",
    "descend_contact",
    "close_gripper",
    "lift_validate",
    "carry_to_bin",
    "place_depth",
    "open_gripper",
    "retreat",
    "settle_and_score",
]


class RunFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    target_position: np.ndarray
    tolerance: float
    max_iters: int
    gripper_effort: float | None = None


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


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
    pseudo_root = stage.GetPseudoRoot()
    for prim in Usd.PrimRange(pseudo_root):
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


def _fail(reason: str, message: str) -> None:
    raise RunFailure(reason, message)


def _vector3(value: Any) -> np.ndarray:
    try:
        return np.array([float(value.x), float(value.y), float(value.z)], dtype=float)
    except AttributeError:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)


def _body_pose_position(dc: Any, body: Any) -> np.ndarray:
    pose = dc.get_rigid_body_pose(body)
    return _vector3(pose.p)


def _body_pose_orientation(dc: Any, body: Any) -> dict[str, float] | None:
    try:
        q = dc.get_rigid_body_pose(body).r
        return {"w": float(q.w), "x": float(q.x), "y": float(q.y), "z": float(q.z)}
    except Exception:
        return None


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
        raise RuntimeError(
            f"Requested end-effector body was not found: {requested_body}. "
            f"Available body paths={[path for _, _, _, path in bodies]}"
        )

    preferred = "R_sixforce_link" if arm_side == "right" else "L_sixforce_link"
    preferred_lower = preferred.lower()
    for _, body, name, path in bodies:
        lower = f"{name} {path}".lower()
        if preferred_lower in lower:
            return body, name, path, "preferred_sixforce"

    tokens = RIGHT_EE_TOKENS if arm_side == "right" else LEFT_EE_TOKENS
    candidates: list[tuple[int, Any, str, str]] = []
    for index, body, name, path in bodies:
        lower = f"{name} {path}".lower()
        if any(token in lower for token in EE_EXCLUDE_TOKENS):
            continue
        if any(token in lower for token in tokens):
            candidates.append((index, body, name, path))
    if not candidates:
        raise RuntimeError(
            f"Could not identify {arm_side} end-effector body. "
            f"Tokens={tokens}; available body paths={[path for _, _, _, path in bodies]}"
        )

    _, body, name, path = candidates[-1]
    return body, name, path, "fallback_local_ee"


def _current_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> np.ndarray:
    return np.array([float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs], dtype=float)


def _named_positions(selected_dofs: list[tuple[int, Any, str]], positions: np.ndarray | list[float]) -> dict[str, float]:
    return {name: float(position) for (_, _, name), position in zip(selected_dofs, positions)}


def _targets_from_map(selected_dofs: list[tuple[int, Any, str]], target_by_name: dict[str, float]) -> list[float]:
    missing = [name for _, _, name in selected_dofs if name not in target_by_name]
    if missing:
        raise RuntimeError(f"Missing target values for DOFs: {missing}")
    return [float(target_by_name[name]) for _, _, name in selected_dofs]


def _read_positions(dc: Any, selected_dofs: list[tuple[int, Any, str]]) -> list[float]:
    return [float(dc.get_dof_position(dof)) for _, dof, _ in selected_dofs]


def _select_arm_dofs(dc: Any, articulation: Any, arm_side: str, max_dofs: int) -> list[tuple[int, Any, str]]:
    tokens = RIGHT_ARM_TOKENS if arm_side == "right" else LEFT_ARM_TOKENS
    selected: list[tuple[int, Any, str]] = []
    for index in range(dc.get_articulation_dof_count(articulation)):
        dof = dc.get_articulation_dof(articulation, index)
        name = str(dc.get_dof_name(dof))
        lower = name.lower()
        if any(token in lower for token in ARM_EXCLUDE_TOKENS):
            continue
        if any(token in lower for token in tokens):
            selected.append((index, dof, name))
        if len(selected) >= max_dofs:
            break
    if not selected:
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
        raise RuntimeError(f"No {arm_side}-arm DOFs matched tokens={tokens}; available_dof_names={all_names}")
    return selected


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
        all_names = [
            str(dc.get_dof_name(dc.get_articulation_dof(articulation, index)))
            for index in range(dc.get_articulation_dof_count(articulation))
        ]
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
    interface_cls = module.IsaacSimRobotInterface
    interface = interface_cls(prim_path=prim_path, name=OFFICIAL_ROBOT_NAME, world=None, urdf_path=str(urdf_path))
    joint_map = dict(getattr(interface, "_joint_value_map", {}))
    missing = sorted(OFFICIAL_STARTUP_ARM_JOINT_NAMES - set(joint_map))
    if missing:
        raise RuntimeError(f"Official startup joint map missing expected arm joints: {missing}")
    return {name: float(value) for name, value in joint_map.items()}


def _seed_joint_positions_for_initialization(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    target_positions: list[float] | np.ndarray,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(target_positions) != len(selected_dofs):
        errors.append(f"target length {len(target_positions)} does not match DOF length {len(selected_dofs)}")
    for index, (_, dof, name), target_value in zip(range(len(selected_dofs)), selected_dofs, target_positions):
        target = float(target_value)
        try:
            if hasattr(dc, "set_dof_position"):
                dc.set_dof_position(dof, target)
            if hasattr(dc, "set_dof_position_target"):
                dc.set_dof_position_target(dof, target)
        except Exception as exc:  # pragma: no cover - Isaac runtime API detail.
            errors.append(f"{index}:{name}: {exc}")
    return {
        "supported": not errors,
        "method": "initialization_only_set_dof_position_and_target",
        "dof_names": [name for _, _, name in selected_dofs],
        "errors": errors,
    }


def _restore_joint_state_for_candidate_evaluation(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    positions: np.ndarray,
) -> dict[str, Any]:
    errors: list[str] = []
    for (_, dof, name), target in zip(selected_dofs, positions):
        try:
            if hasattr(dc, "set_dof_position"):
                dc.set_dof_position(dof, float(target))
            if hasattr(dc, "set_dof_position_target"):
                dc.set_dof_position_target(dof, float(target))
        except Exception as exc:  # pragma: no cover - Isaac runtime API detail.
            errors.append(f"{name}: {exc}")
    return {
        "supported": not errors,
        "method": "candidate_evaluation_state_restore_only",
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
        except Exception as exc:  # pragma: no cover - Isaac runtime API detail.
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


def _write_logs(log_root: Path, payload: dict[str, Any], log_suffix: str | None) -> list[str]:
    log_root.mkdir(parents=True, exist_ok=True)
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
            f"selected_target_prim={payload.get('target', {}).get('prim_path')}",
            f"chosen_arm={payload.get('robot', {}).get('chosen_arm')}",
            f"ee_frame={payload.get('robot', {}).get('end_effector_path')}",
            f"object_lifted={payload['result_flags'].get('object_lifted')}",
            f"object_transported={payload['result_flags'].get('object_transported')}",
            f"final_inside_bin={payload['result_flags'].get('final_inside_bin')}",
            f"payload={json.dumps(payload, indent=2, sort_keys=True)}",
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


def _create_debug_marker(stage: Any, path: str, position: np.ndarray | list[float], radius: float, color: tuple[float, float, float]) -> str:
    from pxr import Gf, UsdGeom  # type: ignore

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    point = np.array(position, dtype=float)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return path


def _compute_robot_base_target_components(
    *,
    object_world: np.ndarray,
    robot_base_position: np.ndarray,
    robot_base_yaw_rad: float,
) -> dict[str, Any]:
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


def _compute_grasp_frame(
    *,
    object_world: np.ndarray,
    ee_world: np.ndarray,
    world_down: np.ndarray | None = None,
) -> dict[str, Any]:
    if world_down is None:
        world_down = np.array([0.0, 0.0, -1.0], dtype=float)
    down = np.array(world_down, dtype=float)
    down = down / max(float(np.linalg.norm(down)), 1.0e-9)
    reach_dir = np.array(object_world, dtype=float) - np.array(ee_world, dtype=float)
    reach_dir = reach_dir - np.dot(reach_dir, down) * down
    reach_norm = float(np.linalg.norm(reach_dir))
    if reach_norm < 1.0e-6:
        seed = np.array([1.0, 0.0, 0.0], dtype=float)
        reach_dir = seed - np.dot(seed, down) * down
        reach_norm = float(np.linalg.norm(reach_dir))
    x_grasp = reach_dir / max(reach_norm, 1.0e-9)
    y_grasp = np.cross(down, x_grasp)
    y_norm = float(np.linalg.norm(y_grasp))
    if y_norm < 1.0e-6:
        y_grasp = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        y_grasp = y_grasp / y_norm
    z_grasp = down
    r_grasp = np.column_stack([x_grasp, y_grasp, z_grasp])
    return {
        "R_grasp_world": r_grasp,
        "x_grasp_world": x_grasp,
        "y_grasp_world": y_grasp,
        "z_grasp_world": z_grasp,
    }


def _compute_tcp_target_world(
    *,
    object_world: np.ndarray,
    r_grasp_world: np.ndarray,
    tcp_offset_local: np.ndarray,
    contact_z_world: float,
) -> tuple[np.ndarray, np.ndarray]:
    tcp_offset_world = np.array(r_grasp_world, dtype=float) @ np.array(tcp_offset_local, dtype=float)
    target = np.array(object_world, dtype=float).copy()
    target[2] = float(contact_z_world)
    return target - tcp_offset_world, tcp_offset_world


def _compute_contact_z_world(
    *,
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
    current_ee_world: np.ndarray,
    table_top_z: float,
    bin_bbox: dict[str, list[float]],
    bin_floor_top_z: float,
    tcp_offset_local: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    bbox = object_state["bbox"]
    object_world = _center_from_bbox(bbox)
    contact_z = _compute_contact_z_world(
        bbox_top_z=float(bbox["max"][2]),
        table_top_z=table_top_z,
        descend_clearance=args.descend_clearance,
        grasp_depth_offset=args.grasp_depth_offset,
        min_ee_table_clearance=args.min_ee_table_clearance,
    )
    grasp_frame = _compute_grasp_frame(object_world=object_world, ee_world=current_ee_world)
    contact_target, tcp_offset_world = _compute_tcp_target_world(
        object_world=object_world,
        r_grasp_world=grasp_frame["R_grasp_world"],
        tcp_offset_local=tcp_offset_local,
        contact_z_world=contact_z,
    )

    pregrasp_target = contact_target - grasp_frame["x_grasp_world"] * float(args.pregrasp_standoff)
    pregrasp_target[2] += float(args.pregrasp_clearance)

    align_target = contact_target - grasp_frame["x_grasp_world"] * float(args.pregrasp_standoff * 0.35)
    align_target[2] += float(args.align_clearance)

    lift_target = contact_target.copy()
    lift_target[2] = max(contact_target[2] + float(args.pregrasp_clearance), object_world[2] + float(args.pregrasp_clearance) + float(args.min_lift_delta))

    bin_center = np.array(bin_bbox["center"], dtype=float)
    carry_target = np.array([bin_center[0], bin_center[1], float(bin_bbox["max"][2]) + float(args.safe_drop_height)], dtype=float)
    place_target = np.array(
        [
            bin_center[0],
            bin_center[1],
            max(float(bin_floor_top_z) + 0.12, float(bin_bbox["max"][2]) + float(args.place_clearance)),
        ],
        dtype=float,
    )
    retreat_target = place_target.copy()
    retreat_target[2] += float(args.retreat_lift)

    return {
        "object_center_world": object_world.tolist(),
        "contact_z_world": float(contact_z),
        "R_grasp_world": grasp_frame["R_grasp_world"].tolist(),
        "x_grasp_world": grasp_frame["x_grasp_world"].tolist(),
        "y_grasp_world": grasp_frame["y_grasp_world"].tolist(),
        "z_grasp_world": grasp_frame["z_grasp_world"].tolist(),
        "tcp_offset_local": tcp_offset_local.tolist(),
        "tcp_offset_world": tcp_offset_world.tolist(),
        "pregrasp_target_world": pregrasp_target.tolist(),
        "align_target_world": align_target.tolist(),
        "contact_target_world": contact_target.tolist(),
        "lift_target_world": lift_target.tolist(),
        "carry_target_world": carry_target.tolist(),
        "place_target_world": place_target.tolist(),
        "retreat_target_world": retreat_target.tolist(),
    }


def _pregrasp_candidates(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    base = np.array(geometry["pregrasp_target_world"], dtype=float)
    x_axis = np.array(geometry["x_grasp_world"], dtype=float)
    y_axis = np.array(geometry["y_grasp_world"], dtype=float)
    variants = [
        ("nominal", 0.0, 0.0, 0.0),
        ("higher", 0.0, 0.0, 0.04),
        ("nearer", 0.035, 0.0, 0.02),
        ("farther", -0.035, 0.0, 0.02),
        ("lateral_positive", 0.0, 0.04, 0.02),
        ("lateral_negative", 0.0, -0.04, 0.02),
    ]
    candidates: list[dict[str, Any]] = []
    for label, x_delta, y_delta, z_delta in variants:
        target = base + x_axis * float(x_delta) + y_axis * float(y_delta) + np.array([0.0, 0.0, float(z_delta)], dtype=float)
        candidates.append({"label": label, "target_position": target})
    return candidates


def _apply_positions(
    dc: Any,
    selected_dofs: list[tuple[int, Any, str]],
    positions: np.ndarray,
    sim_app: Any,
    counter: dict[str, int],
    track_steps: int,
    articulation: Any | None = None,
    gripper_dofs: list[tuple[int, Any, str]] | None = None,
    gripper_effort: float | None = None,
) -> None:
    _send_position_targets(dc, selected_dofs, [float(value) for value in positions])
    _run_updates(sim_app, track_steps, counter, dc=dc, gripper_dofs=gripper_dofs, gripper_effort=gripper_effort)
    if articulation is not None:
        dc.wake_up_articulation(articulation)


def _estimate_position_jacobian(
    dc: Any,
    articulation: Any,
    selected_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    base_positions: np.ndarray,
    base_ee_position: np.ndarray,
    sim_app: Any,
    counter: dict[str, int],
    eps: float,
    track_steps: int,
    gripper_dofs: list[tuple[int, Any, str]] | None,
    gripper_effort: float | None,
) -> np.ndarray:
    jacobian = np.zeros((3, len(selected_dofs)), dtype=float)
    for column in range(len(selected_dofs)):
        trial = base_positions.copy()
        trial[column] += float(eps)
        _apply_positions(
            dc,
            selected_dofs,
            trial,
            sim_app,
            counter,
            track_steps,
            articulation=articulation,
            gripper_dofs=gripper_dofs,
            gripper_effort=gripper_effort,
        )
        moved_position = _body_pose_position(dc, end_effector_body)
        jacobian[:, column] = (moved_position - base_ee_position) / float(eps)
    _apply_positions(
        dc,
        selected_dofs,
        base_positions,
        sim_app,
        counter,
        track_steps,
        articulation=articulation,
        gripper_dofs=gripper_dofs,
        gripper_effort=gripper_effort,
    )
    return jacobian


def _execute_cartesian_dls_phase(
    spec: PhaseSpec,
    *,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_step = counter["step"]
    target = np.array(spec.target_position, dtype=float)
    start_ee = _body_pose_position(dc, end_effector_body)
    positions = _current_positions(dc, arm_dofs)
    posture = positions.copy()
    best_ee = start_ee.copy()
    best_error = float(np.linalg.norm(target - best_ee))
    best_positions = positions.copy()
    previous_best_error = best_error
    stale_iters = 0
    stalled = False
    workspace_violation = False
    failure_reason: str | None = None
    trace: list[dict[str, Any]] = []

    target_workspace = _workspace_check(target, args.workspace_x, args.workspace_y, args.workspace_z)
    if not target_workspace["workspace_ok"]:
        workspace_violation = True

    for iteration in range(1, spec.max_iters + 1):
        current_ee = _body_pose_position(dc, end_effector_body)
        error_vector = target - current_ee
        error = float(np.linalg.norm(error_vector))
        observed_positions = _current_positions(dc, arm_dofs)
        if error < best_error:
            best_ee = current_ee.copy()
            best_error = error
            best_positions = positions.copy()
        if error <= spec.tolerance:
            break

        jacobian = _estimate_position_jacobian(
            dc,
            articulation,
            arm_dofs,
            end_effector_body,
            positions,
            current_ee,
            sim_app,
            counter,
            args.dls_eps,
            args.dls_track_steps,
            gripper_dofs,
            spec.gripper_effort,
        )
        lhs = jacobian @ jacobian.T + (float(args.dls_damping) ** 2) * np.eye(3)
        try:
            delta = jacobian.T @ np.linalg.solve(lhs, error_vector)
        except np.linalg.LinAlgError:
            delta = jacobian.T @ np.linalg.pinv(lhs) @ error_vector
        delta += float(args.dls_posture_gain) * (posture - positions)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > float(args.dls_max_joint_step):
            delta *= float(args.dls_max_joint_step) / max(delta_norm, 1.0e-9)

        positions = positions + delta
        _apply_positions(
            dc,
            arm_dofs,
            positions,
            sim_app,
            counter,
            args.dls_track_steps,
            articulation=articulation,
            gripper_dofs=gripper_dofs,
            gripper_effort=spec.gripper_effort,
        )
        updated_ee = _body_pose_position(dc, end_effector_body)
        updated_error = float(np.linalg.norm(target - updated_ee))
        workspace_status = _workspace_check(updated_ee, args.workspace_x, args.workspace_y, args.workspace_z)
        workspace_violation = bool(workspace_violation or not workspace_status["workspace_ok"])
        if updated_error < best_error:
            best_ee = updated_ee.copy()
            best_error = updated_error
            best_positions = positions.copy()

        if previous_best_error - best_error < float(args.dls_stall_eps):
            stale_iters += 1
        else:
            stale_iters = 0
        previous_best_error = best_error
        stalled = stale_iters >= int(args.dls_stall_iters)

        row = {
            "iteration": iteration,
            "target_position": target.tolist(),
            "ee_position": updated_ee.tolist(),
            "error": updated_error,
            "best_error": best_error,
            "delta_norm": delta_norm,
            "workspace_ok": workspace_status["workspace_ok"],
            "observed_joint_positions": _named_positions(arm_dofs, observed_positions),
            "commanded_joint_targets": _named_positions(arm_dofs, positions),
        }
        trace.append(row)
        print(
            f"phase={spec.name} iter={iteration}/{spec.max_iters} "
            f"target={target.tolist()} ee={updated_ee.tolist()} error={updated_error:.4f} best={best_error:.4f}"
        )
        if stalled:
            failure_reason = "stalled"
            break

    final_ee = _body_pose_position(dc, end_effector_body)
    final_error = float(np.linalg.norm(target - final_ee))
    if best_error < final_error:
        _apply_positions(
            dc,
            arm_dofs,
            best_positions,
            sim_app,
            counter,
            args.dls_track_steps,
            articulation=articulation,
            gripper_dofs=gripper_dofs,
            gripper_effort=spec.gripper_effort,
        )
        final_ee = _body_pose_position(dc, end_effector_body)
        final_error = float(np.linalg.norm(target - final_ee))

    success = bool(best_error <= float(spec.tolerance))
    if not success and failure_reason is None:
        failure_reason = "tolerance_not_met"
    if workspace_violation and not success:
        failure_reason = failure_reason or "workspace_violation"

    details: dict[str, Any] = {
        "target_position": target.tolist(),
        "start_ee_position": start_ee.tolist(),
        "final_ee_position": final_ee.tolist(),
        "best_ee_position": best_ee.tolist(),
        "final_error": final_error,
        "best_error": best_error,
        "tolerance": float(spec.tolerance),
        "iteration_count": len(trace),
        "failure_reason": None if success else failure_reason,
        "stalled": bool(stalled),
        "workspace_violation": bool(workspace_violation),
        "target_workspace_check": target_workspace,
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
        "start_ee_orientation_quat": _body_pose_orientation(dc, end_effector_body),
        "final_joint_positions": _named_positions(arm_dofs, _current_positions(dc, arm_dofs)),
        "dls_parameters": {
            "measured_body_positions": True,
            "position_only": True,
            "max_iters": int(spec.max_iters),
            "eps": float(args.dls_eps),
            "damping": float(args.dls_damping),
            "max_joint_step": float(args.dls_max_joint_step),
            "track_steps": int(args.dls_track_steps),
            "posture_gain": float(args.dls_posture_gain),
            "stall_iters": int(args.dls_stall_iters),
            "stall_eps": float(args.dls_stall_eps),
        },
        "trace": trace,
    }
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
    print(f"phase={spec.name} condition_met={success} final_error={final_error:.4f} best_error={best_error:.4f}")
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
) -> bool:
    start_step = counter["step"]
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

    details = {
        "target_position": None,
        "start_ee_position": None,
        "final_ee_position": _body_pose_position(dc, end_effector_body).tolist(),
        "best_ee_position": _body_pose_position(dc, end_effector_body).tolist(),
        "final_error": None,
        "best_error": None,
        "iteration_count": int(steps),
        "failure_reason": None if command_supported else "gripper_command_empty",
        "stalled": False,
        "workspace_violation": False,
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
    }
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


def _evaluate_pregrasp_candidates(
    *,
    dc: Any,
    articulation: Any,
    arm_dofs: list[tuple[int, Any, str]],
    end_effector_body: Any,
    end_effector_name: str,
    end_effector_path: str,
    end_effector_policy: str,
    gripper_dofs: list[tuple[int, Any, str]],
    sim_app: Any,
    args: argparse.Namespace,
    counter: dict[str, int],
    phase_log: list[dict[str, Any]],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    start_step = counter["step"]
    reference_positions = _current_positions(dc, arm_dofs)
    reference_ee = _body_pose_position(dc, end_effector_body)
    candidates = _pregrasp_candidates(geometry)
    results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for index, candidate in enumerate(candidates):
        restore_before = _restore_joint_state_for_candidate_evaluation(dc, arm_dofs, reference_positions)
        _run_updates(sim_app, args.candidate_restore_steps, counter)
        candidate_start_ee = _body_pose_position(dc, end_effector_body)
        candidate_start_positions = _current_positions(dc, arm_dofs)
        temp_log: list[dict[str, Any]] = []
        spec = PhaseSpec(
            name=f"candidate_pregrasp_{index}_{candidate['label']}",
            target_position=np.array(candidate["target_position"], dtype=float),
            tolerance=float(args.pregrasp_tolerance),
            max_iters=int(args.candidate_dls_max_iters),
        )
        details = _execute_cartesian_dls_phase(
            spec,
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=temp_log,
            extra_details={
                "candidate_index": index,
                "candidate_label": candidate["label"],
                "candidate_evaluation_start_ee_position": candidate_start_ee.tolist(),
                "candidate_evaluation_start_joint_positions": _named_positions(arm_dofs, candidate_start_positions),
                "candidate_restore_before_trial": restore_before,
            },
        )
        candidate_success = bool(details["best_error"] <= float(args.pregrasp_tolerance))
        results.append(
            {
                "candidate_index": index,
                "candidate_label": candidate["label"],
                "target_position": np.array(candidate["target_position"], dtype=float).tolist(),
                "success": candidate_success,
                "best_error": details["best_error"],
                "final_error": details["final_error"],
                "failure_reason": details["failure_reason"],
                "candidate_evaluation_start_ee_position": candidate_start_ee.tolist(),
                "candidate_evaluation_start_joint_positions": _named_positions(arm_dofs, candidate_start_positions),
                "restore_before_trial": restore_before,
            }
        )
        restore_after = _restore_joint_state_for_candidate_evaluation(dc, arm_dofs, reference_positions)
        _run_updates(sim_app, args.candidate_restore_steps, counter)
        results[-1]["restore_after_trial"] = restore_after
        results[-1]["post_restore_ee_position"] = _body_pose_position(dc, end_effector_body).tolist()
        if candidate_success and selected is None:
            selected = results[-1]
            break

    success = selected is not None
    details = {
        "target_position": None if selected is None else selected["target_position"],
        "start_ee_position": reference_ee.tolist(),
        "final_ee_position": _body_pose_position(dc, end_effector_body).tolist(),
        "best_ee_position": None,
        "final_error": None if selected is None else selected["final_error"],
        "best_error": None if selected is None else selected["best_error"],
        "iteration_count": len(results),
        "failure_reason": None if success else "no_reachable_pregrasp_candidate",
        "stalled": False,
        "workspace_violation": False,
        "chosen_ee_frame_name": end_effector_name,
        "chosen_ee_frame_path": end_effector_path,
        "chosen_ee_frame_policy": end_effector_policy,
        "candidate_reference_ee_position": reference_ee.tolist(),
        "candidate_reference_joint_positions": _named_positions(arm_dofs, reference_positions),
        "candidate_restore_policy": "restore_snapshot_positions_and_targets_before_each_trial_and_after_failed_trials",
        "candidate_results": results,
        "selected_candidate": selected,
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
        _fail("pregrasp_candidate_failed", "No pregrasp candidate reached the conservative DLS tolerance")
    return selected


def _choose_arm_side(args: argparse.Namespace, target_components: dict[str, Any]) -> str:
    if args.arm in ("left", "right"):
        return str(args.arm)
    lateral = float(target_components["lateral_base"])
    return "left" if lateral > 0.0 else "right"


def _parse_tcp_offset(cfg: dict[str, Any]) -> np.ndarray:
    raw = cfg.get("grasp", {}).get("tcp_offset", [0.0, 0.0, 0.0])
    if isinstance(raw, dict):
        raw = [raw.get("x", 0.0), raw.get("y", 0.0), raw.get("z", 0.0)]
    values = list(raw)
    if len(values) != 3:
        return np.zeros(3, dtype=float)
    return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--asset-root")
    parser.add_argument(
        "--prim-path",
        help="Optional manual robot prim fallback. By default the script uses scene.robot_prim_path after SceneBuilder.build_robot().",
    )
    parser.add_argument("--end-effector-body")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--arm", choices=("auto", "right", "left"), default="auto")
    parser.add_argument("--max-arm-dofs", type=int, default=7)
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--gripper-steps", type=int, default=DEFAULT_GRIPPER_STEPS)
    parser.add_argument("--candidate-restore-steps", type=int, default=4)
    parser.add_argument("--candidate-dls-max-iters", type=int, default=18)
    parser.add_argument("--dls-max-iters", type=int, default=DEFAULT_DLS_MAX_ITERS)
    parser.add_argument("--dls-eps", type=float, default=DEFAULT_DLS_EPS)
    parser.add_argument("--dls-damping", type=float, default=DEFAULT_DLS_DAMPING)
    parser.add_argument("--dls-max-joint-step", type=float, default=DEFAULT_DLS_MAX_JOINT_STEP)
    parser.add_argument("--dls-track-steps", type=int, default=DEFAULT_DLS_TRACK_STEPS)
    parser.add_argument("--dls-posture-gain", type=float, default=DEFAULT_DLS_POSTURE_GAIN)
    parser.add_argument("--dls-stall-iters", type=int, default=DEFAULT_DLS_STALL_ITERS)
    parser.add_argument("--dls-stall-eps", type=float, default=DEFAULT_DLS_STALL_EPS)
    parser.add_argument("--pregrasp-tolerance", type=float, default=DEFAULT_PREGRASP_TOLERANCE)
    parser.add_argument("--align-tolerance", type=float, default=DEFAULT_ALIGN_TOLERANCE)
    parser.add_argument("--descend-tolerance", type=float, default=DEFAULT_DESCEND_TOLERANCE)
    parser.add_argument("--lift-tolerance", type=float, default=DEFAULT_LIFT_TOLERANCE)
    parser.add_argument("--carry-tolerance", type=float, default=DEFAULT_CARRY_TOLERANCE)
    parser.add_argument("--place-tolerance", type=float, default=DEFAULT_PLACE_TOLERANCE)
    parser.add_argument("--retreat-tolerance", type=float, default=DEFAULT_RETREAT_TOLERANCE)
    parser.add_argument("--pregrasp-clearance", type=float, default=DEFAULT_PREGRASP_CLEARANCE)
    parser.add_argument("--pregrasp-standoff", type=float, default=DEFAULT_PREGRASP_STANDOFF)
    parser.add_argument("--align-clearance", type=float, default=DEFAULT_ALIGN_CLEARANCE)
    parser.add_argument("--descend-clearance", type=float, default=DEFAULT_DESCEND_CLEARANCE)
    parser.add_argument("--grasp-depth-offset", type=float, default=0.0)
    parser.add_argument("--safe-drop-height", type=float, default=DEFAULT_SAFE_DROP_HEIGHT)
    parser.add_argument("--place-clearance", type=float, default=DEFAULT_PLACE_CLEARANCE)
    parser.add_argument("--retreat-lift", type=float, default=DEFAULT_RETREAT_LIFT)
    parser.add_argument("--min-lift-delta", type=float, default=DEFAULT_MIN_LIFT_DELTA)
    parser.add_argument("--min-transport-distance", type=float, default=DEFAULT_MIN_TRANSPORT_DISTANCE)
    parser.add_argument("--stable-jitter", type=float, default=DEFAULT_STABLE_JITTER)
    parser.add_argument("--min-ee-table-clearance", type=float, default=DEFAULT_MIN_EE_TABLE_CLEARANCE)
    parser.add_argument("--workspace-x", type=float, nargs=2, default=DEFAULT_WORKSPACE_X)
    parser.add_argument("--workspace-y", type=float, nargs=2, default=DEFAULT_WORKSPACE_Y)
    parser.add_argument("--workspace-z", type=float, nargs=2, default=DEFAULT_WORKSPACE_Z)
    parser.add_argument("--gripper-hold-effort", type=float, default=DEFAULT_GRIPPER_HOLD_EFFORT)
    parser.add_argument("--joint-tolerance", type=float, default=0.06)
    parser.add_argument("--skip-gripper-close", action="store_true")
    parser.add_argument("--skip-release", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--log-suffix")
    args = parser.parse_args()

    if args.target_index < 0:
        raise RuntimeError("--target-index must be non-negative")
    if args.max_arm_dofs < 1:
        raise RuntimeError("--max-arm-dofs must be positive")
    if args.dls_track_steps < 1 or args.dls_max_iters < 1 or args.candidate_dls_max_iters < 1:
        raise RuntimeError("DLS iteration and track-step arguments must be positive")
    if args.prim_path is not None and not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")
    args.workspace_x = tuple(float(value) for value in args.workspace_x)
    args.workspace_y = tuple(float(value) for value in args.workspace_y)
    args.workspace_z = tuple(float(value) for value in args.workspace_z)

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
            "architecture": "phase_based_measured_isaac_cartesian_dls_position_only",
            "old_reach_stack_imported": False,
            "pinocchio_6d_solve_active": False,
            "orientation_control_active": False,
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
    log_paths: list[str] = []

    try:
        paths = _validate_environment()
        baseline_root = _as_path(args.baseline_root, paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
        asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
        config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
        box_usd = asset_root / DEFAULT_BOX_RELATIVE
        log_root = Path(paths["LOG_ROOT"]).resolve()
        payload["run_metadata"].update(
            {
                "repo_path": str(paths["HRC_REPO"]),
                "baseline_root": str(baseline_root),
                "asset_root": str(asset_root),
                "yaml_path": str(config_path),
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
        print(f"stage_debug root_valid={_valid_prim_path(stage, '/Root')} world_valid={_valid_prim_path(stage, '/World')}")
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        print("scene_build_debug before=build_table")
        scene.build_table()
        print("scene_build_debug after=build_table")
        print("scene_build_debug before=build_parts")
        scene.build_parts()
        print("scene_build_debug after=build_parts")

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
        print("scene_build_debug before=build_robot")
        scene.build_robot()
        print("scene_build_debug after=build_robot")
        scene_robot_prim_path = getattr(scene, "robot_prim_path", None)

        rep.orchestrator.step()
        _run_updates(sim_app, args.init_steps, counter)
        detected_articulation_roots = _find_articulation_roots_anywhere(stage)
        robot_path_selection = _choose_robot_prim_path(
            stage,
            scene_robot_prim_path,
            detected_articulation_roots,
            args.prim_path,
        )
        chosen_robot_prim_path = robot_path_selection["chosen_robot_prim_path"]
        articulation_path = detected_articulation_roots[0] if detected_articulation_roots else chosen_robot_prim_path
        payload["robot_path_selection"] = robot_path_selection
        print(
            "robot_path_selection "
            f"scene_robot_prim_path={scene_robot_prim_path} "
            f"detected_articulation_roots={detected_articulation_roots} "
            f"chosen_robot_prim_path={chosen_robot_prim_path} "
            f"fallback_used={robot_path_selection['fallback_used']}"
        )

        part_paths = list(getattr(scene, "parts_prim_paths", []))
        if not part_paths:
            _fail("no_target_parts_found", "SceneBuilder did not expose any Task 1 part prim paths")
        if args.target_index >= len(part_paths):
            _fail("target_index_out_of_range", f"--target-index {args.target_index} out of range for {len(part_paths)} parts")

        table_path = "/Replicator/Ref_Xform"
        table_bbox = _bbox(stage, table_path)
        table_top_z = float(table_bbox["max"][2])
        target_path = part_paths[args.target_index]
        target_category = _category_from_target(stage, target_path, args.target_index, int(cfg["part"].get("num_parts", 2)))
        initial_state = _bbox_state(stage, target_path)
        initial_center = _center_from_bbox(initial_state["bbox"])
        target_components = _compute_robot_base_target_components(
            object_world=initial_center,
            robot_base_position=np.array(configured_robot_position, dtype=float),
            robot_base_yaw_rad=math.radians(float(configured_robot_rotation[2]) if len(configured_robot_rotation) >= 3 else 0.0),
        )
        chosen_arm = _choose_arm_side(args, target_components)

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
            "target_index": args.target_index,
            "initial_pose": initial_state,
            "robot_base_target_components": target_components,
            **target_category,
        }
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
                "start_ee_position": None,
                "final_ee_position": None,
                "best_ee_position": None,
                "final_error": None,
                "best_error": None,
                "iteration_count": 0,
                "failure_reason": None,
                "stalled": False,
                "workspace_violation": False,
                "selected_target": payload["target"],
                "chosen_arm_preliminary": chosen_arm,
            },
        )
        print(f"phase=select_target target={target_path} center={initial_center.tolist()} chosen_arm={chosen_arm}")

        joint_names = _find_joint_names(stage, chosen_robot_prim_path)
        if not detected_articulation_roots and not _valid_prim_path(stage, articulation_path):
            _fail("scene_build_failed", "Walker S2 loaded, but no articulation root was detected")

        timeline = _start_timeline()
        _run_updates(sim_app, 5, counter)

        dc, articulation = _acquire_articulation(articulation_path)
        dof_observation = _read_dof_observation(dc, articulation)
        arm_dofs = _select_arm_dofs(dc, articulation, chosen_arm, args.max_arm_dofs)
        gripper_dofs = _select_gripper_dofs(dc, articulation, chosen_arm)
        end_effector_body, end_effector_name, end_effector_path, end_effector_policy = _identify_end_effector_body(
            dc,
            articulation,
            args.end_effector_body,
            chosen_arm,
        )
        official_startup_joint_map = _load_official_startup_joint_map(baseline_root, chosen_robot_prim_path, asset_root / "s2.urdf")
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
            "fallback_used_for_robot_prim_path": bool(robot_path_selection["fallback_used"]),
            "prim_path": chosen_robot_prim_path,
            "requested_prim_path": args.prim_path,
            "configured_position_from_yaml": configured_robot_position,
            "configured_rotation_xyz_deg_from_yaml": configured_robot_rotation,
            "config_robot_pose_applied_by": "official_SceneBuilder.build_robot",
            "articulation_path": articulation_path,
            "joint_count": len(joint_names),
            "chosen_arm": chosen_arm,
            "arm_dof_names": [name for _, _, name in arm_dofs],
            "gripper_dof_names": [name for _, _, name in gripper_dofs],
            "official_startup_dof_names": [name for _, _, name in startup_dofs],
            "missing_optional_official_startup_dofs": missing_official_startup_optional_dofs,
            "official_startup_source": "lerobot.common.robot_devices.robots.isaac_sim_robot_interface.IsaacSimRobotInterface._joint_value_map",
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
                "start_ee_position": None,
                "final_ee_position": ee_startup.tolist(),
                "best_ee_position": ee_startup.tolist(),
                "final_error": startup_max_error,
                "best_error": startup_max_error,
                "iteration_count": 12,
                "failure_reason": None if startup_ok else "official_startup_pose_failed",
                "stalled": False,
                "workspace_violation": False,
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

        current_ee = _body_pose_position(dc, end_effector_body)
        tcp_offset = _parse_tcp_offset(cfg)
        geometry = _plan_grasp_geometry(
            object_state=initial_state,
            current_ee_world=current_ee,
            table_top_z=table_top_z,
            bin_bbox=bin_bbox,
            bin_floor_top_z=float(bin_collider["floor_top_z"]),
            tcp_offset_local=tcp_offset,
            args=args,
        )
        payload["grasp_geometry"] = geometry
        debug_markers = [
            _create_debug_marker(stage, "/World/DebugCartesianDlsTarget", initial_center, 0.025, (1.0, 0.2, 0.1)),
            _create_debug_marker(stage, "/World/DebugCartesianDlsPregrasp", geometry["pregrasp_target_world"], 0.025, (0.2, 0.6, 1.0)),
            _create_debug_marker(stage, "/World/DebugCartesianDlsContact", geometry["contact_target_world"], 0.022, (1.0, 0.8, 0.1)),
            _create_debug_marker(stage, "/World/DebugCartesianDlsBin", bin_bbox["center"], 0.03, (0.2, 1.0, 0.2)),
        ]
        payload["scene"]["debug_marker_paths"] = debug_markers
        plan_start = counter["step"]
        _append_phase(
            phase_log,
            phase="plan_grasp_geometry",
            start_step=plan_start,
            end_step=counter["step"],
            condition_met=True,
            details={
                "target_position": geometry["contact_target_world"],
                "start_ee_position": current_ee.tolist(),
                "final_ee_position": current_ee.tolist(),
                "best_ee_position": current_ee.tolist(),
                "final_error": None,
                "best_error": None,
                "iteration_count": 0,
                "failure_reason": None,
                "stalled": False,
                "workspace_violation": False,
                "chosen_ee_frame_name": end_effector_name,
                "chosen_ee_frame_path": end_effector_path,
                "chosen_ee_frame_policy": end_effector_policy,
                "geometry": geometry,
            },
        )
        print(f"phase=plan_grasp_geometry contact={geometry['contact_target_world']} pregrasp={geometry['pregrasp_target_world']}")

        selected_candidate = _evaluate_pregrasp_candidates(
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            geometry=geometry,
        )

        pregrasp_result = _execute_cartesian_dls_phase(
            PhaseSpec("move_pregrasp", np.array(selected_candidate["target_position"], dtype=float), args.pregrasp_tolerance, args.dls_max_iters),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            extra_details={"selected_pregrasp_candidate": selected_candidate},
        )
        if pregrasp_result["best_error"] > args.pregrasp_tolerance:
            _fail("pregrasp_failed", "Pregrasp phase did not reach selected Cartesian target")

        align_result = _execute_cartesian_dls_phase(
            PhaseSpec("grasp_align", np.array(geometry["align_target_world"], dtype=float), args.align_tolerance, args.dls_max_iters),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        if align_result["best_error"] > args.align_tolerance:
            _fail("align_failed", "Grasp align phase did not reach Cartesian target")

        descend_result = _execute_cartesian_dls_phase(
            PhaseSpec("descend_contact", np.array(geometry["contact_target_world"], dtype=float), args.descend_tolerance, args.dls_max_iters),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        after_descend = _bbox_state(stage, target_path)
        payload["object_trace"]["after_descend"] = after_descend
        ee_descend = np.array(descend_result["final_ee_position"], dtype=float)
        ee_to_object_before_close = _distance(ee_descend, _center_from_bbox(after_descend["bbox"]))
        phase_log[-1]["details"]["ee_to_object_before_close"] = ee_to_object_before_close
        if descend_result["best_error"] > args.descend_tolerance:
            _fail("descend_failed", "Descend/contact phase failed to reach contact target")

        close_ok = _command_gripper_phase(
            "close_gripper",
            dc=dc,
            gripper_dofs=gripper_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            target_positions=[OFFICIAL_GRIPPER_CLOSE_WIDTH] * len(gripper_dofs),
            sim_app=sim_app,
            steps=args.gripper_steps,
            counter=counter,
            phase_log=phase_log,
            skipped=args.skip_gripper_close,
            effort_value=args.gripper_hold_effort,
        )
        payload["object_trace"]["after_close"] = _bbox_state(stage, target_path)
        if not close_ok:
            _fail("close_gripper_failed", f"{chosen_arm} gripper close command failed")

        validation_before = _bbox_state(stage, target_path)
        validation_before_center = _center_from_bbox(validation_before["bbox"])
        lift_result = _execute_cartesian_dls_phase(
            PhaseSpec(
                "lift_validate",
                np.array(geometry["lift_target_world"], dtype=float),
                args.lift_tolerance,
                args.dls_max_iters,
                gripper_effort=args.gripper_hold_effort,
            ),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
            extra_details={"object_pose_before_lift_validate": validation_before},
        )
        validation_after = _bbox_state(stage, target_path)
        validation_after_center = _center_from_bbox(validation_after["bbox"])
        validation_delta = validation_after_center - validation_before_center
        ee_to_object_after_validation = _distance(np.array(lift_result["final_ee_position"], dtype=float), validation_after_center)
        object_lifted = bool(validation_delta[2] >= args.min_lift_delta and ee_to_object_after_validation <= max(0.18, args.descend_tolerance * 3.0))
        payload["object_trace"]["after_lift_validate"] = validation_after
        payload["result_flags"]["object_lifted"] = object_lifted
        phase_log[-1]["condition_met"] = bool(object_lifted)
        phase_log[-1]["details"].update(
            {
                "object_pose_after_lift_validate": validation_after,
                "object_delta_during_lift_validate_m": validation_delta.tolist(),
                "ee_to_object_after_lift_validate": ee_to_object_after_validation,
                "object_lifted": object_lifted,
            }
        )
        if not object_lifted:
            _fail("object_not_lifted", "Object did not move upward with the gripper during lift validation")

        carry_result = _execute_cartesian_dls_phase(
            PhaseSpec(
                "carry_to_bin",
                np.array(geometry["carry_target_world"], dtype=float),
                args.carry_tolerance,
                args.dls_max_iters,
                gripper_effort=args.gripper_hold_effort,
            ),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        after_carry = _bbox_state(stage, target_path)
        payload["object_trace"]["after_carry_to_bin"] = after_carry
        carry_center = _center_from_bbox(after_carry["bbox"])
        distance_to_bin_initial = _distance(initial_center, np.array(bin_bbox["center"], dtype=float))
        distance_to_bin_after = _distance(carry_center, np.array(bin_bbox["center"], dtype=float))
        object_transported = bool(
            _distance(carry_center, initial_center) >= args.min_transport_distance
            and distance_to_bin_after < distance_to_bin_initial
            and carry_result["best_error"] <= args.carry_tolerance
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
        if carry_result["best_error"] > args.carry_tolerance:
            _fail("carry_failed", "Arm/EE failed to reach carry target")
        if not object_transported:
            _fail("dropped_during_transport", "Object did not move the required minimum distance toward the destination bin")

        place_result = _execute_cartesian_dls_phase(
            PhaseSpec(
                "place_depth",
                np.array(geometry["place_target_world"], dtype=float),
                args.place_tolerance,
                args.dls_max_iters,
                gripper_effort=args.gripper_hold_effort,
            ),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        if place_result["best_error"] > args.place_tolerance:
            _fail("place_failed", "Arm/EE failed to reach place target")

        release_ok = _command_gripper_phase(
            "open_gripper",
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
            skipped=args.skip_release,
            effort_value=0.0,
        )
        payload["object_trace"]["after_release"] = _bbox_state(stage, target_path)
        if not release_ok:
            _fail("release_failed", f"{chosen_arm} gripper release command failed near place target")

        retreat_result = _execute_cartesian_dls_phase(
            PhaseSpec("retreat", np.array(geometry["retreat_target_world"], dtype=float), args.retreat_tolerance, args.dls_max_iters),
            dc=dc,
            articulation=articulation,
            arm_dofs=arm_dofs,
            end_effector_body=end_effector_body,
            end_effector_name=end_effector_name,
            end_effector_path=end_effector_path,
            end_effector_policy=end_effector_policy,
            gripper_dofs=gripper_dofs,
            sim_app=sim_app,
            args=args,
            counter=counter,
            phase_log=phase_log,
        )
        if retreat_result["best_error"] > args.retreat_tolerance:
            _fail("retreat_failed", "Retreat phase failed to reach Cartesian target")

        settle_start = counter["step"]
        final_state, final_jitter = _settle_and_measure(stage, target_path, sim_app, args.settle_steps, counter)
        final_center = _center_from_bbox(final_state["bbox"])
        final_inside_bin = _inside_bin(
            final_center,
            bin_bbox,
            float(bin_collider["wall_thickness"]),
            float(bin_collider["floor_top_z"]),
        )
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
                "start_ee_position": None,
                "final_ee_position": _body_pose_position(dc, end_effector_body).tolist(),
                "best_ee_position": None,
                "final_error": None,
                "best_error": None,
                "iteration_count": args.settle_steps,
                "failure_reason": None if final_inside_bin and object_stable else "object_outside_bin_or_unstable",
                "stalled": False,
                "workspace_violation": False,
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
