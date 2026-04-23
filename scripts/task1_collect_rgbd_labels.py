#!/usr/bin/env python3
"""Collect Task 1 RGB-D samples with simulator truth labels.

This collector is Phase 1 data infrastructure only. It builds the official
Task 1 scene, reuses RobotArticulation.get_cameras_images(step), and writes
synchronized camera arrays plus structured simulator-truth labels/metadata.
It does not run manipulation, replace the planner, or connect a Thinker policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from load_walker_s2 import _create_minimal_scene, _load_simulation_app, _validate_environment  # type: ignore
from validate_task1_object_assets import _bbox  # type: ignore
from validate_task1_scene_builder_scene import (  # type: ignore
    DEFAULT_ASSET_ROOT_RELATIVE,
    DEFAULT_BASELINE_RELATIVE,
    DEFAULT_CONFIG_RELATIVE,
    _NullDataLogger,
    _category_from_reference,
    _load_official_scene_builder,
    _reference_paths,
)


SCRIPT_NAME = "task1_collect_rgbd_labels.py"
LOG_STEM = "task1_rgbd_collection"
OUTPUT_DATASET_RELATIVE = "datasets/task1_rgbd_labels"
OFFICIAL_ROBOT_PRIM_PATH = "/Root/Ref_Xform/Ref"
OFFICIAL_ROBOT_NAME = "walkerS2"
CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")
TABLE_PATH = "/Replicator/Ref_Xform"
PHYSICS_SCENE_PATH = "/World/PhysicsScene"
TABLE_UNIT_M = 0.035
DEFAULT_INIT_STEPS = 120
DEFAULT_CAMERA_WARMUP_STEPS = 5
DEFAULT_SAMPLE_STRIDE = 5
DEFAULT_PHYSICS_DT = 1.0 / 60.0
PHYSICS_READY_SETTLE_STEPS = 5
ROBOT_POSE_POSITION_WARN_M = 0.005
ROBOT_POSE_YAW_WARN_RAD = math.radians(0.5)
COARSE_ORIENTATION_BUCKETS = (
    "front",
    "front_left",
    "left",
    "back_left",
    "back",
    "back_right",
    "right",
    "front_right",
)


class RunFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _fail(reason: str, message: str) -> None:
    raise RunFailure(reason, message)


def _as_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path else default_path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def _log(log_path: Path, message: str, **fields: Any) -> None:
    payload = " ".join(f"{key}={json.dumps(value, default=_json_default, sort_keys=True)}" for key, value in fields.items())
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    if payload:
        line = f"{line} {payload}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_updates(sim_app: Any, steps: int, counter: dict[str, int]) -> None:
    for _ in range(max(0, int(steps))):
        sim_app.update()
        counter["step"] += 1


def _physics_sim_view_ready() -> bool:
    try:
        from isaacsim.core.simulation_manager import SimulationManager  # type: ignore

        return SimulationManager.get_physics_sim_view() is not None
    except Exception:
        return False


def _ensure_physics_ready(
    *,
    sim_app: Any,
    counter: dict[str, int],
    log_path: Path,
    physics_dt: float,
) -> Any:
    """Create Isaac's physics simulation view before RobotArticulation init."""
    ready_before = _physics_sim_view_ready()
    _log(
        log_path,
        "physics_ready_check",
        ready=ready_before,
        simulation_step=counter["step"],
        physics_scene_path=PHYSICS_SCENE_PATH,
    )
    if ready_before:
        return None

    try:
        from isaacsim.core.api.simulation_context import SimulationContext  # type: ignore

        simulation_context = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=float(physics_dt),
            rendering_dt=float(physics_dt),
            physics_prim_path=PHYSICS_SCENE_PATH,
        )
        simulation_context.initialize_physics()
        simulation_context.play()
        _run_updates(sim_app, PHYSICS_READY_SETTLE_STEPS, counter)
    except Exception as exc:
        _fail("physics_initialization_failed", f"Failed to initialize Isaac physics context: {exc}")

    ready_after = _physics_sim_view_ready()
    _log(
        log_path,
        "physics_ready",
        ready=ready_after,
        simulation_step=counter["step"],
        physics_scene_path=PHYSICS_SCENE_PATH,
    )
    if not ready_after:
        _fail(
            "physics_not_ready",
            "Isaac physics simulation view is still unavailable before RobotArticulation.initialize()",
        )
    return simulation_context


def _load_official_robot_articulation(baseline_root: Path) -> Any:
    ubtech_path = baseline_root / "Ubtech_sim"
    if not ubtech_path.exists():
        raise RuntimeError(f"Official Ubtech_sim directory missing: {ubtech_path}")
    if str(ubtech_path) not in sys.path:
        sys.path.insert(0, str(ubtech_path))
    from source.RobotArticulation import RobotArticulation  # type: ignore

    return RobotArticulation


def _array_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    try:
        return np.asarray(current)
    except Exception:
        return None


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _array_summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {"available": False}
    summary: dict[str, Any] = {
        "available": True,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }
    if array.size > 0 and np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        summary["finite_count"] = int(np.count_nonzero(finite))
        if np.any(finite):
            finite_values = array[finite]
            summary["finite_min"] = float(np.min(finite_values))
            summary["finite_max"] = float(np.max(finite_values))
    return summary


def _sample_camera_shape(summary: dict[str, Any]) -> tuple[int, int] | None:
    shape = summary.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return None
    height = int(shape[0])
    width = int(shape[1])
    if height <= 0 or width <= 0:
        return None
    return height, width


def _save_camera_arrays(
    *,
    run_dir: Path,
    sample_id: str,
    camera_data: dict[str, Any],
    save_depth: bool,
) -> tuple[dict[str, Any], list[str]]:
    rgb_dir = run_dir / "rgb" / sample_id
    depth_dir = run_dir / "depth" / sample_id
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    cameras: dict[str, Any] = {}
    warnings: list[str] = []
    for camera_name in CAMERA_NAMES:
        entry = camera_data.get(camera_name, {}) if isinstance(camera_data, dict) else {}
        rgb = _array_to_numpy(entry.get("rgb")) if isinstance(entry, dict) else None
        depth = _array_to_numpy(entry.get("depth")) if isinstance(entry, dict) else None

        rgb_path: Path | None = None
        depth_path: Path | None = None
        if rgb is not None:
            rgb_path = rgb_dir / f"{camera_name}.npy"
            np.save(rgb_path, rgb)
        else:
            warnings.append(f"{camera_name}:missing_rgb")

        if save_depth:
            if depth is not None:
                depth_path = depth_dir / f"{camera_name}.npy"
                np.save(depth_path, depth)
            else:
                warnings.append(f"{camera_name}:missing_depth")

        rgb_summary = _array_summary(rgb)
        depth_summary = _array_summary(depth if save_depth else None)
        shape = _sample_camera_shape(rgb_summary) or _sample_camera_shape(depth_summary)
        cameras[camera_name] = {
            "camera_name": camera_name,
            "source_interface": "RobotArticulation.get_cameras_images",
            "rgb": {
                **rgb_summary,
                "path": _relative_path(rgb_path, run_dir) if rgb_path else None,
            },
            "depth": {
                **depth_summary,
                "path": _relative_path(depth_path, run_dir) if depth_path else None,
            },
            "image_shape_hw": list(shape) if shape else None,
        }

    return cameras, warnings


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.array(fallback, dtype=float)
    return np.array(vector, dtype=float) / norm


def _bbox_corners_from_bbox(box: dict[str, list[float]]) -> np.ndarray:
    mins = np.array(box["min"], dtype=float)
    maxs = np.array(box["max"], dtype=float)
    return np.array(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=float,
    )


def _axis_aligned_with_world_xy(x_axis: np.ndarray, y_axis: np.ndarray) -> bool:
    world_axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    )
    return max(abs(float(np.dot(x_axis, axis))) for axis in world_axes) > 0.999 and max(
        abs(float(np.dot(y_axis, axis))) for axis in world_axes
    ) > 0.999


def _inspect_table_xform_axes(stage: Any, table_path: str) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    log: dict[str, Any] = {"table_path": table_path}
    try:
        prim = stage.GetPrimAtPath(table_path)
        if not prim or not prim.IsValid():
            log.update({"usable_horizontal_axes": False, "source": "missing_table_prim"})
            return log
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        axes = [
            _normalize(np.array([matrix[0][0], matrix[0][1], matrix[0][2]], dtype=float), np.array([1.0, 0.0, 0.0])),
            _normalize(np.array([matrix[1][0], matrix[1][1], matrix[1][2]], dtype=float), np.array([0.0, 1.0, 0.0])),
            _normalize(np.array([matrix[2][0], matrix[2][1], matrix[2][2]], dtype=float), np.array([0.0, 0.0, 1.0])),
        ]
        horizontal_axes = [axis for axis in axes if abs(float(np.dot(axis, np.array([0.0, 0.0, 1.0])))) < 0.25]
        log.update(
            {
                "usable_horizontal_axes": bool(horizontal_axes),
                "raw_axes_world": [axis.tolist() for axis in axes],
                "horizontal_axes_world": [axis.tolist() for axis in horizontal_axes],
                "source": "usd_xformable_local_to_world",
            }
        )
        return log
    except Exception as exc:
        log.update({"usable_horizontal_axes": False, "source": "xform_inspection_failed", "error": str(exc)})
        return log


def build_or_resolve_table_frame(
    *,
    stage: Any,
    table_path: str,
    table_bbox: dict[str, list[float]],
    robot_base_position: np.ndarray,
    robot_base_yaw_rad: float,
) -> dict[str, Any]:
    """Resolve the same robot-facing table convention used by Phase 1 planning."""
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
        "axis_aligned_with_world_xy": _axis_aligned_with_world_xy(x_axis, y_axis),
        "mapping_mode": "explicit_world_table_transform",
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


def _table_rotation_world(table_frame: dict[str, Any]) -> np.ndarray:
    return np.column_stack(
        [
            np.array(table_frame["x_axis_world"], dtype=float),
            np.array(table_frame["y_axis_world"], dtype=float),
            np.array(table_frame["z_axis_world"], dtype=float),
        ]
    )


def _quat_xyzw_normalized(quat_xyzw: Any) -> np.ndarray:
    q = np.array(quat_xyzw, dtype=float)
    if q.shape != (4,):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def _quat_xyzw_to_matrix(quat_xyzw: Any) -> np.ndarray:
    x, y, z, w = _quat_xyzw_normalized(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> list[float]:
    m = np.array(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _quat_xyzw_normalized([x, y, z, w]).tolist()


def _yaw_from_rotation_matrix(matrix: np.ndarray) -> float:
    return float(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def _wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _coarse_orientation(yaw_rad: float) -> str:
    wrapped = _wrap_pi(yaw_rad)
    index = int(round(wrapped / (math.pi / 4.0))) % len(COARSE_ORIENTATION_BUCKETS)
    return COARSE_ORIENTATION_BUCKETS[index]


def _gf_quat_to_xyzw(quat: Any) -> list[float]:
    imaginary = quat.GetImaginary()
    return [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(quat.GetReal())]


def _resolve_robot_frame(stage: Any, robot_prim_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(robot_prim_path)
    if not prim or not prim.IsValid():
        return {
            "available": False,
            "frame_name": "robot_root_usd",
            "source": "missing_robot_prim",
            "robot_prim_path": robot_prim_path,
        }
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transform = cache.GetLocalToWorldTransform(prim)
    translation = transform.ExtractTranslation()
    quat_xyzw = _gf_quat_to_xyzw(transform.ExtractRotationQuat())
    rotation_world = _quat_xyzw_to_matrix(quat_xyzw)
    yaw_rad = _wrap_pi(_yaw_from_rotation_matrix(rotation_world))
    return {
        "available": True,
        "frame_name": "robot_root_usd",
        "source": "usd_xform_cache_local_to_world",
        "semantics": "USD robot root frame for dataset labels; not a replacement for Pinocchio CoordinateTransform",
        "robot_prim_path": robot_prim_path,
        "position_world": [float(translation[0]), float(translation[1]), float(translation[2])],
        "orientation_xyzw_world": quat_xyzw,
        "rotation_world": rotation_world.tolist(),
        "yaw_rad": yaw_rad,
        "yaw_deg": math.degrees(yaw_rad),
    }


def _select_table_frame_robot_pose(
    *,
    configured_robot_position: np.ndarray,
    configured_robot_yaw_rad: float,
    robot_frame: dict[str, Any],
) -> dict[str, Any]:
    config_pose = {
        "source": "Part_Sorting.yaml:robot",
        "position_world": [float(value) for value in configured_robot_position.tolist()],
        "yaw_rad": float(configured_robot_yaw_rad),
        "yaw_deg": math.degrees(float(configured_robot_yaw_rad)),
    }
    if not robot_frame.get("available"):
        return {
            "selected_source": "config_robot_pose_fallback",
            "selected_pose": config_pose,
            "config_pose": config_pose,
            "usd_pose": None,
            "comparison": {
                "available": False,
                "reason": robot_frame.get("source", "usd_robot_frame_unavailable"),
                "warning_thresholds": {
                    "position_m": ROBOT_POSE_POSITION_WARN_M,
                    "yaw_rad": ROBOT_POSE_YAW_WARN_RAD,
                    "yaw_deg": math.degrees(ROBOT_POSE_YAW_WARN_RAD),
                },
            },
            "warnings": ["table_frame_robot_pose_source=config_fallback_usd_robot_frame_unavailable"],
        }

    usd_position = np.array(robot_frame.get("position_world", [0.0, 0.0, 0.0]), dtype=float)
    usd_yaw_rad = float(robot_frame.get("yaw_rad", 0.0))
    usd_pose = {
        "source": "usd_stage_robot_frame",
        "position_world": [float(value) for value in usd_position.tolist()],
        "yaw_rad": usd_yaw_rad,
        "yaw_deg": math.degrees(usd_yaw_rad),
        "robot_prim_path": robot_frame.get("robot_prim_path"),
    }
    position_delta = usd_position - configured_robot_position
    position_delta_norm = float(np.linalg.norm(position_delta))
    yaw_delta_rad = _wrap_pi(usd_yaw_rad - float(configured_robot_yaw_rad))
    differs = position_delta_norm > ROBOT_POSE_POSITION_WARN_M or abs(yaw_delta_rad) > ROBOT_POSE_YAW_WARN_RAD
    warnings: list[str] = []
    if differs:
        warnings.append("config_robot_pose_differs_from_usd_stage_pose")

    return {
        "selected_source": "usd_stage_robot_frame",
        "selected_pose": usd_pose,
        "config_pose": config_pose,
        "usd_pose": usd_pose,
        "comparison": {
            "available": True,
            "position_delta_world_m": [float(value) for value in position_delta.tolist()],
            "position_delta_norm_m": position_delta_norm,
            "yaw_delta_rad": yaw_delta_rad,
            "yaw_delta_deg": math.degrees(yaw_delta_rad),
            "exceeds_warning_threshold": differs,
            "warning_thresholds": {
                "position_m": ROBOT_POSE_POSITION_WARN_M,
                "yaw_rad": ROBOT_POSE_YAW_WARN_RAD,
                "yaw_deg": math.degrees(ROBOT_POSE_YAW_WARN_RAD),
            },
        },
        "warnings": warnings,
    }


def _pose_in_frame(
    *,
    position_world: list[float],
    orientation_xyzw_world: list[float],
    frame_position_world: list[float],
    frame_rotation_world: np.ndarray,
) -> dict[str, Any]:
    world_position = np.array(position_world, dtype=float)
    frame_position = np.array(frame_position_world, dtype=float)
    object_rotation_world = _quat_xyzw_to_matrix(orientation_xyzw_world)
    rotation_in_frame = frame_rotation_world.T @ object_rotation_world
    position_in_frame = frame_rotation_world.T @ (world_position - frame_position)
    yaw = _yaw_from_rotation_matrix(rotation_in_frame)
    return {
        "position_xyz_m": position_in_frame.tolist(),
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "orientation_xyzw": _matrix_to_quat_xyzw(rotation_in_frame),
        "coarse_orientation": _coarse_orientation(yaw),
    }


def _class_for_part(stage: Any, prim_path: str, index: int, num_parts_per_class: int) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    refs = _reference_paths(prim) if prim and prim.IsValid() else []
    category = _category_from_reference(refs)
    if category == "part_a":
        return {"class": "A", "raw_class": category, "source": "usd_reference_path", "reference_paths": refs}
    if category == "part_b":
        return {"class": "B", "raw_class": category, "source": "usd_reference_path", "reference_paths": refs}
    fallback = "part_a" if index < num_parts_per_class else "part_b"
    return {
        "class": "A" if fallback == "part_a" else "B",
        "raw_class": fallback,
        "source": "scene_builder_creation_order_fallback",
        "reference_paths": refs,
    }


def _target_bin_payload(cfg: dict[str, Any], class_name: str) -> dict[str, Any]:
    box_cfg = cfg.get("box", {})
    positions = box_cfg.get("box_position", []) or []
    scales = box_cfg.get("box_scale", []) or []
    return {
        "semantic_target": "task1_sorting_bin" if positions else None,
        "target_bin_id": "box_0" if positions else None,
        "class_specific_mapping": None,
        "class_specific_mapping_status": "not_available_in_current_task_config",
        "object_class": class_name,
        "source": "Part_Sorting.yaml:box" if positions else "unavailable",
        "configured_box_position_world": [float(value) for value in positions[0]] if positions else None,
        "configured_box_scale": [float(value) for value in scales[0]] if scales else None,
    }


def _project_bbox_debug(camera: Any, bbox_world: dict[str, Any], image_shape_hw: tuple[int, int]) -> dict[str, Any]:
    if not isinstance(bbox_world, dict) or bbox_world.get("min") is None or bbox_world.get("max") is None:
        return {"available": False, "reason": "bbox_world_unavailable"}
    try:
        corners_world = _bbox_corners_from_bbox(bbox_world)
        uv = camera.get_image_coords_from_world_points(corners_world.astype(np.float32))
        uv_array = _array_to_numpy(uv)
        if uv_array is None or uv_array.size < 2:
            raise RuntimeError("bbox projection returned no image coordinates")
        points = np.reshape(uv_array, (-1, 2))
        finite_mask = np.all(np.isfinite(points), axis=1)
        finite_points = points[finite_mask]
        if finite_points.size == 0:
            return {
                "available": False,
                "reason": "no_finite_projected_bbox_corners",
                "finite_corner_count": 0,
                "corner_count": int(points.shape[0]),
            }
        height, width = image_shape_hw
        x1 = float(np.min(finite_points[:, 0]))
        y1 = float(np.min(finite_points[:, 1]))
        x2 = float(np.max(finite_points[:, 0]))
        y2 = float(np.max(finite_points[:, 1]))
        in_bounds = (
            finite_mask
            & (points[:, 0] >= 0.0)
            & (points[:, 0] < float(width))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < float(height))
        )
        return {
            "available": True,
            "mode": "projected_3d_bbox_debug_only",
            "roi_2d_px": [x1, y1, x2, y2],
            "center_2d_px": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
            "extent_2d_px": [max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "finite_corner_count": int(np.count_nonzero(finite_mask)),
            "corner_count": int(points.shape[0]),
            "in_bounds_corner_fraction": float(np.count_nonzero(in_bounds)) / float(points.shape[0]),
            "intersects_image_bounds": bool(x2 >= 0.0 and y2 >= 0.0 and x1 < float(width) and y1 < float(height)),
        }
    except Exception as exc:
        return {"available": False, "reason": "bbox_projection_failed", "error": str(exc)}


def _estimate_projection_visibility(
    *,
    robot: Any,
    object_center_world: list[float],
    bbox_world: dict[str, Any],
    camera_summaries: dict[str, Any],
) -> dict[str, Any]:
    per_camera: dict[str, Any] = {}
    known_values: list[bool] = []
    for camera_name in CAMERA_NAMES:
        camera = getattr(robot, "cameras", {}).get(camera_name) if hasattr(robot, "cameras") else None
        shape = _sample_camera_shape(camera_summaries.get(camera_name, {}).get("rgb", {})) or _sample_camera_shape(
            camera_summaries.get(camera_name, {}).get("depth", {})
        )
        if camera is None or shape is None or not hasattr(camera, "get_image_coords_from_world_points"):
            per_camera[camera_name] = {
                "visible_projection": None,
                "visibility_mode": "center_projection_only",
                "source": "projection_method_or_image_shape_unavailable",
                "segmentation_used": False,
                "occlusion_reasoning_used": False,
                "depth_roi_finite_ratio": None,
                "depth_roi_check_status": "not_computed_in_phase1",
                "bbox_projection": {"available": False, "reason": "projection_method_or_image_shape_unavailable"},
            }
            continue
        try:
            uv = camera.get_image_coords_from_world_points(np.array([object_center_world], dtype=np.float32))
            uv_array = _array_to_numpy(uv)
            if uv_array is None or uv_array.size < 2:
                raise RuntimeError("projection returned no image coordinates")
            u = float(np.reshape(uv_array, (-1, 2))[0][0])
            v = float(np.reshape(uv_array, (-1, 2))[0][1])
            height, width = shape
            visible = math.isfinite(u) and math.isfinite(v) and 0.0 <= u < float(width) and 0.0 <= v < float(height)
            known_values.append(bool(visible))
            per_camera[camera_name] = {
                "visible_projection": bool(visible),
                "visibility_mode": "center_projection_only",
                "uv_px": [u, v],
                "image_shape_hw": [height, width],
                "source": "camera_center_projection_only_no_occlusion_test",
                "segmentation_used": False,
                "occlusion_reasoning_used": False,
                "depth_roi_finite_ratio": None,
                "depth_roi_check_status": "not_computed_in_phase1",
                "bbox_projection": _project_bbox_debug(camera, bbox_world, shape),
            }
        except Exception as exc:
            per_camera[camera_name] = {
                "visible_projection": None,
                "visibility_mode": "center_projection_only",
                "source": "projection_failed",
                "error": str(exc),
                "segmentation_used": False,
                "occlusion_reasoning_used": False,
                "depth_roi_finite_ratio": None,
                "depth_roi_check_status": "not_computed_in_phase1",
                "bbox_projection": {"available": False, "reason": "center_projection_failed"},
            }
    return {
        "visible_any_camera": any(known_values) if known_values else None,
        "visibility_mode": "center_projection_only",
        "visibility_truth": False,
        "segmentation_used": False,
        "occlusion_reasoning_used": False,
        "depth_roi_check_status": "not_computed_in_phase1",
        "visibility_semantics": "Weak best-effort center projection only. This is not segmentation, not occlusion truth, and not proof of usable depth.",
        "bbox_projection_semantics": "Projected 3D bbox fields are debug hints only and do not make visible_projection occlusion-aware.",
        "per_camera": per_camera,
    }


def _pose_lookup_from_scene(scene: Any) -> dict[str, dict[str, Any]]:
    try:
        poses = scene.get_parts_world_poses()
    except Exception:
        poses = []
    return {str(entry.get("prim_path")): entry for entry in poses if entry.get("prim_path")}


def _labels_for_sample(
    *,
    stage: Any,
    scene: Any,
    cfg: dict[str, Any],
    table_frame: dict[str, Any] | None,
    robot_frame: dict[str, Any],
    robot: Any,
    camera_summaries: dict[str, Any],
) -> dict[str, Any]:
    part_paths = list(getattr(scene, "parts_prim_paths", []))
    pose_lookup = _pose_lookup_from_scene(scene)
    num_parts_per_class = int(cfg.get("part", {}).get("num_parts", 2))
    robot_rotation_world = np.array(robot_frame.get("rotation_world", np.eye(3)), dtype=float)
    robot_position_world = robot_frame.get("position_world", [0.0, 0.0, 0.0])
    table_rotation_world = _table_rotation_world(table_frame) if table_frame else None

    objects: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, prim_path in enumerate(part_paths):
        pose = pose_lookup.get(prim_path)
        if not pose:
            warnings.append(f"{prim_path}:missing_scene_pose")
            continue
        position_world = [float(value) for value in pose.get("position", [0.0, 0.0, 0.0])]
        orientation_world = _quat_xyzw_normalized(pose.get("orientation", [0.0, 0.0, 0.0, 1.0])).tolist()
        world_rotation = _quat_xyzw_to_matrix(orientation_world)
        world_yaw = _yaw_from_rotation_matrix(world_rotation)
        try:
            bbox = _bbox(stage, str(prim_path))
        except Exception as exc:
            bbox = {"error": str(exc), "min": None, "max": None, "center": position_world, "size": None}
            warnings.append(f"{prim_path}:bbox_failed:{exc}")

        class_payload = _class_for_part(stage, str(prim_path), index, num_parts_per_class)
        class_name = str(class_payload["class"])
        label: dict[str, Any] = {
            "object_id": f"task1_part_{index:03d}",
            "prim_path": str(prim_path),
            "class": class_name,
            "raw_class": class_payload["raw_class"],
            "class_source": class_payload["source"],
            "reference_paths": class_payload["reference_paths"],
            "world_pose": {
                "position_xyz_m": position_world,
                "orientation_xyzw": orientation_world,
                "yaw_rad": world_yaw,
                "yaw_deg": math.degrees(world_yaw),
                "coarse_orientation": _coarse_orientation(world_yaw),
            },
            "base_frame_pose": None,
            "table_frame_pose": None,
            "bbox_world": bbox,
            "target_bin": _target_bin_payload(cfg, class_name),
            "visibility": _estimate_projection_visibility(
                robot=robot,
                object_center_world=position_world,
                bbox_world=bbox,
                camera_summaries=camera_summaries,
            ),
        }
        if robot_frame.get("available"):
            label["base_frame_pose"] = {
                **_pose_in_frame(
                    position_world=position_world,
                    orientation_xyzw_world=orientation_world,
                    frame_position_world=robot_position_world,
                    frame_rotation_world=robot_rotation_world,
                ),
                "frame_name": robot_frame["frame_name"],
                "frame_source": robot_frame["source"],
            }
        if table_frame and table_rotation_world is not None:
            table_position = world_to_table(position_world, table_frame)
            table_rotation = table_rotation_world.T @ world_rotation
            table_yaw = _yaw_from_rotation_matrix(table_rotation)
            label["table_frame_pose"] = {
                "position_xyz_m": table_position.tolist(),
                "x": float(table_position[0]),
                "y": float(table_position[1]),
                "z": float(table_position[2]),
                "yaw_rad": table_yaw,
                "yaw_deg": math.degrees(table_yaw),
                "orientation_xyzw": _matrix_to_quat_xyzw(table_rotation),
                "coarse_orientation": _coarse_orientation(table_yaw),
                "frame_name": table_frame["frame_name"],
                "frame_source": table_frame["source"],
            }
        objects.append(label)

    return {
        "schema_name": "task1_rgbd_truth_labels",
        "schema_version": "0.2.0",
        "object_count": len(objects),
        "objects": objects,
        "warnings": warnings,
        "label_policy": {
            "object_set": "all_spawned_task1_parts",
            "visibility_filter_applied": False,
            "visibility_note": "Per-object visibility is weak center-projection metadata only; no segmentation, depth ROI validation, or true occlusion filtering is applied in Phase 1.",
        },
    }


def _parse_json_arg(raw: str | None, name: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON: {exc}") from exc


def _prepare_run_dirs(output_root: Path, repo_root: Path, run_id: str) -> Path:
    output_root = output_root.expanduser().resolve()
    if _is_relative_to(output_root, repo_root):
        raise RuntimeError(
            f"Output root must stay outside the code repo. Got {output_root}; use $OUTPUT_ROOT or another runtime directory."
        )
    run_dir = output_root / run_id
    for relative in ("rgb", "depth", "labels", "metadata", "sync_debug"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", help="Official baseline repo root. Defaults to $HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline.")
    parser.add_argument("--asset-root", help="Official assets resources root. Defaults to $HRC_ROOT/assets/challenge2026_assets/resources.")
    parser.add_argument("--config", help="Task 1 config path. Defaults to Ubtech_sim/config/Part_Sorting.yaml under baseline root.")
    parser.add_argument("--output-root", help="Dataset output root. Defaults to $OUTPUT_ROOT/datasets/task1_rgbd_labels.")
    parser.add_argument("--run-id", help="Run directory name. Defaults to UTC timestamp plus seed.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to collect.")
    parser.add_argument("--sample-stride", type=int, default=DEFAULT_SAMPLE_STRIDE, help="Simulation updates between samples.")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS, help="Initial scene settle/update steps before camera setup.")
    parser.add_argument("--camera-warmup-steps", type=int, default=DEFAULT_CAMERA_WARMUP_STEPS, help="Updates after RobotArticulation camera initialization.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--physics-dt", type=float, default=DEFAULT_PHYSICS_DT, help="Used only for sim_time_estimate_s metadata.")
    parser.add_argument("--robot-prim-path", default=OFFICIAL_ROBOT_PRIM_PATH)
    parser.add_argument("--gui", action="store_true", help="Open Isaac GUI instead of headless mode.")
    parser.add_argument("--no-headless", action="store_true", help="Alias for --gui.")
    parser.add_argument("--hold-open", action="store_true", help="Keep the GUI open after collection.")
    parser.add_argument("--save-depth", dest="save_depth", action="store_true", help="Save depth .npy arrays.")
    parser.add_argument("--no-save-depth", dest="save_depth", action="store_false", help="Do not save depth arrays. Intended only for quick storage/debug runs.")
    parser.set_defaults(save_depth=True)
    parser.add_argument("--chosen-object-id", "--selected-object-id", dest="chosen_object_id", default=None)
    parser.add_argument("--chosen-arm", "--selected-arm", dest="chosen_arm", choices=("left", "right", "none"), default="none")
    parser.add_argument("--chosen-preset", "--selected-preset", dest="chosen_preset", default=None)
    parser.add_argument("--chosen-candidate-json", default=None, help="Optional JSON payload copied into runtime metadata.")
    parser.add_argument("--planner-target-json", default=None, help="Optional JSON payload copied into runtime metadata.")
    parser.add_argument("--execution-result", default="collection_only_no_manipulation")
    parser.add_argument("--fail-reason", default=None)
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.samples < 1:
        raise RuntimeError("--samples must be at least 1")
    if args.sample_stride < 0:
        raise RuntimeError("--sample-stride must be non-negative")
    if args.init_steps < 0 or args.camera_warmup_steps < 0:
        raise RuntimeError("--init-steps and --camera-warmup-steps must be non-negative")
    if args.physics_dt <= 0.0:
        raise RuntimeError("--physics-dt must be positive")
    if not str(args.robot_prim_path).startswith("/"):
        raise RuntimeError("--robot-prim-path must be an absolute USD prim path")

    planner_target = _parse_json_arg(args.planner_target_json, "--planner-target-json")
    chosen_candidate = _parse_json_arg(args.chosen_candidate_json, "--chosen-candidate-json")
    timestamp = _timestamp_compact()
    run_id = args.run_id or f"{timestamp}_seed{int(args.seed)}"

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    repo_root = paths["HRC_REPO"]
    baseline_root = _as_path(args.baseline_root or os.environ.get("HRC_BASELINE_REPO"), paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
    asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
    config_path = _as_path(args.config, baseline_root / DEFAULT_CONFIG_RELATIVE)
    output_root = _as_path(args.output_root, paths["OUTPUT_ROOT"] / OUTPUT_DATASET_RELATIVE)
    run_dir = _prepare_run_dirs(output_root, repo_root, run_id)
    log_path = paths["LOG_ROOT"] / f"{LOG_STEM}_{run_id}.log"
    manifest_path = run_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    if not baseline_root.exists():
        _fail("environment_invalid", f"Official baseline root missing: {baseline_root}")
    if not asset_root.exists():
        _fail("environment_invalid", f"Asset root missing: {asset_root}")
    if not config_path.exists() or config_path.name != "Part_Sorting.yaml":
        _fail("environment_invalid", f"Task 1 Part_Sorting.yaml config missing or wrong: {config_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    sim_app = None
    robot = None
    simulation_context = None
    counter = {"step": 0}
    try:
        _log(
            log_path,
            "collector_start",
            run_id=run_id,
            samples=args.samples,
            gui=bool(args.gui or args.no_headless),
            save_depth=bool(args.save_depth),
        )
        SimulationApp = _load_simulation_app()
        sim_app = SimulationApp({"headless": not (args.gui or args.no_headless)})

        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        original_root_path = cfg.get("root_path")
        cfg["root_path"] = str(asset_root)
        apply_scatter_config(cfg)

        import omni.replicator.core as rep  # type: ignore
        from pxr import UsdGeom  # type: ignore

        if hasattr(rep, "set_global_seed"):
            rep.set_global_seed(args.seed)

        stage = _create_minimal_scene()
        UsdGeom.Xform.Define(stage, "/Root")
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        scene.build_table()
        scene.build_parts()
        scene.build_robot()
        part_paths = list(getattr(scene, "parts_prim_paths", []))
        _log(
            log_path,
            "scene_built",
            part_count=len(part_paths),
            robot_prim_path=args.robot_prim_path,
            simulation_step=counter["step"],
        )
        rep.orchestrator.step()
        _run_updates(sim_app, args.init_steps, counter)
        simulation_context = _ensure_physics_ready(
            sim_app=sim_app,
            counter=counter,
            log_path=log_path,
            physics_dt=float(args.physics_dt),
        )

        RobotArticulation = _load_official_robot_articulation(baseline_root)
        robot = RobotArticulation(prim_path=args.robot_prim_path, name=OFFICIAL_ROBOT_NAME)
        _log(
            log_path,
            "robot_initialize_start",
            robot_prim_path=args.robot_prim_path,
            simulation_step=counter["step"],
            physics_ready=_physics_sim_view_ready(),
        )
        robot.initialize()
        _log(
            log_path,
            "robot_initialize_success",
            robot_prim_path=args.robot_prim_path,
            simulation_step=counter["step"],
            camera_names=sorted(list(getattr(robot, "cameras", {}).keys())),
        )
        _run_updates(sim_app, args.camera_warmup_steps, counter)

        if not part_paths:
            _fail("no_task1_parts", "SceneBuilder did not expose any Task 1 part prim paths")

        table_bbox = _bbox(stage, TABLE_PATH)
        robot_cfg = cfg.get("robot", {})
        configured_robot_position = np.array(robot_cfg.get("robot_position", [0.0, 0.0, 0.0]), dtype=float)
        configured_robot_rotation = robot_cfg.get("robot_rotation", [0.0, 0.0, 0.0])
        configured_robot_yaw_rad = math.radians(float(configured_robot_rotation[2]) if len(configured_robot_rotation) >= 3 else 0.0)
        robot_frame = _resolve_robot_frame(stage, args.robot_prim_path)
        table_robot_pose = _select_table_frame_robot_pose(
            configured_robot_position=configured_robot_position,
            configured_robot_yaw_rad=configured_robot_yaw_rad,
            robot_frame=robot_frame,
        )
        selected_table_pose = table_robot_pose["selected_pose"]
        table_frame = build_or_resolve_table_frame(
            stage=stage,
            table_path=TABLE_PATH,
            table_bbox=table_bbox,
            robot_base_position=np.array(selected_table_pose["position_world"], dtype=float),
            robot_base_yaw_rad=float(selected_table_pose["yaw_rad"]),
        )
        table_frame["robot_pose_input"] = selected_table_pose
        table_frame["robot_pose_source_comparison"] = table_robot_pose["comparison"]
        if table_robot_pose["warnings"]:
            _log(
                log_path,
                "robot_pose_source_warning",
                warnings=table_robot_pose["warnings"],
                comparison=table_robot_pose["comparison"],
            )
        run_metadata = {
            "schema_name": "task1_rgbd_collection_run",
            "schema_version": "0.2.0",
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": SCRIPT_NAME,
            "phase_scope": "Phase 1 synchronized RGB-D and simulator truth data collection",
            "implemented_scope": [
                "official SceneBuilder Task 1 scene setup",
                "official RobotArticulation camera RGB-D interface",
                "JSONL manifest",
                "per-sample RGB/depth arrays",
                "simulator truth labels",
                "runtime metadata",
                "sync debug logs",
            ],
            "not_implemented_scope": [
                "camera-first manipulation runtime",
                "automatic evaluator execution",
                "Thinker runtime integration",
                "final grasp-pose generation from Thinker output",
            ],
            "warnings": table_robot_pose["warnings"],
            "paths": {
                "repo_root": str(repo_root),
                "baseline_root": str(baseline_root),
                "asset_root": str(asset_root),
                "config_path": str(config_path),
                "output_root": str(output_root),
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "log_path": str(log_path),
            },
            "config": {
                "original_root_path": original_root_path,
                "effective_root_path": cfg.get("root_path"),
                "seed": int(args.seed),
                "samples": int(args.samples),
                "sample_stride": int(args.sample_stride),
                "init_steps": int(args.init_steps),
                "camera_warmup_steps": int(args.camera_warmup_steps),
                "physics_dt_for_metadata": float(args.physics_dt),
                "save_depth": bool(args.save_depth),
            },
            "scene": {
                "part_paths": part_paths,
                "table_path": TABLE_PATH,
                "table_bbox_world": table_bbox,
                "table_frame": table_frame,
                "robot_frame": robot_frame,
                "robot_pose_sources": table_robot_pose,
            },
            "camera_interface": {
                "provider": "source.RobotArticulation.RobotArticulation",
                "method": "get_cameras_images(step)",
                "camera_names": list(CAMERA_NAMES),
                "camera_prim_paths_are_defined_by_official_robot_articulation": True,
            },
        }
        _write_json(run_dir / "run_metadata.json", run_metadata)
        _log(log_path, "scene_ready", part_count=len(part_paths), simulation_step=counter["step"])

        for sample_index in range(int(args.samples)):
            _run_updates(sim_app, args.sample_stride, counter)
            sample_id = f"sample_{sample_index:06d}"
            capture_step = int(counter["step"])
            capture_start = time.perf_counter()
            camera_data = robot.get_cameras_images(capture_step)
            capture_elapsed = time.perf_counter() - capture_start
            camera_summaries, camera_warnings = _save_camera_arrays(
                run_dir=run_dir,
                sample_id=sample_id,
                camera_data=camera_data,
                save_depth=bool(args.save_depth),
            )
            labels = _labels_for_sample(
                stage=stage,
                scene=scene,
                cfg=cfg,
                table_frame=table_frame,
                robot_frame=robot_frame,
                robot=robot,
                camera_summaries=camera_summaries,
            )
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            runtime_metadata = {
                "schema_name": "task1_rgbd_collection_sample_metadata",
                "schema_version": "0.2.0",
                "sample_id": sample_id,
                "sample_index": int(sample_index),
                "timestamp_utc": timestamp_utc,
                "simulation_step": capture_step,
                "sim_time_estimate_s": float(capture_step) * float(args.physics_dt),
                "chosen_object_id": args.chosen_object_id,
                "chosen_arm": None if args.chosen_arm == "none" else args.chosen_arm,
                "chosen_preset": args.chosen_preset,
                "chosen_candidate": chosen_candidate,
                "selected_object_id": args.chosen_object_id,
                "selected_arm": None if args.chosen_arm == "none" else args.chosen_arm,
                "selected_preset": args.chosen_preset,
                "selected_candidate": chosen_candidate,
                "planner_target": planner_target,
                "execution_result": args.execution_result,
                "fail_reason": args.fail_reason,
                "collector_only": True,
            }
            sync_debug = {
                "schema_name": "task1_rgbd_collection_sync_debug",
                "schema_version": "0.2.0",
                "sample_id": sample_id,
                "capture_step": capture_step,
                "capture_step_after": int(counter["step"]),
                "same_simulation_step_for_all_cameras": True,
                "camera_capture_order": list(CAMERA_NAMES),
                "camera_capture_elapsed_s": float(capture_elapsed),
                "camera_summaries": camera_summaries,
                "warnings": camera_warnings + labels.get("warnings", []),
                "sync_note": "RGB and depth are requested through RobotArticulation.get_cameras_images(step) without advancing sim_app.update between per-camera reads.",
            }

            labels_path = run_dir / "labels" / f"{sample_id}.json"
            metadata_path = run_dir / "metadata" / f"{sample_id}.json"
            sync_debug_path = run_dir / "sync_debug" / f"{sample_id}.json"
            _write_json(labels_path, labels)
            _write_json(metadata_path, runtime_metadata)
            _write_json(sync_debug_path, sync_debug)

            manifest_entry = {
                "schema_name": "task1_rgbd_collection_manifest_entry",
                "schema_version": "0.2.0",
                "run_id": run_id,
                "seed": int(args.seed),
                "sample_id": sample_id,
                "sample_index": int(sample_index),
                "simulation_step": capture_step,
                "timestamp_utc": timestamp_utc,
                "paths": {
                    "labels": _relative_path(labels_path, run_dir),
                    "metadata": _relative_path(metadata_path, run_dir),
                    "sync_debug": _relative_path(sync_debug_path, run_dir),
                },
                "cameras": camera_summaries,
                "object_count": int(labels["object_count"]),
                "chosen_object_id": args.chosen_object_id,
                "chosen_arm": None if args.chosen_arm == "none" else args.chosen_arm,
                "chosen_preset": args.chosen_preset,
                "execution_result": args.execution_result,
                "fail_reason": args.fail_reason,
            }
            _append_jsonl(manifest_path, manifest_entry)
            _log(
                log_path,
                "sample_saved",
                sample_id=sample_id,
                simulation_step=capture_step,
                object_count=labels["object_count"],
                warnings=sync_debug["warnings"],
            )

        _log(log_path, "collector_complete", run_dir=str(run_dir), manifest=str(manifest_path))
        if args.hold_open and (args.gui or args.no_headless):
            _log(log_path, "hold_open_start", note="Close Isaac Sim window to exit.")
            while sim_app is not None and sim_app.is_running():
                sim_app.update()
        return 0
    except RunFailure as exc:
        _log(log_path, "collector_failed", failure_reason=exc.reason, error=str(exc))
        raise
    finally:
        if robot is not None and hasattr(robot, "cleanup"):
            try:
                robot.cleanup()
            except Exception:
                pass
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{SCRIPT_NAME} FAILED: {exc}", file=sys.stderr)
        raise
