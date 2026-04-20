#!/usr/bin/env python3
"""Collect a minimal Task 1 RGB dataset with table-frame object labels.

This Phase 1 collector is deliberately narrow: it builds the official Task 1
scene, reuses RobotArticulation.get_cameras_images(step), and exports training
samples for the future camera -> Thinker -> planner path. It does not run or
modify manipulation, IK, grasp planning, or control code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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
DEFAULT_INIT_STEPS = 120
DEFAULT_CAMERA_WARMUP_STEPS = 5
DEFAULT_SAMPLE_STRIDE = 5
DEFAULT_XY_NOISE_SIGMA_M = 0.005
DEFAULT_YAW_NOISE_SIGMA_DEG = 3.0


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
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_updates(sim_app: Any, steps: int, counter: dict[str, int]) -> None:
    for _ in range(max(0, int(steps))):
        sim_app.update()
        counter["step"] += 1


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


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"Expected RGB/RGBA array with shape HxWxC, got {array.shape}")
    array = array[:, :, :3]
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    array = array.astype(np.float32, copy=False)
    finite = np.isfinite(array)
    if np.any(finite) and float(np.nanmax(array[finite])) <= 1.0:
        array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


def _save_rgb_frame(path_stem: Path, rgb: np.ndarray, rgb_format: str) -> tuple[Path, str | None]:
    if rgb_format == "npy":
        path = path_stem.with_suffix(".npy")
        np.save(path, rgb)
        return path, None

    try:
        from PIL import Image  # type: ignore

        path = path_stem.with_suffix(".png")
        Image.fromarray(_to_uint8_rgb(rgb)).save(path)
        return path, None
    except Exception as exc:
        path = path_stem.with_suffix(".npy")
        np.save(path, rgb)
        return path, f"{path_stem.name}:png_fallback_to_npy:{exc}"


def _save_camera_sample(
    *,
    run_dir: Path,
    sample_id: str,
    camera_data: dict[str, Any],
    rgb_format: str,
    save_depth: bool,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    rgb_dir = run_dir / "rgb" / sample_id
    depth_dir = run_dir / "depth" / sample_id
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths: dict[str, str] = {}
    depth_paths: dict[str, str] = {}
    warnings: list[str] = []
    for camera_name in CAMERA_NAMES:
        entry = camera_data.get(camera_name, {}) if isinstance(camera_data, dict) else {}
        rgb = _array_to_numpy(entry.get("rgb")) if isinstance(entry, dict) else None
        depth = _array_to_numpy(entry.get("depth")) if isinstance(entry, dict) else None

        if rgb is not None:
            rgb_path, warning = _save_rgb_frame(rgb_dir / camera_name, rgb, rgb_format)
            rgb_paths[camera_name] = _relative_path(rgb_path, run_dir)
            if warning:
                warnings.append(warning)
        else:
            warnings.append(f"{camera_name}:missing_rgb")

        if save_depth:
            if depth is not None:
                depth_path = depth_dir / f"{camera_name}.npy"
                np.save(depth_path, depth)
                depth_paths[camera_name] = _relative_path(depth_path, run_dir)
            else:
                warnings.append(f"{camera_name}:missing_depth")

    return rgb_paths, depth_paths, warnings


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


def _inspect_table_xform_axes(stage: Any, table_path: str) -> dict[str, Any]:
    from pxr import UsdGeom  # type: ignore

    try:
        prim = stage.GetPrimAtPath(table_path)
        if not prim or not prim.IsValid():
            return {"usable_horizontal_axes": False}
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        axes = [
            _normalize(np.array([matrix[0][0], matrix[0][1], matrix[0][2]], dtype=float), np.array([1.0, 0.0, 0.0])),
            _normalize(np.array([matrix[1][0], matrix[1][1], matrix[1][2]], dtype=float), np.array([0.0, 1.0, 0.0])),
            _normalize(np.array([matrix[2][0], matrix[2][1], matrix[2][2]], dtype=float), np.array([0.0, 0.0, 1.0])),
        ]
        horizontal_axes = [axis for axis in axes if abs(float(np.dot(axis, np.array([0.0, 0.0, 1.0])))) < 0.25]
        return {
            "usable_horizontal_axes": bool(horizontal_axes),
            "horizontal_axes_world": [axis.tolist() for axis in horizontal_axes],
        }
    except Exception:
        return {"usable_horizontal_axes": False}


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
    if xform_log.get("usable_horizontal_axes"):
        horizontal_axes = [np.array(axis, dtype=float) for axis in xform_log.get("horizontal_axes_world", [])]
        best = max(horizontal_axes, key=lambda axis: abs(float(np.dot(axis, robot_forward))))
        if abs(float(np.dot(best, robot_forward))) > 0.25:
            y_seed = best if float(np.dot(best, robot_forward)) >= 0.0 else -best

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
    origin_world = x_axis * float(np.min(x_proj)) + y_axis * float(np.min(y_proj)) + world_up * float(table_bbox["max"][2])
    return {
        "origin_world": origin_world.tolist(),
        "x_axis_world": x_axis.tolist(),
        "y_axis_world": y_axis.tolist(),
        "z_axis_world": world_up.tolist(),
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


def _yaw_from_rotation_matrix(matrix: np.ndarray) -> float:
    return float(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))


def _wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _class_for_part(stage: Any, prim_path: str, index: int, num_parts_per_class: int) -> str:
    prim = stage.GetPrimAtPath(prim_path)
    refs = _reference_paths(prim) if prim and prim.IsValid() else []
    category = _category_from_reference(refs)
    if category == "part_a":
        return "A"
    if category == "part_b":
        return "B"
    return "A" if index < num_parts_per_class else "B"


def _pose_lookup_from_scene(scene: Any) -> dict[str, dict[str, Any]]:
    try:
        poses = scene.get_parts_world_poses()
    except Exception:
        poses = []
    return {str(entry.get("prim_path")): entry for entry in poses if entry.get("prim_path")}


def _raw_table_objects(stage: Any, scene: Any, cfg: dict[str, Any], table_frame: dict[str, Any]) -> list[dict[str, float | str]]:
    part_paths = list(getattr(scene, "parts_prim_paths", []))
    pose_lookup = _pose_lookup_from_scene(scene)
    table_rotation = _table_rotation_world(table_frame)
    num_parts_per_class = int(cfg.get("part", {}).get("num_parts", 2))

    objects: list[dict[str, float | str]] = []
    for index, prim_path in enumerate(part_paths):
        pose = pose_lookup.get(prim_path)
        if not pose:
            continue
        position_world = [float(value) for value in pose.get("position", [0.0, 0.0, 0.0])]
        orientation_world = _quat_xyzw_normalized(pose.get("orientation", [0.0, 0.0, 0.0, 1.0])).tolist()
        table_position = world_to_table(position_world, table_frame)
        table_rotation_object = table_rotation.T @ _quat_xyzw_to_matrix(orientation_world)
        objects.append(
            {
                "class": _class_for_part(stage, str(prim_path), index, num_parts_per_class),
                "x": float(table_position[0]),
                "y": float(table_position[1]),
                "yaw": _wrap_pi(_yaw_from_rotation_matrix(table_rotation_object)),
            }
        )
    return objects


def _export_objects_with_noise(
    raw_objects: list[dict[str, float | str]],
    *,
    rng: np.random.Generator,
    apply_noise: bool,
    xy_sigma_m: float,
    yaw_sigma_rad: float,
) -> list[dict[str, float | str]]:
    exported: list[dict[str, float | str]] = []
    for obj in raw_objects:
        x = float(obj["x"])
        y = float(obj["y"])
        yaw = float(obj["yaw"])
        if apply_noise:
            x += float(rng.normal(0.0, xy_sigma_m))
            y += float(rng.normal(0.0, xy_sigma_m))
            yaw = _wrap_pi(yaw + float(rng.normal(0.0, yaw_sigma_rad)))
        exported.append({"class": str(obj["class"]), "x": x, "y": y, "yaw": yaw})
    return exported


def _prepare_run_dirs(output_root: Path, repo_root: Path, run_id: str, save_depth: bool) -> Path:
    output_root = output_root.expanduser().resolve()
    if _is_relative_to(output_root, repo_root):
        raise RuntimeError(
            f"Output root must stay outside the code repo. Got {output_root}; use $OUTPUT_ROOT or another runtime directory."
        )
    run_dir = output_root / run_id
    (run_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    if save_depth:
        (run_dir / "depth").mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--noise-seed", type=int, help="Optional independent seed for exported label noise. Defaults to --seed.")
    parser.add_argument("--robot-prim-path", default=OFFICIAL_ROBOT_PRIM_PATH)
    parser.add_argument("--gui", action="store_true", help="Open Isaac GUI instead of headless mode.")
    parser.add_argument("--no-headless", action="store_true", help="Alias for --gui.")
    parser.add_argument("--hold-open", action="store_true", help="Keep the GUI open after collection.")
    parser.add_argument("--label-noise", dest="label_noise", action="store_true", help="Enable exported label noise.")
    parser.add_argument("--no-label-noise", dest="label_noise", action="store_false", help="Disable exported label noise.")
    parser.set_defaults(label_noise=True)
    parser.add_argument("--label-xy-noise-sigma-m", type=float, default=DEFAULT_XY_NOISE_SIGMA_M)
    parser.add_argument("--label-yaw-noise-sigma-deg", type=float, default=DEFAULT_YAW_NOISE_SIGMA_DEG)
    parser.add_argument("--save-depth", dest="save_depth", action="store_true", help="Save depth .npy arrays.")
    parser.add_argument("--no-save-depth", dest="save_depth", action="store_false", help="Do not save depth arrays.")
    parser.set_defaults(save_depth=False)
    parser.add_argument("--rgb-format", choices=("png", "npy"), default="png", help="RGB export format. PNG falls back to NPY if Pillow is unavailable.")
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
    if args.label_xy_noise_sigma_m < 0.0:
        raise RuntimeError("--label-xy-noise-sigma-m must be non-negative")
    if args.label_yaw_noise_sigma_deg < 0.0:
        raise RuntimeError("--label-yaw-noise-sigma-deg must be non-negative")
    if not str(args.robot_prim_path).startswith("/"):
        raise RuntimeError("--robot-prim-path must be an absolute USD prim path")

    timestamp = _timestamp_compact()
    run_id = args.run_id or f"{timestamp}_seed{int(args.seed)}"
    noise_seed = int(args.noise_seed if args.noise_seed is not None else args.seed)
    yaw_noise_sigma_rad = math.radians(float(args.label_yaw_noise_sigma_deg))

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    repo_root = paths["HRC_REPO"]
    baseline_root = _as_path(args.baseline_root or os.environ.get("HRC_BASELINE_REPO"), paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
    asset_root = _as_path(args.asset_root, paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE)
    config_path = _as_path(args.config, baseline_root / DEFAULT_CONFIG_RELATIVE)
    output_root = _as_path(args.output_root, paths["OUTPUT_ROOT"] / OUTPUT_DATASET_RELATIVE)
    run_dir = _prepare_run_dirs(output_root, repo_root, run_id, bool(args.save_depth))
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
    noise_rng = np.random.default_rng(noise_seed)

    sim_app = None
    robot = None
    counter = {"step": 0}
    try:
        _log(
            log_path,
            "collector_start",
            run_id=run_id,
            samples=args.samples,
            save_depth=bool(args.save_depth),
            label_noise=bool(args.label_noise),
        )
        SimulationApp = _load_simulation_app()
        sim_app = SimulationApp({"headless": not (args.gui or args.no_headless)})

        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
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
        rep.orchestrator.step()
        _run_updates(sim_app, args.init_steps, counter)

        RobotArticulation = _load_official_robot_articulation(baseline_root)
        robot = RobotArticulation(prim_path=args.robot_prim_path, name=OFFICIAL_ROBOT_NAME)
        robot.initialize()
        _run_updates(sim_app, args.camera_warmup_steps, counter)

        part_paths = list(getattr(scene, "parts_prim_paths", []))
        if not part_paths:
            _fail("no_task1_parts", "SceneBuilder did not expose any Task 1 part prim paths")

        robot_cfg = cfg.get("robot", {})
        configured_robot_position = np.array(robot_cfg.get("robot_position", [0.0, 0.0, 0.0]), dtype=float)
        configured_robot_rotation = robot_cfg.get("robot_rotation", [0.0, 0.0, 0.0])
        robot_base_yaw_rad = math.radians(float(configured_robot_rotation[2]) if len(configured_robot_rotation) >= 3 else 0.0)
        table_frame = build_or_resolve_table_frame(
            stage=stage,
            table_path=TABLE_PATH,
            table_bbox=_bbox(stage, TABLE_PATH),
            robot_base_position=configured_robot_position,
            robot_base_yaw_rad=robot_base_yaw_rad,
        )
        _log(log_path, "scene_ready", part_count=len(part_paths), simulation_step=counter["step"])

        for sample_index in range(int(args.samples)):
            _run_updates(sim_app, args.sample_stride, counter)
            sample_id = f"sample_{sample_index:06d}"
            camera_data = robot.get_cameras_images(int(counter["step"]))
            rgb_paths, depth_paths, warnings = _save_camera_sample(
                run_dir=run_dir,
                sample_id=sample_id,
                camera_data=camera_data,
                rgb_format=str(args.rgb_format),
                save_depth=bool(args.save_depth),
            )

            raw_objects = _raw_table_objects(stage, scene, cfg, table_frame)
            label_objects = _export_objects_with_noise(
                raw_objects,
                rng=noise_rng,
                apply_noise=bool(args.label_noise),
                xy_sigma_m=float(args.label_xy_noise_sigma_m),
                yaw_sigma_rad=yaw_noise_sigma_rad,
            )
            labels_path = run_dir / "labels" / f"{sample_id}.json"
            _write_json(labels_path, {"objects": label_objects})

            manifest_entry = {
                "run_id": run_id,
                "seed": int(args.seed),
                "sample_id": sample_id,
                "sample_index": int(sample_index),
                "rgb": rgb_paths,
                "label": _relative_path(labels_path, run_dir),
                "has_depth": bool(depth_paths),
                "depth": depth_paths,
                "object_count": len(label_objects),
                "label_noise": {
                    "applied": bool(args.label_noise),
                    "xy_sigma_m": float(args.label_xy_noise_sigma_m) if args.label_noise else 0.0,
                    "yaw_sigma_deg": float(args.label_yaw_noise_sigma_deg) if args.label_noise else 0.0,
                    "seed": noise_seed if args.label_noise else None,
                },
            }
            _append_jsonl(manifest_path, manifest_entry)
            _log(
                log_path,
                "sample_saved",
                sample_id=sample_id,
                rgb_count=len(rgb_paths),
                has_depth=bool(depth_paths),
                object_count=len(label_objects),
                warnings=warnings,
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
