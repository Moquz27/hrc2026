#!/usr/bin/env python3
"""Validate official Task 1 workpiece assets in a simple Isaac physics scene.

This diagnostic loads one Part A USD and one Part B USD with the official table
visual asset plus a simple diagnostic tabletop collider. It checks object size,
origin offset, collision APIs, and simple gravity/rest behavior. It intentionally
contains no robot, perception, sorting, or task policy logic.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from load_walker_s2 import _load_simulation_app, _validate_environment  # type: ignore


DEFAULT_PART_A_RELATIVE = "assets/challenge2026_assets/resources/Collected_Task1_PartA_red/Task1_PartA.usd"
DEFAULT_PART_B_RELATIVE = "assets/challenge2026_assets/resources/Collected_Part_B_red/Part_B.usd"
DEFAULT_TABLE_RELATIVE = "assets/challenge2026_assets/resources/Collected_table_v2/table_v2.usd"
DEFAULT_INIT_STEPS = 60
DEFAULT_SIM_STEPS = 240
DEFAULT_DROP_HEIGHT = 0.20
DEFAULT_MAX_REASONABLE_DIM = 0.60
DEFAULT_MIN_REASONABLE_DIM = 0.005
DEFAULT_REST_JITTER = 0.01
DEFAULT_TABLE_TOP_THICKNESS = 0.04
POINTER_MARKER = "version https://git-lfs.github.com/spec/"


def _resolve_usd(raw_path: str | None, default_path: Path, label: str) -> Path:
    usd_path = Path(raw_path).expanduser().resolve() if raw_path else default_path
    if not usd_path.exists():
        raise RuntimeError(f"{label} USD does not exist: {usd_path}")
    if not usd_path.is_file():
        raise RuntimeError(f"{label} USD is not a file: {usd_path}")
    if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise RuntimeError(f"{label} path is not a USD file: {usd_path}")
    header = usd_path.read_text(encoding="utf-8", errors="ignore")[:128]
    if header.startswith(POINTER_MARKER):
        raise RuntimeError(f"{label} USD is a Git LFS pointer, not a payload: {usd_path}")
    return usd_path


def _create_stage() -> Any:
    import omni.usd  # type: ignore
    from pxr import UsdGeom, UsdLux, UsdPhysics  # type: ignore

    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not provide a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(500)
    return stage


def _add_reference(stage: Any, prim_path: str, usd_path: Path) -> Any:
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.GetReferences().AddReference(str(usd_path)):
        raise RuntimeError(f"Could not add reference {usd_path} at {prim_path}")
    return prim


def _bbox(stage: Any, prim_path: str) -> dict[str, list[float]]:
    from pxr import Usd, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim does not exist for bbox: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = box.GetMin()
    maximum = box.GetMax()
    min_v = [float(minimum[0]), float(minimum[1]), float(minimum[2])]
    max_v = [float(maximum[0]), float(maximum[1]), float(maximum[2])]
    size = [max_v[index] - min_v[index] for index in range(3)]
    center = [(min_v[index] + max_v[index]) * 0.5 for index in range(3)]
    return {"min": min_v, "max": max_v, "size": size, "center": center}


def _has_api(prim: Any, api: Any, schema_name: str) -> bool:
    try:
        if prim.HasAPI(api):
            return True
    except Exception:
        pass
    return schema_name in set(prim.GetAppliedSchemas())


def _physics_summary(stage: Any, root_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Prim does not exist for physics summary: {root_path}")
    collision_paths: list[str] = []
    rigid_body_paths: list[str] = []
    mesh_collision_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if _has_api(prim, UsdPhysics.CollisionAPI, "PhysicsCollisionAPI"):
            collision_paths.append(str(prim.GetPath()))
        if _has_api(prim, UsdPhysics.RigidBodyAPI, "PhysicsRigidBodyAPI"):
            rigid_body_paths.append(str(prim.GetPath()))
        schemas = set(prim.GetAppliedSchemas())
        if "PhysicsMeshCollisionAPI" in schemas or "PhysxMeshCollisionAPI" in schemas:
            mesh_collision_paths.append(str(prim.GetPath()))
    return {
        "collision_count": len(collision_paths),
        "rigid_body_count": len(rigid_body_paths),
        "mesh_collision_count": len(mesh_collision_paths),
        "collision_paths_sample": collision_paths[:20],
        "rigid_body_paths_sample": rigid_body_paths[:20],
        "mesh_collision_paths_sample": mesh_collision_paths[:20],
    }


def _apply_test_rigid_body(stage: Any, object_prim: Any, physics_summary: dict[str, Any]) -> bool:
    from pxr import UsdPhysics  # type: ignore

    if physics_summary["rigid_body_count"] > 0:
        return False
    if physics_summary["collision_count"] <= 0:
        return False
    UsdPhysics.RigidBodyAPI.Apply(object_prim)
    mass_api = UsdPhysics.MassAPI.Apply(object_prim)
    mass_api.CreateMassAttr(0.1)
    return True


def _set_translate(prim: Any, xyz: list[float]) -> None:
    from pxr import UsdGeom  # type: ignore

    UsdGeom.XformCommonAPI(prim).SetTranslate((float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _add_table_top_collider(stage: Any, table_bbox: dict[str, list[float]], thickness: float) -> dict[str, Any]:
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    table_min = table_bbox["min"]
    table_max = table_bbox["max"]
    table_size = table_bbox["size"]
    collider_path = "/World/ValidationTableTopCollider"
    cube = UsdGeom.Cube.Define(stage, collider_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(
        Gf.Vec3f(
            max(0.5, float(table_size[0])),
            max(0.5, float(table_size[1])),
            float(thickness),
        )
    )
    cube.AddTranslateOp().Set(
        Gf.Vec3d(
            float((table_min[0] + table_max[0]) * 0.5),
            float((table_min[1] + table_max[1]) * 0.5),
            float(table_max[2] - thickness * 0.5),
        )
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return {"path": collider_path, "top_z": float(table_max[2]), "size": [max(0.5, float(table_size[0])), max(0.5, float(table_size[1])), float(thickness)]}


def _reasonable_size(size: list[float], table_size: list[float], min_dim: float, max_dim: float) -> bool:
    positive = [value for value in size if value > 1.0e-6]
    if not positive:
        return False
    object_max = max(positive)
    table_xy = max(table_size[0], table_size[1], 1.0e-6)
    return min_dim <= min(positive) and object_max <= max_dim and object_max < table_xy


def _object_state(stage: Any, object_path: str, table_top_z: float) -> dict[str, Any]:
    box = _bbox(stage, object_path)
    center = box["center"]
    return {
        "bbox": box,
        "center": center,
        "bottom_z": box["min"][2],
        "penetration_below_table_m": max(0.0, table_top_z - box["min"][2]),
        "finite": all(math.isfinite(value) for value in center + box["min"] + box["max"]),
    }


def _validate_one_object(
    stage: Any,
    sim_app: Any,
    label: str,
    object_usd: Path,
    table_usd: Path,
    init_steps: int,
    sim_steps: int,
    drop_height: float,
    rest_jitter: float,
    min_dim: float,
    max_dim: float,
    table_thickness: float,
) -> dict[str, Any]:
    stage = _create_stage()
    table_prim = _add_reference(stage, "/World/Table", table_usd)
    object_path = f"/World/Object_{label}"
    object_prim = _add_reference(stage, object_path, object_usd)
    sim_app.update()
    for _ in range(init_steps):
        sim_app.update()

    table_bbox = _bbox(stage, "/World/Table")
    table_physics = _physics_summary(stage, "/World/Table")
    table_collider = _add_table_top_collider(stage, table_bbox, table_thickness)
    object_bbox_at_origin = _bbox(stage, object_path)
    object_physics_before = _physics_summary(stage, object_path)
    rigid_body_added_for_test = _apply_test_rigid_body(stage, object_prim, object_physics_before)

    table_center = table_bbox["center"]
    object_center = object_bbox_at_origin["center"]
    object_min = object_bbox_at_origin["min"]
    spawn_translation = [
        table_center[0] - object_center[0],
        table_center[1] - object_center[1],
        table_collider["top_z"] + drop_height - object_min[2],
    ]
    _set_translate(object_prim, spawn_translation)
    for _ in range(init_steps):
        sim_app.update()

    import omni.timeline  # type: ignore

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    state_samples: list[dict[str, Any]] = []
    initial_state = _object_state(stage, object_path, table_collider["top_z"])
    for step in range(sim_steps):
        sim_app.update()
        if step >= max(0, sim_steps - 30):
            state_samples.append(_object_state(stage, object_path, table_collider["top_z"]))
    timeline.stop()
    final_state = _object_state(stage, object_path, table_collider["top_z"])
    object_physics_after = _physics_summary(stage, object_path)

    centers = [sample["center"] for sample in state_samples if sample["finite"]]
    jitter = 0.0
    if centers:
        mean = [sum(center[index] for center in centers) / len(centers) for index in range(3)]
        jitter = max(
            math.sqrt(sum((center[index] - mean[index]) ** 2 for index in range(3)))
            for center in centers
        )

    size_ok = _reasonable_size(object_bbox_at_origin["size"], table_bbox["size"], min_dim, max_dim)
    collision_ok = object_physics_before["collision_count"] > 0
    fell_under_gravity = final_state["center"][2] < initial_state["center"][2] - 0.02
    rested_on_table = (
        final_state["finite"]
        and final_state["bottom_z"] >= table_collider["top_z"] - 0.02
        and final_state["bottom_z"] <= table_collider["top_z"] + 0.06
    )
    stable = jitter <= rest_jitter
    no_explosion = final_state["finite"] and all(abs(value) < 10.0 for value in final_state["center"])
    passed = bool(size_ok and collision_ok and fell_under_gravity and rested_on_table and stable and no_explosion)
    failure_reasons: list[str] = []
    if not size_ok:
        failure_reasons.append("unreasonable_object_scale")
    if not collision_ok:
        failure_reasons.append("no_collision_api_detected")
    if not fell_under_gravity:
        failure_reasons.append("object_did_not_fall_under_gravity")
    if not rested_on_table:
        failure_reasons.append("object_did_not_rest_on_table")
    if not stable:
        failure_reasons.append("object_jitter_exceeded_threshold")
    if not no_explosion:
        failure_reasons.append("object_exploded_or_nonfinite_pose")

    return {
        "label": label,
        "object_usd": str(object_usd),
        "table_usd": str(table_usd),
        "table_bbox": table_bbox,
        "table_physics": table_physics,
        "table_top_collider": table_collider,
        "table_collision_validated": table_physics["collision_count"] > 0,
        "table_rest_surface": "diagnostic_tabletop_collider",
        "object_bbox_at_origin": object_bbox_at_origin,
        "object_origin_offset_from_bbox_center": object_bbox_at_origin["center"],
        "object_physics_before": object_physics_before,
        "object_physics_after": object_physics_after,
        "rigid_body_added_for_test": rigid_body_added_for_test,
        "spawn_translation": spawn_translation,
        "initial_state": initial_state,
        "final_state": final_state,
        "fell_under_gravity": fell_under_gravity,
        "rested_on_table": rested_on_table,
        "jitter_last_30_steps_m": jitter,
        "size_ok": size_ok,
        "collision_ok": collision_ok,
        "stable": stable,
        "no_explosion": no_explosion,
        "validation_passed": passed,
        "failure_reasons": failure_reasons,
    }


def _write_log(log_root: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    status = "task1_object_asset_validation_ok" if all(result["validation_passed"] for result in results) else "task1_object_asset_validation_failed"
    rows = [
        f"status={status}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"init_steps={args.init_steps}",
        f"sim_steps={args.sim_steps}",
        f"drop_height={args.drop_height}",
        f"rest_jitter={args.rest_jitter}",
        f"results={results}",
        "tested=official Part A and Part B object size, origin offset, collision API presence, gravity drop, diagnostic tabletop rest, and stability",
        "unverified=official table collision response, robot grasping, robot transport, perception, sorting, and official task scoring",
    ]
    log_file = log_root / "task1_object_asset_validation.log"
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part-a-usd", help="Part A USD path. Defaults to the collected physics-ready Task1_PartA variant.")
    parser.add_argument("--part-b-usd", help="Part B USD path. Defaults to the collected physics-ready Part_B variant.")
    parser.add_argument("--table-usd", help="Table USD path. Defaults to official Collected_table_v2/table_v2.usd.")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--drop-height", type=float, default=DEFAULT_DROP_HEIGHT)
    parser.add_argument("--rest-jitter", type=float, default=DEFAULT_REST_JITTER)
    parser.add_argument("--min-reasonable-dim", type=float, default=DEFAULT_MIN_REASONABLE_DIM)
    parser.add_argument("--max-reasonable-dim", type=float, default=DEFAULT_MAX_REASONABLE_DIM)
    parser.add_argument("--table-top-thickness", type=float, default=DEFAULT_TABLE_TOP_THICKNESS)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    args = parser.parse_args()

    if args.init_steps < 1 or args.sim_steps < 1:
        raise RuntimeError("--init-steps and --sim-steps must be at least 1")
    if args.drop_height <= 0.0:
        raise RuntimeError("--drop-height must be positive")
    if args.rest_jitter <= 0.0:
        raise RuntimeError("--rest-jitter must be positive")
    if args.min_reasonable_dim <= 0.0 or args.max_reasonable_dim <= args.min_reasonable_dim:
        raise RuntimeError("reasonable dimension thresholds are invalid")
    if args.table_top_thickness <= 0.0:
        raise RuntimeError("--table-top-thickness must be positive")

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    part_a = _resolve_usd(args.part_a_usd, paths["HRC_ROOT"] / DEFAULT_PART_A_RELATIVE, "Part A")
    part_b = _resolve_usd(args.part_b_usd, paths["HRC_ROOT"] / DEFAULT_PART_B_RELATIVE, "Part B")
    table = _resolve_usd(args.table_usd, paths["HRC_ROOT"] / DEFAULT_TABLE_RELATIVE, "Table")

    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
    try:
        results = [
            _validate_one_object(
                None,
                sim_app,
                "A",
                part_a,
                table,
                args.init_steps,
                args.sim_steps,
                args.drop_height,
                args.rest_jitter,
                args.min_reasonable_dim,
                args.max_reasonable_dim,
                args.table_top_thickness,
            ),
            _validate_one_object(
                None,
                sim_app,
                "B",
                part_b,
                table,
                args.init_steps,
                args.sim_steps,
                args.drop_height,
                args.rest_jitter,
                args.min_reasonable_dim,
                args.max_reasonable_dim,
                args.table_top_thickness,
            ),
        ]
        log_file = _write_log(paths["LOG_ROOT"], results, args)
        for result in results:
            print(result)
        print(f"Task 1 object asset validation wrote {log_file}")
        if not all(result["validation_passed"] for result in results):
            raise RuntimeError(f"Task 1 object asset validation failed: {results}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Task 1 object asset validation FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
