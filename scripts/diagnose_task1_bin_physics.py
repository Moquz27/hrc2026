#!/usr/bin/env python3
"""Diagnose Task 1 box/bin physics in isolation and through SceneBuilder.

This script is diagnostic-only. It does not modify the official baseline repo,
does not implement manipulation, and does not change runtime task logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from load_walker_s2 import _load_simulation_app, _validate_environment  # type: ignore
from validate_task1_object_assets import _add_reference, _bbox, _create_stage, _physics_summary  # type: ignore
from validate_task1_scene_builder_scene import (  # type: ignore
    _NullDataLogger,
    _load_official_scene_builder,
    _mass_summary,
    _path_from_env_or_default,
    _rigid_body_schema_issues,
)


DEFAULT_BASELINE_RELATIVE = "baseline/GlobalHumanoidRobotChallenge_2026_Baseline"
DEFAULT_ASSET_ROOT_RELATIVE = "assets/challenge2026_assets/resources"
DEFAULT_CONFIG_RELATIVE = "Ubtech_sim/config/Part_Sorting.yaml"
DEFAULT_BOX_RELATIVE = "Box_blank/box_60_40_23_cut_0.usd"
DEFAULT_TEST_PART_RELATIVE = "Collected_Task1_PartA_red/Task1_PartA.usd"
DEFAULT_STEPS = 240
DEFAULT_REST_JITTER = 0.01
LOG_NAME = "task1_bin_physics_diagnostic.log"


def _get_stage() -> Any:
    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage")
    return stage


def _add_box_collider(stage: Any, path: str, center: list[float], size: list[float]) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    cube.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _add_static_bin_colliders(stage: Any, visual_bbox: dict[str, list[float]]) -> dict[str, Any]:
    min_v = visual_bbox["min"]
    max_v = visual_bbox["max"]
    center = visual_bbox["center"]
    sx = max_v[0] - min_v[0]
    sy = max_v[1] - min_v[1]
    sz = max_v[2] - min_v[2]
    thickness = 0.012
    wall_h = max(0.04, sz)
    colliders = [
        ("/World/DiagnosticBin/Floor", [center[0], center[1], min_v[2] + thickness * 0.5], [sx, sy, thickness]),
        ("/World/DiagnosticBin/WallXMin", [min_v[0] + thickness * 0.5, center[1], min_v[2] + wall_h * 0.5], [thickness, sy, wall_h]),
        ("/World/DiagnosticBin/WallXMax", [max_v[0] - thickness * 0.5, center[1], min_v[2] + wall_h * 0.5], [thickness, sy, wall_h]),
        ("/World/DiagnosticBin/WallYMin", [center[0], min_v[1] + thickness * 0.5, min_v[2] + wall_h * 0.5], [sx, thickness, wall_h]),
        ("/World/DiagnosticBin/WallYMax", [center[0], max_v[1] - thickness * 0.5, min_v[2] + wall_h * 0.5], [sx, thickness, wall_h]),
    ]
    for path, collider_center, collider_size in colliders:
        _add_box_collider(stage, path, collider_center, collider_size)
    return {
        "collider_paths": [item[0] for item in colliders],
        "floor_top_z": min_v[2] + thickness,
        "size": [sx, sy, sz],
        "wall_thickness": thickness,
    }


def _state(stage: Any, prim_path: str, surface_z: float) -> dict[str, Any]:
    box = _bbox(stage, prim_path)
    values = box["min"] + box["max"] + box["center"]
    return {
        "bbox": box,
        "bottom_z": box["min"][2],
        "center": box["center"],
        "penetration_below_surface_m": max(0.0, surface_z - box["min"][2]),
        "finite": all(math.isfinite(value) for value in values),
    }


def _set_pose(stage: Any, prim_path: str, position: list[float]) -> None:
    from pxr import Gf, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def _apply_rigid_if_needed(stage: Any, prim_path: str) -> bool:
    from pxr import UsdPhysics  # type: ignore

    physics = _physics_summary(stage, prim_path)
    if physics["rigid_body_count"] > 0:
        return False
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.2)
    return True


def _disable_physics_under(stage: Any, root_path: str) -> list[str]:
    from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Cannot disable physics for missing prim: {root_path}")
    changed: list[str] = []
    for prim in Usd.PrimRange(root):
        schemas = set(prim.GetAppliedSchemas())
        if "PhysicsRigidBodyAPI" in schemas:
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            changed.append(f"{prim.GetPath()}:RigidBodyAPI")
        if "PhysicsCollisionAPI" in schemas:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            changed.append(f"{prim.GetPath()}:CollisionAPI")
        if "PhysicsMeshCollisionAPI" in schemas:
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            changed.append(f"{prim.GetPath()}:MeshCollisionAPI")
        if "PhysxMeshCollisionAPI" in schemas:
            prim.RemoveAPI(PhysxSchema.PhysxMeshCollisionAPI)
            changed.append(f"{prim.GetPath()}:PhysxMeshCollisionAPI")
    return changed


def _inspect(stage: Any, prim_path: str) -> dict[str, Any]:
    return {
        "prim_path": prim_path,
        "bbox": _bbox(stage, prim_path),
        "physics": _physics_summary(stage, prim_path),
        "mass": _mass_summary(stage, prim_path),
        "rigid_body_schema_issues": _rigid_body_schema_issues(stage, prim_path),
    }


def _validate_static_bin_strategy(sim_app: Any, box_usd: Path, part_usd: Path, steps: int, rest_jitter: float) -> dict[str, Any]:
    import omni.timeline  # type: ignore

    _create_stage()
    stage = _get_stage()
    _add_reference(stage, "/World/BoxVisual", box_usd)
    for _ in range(30):
        sim_app.update()
    visual_before_physics_strip = _inspect(stage, "/World/BoxVisual")
    removed_visual_physics = _disable_physics_under(stage, "/World/BoxVisual")
    visual = _inspect(stage, "/World/BoxVisual")
    colliders = _add_static_bin_colliders(stage, visual_before_physics_strip["bbox"])
    _add_reference(stage, "/World/TestPart", part_usd)
    for _ in range(30):
        sim_app.update()
    rigid_added = _apply_rigid_if_needed(stage, "/World/TestPart")
    part_bbox = _bbox(stage, "/World/TestPart")
    floor_top = float(colliders["floor_top_z"])
    spawn_z = floor_top + 0.18 + (part_bbox["center"][2] - part_bbox["min"][2])
    _set_pose(stage, "/World/TestPart", [visual["bbox"]["center"][0], visual["bbox"]["center"][1], spawn_z])

    for _ in range(10):
        sim_app.update()
    initial = _state(stage, "/World/TestPart", floor_top)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    samples: list[dict[str, Any]] = []
    for step in range(steps):
        sim_app.update()
        if step >= max(0, steps - 30):
            samples.append(_state(stage, "/World/TestPart", floor_top))
    timeline.stop()
    final = _state(stage, "/World/TestPart", floor_top)
    centers = [sample["center"] for sample in samples if sample["finite"]]
    jitter = 0.0
    if centers:
        mean = [sum(center[i] for center in centers) / len(centers) for i in range(3)]
        jitter = max(math.sqrt(sum((center[i] - mean[i]) ** 2 for i in range(3))) for center in centers)
    fell = final["center"][2] < initial["center"][2] - 0.02
    rested = final["finite"] and floor_top - 0.02 <= final["bottom_z"] <= floor_top + 0.08
    no_explosion = final["finite"] and all(abs(value) < 10.0 for value in final["center"])
    stable = jitter <= rest_jitter
    return {
        "strategy": "diagnostic_replacement_static_bin_collider",
        "box_visual_before_physics_strip": visual_before_physics_strip,
        "box_visual_after_physics_strip": visual,
        "removed_visual_physics": removed_visual_physics[:40],
        "colliders": colliders,
        "test_part_usd": str(part_usd),
        "rigid_body_added_for_test": rigid_added,
        "initial_state": initial,
        "final_state": final,
        "fell_under_gravity": fell,
        "rested_on_static_bin_floor": rested,
        "jitter_last_30_steps_m": jitter,
        "stable": stable,
        "no_explosion": no_explosion,
        "validation_passed": bool(fell and rested and stable and no_explosion),
    }


def _write_log(log_root: Path, payload: dict[str, Any]) -> Path:
    log_path = log_root / LOG_NAME
    strategy_validation = payload.get("strategy_validation", {"validation_passed": False})
    lines = [
        f"status={payload['status']}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"root_cause_classification={payload['root_cause_classification']}",
        f"chosen_unblock_strategy={payload['chosen_unblock_strategy']}",
        f"validation_passed={strategy_validation['validation_passed']}",
        f"payload={json.dumps(payload, indent=2, sort_keys=True)}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root")
    parser.add_argument("--asset-root")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--rest-jitter", type=float, default=DEFAULT_REST_JITTER)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    args = parser.parse_args()

    if args.steps < 1:
        raise RuntimeError("--steps must be positive")
    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    baseline_root = _path_from_env_or_default("HRC_BASELINE_REPO", paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
    if args.baseline_root:
        baseline_root = Path(args.baseline_root).expanduser().resolve()
    asset_root = paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE
    if args.asset_root:
        asset_root = Path(args.asset_root).expanduser().resolve()
    config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
    box_usd = asset_root / DEFAULT_BOX_RELATIVE
    part_usd = asset_root / DEFAULT_TEST_PART_RELATIVE
    if not box_usd.exists():
        raise RuntimeError(f"Box USD missing: {box_usd}")

    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
    try:
        print("phase=standalone_box_inspection")
        _create_stage()
        standalone_stage = _get_stage()
        _add_reference(standalone_stage, "/World/StandaloneBox", box_usd)
        for _ in range(30):
            sim_app.update()
        standalone = _inspect(standalone_stage, "/World/StandaloneBox")
        print("phase=standalone_box_done")

        print("phase=scenebuilder_box_inspection")
        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        cfg["root_path"] = str(asset_root)
        apply_scatter_config(cfg)
        _create_stage()
        composed_stage = _get_stage()
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        scene.build_box()
        for _ in range(30):
            sim_app.update()
        composed = _inspect(composed_stage, "/Root/Box")
        print("phase=scenebuilder_box_done")

        print("phase=diagnostic_static_bin_validation")
        strategy_validation = _validate_static_bin_strategy(sim_app, box_usd, part_usd, args.steps, args.rest_jitter)
        print("phase=diagnostic_static_bin_done")

        asset_invalid = bool(standalone["rigid_body_schema_issues"])
        code_added_issues = len(composed["rigid_body_schema_issues"]) > len(standalone["rigid_body_schema_issues"])
        if asset_invalid and code_added_issues:
            root_cause = "both"
        elif asset_invalid:
            root_cause = "asset"
        elif code_added_issues:
            root_cause = "code"
        else:
            root_cause = "not_reproduced"

        payload = {
            "status": "task1_bin_physics_diagnostic_ok" if strategy_validation["validation_passed"] else "task1_bin_physics_diagnostic_failed",
            "root_cause_classification": root_cause,
            "box_usd": str(box_usd),
            "baseline_config_path": str(config_path),
            "standalone_box": standalone,
            "scene_builder_box": composed,
            "scene_builder_trace": {
                "build_box": "stage_utils.add_reference_to_stage(box_usd, /Root/Box), optional clones, XFormPrim positions/scales, then _lock_box_positions when lock_boxes=true",
                "lock_box_positions": "traverses every prim below /Root/Box and applies RigidBodyAPI when CanApply, then sets kinematicEnabled=true",
                "observed_effect": "RigidBodyAPI appears on material/shader and nested child prims after SceneBuilder locking.",
            },
            "chosen_unblock_strategy": "diagnostic_replacement_static_bin_collider",
            "strategy_validation": strategy_validation,
            "remaining_unverified": [
                "robot placing parts into the diagnostic bin",
                "whether official scoring accepts diagnostic bin collider prims",
                "whether a targeted _lock_box_positions patch is preferable after baseline ownership is decided",
            ],
        }
        log_path = _write_log(paths["LOG_ROOT"], payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Task 1 bin physics diagnostic wrote {log_path}")
        if not strategy_validation["validation_passed"]:
            raise RuntimeError(f"Bin diagnostic unblock strategy failed; see {log_path}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Task 1 bin physics diagnostic FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
