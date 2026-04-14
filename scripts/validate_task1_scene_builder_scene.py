#!/usr/bin/env python3
"""Validate Task 1 objects after official SceneBuilder instantiation.

This diagnostic imports the official baseline SceneBuilder, overrides the
loaded Task 1 config root_path to the verified external assets directory, builds
table + boxes + parts only, and checks composed runtime physics. It contains no
robot, perception, sorting, grasping, or task policy logic.
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
from validate_task1_object_assets import _bbox, _has_api, _physics_summary  # type: ignore


DEFAULT_BASELINE_RELATIVE = "baseline/GlobalHumanoidRobotChallenge_2026_Baseline"
DEFAULT_ASSET_ROOT_RELATIVE = "assets/challenge2026_assets/resources"
DEFAULT_CONFIG_RELATIVE = "Ubtech_sim/config/Part_Sorting.yaml"
DEFAULT_INIT_STEPS = 60
DEFAULT_SETTLE_STEPS = 240
DEFAULT_DROP_HEIGHT = 0.20
DEFAULT_REST_JITTER = 0.01
DEFAULT_TABLE_TOLERANCE = 0.025
LOG_NAME = "task1_scene_builder_validation.log"


class _NullDataLogger:
    def log_poses(self, poses_data: Any) -> None:
        return None


def _path_from_env_or_default(env_name: str, default_path: Path) -> Path:
    raw = os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else default_path.resolve()


def _load_official_scene_builder(baseline_root: Path, config_path: Path) -> tuple[dict[str, Any], Any, Any]:
    ubtech_path = baseline_root / "Ubtech_sim"
    if not ubtech_path.exists():
        raise RuntimeError(f"Official Ubtech_sim directory missing: {ubtech_path}")
    if not config_path.exists():
        raise RuntimeError(f"Task 1 config missing: {config_path}")

    sys.path.insert(0, str(ubtech_path))
    from source.SceneBuilder import SceneBuilder  # type: ignore
    from source.config_loader import apply_scatter_config, load_config  # type: ignore

    cfg = load_config(str(config_path))
    return cfg, apply_scatter_config, SceneBuilder


def _create_minimal_world() -> Any:
    import omni.usd  # type: ignore
    from pxr import UsdGeom, UsdLux  # type: ignore
    from isaacsim.core.api import World  # type: ignore

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not provide a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    light = UsdLux.DistantLight.Define(stage, "/Root/Light")
    light.CreateIntensityAttr(500)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 20.0)
    world.initialize_physics()
    return world


def _get_stage() -> Any:
    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage")
    return stage


def _reference_paths(prim: Any) -> list[str]:
    refs: list[str] = []
    stack = prim.GetPrimStack()
    for spec in stack:
        try:
            explicit_items = spec.referenceList.GetExplicitItems()
            added_items = spec.referenceList.GetAddedItems()
        except Exception:
            continue
        for ref in list(explicit_items) + list(added_items):
            asset_path = getattr(ref, "assetPath", "")
            if asset_path and asset_path not in refs:
                refs.append(str(asset_path))
    return refs


def _mass_summary(stage: Any, root_path: str) -> dict[str, Any]:
    from pxr import Usd, UsdPhysics  # type: ignore

    masses: list[dict[str, Any]] = []
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {"mass_count": 0, "masses": []}
    for prim in Usd.PrimRange(root):
        try:
            if not _has_api(prim, UsdPhysics.MassAPI, "PhysicsMassAPI"):
                continue
            mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
            mass = mass_attr.Get() if mass_attr and mass_attr.IsValid() else None
            masses.append({"path": str(prim.GetPath()), "mass": None if mass is None else float(mass)})
        except Exception as exc:
            masses.append({"path": str(prim.GetPath()), "error": str(exc)})
    return {"mass_count": len(masses), "masses": masses[:20]}


def _find_rigid_body_path(stage: Any, base_path: str) -> str | None:
    from pxr import Usd, UsdPhysics  # type: ignore

    prim = stage.GetPrimAtPath(base_path)
    if not prim or not prim.IsValid():
        return None
    for candidate in Usd.PrimRange(prim):
        if _has_api(candidate, UsdPhysics.RigidBodyAPI, "PhysicsRigidBodyAPI"):
            return str(candidate.GetPath())
    return None


def _rigid_body_schema_issues(stage: Any, root_path: str) -> list[dict[str, str]]:
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return [{"path": root_path, "issue": "missing_root_prim"}]

    issues: list[dict[str, str]] = []
    rigid_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if not _has_api(prim, UsdPhysics.RigidBodyAPI, "PhysicsRigidBodyAPI"):
            continue
        path = str(prim.GetPath())
        rigid_paths.append(path)
        if not prim.IsA(UsdGeom.Xformable):
            issues.append({"path": path, "issue": "rigid_body_on_non_xformable_prim"})

    for child_path in rigid_paths:
        for parent_path in rigid_paths:
            if child_path != parent_path and child_path.startswith(parent_path + "/"):
                issues.append({"path": child_path, "issue": f"nested_rigid_body_under:{parent_path}"})
                break
    return issues


def _object_state(stage: Any, prim_path: str, table_top_z: float) -> dict[str, Any]:
    box = _bbox(stage, prim_path)
    values = box["min"] + box["max"] + box["center"]
    return {
        "bbox": box,
        "center": box["center"],
        "bottom_z": box["min"][2],
        "penetration_below_table_m": max(0.0, table_top_z - box["min"][2]),
        "finite": all(math.isfinite(value) for value in values),
    }


def _set_world_pose(stage: Any, prim_path: str, position: np.ndarray, orientation_wxyz: np.ndarray) -> None:
    from pxr import Gf, UsdGeom  # type: ignore

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot set pose for missing prim: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xformable.AddOrientOp().Set(
        Gf.Quatf(float(orientation_wxyz[0]), float(orientation_wxyz[1]), float(orientation_wxyz[2]), float(orientation_wxyz[3]))
    )


def _zero_rigid_velocity(stage: Any, prim_path: str) -> None:
    rigid_path = _find_rigid_body_path(stage, prim_path)
    if rigid_path is None:
        return
    prim = stage.GetPrimAtPath(rigid_path)
    if not prim or not prim.IsValid():
        return
    for attr_name in ("physics:velocity", "physics:angularVelocity"):
        attr = prim.GetAttribute(attr_name)
        if attr and attr.IsValid():
            try:
                attr.Set((0.0, 0.0, 0.0))
            except Exception:
                pass


def _validate_parts_after_lift(
    world: Any,
    stage: Any,
    part_paths: list[str],
    table_bbox: dict[str, list[float]],
    settle_steps: int,
    drop_height: float,
    rest_jitter: float,
    table_tolerance: float,
) -> list[dict[str, Any]]:
    table_center = table_bbox["center"]
    table_top_z = table_bbox["max"][2]
    offsets = [(-0.12, -0.07), (-0.04, 0.06), (0.05, -0.05), (0.12, 0.07), (0.0, 0.0)]
    results: list[dict[str, Any]] = []

    world.pause()
    pre_lift_states = {path: _object_state(stage, path, table_top_z) for path in part_paths}
    for index, path in enumerate(part_paths):
        state = pre_lift_states[path]
        offset = offsets[index % len(offsets)]
        position = np.array(
            [
                table_center[0] + offset[0],
                table_center[1] + offset[1],
                table_top_z + drop_height + (state["center"][2] - state["bottom_z"]),
            ],
            dtype=np.float64,
        )
        _set_world_pose(stage, path, position, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        _zero_rigid_velocity(stage, path)

    for _ in range(10):
        world.step(render=False)

    initial_states = {path: _object_state(stage, path, table_top_z) for path in part_paths}
    world.play()
    samples: dict[str, list[dict[str, Any]]] = {path: [] for path in part_paths}
    for step in range(settle_steps):
        world.step(render=False)
        if step >= max(0, settle_steps - 30):
            for path in part_paths:
                samples[path].append(_object_state(stage, path, table_top_z))
    world.pause()

    for path in part_paths:
        initial = initial_states[path]
        final = _object_state(stage, path, table_top_z)
        centers = [sample["center"] for sample in samples[path] if sample["finite"]]
        jitter = 0.0
        if centers:
            mean = [sum(center[i] for center in centers) / len(centers) for i in range(3)]
            jitter = max(math.sqrt(sum((center[i] - mean[i]) ** 2 for i in range(3))) for center in centers)
        fell = final["center"][2] < initial["center"][2] - 0.02
        rested = (
            final["finite"]
            and final["bottom_z"] >= table_top_z - table_tolerance
            and final["bottom_z"] <= table_top_z + 0.08
        )
        stable = jitter <= rest_jitter
        no_explosion = final["finite"] and all(abs(value) < 10.0 for value in final["center"])
        passed = bool(fell and rested and stable and no_explosion)
        reasons: list[str] = []
        if not fell:
            reasons.append("object_did_not_fall_after_lift")
        if not rested:
            reasons.append("object_did_not_rest_on_table")
        if not stable:
            reasons.append("object_jitter_exceeded_threshold")
        if not no_explosion:
            reasons.append("object_exploded_or_nonfinite_pose")
        results.append(
            {
                "prim_path": path,
                "initial_state": initial,
                "final_state": final,
                "fell_under_gravity": fell,
                "rested_on_table": rested,
                "jitter_last_30_steps_m": jitter,
                "stable": stable,
                "no_explosion": no_explosion,
                "validation_passed": passed,
                "failure_reasons": reasons,
            }
        )
    return results


def _category_from_reference(refs: list[str]) -> str:
    joined = " ".join(refs)
    if "Task1_PartA" in joined:
        return "part_a"
    if "Part_B" in joined:
        return "part_b"
    return "unknown"


def _write_log(log_root: Path, payload: dict[str, Any]) -> Path:
    log_path = log_root / LOG_NAME
    lines = [
        f"status={payload['status']}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"ready_for_manipulation={payload['ready_for_manipulation']}",
        f"payload={json.dumps(payload, indent=2, sort_keys=True)}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", help="Official baseline repo root. Defaults to $HRC_ROOT/baseline/GlobalHumanoidRobotChallenge_2026_Baseline.")
    parser.add_argument("--asset-root", help="Official assets resources root. Defaults to $HRC_ROOT/assets/challenge2026_assets/resources.")
    parser.add_argument("--config", help="Task 1 config path. Defaults to Ubtech_sim/config/Part_Sorting.yaml under baseline root.")
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS)
    parser.add_argument("--drop-height", type=float, default=DEFAULT_DROP_HEIGHT)
    parser.add_argument("--rest-jitter", type=float, default=DEFAULT_REST_JITTER)
    parser.add_argument("--table-tolerance", type=float, default=DEFAULT_TABLE_TOLERANCE)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    args = parser.parse_args()

    if args.init_steps < 1 or args.settle_steps < 1:
        raise RuntimeError("--init-steps and --settle-steps must be at least 1")
    if args.drop_height <= 0.0:
        raise RuntimeError("--drop-height must be positive")

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    baseline_root = _path_from_env_or_default("HRC_BASELINE_REPO", paths["HRC_ROOT"] / DEFAULT_BASELINE_RELATIVE)
    if args.baseline_root:
        baseline_root = Path(args.baseline_root).expanduser().resolve()
    asset_root = paths["HRC_ROOT"] / DEFAULT_ASSET_ROOT_RELATIVE
    if args.asset_root:
        asset_root = Path(args.asset_root).expanduser().resolve()
    config_path = baseline_root / DEFAULT_CONFIG_RELATIVE
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    if not asset_root.exists():
        raise RuntimeError(f"Verified asset root missing: {asset_root}")

    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
    try:
        cfg, apply_scatter_config, SceneBuilder = _load_official_scene_builder(baseline_root, config_path)
        original_root_path = cfg.get("root_path")
        cfg["root_path"] = str(asset_root)
        apply_scatter_config(cfg)

        world = _create_minimal_world()
        scene = SceneBuilder(cfg, data_logger=_NullDataLogger())
        scene.build_table()
        scene.build_box()
        scene.build_parts()

        import omni.replicator.core as rep  # type: ignore

        rep.orchestrator.step()
        for _ in range(args.init_steps):
            world.step(render=False)

        stage = _get_stage()
        table_paths = getattr(scene, "table_prim_paths", [])
        box_paths = ["/Root/Box"]
        part_paths = list(getattr(scene, "parts_prim_paths", []))
        table_path = "/Replicator/Ref_Xform"
        table_bbox = _bbox(stage, table_path)

        part_mappings: list[dict[str, Any]] = []
        num_parts_per_class = int(cfg["part"].get("num_parts", 2))
        for index, path in enumerate(part_paths):
            prim = stage.GetPrimAtPath(path)
            refs = _reference_paths(prim) if prim and prim.IsValid() else []
            category_from_refs = _category_from_reference(refs)
            category_from_order = "part_a" if index < num_parts_per_class else "part_b"
            physics = _physics_summary(stage, path)
            mass = _mass_summary(stage, path)
            box = _bbox(stage, path)
            part_mappings.append(
                {
                    "prim_path": path,
                    "category_from_reference": category_from_refs,
                    "category_from_scene_builder_order": category_from_order,
                    "referenced_usd_paths": refs,
                    "bbox": box,
                    "physics": physics,
                    "rigid_body_schema_issues": _rigid_body_schema_issues(stage, path),
                    "mass": mass,
                    "safe_physics_metadata": physics["collision_count"] > 0 and physics["rigid_body_count"] > 0,
                }
            )

        table_summary = {
            "configured_usd": cfg["table"]["table_usd"],
            "scene_table_path": table_path,
            "table_prim_paths_from_scene_builder": table_paths,
            "bbox": table_bbox,
            "physics": _physics_summary(stage, table_path),
            "rigid_body_schema_issues": _rigid_body_schema_issues(stage, table_path),
        }
        box_summary = []
        for path in box_paths:
            if stage.GetPrimAtPath(path).IsValid():
                box_summary.append(
                    {
                        "prim_path": path,
                        "bbox": _bbox(stage, path),
                        "physics": _physics_summary(stage, path),
                        "rigid_body_schema_issues": _rigid_body_schema_issues(stage, path),
                        "mass": _mass_summary(stage, path),
                    }
                )

        physics_results = _validate_parts_after_lift(
            world=world,
            stage=stage,
            part_paths=part_paths,
            table_bbox=table_bbox,
            settle_steps=args.settle_steps,
            drop_height=args.drop_height,
            rest_jitter=args.rest_jitter,
            table_tolerance=args.table_tolerance,
        )

        metadata_ok = bool(part_paths) and all(item["safe_physics_metadata"] for item in part_mappings)
        part_schema_ok = all(not item["rigid_body_schema_issues"] for item in part_mappings)
        physics_ok = bool(physics_results) and all(item["validation_passed"] for item in physics_results)
        table_ok = table_summary["physics"]["collision_count"] > 0 and not table_summary["rigid_body_schema_issues"]
        box_ok = bool(box_summary) and all(
            item["physics"]["collision_count"] > 0 and not item["rigid_body_schema_issues"]
            for item in box_summary
        )
        ready = bool(metadata_ok and part_schema_ok and physics_ok and table_ok and box_ok)
        status = "task1_scene_builder_validation_ok" if ready else "task1_scene_builder_validation_failed"
        payload = {
            "status": status,
            "ready_for_manipulation": ready,
            "baseline_root": str(baseline_root),
            "config_path": str(config_path),
            "original_config_root_path": original_root_path,
            "overridden_root_path": str(asset_root),
            "configured_asset_pools": {
                "part_a_assets": cfg["part"].get("part_a_assets", []),
                "part_b_assets": cfg["part"].get("part_b_assets", []),
                "num_parts_per_class": cfg["part"].get("num_parts"),
            },
            "table": table_summary,
            "boxes": box_summary,
            "parts": part_mappings,
            "physics_validation": physics_results,
            "mismatch_vs_standalone": {
                "expected_part_count": cfg["part"].get("num_parts", 2) * 2,
                "actual_part_count": len(part_paths),
                "all_parts_have_collision_and_rigid_body": metadata_ok,
                "all_parts_have_valid_rigid_body_schema": part_schema_ok,
                "all_parts_fell_and_rested": physics_ok,
                "table_collision_and_schema_ok": table_ok,
                "box_collision_and_schema_ok": box_ok,
                "note": "Standalone validation used isolated object USDs; this run validates objects after SceneBuilder Replicator creation and physics setup.",
            },
            "unverified": [
                "robot grasping",
                "object transport",
                "sorting correctness",
                "official scoring/reset integration",
            ],
        }
        log_path = _write_log(paths["LOG_ROOT"], payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Task 1 SceneBuilder validation wrote {log_path}")
        if not ready:
            raise RuntimeError(f"Task 1 SceneBuilder validation failed; see {log_path}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Task 1 SceneBuilder validation FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
