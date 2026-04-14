#!/usr/bin/env python3
"""Load one official HRC asset USD with Walker S2 in Isaac Sim.

This diagnostic validates resource usability only. It does not implement task
logic, scoring, manipulation, perception, or learning.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from load_walker_s2 import (  # type: ignore
    DEFAULT_INIT_STEPS,
    DEFAULT_PRIM_PATH,
    _find_articulation_roots,
    _find_joint_names,
    _load_robot_reference,
    _load_simulation_app,
    _resolve_robot_usd,
    _validate_environment,
)


DEFAULT_SCENE_RELATIVE = "assets/challenge2026_assets/resources/Collected_table_v2/table_v2.usd"


def _resolve_scene_usd(raw_path: str | None, hrc_root: Path) -> Path:
    if not raw_path:
        raw_path = os.environ.get("OFFICIAL_SCENE_USD")
    scene_usd = Path(raw_path).expanduser().resolve() if raw_path else hrc_root / DEFAULT_SCENE_RELATIVE
    if not scene_usd.exists():
        raise RuntimeError(f"Official scene USD does not exist: {scene_usd}")
    if not scene_usd.is_file():
        raise RuntimeError(f"Official scene USD is not a file: {scene_usd}")
    if scene_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise RuntimeError(f"Official scene path is not a USD file: {scene_usd}")
    header = scene_usd.read_text(encoding="utf-8", errors="ignore")[:128]
    if header.startswith("version https://git-lfs.github.com/spec/"):
        raise RuntimeError(f"Official scene USD is a Git LFS pointer, not a payload: {scene_usd}")
    return scene_usd


def _dependency_report(usd_path: Path) -> tuple[list[str], list[str], list[str]]:
    try:
        from pxr import UsdUtils  # type: ignore
    except Exception:
        return [], [], ["UsdUtils import failed; dependency validation unavailable"]

    try:
        result = UsdUtils.ComputeAllDependencies(str(usd_path))
    except Exception as exc:
        return [], [], [f"ComputeAllDependencies failed: {exc}"]

    dependencies: list[str] = []
    unresolved: list[str] = []
    if isinstance(result, tuple):
        for value in result:
            if isinstance(value, (list, tuple)):
                for item in value:
                    text = str(item)
                    if text and text not in dependencies:
                        dependencies.append(text)
        if len(result) >= 3 and isinstance(result[2], (list, tuple)):
            unresolved = [str(item) for item in result[2]]
    return dependencies, unresolved, []


def _create_stage_with_scene_and_robot(stage: Any, scene_usd: Path, robot_usd: Path, robot_prim_path: str) -> None:
    from pxr import UsdGeom, UsdLux, UsdPhysics  # type: ignore

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(500)

    scene_prim = stage.DefinePrim("/World/OfficialScene", "Xform")
    if not scene_prim.GetReferences().AddReference(str(scene_usd)):
        raise RuntimeError(f"Could not add official scene reference: {scene_usd}")
    _load_robot_reference(stage, robot_usd, robot_prim_path)


def _prim_count(stage: Any, root_path: str) -> int:
    from pxr import Usd  # type: ignore

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    return sum(1 for _ in Usd.PrimRange(root))


def _write_log(
    log_root: Path,
    scene_usd: Path,
    robot_usd: Path,
    robot_prim_path: str,
    scene_prim_count: int,
    robot_prim_count: int,
    articulation_roots: list[str],
    joint_names: list[str],
    scene_unresolved: list[str],
    robot_unresolved: list[str],
    dependency_warnings: list[str],
) -> Path:
    status = "official_scene_smoke_ok" if scene_prim_count > 0 and robot_prim_count > 0 and articulation_roots and not scene_unresolved and not robot_unresolved else "official_scene_smoke_failed"
    rows = [
        f"status={status}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"scene_usd_path={scene_usd}",
        f"robot_usd_path={robot_usd}",
        f"robot_prim_path={robot_prim_path}",
        f"scene_prim_path=/World/OfficialScene",
        f"scene_prim_count={scene_prim_count}",
        f"robot_prim_count={robot_prim_count}",
        f"articulation_roots={articulation_roots}",
        f"joint_count={len(joint_names)}",
        f"scene_unresolved_dependencies={scene_unresolved}",
        f"robot_unresolved_dependencies={robot_unresolved}",
        f"dependency_validation_warnings={dependency_warnings}",
        "tested=official USD reference + Walker S2 reference in one Isaac stage",
        "unverified=task reset logic, scoring, perception, and physical manipulation",
    ]
    log_file = log_root / "official_scene_smoke.log"
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-usd", help="Official scene/object USD to load. Defaults to OFFICIAL_SCENE_USD or table_v2.")
    parser.add_argument("--robot-usd", help="Walker S2 USD path. Defaults to WALKER_S2_USD.")
    parser.add_argument("--prim-path", default=DEFAULT_PRIM_PATH)
    parser.add_argument("--init-steps", type=int, default=DEFAULT_INIT_STEPS)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--gui", action="store_true", help="Alias for --no-headless.")
    args = parser.parse_args()

    if args.init_steps < 1:
        raise RuntimeError("--init-steps must be at least 1")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path")

    sys.argv = [sys.argv[0]]
    paths = _validate_environment()
    scene_usd = _resolve_scene_usd(args.scene_usd, paths["HRC_ROOT"])
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])

    _, scene_unresolved, scene_dependency_warnings = _dependency_report(scene_usd)
    _, robot_unresolved, robot_dependency_warnings = _dependency_report(robot_usd)
    if scene_unresolved or robot_unresolved:
        raise RuntimeError(
            "Unresolved USD dependencies detected before Isaac load: "
            f"scene={scene_unresolved}, robot={robot_unresolved}"
        )

    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not (args.no_headless or args.gui)})
    try:
        import omni.usd  # type: ignore

        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not provide a USD stage")
        _create_stage_with_scene_and_robot(stage, scene_usd, robot_usd, args.prim_path)
        for _ in range(args.init_steps):
            sim_app.update()

        scene_prim_count = _prim_count(stage, "/World/OfficialScene")
        robot_prim_count = _prim_count(stage, args.prim_path)
        articulation_roots = _find_articulation_roots(stage, args.prim_path)
        joint_names = _find_joint_names(stage, args.prim_path)
        if scene_prim_count <= 0:
            raise RuntimeError("Official scene prim hierarchy is empty")
        if robot_prim_count <= 0:
            raise RuntimeError("Walker S2 prim hierarchy is empty")
        if not articulation_roots:
            raise RuntimeError("Walker S2 articulation root was not detected")
        if not joint_names:
            raise RuntimeError("Walker S2 joint prims were not detected")

        log_file = _write_log(
            paths["LOG_ROOT"],
            scene_usd,
            robot_usd,
            args.prim_path,
            scene_prim_count,
            robot_prim_count,
            articulation_roots,
            joint_names,
            scene_unresolved,
            robot_unresolved,
            scene_dependency_warnings + robot_dependency_warnings,
        )
        print(f"Official scene smoke OK; wrote {log_file}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Official scene smoke FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
