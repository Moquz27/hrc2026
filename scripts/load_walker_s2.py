#!/usr/bin/env python3
"""Load and inspect the Walker S2 USD model in Isaac Sim.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script intentionally stops at robot loading, articulation inspection, and
logging; it contains no task, control, perception, dataset, or learning logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ENV = (
    "HRC_ROOT",
    "HRC_REPO",
    "DATA_ROOT",
    "CKPT_ROOT",
    "OUTPUT_ROOT",
    "LOG_ROOT",
)

DEFAULT_PRIM_PATH = "/World/WalkerS2"
DEFAULT_INIT_STEPS = 120


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_simulation_app() -> Any:
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp
    except Exception:
        try:
            from omni.isaac.kit import SimulationApp  # type: ignore

            return SimulationApp
        except Exception as old_api_error:
            raise RuntimeError(
                "Could not import Isaac Sim SimulationApp. Run this script with "
                "Isaac Sim's Python interpreter or python.sh."
            ) from old_api_error


def _validate_environment() -> dict[str, Path]:
    paths = {name: _env_path(name) for name in REQUIRED_ENV}

    repo_root = Path(__file__).resolve().parents[1]
    if paths["HRC_REPO"] != repo_root:
        raise RuntimeError(
            f"HRC_REPO must point to this repo: expected {repo_root}, "
            f"got {paths['HRC_REPO']}"
        )

    paths["LOG_ROOT"].mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_robot_usd(raw_path: str | None, repo_root: Path) -> Path:
    if not raw_path:
        raw_path = os.environ.get("WALKER_S2_USD")
    if not raw_path:
        raise RuntimeError(
            "Missing Walker S2 USD path. Pass --robot-usd or set WALKER_S2_USD."
        )

    robot_usd = Path(raw_path).expanduser().resolve()
    if not robot_usd.exists():
        raise RuntimeError(f"Wrong USD path; file does not exist: {robot_usd}")
    if not robot_usd.is_file():
        raise RuntimeError(f"Wrong USD path; expected a file: {robot_usd}")
    if robot_usd.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise RuntimeError(f"Wrong USD path; expected .usd/.usda/.usdc: {robot_usd}")
    if _is_relative_to(robot_usd, repo_root):
        raise RuntimeError(
            f"Robot asset must stay outside the code repo: {robot_usd} is under {repo_root}"
        )
    try:
        header = robot_usd.read_text(encoding="utf-8", errors="ignore")[:128]
    except OSError as exc:
        raise RuntimeError(f"Could not read USD file header: {robot_usd}") from exc
    if header.startswith("version https://git-lfs.github.com/spec/"):
        raise RuntimeError(
            f"USD path is a Git LFS pointer, not a downloaded asset payload: {robot_usd}. "
            "Run git lfs pull or otherwise fetch the real Walker S2 assets outside the code repo."
        )
    return robot_usd


def _create_minimal_scene() -> Any:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics  # type: ignore

    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not provide a USD stage.")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.02))
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(500)

    return stage


def _load_robot_reference(stage: Any, robot_usd: Path, prim_path: str) -> Any:
    from pxr import UsdGeom  # type: ignore

    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to define robot prim: {prim_path}")

    if not prim.GetReferences().AddReference(str(robot_usd)):
        raise RuntimeError(f"Isaac/USD could not add robot reference: {robot_usd}")

    UsdGeom.XformCommonAPI(prim).SetTranslate((0.0, 0.0, 0.0))
    return prim


def _prim_has_articulation_api(prim: Any) -> bool:
    from pxr import UsdPhysics  # type: ignore

    try:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return True
    except Exception:
        pass

    applied_schemas = set(prim.GetAppliedSchemas())
    if "PhysicsArticulationRootAPI" in applied_schemas:
        return True

    try:
        from pxr import PhysxSchema  # type: ignore

        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            return True
    except Exception:
        pass

    return "PhysxArticulationAPI" in applied_schemas


def _find_articulation_roots(stage: Any, prim_path: str) -> list[str]:
    from pxr import Usd  # type: ignore

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot prim was not created or did not load: {prim_path}")

    roots: list[str] = []
    for prim in Usd.PrimRange(root):
        if _prim_has_articulation_api(prim):
            roots.append(str(prim.GetPath()))
    return roots


def _find_joint_names(stage: Any, prim_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics  # type: ignore

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Robot prim was not created or did not load: {prim_path}")

    joint_names: list[str] = []
    for prim in Usd.PrimRange(root):
        is_joint = False
        try:
            is_joint = prim.IsA(UsdPhysics.Joint)
        except Exception:
            is_joint = prim.GetTypeName().endswith("Joint")

        if is_joint:
            joint_names.append(str(prim.GetPath()))

    return joint_names


def _try_read_joint_positions(prim_path: str) -> tuple[list[str], list[float] | None, str | None]:
    errors: list[str] = []

    try:
        from omni.isaac.dynamic_control import _dynamic_control  # type: ignore

        dc = _dynamic_control.acquire_dynamic_control_interface()
        articulation = dc.get_articulation(prim_path)
        if articulation:
            dof_count = dc.get_articulation_dof_count(articulation)
            names: list[str] = []
            positions: list[float] = []
            for index in range(dof_count):
                dof = dc.get_articulation_dof(articulation, index)
                names.append(str(dc.get_dof_name(dof)))
                positions.append(float(dc.get_dof_position(dof)))
            return names, positions, None
        errors.append("omni.isaac.dynamic_control: articulation handle was unavailable")
    except Exception as exc:
        errors.append(f"omni.isaac.dynamic_control: {exc}")

    candidates = (
        ("isaacsim.core.api.articulations", "Articulation"),
        ("omni.isaac.core.articulations", "Articulation"),
    )

    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            articulation_cls = getattr(module, class_name)
            robot = articulation_cls(prim_path=prim_path, name="walker_s2_inspect")
            robot.initialize()

            names = list(getattr(robot, "dof_names", []) or [])
            positions = robot.get_joint_positions()
            if positions is None:
                return names, None, None
            if hasattr(positions, "tolist"):
                positions = positions.tolist()
            return names, [float(value) for value in positions], None
        except Exception as exc:
            errors.append(f"{module_name}.{class_name}: {exc}")

    return [], None, "; ".join(errors)


def _write_log(
    log_root: Path,
    robot_usd: Path,
    prim_path: str,
    articulation_available: bool,
    joint_count: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    status = "walker_s2_load_ok" if articulation_available and joint_count > 0 else "walker_s2_load_failed"
    log_file = log_root / "walker_s2_load_ok.txt"
    log_file.write_text(
        "\n".join(
            (
                f"status={status}",
                f"timestamp_utc={timestamp}",
                f"robot_usd_path={robot_usd}",
                f"prim_path={prim_path}",
                f"articulation_available={str(articulation_available).lower()}",
                f"joint_count={joint_count}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def _hold_gui_open(sim_app: Any) -> None:
    print("Holding Isaac Sim GUI open. Close the window or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        print("Ctrl+C received; closing Isaac Sim.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-usd",
        help="Path to the Walker S2 root USD file. Defaults to WALKER_S2_USD.",
    )
    parser.add_argument(
        "--prim-path",
        default=DEFAULT_PRIM_PATH,
        help=f"Stage prim path for the robot reference. Default: {DEFAULT_PRIM_PATH}",
    )
    parser.add_argument(
        "--init-steps",
        type=int,
        default=DEFAULT_INIT_STEPS,
        help="Simulation updates to allow robot initialization.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Launch Isaac Sim with a visible window instead of headless mode.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Alias for --no-headless; launch Isaac Sim with a visible window.",
    )
    parser.add_argument(
        "--hold-open",
        action="store_true",
        help="After inspection, keep the GUI open until the window closes or Ctrl+C is pressed.",
    )
    args = parser.parse_args()

    if args.init_steps < 1:
        raise RuntimeError("--init-steps must be at least 1")
    if not args.prim_path.startswith("/"):
        raise RuntimeError("--prim-path must be an absolute USD prim path, for example /World/WalkerS2")

    # Prevent Isaac Kit from consuming this script's CLI arguments.
    sys.argv = [sys.argv[0]]

    paths = _validate_environment()
    robot_usd = _resolve_robot_usd(args.robot_usd, paths["HRC_REPO"])

    print(f"HRC_REPO={paths['HRC_REPO']}")
    print(f"LOG_ROOT={paths['LOG_ROOT']}")
    print(f"robot_usd_path={robot_usd}")
    print(f"robot_prim_path={args.prim_path}")

    show_gui = args.no_headless or args.gui
    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not show_gui})

    try:
        stage = _create_minimal_scene()
        _load_robot_reference(stage, robot_usd, args.prim_path)
        sim_app.update()

        articulation_roots: list[str] = []
        joint_names: list[str] = []
        for step in range(args.init_steps):
            sim_app.update()
            articulation_roots = _find_articulation_roots(stage, args.prim_path)
            joint_names = _find_joint_names(stage, args.prim_path)
            if articulation_roots and joint_names:
                break
        else:
            raise RuntimeError(
                "Robot did not expose both articulation and joints within "
                f"{args.init_steps} update steps. Increase --init-steps only after "
                "checking the USD path and Isaac console errors."
            )

        articulation_available = bool(articulation_roots)
        if not articulation_available:
            raise RuntimeError("Robot loaded, but no articulation root API was detected.")
        if not joint_names:
            raise RuntimeError("Robot articulation loaded, but no joint prims were detected.")

        dof_names, joint_positions, state_error = _try_read_joint_positions(args.prim_path)
        if state_error:
            print(f"joint_state_read_warning={state_error}")

        print(f"robot_prim_path={args.prim_path}")
        print(f"articulation_available={str(articulation_available).lower()}")
        print(f"articulation_roots={articulation_roots}")
        print(f"joint_count={len(joint_names)}")
        print("joint_names:")
        for joint_name in joint_names:
            print(f"  {joint_name}")

        if dof_names:
            print(f"dof_count={len(dof_names)}")
            print("dof_names:")
            for dof_name in dof_names:
                print(f"  {dof_name}")
        if joint_positions is not None:
            print(f"joint_positions={joint_positions}")

        log_file = _write_log(
            paths["LOG_ROOT"],
            robot_usd,
            args.prim_path,
            articulation_available,
            len(joint_names),
        )
        print(f"Walker S2 load OK; wrote {log_file}")

        if show_gui and args.hold_open:
            _hold_gui_open(sim_app)
    finally:
        sim_app.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Walker S2 load FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
