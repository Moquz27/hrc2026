#!/usr/bin/env python3
"""Minimal reproducible Isaac Sim scene baseline.

Run this on the Linux runtime machine with Isaac Sim's Python environment. This
script intentionally contains no robot model, task logic, dataset access,
perception, or learning code.
"""

from __future__ import annotations

import argparse
import json
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

DEFAULT_FRAMES = 60


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


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
    metrics_root = paths["OUTPUT_ROOT"] / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    paths["METRICS_ROOT"] = metrics_root
    return paths


def _create_static_cube_scene() -> str:
    import omni.usd  # type: ignore
    from pxr import Gf, UsdGeom, UsdLux  # type: ignore

    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac Sim did not provide a USD stage.")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube_path = "/World/StaticCube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))

    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(500)

    return cube_path


def _write_text_log(log_root: Path, frames: int, scene_prim: str) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file = log_root / "minimal_scene_baseline.log"
    log_file.write_text(
        "\n".join(
            (
                "minimal_scene_baseline_ok",
                f"timestamp_utc={timestamp}",
                f"frames={frames}",
                f"scene_prim={scene_prim}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def _write_metric(metrics_root: Path, frames: int, scene_prim: str, log_file: Path) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    metric_file = metrics_root / "minimal_scene_baseline.json"
    metric = {
        "status": "ok",
        "phase": "minimal_simulation_baseline",
        "timestamp_utc": timestamp,
        "frames_requested": frames,
        "frames_stepped": frames,
        "scene_prim": scene_prim,
        "robot_model": None,
        "dataset": None,
        "task_logic": None,
        "perception": None,
        "learning": None,
        "log_file": str(log_file),
    }
    metric_file.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metric_file


def _hold_gui_open(sim_app: Any) -> None:
    print("Holding Isaac Sim GUI open. Close the window or press Ctrl+C to exit.")
    try:
        while sim_app.is_running():
            sim_app.update()
    except KeyboardInterrupt:
        print("Ctrl+C received; closing Isaac Sim.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Frames to step.")
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
        help="After writing artifacts, keep the GUI open until the window closes or Ctrl+C is pressed.",
    )
    args = parser.parse_args()

    if args.frames < 1:
        raise RuntimeError("--frames must be at least 1")

    # Prevent Isaac Kit from consuming this script's CLI arguments.
    sys.argv = [sys.argv[0]]

    paths = _validate_environment()
    print(f"HRC_REPO={paths['HRC_REPO']}")
    print(f"LOG_ROOT={paths['LOG_ROOT']}")
    print(f"METRICS_ROOT={paths['METRICS_ROOT']}")

    show_gui = args.no_headless or args.gui
    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not show_gui})

    try:
        scene_prim = _create_static_cube_scene()
        sim_app.update()

        for frame in range(args.frames):
            sim_app.update()
            print(f"stepped {frame + 1}/{args.frames}")

        log_file = _write_text_log(paths["LOG_ROOT"], args.frames, scene_prim)
        metric_file = _write_metric(paths["METRICS_ROOT"], args.frames, scene_prim, log_file)
        print(f"Minimal scene baseline OK; wrote {log_file}")
        print(f"Minimal scene baseline metrics OK; wrote {metric_file}")

        if show_gui and args.hold_open:
            _hold_gui_open(sim_app)
    finally:
        sim_app.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Minimal scene baseline FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
