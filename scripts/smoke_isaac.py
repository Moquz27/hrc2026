#!/usr/bin/env python3
"""Minimal Isaac Sim runtime smoke test.

Run this on the Linux runtime machine from the repository root with Isaac Sim's
Python environment. It intentionally avoids robot assets, datasets, and task
logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ENV = (
    "HRC_ROOT",
    "HRC_REPO",
    "DATA_ROOT",
    "CKPT_ROOT",
    "OUTPUT_ROOT",
    "LOG_ROOT",
)


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


def _load_simulation_app():
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp
    except Exception as new_api_error:
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


def _write_log(log_root: Path, steps: int, name: str = "isaac_smoke_ok.txt") -> Path:
    log_file = log_root / name
    timestamp = datetime.now(timezone.utc).isoformat()
    log_file.write_text(
        "\n".join(
            (
                "isaac_smoke_ok",
                f"timestamp_utc={timestamp}",
                f"steps={steps}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return log_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="Simulation frames to step.")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Launch Isaac Sim with a visible window instead of headless mode.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate environment variables and LOG_ROOT write access without launching Isaac Sim.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise RuntimeError("--steps must be at least 1")

    paths = _validate_environment()
    print(f"HRC_REPO={paths['HRC_REPO']}")
    print(f"LOG_ROOT={paths['LOG_ROOT']}")

    if args.preflight_only:
        log_file = _write_log(paths["LOG_ROOT"], 0, "isaac_smoke_preflight_ok.txt")
        print(f"Preflight OK; wrote {log_file}")
        return 0

    SimulationApp = _load_simulation_app()
    sim_app = SimulationApp({"headless": not args.no_headless})

    try:
        import omni.usd  # type: ignore

        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        sim_app.update()

        for step in range(args.steps):
            sim_app.update()
            print(f"stepped {step + 1}/{args.steps}")

        log_file = _write_log(paths["LOG_ROOT"], args.steps)
        print(f"Isaac smoke OK; wrote {log_file}")
    finally:
        sim_app.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Isaac smoke FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
