#!/usr/bin/env python3
"""Inspect official HRC 2026 external resources on the Linux runtime machine.

This is a filesystem diagnostic only. It does not launch Isaac Sim and does not
run task logic.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ENV = ("HRC_ROOT", "HRC_REPO", "DATA_ROOT", "CKPT_ROOT", "OUTPUT_ROOT", "LOG_ROOT")
POINTER_MARKER = "version https://git-lfs.github.com/spec/"


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value).expanduser().resolve()


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 8192:
            return False
        return path.read_text(encoding="utf-8", errors="ignore").startswith(POINTER_MARKER)
    except OSError:
        return False


def _key_files(path: Path, limit: int = 30) -> list[str]:
    if not path.exists():
        return []
    suffixes = {".usd", ".usda", ".usdc", ".urdf", ".json", ".yaml", ".yml", ".parquet"}
    files = [
        item
        for item in path.rglob("*")
        if item.is_file() and (item.suffix.lower() in suffixes or item.name in {"README.md", "run.sh"})
    ]
    files.sort(key=lambda item: (item.suffix.lower(), str(item)))
    return [f"{item.stat().st_size} {item}" for item in files[:limit]]


def _resource_status(path: Path, min_size_bytes: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not path.exists():
        return "MISSING", ["path does not exist"]
    if not path.is_dir():
        return "BROKEN", ["path exists but is not a directory"]

    size = _directory_size_bytes(path)
    if size < min_size_bytes:
        reasons.append(f"directory too small: {size} bytes < {min_size_bytes} bytes")

    pointers = [str(item) for item in path.rglob("*") if item.is_file() and _is_lfs_pointer(item)]
    if pointers:
        reasons.append(f"git_lfs_pointer_files={pointers[:20]}")

    return ("BROKEN" if reasons else "OK"), reasons


def main() -> int:
    paths = {name: _env_path(name) for name in REQUIRED_ENV}
    resources = {
        "baseline_repo": (
            paths["HRC_ROOT"] / "baseline" / "GlobalHumanoidRobotChallenge_2026_Baseline",
            500_000,
        ),
        "walker_s2_model": (
            paths["HRC_ROOT"] / "assets" / "WalkerS2-Model-Challenge",
            100_000_000,
        ),
        "official_assets": (
            paths["HRC_ROOT"] / "assets" / "challenge2026_assets",
            100_000_000,
        ),
        "official_dataset": (
            paths["DATA_ROOT"] / "challenge2026_dataset",
            100_000_000,
        ),
    }

    rows: list[str] = [
        "status=official_resources_inventory",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
    ]
    for name in REQUIRED_ENV:
        rows.append(f"env_{name}={paths[name]}")

    for name, (path, min_size) in resources.items():
        status, reasons = _resource_status(path, min_size)
        rows.extend(
            (
                f"resource={name}",
                f"{name}_path={path}",
                f"{name}_size_bytes={_directory_size_bytes(path)}",
                f"{name}_status={status}",
                f"{name}_reasons={reasons}",
                f"{name}_key_files={_key_files(path)}",
            )
        )

    log_file = paths["LOG_ROOT"] / "official_resources_inventory.log"
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print(f"wrote_log={log_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"official resource inspection FAILED: {exc}")
        raise SystemExit(1)
