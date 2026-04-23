#!/usr/bin/env python3
"""Validate and score Phase 1 Task 1 RGB-D collection runs.

This is the Phase 2 dataset/evaluator entrypoint. It validates the collected
sample structure and, when optional prediction files are provided, computes
prediction-vs-truth metrics without touching manipulation, IK, planner, or
camera runtime code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")
METRIC_NAMES = (
    "class_accuracy",
    "selected_object_accuracy",
    "center_2d_error_px",
    "yaw_bucket_accuracy",
    "arm_recommendation_accuracy",
    "preset_recommendation_accuracy",
    "conversion_3d_error_m",
    "task_success_rate",
    "wrong_bin_rate",
    "drop_rate",
    "cycle_time_s",
)
SUCCESS_EXECUTION_RESULTS = {
    "success",
    "task_success",
    "completed",
    "pass",
    "passed",
    "ok",
    "succeeded",
}
FAILURE_EXECUTION_RESULTS = {
    "fail",
    "failed",
    "failure",
    "task_failed",
    "object_outside_bin",
    "wrong_bin",
    "drop",
    "dropped",
}
COLLECTOR_ONLY_RESULTS = {"collection_only_no_manipulation"}


class EvaluationError(RuntimeError):
    """Raised for CLI/configuration errors before dataset validation starts."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}: invalid JSONL line {line_index}: {exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationError(f"{path}: JSONL line {line_index} is not an object")
            records.append(record)
    return records


def _as_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _resolve_existing_or_candidate(
    value: str | None,
    *,
    dataset_root: Path | None,
    base_dir: Path | None = None,
) -> Path | None:
    if not value:
        return None
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw
    candidates: list[Path] = []
    if dataset_root is not None:
        candidates.append(dataset_root / raw)
    if base_dir is not None:
        candidates.append(base_dir / raw)
    candidates.append(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_dataset_root_for_run(run_id: str) -> Path:
    output_root = os.environ.get("OUTPUT_ROOT")
    if not output_root:
        raise EvaluationError("OUTPUT_ROOT is required when --dataset-root is omitted")
    return Path(output_root) / "datasets" / "task1_rgbd_labels" / run_id


def _load_optional_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def _load_evaluator_io(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise EvaluationError(f"{path}: evaluator I/O file must be a JSON object")
    for field in ("schema_version", "run_id", "inputs", "outputs"):
        if field not in payload:
            raise EvaluationError(f"{path}: missing required evaluator I/O field {field!r}")
    if not isinstance(payload.get("inputs"), dict):
        raise EvaluationError(f"{path}: evaluator I/O field 'inputs' must be an object")
    return payload


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.size > 0 and np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        finite_count = int(np.count_nonzero(finite))
        summary["finite_count"] = finite_count
        if finite_count:
            finite_values = array[finite]
            summary["finite_min"] = float(np.min(finite_values))
            summary["finite_max"] = float(np.max(finite_values))
    return summary


def _load_array_for_validation(
    path: Path,
    *,
    expected_shape: list[Any] | None,
    array_kind: str,
    errors: list[str],
    warnings: list[str],
    context: str,
) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"{context}: missing {array_kind} array file: {path}")
        return None
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        errors.append(f"{context}: failed to load {array_kind} array {path}: {exc}")
        return None

    summary = _array_summary(array)
    shape = summary["shape"]
    if expected_shape is not None and shape != [int(value) for value in expected_shape]:
        errors.append(
            f"{context}: {array_kind} shape mismatch for {path}: manifest={expected_shape}, actual={shape}"
        )

    if array_kind == "rgb":
        if array.ndim != 3 or array.shape[2] < 3:
            errors.append(f"{context}: RGB array must be HxWxC with at least 3 channels, got {shape}")
    elif array_kind == "depth":
        if array.ndim != 2:
            errors.append(f"{context}: depth array must be HxW, got {shape}")
        if summary.get("finite_count", 0) <= 0:
            errors.append(f"{context}: depth array has no finite values: {path}")

    if array.size == 0:
        warnings.append(f"{context}: {array_kind} array is empty: {path}")
    return summary


def _append_if_missing(mapping: dict[str, Any], field: str, errors: list[str], context: str) -> None:
    if field not in mapping:
        errors.append(f"{context}: missing required field {field!r}")


def _require_object_fields(
    obj: dict[str, Any],
    *,
    sample_id: str,
    object_index: int,
    errors: list[str],
) -> None:
    context = f"{sample_id}: labels.objects[{object_index}]"
    for field in (
        "object_id",
        "prim_path",
        "class",
        "raw_class",
        "world_pose",
        "base_frame_pose",
        "table_frame_pose",
        "bbox_world",
        "target_bin",
        "visibility",
    ):
        _append_if_missing(obj, field, errors, context)
    table_pose = obj.get("table_frame_pose")
    if isinstance(table_pose, dict):
        for field in ("x", "y", "z", "yaw_rad", "coarse_orientation"):
            _append_if_missing(table_pose, field, errors, f"{context}.table_frame_pose")
    world_pose = obj.get("world_pose")
    if isinstance(world_pose, dict):
        for field in ("position_xyz_m", "orientation_xyzw", "yaw_rad", "coarse_orientation"):
            _append_if_missing(world_pose, field, errors, f"{context}.world_pose")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if value) / len(values))


def _truth_table_position(obj: dict[str, Any]) -> list[float] | None:
    pose = obj.get("table_frame_pose")
    if not isinstance(pose, dict):
        return None
    explicit = pose.get("position_xyz_m")
    if isinstance(explicit, list) and len(explicit) >= 3:
        values = [_finite_float(value) for value in explicit[:3]]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    xyz = [_finite_float(pose.get(axis)) for axis in ("x", "y", "z")]
    if all(value is not None for value in xyz):
        return [float(value) for value in xyz if value is not None]
    return None


def _prediction_position(obj: dict[str, Any]) -> list[float] | None:
    for field in (
        "table_frame_position_m",
        "position_xyz_m",
        "position_m",
        "center_3d_m",
        "xyz_m",
        "predicted_position_xyz_m",
    ):
        value = obj.get(field)
        if isinstance(value, list) and len(value) >= 3:
            values = [_finite_float(item) for item in value[:3]]
            if all(item is not None for item in values):
                return [float(item) for item in values if item is not None]

    for nested_field in ("table_frame_pose", "geometry", "prediction"):
        nested = obj.get(nested_field)
        if isinstance(nested, dict):
            nested_position = _prediction_position(nested)
            if nested_position is not None:
                return nested_position
            xyz = [_finite_float(nested.get(axis)) for axis in ("x", "y", "z")]
            if all(value is not None for value in xyz):
                return [float(value) for value in xyz if value is not None]
    return None


def _object_center_2d(obj: dict[str, Any]) -> list[float] | None:
    center = obj.get("center_2d")
    if isinstance(center, list) and len(center) >= 2:
        values = [_finite_float(item) for item in center[:2]]
        if all(value is not None for value in values):
            return [float(value) for value in values if value is not None]
    roi = obj.get("roi")
    if isinstance(roi, list) and len(roi) >= 4:
        values = [_finite_float(item) for item in roi[:4]]
        if all(value is not None for value in values):
            x1, y1, x2, y2 = [float(value) for value in values if value is not None]
            return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]
    for nested_field in ("geometry", "prediction"):
        nested = obj.get(nested_field)
        if isinstance(nested, dict):
            nested_center = _object_center_2d(nested)
            if nested_center is not None:
                return nested_center
    return None


def _truth_center_2d(obj: dict[str, Any], camera_name: str) -> list[float] | None:
    visibility = obj.get("visibility")
    if not isinstance(visibility, dict):
        return None
    per_camera = visibility.get("per_camera")
    if not isinstance(per_camera, dict):
        return None
    camera_record = per_camera.get(camera_name)
    if not isinstance(camera_record, dict):
        return None
    for field in ("uv_px", "center_2d_px"):
        value = camera_record.get(field)
        if isinstance(value, list) and len(value) >= 2:
            values = [_finite_float(item) for item in value[:2]]
            if all(item is not None for item in values):
                return [float(item) for item in values if item is not None]
    bbox_projection = camera_record.get("bbox_projection")
    if isinstance(bbox_projection, dict):
        value = bbox_projection.get("center_2d_px")
        if isinstance(value, list) and len(value) >= 2:
            values = [_finite_float(item) for item in value[:2]]
            if all(item is not None for item in values):
                return [float(item) for item in values if item is not None]
    return None


def _truth_selected_object_id(metadata: dict[str, Any]) -> str | None:
    for field in ("selected_object_id", "chosen_object_id"):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _prediction_selected_object_id(prediction: dict[str, Any]) -> str | None:
    for field in ("selected_object_id", "chosen_object_id", "target_object_id"):
        value = prediction.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _truth_arm(metadata: dict[str, Any]) -> str | None:
    for field in ("selected_arm", "chosen_arm"):
        value = metadata.get(field)
        if isinstance(value, str) and value in {"left", "right"}:
            return value
    return None


def _truth_preset(metadata: dict[str, Any]) -> str | None:
    for field in ("selected_preset", "chosen_preset"):
        value = metadata.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _prediction_execution_bool(prediction: dict[str, Any]) -> bool | None:
    for field in ("task_success", "success", "succeeded"):
        value = prediction.get(field)
        if isinstance(value, bool):
            return value
    result = prediction.get("execution_result")
    if isinstance(result, str):
        normalized = result.strip().lower()
        if normalized in COLLECTOR_ONLY_RESULTS:
            return None
        if normalized in SUCCESS_EXECUTION_RESULTS:
            return True
        if normalized in FAILURE_EXECUTION_RESULTS:
            return False
    return None


def _prediction_bool(prediction: dict[str, Any], fields: tuple[str, ...]) -> bool | None:
    for field in fields:
        value = prediction.get(field)
        if isinstance(value, bool):
            return value
    return None


def _prediction_cycle_time(prediction: dict[str, Any]) -> float | None:
    for field in ("cycle_time_s", "elapsed_s", "duration_s"):
        value = _finite_float(prediction.get(field))
        if value is not None:
            return value
    return None


def _distance(values_a: list[float], values_b: list[float]) -> float:
    return float(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(values_a, values_b))))


def _normalize_sample_prediction(record: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    sample_id = record.get("sample_id") or record.get("frame_id") or record.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        return None
    normalized = dict(record)
    normalized["sample_id"] = sample_id
    normalized.setdefault("_prediction_sources", [])
    if source not in normalized["_prediction_sources"]:
        normalized["_prediction_sources"].append(source)
    return normalized


def _payload_to_prediction_records(payload: Any, *, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                normalized = _normalize_sample_prediction(item, source=source)
                if normalized is not None:
                    records.append(normalized)
        return records

    if not isinstance(payload, dict):
        return records

    for key in ("samples", "predictions", "frames", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return _payload_to_prediction_records(value, source=source)

    if isinstance(payload.get("objects"), list) and (
        isinstance(payload.get("sample_id"), str) or isinstance(payload.get("frame_id"), str)
    ):
        normalized = _normalize_sample_prediction(payload, source=source)
        return [normalized] if normalized is not None else []

    # Accept mapping form: {"sample_000000": {"objects": [...]}, ...}
    for key, value in payload.items():
        if isinstance(key, str) and key.startswith("sample_") and isinstance(value, dict):
            item = dict(value)
            item.setdefault("sample_id", key)
            normalized = _normalize_sample_prediction(item, source=source)
            if normalized is not None:
                records.append(normalized)
    return records


def _merge_objects(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(item) for item in existing if isinstance(item, dict)]
    id_to_index = {
        str(item.get("object_id")): index
        for index, item in enumerate(merged)
        if isinstance(item.get("object_id"), str)
    }
    for item in incoming:
        if not isinstance(item, dict):
            continue
        object_id = item.get("object_id")
        if isinstance(object_id, str) and object_id in id_to_index:
            merged[id_to_index[object_id]].update(item)
        else:
            merged.append(dict(item))
    return merged


def _merge_prediction_record(
    prediction_by_sample: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    sample_id = record["sample_id"]
    existing = prediction_by_sample.setdefault(sample_id, {"sample_id": sample_id})
    incoming_sources = record.get("_prediction_sources", [])
    existing_sources = existing.setdefault("_prediction_sources", [])
    for source in incoming_sources:
        if source not in existing_sources:
            existing_sources.append(source)

    if isinstance(record.get("objects"), list):
        existing["objects"] = _merge_objects(existing.get("objects", []), record["objects"])

    for key, value in record.items():
        if key in {"objects", "sample_id", "_prediction_sources"}:
            continue
        if value is not None:
            existing[key] = value


def _load_prediction_bundle(
    paths: dict[str, Path | None],
    *,
    warnings: list[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    prediction_by_sample: dict[str, dict[str, Any]] = {}
    loaded_sources: list[str] = []
    for source, path in paths.items():
        if path is None:
            continue
        if not path.exists():
            warnings.append(f"prediction source {source!r} does not exist: {path}")
            continue
        try:
            payload = _load_optional_json_or_jsonl(path)
        except Exception as exc:
            warnings.append(f"prediction source {source!r} could not be loaded from {path}: {exc}")
            continue
        records = _payload_to_prediction_records(payload, source=source)
        if not records:
            warnings.append(f"prediction source {source!r} had no sample prediction records: {path}")
            continue
        for record in records:
            _merge_prediction_record(prediction_by_sample, record)
        loaded_sources.append(f"{source}:{path}")
    return prediction_by_sample, loaded_sources


def _validate_dataset(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not dataset_root.exists():
        raise EvaluationError(f"dataset root does not exist: {dataset_root}")
    if not manifest_path.exists():
        raise EvaluationError(f"manifest does not exist: {manifest_path}")

    run_metadata_path = dataset_root / "run_metadata.json"
    run_metadata: dict[str, Any] | None = None
    if run_metadata_path.exists():
        try:
            payload = _read_json(run_metadata_path)
            if isinstance(payload, dict):
                run_metadata = payload
            else:
                errors.append(f"run_metadata.json is not a JSON object: {run_metadata_path}")
        except Exception as exc:
            errors.append(f"failed to read run_metadata.json: {exc}")
    else:
        warnings.append(f"missing run_metadata.json: {run_metadata_path}")

    manifest_entries = _read_jsonl(manifest_path)
    if not manifest_entries:
        errors.append(f"manifest has no sample entries: {manifest_path}")

    sample_records: list[dict[str, Any]] = []
    depth_finite_by_camera: dict[str, list[int]] = {camera: [] for camera in CAMERA_NAMES}
    object_count_values: list[int] = []
    camera_complete_sample_count = 0
    rgb_array_count = 0
    depth_array_count = 0
    label_file_count = 0
    metadata_file_count = 0
    sync_debug_file_count = 0

    for sample_index, entry in enumerate(manifest_entries):
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            sample_id = f"manifest_index_{sample_index}"
            errors.append(f"manifest entry {sample_index}: missing sample_id")
        context = sample_id
        cameras = entry.get("cameras")
        if not isinstance(cameras, dict):
            errors.append(f"{context}: manifest.cameras must be an object")
            cameras = {}

        sample_camera_complete = True
        loaded_camera_summaries: dict[str, Any] = {}
        for camera_name in CAMERA_NAMES:
            camera_record = cameras.get(camera_name)
            if not isinstance(camera_record, dict):
                errors.append(f"{context}: missing camera record {camera_name}")
                sample_camera_complete = False
                continue
            rgb_record = camera_record.get("rgb") if isinstance(camera_record.get("rgb"), dict) else {}
            depth_record = camera_record.get("depth") if isinstance(camera_record.get("depth"), dict) else {}

            rgb_path = _resolve_existing_or_candidate(
                rgb_record.get("path") if isinstance(rgb_record.get("path"), str) else None,
                dataset_root=dataset_root,
            )
            depth_path = _resolve_existing_or_candidate(
                depth_record.get("path") if isinstance(depth_record.get("path"), str) else None,
                dataset_root=dataset_root,
            )

            if rgb_path is None:
                errors.append(f"{context}:{camera_name}: missing RGB path in manifest")
                sample_camera_complete = False
                rgb_summary = None
            else:
                rgb_summary = _load_array_for_validation(
                    rgb_path,
                    expected_shape=rgb_record.get("shape") if isinstance(rgb_record.get("shape"), list) else None,
                    array_kind="rgb",
                    errors=errors,
                    warnings=warnings,
                    context=f"{context}:{camera_name}",
                )
                if rgb_summary is not None:
                    rgb_array_count += 1

            if depth_path is None:
                errors.append(f"{context}:{camera_name}: missing depth path in manifest")
                sample_camera_complete = False
                depth_summary = None
            else:
                depth_summary = _load_array_for_validation(
                    depth_path,
                    expected_shape=depth_record.get("shape") if isinstance(depth_record.get("shape"), list) else None,
                    array_kind="depth",
                    errors=errors,
                    warnings=warnings,
                    context=f"{context}:{camera_name}",
                )
                if depth_summary is not None:
                    depth_array_count += 1
                    finite_count = depth_summary.get("finite_count")
                    if isinstance(finite_count, int):
                        depth_finite_by_camera[camera_name].append(finite_count)

            if rgb_summary is not None and depth_summary is not None:
                rgb_hw = rgb_summary["shape"][:2]
                depth_hw = depth_summary["shape"][:2]
                if rgb_hw != depth_hw:
                    errors.append(f"{context}:{camera_name}: RGB/depth shape mismatch: rgb={rgb_hw}, depth={depth_hw}")

            loaded_camera_summaries[camera_name] = {
                "rgb": rgb_summary,
                "depth": depth_summary,
            }

        if sample_camera_complete:
            camera_complete_sample_count += 1

        paths = entry.get("paths")
        if not isinstance(paths, dict):
            errors.append(f"{context}: manifest.paths must be an object")
            paths = {}

        labels_path = _resolve_existing_or_candidate(paths.get("labels"), dataset_root=dataset_root)
        metadata_path = _resolve_existing_or_candidate(paths.get("metadata"), dataset_root=dataset_root)
        sync_debug_path = _resolve_existing_or_candidate(paths.get("sync_debug"), dataset_root=dataset_root)

        labels: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        sync_debug: dict[str, Any] = {}

        if labels_path is None or not labels_path.exists():
            errors.append(f"{context}: missing labels file: {labels_path}")
        else:
            label_file_count += 1
            try:
                payload = _read_json(labels_path)
                if isinstance(payload, dict):
                    labels = payload
                else:
                    errors.append(f"{context}: labels file is not a JSON object: {labels_path}")
            except Exception as exc:
                errors.append(f"{context}: failed to read labels file {labels_path}: {exc}")

        if metadata_path is None or not metadata_path.exists():
            errors.append(f"{context}: missing metadata file: {metadata_path}")
        else:
            metadata_file_count += 1
            try:
                payload = _read_json(metadata_path)
                if isinstance(payload, dict):
                    metadata = payload
                else:
                    errors.append(f"{context}: metadata file is not a JSON object: {metadata_path}")
            except Exception as exc:
                errors.append(f"{context}: failed to read metadata file {metadata_path}: {exc}")

        if sync_debug_path is None or not sync_debug_path.exists():
            errors.append(f"{context}: missing sync debug file: {sync_debug_path}")
        else:
            sync_debug_file_count += 1
            try:
                payload = _read_json(sync_debug_path)
                if isinstance(payload, dict):
                    sync_debug = payload
                else:
                    errors.append(f"{context}: sync debug file is not a JSON object: {sync_debug_path}")
            except Exception as exc:
                errors.append(f"{context}: failed to read sync debug file {sync_debug_path}: {exc}")

        if labels:
            for field in ("object_count", "objects", "label_policy"):
                _append_if_missing(labels, field, errors, f"{context}: labels")
            objects = labels.get("objects")
            if not isinstance(objects, list):
                errors.append(f"{context}: labels.objects must be a list")
                objects = []
            for object_index, obj in enumerate(objects):
                if isinstance(obj, dict):
                    _require_object_fields(obj, sample_id=context, object_index=object_index, errors=errors)
                else:
                    errors.append(f"{context}: labels.objects[{object_index}] is not an object")

            label_object_count = labels.get("object_count")
            manifest_object_count = entry.get("object_count")
            if isinstance(label_object_count, int):
                object_count_values.append(label_object_count)
            if isinstance(manifest_object_count, int) and isinstance(label_object_count, int):
                if manifest_object_count != label_object_count:
                    errors.append(
                        f"{context}: manifest.object_count={manifest_object_count} "
                        f"!= labels.object_count={label_object_count}"
                    )
            if isinstance(label_object_count, int) and label_object_count != len(objects):
                errors.append(f"{context}: labels.object_count={label_object_count} != len(objects)={len(objects)}")

        if metadata:
            for field in (
                "sample_id",
                "sample_index",
                "chosen_object_id",
                "chosen_arm",
                "chosen_preset",
                "chosen_candidate",
                "planner_target",
                "execution_result",
                "fail_reason",
                "simulation_step",
                "sim_time_estimate_s",
                "timestamp_utc",
                "collector_only",
            ):
                _append_if_missing(metadata, field, errors, f"{context}: metadata")
            if metadata.get("sample_id") != entry.get("sample_id"):
                errors.append(f"{context}: metadata.sample_id does not match manifest.sample_id")

        if sync_debug:
            for field in (
                "capture_step",
                "same_simulation_step_for_all_cameras",
                "camera_capture_order",
                "camera_summaries",
            ):
                _append_if_missing(sync_debug, field, errors, f"{context}: sync_debug")
            if sync_debug.get("same_simulation_step_for_all_cameras") is not True:
                errors.append(f"{context}: sync_debug.same_simulation_step_for_all_cameras is not true")
            capture_order = sync_debug.get("camera_capture_order")
            if isinstance(capture_order, list) and capture_order != list(CAMERA_NAMES):
                warnings.append(f"{context}: sync_debug.camera_capture_order differs from expected camera order")

        sample_records.append(
            {
                "sample_id": sample_id,
                "manifest": entry,
                "labels": labels,
                "metadata": metadata,
                "sync_debug": sync_debug,
                "camera_summaries": loaded_camera_summaries,
            }
        )

    depth_summary = {}
    for camera_name, values in depth_finite_by_camera.items():
        depth_summary[camera_name] = {
            "sample_count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": _mean([float(value) for value in values]),
            "total": int(sum(values)) if values else 0,
        }

    return {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "run_metadata": run_metadata,
        "samples": sample_records,
        "summary": {
            "sample_count": len(manifest_entries),
            "camera_expected_count": len(CAMERA_NAMES),
            "camera_complete_sample_count": camera_complete_sample_count,
            "rgb_array_count": rgb_array_count,
            "depth_array_count": depth_array_count,
            "label_file_count": label_file_count,
            "metadata_file_count": metadata_file_count,
            "sync_debug_file_count": sync_debug_file_count,
            "depth_finite_count_by_camera": depth_summary,
            "object_count_min": min(object_count_values) if object_count_values else None,
            "object_count_max": max(object_count_values) if object_count_values else None,
            "object_count_values": sorted(set(object_count_values)),
        },
        "errors": errors,
        "warnings": warnings,
    }


def _match_prediction_objects(
    pred_objects: list[Any],
    truth_objects: list[Any],
    *,
    sample_id: str,
    warnings: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    truth_dicts = [obj for obj in truth_objects if isinstance(obj, dict)]
    truth_by_id = {
        str(obj.get("object_id")): obj
        for obj in truth_dicts
        if isinstance(obj.get("object_id"), str)
    }
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_truth_ids: set[str] = set()
    for index, pred_obj in enumerate(pred_objects):
        if not isinstance(pred_obj, dict):
            continue
        object_id = pred_obj.get("object_id") or pred_obj.get("truth_object_id")
        truth_obj = truth_by_id.get(str(object_id)) if object_id is not None else None
        if truth_obj is None and object_id is None and index < len(truth_dicts):
            truth_obj = truth_dicts[index]
            warnings.append(f"{sample_id}: prediction object {index} matched by order because object_id is absent")
        if truth_obj is None:
            warnings.append(f"{sample_id}: could not match prediction object {index} to truth object_id={object_id!r}")
            continue
        truth_id = str(truth_obj.get("object_id"))
        if truth_id in used_truth_ids:
            warnings.append(f"{sample_id}: duplicate prediction match for truth object_id={truth_id}")
            continue
        used_truth_ids.add(truth_id)
        matched.append((pred_obj, truth_obj))
    return matched


def _evaluate_predictions(
    validation: dict[str, Any],
    predictions_by_sample: dict[str, dict[str, Any]],
    *,
    metric_camera: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    metric_values: dict[str, list[float | bool]] = {name: [] for name in METRIC_NAMES}

    for sample in validation["samples"]:
        sample_id = sample["sample_id"]
        prediction = predictions_by_sample.get(sample_id)
        labels = sample["labels"]
        metadata = sample["metadata"]
        if not prediction:
            continue

        truth_objects = labels.get("objects") if isinstance(labels.get("objects"), list) else []
        pred_objects = prediction.get("objects") if isinstance(prediction.get("objects"), list) else []
        matched_objects = _match_prediction_objects(
            pred_objects,
            truth_objects,
            sample_id=sample_id,
            warnings=warnings,
        )

        for pred_obj, truth_obj in matched_objects:
            pred_class = pred_obj.get("class")
            truth_class = truth_obj.get("class")
            if isinstance(pred_class, str) and isinstance(truth_class, str):
                metric_values["class_accuracy"].append(pred_class == truth_class)

            pred_bucket = pred_obj.get("orientation_bucket") or pred_obj.get("coarse_orientation")
            truth_pose = truth_obj.get("table_frame_pose")
            truth_bucket = truth_pose.get("coarse_orientation") if isinstance(truth_pose, dict) else None
            if isinstance(pred_bucket, str) and isinstance(truth_bucket, str):
                metric_values["yaw_bucket_accuracy"].append(pred_bucket == truth_bucket)

            center_camera = pred_obj.get("camera_name") if isinstance(pred_obj.get("camera_name"), str) else metric_camera
            pred_center = _object_center_2d(pred_obj)
            truth_center = _truth_center_2d(truth_obj, center_camera)
            if pred_center is not None and truth_center is not None:
                metric_values["center_2d_error_px"].append(_distance(pred_center, truth_center))

            pred_position = _prediction_position(pred_obj)
            truth_position = _truth_table_position(truth_obj)
            if pred_position is not None and truth_position is not None:
                metric_values["conversion_3d_error_m"].append(_distance(pred_position, truth_position))

            truth_arm = _truth_arm(metadata)
            pred_arm = pred_obj.get("recommended_arm")
            if truth_arm is not None and isinstance(pred_arm, str):
                metric_values["arm_recommendation_accuracy"].append(pred_arm == truth_arm)

            truth_preset = _truth_preset(metadata)
            pred_preset = pred_obj.get("recommended_preset")
            if truth_preset is not None and isinstance(pred_preset, str):
                metric_values["preset_recommendation_accuracy"].append(pred_preset == truth_preset)

        truth_selected = _truth_selected_object_id(metadata)
        pred_selected = _prediction_selected_object_id(prediction)
        if truth_selected is not None and pred_selected is not None:
            metric_values["selected_object_accuracy"].append(pred_selected == truth_selected)

        success = _prediction_execution_bool(prediction)
        if success is not None:
            metric_values["task_success_rate"].append(success)

        wrong_bin = _prediction_bool(prediction, ("wrong_bin", "wrong_bin_result"))
        if wrong_bin is not None:
            metric_values["wrong_bin_rate"].append(wrong_bin)

        dropped = _prediction_bool(prediction, ("drop", "dropped", "object_dropped"))
        if dropped is not None:
            metric_values["drop_rate"].append(dropped)

        cycle_time = _prediction_cycle_time(prediction)
        if cycle_time is not None:
            metric_values["cycle_time_s"].append(cycle_time)

    metrics: dict[str, float | None] = {}
    metric_counts: dict[str, int] = {}
    for name, values in metric_values.items():
        metric_counts[name] = len(values)
        if not values:
            metrics[name] = None
        elif name.endswith("_accuracy") or name.endswith("_rate"):
            metrics[name] = _rate([bool(value) for value in values])
        else:
            metrics[name] = _mean([float(value) for value in values])

    pending_metrics = [name for name, value in metrics.items() if value is None]
    return {
        "metrics": metrics,
        "metric_counts": metric_counts,
        "pending_metrics": pending_metrics,
        "warnings": warnings,
    }


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, str, dict[str, Path | None], dict[str, Any] | None]:
    evaluator_io: dict[str, Any] | None = None
    evaluator_io_path = _as_path(args.evaluator_io)
    evaluator_base_dir = evaluator_io_path.parent if evaluator_io_path is not None else None

    if evaluator_io_path is not None:
        evaluator_io = _load_evaluator_io(evaluator_io_path)
        io_inputs = evaluator_io["inputs"]
    else:
        io_inputs = {}

    run_id = args.run_id or (evaluator_io.get("run_id") if evaluator_io is not None else None)

    dataset_root: Path | None = _as_path(args.dataset_root)
    if dataset_root is None and isinstance(io_inputs.get("dataset_root"), str):
        dataset_root = _resolve_existing_or_candidate(
            io_inputs.get("dataset_root"),
            dataset_root=None,
            base_dir=evaluator_base_dir,
        )
    if dataset_root is None:
        if not run_id:
            raise EvaluationError("Provide --dataset-root, --run-id, or --evaluator-io with inputs.dataset_root")
        dataset_root = _default_dataset_root_for_run(str(run_id))
    dataset_root = dataset_root.resolve()

    manifest_path = _as_path(args.manifest)
    if manifest_path is None and isinstance(io_inputs.get("manifest_path"), str):
        manifest_path = _resolve_existing_or_candidate(
            io_inputs.get("manifest_path"),
            dataset_root=dataset_root,
            base_dir=evaluator_base_dir,
        )
    if manifest_path is None:
        manifest_path = dataset_root / "manifest.jsonl"
    manifest_path = manifest_path.resolve()

    if run_id is None:
        run_id = dataset_root.name

    prediction_paths: dict[str, Path | None] = {
        "predictions": _as_path(args.predictions),
        "thinker_output": _as_path(args.thinker_output),
        "geometry_output": _as_path(args.geometry_output),
        "planner_trace": _as_path(args.planner_trace),
        "execution_log": _as_path(args.execution_log),
    }
    io_field_to_source = {
        "thinker_output_path": "thinker_output",
        "geometry_output_path": "geometry_output",
        "planner_trace_path": "planner_trace",
        "execution_log_path": "execution_log",
    }
    for io_field, source in io_field_to_source.items():
        if prediction_paths[source] is None and isinstance(io_inputs.get(io_field), str):
            prediction_paths[source] = _resolve_existing_or_candidate(
                io_inputs.get(io_field),
                dataset_root=dataset_root,
                base_dir=evaluator_base_dir,
            )
    prediction_paths = {
        key: (value.resolve() if value is not None else None)
        for key, value in prediction_paths.items()
    }
    return dataset_root, manifest_path, str(run_id), prediction_paths, evaluator_io


def _make_notes(
    *,
    validation: dict[str, Any],
    prediction_eval: dict[str, Any],
    loaded_prediction_sources: list[str],
) -> str:
    summary = validation["summary"]
    structural_status = "pass" if not validation["errors"] else "fail"
    prediction_status = "available" if loaded_prediction_sources else "not_provided"
    pending = prediction_eval["pending_metrics"]
    return (
        f"structural_status={structural_status}; "
        f"samples={summary['sample_count']}; "
        f"camera_complete_samples={summary['camera_complete_sample_count']}; "
        f"rgb_arrays={summary['rgb_array_count']}; "
        f"depth_arrays={summary['depth_array_count']}; "
        f"prediction_status={prediction_status}; "
        f"pending_metrics={','.join(pending) if pending else 'none'}"
    )


def _build_report(
    *,
    dataset_root: Path,
    manifest_path: Path,
    run_id: str,
    prediction_paths: dict[str, Path | None],
    validation: dict[str, Any],
    prediction_eval: dict[str, Any],
    loaded_prediction_sources: list[str],
) -> dict[str, Any]:
    structural_errors = validation["errors"]
    status = "fail" if structural_errors else ("pass" if loaded_prediction_sources else "skip")
    failure_reason = None
    if structural_errors:
        failure_reason = "structural_dataset_validation_failed"
    elif not loaded_prediction_sources:
        failure_reason = "prediction_inputs_not_provided"

    inputs = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "thinker_output_path": str(prediction_paths["thinker_output"]) if prediction_paths["thinker_output"] else None,
        "geometry_output_path": str(prediction_paths["geometry_output"]) if prediction_paths["geometry_output"] else None,
        "planner_trace_path": str(prediction_paths["planner_trace"]) if prediction_paths["planner_trace"] else None,
        "execution_log_path": str(prediction_paths["execution_log"]) if prediction_paths["execution_log"] else None,
    }
    report = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "sample_id": None,
        "inputs": inputs,
        "outputs": {
            "status": status,
            "metrics": prediction_eval["metrics"],
            "failure_reason": failure_reason,
            "notes": _make_notes(
                validation=validation,
                prediction_eval=prediction_eval,
                loaded_prediction_sources=loaded_prediction_sources,
            ),
        },
        "structural_validation": {
            **validation["summary"],
            "status": "pass" if not structural_errors else "fail",
            "error_count": len(validation["errors"]),
            "warning_count": len(validation["warnings"]) + len(prediction_eval["warnings"]),
            "errors": validation["errors"],
            "warnings": validation["warnings"] + prediction_eval["warnings"],
        },
        "prediction_evaluation": {
            "loaded_prediction_sources": loaded_prediction_sources,
            "metric_counts": prediction_eval["metric_counts"],
            "pending_metrics": prediction_eval["pending_metrics"],
        },
    }
    prediction_path = prediction_paths.get("predictions")
    if prediction_path is not None:
        report["inputs"]["prediction_path"] = str(prediction_path)
    return report


def _print_report(report: dict[str, Any], *, max_messages: int) -> None:
    structural = report["structural_validation"]
    outputs = report["outputs"]
    prediction = report["prediction_evaluation"]
    print("Task 1 dataset evaluator")
    print(f"Dataset root: {report['inputs']['dataset_root']}")
    print(f"Manifest: {report['inputs']['manifest_path']}")
    print(
        "Structural: "
        f"{structural['status'].upper()} "
        f"(samples={structural['sample_count']}, "
        f"errors={structural['error_count']}, "
        f"warnings={structural['warning_count']})"
    )
    print(
        "Files: "
        f"rgb_arrays={structural['rgb_array_count']}, "
        f"depth_arrays={structural['depth_array_count']}, "
        f"labels={structural['label_file_count']}, "
        f"metadata={structural['metadata_file_count']}, "
        f"sync_debug={structural['sync_debug_file_count']}"
    )
    print(
        "Cameras: "
        f"complete_samples={structural['camera_complete_sample_count']}/{structural['sample_count']}, "
        f"expected_per_sample={structural['camera_expected_count']}"
    )
    print("Depth finite_count by camera:")
    for camera_name in CAMERA_NAMES:
        camera_summary = structural["depth_finite_count_by_camera"].get(camera_name, {})
        print(
            f"  {camera_name}: "
            f"samples={camera_summary.get('sample_count')}, "
            f"min={camera_summary.get('min')}, "
            f"max={camera_summary.get('max')}, "
            f"mean={camera_summary.get('mean')}"
        )
    print(
        "Objects: "
        f"object_count_values={structural['object_count_values']}, "
        f"min={structural['object_count_min']}, "
        f"max={structural['object_count_max']}"
    )

    loaded_sources = prediction["loaded_prediction_sources"]
    if loaded_sources:
        print("Prediction sources:")
        for source in loaded_sources:
            print(f"  {source}")
    else:
        print("Prediction sources: none provided")

    print("Metrics:")
    metrics = outputs["metrics"]
    counts = prediction["metric_counts"]
    for metric_name in METRIC_NAMES:
        value = metrics.get(metric_name)
        status = "pending" if value is None else value
        print(f"  {metric_name}: {status} (n={counts.get(metric_name, 0)})")

    for label, messages in (("Errors", structural["errors"]), ("Warnings", structural["warnings"])):
        if not messages:
            continue
        print(f"{label}:")
        for message in messages[:max_messages]:
            print(f"  {message}")
        remaining = len(messages) - max_messages
        if remaining > 0:
            print(f"  ... {remaining} more")
    print(f"Evaluator status: {outputs['status']}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", help="Path to one Phase 1 run directory. Defaults to $OUTPUT_ROOT/datasets/task1_rgbd_labels/<run-id>.")
    parser.add_argument("--run-id", help="Run id under $OUTPUT_ROOT/datasets/task1_rgbd_labels.")
    parser.add_argument("--manifest", help="Manifest path. Defaults to <dataset-root>/manifest.jsonl.")
    parser.add_argument("--evaluator-io", help="Optional evaluator I/O JSON using docs/schemas/task1_evaluator_io.schema.json.")
    parser.add_argument("--predictions", help="Optional direct prediction JSON/JSONL. Supports Thinker-style frame records or {samples:[...]} payloads.")
    parser.add_argument("--thinker-output", help="Optional Thinker structured output JSON/JSONL.")
    parser.add_argument("--geometry-output", help="Optional geometry output JSON/JSONL with table-frame object positions.")
    parser.add_argument("--planner-trace", help="Optional planner trace JSON/JSONL with selected object/arm/preset fields.")
    parser.add_argument("--execution-log", help="Optional execution result JSON/JSONL with task success, wrong-bin/drop, or cycle-time fields.")
    parser.add_argument("--metric-camera", default="head_left", choices=CAMERA_NAMES, help="Camera used for 2D center truth lookup when predictions do not specify camera_name.")
    parser.add_argument("--report", help="Optional JSON summary report path. Keep this under OUTPUT_ROOT/LOG_ROOT, not the repo.")
    parser.add_argument("--max-messages", type=int, default=20, help="Maximum errors/warnings printed to terminal.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when structural validation fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        dataset_root, manifest_path, run_id, prediction_paths, _evaluator_io = _resolve_inputs(args)
        validation = _validate_dataset(dataset_root, manifest_path)
        prediction_warnings: list[str] = []
        predictions_by_sample, loaded_prediction_sources = _load_prediction_bundle(
            prediction_paths,
            warnings=prediction_warnings,
        )
        validation["warnings"].extend(prediction_warnings)
        prediction_eval = _evaluate_predictions(
            validation,
            predictions_by_sample,
            metric_camera=args.metric_camera,
        )
        report = _build_report(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            run_id=run_id,
            prediction_paths=prediction_paths,
            validation=validation,
            prediction_eval=prediction_eval,
            loaded_prediction_sources=loaded_prediction_sources,
        )
        _print_report(report, max_messages=max(0, args.max_messages))
        if args.report:
            report_path = Path(args.report).expanduser()
            _write_json(report_path, report)
            print(f"Report saved: {report_path}")
        if args.strict and validation["errors"]:
            return 1
        return 0
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
