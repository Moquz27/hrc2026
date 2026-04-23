#!/usr/bin/env python3
"""Run a 10-case input-correction evaluation for Task 1.

This script evaluates whether AI output improves input fields before any full
runtime integration. It does not run the robot, generate grasp poses, call
DualArmIK, or modify planner/execution code.
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

from task1_input_correction import CorrectionConfig, apply_input_corrections


CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")
INPUT_FORMAT_VERSION = "0.1.0"
OUTPUT_FORMAT_VERSION = "0.1.0"
DEFAULT_LIMIT = 10


class InputCorrectionEvalError(RuntimeError):
    """Raised for configuration or input-format errors."""


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
                raise InputCorrectionEvalError(f"{path}: invalid JSONL line {line_index}: {exc}") from exc
            if not isinstance(record, dict):
                raise InputCorrectionEvalError(f"{path}: JSONL line {line_index} is not an object")
            records.append(record)
    return records


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def _dataset_root_from_args(args: argparse.Namespace) -> Path | None:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    if args.run_id:
        output_root = os.environ.get("OUTPUT_ROOT")
        if not output_root:
            raise InputCorrectionEvalError("OUTPUT_ROOT is required when --run-id is used without --dataset-root")
        return (Path(output_root) / "datasets" / "task1_rgbd_labels" / args.run_id).resolve()
    return None


def _default_output_dir(run_label: str) -> Path:
    output_root = os.environ.get("OUTPUT_ROOT")
    if not output_root:
        raise InputCorrectionEvalError("OUTPUT_ROOT is required when --output-dir is omitted")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(output_root) / "test_runs" / "task1_input_correction_eval" / f"{run_label}_{timestamp}"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _finite_vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < length:
        return None
    values = [_finite_float(item) for item in value[:length]]
    if any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _distance_2d(a: list[float], b: list[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _roi_center(roi: list[float]) -> list[float]:
    return [(float(roi[0]) + float(roi[2])) * 0.5, (float(roi[1]) + float(roi[3])) * 0.5]


def _center_from_record(record: dict[str, Any]) -> list[float] | None:
    center = _finite_vector(record.get("center_2d"), 2)
    if center is not None:
        return center
    roi = _finite_vector(record.get("roi"), 4)
    if roi is not None:
        return _roi_center(roi)
    return None


def _object_id(record: dict[str, Any]) -> str | None:
    for field in ("selected_object_id", "object_id", "chosen_object_id", "target_object_id"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _class_value(record: dict[str, Any]) -> str | None:
    value = record.get("class")
    return value if isinstance(value, str) and value else None


def _orientation_value(record: dict[str, Any]) -> str | None:
    for field in ("orientation_bucket", "coarse_orientation", "yaw_bucket"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _arm_value(record: dict[str, Any]) -> str | None:
    for field in ("recommended_arm", "arm", "selected_arm"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _preset_value(record: dict[str, Any]) -> str | None:
    for field in ("recommended_preset", "preset", "selected_preset"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _safe_bool_equal(left: Any, right: Any) -> bool | None:
    if left is None or right is None:
        return None
    return left == right


def _metric_pass_bool(value: bool | None) -> bool | None:
    return value if isinstance(value, bool) else None


def _metric_pass_center(error_px: float | None, threshold_px: float) -> bool | None:
    if error_px is None:
        return None
    return error_px <= threshold_px


def _score_correctness(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _extract_truth_center_from_visibility(obj: dict[str, Any], preferred_camera: str) -> tuple[list[float] | None, str | None]:
    visibility = obj.get("visibility")
    if not isinstance(visibility, dict):
        return None, None
    per_camera = visibility.get("per_camera")
    if not isinstance(per_camera, dict):
        return None, None
    ordered_cameras = [preferred_camera] + [camera for camera in CAMERA_NAMES if camera != preferred_camera]
    for camera_name in ordered_cameras:
        camera_record = per_camera.get(camera_name)
        if not isinstance(camera_record, dict):
            continue
        for field in ("uv_px", "center_2d_px"):
            center = _finite_vector(camera_record.get(field), 2)
            if center is not None:
                return center, camera_name
        bbox_projection = camera_record.get("bbox_projection")
        if isinstance(bbox_projection, dict):
            center = _finite_vector(bbox_projection.get("center_2d_px"), 2)
            if center is not None:
                return center, camera_name
    return None, None


def _extract_truth_roi_from_visibility(obj: dict[str, Any], camera_name: str | None) -> list[float] | None:
    if camera_name is None:
        return None
    visibility = obj.get("visibility")
    if not isinstance(visibility, dict):
        return None
    per_camera = visibility.get("per_camera")
    if not isinstance(per_camera, dict):
        return None
    camera_record = per_camera.get(camera_name)
    if not isinstance(camera_record, dict):
        return None
    bbox_projection = camera_record.get("bbox_projection")
    if isinstance(bbox_projection, dict):
        return _finite_vector(bbox_projection.get("roi_2d_px"), 4)
    return None


def _reference_arm(table_x: float | None, x_extent: float | None) -> str:
    if table_x is None:
        return "right"
    split = (x_extent * 0.5) if x_extent is not None and x_extent > 0 else 0.75
    return "left" if table_x <= split else "right"


def _reference_preset(orientation_bucket: str | None) -> str:
    if orientation_bucket in {"left", "right", "front_left", "front_right"}:
        return "side_approach"
    return "topdown"


def _flip_class(class_name: str | None) -> str:
    if class_name == "A":
        return "B"
    if class_name == "B":
        return "A"
    return "unknown"


def _alternate_bucket(bucket: str | None) -> str:
    order = [
        "front",
        "front_left",
        "left",
        "back_left",
        "back",
        "back_right",
        "right",
        "front_right",
    ]
    if bucket not in order:
        return "front"
    return order[(order.index(bucket) + 2) % len(order)]


def _alternate_arm(arm: str | None) -> str:
    return "left" if arm == "right" else "right"


def _alternate_preset(preset: str | None) -> str:
    return "side_approach" if preset == "topdown" else "topdown"


def _shift_center(center: list[float] | None, dx: float, dy: float) -> list[float] | None:
    if center is None:
        return None
    return [float(center[0]) + dx, float(center[1]) + dy]


def _shift_roi(roi: list[float] | None, dx: float, dy: float) -> list[float] | None:
    if roi is None:
        return None
    return [float(roi[0]) + dx, float(roi[1]) + dy, float(roi[2]) + dx, float(roi[3]) + dy]


def _load_cases_from_file(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = _load_json_or_jsonl(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            cases = payload["cases"]
        else:
            raise InputCorrectionEvalError(f"{path}: JSON case file must contain a cases list")
    elif isinstance(payload, list):
        cases = payload
    else:
        raise InputCorrectionEvalError(f"{path}: cases input must be a JSON object or list")

    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases[:limit]):
        if not isinstance(case, dict):
            raise InputCorrectionEvalError(f"{path}: case {index} is not an object")
        for field in ("case_id", "original_input", "ai_output", "truth"):
            if field not in case:
                raise InputCorrectionEvalError(f"{path}: case {index} missing {field!r}")
        normalized.append(case)
    return normalized


def _manifest_records(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise InputCorrectionEvalError(f"manifest does not exist: {manifest_path}")
    return _read_jsonl(manifest_path)


def _load_dataset_cases(
    *,
    dataset_root: Path,
    limit: int,
    seed: int,
    metric_camera: str,
) -> list[dict[str, Any]]:
    run_metadata_path = dataset_root / "run_metadata.json"
    run_metadata = _read_json(run_metadata_path) if run_metadata_path.exists() else {}
    table_frame = {}
    if isinstance(run_metadata, dict):
        scene = run_metadata.get("scene")
        if isinstance(scene, dict) and isinstance(scene.get("table_frame"), dict):
            table_frame = scene["table_frame"]
    x_extent = _finite_float(table_frame.get("x_extent_m"))

    rng = random.Random(seed)
    manifest_entries = _manifest_records(dataset_root)
    raw_cases: list[dict[str, Any]] = []

    for entry in manifest_entries:
        sample_id = entry.get("sample_id")
        paths = entry.get("paths")
        if not isinstance(sample_id, str) or not isinstance(paths, dict):
            continue
        labels_rel = paths.get("labels")
        if not isinstance(labels_rel, str):
            continue
        labels_path = dataset_root / labels_rel
        if not labels_path.exists():
            continue
        labels = _read_json(labels_path)
        objects = labels.get("objects") if isinstance(labels, dict) else None
        if not isinstance(objects, list):
            continue
        object_ids = [obj.get("object_id") for obj in objects if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)]
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict) or not isinstance(obj.get("object_id"), str):
                continue
            table_pose = obj.get("table_frame_pose") if isinstance(obj.get("table_frame_pose"), dict) else {}
            table_x = _finite_float(table_pose.get("x"))
            truth_center, truth_camera = _extract_truth_center_from_visibility(obj, metric_camera)
            truth_roi = _extract_truth_roi_from_visibility(obj, truth_camera)
            orientation_bucket = table_pose.get("coarse_orientation")
            if not isinstance(orientation_bucket, str):
                orientation_bucket = obj.get("world_pose", {}).get("coarse_orientation") if isinstance(obj.get("world_pose"), dict) else None
            truth_arm = _reference_arm(table_x, x_extent)
            truth_preset = _reference_preset(orientation_bucket)
            truth = {
                "selected_object_id": obj["object_id"],
                "object_id": obj["object_id"],
                "class": obj.get("class"),
                "center_2d": truth_center,
                "roi": truth_roi,
                "orientation_bucket": orientation_bucket,
                "recommended_arm": truth_arm,
                "recommended_preset": truth_preset,
                "camera_name": truth_camera,
                "reference_note": "arm/preset are deterministic evaluator references, not simulator-truth control labels",
            }

            case_number = len(raw_cases)
            wrong_object_id = object_ids[(object_ids.index(obj["object_id"]) + 1) % len(object_ids)] if len(object_ids) > 1 else obj["object_id"]
            center_dx = 22.0 + float((case_number % 3) * 5)
            center_dy = -15.0 + float((case_number % 2) * 8)
            original = {
                "selected_object_id": wrong_object_id if case_number % 4 == 0 else obj["object_id"],
                "object_id": wrong_object_id if case_number % 4 == 0 else obj["object_id"],
                "class": _flip_class(obj.get("class")) if case_number % 3 == 0 else obj.get("class"),
                "center_2d": _shift_center(truth_center, center_dx, center_dy),
                "roi": _shift_roi(truth_roi, center_dx, center_dy),
                "orientation_bucket": _alternate_bucket(orientation_bucket) if case_number % 2 == 0 else orientation_bucket,
                "recommended_arm": _alternate_arm(truth_arm) if case_number % 3 == 1 else truth_arm,
                "recommended_preset": _alternate_preset(truth_preset) if case_number % 3 == 2 else truth_preset,
                "camera_name": truth_camera,
            }

            ai_center = _shift_center(truth_center, rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
            ai_roi = _shift_roi(truth_roi, rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
            ai_class = obj.get("class")
            ai_bucket = orientation_bucket
            ai_object_id = obj["object_id"]
            ai_arm = truth_arm
            ai_preset = truth_preset
            confidences = {
                "selected_object_id": 0.92,
                "class": 0.91,
                "center_2d": 0.90,
                "roi": 0.90,
                "orientation_bucket": 0.88,
                "recommended_arm": 0.87,
                "recommended_preset": 0.86,
            }

            if case_number % 7 == 3:
                ai_class = _flip_class(obj.get("class"))
                confidences["class"] = 0.93
            if case_number % 6 == 2:
                confidences["orientation_bucket"] = 0.35
            if case_number % 5 == 4:
                ai_center = _shift_center(truth_center, 140.0, -120.0)
                confidences["center_2d"] = 0.95
            if case_number % 8 == 5:
                ai_object_id = "not_a_spawned_task1_object"
                confidences["selected_object_id"] = 0.96
            if case_number % 9 == 6:
                confidences["recommended_arm"] = 0.40

            ai_output = {
                "selected_object_id": ai_object_id,
                "object_id": ai_object_id,
                "class": ai_class,
                "center_2d": ai_center,
                "roi": ai_roi,
                "orientation_bucket": ai_bucket,
                "recommended_arm": ai_arm,
                "recommended_preset": ai_preset,
                "confidences": confidences,
                "source": "synthetic_ai_output_for_input_correction_pipeline_test",
            }
            raw_cases.append(
                {
                    "format_version": INPUT_FORMAT_VERSION,
                    "case_id": f"case_{case_number:03d}",
                    "sample_id": sample_id,
                    "source": {
                        "mode": "generated_from_phase1_dataset",
                        "dataset_root": str(dataset_root),
                        "sample_id": sample_id,
                        "object_index": object_index,
                        "truth_camera": truth_camera,
                    },
                    "original_input": original,
                    "ai_output": ai_output,
                    "truth": truth,
                    "allowed_object_ids": object_ids,
                }
            )
            if len(raw_cases) >= limit:
                return raw_cases
    return raw_cases


def _evaluate_record(record: dict[str, Any], truth: dict[str, Any], center_pass_threshold_px: float) -> dict[str, Any]:
    center = _center_from_record(record)
    truth_center = _center_from_record(truth)
    center_error = _distance_2d(center, truth_center) if center is not None and truth_center is not None else None
    class_correct = _safe_bool_equal(_class_value(record), _class_value(truth))
    selected_correct = _safe_bool_equal(_object_id(record), _object_id(truth))
    orientation_correct = _safe_bool_equal(_orientation_value(record), _orientation_value(truth))
    arm_correct = _safe_bool_equal(_arm_value(record), _arm_value(truth))
    preset_correct = _safe_bool_equal(_preset_value(record), _preset_value(truth))
    return {
        "class_correct": class_correct,
        "selected_object_correct": selected_correct,
        "center_2d_error_px": center_error,
        "orientation_bucket_correct": orientation_correct,
        "arm_recommendation_correct": arm_correct,
        "preset_recommendation_correct": preset_correct,
        "pass_flags": {
            "class": _metric_pass_bool(class_correct),
            "selected_object": _metric_pass_bool(selected_correct),
            "center_2d": _metric_pass_center(center_error, center_pass_threshold_px),
            "orientation_bucket": _metric_pass_bool(orientation_correct),
            "arm_recommendation": _metric_pass_bool(arm_correct),
            "preset_recommendation": _metric_pass_bool(preset_correct),
        },
    }


def _compare_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    metric_changes: dict[str, Any] = {}
    improved = 0
    worsened = 0
    bool_fields = (
        "class_correct",
        "selected_object_correct",
        "orientation_bucket_correct",
        "arm_recommendation_correct",
        "preset_recommendation_correct",
    )
    for field in bool_fields:
        before_score = _score_correctness(before.get(field))
        after_score = _score_correctness(after.get(field))
        if before_score is None or after_score is None:
            status = "not_applicable"
        elif after_score > before_score:
            status = "improved"
            improved += 1
        elif after_score < before_score:
            status = "worsened"
            worsened += 1
        else:
            status = "unchanged"
        metric_changes[field] = {"before": before.get(field), "after": after.get(field), "status": status}

    before_center = before.get("center_2d_error_px")
    after_center = after.get("center_2d_error_px")
    if isinstance(before_center, (int, float)) and isinstance(after_center, (int, float)):
        delta = float(after_center) - float(before_center)
        if delta < -1e-6:
            center_status = "improved"
            improved += 1
        elif delta > 1e-6:
            center_status = "worsened"
            worsened += 1
        else:
            center_status = "unchanged"
        metric_changes["center_2d_error_px"] = {
            "before": float(before_center),
            "after": float(after_center),
            "delta_after_minus_before": delta,
            "status": center_status,
        }
    else:
        metric_changes["center_2d_error_px"] = {
            "before": before_center,
            "after": after_center,
            "delta_after_minus_before": None,
            "status": "not_applicable",
        }

    if improved > worsened:
        case_status = "improved"
    elif worsened > improved:
        case_status = "worsened"
    else:
        case_status = "unchanged"
    return {
        "metric_changes": metric_changes,
        "improved_metric_count": improved,
        "worsened_metric_count": worsened,
        "case_outcome": case_status,
    }


def _accepted_rejected_counts(decisions: dict[str, Any]) -> tuple[int, int]:
    accepted = 0
    rejected = 0
    for decision in decisions.values():
        if decision.get("accepted") is True:
            accepted += 1
        elif decision.get("reason") not in {"missing_ai_value", "correction_disabled"}:
            rejected += 1
    return accepted, rejected


def _aggregate_case_logs(case_logs: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = 0
    rejected = 0
    improved = 0
    unchanged = 0
    worsened = 0
    before_correct_counts = {
        "class": 0,
        "selected_object": 0,
        "orientation_bucket": 0,
        "arm_recommendation": 0,
        "preset_recommendation": 0,
    }
    after_correct_counts = {key: 0 for key in before_correct_counts}
    before_center_errors: list[float] = []
    after_center_errors: list[float] = []

    for case_log in case_logs:
        accepted += int(case_log["correction_summary"]["accepted_count"])
        rejected += int(case_log["correction_summary"]["rejected_count"])
        outcome = case_log["before_after_deltas"]["case_outcome"]
        if outcome == "improved":
            improved += 1
        elif outcome == "worsened":
            worsened += 1
        else:
            unchanged += 1

        before = case_log["metrics"]["before"]
        after = case_log["metrics"]["after"]
        bool_mapping = {
            "class": "class_correct",
            "selected_object": "selected_object_correct",
            "orientation_bucket": "orientation_bucket_correct",
            "arm_recommendation": "arm_recommendation_correct",
            "preset_recommendation": "preset_recommendation_correct",
        }
        for aggregate_key, metric_key in bool_mapping.items():
            if before.get(metric_key) is True:
                before_correct_counts[aggregate_key] += 1
            if after.get(metric_key) is True:
                after_correct_counts[aggregate_key] += 1
        if isinstance(before.get("center_2d_error_px"), (int, float)):
            before_center_errors.append(float(before["center_2d_error_px"]))
        if isinstance(after.get("center_2d_error_px"), (int, float)):
            after_center_errors.append(float(after["center_2d_error_px"]))

    case_count = len(case_logs)
    return {
        "case_count": case_count,
        "accepted_ai_corrections": accepted,
        "rejected_ai_corrections": rejected,
        "cases_improved": improved,
        "cases_unchanged": unchanged,
        "cases_worsened": worsened,
        "before_correct_counts": before_correct_counts,
        "after_correct_counts": after_correct_counts,
        "before_accuracy": {
            key: (value / case_count if case_count else None)
            for key, value in before_correct_counts.items()
        },
        "after_accuracy": {
            key: (value / case_count if case_count else None)
            for key, value in after_correct_counts.items()
        },
        "before_center_2d_error_mean_px": (
            sum(before_center_errors) / len(before_center_errors) if before_center_errors else None
        ),
        "after_center_2d_error_mean_px": (
            sum(after_center_errors) / len(after_center_errors) if after_center_errors else None
        ),
    }


def _evaluate_cases(
    *,
    cases: list[dict[str, Any]],
    output_dir: Path,
    config: CorrectionConfig,
    center_pass_threshold_px: float,
) -> dict[str, Any]:
    case_logs: list[dict[str, Any]] = []
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{index:03d}")
        original_input = case.get("original_input")
        ai_output = case.get("ai_output")
        truth = case.get("truth")
        if not isinstance(original_input, dict) or not isinstance(ai_output, dict) or not isinstance(truth, dict):
            raise InputCorrectionEvalError(f"{case_id}: original_input, ai_output, and truth must be objects")
        allowed_object_ids = case.get("allowed_object_ids")
        if not isinstance(allowed_object_ids, list):
            truth_id = _object_id(truth)
            allowed_object_ids = [truth_id] if truth_id else []

        correction = apply_input_corrections(
            original_input,
            ai_output,
            config=config,
            allowed_object_ids=[str(item) for item in allowed_object_ids if isinstance(item, str)],
        )
        corrected_input = correction["corrected_input"]
        before_metrics = _evaluate_record(original_input, truth, center_pass_threshold_px)
        after_metrics = _evaluate_record(corrected_input, truth, center_pass_threshold_px)
        deltas = _compare_metrics(before_metrics, after_metrics)
        accepted_count, rejected_count = _accepted_rejected_counts(correction["decisions"])

        case_log = {
            "format_version": OUTPUT_FORMAT_VERSION,
            "case_id": case_id,
            "sample_id": case.get("sample_id"),
            "original_input": original_input,
            "ai_output": ai_output,
            "corrected_input": corrected_input,
            "truth": truth,
            "correction_decisions": correction["decisions"],
            "confidence_values": {
                field: decision.get("confidence")
                for field, decision in correction["decisions"].items()
            },
            "ignored_forbidden_fields": correction["ignored_forbidden_fields"],
            "correction_summary": {
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "correction_enabled": correction["correction_enabled"],
                "scope": correction["scope"],
            },
            "metrics": {
                "before": before_metrics,
                "after": after_metrics,
            },
            "before_after_deltas": deltas,
        }
        _write_json(cases_dir / f"{case_id}.json", case_log)
        case_logs.append(case_log)

    aggregate = _aggregate_case_logs(case_logs)
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "input_correction_only_no_runtime_no_grasp_pose_no_motion_control",
        "correction_config": {
            "enabled": config.enabled,
            "min_confidence": config.min_confidence,
            "field_min_confidence": config.field_min_confidence,
            "max_center_shift_px": config.max_center_shift_px,
            "max_roi_center_shift_px": config.max_roi_center_shift_px,
            "max_roi_corner_shift_px": config.max_roi_corner_shift_px,
            "max_roi_scale_ratio": config.max_roi_scale_ratio,
        },
        "aggregate": aggregate,
        "case_log_paths": [
            str((output_dir / "cases" / f"{case_log['case_id']}.json").resolve())
            for case_log in case_logs
        ],
    }


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    aggregate = summary["aggregate"]
    print("Task 1 input-correction evaluation")
    print(f"Scope: {summary['scope']}")
    print(f"Output dir: {output_dir}")
    print(
        "Cases: "
        f"{aggregate['case_count']} "
        f"(improved={aggregate['cases_improved']}, "
        f"unchanged={aggregate['cases_unchanged']}, "
        f"worsened={aggregate['cases_worsened']})"
    )
    print(
        "AI corrections: "
        f"accepted={aggregate['accepted_ai_corrections']}, "
        f"rejected={aggregate['rejected_ai_corrections']}"
    )
    print(
        "Class accuracy: "
        f"{aggregate['before_accuracy']['class']} -> {aggregate['after_accuracy']['class']}"
    )
    print(
        "Selected-object accuracy: "
        f"{aggregate['before_accuracy']['selected_object']} -> "
        f"{aggregate['after_accuracy']['selected_object']}"
    )
    print(
        "Center 2D mean error px: "
        f"{aggregate['before_center_2d_error_mean_px']} -> "
        f"{aggregate['after_center_2d_error_mean_px']}"
    )
    print(
        "Orientation accuracy: "
        f"{aggregate['before_accuracy']['orientation_bucket']} -> "
        f"{aggregate['after_accuracy']['orientation_bucket']}"
    )
    print(
        "Arm recommendation accuracy: "
        f"{aggregate['before_accuracy']['arm_recommendation']} -> "
        f"{aggregate['after_accuracy']['arm_recommendation']}"
    )
    print(
        "Preset recommendation accuracy: "
        f"{aggregate['before_accuracy']['preset_recommendation']} -> "
        f"{aggregate['after_accuracy']['preset_recommendation']}"
    )
    print(f"Summary JSON: {output_dir / 'summary.json'}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", help="Phase 1 run directory used to generate cases when --cases-json is omitted.")
    parser.add_argument("--run-id", help="Run id under $OUTPUT_ROOT/datasets/task1_rgbd_labels.")
    parser.add_argument("--cases-json", help="Prepared JSON/JSONL cases file. If omitted, cases are generated from labels.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for generated synthetic AI cases.")
    parser.add_argument("--metric-camera", default="head_left", choices=CAMERA_NAMES, help="Preferred camera for generated 2D truth.")
    parser.add_argument("--output-dir", help="Output directory. Defaults under $OUTPUT_ROOT/test_runs/task1_input_correction_eval/.")
    parser.add_argument("--disable-ai-correction", action="store_true", help="Keep original input and log all AI corrections as disabled.")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--max-center-shift-px", type=float, default=45.0)
    parser.add_argument("--max-roi-center-shift-px", type=float, default=45.0)
    parser.add_argument("--max-roi-corner-shift-px", type=float, default=70.0)
    parser.add_argument("--max-roi-scale-ratio", type=float, default=3.0)
    parser.add_argument("--center-pass-threshold-px", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        if args.limit <= 0:
            raise InputCorrectionEvalError("--limit must be positive")
        dataset_root = _dataset_root_from_args(args)
        if args.cases_json:
            cases = _load_cases_from_file(Path(args.cases_json).expanduser().resolve(), args.limit)
            run_label = Path(args.cases_json).stem
        else:
            if dataset_root is None:
                raise InputCorrectionEvalError("Provide --cases-json, --dataset-root, or --run-id")
            cases = _load_dataset_cases(
                dataset_root=dataset_root,
                limit=args.limit,
                seed=args.seed,
                metric_camera=args.metric_camera,
            )
            run_label = dataset_root.name
        if len(cases) < args.limit:
            print(f"warning: requested {args.limit} cases but only loaded {len(cases)}", file=sys.stderr)
        if not cases:
            raise InputCorrectionEvalError("no input-correction cases available")

        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(run_label)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = CorrectionConfig(
            enabled=not args.disable_ai_correction,
            min_confidence=float(args.min_confidence),
            max_center_shift_px=float(args.max_center_shift_px),
            max_roi_center_shift_px=float(args.max_roi_center_shift_px),
            max_roi_corner_shift_px=float(args.max_roi_corner_shift_px),
            max_roi_scale_ratio=float(args.max_roi_scale_ratio),
        )
        summary = _evaluate_cases(
            cases=cases,
            output_dir=output_dir,
            config=config,
            center_pass_threshold_px=float(args.center_pass_threshold_px),
        )
        summary["inputs"] = {
            "dataset_root": str(dataset_root) if dataset_root is not None else None,
            "cases_json": str(Path(args.cases_json).expanduser().resolve()) if args.cases_json else None,
            "limit": args.limit,
            "loaded_case_count": len(cases),
            "metric_camera": args.metric_camera,
            "case_generation_mode": "prepared_cases" if args.cases_json else "generated_from_phase1_dataset",
            "case_generation_note": (
                "Generated cases use deterministic synthetic AI outputs to test gating; "
                "they are not real Thinker runtime predictions."
                if not args.cases_json
                else "Prepared cases provided by caller."
            ),
        }
        _write_json(output_dir / "summary.json", summary)
        _print_summary(summary, output_dir)
        return 0
    except InputCorrectionEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
