#!/usr/bin/env python3
"""Input-level AI correction gates for Task 1.

This module intentionally corrects only perception/advisory input fields. It
does not generate 3D grasp poses, joint commands, planner targets, or execution
phases.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_FIELDS = (
    "selected_object_id",
    "class",
    "center_2d",
    "roi",
    "orientation_bucket",
    "recommended_arm",
    "recommended_preset",
)
ORIENTATION_BUCKETS = (
    "front",
    "front_left",
    "left",
    "back_left",
    "back",
    "back_right",
    "right",
    "front_right",
    "unknown",
)
OBJECT_CLASSES = ("A", "B", "unknown")
ARMS = ("left", "right")
FORBIDDEN_AI_FIELDS = (
    "grasp_pose",
    "final_grasp_pose",
    "target_pose",
    "target_pose_world",
    "pose_6d",
    "joint_command",
    "joint_commands",
    "motion_command",
    "motion_commands",
    "trajectory",
    "waypoints",
    "center_3d",
    "position_3d",
    "position_xyz",
    "position_xyz_m",
    "world_position",
)


@dataclass(frozen=True)
class CorrectionConfig:
    enabled: bool = True
    min_confidence: float = 0.70
    field_min_confidence: dict[str, float] = field(default_factory=dict)
    max_center_shift_px: float = 45.0
    max_roi_center_shift_px: float = 45.0
    max_roi_corner_shift_px: float = 70.0
    max_roi_scale_ratio: float = 3.0


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


def _field_aliases(field_name: str) -> tuple[str, ...]:
    aliases = {
        "selected_object_id": ("selected_object_id", "object_id", "chosen_object_id", "target_object_id"),
        "class": ("class", "object_class", "predicted_class"),
        "center_2d": ("center_2d", "center_2d_px", "uv_px"),
        "roi": ("roi", "roi_2d", "roi_2d_px", "bbox_2d", "bbox_2d_px"),
        "orientation_bucket": ("orientation_bucket", "coarse_orientation", "yaw_bucket"),
        "recommended_arm": ("recommended_arm", "arm", "selected_arm"),
        "recommended_preset": ("recommended_preset", "preset", "selected_preset"),
    }
    return aliases[field_name]


def _get_field(record: dict[str, Any], field_name: str) -> Any:
    for alias in _field_aliases(field_name):
        if alias in record:
            return record[alias]
    return None


def _set_field(record: dict[str, Any], field_name: str, value: Any) -> None:
    primary = {
        "selected_object_id": "selected_object_id",
        "class": "class",
        "center_2d": "center_2d",
        "roi": "roi",
        "orientation_bucket": "orientation_bucket",
        "recommended_arm": "recommended_arm",
        "recommended_preset": "recommended_preset",
    }[field_name]
    record[primary] = value
    if field_name == "selected_object_id":
        record["object_id"] = value
    elif field_name == "orientation_bucket":
        record["coarse_orientation"] = value
    elif field_name == "recommended_arm":
        record["arm"] = value
    elif field_name == "recommended_preset":
        record["preset"] = value


def _confidence_aliases(field_name: str) -> tuple[str, ...]:
    return {
        "selected_object_id": (
            "selected_object_confidence",
            "object_id_confidence",
            "selection_confidence",
            "confidence",
        ),
        "class": ("class_confidence", "object_class_confidence", "confidence"),
        "center_2d": ("center_2d_confidence", "center_confidence", "confidence"),
        "roi": ("roi_confidence", "bbox_confidence", "confidence"),
        "orientation_bucket": ("orientation_bucket_confidence", "orientation_confidence", "yaw_bucket_confidence", "confidence"),
        "recommended_arm": ("recommended_arm_confidence", "arm_confidence", "confidence"),
        "recommended_preset": ("recommended_preset_confidence", "preset_confidence", "confidence"),
    }[field_name]


def _get_confidence(ai_output: dict[str, Any], field_name: str) -> float | None:
    confidences = ai_output.get("confidences")
    if isinstance(confidences, dict):
        value = _finite_float(confidences.get(field_name))
        if value is not None:
            return value
    for alias in _confidence_aliases(field_name):
        value = _finite_float(ai_output.get(alias))
        if value is not None:
            return value
    return None


def _threshold(config: CorrectionConfig, field_name: str) -> float:
    return float(config.field_min_confidence.get(field_name, config.min_confidence))


def _distance_2d(a: list[float], b: list[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _roi_center(roi: list[float]) -> list[float]:
    return [(float(roi[0]) + float(roi[2])) * 0.5, (float(roi[1]) + float(roi[3])) * 0.5]


def _roi_size(roi: list[float]) -> list[float]:
    return [abs(float(roi[2]) - float(roi[0])), abs(float(roi[3]) - float(roi[1]))]


def _ratio_larger_to_smaller(a: float, b: float) -> float | None:
    if a <= 0.0 or b <= 0.0:
        return None
    return max(a, b) / min(a, b)


def _forbidden_fields(ai_output: dict[str, Any]) -> list[str]:
    forbidden: list[str] = []
    for key in ai_output:
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_AI_FIELDS):
            forbidden.append(key)
    return sorted(forbidden)


def _decision(
    *,
    field_name: str,
    accepted: bool,
    reason: str,
    original: Any,
    ai_value: Any,
    corrected: Any,
    confidence: float | None,
    threshold: float,
    delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "field": field_name,
        "accepted": bool(accepted),
        "source": "ai" if accepted else "original",
        "reason": reason,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "original": original,
        "ai": ai_value,
        "corrected": corrected,
        "delta": delta or {},
    }


def _validate_discrete(
    *,
    field_name: str,
    ai_value: Any,
    allowed_values: tuple[str, ...] | set[str] | None,
) -> str | None:
    if not isinstance(ai_value, str) or not ai_value:
        return "invalid_or_missing_ai_value"
    if allowed_values is not None and ai_value not in allowed_values:
        return "ai_value_not_allowed"
    return None


def _evaluate_field(
    *,
    field_name: str,
    original_input: dict[str, Any],
    ai_output: dict[str, Any],
    corrected_input: dict[str, Any],
    config: CorrectionConfig,
    allowed_object_ids: set[str] | None,
) -> dict[str, Any]:
    original = _get_field(original_input, field_name)
    ai_value = _get_field(ai_output, field_name)
    confidence = _get_confidence(ai_output, field_name)
    threshold = _threshold(config, field_name)

    if not config.enabled:
        return _decision(
            field_name=field_name,
            accepted=False,
            reason="correction_disabled",
            original=original,
            ai_value=ai_value,
            corrected=original,
            confidence=confidence,
            threshold=threshold,
        )
    if ai_value is None:
        return _decision(
            field_name=field_name,
            accepted=False,
            reason="missing_ai_value",
            original=original,
            ai_value=ai_value,
            corrected=original,
            confidence=confidence,
            threshold=threshold,
        )
    if confidence is None:
        return _decision(
            field_name=field_name,
            accepted=False,
            reason="missing_confidence",
            original=original,
            ai_value=ai_value,
            corrected=original,
            confidence=confidence,
            threshold=threshold,
        )
    if confidence < threshold:
        return _decision(
            field_name=field_name,
            accepted=False,
            reason="low_confidence",
            original=original,
            ai_value=ai_value,
            corrected=original,
            confidence=confidence,
            threshold=threshold,
        )

    delta: dict[str, Any] = {}
    rejection_reason: str | None = None
    normalized_ai = ai_value

    if field_name == "selected_object_id":
        rejection_reason = _validate_discrete(
            field_name=field_name,
            ai_value=ai_value,
            allowed_values=allowed_object_ids,
        )
        delta["changed"] = original != ai_value
    elif field_name == "class":
        rejection_reason = _validate_discrete(
            field_name=field_name,
            ai_value=ai_value,
            allowed_values=OBJECT_CLASSES,
        )
        delta["changed"] = original != ai_value
    elif field_name == "orientation_bucket":
        rejection_reason = _validate_discrete(
            field_name=field_name,
            ai_value=ai_value,
            allowed_values=ORIENTATION_BUCKETS,
        )
        delta["changed"] = original != ai_value
    elif field_name == "recommended_arm":
        rejection_reason = _validate_discrete(
            field_name=field_name,
            ai_value=ai_value,
            allowed_values=ARMS,
        )
        delta["changed"] = original != ai_value
    elif field_name == "recommended_preset":
        rejection_reason = _validate_discrete(
            field_name=field_name,
            ai_value=ai_value,
            allowed_values=None,
        )
        delta["changed"] = original != ai_value
    elif field_name == "center_2d":
        normalized_ai = _finite_vector(ai_value, 2)
        if normalized_ai is None:
            rejection_reason = "invalid_or_missing_ai_value"
        else:
            original_center = _finite_vector(original, 2)
            if original_center is not None:
                shift = _distance_2d(original_center, normalized_ai)
                delta["shift_px"] = shift
                if shift > config.max_center_shift_px:
                    rejection_reason = "center_shift_too_large"
    elif field_name == "roi":
        normalized_ai = _finite_vector(ai_value, 4)
        if normalized_ai is None:
            rejection_reason = "invalid_or_missing_ai_value"
        else:
            original_roi = _finite_vector(original, 4)
            if original_roi is not None:
                center_shift = _distance_2d(_roi_center(original_roi), _roi_center(normalized_ai))
                corner_shift = max(abs(float(a) - float(b)) for a, b in zip(original_roi, normalized_ai))
                original_size = _roi_size(original_roi)
                ai_size = _roi_size(normalized_ai)
                width_ratio = _ratio_larger_to_smaller(original_size[0], ai_size[0])
                height_ratio = _ratio_larger_to_smaller(original_size[1], ai_size[1])
                delta.update(
                    {
                        "center_shift_px": center_shift,
                        "corner_shift_px": corner_shift,
                        "width_scale_ratio": width_ratio,
                        "height_scale_ratio": height_ratio,
                    }
                )
                if center_shift > config.max_roi_center_shift_px:
                    rejection_reason = "roi_center_shift_too_large"
                elif corner_shift > config.max_roi_corner_shift_px:
                    rejection_reason = "roi_corner_shift_too_large"
                elif width_ratio is not None and width_ratio > config.max_roi_scale_ratio:
                    rejection_reason = "roi_width_scale_too_large"
                elif height_ratio is not None and height_ratio > config.max_roi_scale_ratio:
                    rejection_reason = "roi_height_scale_too_large"

    if rejection_reason is not None:
        return _decision(
            field_name=field_name,
            accepted=False,
            reason=rejection_reason,
            original=original,
            ai_value=ai_value,
            corrected=original,
            confidence=confidence,
            threshold=threshold,
            delta=delta,
        )

    _set_field(corrected_input, field_name, normalized_ai)
    return _decision(
        field_name=field_name,
        accepted=True,
        reason="accepted",
        original=original,
        ai_value=ai_value,
        corrected=normalized_ai,
        confidence=confidence,
        threshold=threshold,
        delta=delta,
    )


def apply_input_corrections(
    original_input: dict[str, Any],
    ai_output: dict[str, Any],
    *,
    config: CorrectionConfig | None = None,
    allowed_object_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Apply gated AI corrections to input-level fields only."""
    active_config = config or CorrectionConfig()
    corrected_input = copy.deepcopy(original_input)
    allowed_set = set(allowed_object_ids) if allowed_object_ids is not None else None
    decisions: dict[str, Any] = {}
    for field_name in SUPPORTED_FIELDS:
        decisions[field_name] = _evaluate_field(
            field_name=field_name,
            original_input=original_input,
            ai_output=ai_output,
            corrected_input=corrected_input,
            config=active_config,
            allowed_object_ids=allowed_set,
        )

    forbidden_fields = _forbidden_fields(ai_output)
    return {
        "corrected_input": corrected_input,
        "decisions": decisions,
        "accepted_count": sum(1 for item in decisions.values() if item["accepted"]),
        "rejected_count": sum(
            1
            for item in decisions.values()
            if not item["accepted"] and item["reason"] not in {"missing_ai_value", "correction_disabled"}
        ),
        "ignored_forbidden_fields": forbidden_fields,
        "correction_enabled": active_config.enabled,
        "scope": "input_fields_only_no_3d_pose_no_motion_control",
    }
