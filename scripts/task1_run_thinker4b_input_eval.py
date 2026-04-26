#!/usr/bin/env python3
"""Run Thinker4B input evaluation on recorded Task 1 camera samples.

This script is input-evaluation only. It reads recorded RGB arrays from the
Phase 1 dataset, sends selected camera images to a configured Thinker4B
provider, applies the existing input-correction gates, compares original /
Thinker4B / corrected fields against labels, and writes structured logs plus a
human-readable report.

It does not run robot execution, produce grasp poses, call IK, or modify the
planner/manipulation backend.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from task1_input_correction import CorrectionConfig, apply_input_corrections
from task1_run_input_correction_eval import (
    _accepted_rejected_counts,
    _alternate_arm,
    _alternate_bucket,
    _alternate_preset,
    _center_from_record,
    _class_value,
    _compare_metrics,
    _distance_2d,
    _evaluate_record,
    _extract_truth_center_from_visibility,
    _extract_truth_roi_from_visibility,
    _finite_float,
    _finite_vector,
    _flip_class,
    _object_id,
    _read_json,
    _read_jsonl,
    _reference_arm,
    _reference_preset,
    _roi_center,
    _shift_center,
    _shift_roi,
    _write_json,
)


CAMERA_NAMES = ("head_left", "head_right", "wrist_left", "wrist_right")
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
DEFAULT_CASES_PER_SEED = 10
OUTPUT_FORMAT_VERSION = "0.1.0"
PROMPT_MODES = ("image-only", "refine-original")
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


class Thinker4BEvalError(RuntimeError):
    """Raised for configuration or evaluation setup errors."""


class ThinkerProviderUnavailable(Thinker4BEvalError):
    """Raised when no real Thinker4B provider is configured."""


class ThinkerProviderError(Thinker4BEvalError):
    """Raised when a configured Thinker4B provider fails for one case."""


@dataclass
class ThinkerInferResult:
    payload: dict[str, Any]
    raw_text: str | None = None


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    seeds: list[int] = []
    for item in _parse_csv(value):
        try:
            seeds.append(int(item))
        except ValueError as exc:
            raise Thinker4BEvalError(f"invalid seed {item!r}") from exc
    if not seeds:
        raise Thinker4BEvalError("at least one seed is required")
    return seeds


def _dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    if not args.run_id:
        raise Thinker4BEvalError("provide --dataset-root or --run-id")
    output_root = os.environ.get("OUTPUT_ROOT")
    if not output_root:
        raise Thinker4BEvalError("OUTPUT_ROOT is required when --run-id is used")
    return (Path(output_root) / "datasets" / "task1_rgbd_labels" / args.run_id).resolve()


def _default_output_dir(run_label: str) -> Path:
    output_root = os.environ.get("OUTPUT_ROOT")
    if not output_root:
        raise Thinker4BEvalError("OUTPUT_ROOT is required when --output-dir is omitted")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(output_root) / "test_runs" / "task1_thinker4b_input_eval" / f"{run_label}_{timestamp}"


def _load_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise Thinker4BEvalError(f"manifest does not exist: {manifest_path}")
    return _read_jsonl(manifest_path)


def _load_run_metadata(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "run_metadata.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _truth_from_object(
    *,
    obj: dict[str, Any],
    preferred_camera: str,
    x_extent_m: float | None,
) -> dict[str, Any]:
    table_pose = obj.get("table_frame_pose") if isinstance(obj.get("table_frame_pose"), dict) else {}
    table_x = _finite_float(table_pose.get("x"))
    truth_center, truth_camera = _extract_truth_center_from_visibility(obj, preferred_camera)
    truth_roi = _extract_truth_roi_from_visibility(obj, truth_camera)
    orientation_bucket = table_pose.get("coarse_orientation")
    if not isinstance(orientation_bucket, str):
        world_pose = obj.get("world_pose") if isinstance(obj.get("world_pose"), dict) else {}
        orientation_bucket = world_pose.get("coarse_orientation")
    if not isinstance(orientation_bucket, str):
        orientation_bucket = "unknown"
    return {
        "selected_object_id": obj.get("object_id"),
        "object_id": obj.get("object_id"),
        "class": obj.get("class"),
        "center_2d": truth_center,
        "roi": truth_roi,
        "orientation_bucket": orientation_bucket,
        "recommended_arm": _reference_arm(table_x, x_extent_m),
        "recommended_preset": _reference_preset(orientation_bucket),
        "camera_name": truth_camera,
        "reference_note": (
            "class/object/pose truth comes from Phase 1 labels; arm/preset are "
            "deterministic evaluator references for this input-only test."
        ),
    }


def _truth_from_object_for_exact_camera(
    *,
    obj: dict[str, Any],
    camera_name: str,
    x_extent_m: float | None,
) -> dict[str, Any] | None:
    table_pose = obj.get("table_frame_pose") if isinstance(obj.get("table_frame_pose"), dict) else {}
    table_x = _finite_float(table_pose.get("x"))
    truth_center, truth_camera = _extract_truth_center_from_visibility(obj, camera_name)
    if truth_camera != camera_name:
        truth_center = None
        truth_camera = None
    truth_roi = _extract_truth_roi_from_visibility(obj, camera_name)
    if truth_center is None and truth_roi is None:
        return None
    orientation_bucket = table_pose.get("coarse_orientation")
    if not isinstance(orientation_bucket, str):
        world_pose = obj.get("world_pose") if isinstance(obj.get("world_pose"), dict) else {}
        orientation_bucket = world_pose.get("coarse_orientation")
    if not isinstance(orientation_bucket, str):
        orientation_bucket = "unknown"
    return {
        "selected_object_id": obj.get("object_id"),
        "object_id": obj.get("object_id"),
        "class": obj.get("class"),
        "center_2d": truth_center,
        "roi": truth_roi,
        "orientation_bucket": orientation_bucket,
        "recommended_arm": _reference_arm(table_x, x_extent_m),
        "recommended_preset": _reference_preset(orientation_bucket),
        "camera_name": truth_camera,
    }


def _camera_paths_from_manifest(dataset_root: Path, entry: dict[str, Any]) -> dict[str, Path]:
    cameras = entry.get("cameras") if isinstance(entry.get("cameras"), dict) else {}
    paths: dict[str, Path] = {}
    for camera_name in CAMERA_NAMES:
        camera_record = cameras.get(camera_name)
        if not isinstance(camera_record, dict):
            continue
        rgb_record = camera_record.get("rgb")
        if not isinstance(rgb_record, dict):
            continue
        rel_path = rgb_record.get("path")
        if isinstance(rel_path, str) and rel_path:
            paths[camera_name] = dataset_root / rel_path
    return paths


def _build_case_pool(
    *,
    dataset_root: Path,
    preferred_camera: str,
) -> list[dict[str, Any]]:
    run_metadata = _load_run_metadata(dataset_root)
    table_frame = {}
    scene = run_metadata.get("scene") if isinstance(run_metadata.get("scene"), dict) else {}
    if isinstance(scene.get("table_frame"), dict):
        table_frame = scene["table_frame"]
    x_extent_m = _finite_float(table_frame.get("x_extent_m"))
    run_id = str(run_metadata.get("run_id") or dataset_root.name)

    pool: list[dict[str, Any]] = []
    for entry in _load_manifest(dataset_root):
        sample_id = entry.get("sample_id")
        paths = entry.get("paths")
        if not isinstance(sample_id, str) or not isinstance(paths, dict):
            continue
        label_rel_path = paths.get("labels")
        if not isinstance(label_rel_path, str):
            continue
        labels_path = dataset_root / label_rel_path
        if not labels_path.exists():
            continue
        labels = _read_json(labels_path)
        objects = labels.get("objects") if isinstance(labels, dict) else None
        if not isinstance(objects, list):
            continue
        allowed_ids = [
            str(obj.get("object_id"))
            for obj in objects
            if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
        ]
        camera_paths = _camera_paths_from_manifest(dataset_root, entry)
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict) or not isinstance(obj.get("object_id"), str):
                continue
            truth = _truth_from_object(
                obj=obj,
                preferred_camera=preferred_camera,
                x_extent_m=x_extent_m,
            )
            pool.append(
                {
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "object_index": object_index,
                    "object_id": obj["object_id"],
                    "truth": truth,
                    "allowed_object_ids": allowed_ids,
                    "scene_objects": objects,
                    "x_extent_m": x_extent_m,
                    "camera_paths": {key: str(value) for key, value in camera_paths.items()},
                    "label_object": obj,
                    "dataset_root": str(dataset_root),
                }
            )
    if not pool:
        raise Thinker4BEvalError(f"no object cases could be built from {dataset_root}")
    return pool


def _make_original_input(case_source: dict[str, Any], *, case_number: int) -> dict[str, Any]:
    truth = case_source["truth"]
    allowed_ids = case_source["allowed_object_ids"]
    truth_id = _object_id(truth)
    wrong_id = truth_id
    if truth_id in allowed_ids and len(allowed_ids) > 1:
        wrong_id = allowed_ids[(allowed_ids.index(truth_id) + 1) % len(allowed_ids)]
    center = _center_from_record(truth)
    roi = _finite_vector(truth.get("roi"), 4)
    dx = 18.0 + float((case_number % 4) * 7)
    dy = -14.0 + float((case_number % 3) * 6)
    original_id = wrong_id if case_number % 4 == 0 else truth_id
    return {
        "selected_object_id": original_id,
        "object_id": original_id,
        "class": _flip_class(_class_value(truth)) if case_number % 3 == 0 else _class_value(truth),
        "center_2d": _shift_center(center, dx, dy),
        "roi": _shift_roi(roi, dx, dy),
        "orientation_bucket": (
            _alternate_bucket(truth.get("orientation_bucket"))
            if case_number % 2 == 0
            else truth.get("orientation_bucket")
        ),
        "recommended_arm": (
            _alternate_arm(truth.get("recommended_arm"))
            if case_number % 3 == 1
            else truth.get("recommended_arm")
        ),
        "recommended_preset": (
            _alternate_preset(truth.get("recommended_preset"))
            if case_number % 3 == 2
            else truth.get("recommended_preset")
        ),
        "camera_name": truth.get("camera_name"),
        "source": "deterministic_generated_original_input_estimate",
    }


def _select_camera_names(
    *,
    requested: list[str],
    truth_camera: str | None,
    include_truth_camera: bool,
    available: dict[str, str],
) -> list[str]:
    selected: list[str] = []
    for camera_name in requested:
        if camera_name in available and camera_name not in selected:
            selected.append(camera_name)
    if include_truth_camera and truth_camera in available and truth_camera not in selected:
        selected.append(str(truth_camera))
    if not selected:
        for camera_name in CAMERA_NAMES:
            if camera_name in available:
                selected.append(camera_name)
                break
    return selected


def _select_cases(
    *,
    pool: list[dict[str, Any]],
    seeds: list[int],
    cases_per_seed: int,
    requested_cameras: list[str],
    include_truth_camera: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    global_case_number = 0
    for seed in seeds:
        rng = random.Random(seed)
        if len(pool) >= cases_per_seed:
            indices = rng.sample(range(len(pool)), cases_per_seed)
            reuse_reason = None
        else:
            indices = [rng.randrange(len(pool)) for _ in range(cases_per_seed)]
            reuse_reason = "candidate_pool_smaller_than_cases_per_seed"
        for case_index, pool_index in enumerate(indices):
            source = pool[pool_index]
            truth = source["truth"]
            camera_names = _select_camera_names(
                requested=requested_cameras,
                truth_camera=truth.get("camera_name") if isinstance(truth.get("camera_name"), str) else None,
                include_truth_camera=include_truth_camera,
                available=source["camera_paths"],
            )
            case_id = f"seed_{seed}_case_{case_index:02d}"
            original_input = _make_original_input(source, case_number=global_case_number)
            cases.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "case_index": case_index,
                    "source_pool_index": pool_index,
                    "reuse_reason": reuse_reason,
                    "run_id": source["run_id"],
                    "sample_id": source["sample_id"],
                    "object_index": source["object_index"],
                    "object_id": source["object_id"],
                    "allowed_object_ids": source["allowed_object_ids"],
                    "scene_objects": source["scene_objects"],
                    "x_extent_m": source["x_extent_m"],
                    "camera_names": camera_names,
                    "camera_files": {
                        camera_name: source["camera_paths"][camera_name]
                        for camera_name in camera_names
                    },
                    "all_camera_files": dict(source["camera_paths"]),
                    "truth": truth,
                    "original_input": original_input,
                }
            )
            global_case_number += 1
    return cases


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    rgb = np.asarray(array)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ThinkerProviderError(f"expected RGB/RGBA HxWxC array, got shape {list(rgb.shape)}")
    rgb = rgb[:, :, :3]
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    rgb = rgb.astype(np.float32, copy=False)
    finite = np.isfinite(rgb)
    if np.any(finite) and float(np.nanmax(rgb[finite])) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def _png_bytes_from_image(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_png_bytes(path: Path, *, max_side: int) -> bytes:
    if not path.exists():
        raise ThinkerProviderError(f"camera RGB file does not exist: {path}")
    array = np.load(path, allow_pickle=False)
    image = Image.fromarray(_to_uint8_rgb(array))
    if max_side > 0 and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return _png_bytes_from_image(image)


def _active_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _image_stats(rgb: np.ndarray, *, dtype_name: str, original_shape: tuple[int, ...]) -> dict[str, Any]:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    gray = rgb_u8.astype(np.float32).mean(axis=2)
    non_black_mask = gray > 1.0
    bright_mask = gray > 16.0
    return {
        "shape": list(original_shape),
        "dtype": dtype_name,
        "min": float(rgb_u8.min()),
        "max": float(rgb_u8.max()),
        "mean_brightness": float(gray.mean()),
        "non_black_ratio": float(non_black_mask.mean()),
        "bright_ratio": float(bright_mask.mean()),
        "non_black_bbox": _active_bbox(non_black_mask),
        "bright_bbox": _active_bbox(bright_mask),
    }


def _resize_rgb(rgb: np.ndarray, *, max_side: int) -> np.ndarray:
    image = Image.fromarray(rgb)
    if max_side > 0 and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _camera_image_artifact(path: Path, *, max_side: int) -> dict[str, Any]:
    if not path.exists():
        raise ThinkerProviderError(f"camera RGB file does not exist: {path}")
    raw_array = np.load(path, allow_pickle=False)
    raw_rgb = _to_uint8_rgb(raw_array)
    model_input_rgb = _resize_rgb(raw_rgb, max_side=max_side)
    raw_png_bytes = _png_bytes_from_image(Image.fromarray(raw_rgb))
    model_input_png_bytes = _png_bytes_from_image(Image.fromarray(model_input_rgb))
    return {
        "path": str(path),
        "raw_png_bytes": raw_png_bytes,
        "model_input_png_bytes": model_input_png_bytes,
        "png_base64": base64.b64encode(model_input_png_bytes).decode("ascii"),
        "mime_type": "image/png",
        "raw_stats": _image_stats(raw_rgb, dtype_name=str(raw_array.dtype), original_shape=tuple(raw_array.shape)),
        "model_input_stats": _image_stats(
            model_input_rgb,
            dtype_name="uint8",
            original_shape=tuple(model_input_rgb.shape),
        ),
    }


def _case_camera_artifacts(case: dict[str, Any], *, max_side: int) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for camera_name in CAMERA_NAMES:
        camera_path = case["all_camera_files"].get(camera_name)
        if not isinstance(camera_path, str):
            continue
        artifact = _camera_image_artifact(Path(camera_path), max_side=max_side)
        artifact["camera_name"] = camera_name
        artifact["sent_to_model"] = camera_name in case["camera_names"]
        artifacts[camera_name] = artifact
    return artifacts


def _prompt_for_case(case: dict[str, Any], *, prompt_mode: str) -> str:
    primary_camera = case["camera_names"][0] if case["camera_names"] else "head_left"
    payload = {
        "case_id": case["case_id"],
        "run_id": case["run_id"],
        "sample_id": case["sample_id"],
        "camera_names": case["camera_names"],
        "primary_image_camera_for_2d_outputs": primary_camera,
    }
    if prompt_mode == "image-only":
        return (
            "You are Thinker4B analyzing recorded camera images from HRC Task 1.\n"
            "Use only visual evidence visible in the provided image(s).\n"
            "Do not use any prior estimate, prior object id hint, or hidden state.\n"
            "Do not guess dataset object ids such as task1_part_XXX from the image.\n"
            "All 2D outputs must be expressed in the first camera image listed in "
            f"camera_names, which is {primary_camera} for this case.\n"
            "Order objects by confidence or task relevance, with the primary candidate first.\n"
            "If something is uncertain, return unknown or null and low confidence.\n"
            "Return ONLY one JSON object with this shape:\n"
            "{\n"
            '  "frame_id": "sample or case id",\n'
            '  "selected_object_id": null,\n'
            '  "global_confidence": 0.0,\n'
            '  "objects": [\n'
            "    {\n"
            '      "class": "A|B|unknown",\n'
            '      "center_2d": [u, v] or null,\n'
            '      "roi": [x1, y1, x2, y2] or null,\n'
            '      "orientation_bucket": "front|front_left|left|back_left|back|back_right|right|front_right|unknown",\n'
            '      "confidence": 0.0,\n'
            '      "recommended_arm": "left|right|null",\n'
            '      "recommended_preset": "short string or null",\n'
            '      "notes": "optional short note"\n'
            "    }\n"
            "  ],\n"
            '  "model_notes": "optional short note"\n'
            "}\n"
            "Do not output final 3D pose, grasp pose, IK targets, motion commands, "
            "waypoints, or trajectories.\n"
            "Context JSON:\n"
            f"{json.dumps(payload, indent=2, sort_keys=True)}"
        )
    payload["allowed_object_ids"] = case["allowed_object_ids"]
    payload["original_input_estimate"] = case["original_input"]
    return (
        "You are Thinker4B analyzing recorded camera images from HRC Task 1.\n"
        "Return ONLY one JSON object compatible with this structure:\n"
        "{\n"
        '  "frame_id": "sample or case id",\n'
        '  "selected_object_id": "one allowed object id or null",\n'
        '  "global_confidence": 0.0,\n'
        '  "objects": [\n'
        "    {\n"
        '      "object_id": "one allowed object id",\n'
        '      "class": "A|B|unknown",\n'
        '      "center_2d": [u, v],\n'
        '      "roi": [x1, y1, x2, y2],\n'
        '      "orientation_bucket": "front|front_left|left|back_left|back|back_right|right|front_right|unknown",\n'
        '      "difficulty": 0.0,\n'
        '      "occlusion": 0.0,\n'
        '      "confidence": 0.0,\n'
        '      "recommended_arm": "left|right",\n'
        '      "recommended_preset": "short string",\n'
        '      "notes": "optional short note"\n'
        "    }\n"
        "  ],\n"
        '  "model_notes": "optional short note"\n'
        "}\n"
        "Allowed correction targets are only selected object, class, 2D center/ROI, "
        "coarse orientation bucket, recommended arm, and recommended preset. "
        "Do not output final 3D grasp pose, 6D pose, world pose, joint commands, "
        "motion commands, waypoints, or trajectories.\n"
        "If object ids cannot be visually grounded, use the original estimate only "
        "when it appears consistent; otherwise use null or unknown with low confidence.\n"
        "Context JSON:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def _extract_json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ThinkerProviderError("Thinker4B response did not contain a JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ThinkerProviderError("Thinker4B response JSON is not an object")
    return payload


def _content_text_from_openai_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ThinkerProviderError("OpenAI-compatible response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ThinkerProviderError("OpenAI-compatible response has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    raise ThinkerProviderError("OpenAI-compatible response message has no text content")


class Thinker4BProvider:
    def infer(
        self,
        case: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        prompt: str,
        debug_raw_output_path: Path | None = None,
    ) -> ThinkerInferResult:
        raise NotImplementedError


class OpenAICompatibleProvider(Thinker4BProvider):
    def __init__(self, *, api_base: str | None, api_key: str | None, model: str | None, timeout_s: float) -> None:
        if not api_base:
            raise ThinkerProviderUnavailable("THINKER4B_API_BASE or --api-base is required for openai-compatible provider")
        if not model:
            raise ThinkerProviderUnavailable("THINKER4B_MODEL or --model is required for openai-compatible provider")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def infer(
        self,
        case: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        prompt: str,
        debug_raw_output_path: Path | None = None,
    ) -> ThinkerInferResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image['mime_type']};base64,{image['png_base64']}",
                        "detail": "high",
                    },
                }
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You return strict JSON for input-level perception only.",
                },
                {"role": "user", "content": content},
            ],
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=request_payload,
            timeout=self.timeout_s,
        )
        if response.status_code >= 400:
            raise ThinkerProviderError(f"OpenAI-compatible provider returned HTTP {response.status_code}: {response.text[:500]}")
        text = _content_text_from_openai_response(response.json())
        return ThinkerInferResult(payload=_extract_json_from_text(text), raw_text=text)


class OllamaProvider(Thinker4BProvider):
    def __init__(self, *, host: str | None, model: str | None, timeout_s: float) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("THINKER4B_MODEL") or "thinker4b"
        self.timeout_s = timeout_s

    def infer(
        self,
        case: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        prompt: str,
        debug_raw_output_path: Path | None = None,
    ) -> ThinkerInferResult:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image["png_base64"] for image in images],
                    }
                ],
            },
            timeout=self.timeout_s,
        )
        if response.status_code >= 400:
            raise ThinkerProviderError(f"Ollama provider returned HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        message = payload.get("message") if isinstance(payload, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ThinkerProviderError("Ollama response has no message.content")
        return ThinkerInferResult(payload=_extract_json_from_text(content), raw_text=content)


class CommandProvider(Thinker4BProvider):
    def __init__(self, *, command: str | None, timeout_s: float) -> None:
        if not command:
            raise ThinkerProviderUnavailable("THINKER4B_CMD or --thinker-command is required for command provider")
        self.command = command
        self.timeout_s = timeout_s

    def infer(
        self,
        case: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        prompt: str,
        debug_raw_output_path: Path | None = None,
    ) -> ThinkerInferResult:
        request_payload = {
            "prompt": prompt,
            "case": case,
            "images": images,
            "output_contract": "return JSON matching docs/schemas/task1_thinker_structured_output.schema.json",
        }
        if debug_raw_output_path is not None:
            request_payload["debug_raw_output_path"] = str(debug_raw_output_path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(request_payload, handle)
            request_path = handle.name
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps({"request_path": request_path}),
                text=True,
                shell=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        finally:
            try:
                Path(request_path).unlink(missing_ok=True)
            except Exception:
                pass
        if completed.returncode != 0:
            raise ThinkerProviderError(
                f"command provider failed with code {completed.returncode}: {completed.stderr[:500]}"
            )
        return ThinkerInferResult(payload=_extract_json_from_text(completed.stdout), raw_text=completed.stdout)


class CacheProvider(Thinker4BProvider):
    def __init__(self, *, path: str | None) -> None:
        if not path:
            raise ThinkerProviderUnavailable("--thinker-output-cache is required for cache provider")
        cache_path = Path(path).expanduser().resolve()
        if not cache_path.exists():
            raise ThinkerProviderUnavailable(f"Thinker4B output cache does not exist: {cache_path}")
        payload: Any
        if cache_path.suffix.lower() == ".jsonl":
            payload = _read_jsonl(cache_path)
        else:
            payload = _read_json(cache_path)
        records = payload.get("cases") if isinstance(payload, dict) and isinstance(payload.get("cases"), list) else payload
        if not isinstance(records, list):
            raise ThinkerProviderUnavailable(f"Thinker4B cache must be a JSON/JSONL case list: {cache_path}")
        self.by_case_id: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            case_id = record.get("case_id")
            raw = record.get("thinker4b_raw_output") or record.get("thinker_output") or record.get("raw_output")
            if not isinstance(raw, dict):
                thinker_block = record.get("thinker4b")
                if isinstance(thinker_block, dict) and isinstance(thinker_block.get("raw_output"), dict):
                    raw = thinker_block.get("raw_output")
            if isinstance(case_id, str) and isinstance(raw, dict):
                self.by_case_id[case_id] = raw

    def infer(
        self,
        case: dict[str, Any],
        images: list[dict[str, Any]],
        *,
        prompt: str,
        debug_raw_output_path: Path | None = None,
    ) -> ThinkerInferResult:
        case_id = case["case_id"]
        if case_id not in self.by_case_id:
            raise ThinkerProviderError(f"no cached Thinker4B output for case_id={case_id}")
        payload = self.by_case_id[case_id]
        return ThinkerInferResult(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False))


def _make_provider(args: argparse.Namespace) -> Thinker4BProvider:
    provider = args.provider
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(
            api_base=args.api_base or os.environ.get("THINKER4B_API_BASE"),
            api_key=args.api_key or os.environ.get("THINKER4B_API_KEY"),
            model=args.model or os.environ.get("THINKER4B_MODEL"),
            timeout_s=float(args.timeout_s),
        )
    if provider == "ollama":
        return OllamaProvider(
            host=args.ollama_host,
            model=args.model or os.environ.get("THINKER4B_MODEL"),
            timeout_s=float(args.timeout_s),
        )
    if provider == "command":
        return CommandProvider(
            command=args.thinker_command or os.environ.get("THINKER4B_CMD"),
            timeout_s=float(args.timeout_s),
        )
    if provider == "cache":
        return CacheProvider(path=args.thinker_output_cache)
    raise Thinker4BEvalError(f"unsupported provider: {provider}")


def _clamp_unit(value: Any) -> float:
    numeric = _finite_float(value)
    if numeric is None:
        return 0.0
    return max(0.0, min(1.0, float(numeric)))


def _clean_center_2d(value: Any) -> list[float] | None:
    return _finite_vector(value, 2)


def _clean_roi(value: Any) -> list[float] | None:
    roi = _finite_vector(value, 4)
    if roi is None:
        return None
    if not (roi[0] < roi[2] and roi[1] < roi[3]):
        return None
    return roi


def _clean_object_class(value: Any) -> str:
    return value if isinstance(value, str) and value in {"A", "B", "unknown"} else "unknown"


def _clean_orientation_bucket(value: Any) -> str:
    return value if isinstance(value, str) and value in ORIENTATION_BUCKETS else "unknown"


def _clean_recommended_arm(value: Any) -> str | None:
    return value if isinstance(value, str) and value in {"left", "right"} else None


def _clean_recommended_preset(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _clean_object_candidate(item: dict[str, Any]) -> dict[str, Any]:
    object_id = item.get("object_id")
    notes = item.get("notes")
    cleaned = {
        "object_id": object_id if isinstance(object_id, str) and object_id else None,
        "class": _clean_object_class(item.get("class")),
        "center_2d": _clean_center_2d(item.get("center_2d")),
        "roi": _clean_roi(item.get("roi")),
        "orientation_bucket": _clean_orientation_bucket(item.get("orientation_bucket")),
        "difficulty": _clamp_unit(item.get("difficulty")),
        "occlusion": _clamp_unit(item.get("occlusion")),
        "confidence": _clamp_unit(item.get("confidence")),
        "recommended_arm": _clean_recommended_arm(item.get("recommended_arm")),
        "recommended_preset": _clean_recommended_preset(item.get("recommended_preset")),
        "notes": notes if isinstance(notes, str) and notes.strip() else None,
    }
    return cleaned


def _clean_raw_output(raw: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    objects = raw.get("objects")
    cleaned_objects: list[dict[str, Any]] = []
    if isinstance(objects, list):
        for item in objects:
            if isinstance(item, dict):
                cleaned_objects.append(_clean_object_candidate(item))
    else:
        fallback_item = {
            "object_id": raw.get("object_id") or raw.get("selected_object_id"),
            "class": raw.get("class"),
            "center_2d": raw.get("center_2d"),
            "roi": raw.get("roi"),
            "orientation_bucket": raw.get("orientation_bucket"),
            "difficulty": raw.get("difficulty"),
            "occlusion": raw.get("occlusion"),
            "confidence": raw.get("confidence", raw.get("global_confidence")),
            "recommended_arm": raw.get("recommended_arm"),
            "recommended_preset": raw.get("recommended_preset"),
            "notes": raw.get("notes"),
        }
        if any(
            fallback_item.get(field) is not None
            for field in ("object_id", "class", "center_2d", "roi", "orientation_bucket", "recommended_arm", "recommended_preset", "notes")
        ):
            cleaned_objects.append(_clean_object_candidate(fallback_item))
    selected_object_id = raw.get("selected_object_id")
    if not isinstance(selected_object_id, str) or not selected_object_id:
        selected_object_id = None
    model_notes = raw.get("model_notes")
    return {
        "frame_id": str(raw.get("frame_id") or case["case_id"]),
        "selected_object_id": selected_object_id,
        "global_confidence": _clamp_unit(raw.get("global_confidence", raw.get("confidence"))),
        "objects": cleaned_objects,
        "model_notes": model_notes if isinstance(model_notes, str) and model_notes.strip() else None,
    }


def _selected_object_from_schema(raw: dict[str, Any]) -> dict[str, Any]:
    objects = raw.get("objects") if isinstance(raw.get("objects"), list) else []
    selected_id = raw.get("selected_object_id")
    selected_obj: dict[str, Any] | None = None
    if isinstance(selected_id, str):
        for obj in objects:
            if isinstance(obj, dict) and obj.get("object_id") == selected_id:
                selected_obj = obj
                break
    if selected_obj is None:
        for obj in objects:
            if isinstance(obj, dict):
                selected_obj = obj
                break
    return selected_obj or {}


def _record_center_2d(record: dict[str, Any]) -> list[float] | None:
    center = _clean_center_2d(record.get("center_2d"))
    if center is not None:
        return center
    roi = _clean_roi(record.get("roi"))
    if roi is not None:
        return _roi_center(roi)
    return None


def _truth_candidates_for_case(case: dict[str, Any], *, camera_name: str) -> list[dict[str, Any]]:
    truth_candidates: list[dict[str, Any]] = []
    for obj in case.get("scene_objects", []):
        if not isinstance(obj, dict):
            continue
        candidate = _truth_from_object_for_exact_camera(
            obj=obj,
            camera_name=camera_name,
            x_extent_m=_finite_float(case.get("x_extent_m")),
        )
        if candidate is not None:
            truth_candidates.append(candidate)
    return truth_candidates


def _match_objects_by_geometry(
    raw: dict[str, Any],
    case: dict[str, Any],
    *,
    max_match_distance_px: float,
) -> dict[str, Any]:
    reference_camera = case["camera_names"][0] if case["camera_names"] else None
    if reference_camera is None:
        return {
            "enabled": False,
            "reference_camera": None,
            "selected_prediction_index": None,
            "selected_match_distance_px": None,
            "selected_object_id": None,
            "object_matches": [],
        }
    truth_candidates = _truth_candidates_for_case(case, camera_name=reference_camera)
    predicted_objects = raw.get("objects") if isinstance(raw.get("objects"), list) else []
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred_obj in enumerate(predicted_objects):
        if not isinstance(pred_obj, dict):
            continue
        pred_center = _record_center_2d(pred_obj)
        if pred_center is None:
            continue
        for truth_index, truth_obj in enumerate(truth_candidates):
            truth_center = _record_center_2d(truth_obj)
            if truth_center is None:
                continue
            pairs.append((_distance_2d(pred_center, truth_center), pred_index, truth_index))
    pairs.sort(key=lambda item: item[0])
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches_by_prediction: dict[int, dict[str, Any]] = {}
    for distance_px, pred_index, truth_index in pairs:
        if distance_px > max_match_distance_px:
            continue
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        truth_obj = truth_candidates[truth_index]
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
        matches_by_prediction[pred_index] = {
            "prediction_index": pred_index,
            "matched_truth_object_id": truth_obj.get("object_id"),
            "distance_px": distance_px,
            "truth_center_2d": truth_obj.get("center_2d"),
            "truth_roi": truth_obj.get("roi"),
        }
    selected_prediction_index: int | None = None
    selected_id = raw.get("selected_object_id")
    if isinstance(selected_id, str):
        for index, pred_obj in enumerate(predicted_objects):
            if isinstance(pred_obj, dict) and pred_obj.get("object_id") == selected_id:
                selected_prediction_index = index
                break
    if selected_prediction_index is None and predicted_objects:
        selected_prediction_index = 0
    selected_match = matches_by_prediction.get(selected_prediction_index) if selected_prediction_index is not None else None
    return {
        "enabled": True,
        "reference_camera": reference_camera,
        "selected_prediction_index": selected_prediction_index,
        "selected_match_distance_px": selected_match.get("distance_px") if isinstance(selected_match, dict) else None,
        "selected_object_id": selected_match.get("matched_truth_object_id") if isinstance(selected_match, dict) else None,
        "object_matches": [
            matches_by_prediction[index]
            for index in sorted(matches_by_prediction)
        ],
    }


def _flat_ai_output(
    raw: dict[str, Any],
    *,
    prompt_mode: str,
    geometry_match: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_obj = _selected_object_from_schema(raw)
    selected_id = raw.get("selected_object_id") or selected_obj.get("object_id")
    if prompt_mode == "image-only" and isinstance(geometry_match, dict):
        selected_id = geometry_match.get("selected_object_id")
    object_confidence = _finite_float(selected_obj.get("confidence"))
    global_confidence = _finite_float(raw.get("global_confidence"))
    confidence = object_confidence if object_confidence is not None else global_confidence
    if confidence is None:
        confidence = 0.0
    flat = {
        "selected_object_id": selected_id,
        "object_id": selected_id,
        "class": selected_obj.get("class") or raw.get("class"),
        "center_2d": selected_obj.get("center_2d") or raw.get("center_2d"),
        "roi": selected_obj.get("roi") or raw.get("roi"),
        "orientation_bucket": selected_obj.get("orientation_bucket") or raw.get("orientation_bucket"),
        "recommended_arm": selected_obj.get("recommended_arm") or raw.get("recommended_arm"),
        "recommended_preset": selected_obj.get("recommended_preset") or raw.get("recommended_preset"),
        "confidence": confidence,
        "confidences": {
            "selected_object_id": confidence,
            "class": confidence,
            "center_2d": confidence,
            "roi": confidence,
            "orientation_bucket": confidence,
            "recommended_arm": confidence,
            "recommended_preset": confidence,
        },
        "model_notes": raw.get("model_notes") or selected_obj.get("notes"),
    }
    return flat


def _case_camera_metadata(case: dict[str, Any], camera_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for camera_name in CAMERA_NAMES:
        artifact = camera_artifacts.get(camera_name)
        if artifact is None:
            continue
        cameras[camera_name] = {
            "npy_path": artifact["path"],
            "sent_to_model": bool(artifact["sent_to_model"]),
            "raw": artifact["raw_stats"],
            "model_input": artifact["model_input_stats"],
        }
    return {
        "case_id": case["case_id"],
        "sample_id": case["sample_id"],
        "seed": case["seed"],
        "camera_names_used_by_model": list(case["camera_names"]),
        "cameras": cameras,
    }


def _write_case_visual_artifacts(
    case_dir: Path,
    *,
    prompt_text: str,
    raw_model_output_text: str | None,
    camera_artifacts: dict[str, dict[str, Any]],
    camera_metadata: dict[str, Any],
    raw_output: dict[str, Any] | None,
    ai_output: dict[str, Any],
    truth: dict[str, Any],
    original_input: dict[str, Any],
    corrected_input: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    raw_model_output_path = case_dir / "raw_model_output.txt"
    if not raw_model_output_path.exists():
        raw_model_output_path.write_text(raw_model_output_text or "", encoding="utf-8")
    for camera_name in CAMERA_NAMES:
        artifact = camera_artifacts.get(camera_name)
        if artifact is None:
            continue
        (case_dir / f"{camera_name}_raw.png").write_bytes(artifact["raw_png_bytes"])
        (case_dir / f"{camera_name}_model_input.png").write_bytes(artifact["model_input_png_bytes"])
    _write_json(case_dir / "metadata.json", camera_metadata)
    _write_json(case_dir / "normalized_output.json", ai_output)
    _write_json(case_dir / "truth.json", truth)
    _write_json(case_dir / "original_input.json", original_input)
    _write_json(case_dir / "corrected_input.json", corrected_input)
    _write_json(case_dir / "metrics.json", metrics)
    if raw_output is not None:
        _write_json(case_dir / "raw_output.json", raw_output)


def _case_outcome(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return _compare_metrics(before, after)


def _evaluate_cases(
    *,
    cases: list[dict[str, Any]],
    provider: Thinker4BProvider | None,
    provider_error: str | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_logs: list[dict[str, Any]] = []
    jsonl_path = output_dir / "cases.jsonl"
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    correction_config = CorrectionConfig(
        enabled=not args.disable_ai_correction,
        min_confidence=float(args.min_confidence),
        max_center_shift_px=float(args.max_center_shift_px),
        max_roi_center_shift_px=float(args.max_roi_center_shift_px),
        max_roi_corner_shift_px=float(args.max_roi_corner_shift_px),
        max_roi_scale_ratio=float(args.max_roi_scale_ratio),
    )
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for case in cases:
            start = time.monotonic()
            raw_output: dict[str, Any] | None = None
            ai_output: dict[str, Any] = {}
            geometry_match: dict[str, Any] | None = None
            thinker_status = "not_run"
            thinker_error = provider_error
            images_meta: list[dict[str, Any]] = []
            prompt_text = _prompt_for_case(case, prompt_mode=args.mode)
            raw_model_output_text: str | None = None
            case_dir = output_dir / case["case_id"] if args.save_debug_artifacts else None
            camera_artifacts: dict[str, dict[str, Any]] = {}
            camera_metadata = {
                "case_id": case["case_id"],
                "sample_id": case["sample_id"],
                "seed": case["seed"],
                "camera_names_used_by_model": list(case["camera_names"]),
                "cameras": {},
            }
            try:
                camera_artifacts = _case_camera_artifacts(case, max_side=int(args.max_image_side))
                camera_metadata = _case_camera_metadata(case, camera_artifacts)
                images_meta = [
                    {
                        "camera_name": camera_name,
                        "path": artifact["path"],
                        "mime_type": artifact["mime_type"],
                        "sent_to_model": bool(artifact["sent_to_model"]),
                        "raw_stats": artifact["raw_stats"],
                        "model_input_stats": artifact["model_input_stats"],
                    }
                    for camera_name, artifact in camera_artifacts.items()
                ]
                provider_images = [
                    {
                        key: value
                        for key, value in camera_artifacts[camera_name].items()
                        if key in {"camera_name", "path", "png_base64", "mime_type"}
                    }
                    for camera_name in case["camera_names"]
                    if camera_name in camera_artifacts
                ]
                if provider is None:
                    raise ThinkerProviderUnavailable(provider_error or "Thinker4B provider unavailable")
                infer_result = provider.infer(
                    case,
                    provider_images,
                    prompt=prompt_text,
                    debug_raw_output_path=(case_dir / "raw_model_output.txt") if case_dir is not None else None,
                )
                raw_model_output_text = infer_result.raw_text
                if case_dir is not None and infer_result.raw_text and not (case_dir / "raw_model_output.txt").exists():
                    case_dir.mkdir(parents=True, exist_ok=True)
                    (case_dir / "raw_model_output.txt").write_text(infer_result.raw_text, encoding="utf-8")
                raw_output = _clean_raw_output(infer_result.payload, case)
                if args.mode == "image-only":
                    geometry_match = _match_objects_by_geometry(
                        raw_output,
                        case,
                        max_match_distance_px=float(args.image_only_match_max_distance_px),
                    )
                ai_output = _flat_ai_output(
                    raw_output,
                    prompt_mode=args.mode,
                    geometry_match=geometry_match,
                )
                thinker_status = "ok"
                thinker_error = None
            except Exception as exc:
                if not args.allow_provider_failure:
                    raise
                thinker_status = "failed"
                thinker_error = str(exc)
                raw_model_output_text = raw_model_output_text or str(exc)

            correction = apply_input_corrections(
                case["original_input"],
                ai_output,
                config=correction_config,
                allowed_object_ids=case["allowed_object_ids"],
            )
            before_metrics = _evaluate_record(
                case["original_input"],
                case["truth"],
                float(args.center_pass_threshold_px),
            )
            thinker_metrics = _evaluate_record(
                ai_output,
                case["truth"],
                float(args.center_pass_threshold_px),
            )
            after_metrics = _evaluate_record(
                correction["corrected_input"],
                case["truth"],
                float(args.center_pass_threshold_px),
            )
            deltas = _case_outcome(before_metrics, after_metrics)
            accepted_count, rejected_count = _accepted_rejected_counts(correction["decisions"])
            log = {
                "format_version": OUTPUT_FORMAT_VERSION,
                "scope": "thinker4b_recorded_camera_input_eval_no_motion",
                "seed": case["seed"],
                "case_id": case["case_id"],
                "case_index": case["case_index"],
                "run_id": case["run_id"],
                "sample_id": case["sample_id"],
                "object_index": case["object_index"],
                "source_pool_index": case["source_pool_index"],
                "reuse_reason": case["reuse_reason"],
                "camera_inputs_used": images_meta,
                "camera_metadata": camera_metadata,
                "original_input": case["original_input"],
                "thinker4b": {
                    "provider": args.provider,
                    "model": args.model or os.environ.get("THINKER4B_MODEL"),
                    "prompt_mode": args.mode,
                    "status": thinker_status,
                    "error": thinker_error,
                    "raw_model_output_text": raw_model_output_text,
                    "raw_output": raw_output,
                    "geometry_match": geometry_match,
                    "normalized_input_output": ai_output,
                },
                "corrected_input": correction["corrected_input"],
                "truth": case["truth"],
                "allowed_object_ids": case["allowed_object_ids"],
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
                    "thinker_raw": thinker_metrics,
                    "after": after_metrics,
                },
                "before_after_deltas": deltas,
                "duration_s": time.monotonic() - start,
            }
            if case_dir is not None:
                _write_case_visual_artifacts(
                    case_dir,
                    prompt_text=prompt_text,
                    raw_model_output_text=raw_model_output_text,
                    camera_artifacts=camera_artifacts,
                    camera_metadata=camera_metadata,
                    raw_output=raw_output,
                    ai_output=ai_output,
                    truth=case["truth"],
                    original_input=case["original_input"],
                    corrected_input=correction["corrected_input"],
                    metrics=log["metrics"],
                )
            _write_json(cases_dir / f"{case['case_id']}.json", log)
            jsonl.write(json.dumps(log, sort_keys=True) + "\n")
            case_logs.append(log)
    summary = _aggregate(case_logs, args=args, provider_error=provider_error, output_dir=output_dir)
    return case_logs, summary


def _count_true(logs: list[dict[str, Any]], path: tuple[str, ...]) -> int:
    count = 0
    for log in logs:
        value: Any = log
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is True:
            count += 1
    return count


def _accuracy(logs: list[dict[str, Any]], phase: str, metric: str) -> float | None:
    applicable = 0
    correct = 0
    for log in logs:
        value = log.get("metrics", {}).get(phase, {}).get(metric)
        if isinstance(value, bool):
            applicable += 1
            if value:
                correct += 1
    if applicable == 0:
        return None
    return correct / applicable


def _mean_metric(logs: list[dict[str, Any]], phase: str, metric: str) -> float | None:
    values: list[float] = []
    for log in logs:
        value = log.get("metrics", {}).get(phase, {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _case_has_usable_center_or_roi(log: dict[str, Any]) -> bool:
    normalized = log.get("thinker4b", {}).get("normalized_input_output", {})
    if not isinstance(normalized, dict):
        return False
    camera_names = log.get("camera_metadata", {}).get("camera_names_used_by_model")
    if not isinstance(camera_names, list) or not camera_names:
        return False
    primary_camera = camera_names[0]
    camera_payload = log.get("camera_metadata", {}).get("cameras", {}).get(primary_camera)
    model_input_stats = camera_payload.get("model_input") if isinstance(camera_payload, dict) else None
    shape = model_input_stats.get("shape") if isinstance(model_input_stats, dict) else None
    if not isinstance(shape, list) or len(shape) < 2:
        return False
    height = _finite_float(shape[0])
    width = _finite_float(shape[1])
    if height is None or width is None or height <= 0 or width <= 0:
        return False
    center = _finite_vector(normalized.get("center_2d"), 2)
    if center is not None:
        if 0.0 <= center[0] < width and 0.0 <= center[1] < height:
            return True
    roi = _finite_vector(normalized.get("roi"), 4)
    if roi is not None:
        if roi[0] < roi[2] and roi[1] < roi[3]:
            if roi[2] > 0.0 and roi[3] > 0.0 and roi[0] < width and roi[1] < height:
                return True
    return False


def _aggregate(logs: list[dict[str, Any]], *, args: argparse.Namespace, provider_error: str | None, output_dir: Path) -> dict[str, Any]:
    total = len(logs)
    requested_total = len(_parse_seeds(args.seeds)) * int(args.cases_per_seed)
    accepted = sum(int(log["correction_summary"]["accepted_count"]) for log in logs)
    rejected = sum(int(log["correction_summary"]["rejected_count"]) for log in logs)
    total_runtime_s = sum(float(log.get("duration_s") or 0.0) for log in logs)
    outcomes = [log["before_after_deltas"]["case_outcome"] for log in logs]
    empty_object_outputs = 0
    usable_center_or_roi = 0
    status_counts: dict[str, int] = {}
    camera_debug_values: dict[str, dict[str, list[Any]]] = {}
    for log in logs:
        status = log["thinker4b"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        raw_output = log.get("thinker4b", {}).get("raw_output")
        objects = raw_output.get("objects") if isinstance(raw_output, dict) else None
        if isinstance(objects, list) and len(objects) == 0:
            empty_object_outputs += 1
        if _case_has_usable_center_or_roi(log):
            usable_center_or_roi += 1
        camera_metadata = log.get("camera_metadata", {}).get("cameras")
        if isinstance(camera_metadata, dict):
            for camera_name, payload in camera_metadata.items():
                if not isinstance(payload, dict):
                    continue
                raw_stats = payload.get("raw")
                if not isinstance(raw_stats, dict):
                    continue
                bucket = camera_debug_values.setdefault(
                    camera_name,
                    {
                        "shape": [],
                        "dtype": [],
                        "mean_brightness": [],
                        "non_black_ratio": [],
                        "bright_ratio": [],
                        "non_black_bbox": [],
                        "bright_bbox": [],
                    },
                )
                for key in ("shape", "dtype", "non_black_bbox", "bright_bbox"):
                    if raw_stats.get(key) is not None:
                        bucket[key].append(raw_stats.get(key))
                for key in ("mean_brightness", "non_black_ratio", "bright_ratio"):
                    value = _finite_float(raw_stats.get(key))
                    if value is not None:
                        bucket[key].append(value)
    bool_metrics = {
        "class": "class_correct",
        "selected_object": "selected_object_correct",
        "orientation_bucket": "orientation_bucket_correct",
        "arm_recommendation": "arm_recommendation_correct",
        "preset_recommendation": "preset_recommendation_correct",
    }
    before_accuracy = {}
    thinker_accuracy = {}
    after_accuracy = {}
    for label, metric in bool_metrics.items():
        before_accuracy[label] = _accuracy(logs, "before", metric)
        thinker_accuracy[label] = _accuracy(logs, "thinker_raw", metric)
        after_accuracy[label] = _accuracy(logs, "after", metric)
    camera_debug_overview: dict[str, Any] = {}
    for camera_name, values in camera_debug_values.items():
        camera_debug_overview[camera_name] = {
            "shapes_seen": sorted({tuple(item) for item in values["shape"]}),
            "dtypes_seen": sorted({str(item) for item in values["dtype"]}),
            "mean_brightness_range": (
                [min(values["mean_brightness"]), max(values["mean_brightness"])]
                if values["mean_brightness"]
                else None
            ),
            "non_black_ratio_range": (
                [min(values["non_black_ratio"]), max(values["non_black_ratio"])]
                if values["non_black_ratio"]
                else None
            ),
            "bright_ratio_range": (
                [min(values["bright_ratio"]), max(values["bright_ratio"])]
                if values["bright_ratio"]
                else None
            ),
            "example_non_black_bbox": values["non_black_bbox"][0] if values["non_black_bbox"] else None,
            "example_bright_bbox": values["bright_bbox"][0] if values["bright_bbox"] else None,
        }
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "thinker4b_recorded_camera_input_eval_no_motion",
        "provider": {
            "name": args.provider,
            "model": args.model or os.environ.get("THINKER4B_MODEL"),
            "api_base_configured": bool(args.api_base or os.environ.get("THINKER4B_API_BASE")),
            "command_configured": bool(args.thinker_command or os.environ.get("THINKER4B_CMD")),
            "provider_setup_error": provider_error,
            "allow_provider_failure": bool(args.allow_provider_failure),
        },
        "prompt": {
            "mode": args.mode,
            "original_input_in_prompt": args.mode != "image-only",
            "allowed_object_ids_in_prompt": args.mode != "image-only",
            "save_debug_artifacts": bool(args.save_debug_artifacts),
            "image_only_match_max_distance_px": (
                float(args.image_only_match_max_distance_px) if args.mode == "image-only" else None
            ),
        },
        "inputs": {
            "run_id": args.run_id,
            "dataset_root": str(_dataset_root(args)),
            "seeds": _parse_seeds(args.seeds),
            "cases_per_seed": int(args.cases_per_seed),
            "requested_cameras": _parse_csv(args.cameras),
            "include_truth_camera": bool(args.include_truth_camera),
        },
        "output_dir": str(output_dir),
        "requested_case_count": requested_total,
        "case_count": total,
        "runtime_s": total_runtime_s,
        "thinker_status_counts": status_counts,
        "objects_empty_count": empty_object_outputs,
        "cases_with_usable_center_or_roi": usable_center_or_roi,
        "accepted_ai_corrections": accepted,
        "rejected_ai_corrections": rejected,
        "cases_improved": outcomes.count("improved"),
        "cases_unchanged": outcomes.count("unchanged"),
        "cases_worsened": outcomes.count("worsened"),
        "before_accuracy": before_accuracy,
        "thinker_raw_accuracy": thinker_accuracy,
        "after_accuracy": after_accuracy,
        "before_center_2d_error_mean_px": _mean_metric(logs, "before", "center_2d_error_px"),
        "thinker_center_2d_error_mean_px": _mean_metric(logs, "thinker_raw", "center_2d_error_px"),
        "after_center_2d_error_mean_px": _mean_metric(logs, "after", "center_2d_error_px"),
        "camera_debug_overview": camera_debug_overview,
        "limitations": [
            "This is recorded-camera input evaluation only.",
            "No robot execution, IK, planner integration, final grasp pose, or motion control is performed.",
            "Original inputs are deterministic generated estimates unless an external original-input source is added later.",
            (
                "In image-only mode, dataset object ids are excluded from the prompt and "
                "selected-object scoring uses external 2D geometry matching."
            ),
        ],
    }


def _format_json_for_report(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _portable_path(path_value: str) -> str:
    path = str(path_value)
    replacements = (
        ("OUTPUT_ROOT", os.environ.get("OUTPUT_ROOT")),
        ("LOG_ROOT", os.environ.get("LOG_ROOT")),
        ("DATA_ROOT", os.environ.get("DATA_ROOT")),
        ("CKPT_ROOT", os.environ.get("CKPT_ROOT")),
        ("HRC_REPO", os.environ.get("HRC_REPO")),
        ("HRC_ROOT", os.environ.get("HRC_ROOT")),
    )
    for env_name, env_value in replacements:
        if env_value and path == env_value:
            return f"${env_name}"
        if env_value and path.startswith(env_value.rstrip("/") + "/"):
            return f"${env_name}/{path[len(env_value.rstrip('/')) + 1:]}"
    return path


def _portable_for_report(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_for_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_for_report(item) for item in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return _portable_path(value)
        return value
    return value


def _write_human_report(path: Path, *, case_logs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines: list[str] = []
    portable_summary = _portable_for_report(summary)
    lines.append("Task 1 Thinker4B Recorded-Camera Input Evaluation")
    lines.append("=" * 58)
    lines.append("")
    lines.append("Scope: input-evaluation only. No robot execution, no final grasp pose, no IK, no planner or motion integration.")
    lines.append(f"Generated at UTC: {summary['generated_at_utc']}")
    lines.append(f"Output dir: {portable_summary['output_dir']}")
    lines.append("")
    lines.append("Provider")
    lines.append("-" * 8)
    lines.append(_format_json_for_report(portable_summary["provider"]))
    lines.append("")
    lines.append("Prompt")
    lines.append("-" * 6)
    lines.append(_format_json_for_report(portable_summary["prompt"]))
    lines.append("")
    lines.append("Cases")
    lines.append("-" * 5)
    for log in case_logs:
        portable_log = _portable_for_report(log)
        lines.append("")
        lines.append(f"Case {log['case_id']}")
        lines.append("~" * (5 + len(log["case_id"])))
        lines.append(f"Seed: {log['seed']}")
        lines.append(f"Run ID: {log['run_id']}")
        lines.append(f"Sample ID: {log['sample_id']}")
        lines.append(f"Object index: {log['object_index']}")
        lines.append(f"Camera inputs used: {_format_json_for_report(portable_log['camera_inputs_used'])}")
        lines.append(f"Thinker4B status: {log['thinker4b']['status']}")
        if log["thinker4b"].get("error"):
            lines.append(f"Thinker4B error: {log['thinker4b']['error']}")
        lines.append("")
        lines.append("Truth data:")
        lines.append(_format_json_for_report(portable_log["truth"]))
        lines.append("")
        lines.append("Original input:")
        lines.append(_format_json_for_report(portable_log["original_input"]))
        lines.append("")
        lines.append("Thinker4B raw data:")
        lines.append(_format_json_for_report(portable_log["thinker4b"]["raw_output"]))
        lines.append("")
        lines.append("Thinker4B geometry match:")
        lines.append(_format_json_for_report(portable_log["thinker4b"]["geometry_match"]))
        lines.append("")
        lines.append("Thinker4B normalized input-level output:")
        lines.append(_format_json_for_report(portable_log["thinker4b"]["normalized_input_output"]))
        lines.append("")
        lines.append("Corrected data:")
        lines.append(_format_json_for_report(portable_log["corrected_input"]))
        lines.append("")
        lines.append("Correction decisions:")
        lines.append(_format_json_for_report(portable_log["correction_decisions"]))
        lines.append("")
        lines.append("Before / Thinker / After metrics:")
        lines.append(_format_json_for_report(portable_log["metrics"]))
        lines.append("")
        lines.append("Decision:")
        lines.append(_format_json_for_report(portable_log["before_after_deltas"]))
    lines.append("")
    lines.append("Aggregate Summary")
    lines.append("-" * 17)
    lines.append(_format_json_for_report(portable_summary))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any], report_path: Path) -> None:
    print("Task 1 Thinker4B input evaluation")
    print(f"Scope: {summary['scope']}")
    print(f"Output dir: {summary['output_dir']}")
    print(f"Report: {report_path}")
    print(f"Cases: {summary['case_count']}")
    print(f"Thinker statuses: {summary['thinker_status_counts']}")
    print(
        "Correction counts: "
        f"accepted={summary['accepted_ai_corrections']}, "
        f"rejected={summary['rejected_ai_corrections']}"
    )
    print(
        "Case outcomes: "
        f"improved={summary['cases_improved']}, "
        f"unchanged={summary['cases_unchanged']}, "
        f"worsened={summary['cases_worsened']}"
    )
    print(
        "Center 2D mean error px before/Thinker/after: "
        f"{summary['before_center_2d_error_mean_px']} / "
        f"{summary['thinker_center_2d_error_mean_px']} / "
        f"{summary['after_center_2d_error_mean_px']}"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", help="Phase 1 run directory.")
    parser.add_argument("--run-id", default="test_phase1_initfix_1", help="Run id under $OUTPUT_ROOT/datasets/task1_rgbd_labels.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--cases-per-seed", type=int, default=DEFAULT_CASES_PER_SEED)
    parser.add_argument("--mode", choices=PROMPT_MODES, default="image-only", help="Prompt/evaluation mode. Default: image-only.")
    parser.add_argument("--cameras", default="head_left,head_right", help="Comma-separated preferred camera inputs.")
    parser.add_argument("--include-truth-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider", choices=("openai-compatible", "ollama", "command", "cache"), default="openai-compatible")
    parser.add_argument("--model", help="Thinker4B model id. Defaults to $THINKER4B_MODEL.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base. Defaults to $THINKER4B_API_BASE.")
    parser.add_argument("--api-key", help="OpenAI-compatible API key. Defaults to $THINKER4B_API_KEY.")
    parser.add_argument("--ollama-host", help="Ollama host. Defaults to $OLLAMA_HOST or http://localhost:11434.")
    parser.add_argument("--thinker-command", help="Command provider. Defaults to $THINKER4B_CMD.")
    parser.add_argument("--thinker-output-cache", help="JSON/JSONL cache of real Thinker4B outputs for provider=cache.")
    parser.add_argument("--allow-provider-failure", action="store_true", help="Write logs/report even if Thinker4B provider is unavailable or fails.")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--image-only-match-max-distance-px", type=float, default=80.0)
    parser.add_argument("--save-debug-artifacts", action="store_true", help="Save prompt, PNG inputs, and raw provider text under <output_dir>/debug/<case_id>.")
    parser.add_argument("--output-dir", help="Runtime output dir. Defaults under $OUTPUT_ROOT/test_runs/task1_thinker4b_input_eval/.")
    parser.add_argument("--report-path", default="docs/output02.txt")
    parser.add_argument("--disable-ai-correction", action="store_true")
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
        seeds = _parse_seeds(args.seeds)
        if args.cases_per_seed <= 0:
            raise Thinker4BEvalError("--cases-per-seed must be positive")
        requested_cameras = _parse_csv(args.cameras)
        for camera_name in requested_cameras:
            if camera_name not in CAMERA_NAMES:
                raise Thinker4BEvalError(f"unsupported camera {camera_name!r}; expected one of {CAMERA_NAMES}")
        dataset_root = _dataset_root(args)
        pool = _build_case_pool(dataset_root=dataset_root, preferred_camera=requested_cameras[0] if requested_cameras else "head_left")
        if args.mode == "image-only" and requested_cameras:
            primary_camera = requested_cameras[0]
            visible_pool = [
                item
                for item in pool
                if item.get("truth", {}).get("camera_name") == primary_camera
            ]
            if visible_pool:
                pool = visible_pool
            else:
                raise Thinker4BEvalError(
                    f"no cases have the target visible in primary image-only camera {primary_camera!r}"
                )
        cases = _select_cases(
            pool=pool,
            seeds=seeds,
            cases_per_seed=int(args.cases_per_seed),
            requested_cameras=requested_cameras,
            include_truth_camera=bool(args.include_truth_camera),
        )
        provider: Thinker4BProvider | None = None
        provider_error: str | None = None
        try:
            provider = _make_provider(args)
        except ThinkerProviderUnavailable as exc:
            provider_error = str(exc)
            if not args.allow_provider_failure:
                raise
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(dataset_root.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        case_logs, summary = _evaluate_cases(
            cases=cases,
            provider=provider,
            provider_error=provider_error,
            args=args,
            output_dir=output_dir,
        )
        _write_json(output_dir / "summary.json", summary)
        report_path = Path(args.report_path).expanduser()
        _write_human_report(report_path, case_logs=case_logs, summary=summary)
        _print_summary(summary, report_path)
        if provider_error and not args.allow_provider_failure:
            return 2
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
