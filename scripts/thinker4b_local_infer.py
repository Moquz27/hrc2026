#!/usr/bin/env python3
"""Local Thinker-4B command wrapper for recorded-camera input evaluation.

This wrapper is intentionally limited to input-level perception output. It can
be called directly with image paths, or by
`scripts/task1_run_thinker4b_input_eval.py --provider command`, which passes a
JSON object on stdin containing a `request_path`.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL_ID = "UBTECH-Robotics/Thinker-4B"
DEFAULT_MAX_NEW_TOKENS = 768


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _coerce_schema(payload: dict[str, Any] | None, *, frame_id: str, raw_text: str | None) -> dict[str, Any]:
    """Return a schema-compatible object without inventing missing predictions."""
    if payload is None:
        return {
            "frame_id": frame_id,
            "selected_object_id": None,
            "global_confidence": 0.0,
            "objects": [],
            "model_notes": (
                "Thinker4B returned text that could not be parsed as JSON. "
                f"Raw output: {raw_text[:1000] if raw_text else ''}"
            ),
        }

    objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    cleaned_objects: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key in (
            "object_id",
            "class",
            "roi",
            "center_2d",
            "orientation_bucket",
            "difficulty",
            "occlusion",
            "confidence",
            "recommended_arm",
            "recommended_preset",
            "notes",
        ):
            if key in item:
                cleaned[key] = item[key]
        cleaned_objects.append(cleaned)

    selected_object_id = payload.get("selected_object_id")
    if selected_object_id is not None and not isinstance(selected_object_id, str):
        selected_object_id = None

    confidence = payload.get("global_confidence", payload.get("confidence", 0.0))
    try:
        global_confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        global_confidence = 0.0

    result = {
        "frame_id": str(payload.get("frame_id") or frame_id),
        "selected_object_id": selected_object_id,
        "global_confidence": global_confidence,
        "objects": cleaned_objects,
    }
    if isinstance(payload.get("model_notes"), str):
        result["model_notes"] = payload["model_notes"]
    elif raw_text:
        result["model_notes"] = raw_text[:1000]
    return result


def _image_from_base64(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def _resize_image(image: Image.Image, *, max_side: int) -> Image.Image:
    if max_side > 0 and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    return image.convert("RGB")


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    rgb = np.asarray(array)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected RGB/RGBA HxWxC array, got shape {list(rgb.shape)}")
    rgb = rgb[:, :, :3]
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    rgb = rgb.astype(np.float32, copy=False)
    finite = np.isfinite(rgb)
    if np.any(finite) and float(np.nanmax(rgb[finite])) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def _open_image(path: str) -> Image.Image:
    image_path = Path(path)
    if image_path.suffix.lower() == ".npy":
        array = np.load(image_path, allow_pickle=False)
        return Image.fromarray(_to_uint8_rgb(array))
    return Image.open(image_path).convert("RGB")


def _load_images_from_request(request: dict[str, Any], *, max_side: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for item in request.get("images", []):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("png_base64"), str):
            images.append(_resize_image(_image_from_base64(item["png_base64"]), max_side=max_side))
        elif isinstance(item.get("path"), str):
            images.append(_resize_image(_open_image(item["path"]), max_side=max_side))
    return images


def _load_images_from_paths(paths: list[str], *, max_side: int) -> list[Image.Image]:
    return [_resize_image(_open_image(path), max_side=max_side) for path in paths]


def _request_from_stdin() -> dict[str, Any] | None:
    if sys.stdin.isatty():
        return None
    text = sys.stdin.read().strip()
    if not text:
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("stdin JSON must be an object")
    request_path = payload.get("request_path")
    if isinstance(request_path, str):
        return _read_json(Path(request_path))
    return payload


def _build_direct_request(args: argparse.Namespace) -> dict[str, Any]:
    prompt = args.prompt or (
        "You are Thinker4B analyzing recorded camera images from HRC Task 1. "
        "Return EXACTLY one JSON object with keys frame_id, selected_object_id, "
        "global_confidence, objects, and optional model_notes. "
        "Each object may contain object_id, class, center_2d, roi, "
        "orientation_bucket, difficulty, occlusion, confidence, "
        "recommended_arm, recommended_preset, and notes. "
        "Do not use markdown fences. Do not output final grasp poses, world poses, "
        "joint commands, trajectories, or extra prose. "
        "If uncertain, use null or unknown and low confidence."
    )
    return {
        "prompt": prompt,
        "case": {"case_id": args.frame_id, "sample_id": args.frame_id},
        "images": [{"path": path, "camera_name": f"image_{index}"} for index, path in enumerate(args.image)],
    }


class LocalThinker4B:
    def __init__(
        self,
        *,
        model: str,
        device_map: str,
        dtype: str,
        attn_implementation: str | None,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": dtype,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model)
        self.torch = torch

    def generate(
        self,
        *,
        prompt: str,
        images: list[Image.Image],
        max_new_tokens: int,
    ) -> str:
        temp_dir = Path(tempfile.mkdtemp(prefix="thinker4b_images_"))
        try:
            content: list[dict[str, Any]] = []
            for index, image in enumerate(images):
                image_path = temp_dir / f"image_{index:02d}.png"
                image.save(image_path, format="PNG")
                content.append({"type": "image", "image": str(image_path)})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                )
            generated_ids_trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0] if decoded else ""
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("THINKER4B_MODEL_PATH") or os.environ.get("THINKER4B_MODEL") or DEFAULT_MODEL_ID)
    parser.add_argument("--image", action="append", default=[], help="Direct image path. May be provided multiple times.")
    parser.add_argument("--prompt", help="Direct-mode prompt. Ignored when stdin/request JSON provides a prompt.")
    parser.add_argument("--frame-id", default="direct_image")
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--debug-raw-output", help="Optional path to save the raw model text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        request = _request_from_stdin() or _build_direct_request(args)
        if not request.get("prompt"):
            raise ValueError("request prompt is required")
        if not request.get("images") and not args.image:
            raise ValueError("at least one image is required")
        images = _load_images_from_request(request, max_side=args.max_image_side)
        if not images:
            images = _load_images_from_paths(args.image, max_side=args.max_image_side)
        if not images:
            raise ValueError("no loadable images found")

        case = request.get("case") if isinstance(request.get("case"), dict) else {}
        frame_id = str(case.get("case_id") or case.get("sample_id") or args.frame_id)

        _stderr(f"loading Thinker4B model: {args.model}")
        runner = LocalThinker4B(
            model=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        _stderr(f"running Thinker4B inference on {len(images)} image(s)")
        raw_text = runner.generate(
            prompt=str(request["prompt"]),
            images=images,
            max_new_tokens=args.max_new_tokens,
        )
        if args.debug_raw_output:
            debug_path = Path(args.debug_raw_output).expanduser()
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw_text, encoding="utf-8")

        payload = _extract_json_object(raw_text)
        print(json.dumps(_coerce_schema(payload, frame_id=frame_id, raw_text=raw_text), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"thinker4b_local_infer_error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
