#!/usr/bin/env python3
"""Mask selection, deterministic compositing, resizing, preview, and auditing."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageColor


HERE = pathlib.Path(__file__).resolve().parent
RUNTIME_CONTRACT = HERE.parent / "references" / "runtime-contract.json"


def runtime_contract() -> dict[str, Any]:
    """Load the deployment-owned model identities used by regular operations."""

    value = json.loads(RUNTIME_CONTRACT.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "pixeltops-image-runtime.v1":
        raise RuntimeError("runtime contract is invalid")
    return value


def model_declaration(name: str) -> tuple[str, str]:
    """Return one exact Hugging Face model identity and revision."""

    models = runtime_contract().get("huggingface_models")
    if not isinstance(models, dict):
        raise RuntimeError("runtime contract huggingface_models must be an object")
    declaration = models.get(name)
    if not isinstance(declaration, dict):
        raise RuntimeError(f"runtime model declaration is missing: {name}")
    model_id = declaration.get("id")
    revision = declaration.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise RuntimeError(f"runtime model declaration is invalid: {name}")
    return model_id, revision


GROUNDING_MODEL_ID, GROUNDING_REVISION = model_declaration("grounding_dino")
SAM2_MODEL_ID, SAM2_REVISION = model_declaration("sam2")


def emit(payload: dict[str, Any], *, error: bool = False) -> int:
    print(json.dumps(payload, separators=(",", ":")), file=sys.stderr if error else sys.stdout)
    return 1 if error else 0


def models_root() -> pathlib.Path:
    codex = pathlib.Path(os.environ.get("CODEX_HOME") or pathlib.Path.home() / ".codex")
    contract = runtime_contract()
    runtime_path = contract.get("runtime_relative_path")
    cache_path = contract.get("huggingface_cache")
    if not isinstance(runtime_path, str) or not isinstance(cache_path, str):
        raise RuntimeError("runtime contract model paths are invalid")
    return codex / pathlib.Path(runtime_path) / pathlib.Path(cache_path)


def model_snapshot(model_id: str, revision: str) -> pathlib.Path:
    return models_root() / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision


def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rgb(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(require_file(path)).convert("RGB"), dtype=np.uint8)


def load_rgba(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(require_file(path)).convert("RGBA"), dtype=np.uint8)


def load_mask(path: pathlib.Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(require_file(path)).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8) * 255


def atomic_save(array: np.ndarray, path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix or '.png'}")
    mode = "L" if array.ndim == 2 else "RGBA" if array.shape[2] == 4 else "RGB"
    Image.fromarray(array, mode=mode).save(temporary)
    os.replace(temporary, path)


def morph_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels == 0:
        return mask
    radius = abs(pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    operation = cv2.dilate if pixels > 0 else cv2.erode
    return operation(mask, kernel, iterations=1)


def parse_polygon(text: str, width: int, height: int) -> np.ndarray:
    points: list[tuple[int, int]] = []
    for pair in text.split(";"):
        x_text, y_text = pair.split(",", 1)
        x, y = int(round(float(x_text))), int(round(float(y_text)))
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"polygon point outside image: {x},{y}")
        points.append((x, y))
    if len(points) < 3:
        raise ValueError("polygon lasso requires at least three points")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


def text_object_mask(image: Image.Image, text: str, threshold: float, box_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Config,
        Sam2Model,
        Sam2Processor,
        Sam2VideoConfig,
    )

    grounding_path = model_snapshot(GROUNDING_MODEL_ID, GROUNDING_REVISION)
    sam_path = model_snapshot(SAM2_MODEL_ID, SAM2_REVISION)
    if not grounding_path.is_dir() or not sam_path.is_dir():
        raise FileNotFoundError("pinned Grounding DINO or SAM 2.1 snapshot is missing")

    prompt = text.strip().lower()
    if not prompt:
        raise ValueError("text selection cannot be empty")
    if not prompt.endswith("."):
        prompt += "."

    grounding_processor = AutoProcessor.from_pretrained(grounding_path, local_files_only=True)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_path, local_files_only=True).eval()
    grounding_inputs = grounding_processor(images=image, text=prompt, return_tensors="pt")
    with torch.inference_mode():
        grounding_outputs = grounding_model(**grounding_inputs)
    detected = grounding_processor.post_process_grounded_object_detection(
        grounding_outputs,
        grounding_inputs.input_ids,
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    order = torch.argsort(detected["scores"], descending=True)
    boxes = detected["boxes"][order]
    scores = detected["scores"][order]
    labels = [detected["text_labels"][int(index)] for index in order]
    if box_index < 0 or box_index >= len(boxes):
        raise ValueError(f"box index {box_index} is unavailable; detected {len(boxes)} candidate(s)")
    selected_box = boxes[box_index].tolist()

    video_config = Sam2VideoConfig.from_pretrained(sam_path, local_files_only=True)
    image_config = Sam2Config(
        vision_config=video_config.vision_config,
        prompt_encoder_config=video_config.prompt_encoder_config,
        mask_decoder_config=video_config.mask_decoder_config,
        initializer_range=video_config.initializer_range,
    )
    sam_model = Sam2Model.from_pretrained(sam_path, config=image_config, local_files_only=True).eval()
    sam_processor = Sam2Processor.from_pretrained(sam_path, local_files_only=True)
    sam_inputs = sam_processor(images=image, input_boxes=[[selected_box]], return_tensors="pt")
    with torch.inference_mode():
        sam_outputs = sam_model(**sam_inputs)
    masks = sam_processor.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
    best = int(sam_outputs.iou_scores[0, 0].argmax())
    mask = (masks[0, best].numpy() > 0).astype(np.uint8) * 255
    metadata = {
        "method": "grounding-dino+sam2.1",
        "query": text,
        "candidateCount": len(boxes),
        "selectedIndex": box_index,
        "selectedBox": [round(float(value), 2) for value in selected_box],
        "selectedScore": round(float(scores[box_index]), 6),
        "selectedLabel": labels[box_index],
        "groundingRevision": GROUNDING_REVISION,
        "sam2Revision": SAM2_REVISION,
    }
    return mask, metadata


def transition_alpha(mask: np.ndarray, feather: float) -> tuple[np.ndarray, np.ndarray]:
    binary = mask >= 128
    if feather <= 0:
        return binary.astype(np.float32), binary
    radius = max(1, int(math.ceil(feather * 3)))
    allowed = morph_mask(mask, radius) >= 128
    alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=feather, sigmaY=feather)
    alpha[~allowed] = 0.0
    return np.clip(alpha, 0.0, 1.0), allowed


def stable_foreground(mask: np.ndarray, feather: float) -> np.ndarray:
    binary = mask >= 128
    if feather <= 0:
        return binary
    radius = max(1, int(math.ceil(feather * 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)


def blend_exact(base: np.ndarray, edited: np.ndarray, alpha: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    result = np.rint(edited.astype(np.float32) * alpha[..., None] + base.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)
    result[~allowed] = base[~allowed]
    return result


def verify_unchanged(original: np.ndarray, result: np.ndarray, allowed: np.ndarray) -> dict[str, Any]:
    if original.shape != result.shape:
        return {"status": "ERROR", "reason": f"shape mismatch: {original.shape} != {result.shape}"}
    changed = np.any(original != result, axis=2)
    outside_changed = int(np.count_nonzero(changed & ~allowed))
    return {
        "status": "OK" if outside_changed == 0 else "ERROR",
        "outsideChangedPixels": outside_changed,
        "allowedPixels": int(np.count_nonzero(allowed)),
        "changedPixels": int(np.count_nonzero(changed)),
    }


def color_rgba(text: str) -> tuple[int, int, int, int]:
    if text.lower() == "transparent":
        return 0, 0, 0, 0
    rgb = ImageColor.getrgb(text)
    return (*rgb[:3], 255)


def fit_image(image: Image.Image, width: int, height: int, mode: str, color: str = "transparent") -> Image.Image:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    source_width, source_height = image.size
    scale = min(width / source_width, height / source_height) if mode == "contain" else max(width / source_width, height / source_height)
    if mode == "cover":
        resized_size = (max(width, math.ceil(source_width * scale)), max(height, math.ceil(source_height * scale)))
    else:
        resized_size = (min(width, max(1, math.floor(source_width * scale))), min(height, max(1, math.floor(source_height * scale))))
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    if mode == "cover":
        left = max(0, (resized.width - width) // 2)
        top = max(0, (resized.height - height) // 2)
        return resized.crop((left, top, left + width, top + height))
    canvas = Image.new("RGBA", (width, height), color_rgba(color))
    canvas.alpha_composite(resized.convert("RGBA"), ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def prepare_resize_output(image: Image.Image, output: pathlib.Path, color: str) -> Image.Image:
    """Flatten transparency only when required by a JPEG destination."""

    if output.suffix.lower() not in {".jpg", ".jpeg"}:
        return image
    rgba = image.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] < 255:
        if color.lower() == "transparent":
            raise ValueError("JPEG output cannot contain transparency; pass an opaque --color")
        matte = Image.new("RGBA", rgba.size, color_rgba(color))
        matte.alpha_composite(rgba)
        rgba = matte
    return rgba.convert("RGB")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("input", type=pathlib.Path)
    select.add_argument("output", type=pathlib.Path)
    source = select.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--polygon")
    source.add_argument("--mask", type=pathlib.Path)
    select.add_argument("--box-index", type=int, default=0)
    select.add_argument("--threshold", type=float, default=0.25)
    select.add_argument("--expand", type=int, default=0)
    select.add_argument("--preview", type=pathlib.Path)
    select.add_argument("--overwrite", action="store_true")

    remove = commands.add_parser("remove-background")
    remove.add_argument("input", type=pathlib.Path)
    remove.add_argument("mask", type=pathlib.Path)
    remove.add_argument("output", type=pathlib.Path)
    remove.add_argument("--allowed-mask-output", type=pathlib.Path, required=True)
    remove.add_argument("--feather", type=float, default=0.0)
    remove.add_argument("--overwrite", action="store_true")

    replace = commands.add_parser("replace-background")
    replace.add_argument("input", type=pathlib.Path)
    replace.add_argument("mask", type=pathlib.Path)
    replace.add_argument("output", type=pathlib.Path)
    background = replace.add_mutually_exclusive_group(required=True)
    background.add_argument("--background", type=pathlib.Path)
    background.add_argument("--color")
    replace.add_argument("--background-mode", choices=("cover", "contain", "stretch"), default="cover")
    replace.add_argument("--allowed-mask-output", type=pathlib.Path, required=True)
    replace.add_argument("--feather", type=float, default=0.0)
    replace.add_argument("--overwrite", action="store_true")

    composite = commands.add_parser("composite")
    composite.add_argument("base", type=pathlib.Path)
    composite.add_argument("edited", type=pathlib.Path)
    composite.add_argument("mask", type=pathlib.Path)
    composite.add_argument("output", type=pathlib.Path)
    composite.add_argument("--method", choices=("hard", "feather", "seamless"), default="hard")
    composite.add_argument("--feather", type=float, default=0.0)
    composite.add_argument("--transition", type=int, default=8)
    composite.add_argument("--allowed-mask-output", type=pathlib.Path, required=True)
    composite.add_argument("--overwrite", action="store_true")

    resize = commands.add_parser("resize")
    resize.add_argument("input", type=pathlib.Path)
    resize.add_argument("output", type=pathlib.Path)
    resize.add_argument("width", type=int)
    resize.add_argument("height", type=int)
    resize.add_argument("--mode", choices=("contain", "cover", "stretch"), default="contain")
    resize.add_argument("--color", default="transparent")
    resize.add_argument("--overwrite", action="store_true")

    preview = commands.add_parser("preview")
    preview.add_argument("input", type=pathlib.Path)
    preview.add_argument("mask", type=pathlib.Path)
    preview.add_argument("output", type=pathlib.Path)
    preview.add_argument("--color", default="#ff00ff")
    preview.add_argument("--opacity", type=float, default=0.45)
    preview.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("original", type=pathlib.Path)
    verify.add_argument("result", type=pathlib.Path)
    verify.add_argument("allowed_mask", type=pathlib.Path)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "select":
            image = Image.open(require_file(args.input)).convert("RGB")
            if args.text:
                mask, metadata = text_object_mask(image, args.text, args.threshold, args.box_index)
            elif args.polygon:
                mask = parse_polygon(args.polygon, image.width, image.height)
                metadata = {"method": "polygon-lasso"}
            else:
                mask = load_mask(args.mask, image.size)
                metadata = {"method": "supplied-mask", "source": str(args.mask.resolve())}
            mask = morph_mask(mask, args.expand)
            atomic_save(mask, args.output, args.overwrite)
            if args.preview:
                rgb = np.asarray(image, dtype=np.uint8)
                overlay = rgb.copy()
                overlay[mask >= 128] = (255, 0, 255)
                preview = np.rint(rgb * 0.55 + overlay * 0.45).astype(np.uint8)
                atomic_save(preview, args.preview, args.overwrite)
            metadata.update({"status": "OK", "mask": str(args.output.resolve()), "pixels": int(np.count_nonzero(mask)), "width": image.width, "height": image.height})
            return emit(metadata)

        if args.command == "remove-background":
            base = load_rgb(args.input)
            mask = load_mask(args.mask, (base.shape[1], base.shape[0]))
            alpha, _ = transition_alpha(mask, args.feather)
            rgba = np.dstack((base, np.rint(alpha * 255).astype(np.uint8)))
            allowed = ~stable_foreground(mask, args.feather)
            atomic_save(rgba, args.output, args.overwrite)
            atomic_save(allowed.astype(np.uint8) * 255, args.allowed_mask_output, args.overwrite)
            return emit({"status": "OK", "output": str(args.output.resolve()), "allowedMask": str(args.allowed_mask_output.resolve()), "foregroundRgbPreserved": True})

        if args.command == "replace-background":
            base = load_rgb(args.input)
            height, width = base.shape[:2]
            mask = load_mask(args.mask, (width, height))
            alpha, _ = transition_alpha(mask, args.feather)
            if args.background:
                background_image = fit_image(Image.open(require_file(args.background)).convert("RGB"), width, height, args.background_mode, "black").convert("RGB")
                background = np.asarray(background_image, dtype=np.uint8)
            else:
                background = np.full_like(base, ImageColor.getrgb(args.color))
            result = np.rint(base.astype(np.float32) * alpha[..., None] + background.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)
            stable = stable_foreground(mask, args.feather)
            result[stable] = base[stable]
            allowed = ~stable
            atomic_save(result, args.output, args.overwrite)
            atomic_save(allowed.astype(np.uint8) * 255, args.allowed_mask_output, args.overwrite)
            return emit({"status": "OK", "output": str(args.output.resolve()), "allowedMask": str(args.allowed_mask_output.resolve()), "preservedForegroundPixels": int(np.count_nonzero(stable))})

        if args.command == "composite":
            base = load_rgb(args.base)
            edited = load_rgb(args.edited)
            if base.shape != edited.shape:
                raise ValueError("base and edited images must have identical dimensions")
            mask = load_mask(args.mask, (base.shape[1], base.shape[0]))
            if args.method == "seamless":
                allowed = morph_mask(mask, max(0, args.transition)) >= 128
                ys, xs = np.nonzero(mask >= 128)
                if len(xs) == 0:
                    raise ValueError("mask is empty")
                center = (int(round((xs.min() + xs.max()) / 2)), int(round((ys.min() + ys.max()) / 2)))
                cloned = cv2.seamlessClone(edited[:, :, ::-1], base[:, :, ::-1], mask, center, cv2.NORMAL_CLONE)[:, :, ::-1]
                result = cloned
                result[~allowed] = base[~allowed]
            else:
                feather = args.feather if args.method == "feather" else 0.0
                alpha, allowed = transition_alpha(mask, feather)
                result = blend_exact(base, edited, alpha, allowed)
            audit = verify_unchanged(base, result, allowed)
            if audit["status"] != "OK":
                raise RuntimeError(f"outside-mask verification failed: {audit}")
            atomic_save(result, args.output, args.overwrite)
            atomic_save(allowed.astype(np.uint8) * 255, args.allowed_mask_output, args.overwrite)
            return emit({"status": "OK", "output": str(args.output.resolve()), "allowedMask": str(args.allowed_mask_output.resolve()), "audit": audit, "method": args.method})

        if args.command == "resize":
            source = Image.open(require_file(args.input)).convert("RGBA")
            result = fit_image(source, args.width, args.height, args.mode, args.color)
            result = prepare_resize_output(result, args.output, args.color)
            array = np.asarray(result, dtype=np.uint8)
            atomic_save(array, args.output, args.overwrite)
            return emit({"status": "OK", "output": str(args.output.resolve()), "width": args.width, "height": args.height, "mode": args.mode})

        if args.command == "preview":
            base = load_rgb(args.input)
            mask = load_mask(args.mask, (base.shape[1], base.shape[0])) >= 128
            opacity = min(1.0, max(0.0, args.opacity))
            color = np.asarray(ImageColor.getrgb(args.color), dtype=np.float32)
            result = base.copy()
            result[mask] = np.rint(base[mask].astype(np.float32) * (1.0 - opacity) + color * opacity).astype(np.uint8)
            atomic_save(result, args.output, args.overwrite)
            return emit({"status": "OK", "output": str(args.output.resolve()), "maskPixels": int(np.count_nonzero(mask))})

        if args.command == "verify":
            original = load_rgb(args.original)
            result = load_rgb(args.result)
            allowed = load_mask(args.allowed_mask, (original.shape[1], original.shape[0])) >= 128
            audit = verify_unchanged(original, result, allowed)
            return emit(audit, error=audit["status"] != "OK")

        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        return emit({"status": "ERROR", "command": args.command, "error": str(exc)}, error=True)


if __name__ == "__main__":
    sys.exit(main())
