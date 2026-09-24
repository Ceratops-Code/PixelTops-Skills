#!/usr/bin/env python3
"""Headless Big-LaMa worker with hard outside-mask preservation and auditing."""

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


MODEL_ENVIRONMENT = "PIXELTOPS_LAMA_MODEL"
MODEL_FILENAME = "big-lama.pt"
PAD_MODULO = 8
CROP_TRIGGER_SIZE = 800
CROP_MARGIN = 128


def emit(payload: dict[str, Any], *, error: bool = False) -> int:
    print(json.dumps(payload, separators=(",", ":")), file=sys.stderr if error else sys.stdout)
    return 1 if error else 0


def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_image(path: pathlib.Path, flags: int) -> np.ndarray:
    """Decode one required image without adding a Pillow runtime dependency."""

    source = require_file(path)
    encoded = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def atomic_save(array: np.ndarray, path: pathlib.Path, overwrite: bool) -> None:
    """Encode beside the destination and atomically activate the complete file."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    payload = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 else array
    encoded_ok, encoded = cv2.imencode(extension, payload)
    if not encoded_ok:
        raise OSError(f"failed to encode output: {path}")
    temporary = path.with_name(f".{path.stem}.tmp{extension}")
    try:
        encoded.tofile(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(torch: Any) -> pathlib.Path:
    """Resolve the deployed checkpoint while retaining direct-worker compatibility."""

    configured = os.environ.get(MODEL_ENVIRONMENT)
    if configured:
        return require_file(pathlib.Path(configured).expanduser())
    return require_file(pathlib.Path(torch.hub.get_dir()) / "checkpoints" / MODEL_FILENAME)


def pad_to_modulo(array: np.ndarray, modulo: int = PAD_MODULO) -> np.ndarray:
    """Symmetrically pad the bottom and right edges for TorchScript inference."""

    height, width = array.shape[:2]
    pad_height = (-height) % modulo
    pad_width = (-width) % modulo
    if not pad_height and not pad_width:
        return array
    padding = (
        ((0, pad_height), (0, pad_width))
        if array.ndim == 2
        else ((0, pad_height), (0, pad_width), (0, 0))
    )
    return np.pad(array, padding, mode="symmetric")


def infer_crop(torch: Any, model: Any, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run one RGB crop through Big-LaMa and restore its original dimensions."""

    height, width = image.shape[:2]
    padded_image = pad_to_modulo(image)
    padded_mask = pad_to_modulo(mask)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(padded_image.transpose(2, 0, 1), dtype=np.float32)
        / 255.0
    ).unsqueeze(0)
    mask_tensor = torch.from_numpy(
        np.ascontiguousarray((padded_mask > 0).astype(np.float32)[None, ...])
    ).unsqueeze(0)
    with torch.inference_mode():
        output = model(image_tensor, mask_tensor)
    generated = output[0].permute(1, 2, 0).detach().cpu().numpy()
    if generated.ndim != 3 or generated.shape[2] != 3:
        raise RuntimeError("LaMa returned an invalid image tensor")
    if generated.shape[0] < height or generated.shape[1] < width:
        raise RuntimeError("LaMa returned an undersized image tensor")
    return np.clip(generated[:height, :width] * 255.0, 0, 255).astype(np.uint8)


def expanded_crop_box(
    image_shape: tuple[int, ...], bounds: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Expand one mask component by the established 128-pixel context margin."""

    x, y, width, height = bounds
    center_x = x + width // 2
    center_y = y + height // 2
    target_width = width + CROP_MARGIN * 2
    target_height = height + CROP_MARGIN * 2
    image_height, image_width = image_shape[:2]

    raw_left = center_x - target_width // 2
    raw_right = center_x + target_width // 2
    raw_top = center_y - target_height // 2
    raw_bottom = center_y + target_height // 2
    left = max(raw_left, 0)
    right = min(raw_right, image_width)
    top = max(raw_top, 0)
    bottom = min(raw_bottom, image_height)
    if raw_left < 0:
        right += -raw_left
    if raw_right > image_width:
        left -= raw_right - image_width
    if raw_top < 0:
        bottom += -raw_top
    if raw_bottom > image_height:
        top -= raw_bottom - image_height
    return (
        max(left, 0),
        max(top, 0),
        min(right, image_width),
        min(bottom, image_height),
    )


def run_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Load the verified TorchScript checkpoint and run CPU-only LaMa inference."""

    import torch

    device = torch.device("cpu")
    model = torch.jit.load(str(checkpoint_path(torch)), map_location=device)
    model.eval()
    model.to(device)
    if max(image.shape) <= CROP_TRIGGER_SIZE:
        return infer_crop(torch, model, image, mask)

    generated = image.copy()
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        left, top, right, bottom = expanded_crop_box(
            image.shape, (x, y, width, height)
        )
        generated[top:bottom, left:right] = infer_crop(
            torch,
            model,
            image[top:bottom, left:right],
            mask[top:bottom, left:right],
        )
    return generated


def transition_alpha(mask: np.ndarray, feather: float) -> tuple[np.ndarray, np.ndarray]:
    binary = mask >= 128
    if feather <= 0:
        return binary.astype(np.float32), binary
    radius = max(1, int(math.ceil(feather * 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    allowed = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=feather, sigmaY=feather)
    alpha[~allowed] = 0.0
    return np.clip(alpha, 0.0, 1.0), allowed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    erase = commands.add_parser("erase")
    erase.add_argument("input", type=pathlib.Path)
    erase.add_argument("mask", type=pathlib.Path)
    erase.add_argument("output", type=pathlib.Path)
    erase.add_argument("--allowed-mask-output", type=pathlib.Path, required=True)
    erase.add_argument("--feather", type=float, default=0.0)
    erase.add_argument("--overwrite", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "erase":
            original_bgr = read_image(args.input, cv2.IMREAD_COLOR)
            original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            mask_image = read_image(args.mask, cv2.IMREAD_GRAYSCALE)
            if mask_image.shape != original.shape[:2]:
                mask_image = cv2.resize(
                    mask_image,
                    (original.shape[1], original.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = (mask_image >= 128).astype(np.uint8) * 255
            if not np.any(mask):
                raise ValueError("mask is empty")

            generated = run_lama(original, mask)
            if generated.shape != original.shape:
                raise RuntimeError("LaMa output dimensions do not match the input")
            alpha, allowed = transition_alpha(mask, args.feather)
            result = np.rint(generated.astype(np.float32) * alpha[..., None] + original.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)
            result[~allowed] = original[~allowed]
            changed = np.any(result != original, axis=2)
            outside_changed = int(np.count_nonzero(changed & ~allowed))
            if outside_changed:
                raise RuntimeError(f"outside-mask verification failed: {outside_changed} changed pixels")
            atomic_save(result, args.output, args.overwrite)
            atomic_save(allowed.astype(np.uint8) * 255, args.allowed_mask_output, args.overwrite)
            return emit({
                "status": "OK",
                "output": str(args.output.resolve()),
                "allowedMask": str(args.allowed_mask_output.resolve()),
                "allowedPixels": int(np.count_nonzero(allowed)),
                "changedPixels": int(np.count_nonzero(changed)),
                "outsideChangedPixels": outside_changed,
                "model": "LaMa",
                "device": "cpu",
            })

        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        return emit({"status": "ERROR", "command": args.command, "error": str(exc)}, error=True)


if __name__ == "__main__":
    sys.exit(main())
