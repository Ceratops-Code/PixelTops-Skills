#!/usr/bin/env python3
"""IOPaint/LaMa worker with hard outside-mask preservation and auditing."""

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
from PIL import Image


def emit(payload: dict[str, Any], *, error: bool = False) -> int:
    print(json.dumps(payload, separators=(",", ":")), file=sys.stderr if error else sys.stdout)
    return 1 if error else 0


def require_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def atomic_save(array: np.ndarray, path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix or '.png'}")
    Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB").save(temporary)
    os.replace(temporary, path)


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
    commands.add_parser("doctor")
    erase = commands.add_parser("erase")
    erase.add_argument("input", type=pathlib.Path)
    erase.add_argument("mask", type=pathlib.Path)
    erase.add_argument("output", type=pathlib.Path)
    erase.add_argument("--allowed-mask-output", type=pathlib.Path, required=True)
    erase.add_argument("--feather", type=float, default=0.0)
    erase.add_argument("--overwrite", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "doctor":
            import importlib.metadata
            import torch

            torch_home = pathlib.Path(os.environ.get("TORCH_HOME", ""))
            model = torch_home / "hub" / "checkpoints" / "big-lama.pt"
            present = model.is_file()
            return emit({"status": "OK" if present else "ERROR", "python": sys.version.split()[0], "iopaint": importlib.metadata.version("iopaint"), "torch": torch.__version__, "lama": str(model), "modelPresent": present}, error=not present)

        if args.command == "erase":
            from iopaint.model_manager import ModelManager
            from iopaint.schema import HDStrategy, InpaintRequest

            # IOPaint crop mode mutates its input array. Keep an immutable source
            # and pass a separate writable copy so the preservation audit cannot
            # accidentally compare the result against an already-edited buffer.
            original = np.asarray(Image.open(require_file(args.input)).convert("RGB"), dtype=np.uint8).copy()
            working = original.copy()
            mask_image = Image.open(require_file(args.mask)).convert("L")
            if mask_image.size != (original.shape[1], original.shape[0]):
                mask_image = mask_image.resize((original.shape[1], original.shape[0]), Image.Resampling.NEAREST)
            mask = (np.asarray(mask_image, dtype=np.uint8) >= 128).astype(np.uint8) * 255
            if not np.any(mask):
                raise ValueError("mask is empty")

            manager = ModelManager(name="lama", device="cpu")
            config = InpaintRequest(
                image="",
                mask="",
                hd_strategy=HDStrategy.CROP,
                hd_strategy_crop_trigger_size=800,
                hd_strategy_crop_margin=128,
                hd_strategy_resize_limit=1280,
            )
            generated_bgr = manager(working, mask, config)
            generated = generated_bgr[:, :, ::-1]
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
