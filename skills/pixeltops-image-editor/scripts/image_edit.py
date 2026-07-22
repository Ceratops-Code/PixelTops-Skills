#!/usr/bin/env python3
"""One public CLI for deterministic-mask image editing with isolated workers.

The launcher itself uses only the Python standard library. It dispatches model
work to the installed mask and IOPaint environments, preserving their
incompatible dependency sets. Every successful command emits compact JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


VERSION = "0.1.0"
HERE = pathlib.Path(__file__).resolve().parent


def codex_home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CODEX_HOME") or pathlib.Path.home() / ".codex")


def runtime_paths() -> dict[str, pathlib.Path]:
    root = codex_home() / "tools" / "masked-image-edit"
    return {
        "root": root,
        "mask_python": root / "envs" / "mask" / "Scripts" / "python.exe",
        "inpaint_python": root / "envs" / "inpaint" / "Scripts" / "python.exe",
        "hf_models": root / "models" / "huggingface",
        "torch_home": root / "models" / "iopaint" / "torch",
    }


def emit(payload: dict[str, Any], *, error: bool = False) -> int:
    stream = sys.stderr if error else sys.stdout
    print(json.dumps(payload, separators=(",", ":")), file=stream)
    return 1 if error else 0


def run_worker(python_exe: pathlib.Path, worker: str, arguments: list[str]) -> dict[str, Any]:
    if not python_exe.is_file():
        raise RuntimeError(f"runtime missing: {python_exe}")
    worker_path = HERE / worker
    if not worker_path.is_file():
        raise RuntimeError(f"worker missing: {worker_path}")
    environment = os.environ.copy()
    paths = runtime_paths()
    if worker == "mask_worker.py":
        environment["HF_HOME"] = str(paths["hf_models"])
    if worker == "inpaint_worker.py":
        environment["TORCH_HOME"] = str(paths["torch_home"])
    completed = subprocess.run(
        [str(python_exe), str(worker_path), *arguments],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
        timeout=900,
    )
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        detail = (completed.stderr or stdout or "worker failed").strip()[-2000:]
        raise RuntimeError(detail)
    if not stdout:
        raise RuntimeError("worker returned no result")
    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"worker returned invalid JSON: {stdout[-1000:]}") from exc


def ensure_output_target(path: pathlib.Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def artifact_path(output: pathlib.Path, label: str) -> pathlib.Path:
    return output.with_name(f"{output.stem}.{label}.png")


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--text", help="Object description for Grounding DINO + SAM 2.1.")
    selection.add_argument("--polygon", help='Manual lasso as "x1,y1;x2,y2;...".')
    selection.add_argument("--mask", type=pathlib.Path, help="Existing white-editable mask image.")
    parser.add_argument("--box-index", type=int, default=0, help="Grounding result ranked by confidence (default: 0).")
    parser.add_argument("--threshold", type=float, default=0.25, help="Grounding box/text threshold (default: 0.25).")
    parser.add_argument("--expand", type=int, default=0, help="Expand mask by N pixels; negative values shrink it.")


def selection_arguments(args: argparse.Namespace, output_mask: pathlib.Path) -> list[str]:
    command = ["select", str(args.input), str(output_mask), "--box-index", str(args.box_index), "--threshold", str(args.threshold), "--expand", str(args.expand), "--overwrite"]
    if args.text:
        command += ["--text", args.text]
    elif args.polygon:
        command += ["--polygon", args.polygon]
    else:
        command += ["--mask", str(args.mask)]
    preview = getattr(args, "preview", None)
    if preview:
        command += ["--preview", str(preview)]
    return command


def resolve_selection(args: argparse.Namespace, output: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    mask_output = args.selection_mask_output or artifact_path(output, "selection-mask")
    ensure_output_target(mask_output, args.overwrite)
    result = run_worker(runtime_paths()["mask_python"], "mask_worker.py", selection_arguments(args, mask_output))
    return mask_output, result


def add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")


def add_masked_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection-mask-output", type=pathlib.Path, help="Path for the exact resolved selection mask.")
    parser.add_argument("--allowed-mask-output", type=pathlib.Path, help="Path for the exact region allowed to change.")
    parser.add_argument("--feather", type=float, default=0.0, help="Edge transition radius in pixels.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Validate runtimes, packages, models, and workers.")

    select = subparsers.add_parser("select", help="Create an object-shaped mask from text, polygon lasso, or an existing mask.")
    select.add_argument("input", type=pathlib.Path)
    select.add_argument("output_mask", type=pathlib.Path)
    add_selection_arguments(select)
    select.add_argument("--preview", type=pathlib.Path, help="Optional mask-overlay preview.")
    add_overwrite(select)

    for name in ("erase", "fill"):
        erase = subparsers.add_parser(name, help="Erase/fill only inside the resolved mask using LaMa.")
        erase.add_argument("input", type=pathlib.Path)
        erase.add_argument("output", type=pathlib.Path)
        add_selection_arguments(erase)
        add_masked_output_arguments(erase)
        erase.add_argument("--preview", type=pathlib.Path, help="Optional selection-overlay preview.")
        add_overwrite(erase)

    remove = subparsers.add_parser("remove-background", help="Keep the selected foreground and write transparent PNG background.")
    remove.add_argument("input", type=pathlib.Path)
    remove.add_argument("output", type=pathlib.Path)
    add_selection_arguments(remove)
    add_masked_output_arguments(remove)
    remove.add_argument("--preview", type=pathlib.Path)
    add_overwrite(remove)

    replace = subparsers.add_parser("replace-background", help="Place the selected foreground over an image or solid color.")
    replace.add_argument("input", type=pathlib.Path)
    replace.add_argument("output", type=pathlib.Path)
    add_selection_arguments(replace)
    add_masked_output_arguments(replace)
    background = replace.add_mutually_exclusive_group(required=True)
    background.add_argument("--background", type=pathlib.Path)
    background.add_argument("--color", default=None, help="Background color such as #202020 or white.")
    replace.add_argument("--background-mode", choices=("cover", "contain", "stretch"), default="cover")
    replace.add_argument("--preview", type=pathlib.Path)
    add_overwrite(replace)

    composite = subparsers.add_parser("composite", help="Blend an edited image into a base only within a supplied mask.")
    composite.add_argument("base", type=pathlib.Path)
    composite.add_argument("edited", type=pathlib.Path)
    composite.add_argument("mask", type=pathlib.Path)
    composite.add_argument("output", type=pathlib.Path)
    composite.add_argument("--method", choices=("hard", "feather", "seamless"), default="hard")
    composite.add_argument("--feather", type=float, default=0.0)
    composite.add_argument("--transition", type=int, default=8, help="Allowed dilation for seamless blending.")
    composite.add_argument("--allowed-mask-output", type=pathlib.Path)
    add_overwrite(composite)

    resize = subparsers.add_parser("resize", help="Resize to exact dimensions with explicit aspect-ratio policy.")
    resize.add_argument("input", type=pathlib.Path)
    resize.add_argument("output", type=pathlib.Path)
    resize.add_argument("width", type=int)
    resize.add_argument("height", type=int)
    resize.add_argument("--mode", choices=("contain", "cover", "stretch"), default="contain")
    resize.add_argument("--color", default="transparent")
    add_overwrite(resize)

    preview = subparsers.add_parser("preview", help="Render an overlay showing the exact mask.")
    preview.add_argument("input", type=pathlib.Path)
    preview.add_argument("mask", type=pathlib.Path)
    preview.add_argument("output", type=pathlib.Path)
    preview.add_argument("--color", default="#ff00ff")
    preview.add_argument("--opacity", type=float, default=0.45)
    add_overwrite(preview)

    verify = subparsers.add_parser("verify", help="Assert that pixels outside an allowed-change mask are byte-identical.")
    verify.add_argument("original", type=pathlib.Path)
    verify.add_argument("result", type=pathlib.Path)
    verify.add_argument("allowed_mask", type=pathlib.Path)

    return parser


def worker_overwrite(args: argparse.Namespace) -> list[str]:
    return ["--overwrite"] if args.overwrite else []


def main() -> int:
    args = build_parser().parse_args()
    paths = runtime_paths()
    try:
        if args.command == "doctor":
            mask = run_worker(paths["mask_python"], "mask_worker.py", ["doctor"])
            inpaint = run_worker(paths["inpaint_python"], "inpaint_worker.py", ["doctor"])
            return emit({"status": "OK", "version": VERSION, "mask": mask, "inpaint": inpaint})

        if args.command == "select":
            ensure_output_target(args.output_mask, args.overwrite)
            if args.preview:
                ensure_output_target(args.preview, args.overwrite)
            return emit(run_worker(paths["mask_python"], "mask_worker.py", selection_arguments(args, args.output_mask)))

        if args.command in {"erase", "fill"}:
            ensure_output_target(args.output, args.overwrite)
            selection_mask, selection = resolve_selection(args, args.output)
            allowed = args.allowed_mask_output or artifact_path(args.output, "allowed-mask")
            ensure_output_target(allowed, args.overwrite)
            operation = run_worker(
                paths["inpaint_python"],
                "inpaint_worker.py",
                ["erase", str(args.input), str(selection_mask), str(args.output), "--allowed-mask-output", str(allowed), "--feather", str(args.feather), "--overwrite"],
            )
            return emit({"status": "OK", "command": args.command, "selection": selection, "operation": operation})

        if args.command in {"remove-background", "replace-background"}:
            ensure_output_target(args.output, args.overwrite)
            selection_mask, selection = resolve_selection(args, args.output)
            allowed = args.allowed_mask_output or artifact_path(args.output, "allowed-mask")
            ensure_output_target(allowed, args.overwrite)
            command = [args.command, str(args.input), str(selection_mask), str(args.output), "--allowed-mask-output", str(allowed), "--feather", str(args.feather), "--overwrite"]
            if args.command == "replace-background":
                if args.background:
                    command += ["--background", str(args.background)]
                else:
                    command += ["--color", args.color]
                command += ["--background-mode", args.background_mode]
            operation = run_worker(paths["mask_python"], "mask_worker.py", command)
            return emit({"status": "OK", "command": args.command, "selection": selection, "operation": operation})

        if args.command == "composite":
            ensure_output_target(args.output, args.overwrite)
            allowed = args.allowed_mask_output or artifact_path(args.output, "allowed-mask")
            ensure_output_target(allowed, args.overwrite)
            command = [
                "composite", str(args.base), str(args.edited), str(args.mask), str(args.output),
                "--method", args.method, "--feather", str(args.feather), "--transition", str(args.transition),
                "--allowed-mask-output", str(allowed),
            ] + worker_overwrite(args)
            return emit(run_worker(paths["mask_python"], "mask_worker.py", command))

        if args.command == "resize":
            ensure_output_target(args.output, args.overwrite)
            command = ["resize", str(args.input), str(args.output), str(args.width), str(args.height), "--mode", args.mode, "--color", args.color] + worker_overwrite(args)
            return emit(run_worker(paths["mask_python"], "mask_worker.py", command))

        if args.command == "preview":
            ensure_output_target(args.output, args.overwrite)
            command = ["preview", str(args.input), str(args.mask), str(args.output), "--color", args.color, "--opacity", str(args.opacity)] + worker_overwrite(args)
            return emit(run_worker(paths["mask_python"], "mask_worker.py", command))

        if args.command == "verify":
            result = run_worker(paths["mask_python"], "mask_worker.py", ["verify", str(args.original), str(args.result), str(args.allowed_mask)])
            return emit(result, error=result.get("status") != "OK")

        raise ValueError(f"unsupported command: {args.command}")
    except Exception as exc:
        return emit({"status": "ERROR", "command": args.command, "error": str(exc)}, error=True)


if __name__ == "__main__":
    sys.exit(main())
