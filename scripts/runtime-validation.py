#!/usr/bin/env python3
"""Validate the deployed PixelTops runtime without modifying it.

This deploy-only helper verifies the pinned environments and model artifacts
declared by the repository runtime contract. It never installs, repairs, or
downloads state, and regular skill usage never invokes it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "skills"
    / "pixeltops-image-editor"
    / "references"
    / "runtime-contract.json"
)


class ValidationError(RuntimeError):
    """One compact runtime-validation failure."""


def required_mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one contract object or fail with its precise field label."""

    if not isinstance(value, Mapping):
        raise ValidationError(f"runtime contract {label} must be an object")
    return value


def required_text(value: object, label: str) -> str:
    """Return one non-empty contract string."""

    if not isinstance(value, str) or not value:
        raise ValidationError(f"runtime contract {label} must be a non-empty string")
    return value


def portable_relative(value: object, label: str) -> pathlib.Path:
    """Resolve a portable relative contract path without permitting traversal."""

    text = required_text(value, label).replace("\\", "/")
    path = pathlib.PurePosixPath(text)
    windows = pathlib.PureWindowsPath(text)
    if path.is_absolute() or windows.is_absolute() or windows.drive or ".." in path.parts:
        raise ValidationError(f"runtime contract {label} must be portable and relative")
    return pathlib.Path(*path.parts)


def load_contract() -> Mapping[str, object]:
    """Load the shared runtime identity and layout contract."""

    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"runtime contract is unreadable: {exc}") from exc
    mapping = required_mapping(contract, "root")
    if mapping.get("schema") != "pixeltops-image-runtime.v1":
        raise ValidationError("runtime contract schema is unsupported")
    return mapping


def codex_home() -> pathlib.Path:
    """Return the configured Codex home without embedding a user-local path."""

    configured = os.environ.get("CODEX_HOME")
    return (
        pathlib.Path(configured).expanduser()
        if configured
        else pathlib.Path.home() / ".codex"
    )


def run_checked(arguments: Sequence[str]) -> str:
    """Run one read-only probe and retain compact failure evidence."""

    completed = subprocess.run(
        list(arguments),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "probe failed").strip()
        raise ValidationError(detail[-2000:])
    return completed.stdout.strip()


def pinned_requirements(path: pathlib.Path) -> dict[str, str]:
    """Parse the closed ``name==version`` lock format used by deployment."""

    if not path.is_file():
        raise ValidationError(f"missing requirements: {path}")
    pinned: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or "==" in version:
            raise ValidationError(
                f"invalid pinned requirement at {path}:{line_number}"
            )
        pinned[name] = version
    if not pinned:
        raise ValidationError(f"requirements are empty: {path}")
    return pinned


def validate_environment(
    python_exe: pathlib.Path,
    *,
    python_version: str,
    requirements: pathlib.Path,
    imports: Sequence[str],
) -> None:
    """Verify one interpreter, its exact lock, and operation-critical imports."""

    if not python_exe.is_file():
        raise ValidationError(f"missing runtime interpreter: {python_exe}")
    expected = json.dumps(pinned_requirements(requirements), separators=(",", ":"))
    probe = """
import importlib
import importlib.metadata as metadata
import json
import sys

expected_python = sys.argv[1]
expected_packages = json.loads(sys.argv[2])
problems = {}
actual_python = sys.version.split()[0]
if actual_python != expected_python:
    problems["python"] = {"expected": expected_python, "actual": actual_python}
for name, expected_version in expected_packages.items():
    try:
        actual_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        problems[name] = {"expected": expected_version, "actual": None}
        continue
    if not (
        actual_version == expected_version
        or actual_version.startswith(expected_version + "+")
    ):
        problems[name] = {"expected": expected_version, "actual": actual_version}
for module in sys.argv[3:]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        problems[f"import:{module}"] = f"{type(exc).__name__}: {exc}"
if problems:
    print(json.dumps(problems, separators=(",", ":")))
    raise SystemExit(1)
print("OK")
"""
    run_checked(
        [
            str(python_exe),
            "-c",
            probe,
            python_version,
            expected,
            *imports,
        ]
    )


def model_snapshot(cache: pathlib.Path, model_id: str, revision: str) -> pathlib.Path:
    """Return the immutable Hugging Face snapshot location."""

    storage = "models--" + model_id.replace("/", "--")
    return cache / storage / "snapshots" / revision


def file_md5(path: pathlib.Path) -> str:
    """Return the upstream integrity digest declared for LaMa."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_models(root: pathlib.Path, contract: Mapping[str, object]) -> None:
    """Verify every pinned model artifact without invoking a downloader."""

    cache = root / portable_relative(
        contract.get("huggingface_cache"),
        "huggingface_cache",
    )
    models = required_mapping(contract.get("huggingface_models"), "huggingface_models")
    for name in ("grounding_dino", "sam2"):
        declaration = required_mapping(models.get(name), f"huggingface_models.{name}")
        model_id = required_text(declaration.get("id"), f"huggingface_models.{name}.id")
        revision = required_text(
            declaration.get("revision"),
            f"huggingface_models.{name}.revision",
        )
        snapshot = model_snapshot(cache, model_id, revision)
        if not snapshot.is_dir():
            raise ValidationError(f"missing model snapshot: {snapshot}")

    lama = required_mapping(contract.get("lama"), "lama")
    lama_path = root / portable_relative(lama.get("relative_path"), "lama.relative_path")
    expected_md5 = required_text(lama.get("md5"), "lama.md5")
    if not lama_path.is_file():
        raise ValidationError(f"missing LaMa model: {lama_path}")
    if file_md5(lama_path) != expected_md5:
        raise ValidationError(f"LaMa model failed integrity validation: {lama_path}")


def main() -> int:
    """Validate declared runtime state and emit only the compact result."""

    try:
        if sys.argv[1:]:
            raise ValidationError("runtime-validation accepts no arguments")
        if os.name != "nt":
            raise ValidationError("the current image-editing runtime supports Windows only")
        contract = load_contract()
        root = codex_home() / portable_relative(
            contract.get("runtime_relative_path"),
            "runtime_relative_path",
        )
        environments = required_mapping(contract.get("environments"), "environments")
        declarations = (
            (
                "mask",
                ("numpy", "PIL", "torch", "transformers", "pymatting", "cv2"),
            ),
            ("inpaint", ("numpy", "PIL", "torch", "iopaint")),
        )
        for name, imports in declarations:
            declaration = required_mapping(
                environments.get(name),
                f"environments.{name}",
            )
            environment_root = root / portable_relative(
                declaration.get("relative_path"),
                f"environments.{name}.relative_path",
            )
            validate_environment(
                environment_root / "Scripts" / "python.exe",
                python_version=required_text(
                    declaration.get("python"),
                    f"environments.{name}.python",
                ),
                requirements=REPO_ROOT
                / portable_relative(
                    declaration.get("requirements"),
                    f"environments.{name}.requirements",
                ),
                imports=imports,
            )
        validate_models(root, contract)
    except (ValidationError, OSError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(exc)},
                separators=(",", ":"),
            )
        )
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
