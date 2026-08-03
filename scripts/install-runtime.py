#!/usr/bin/env python3
"""Install the PixelTops image-editing runtime for repository deployment.

This explicit deploy helper is the only repository entry point that creates or
updates ``$CODEX_HOME/tools/masked-image-edit``. It stages replacement Python
environments before activation, downloads pinned model artifacts, and never
runs from regular skill usage. Runtime validation is a separate deploy step.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import uuid
from collections.abc import Mapping, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "skills"
    / "pixeltops-image-editor"
    / "references"
    / "runtime-contract.json"
)
ENVIRONMENT_MARKER = ".pixeltops-runtime.json"
MARKER_SCHEMA = "pixeltops-runtime-environment.v1"
LOCK_NAME = ".install-runtime.lock"


class InstallError(RuntimeError):
    """One compact deployment-installation failure."""


def required_mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one contract object or fail with its precise field label."""

    if not isinstance(value, Mapping):
        raise InstallError(f"runtime contract {label} must be an object")
    return value


def required_text(value: object, label: str) -> str:
    """Return one non-empty contract string."""

    if not isinstance(value, str) or not value:
        raise InstallError(f"runtime contract {label} must be a non-empty string")
    return value


def portable_relative(value: object, label: str) -> pathlib.Path:
    """Resolve a portable relative contract path without permitting traversal."""

    text = required_text(value, label).replace("\\", "/")
    path = pathlib.PurePosixPath(text)
    windows = pathlib.PureWindowsPath(text)
    if path.is_absolute() or windows.is_absolute() or windows.drive or ".." in path.parts:
        raise InstallError(f"runtime contract {label} must be portable and relative")
    return pathlib.Path(*path.parts)


def load_contract() -> Mapping[str, object]:
    """Load the single runtime identity, dependency, and model source of truth."""

    try:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"runtime contract is unreadable: {exc}") from exc
    mapping = required_mapping(contract, "root")
    if mapping.get("schema") != "pixeltops-image-runtime.v1":
        raise InstallError("runtime contract schema is unsupported")
    return mapping


RUNTIME_CONTRACT = load_contract()
RUNTIME_RELATIVE = portable_relative(
    RUNTIME_CONTRACT.get("runtime_relative_path"),
    "runtime_relative_path",
)


def environment_declarations() -> tuple[tuple[str, str, pathlib.Path], ...]:
    """Return the two closed deployment environment declarations."""

    environments = required_mapping(
        RUNTIME_CONTRACT.get("environments"),
        "environments",
    )
    declarations: list[tuple[str, str, pathlib.Path]] = []
    for name in ("mask", "inpaint"):
        declaration = required_mapping(environments.get(name), f"environments.{name}")
        python_version = required_text(
            declaration.get("python"),
            f"environments.{name}.python",
        )
        requirements = portable_relative(
            declaration.get("requirements"),
            f"environments.{name}.requirements",
        )
        declarations.append((name, python_version, REPO_ROOT / requirements))
    return tuple(declarations)


def huggingface_declarations() -> tuple[tuple[str, str], ...]:
    """Return exact model identities and immutable revisions."""

    models = required_mapping(
        RUNTIME_CONTRACT.get("huggingface_models"),
        "huggingface_models",
    )
    declarations: list[tuple[str, str]] = []
    for name in ("grounding_dino", "sam2"):
        declaration = required_mapping(
            models.get(name),
            f"huggingface_models.{name}",
        )
        declarations.append(
            (
                required_text(declaration.get("id"), f"huggingface_models.{name}.id"),
                required_text(
                    declaration.get("revision"),
                    f"huggingface_models.{name}.revision",
                ),
            )
        )
    return tuple(declarations)


def codex_home() -> pathlib.Path:
    """Return the configured Codex home without embedding a user-local path."""

    configured = os.environ.get("CODEX_HOME")
    return (
        pathlib.Path(configured).expanduser()
        if configured
        else pathlib.Path.home() / ".codex"
    )


def runtime_root() -> pathlib.Path:
    """Return the machine-local runtime root owned by this installer."""

    return codex_home() / RUNTIME_RELATIVE


def environment_relative(name: str) -> pathlib.Path:
    """Return one environment location from the runtime contract."""

    environments = required_mapping(
        RUNTIME_CONTRACT.get("environments"),
        "environments",
    )
    declaration = required_mapping(environments.get(name), f"environments.{name}")
    return portable_relative(
        declaration.get("relative_path"),
        f"environments.{name}.relative_path",
    )


def environment_python(root: pathlib.Path, name: str) -> pathlib.Path:
    """Return the Windows interpreter path for one isolated environment."""

    return root / environment_relative(name) / "Scripts" / "python.exe"


def run_checked(
    arguments: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> str:
    """Run one deterministic installation command and retain compact failure."""

    completed = subprocess.run(
        list(arguments),
        capture_output=True,
        text=True,
        check=False,
        env=dict(environment) if environment is not None else None,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()
        raise InstallError(detail[-2000:])
    return completed.stdout.strip()


def requirements_digest(path: pathlib.Path) -> str:
    """Hash one pinned dependency set for idempotent environment reuse."""

    if not path.is_file():
        raise InstallError(f"missing requirements: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_marker(path: pathlib.Path) -> Mapping[str, object] | None:
    """Read an environment marker, treating malformed state as stale."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def environment_is_current(
    python_exe: pathlib.Path,
    *,
    python_version: str,
    requirements_sha256: str,
) -> bool:
    """Check only installer-owned state; runtime readiness is validated later."""

    marker = read_marker(python_exe.parents[1] / ENVIRONMENT_MARKER)
    return bool(
        python_exe.is_file()
        and marker
        and marker.get("schema") == MARKER_SCHEMA
        and marker.get("python") == python_version
        and marker.get("requirements_sha256") == requirements_sha256
    )


def write_marker(
    environment_root: pathlib.Path,
    *,
    python_version: str,
    requirements_sha256: str,
) -> None:
    """Record the exact installer inputs after package synchronization succeeds."""

    marker = environment_root / ENVIRONMENT_MARKER
    marker.write_text(
        json.dumps(
            {
                "schema": MARKER_SCHEMA,
                "python": python_version,
                "requirements_sha256": requirements_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def install_environment(
    uv: str,
    root: pathlib.Path,
    name: str,
    python_version: str,
    requirements: pathlib.Path,
) -> None:
    """Stage and atomically activate one pinned Python environment."""

    digest = requirements_digest(requirements)
    target = root / environment_relative(name)
    python_exe = environment_python(root, name)
    if environment_is_current(
        python_exe,
        python_version=python_version,
        requirements_sha256=digest,
    ):
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stage = target.with_name(f".{name}.stage-{token}")
    backup = target.with_name(f".{name}.backup-{token}")
    try:
        run_checked(
            [
                uv,
                "venv",
                str(stage),
                "--python",
                python_version,
                "--managed-python",
                "--no-config",
            ]
        )
        staged_python = stage / "Scripts" / "python.exe"
        run_checked(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(staged_python),
                "--requirements",
                str(requirements),
                "--exact",
                "--strict",
                "--torch-backend",
                "cpu",
                "--no-progress",
                "--no-config",
            ]
        )
        write_marker(
            stage,
            python_version=python_version,
            requirements_sha256=digest,
        )

        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except OSError:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def install_huggingface_models(root: pathlib.Path) -> None:
    """Download exact model revisions through the pinned mask environment."""

    python_exe = environment_python(root, "mask")
    cache = root / portable_relative(
        RUNTIME_CONTRACT.get("huggingface_cache"),
        "huggingface_cache",
    )
    cache.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
        "cache_dir=sys.argv[3])"
    )
    for model_id, revision in huggingface_declarations():
        run_checked([str(python_exe), "-c", code, model_id, revision, str(cache)])


def file_md5(path: pathlib.Path) -> str:
    """Return the upstream integrity digest used by IOPaint for LaMa."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_lama_model(root: pathlib.Path) -> None:
    """Download LaMa to a temporary peer and activate only after verification."""

    lama = required_mapping(RUNTIME_CONTRACT.get("lama"), "lama")
    target = root / portable_relative(lama.get("relative_path"), "lama.relative_path")
    expected_md5 = required_text(lama.get("md5"), "lama.md5")
    source_url = required_text(lama.get("url"), "lama.url")
    if target.is_file() and file_md5(target) == expected_md5:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "PixelTops-Runtime-Installer/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if file_md5(temporary) != expected_md5:
            raise InstallError("downloaded LaMa model failed integrity validation")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_lock(root: pathlib.Path) -> pathlib.Path:
    """Block concurrent installers without touching unrelated runtime state."""

    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise InstallError(f"runtime installation is already locked: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{os.getpid()}\n")
    return lock


def main() -> int:
    """Install declared runtime inputs and emit only the compact result."""

    try:
        if sys.argv[1:]:
            raise InstallError("install-runtime accepts no arguments")
        if os.name != "nt":
            raise InstallError("the current image-editing runtime supports Windows only")
        uv = shutil.which("uv")
        if uv is None:
            raise InstallError("uv is required on PATH for install-runtime")
        root = runtime_root()
        lock = acquire_lock(root)
        try:
            for name, python_version, requirements in environment_declarations():
                install_environment(
                    uv,
                    root,
                    name,
                    python_version,
                    requirements,
                )
            install_huggingface_models(root)
            install_lama_model(root)
        finally:
            lock.unlink(missing_ok=True)
    except (InstallError, OSError, subprocess.SubprocessError) as exc:
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
