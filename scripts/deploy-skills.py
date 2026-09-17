#!/usr/bin/env python3
"""Deploy this repository's declared skills without lifecycle dependencies.

This independent installer renders selected skills in a temporary staging
directory, then copies them over existing installations. It runs no skill or
repository validation and retains destination-only files and unselected skills.
Only input parsing and path safety constrain copying. A copy failure can leave
partial updates; this helper cleans only its own staging directory and lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from typing import cast

INSTALLER_VERSION = 16
MANIFEST_NAME = ".runtime-manifest.json"
RUNTIME_MANIFEST_SCHEMA = "ceratops-runtime-skill.v3"
START = "<!-- CERATOPS_SHARED_SECTIONS_START -->"
END = "<!-- CERATOPS_SHARED_SECTIONS_END -->"
SOURCE_PREFIX = "<!-- SECTION SOURCE: "
SOURCE_SUFFIX = " -->"
LOCK_NAME = ".ceratops-bootstrap.lock"
STAGE_RE = re.compile(r"^\.ceratops-bootstrap-stage-[0-9a-f]{32}$")
SKILL_NAME_RE = re.compile(
    r"^(?![a-z0-9-]*--)[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$"
)
IGNORED_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def fail(message: str) -> int:
    """Emit one concise fatal error."""

    print(message, file=sys.stderr)
    return 1


def safe_relative(value: str) -> bool:
    """Accept only repository-relative manifest paths and patterns."""

    posix = pathlib.PurePosixPath(value.replace("\\", "/"))
    windows = pathlib.PureWindowsPath(value)
    return bool(
        value
        and not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
    )


def unsafe_link(path: pathlib.Path) -> bool:
    """Reject links and Windows reparse points from copied input."""

    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    attributes = getattr(
        path.stat(follow_symlinks=False), "st_file_attributes", 0
    )
    return bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def require_inside(path: pathlib.Path, root: pathlib.Path) -> None:
    """Reject any resolved path that escapes its declared root."""

    path.resolve(strict=False).relative_to(root.resolve())


def validate_tree(root: pathlib.Path) -> None:
    """Reject links or reparse points anywhere in one staged tree."""

    if unsafe_link(root):
        raise ValueError(f"unsafe staged tree root: {root}")
    for path in root.rglob("*"):
        if unsafe_link(path):
            raise ValueError(f"unsafe staged tree entry: {path}")


def prepare_python_runtime(repo_root: pathlib.Path, install_root: pathlib.Path) -> pathlib.Path | None:
    """Build a new locked venv version without changing one used by a helper.

    Source declarations stay in the repository. A valid existing version is
    checked without synchronization; a damaged version gets a fresh path.
    Failed new environments and atomic-index scratch files belong to this call.
    """

    project = repo_root / "skills/sections/python"
    declaration = project / "pyproject.toml"
    lock = project / "uv.lock"
    if not declaration.exists() and not lock.exists():
        return None
    if not declaration.is_file() or not lock.is_file() or unsafe_link(project) or unsafe_link(declaration) or unsafe_link(lock):
        raise ValueError("source skill runtime requires regular pyproject.toml and uv.lock")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required to prepare the shared skill Python runtime")
    digest = hashlib.sha256(declaration.read_bytes() + b"\0" + lock.read_bytes()).hexdigest()
    root = install_root.parent / "runtimes/ceratops"
    versions = root / "versions"
    if any(path.is_symlink() or (path.exists() and unsafe_link(path)) for path in (root, versions)):
        raise ValueError("shared skill runtime path cannot be a link")
    versions.mkdir(parents=True, exist_ok=True)
    index = root / "current.json"
    if index.is_symlink() or index.is_junction() or (index.exists() and unsafe_link(index)):
        raise ValueError("shared skill runtime index cannot be a link")
    current: dict[str, object] = {}
    if index.is_file():
        current = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise ValueError("shared skill runtime index is invalid")
    # A 96-bit directory key keeps nested Windows test and task paths below
    # CreateProcess path limits; current.json retains the full lock digest.
    version = digest[:24]
    if current.get("digest") == digest:
        recorded = current.get("version")
        if isinstance(recorded, str) and re.fullmatch(r"[0-9a-f]{24}(?:-[0-9a-f]{8})?", recorded):
            version = recorded
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "UV_WORKING_DIRECTORY", "UV_NO_SYNC", "UV_FROZEN", "UV_PYTHON"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"

    def interpreter(name: str) -> pathlib.Path:
        return versions / name / ".venv" / scripts / executable

    selected = versions / version
    environment["UV_PROJECT_ENVIRONMENT"] = str(selected / ".venv")
    if selected.is_symlink() or selected.is_junction():
        raise ValueError("shared skill runtime version cannot be a link")
    if selected.exists():
        if unsafe_link(selected):
            raise ValueError("shared skill runtime version cannot be a link")
        checked = subprocess.run(
            [uv, "sync", "--quiet", "--project", str(project), "--locked", "--check", "--no-active"],
            env=environment, capture_output=True, text=True, check=False,
        )
        if checked.returncode or not interpreter(version).is_file():
            version = digest[:24] + "-" + uuid.uuid4().hex[:8]
            selected = versions / version
            environment["UV_PROJECT_ENVIRONMENT"] = str(selected / ".venv")
    created = not selected.exists()
    if created:
        result = subprocess.run(
            [uv, "sync", "--quiet", "--project", str(project), "--locked", "--no-active"],
            env=environment, capture_output=True, text=True, check=False,
        )
        if result.returncode or not interpreter(version).is_file():
            if selected.exists():
                shutil.rmtree(selected)
            raise ValueError("shared skill runtime setup failed: " + (result.stderr or result.stdout).strip()[-1000:])
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".current-", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            json.dump({"digest": digest, "version": version}, handle, sort_keys=True)
        os.replace(temporary, index)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return interpreter(version)


def read_manifest(repo_root: pathlib.Path) -> dict[str, object]:
    """Read and validate the declarations required to render every skill."""

    path = repo_root / "skills" / "skill-sections.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("skill-sections.json must contain an object")
    source_id = value.get("runtime_source_id")
    profile = value.get("validation_profile", "ceratops-compatible")
    sections = value.get("sections")
    skills = value.get("skills")
    payloads = value.get("runtime_payloads", {})
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("runtime_source_id must be a nonempty string")
    if profile not in {"ceratops", "ceratops-compatible"}:
        raise ValueError("validation_profile is unsupported")
    if not isinstance(sections, dict) or not all(
        isinstance(name, str) and isinstance(relative, str)
        for name, relative in sections.items()
    ):
        raise ValueError("sections must map strings to strings")
    if not isinstance(skills, dict) or not all(
        isinstance(name, str) and isinstance(selected, list)
        for name, selected in skills.items()
    ):
        raise ValueError("skills must map names to section lists")
    if not isinstance(payloads, dict):
        raise ValueError("runtime_payloads must be an object")
    return value


def declared_skills(
    manifest: Mapping[str, object], requested: Sequence[str]
) -> list[str]:
    """Resolve the exact declared skill set before staging output."""

    assignments = cast(Mapping[str, object], manifest["skills"])
    names = list(requested) if requested else sorted(assignments)
    if len(names) != len(set(names)):
        raise ValueError("duplicate --skill selection")
    for name in names:
        if not isinstance(name, str) or not SKILL_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid skill name: {name!r}")
        if name not in assignments:
            raise ValueError(f"undeclared skill: {name}")
    return names


def declared_python_skills(
    repo_root: pathlib.Path, manifest: Mapping[str, object], selected: Sequence[str],
) -> set[str]:
    """Select only skills declared to use the shared locked Python project.

    An older manifest without this field retains its prior behavior when it
    already has the source project; compatibility application writes the field.
    """

    declared = manifest.get("python_runtime_skills")
    if declared is None:
        project = repo_root / "skills/sections/python"
        return set(selected) if (project / "pyproject.toml").is_file() else set()
    assignments = cast(Mapping[str, object], manifest["skills"])
    if (
        not isinstance(declared, list)
        or len(declared) != len({item for item in declared if isinstance(item, str)})
        or not all(isinstance(item, str) and item in assignments for item in declared)
    ):
        raise ValueError("python_runtime_skills must list unique declared skills")
    return set(selected).intersection(declared)


def action_assignments(
    repo_root: pathlib.Path,
    manifest: Mapping[str, object],
    selected: set[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Resolve only declared public action targets before any destination writes.

    An absent map preserves skill-only manifests. Explicit paths must be direct,
    uniquely routed action references; sections cannot repeat within an action
    or duplicate its parent skill's shared content, including source aliases.
    """

    raw = manifest.get("actions", {})
    skills = manifest.get("skills", {})
    sections = manifest.get("sections", {})
    if not isinstance(raw, Mapping):
        raise ValueError("section manifest actions must be an object")
    if not isinstance(skills, Mapping) or not isinstance(sections, Mapping):
        raise ValueError("action assignments require skills and sections objects")
    result: dict[str, dict[str, list[str]]] = {}
    for skill, actions in raw.items():
        if selected is not None and skill not in selected:
            continue
        if not isinstance(skill, str) or SKILL_NAME_RE.fullmatch(skill) is None or skill not in skills:
            raise ValueError(f"unknown action assignment skill: {skill}")
        if not isinstance(actions, Mapping) or not actions:
            raise ValueError(f"{skill}: action assignments must be a nonempty object")
        skill_dir = repo_root / "skills" / skill
        require_inside(skill_dir, repo_root)
        parent = skill_dir / "SKILL.md"
        if not parent.is_file() or unsafe_link(skill_dir) or unsafe_link(parent):
            raise ValueError(f"{skill}: unavailable action index")
        lines = parent.read_text(encoding="utf-8").splitlines()
        if lines.count("### Action References") != 1:
            raise ValueError(f"{skill}: requires one Action References index")
        start = lines.index("### Action References") + 1
        end = next((i for i in range(start, len(lines)) if re.match(r"^#{1,3}\s", lines[i])), len(lines))
        routes = re.findall(r"`(references/[^`\s]+\.md)`", "\n".join(lines[start:end]))
        parent_sections = skills[skill]
        if not isinstance(parent_sections, list) or not all(isinstance(item, str) for item in parent_sections):
            raise ValueError(f"{skill}: invalid parent section assignment")
        inherited = {
            (repo_root / path).resolve()
            for name in parent_sections
            if isinstance(path := sections.get(name), str)
        }
        resolved: dict[str, list[str]] = {}
        for relative, names in actions.items():
            label = f"{skill}: {relative}"
            if not isinstance(relative, str) or re.fullmatch(r"references/[a-z0-9]+(?:-[a-z0-9]+)*\.md", relative) is None:
                raise ValueError(f"{label}: action target must be one direct references/*.md path")
            if routes.count(relative) != 1:
                raise ValueError(f"{label}: action target must be routed exactly once")
            source = skill_dir / relative
            require_inside(source, skill_dir)
            if not source.is_file() or unsafe_link(source.parent) or unsafe_link(source):
                raise ValueError(f"{label}: unavailable action reference")
            # Rendering also checks the reserved H1 and source-only boundary.
            render_action(source.read_text(encoding="utf-8"), "", label)
            if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
                raise ValueError(f"{label}: section assignment must be a nonempty string list")
            seen = set(inherited)
            for name in names:
                section = sections.get(name)
                if not isinstance(section, str) or not safe_relative(section):
                    raise ValueError(f"{label}: invalid or unknown section assignment {name!r}")
                path = repo_root / section
                require_inside(path, repo_root)
                if not path.is_file() or unsafe_link(path.parent) or unsafe_link(path):
                    raise ValueError(f"{label}: unavailable section {section}")
                if path.resolve() in seen:
                    raise ValueError(f"{label}: duplicate or inherited section {name}")
                seen.add(path.resolve())
                if any(marker in path.read_text(encoding="utf-8") for marker in (START, END, SOURCE_PREFIX)):
                    raise ValueError(f"{label}: section source contains generated markers")
            resolved[relative] = names
        result[skill] = resolved
    return result


def render_action(source: str, shared: str, label: str) -> str:
    """Insert a single generated block directly after a public action's H1."""

    if any(marker in source for marker in (START, END, SOURCE_PREFIX)):
        raise ValueError(f"{label}: source action must be delta-only")
    lines = source.replace("\r\n", "\n").split("\n")
    if not lines or re.fullmatch(r"# .+ Action", lines[0]) is None:
        raise ValueError(f"{label}: action must be titled # <Action Name> Action")
    after = "\n".join(lines[1:]).strip("\n")
    return f"{lines[0]}\n\n{shared}\n\n{after}\n" if after else f"{lines[0]}\n\n{shared}\n"


def section_block(
    repo_root: pathlib.Path,
    manifest: Mapping[str, object],
    skill: str,
    section_names: Sequence[str] | None = None,
) -> str:
    """Resolve one skill's shared sections without lifecycle runtime code."""

    sections = cast(Mapping[str, object], manifest["sections"])
    assignments = cast(Mapping[str, object], manifest["skills"])
    selected = assignments[skill] if section_names is None else section_names
    if not isinstance(selected, Sequence) or isinstance(selected, str) or not selected:
        raise ValueError(f"{skill}: section assignment must be a nonempty list")
    rendered: list[str] = []
    for name in selected:
        if not isinstance(name, str) or name not in sections:
            raise ValueError(f"{skill}: unresolved section {name!r}")
        relative = sections[name]
        if not isinstance(relative, str) or not safe_relative(relative):
            raise ValueError(f"{skill}: invalid section path {relative!r}")
        path = repo_root / relative
        require_inside(path, repo_root)
        if not path.is_file() or unsafe_link(path):
            raise ValueError(f"{skill}: unavailable section {relative}")
        lines = path.read_text(encoding="utf-8").splitlines()
        # Remove complete standalone author notes, including multiline comments.
        text = re.sub(
            r"(?ms)^[ \t]*<!--[ \t]*INTERNAL:(?:(?!-->).)*-->[ \t]*(?:\n|$)",
            "",
            "\n".join(lines),
        ).strip("\n")
        rendered.extend((f"{SOURCE_PREFIX}{relative}{SOURCE_SUFFIX}", text))
    return f"{START}\n" + "\n\n".join(rendered) + f"\n{END}"


def render_skill(source: str, shared: str, skill: str) -> str:
    """Insert resolved shared text after frontmatter and an optional H1."""

    if START in source or END in source:
        raise ValueError(
            f"{skill}: source SKILL.md must not contain generated sections"
        )
    lines = source.replace("\r\n", "\n").split("\n")
    if not lines or lines[0] != "---":
        raise ValueError(f"{skill}: missing frontmatter")
    try:
        frontmatter_end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise ValueError(
            f"{skill}: missing closing frontmatter marker"
        ) from exc
    insert_after = frontmatter_end
    for index in range(frontmatter_end + 1, len(lines)):
        if not lines[index].strip():
            continue
        if lines[index].startswith("# "):
            insert_after = index
        break
    before = "\n".join(lines[: insert_after + 1]).rstrip()
    after = "\n".join(lines[insert_after + 1 :]).strip("\n")
    if after:
        return f"{before}\n\n{shared}\n\n{after}\n"
    return f"{before}\n\n{shared}\n"


def payload_parts(
    value: object, label: str
) -> tuple[str, str | None]:
    """Normalize one portable payload pattern or exact source-target mapping."""

    if isinstance(value, str):
        if not safe_relative(value):
            raise ValueError(f"{label} has unsafe source path: {value!r}")
        return value, None
    if not isinstance(value, dict) or set(value) != {"source", "target"}:
        raise ValueError(f"{label} must be a path or source-target mapping")
    source = value.get("source")
    destination = value.get("target")
    if not isinstance(source, str) or not isinstance(destination, str):
        raise ValueError(f"{label} source and target must be strings")
    if (
        not safe_relative(source)
        or not safe_relative(destination)
        or any(token in source for token in "*?[")
        or any(token in destination for token in "*?[")
        or pathlib.PurePosixPath(destination).as_posix()
        in {".", "SKILL.md", MANIFEST_NAME}
    ):
        raise ValueError(f"{label} has unsafe exact mapping")
    return source, destination


def payload_declarations(
    manifest: Mapping[str, object], skill: str
) -> list[object]:
    """Return validated global and skill-specific payload declarations."""

    payloads = cast(Mapping[str, object], manifest.get("runtime_payloads", {}))
    result: list[object] = []
    for key in ("*", skill):
        values = payloads.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"runtime_payloads.{key} must be a list")
        for index, value in enumerate(values):
            payload_parts(value, f"runtime_payloads.{key}[{index}]")
            result.append(value)
    return result


def copy_payload(
    repo_root: pathlib.Path, declaration: object, target: pathlib.Path
) -> None:
    """Copy one payload declaration into its staged installed-skill target."""

    pattern, mapped_target = payload_parts(declaration, "runtime payload")
    matches = sorted(repo_root.glob(pattern))
    if not matches and not any(token in pattern for token in "*?["):
        raise ValueError(f"runtime payload does not exist: {pattern}")
    if mapped_target is not None and (
        len(matches) != 1 or not matches[0].is_file()
    ):
        raise ValueError("mapped runtime payload source must be one file")
    for source in matches:
        require_inside(source, repo_root)
        if unsafe_link(source):
            raise ValueError(f"runtime payload cannot be a link: {source}")
        relative = (
            pathlib.PurePosixPath(mapped_target)
            if mapped_target is not None
            else pathlib.PurePosixPath(
                source.relative_to(repo_root).as_posix()
            )
        )
        destination = target.joinpath(*relative.parts)
        require_inside(destination, target)
        if mapped_target is not None and (
            destination.exists() or destination.is_symlink()
        ):
            raise ValueError(
                f"runtime payload target collides with skill source: {mapped_target}"
            )
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(*IGNORED_NAMES),
                dirs_exist_ok=True,
            )
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def build_skill(
    repo_root: pathlib.Path,
    staging: pathlib.Path,
    manifest: Mapping[str, object],
    skill: str,
    python_runtime: pathlib.Path | None,
) -> None:
    """Fully resolve and stage one skill without touching its destination."""

    source = repo_root / "skills" / skill
    skill_md = source / "SKILL.md"
    if not skill_md.is_file() or unsafe_link(source) or unsafe_link(skill_md):
        raise ValueError(f"{skill}: missing or unsafe source SKILL.md")
    validate_tree(source)
    target = staging / skill
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(*IGNORED_NAMES),
    )
    rendered = render_skill(
        skill_md.read_text(encoding="utf-8"),
        section_block(repo_root, manifest, skill),
        skill,
    )
    (target / "SKILL.md").write_text(
        rendered, encoding="utf-8", newline="\n"
    )
    for relative, names in action_assignments(repo_root, manifest, {skill}).get(skill, {}).items():
        action_text = (source / relative).read_text(encoding="utf-8")
        (target / relative).write_text(
            render_action(action_text, section_block(repo_root, manifest, skill, names), f"{skill}: {relative}"),
            encoding="utf-8", newline="\n",
        )
    declarations = payload_declarations(manifest, skill)
    for declaration in declarations:
        copy_payload(repo_root, declaration, target)
    metadata = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "skill": skill,
        "runtime_source_id": manifest["runtime_source_id"],
        "validation_profile": manifest.get(
            "validation_profile", "ceratops-compatible"
        ),
        "source_path": f"skills/{skill}",
        "source_repository_root": str(repo_root),
        "generated_from": "skills/skill-sections.json",
        "payload_patterns": declarations,
    }
    if python_runtime is not None:
        metadata["python_runtime"] = str(python_runtime)
    (target / MANIFEST_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def remove_stage(staging: pathlib.Path, install_root: pathlib.Path) -> None:
    """Remove only the uniquely named bootstrap staging tree we created."""

    require_inside(staging, install_root)
    if staging.parent != install_root or STAGE_RE.fullmatch(staging.name) is None:
        raise ValueError("refusing to remove a non-bootstrap staging path")
    if staging.exists() or staging.is_symlink():
        if unsafe_link(staging) or not staging.is_dir():
            raise ValueError("bootstrap staging path is unsafe")
        shutil.rmtree(staging)


def remove_legacy_skill_runtime(skill_root: pathlib.Path) -> None:
    """Retire only the known per-skill launcher and declaration pair."""

    launcher = skill_root / "scripts/run-skill.py"
    project = skill_root / "scripts/python-runtime"
    if any(path.is_symlink() or path.is_junction() or (path.exists() and unsafe_link(path)) for path in (launcher, project)):
        raise ValueError("legacy skill runtime cannot be a link")
    if launcher.exists():
        if not launcher.is_file():
            raise ValueError("legacy skill launcher is not a file")
        launcher.unlink()
    if project.exists():
        if not project.is_dir() or {item.name for item in project.iterdir()} - {"pyproject.toml", "uv.lock"}:
            raise ValueError("legacy skill runtime contains unexpected files")
        for name in ("pyproject.toml", "uv.lock"):
            path = project / name
            if path.exists() and (not path.is_file() or unsafe_link(path)):
                raise ValueError("legacy skill runtime declaration is not a regular file")
        for name in ("pyproject.toml", "uv.lock"):
            (project / name).unlink(missing_ok=True)
        project.rmdir()


def install_batch(
    repo_root: pathlib.Path,
    install_root: pathlib.Path,
    skills: Sequence[str],
    manifest: Mapping[str, object],
    python_runtime: pathlib.Path | None,
    python_skills: set[str],
) -> None:
    """Render and overlay selected skills, preserving all destination-only data."""

    action_assignments(repo_root, manifest, set(skills) if skills else None)
    install_root.mkdir(parents=True, exist_ok=True)
    lock = install_root / LOCK_NAME
    staging = install_root / f".ceratops-bootstrap-stage-{uuid.uuid4().hex}"
    lock_created = False
    try:
        lock.mkdir()
        lock_created = True
        staging.mkdir()
        for skill in skills:
            build_skill(
                repo_root, staging, manifest, skill,
                python_runtime if skill in python_skills else None,
            )
        # Inspect only paths being written; retained files are not audited.
        for source in staging.rglob("*"):
            target = install_root / source.relative_to(staging)
            require_inside(target, install_root)
            if target.is_symlink() or (target.exists() and unsafe_link(target)):
                raise ValueError(f"bootstrap destination cannot be a link: {target}")
        for skill in skills:
            shutil.copytree(staging / skill, install_root / skill, dirs_exist_ok=True)
            remove_legacy_skill_runtime(install_root / skill)
    finally:
        if lock_created:
            try:
                remove_stage(staging, install_root)
            finally:
                lock.rmdir()


def main() -> int:
    """Install or update selected skills without validation or retirement."""

    parser = argparse.ArgumentParser(
        description="Independently install or update declared repository skills."
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        help="Source repository root; defaults to this script's repository.",
    )
    parser.add_argument(
        "--install-root",
        type=pathlib.Path,
        help="Destination; defaults to $CODEX_HOME/skills.",
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Install only this declared skill; repeat as needed.",
    )
    args = parser.parse_args()
    repo_root = (
        args.repo_root or pathlib.Path(__file__).resolve().parents[1]
    ).resolve()
    default_root = (
        pathlib.Path(
            os.environ.get(
                "CODEX_HOME", pathlib.Path.home() / ".codex"
            )
        )
        / "skills"
    )
    destination = (
        args.install_root or default_root
    ).expanduser().resolve()
    try:
        manifest = read_manifest(repo_root)
        skills = declared_skills(manifest, args.skill)
        action_assignments(repo_root, manifest, set(skills) if args.skill else None)
        if skills:
            python_skills = declared_python_skills(repo_root, manifest, skills)
            python_runtime = (
                prepare_python_runtime(repo_root, destination) if python_skills else None
            )
            if python_skills and python_runtime is None:
                raise ValueError("declared Python skills require the source locked project")
            install_batch(
                repo_root, destination, skills, manifest, python_runtime, python_skills,
            )
    except (
        OSError,
        UnicodeError,
        ValueError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        return fail(str(exc))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
