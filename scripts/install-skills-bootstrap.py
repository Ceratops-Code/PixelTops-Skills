#!/usr/bin/env python3
"""Bootstrap this repository's declared skills without lifecycle dependencies.

This first-install-only helper stages one complete selected batch in a uniquely
named hidden directory under the install root. It validates that batch before
activation, never replaces an existing skill, and cleans only staging and lock
paths that it created.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from typing import cast


INSTALLER_VERSION = 10
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
    source_names = {
        skill.parent.name
        for skill in (repo_root / "skills").glob("*/SKILL.md")
    }
    if set(skills) != source_names:
        raise ValueError("skill assignments must match source SKILL.md folders")
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


def section_block(
    repo_root: pathlib.Path,
    manifest: Mapping[str, object],
    skill: str,
) -> str:
    """Resolve one skill's shared sections without lifecycle runtime code."""

    sections = cast(Mapping[str, object], manifest["sections"])
    assignments = cast(Mapping[str, object], manifest["skills"])
    selected = assignments[skill]
    if not isinstance(selected, list) or not selected:
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
        text = "\n".join(
            line
            for line in lines
            if not line.strip().startswith("<!-- INTERNAL:")
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


def payload_patterns(
    manifest: Mapping[str, object], skill: str
) -> list[str]:
    """Return only globally and directly declared payload patterns."""

    payloads = cast(Mapping[str, object], manifest.get("runtime_payloads", {}))
    result: list[str] = []
    for key in ("*", skill):
        values = payloads.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"runtime_payloads.{key} must be a string list")
        result.extend(cast(list[str], values))
    return result


def copy_payload(
    repo_root: pathlib.Path, pattern: str, target: pathlib.Path
) -> None:
    """Copy one resolved payload pattern into a staged skill tree."""

    if not safe_relative(pattern):
        raise ValueError(f"unsafe runtime payload pattern: {pattern!r}")
    matches = sorted(repo_root.glob(pattern))
    if not matches and not any(token in pattern for token in "*?["):
        raise ValueError(f"runtime payload does not exist: {pattern}")
    for source in matches:
        require_inside(source, repo_root)
        if unsafe_link(source):
            raise ValueError(f"runtime payload cannot be a link: {source}")
        relative = source.relative_to(repo_root)
        destination = target / relative
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
    patterns = payload_patterns(manifest, skill)
    for pattern in patterns:
        copy_payload(repo_root, pattern, target)
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
        "payload_patterns": patterns,
    }
    (target / MANIFEST_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_staged_batch(
    staging: pathlib.Path,
    manifest: Mapping[str, object],
    skills: Sequence[str],
) -> None:
    """Validate every staged tree and runtime identity before activation."""

    expected_source = manifest["runtime_source_id"]
    staged_names = sorted(
        path.name for path in staging.iterdir() if path.is_dir()
    )
    if staged_names != sorted(skills):
        raise ValueError("staged skill batch does not match the selection")
    for skill in skills:
        target = staging / skill
        validate_tree(target)
        skill_text = (target / "SKILL.md").read_text(encoding="utf-8")
        if skill_text.count(START) != 1 or skill_text.count(END) != 1:
            raise ValueError(f"{skill}: staged shared sections are invalid")
        metadata = json.loads(
            (target / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise ValueError(f"{skill}: staged runtime manifest is invalid")
        expected = {
            "schema": RUNTIME_MANIFEST_SCHEMA,
            "skill": skill,
            "runtime_source_id": expected_source,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{skill}: staged runtime identity is invalid")
        if "installer_version" in metadata:
            raise ValueError(
                f"{skill}: staged runtime manifest has obsolete installer_version"
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


def rollback_activation(
    staging: pathlib.Path,
    install_root: pathlib.Path,
    activated: Sequence[str],
) -> list[str]:
    """Move this run's activated skills back into staging before cleanup."""

    errors: list[str] = []
    for skill in reversed(activated):
        target = install_root / skill
        restored = staging / skill
        try:
            if restored.exists() or not target.is_dir() or unsafe_link(target):
                raise ValueError("activated path cannot be safely rolled back")
            target.rename(restored)
        except (OSError, ValueError) as exc:
            errors.append(f"{skill}: {exc}")
    return errors


def install_batch(
    repo_root: pathlib.Path,
    install_root: pathlib.Path,
    skills: Sequence[str],
    manifest: Mapping[str, object],
) -> None:
    """Stage, validate, and atomically activate one first-install batch."""

    install_root.mkdir(parents=True, exist_ok=True)
    lock = install_root / LOCK_NAME
    staging = install_root / f".ceratops-bootstrap-stage-{uuid.uuid4().hex}"
    lock_created = False
    activated: list[str] = []
    try:
        lock.mkdir()
        lock_created = True
        existing = [
            skill
            for skill in skills
            if (install_root / skill).exists()
            or (install_root / skill).is_symlink()
        ]
        if existing:
            raise ValueError(
                "bootstrap is first-install-only; destinations already exist: "
                + ", ".join(sorted(existing))
            )
        staging.mkdir()
        for skill in skills:
            build_skill(repo_root, staging, manifest, skill)
        validate_staged_batch(staging, manifest, skills)
        for skill in skills:
            target = install_root / skill
            if target.exists() or target.is_symlink():
                raise ValueError(
                    f"bootstrap destination appeared during activation: {target}"
                )
            (staging / skill).rename(target)
            activated.append(skill)
    except Exception:
        rollback_errors = rollback_activation(
            staging, install_root, activated
        )
        cleanup_error = ""
        try:
            remove_stage(staging, install_root)
        except (OSError, ValueError) as exc:
            cleanup_error = str(exc)
        if rollback_errors or cleanup_error:
            details = [*rollback_errors]
            if cleanup_error:
                details.append(cleanup_error)
            raise RuntimeError(
                "bootstrap rollback or cleanup failed: " + "; ".join(details)
            )
        raise
    else:
        remove_stage(staging, install_root)
    finally:
        if lock_created:
            try:
                lock.rmdir()
            except OSError as exc:
                raise RuntimeError(
                    f"bootstrap lock cleanup failed: {exc}"
                ) from exc


def main() -> int:
    """Install a complete selected batch only when every target is absent."""

    parser = argparse.ArgumentParser(
        description="Bootstrap declared repository skills."
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
        help="Bootstrap only this declared skill; repeat as needed.",
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
        if skills:
            install_batch(repo_root, destination, skills, manifest)
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
