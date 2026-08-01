#!/usr/bin/env python3
"""Run the repository's local-and-CI validation contract.

Development dependencies are installed by the caller. The command writes full
subprocess evidence to the requested JSON Lines file and emits only ``OK`` or
one compact failure object to standard output.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Never, TextIO


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    """One ordered repository validation command."""

    name: str
    command: tuple[str, ...]


class UsageError(ValueError):
    """Represent a CLI error without argparse's multi-line console output."""


class CompactArgumentParser(argparse.ArgumentParser):
    """Raise usage errors so the public output contract remains machine-safe."""

    def error(self, message: str) -> Never:
        raise UsageError(message)


def repository_checks() -> tuple[Check, ...]:
    """Return the single ordered validation contract used locally and in CI."""

    return (
        Check(
            "section-manifest",
            (
                sys.executable,
                "-m",
                "json.tool",
                "templates/skill-sections.json",
            ),
        ),
        Check(
            "python-compilation",
            (
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "scripts",
                "skills/pixeltops-image-editor/scripts",
            ),
        ),
        Check("mypy", (sys.executable, "-m", "mypy")),
        Check(
            "unit-tests",
            (
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ),
        ),
    )


def _run_command(
    command: tuple[str, ...], repo_root: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    """Run one check without a shell and retain all detailed diagnostics."""

    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_evidence(evidence: TextIO, record: dict[str, object]) -> None:
    """Append one complete, immediately durable JSON Lines evidence record."""

    evidence.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    evidence.write("\n")
    evidence.flush()


def _run_validation(
    repo_root: pathlib.Path, evidence: TextIO, evidence_file: pathlib.Path
) -> tuple[int, dict[str, object] | None]:
    """Run checks in order, stopping at the first failure as CI previously did."""

    for check in repository_checks():
        base_record: dict[str, object] = {
            "check": check.name,
            "command": list(check.command),
            "cwd": str(repo_root),
        }
        try:
            result = _run_command(check.command, repo_root)
        except OSError as exc:
            _write_evidence(
                evidence,
                {
                    **base_record,
                    "status": "ERROR",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return 1, {
                "status": "ERROR",
                "check": check.name,
                "error": "execution",
                "evidenceFile": str(evidence_file),
            }

        _write_evidence(
            evidence,
            {
                **base_record,
                "status": "OK" if result.returncode == 0 else "ERROR",
                "exitCode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        if result.returncode != 0:
            exit_code = result.returncode if 1 <= result.returncode <= 255 else 1
            return exit_code, {
                "status": "ERROR",
                "check": check.name,
                "exitCode": result.returncode,
                "evidenceFile": str(evidence_file),
            }

    return 0, None


def _emit_json(payload: dict[str, object]) -> None:
    """Emit exactly one compact JSON object."""

    print(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    )


def _parse_evidence_file(argv: list[str] | None) -> pathlib.Path:
    parser = CompactArgumentParser(add_help=False)
    parser.add_argument("--evidence-file", required=True, type=pathlib.Path)
    return parser.parse_args(argv).evidence_file


def main(
    argv: list[str] | None = None, *, repo_root: pathlib.Path = REPO_ROOT
) -> int:
    """Run validation and preserve the compact public output contract."""

    try:
        requested_evidence_file = _parse_evidence_file(argv)
    except UsageError as exc:
        _emit_json({"status": "ERROR", "error": "usage", "message": str(exc)})
        return 2

    try:
        evidence_file = requested_evidence_file.expanduser()
        if not evidence_file.is_absolute():
            evidence_file = pathlib.Path.cwd() / evidence_file
        evidence_file = evidence_file.resolve()
        repo_root = repo_root.resolve()
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        with evidence_file.open("w", encoding="utf-8", newline="\n") as evidence:
            exit_code, failure = _run_validation(
                repo_root, evidence, evidence_file
            )
    except (OSError, RuntimeError) as exc:
        _emit_json(
            {
                "status": "ERROR",
                "error": "evidence",
                "message": str(exc),
            }
        )
        return 1

    if failure is not None:
        _emit_json(failure)
        return exit_code

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
