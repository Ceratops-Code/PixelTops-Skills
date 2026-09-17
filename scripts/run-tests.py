#!/usr/bin/env python3
"""Run repository-owned Python tests in the scripts project's locked environment.

Run with ``uv run --project scripts --locked python scripts/run-tests.py`` to select the scripts project's
environment. Edit this generated runner when the repository's suites need different
behavior; compatibility never replaces an existing test implementation. Each run
owns and removes its pytest temporary directories after the child exits, including
when tests fail.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ["tests"]


def without_basetemp(arguments: list[str]) -> list[str]:
    """Remove outer-run locations before pytest validates every occurrence."""
    options = iter(arguments)
    retained = []
    for option in options:
        if option == "--basetemp":
            next(options, None)
        elif not option.startswith("--basetemp="):
            retained.append(option)
    return retained


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Repository-relative test files or directories.")
    parser.add_argument("--pytest-arg", action="append", default=[], help="One pytest argument; repeat using --pytest-arg=VALUE.")
    args = parser.parse_args()
    targets = args.targets or DEFAULT_TARGETS
    for target in targets:
        path = (ROOT / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(ROOT) or not path.exists():
            parser.error(f"test target must exist inside this repository: {target}")
    with tempfile.TemporaryDirectory(prefix="repository-tests-") as directory:
        temporary = pathlib.Path(directory)
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(temporary / "python-cache")
        # Pytest validates inherited basetemp values even when later CLI values
        # override them. An outer pytest location is an invalid ancestor here.
        try:
            environment["PYTEST_ADDOPTS"] = shlex.join(without_basetemp(
                shlex.split(environment.get("PYTEST_ADDOPTS", ""))
            ))
        except ValueError as exc:
            parser.error(f"invalid PYTEST_ADDOPTS: {exc}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *targets, *without_basetemp(args.pytest_arg),
                 "--basetemp", str(temporary / "pytest"),
                 "-o", "cache_dir=" + str(temporary / "pytest-cache")],
                cwd=ROOT, env=environment, check=False,
            )
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return result.returncode if result.returncode >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
