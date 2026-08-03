"""Behavior tests for the shared repository validation runner."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import validate_repository as validator  # noqa: E402


def completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build a deterministic subprocess result for orchestration tests."""

    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class ValidateRepositoryTests(unittest.TestCase):
    def invoke(
        self,
        evidence_file: pathlib.Path,
        repo_root: pathlib.Path,
        results: list[subprocess.CompletedProcess[str]],
    ) -> tuple[int, str, str, mock.Mock]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(validator, "_run_command", side_effect=results) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = validator.main(
                ["--evidence-file", str(evidence_file)], repo_root=repo_root
            )
        return exit_code, stdout.getvalue(), stderr.getvalue(), run

    def test_orchestrates_all_checks_in_order_and_emits_only_ok(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_file = pathlib.Path(directory) / "validation.jsonl"
            results = [completed() for _ in validator.repository_checks()]

            exit_code, stdout, stderr, run = self.invoke(
                evidence_file, REPO_ROOT, results
            )

            expected_commands = [
                (
                    sys.executable,
                    "-m",
                    "json.tool",
                    "skills/skill-sections.json",
                ),
                (
                    sys.executable,
                    "-m",
                    "json.tool",
                    "skills/pixeltops-image-editor/references/runtime-contract.json",
                ),
                (
                    sys.executable,
                    "-m",
                    "compileall",
                    "-q",
                    "scripts",
                    "skills/pixeltops-image-editor/scripts",
                ),
                (sys.executable, "-m", "mypy"),
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
            ]
            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout, "OK\n")
            self.assertEqual(stderr, "")
            self.assertEqual(
                [call.args[0] for call in run.call_args_list], expected_commands
            )
            evidence = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["check"] for record in evidence],
                [
                    "section-manifest",
                    "runtime-contract",
                    "python-compilation",
                    "mypy",
                    "unit-tests",
                ],
            )
            self.assertTrue(all(record["status"] == "OK" for record in evidence))

    def test_propagates_failure_and_stops_with_compact_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_file = pathlib.Path(directory) / "validation.jsonl"
            results = [
                completed(),
                completed(),
                completed(7, "detail out", "detail err"),
            ]

            exit_code, stdout, stderr, run = self.invoke(
                evidence_file, REPO_ROOT, results
            )

            expected = {
                "status": "ERROR",
                "check": "python-compilation",
                "exitCode": 7,
                "evidenceFile": str(evidence_file.resolve()),
            }
            self.assertEqual(exit_code, 7)
            self.assertEqual(
                stdout,
                json.dumps(expected, separators=(",", ":"), sort_keys=True) + "\n",
            )
            self.assertEqual(stderr, "")
            self.assertEqual(run.call_count, 3)
            evidence = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(evidence[-1]["stdout"], "detail out")
            self.assertEqual(evidence[-1]["stderr"], "detail err")

    def test_handles_repository_and_evidence_paths_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = pathlib.Path(directory) / "repository with spaces"
            (repo_root / "skills").mkdir(parents=True)
            (repo_root / "scripts").mkdir()
            (repo_root / "skills" / "pixeltops-image-editor" / "scripts").mkdir(
                parents=True
            )
            (repo_root / "tests").mkdir()
            (repo_root / "skills" / "skill-sections.json").write_text(
                "{}\n", encoding="utf-8"
            )
            runtime_references = (
                repo_root / "skills" / "pixeltops-image-editor" / "references"
            )
            runtime_references.mkdir(parents=True)
            (runtime_references / "runtime-contract.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (repo_root / "scripts" / "example.py").write_text(
                "value = 1\n", encoding="utf-8"
            )
            (repo_root / "tests" / "test_smoke.py").write_text(
                "import unittest\n\n"
                "class SmokeTest(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            (repo_root / "pyproject.toml").write_text(
                '[tool.mypy]\npython_version = "3.12"\nfiles = ["scripts"]\n',
                encoding="utf-8",
            )
            evidence_file = (
                repo_root / "evidence folder" / "validation details.jsonl"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = validator.main(
                    ["--evidence-file", str(evidence_file)], repo_root=repo_root
                )

            self.assertEqual(
                (exit_code, stdout.getvalue(), stderr.getvalue()), (0, "OK\n", "")
            )
            self.assertTrue(evidence_file.is_file())
            evidence = [
                json.loads(line)
                for line in evidence_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(record["cwd"] == str(repo_root.resolve()) for record in evidence)
            )

    def test_usage_errors_are_compact_json_only(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = validator.main([], repo_root=REPO_ROOT)

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["status"], "ERROR")
        self.assertEqual(payload["error"], "usage")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().count("\n"), 1)

    def test_evidence_write_errors_are_compact_json_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = validator.main(
                    ["--evidence-file", directory], repo_root=REPO_ROOT
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["status"], "ERROR")
        self.assertEqual(payload["error"], "evidence")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(stdout.getvalue().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
