"""Behavior tests for the shared repository validation runner."""

from __future__ import annotations

import contextlib
import importlib.util
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

INSTALLER_PATH = SCRIPTS_ROOT / "install-runtime.py"
INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "pixeltops_install_runtime",
    INSTALLER_PATH,
)
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
INSTALLER_SPEC.loader.exec_module(installer)

CANONICAL_VALIDATOR_PATH = SCRIPTS_ROOT / "validate-repository.py"
CANONICAL_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "pixeltops_canonical_validator",
    CANONICAL_VALIDATOR_PATH,
)
assert (
    CANONICAL_VALIDATOR_SPEC is not None
    and CANONICAL_VALIDATOR_SPEC.loader is not None
)
canonical_validator = importlib.util.module_from_spec(CANONICAL_VALIDATOR_SPEC)
CANONICAL_VALIDATOR_SPEC.loader.exec_module(canonical_validator)


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

    def test_lama_runtime_contract_uses_neutral_storage_identity(self) -> None:
        contract = json.loads(
            (
                REPO_ROOT
                / "skills"
                / "pixeltops-image-editor"
                / "references"
                / "runtime-contract.json"
            ).read_text(encoding="utf-8")
        )
        lama = contract["lama"]

        self.assertEqual(lama["relative_path"], "models/lama/big-lama.pt")
        self.assertEqual(lama["backend"], "repository-torchscript")
        self.assertEqual(lama["device"], "cpu")
        self.assertEqual(lama["md5"], "e3aa4aaa15225a33ec84f9f4bc47e500")

    def test_current_marker_requires_matching_environment_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = pathlib.Path(directory) / "environment"
            python_exe = environment / "Scripts" / "python.exe"
            python_exe.parent.mkdir(parents=True)
            python_exe.touch()
            installer.write_marker(
                environment,
                python_version="3.10.20",
                requirements_sha256="requirements",
                environment_sha256="a" * 64,
            )

            with mock.patch.object(
                installer,
                "environment_fingerprint",
                return_value="a" * 64,
            ):
                self.assertTrue(
                    installer.environment_is_current(
                        python_exe,
                        python_version="3.10.20",
                        requirements_sha256="requirements",
                    )
                )
            with mock.patch.object(
                installer,
                "environment_fingerprint",
                return_value="b" * 64,
            ):
                self.assertFalse(
                    installer.environment_is_current(
                        python_exe,
                        python_version="3.10.20",
                        requirements_sha256="requirements",
                    )
                )

    def test_legacy_marker_is_stale_without_running_fingerprint_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = pathlib.Path(directory) / "environment"
            python_exe = environment / "Scripts" / "python.exe"
            python_exe.parent.mkdir(parents=True)
            python_exe.touch()
            (environment / installer.ENVIRONMENT_MARKER).write_text(
                json.dumps(
                    {
                        "schema": "pixeltops-runtime-environment.v1",
                        "python": "3.10.20",
                        "requirements_sha256": "requirements",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(installer, "environment_fingerprint") as probe:
                self.assertFalse(
                    installer.environment_is_current(
                        python_exe,
                        python_version="3.10.20",
                        requirements_sha256="requirements",
                    )
                )
            probe.assert_not_called()

    def test_environment_fingerprint_is_deterministic_sha256(self) -> None:
        first = installer.environment_fingerprint(pathlib.Path(sys.executable))
        second = installer.environment_fingerprint(pathlib.Path(sys.executable))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in first))

    def test_environment_fingerprint_uses_isolated_interpreter(self) -> None:
        python_exe = pathlib.Path("environment") / "Scripts" / "python.exe"
        with mock.patch.object(
            installer,
            "run_checked",
            return_value="a" * 64,
        ) as run:
            self.assertEqual(installer.environment_fingerprint(python_exe), "a" * 64)

        run.assert_called_once_with(
            [
                str(python_exe),
                "-I",
                "-c",
                installer.ENVIRONMENT_FINGERPRINT_CODE,
            ]
        )

    def test_huggingface_install_uses_no_symlink_cache_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            python_exe = root / "envs" / "mask" / "Scripts" / "python.exe"
            with (
                mock.patch.object(
                    installer,
                    "environment_python",
                    return_value=python_exe,
                ),
                mock.patch.object(
                    installer,
                    "huggingface_declarations",
                    return_value=(("example/model", "a" * 40),),
                ),
                mock.patch.object(installer, "run_checked") as run,
            ):
                installer.install_huggingface_models(root)

        arguments = run.call_args.args[0]
        environment = run.call_args.kwargs["environment"]
        self.assertEqual(arguments[0], str(python_exe))
        self.assertEqual(arguments[-3:-1], ["example/model", "a" * 40])
        self.assertEqual(environment["HF_HUB_DISABLE_SYMLINKS"], "1")
        if "PATH" in installer.os.environ:
            self.assertEqual(environment["PATH"], installer.os.environ["PATH"])

    def test_canonical_validator_reports_evidence_write_errors_compactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            evidence_file = root / "evidence"
            evidence_file.mkdir()
            definition = {
                "id": "failing-check",
                "command": [sys.executable, "-c", "raise SystemExit(7)"],
                "cwd": ".",
                "exclusive": True,
                "python_packages": [],
            }
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    canonical_validator,
                    "CHECK_DEFINITIONS",
                    [definition],
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "validate-repository.py",
                        "--evidence-file",
                        str(evidence_file),
                    ],
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = canonical_validator.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["check"], "evidence-write")
            self.assertEqual(payload["exit_code"], 1)
            self.assertEqual(payload["evidence_file"], str(evidence_file.resolve()))
            self.assertIn("write_error", payload)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(stdout.getvalue().count("\n"), 1)
            self.assertFalse((root / ".evidence.tmp").exists())

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
