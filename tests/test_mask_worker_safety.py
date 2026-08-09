"""Functional safety checks for the image-worker command surfaces."""

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

import numpy as np
from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK_WORKER = (
    REPO_ROOT / "skills" / "pixeltops-image-editor" / "scripts" / "mask_worker.py"
)
WORKER_ROOT = MASK_WORKER.parent
sys.path.insert(0, str(WORKER_ROOT))

import inpaint_worker  # noqa: E402


class MaskWorkerSafetyTests(unittest.TestCase):
    def test_composite_preserves_pixels_outside_allowed_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            base_path = root / "base.png"
            edited_path = root / "edited.png"
            mask_path = root / "mask.png"
            output_path = root / "output.png"
            allowed_path = root / "allowed.png"
            tampered_path = root / "tampered.png"

            base_pixels = [
                ((index * 17) % 256, (index * 31) % 256, (index * 47) % 256)
                for index in range(16)
            ]
            edited_pixels = [(240, 120, 60)] * 16
            mask_pixels = [255 if index in {5, 6, 9, 10} else 0 for index in range(16)]

            base = Image.new("RGB", (4, 4))
            base.putdata(base_pixels)
            base.save(base_path)
            edited = Image.new("RGB", (4, 4))
            edited.putdata(edited_pixels)
            edited.save(edited_path)
            mask = Image.new("L", (4, 4))
            mask.putdata(mask_pixels)
            mask.save(mask_path)

            composite = subprocess.run(
                [
                    sys.executable,
                    str(MASK_WORKER),
                    "composite",
                    str(base_path),
                    str(edited_path),
                    str(mask_path),
                    str(output_path),
                    "--method",
                    "hard",
                    "--allowed-mask-output",
                    str(allowed_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(composite.returncode, 0, composite.stderr)
            audit = json.loads(composite.stdout)["audit"]
            self.assertEqual(audit["status"], "OK")
            self.assertEqual(audit["outsideChangedPixels"], 0)

            result_pixels = list(
                Image.open(output_path).convert("RGB").get_flattened_data()
            )
            for index, pixel in enumerate(result_pixels):
                expected = edited_pixels[index] if mask_pixels[index] else base_pixels[index]
                self.assertEqual(pixel, expected)

            tampered_pixels = result_pixels.copy()
            tampered_pixels[0] = (255, 255, 255)
            tampered = Image.new("RGB", (4, 4))
            tampered.putdata(tampered_pixels)
            tampered.save(tampered_path)

            verify = subprocess.run(
                [
                    sys.executable,
                    str(MASK_WORKER),
                    "verify",
                    str(base_path),
                    str(tampered_path),
                    str(allowed_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 1, verify.stdout)
            failure = json.loads(verify.stderr)
            self.assertEqual(failure["status"], "ERROR")
            self.assertEqual(failure["outsideChangedPixels"], 1)


class InpaintWorkerSafetyTests(unittest.TestCase):
    def test_headless_lama_preserves_dimensions_mask_semantics_and_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_path = root / "input.png"
            mask_path = root / "mask.png"
            output_path = root / "output.png"
            allowed_path = root / "allowed.png"
            base_pixels = [
                ((index * 13) % 256, (index * 29) % 256, (index * 43) % 256)
                for index in range(15)
            ]
            mask_pixels = [0] * 15
            mask_pixels[6] = 128
            mask_pixels[7] = 255
            mask_pixels[8] = 127

            source = Image.new("RGB", (5, 3))
            source.putdata(base_pixels)
            source.save(input_path)
            mask = Image.new("L", (5, 3))
            mask.putdata(mask_pixels)
            mask.save(mask_path)
            generated = np.full((3, 5, 3), (240, 120, 60), dtype=np.uint8)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                mock.patch.object(inpaint_worker, "run_lama", return_value=generated) as run,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = inpaint_worker.main(
                    [
                        "erase",
                        str(input_path),
                        str(mask_path),
                        str(output_path),
                        "--allowed-mask-output",
                        str(allowed_path),
                    ]
                )

            self.assertEqual((exit_code, stderr.getvalue()), (0, ""))
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["model"], "LaMa")
            self.assertEqual(payload["device"], "cpu")
            self.assertEqual(payload["allowedPixels"], 2)
            self.assertEqual(payload["outsideChangedPixels"], 0)
            self.assertEqual(run.call_args.args[0].shape, (3, 5, 3))
            self.assertEqual(int(np.count_nonzero(run.call_args.args[1])), 2)

            output = Image.open(output_path).convert("RGB")
            self.assertEqual(output.size, (5, 3))
            output_pixels = list(output.get_flattened_data())
            for index, pixel in enumerate(output_pixels):
                expected = (240, 120, 60) if index in {6, 7} else base_pixels[index]
                self.assertEqual(pixel, expected)
            allowed_pixels = list(Image.open(allowed_path).convert("L").get_flattened_data())
            self.assertEqual(
                allowed_pixels,
                [255 if index in {6, 7} else 0 for index in range(15)],
            )

    def test_empty_mask_uses_compact_error_contract_without_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_path = root / "input.png"
            mask_path = root / "mask.png"
            output_path = root / "output.png"
            allowed_path = root / "allowed.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(input_path)
            Image.new("L", (4, 4), 0).save(mask_path)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                mock.patch.object(inpaint_worker, "run_lama") as run,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = inpaint_worker.main(
                    [
                        "erase",
                        str(input_path),
                        str(mask_path),
                        str(output_path),
                        "--allowed-mask-output",
                        str(allowed_path),
                    ]
                )

            self.assertEqual((exit_code, stdout.getvalue()), (1, ""))
            self.assertEqual(json.loads(stderr.getvalue())["error"], "mask is empty")
            run.assert_not_called()
            self.assertFalse(output_path.exists())
            self.assertFalse(allowed_path.exists())


if __name__ == "__main__":
    unittest.main()
