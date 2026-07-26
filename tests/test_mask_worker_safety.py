"""Functional safety checks for the mask-worker command surface."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MASK_WORKER = (
    REPO_ROOT / "skills" / "pixeltops-image-editor" / "scripts" / "mask_worker.py"
)


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


if __name__ == "__main__":
    unittest.main()
