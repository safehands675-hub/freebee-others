from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from textbook_engine.ocr import resolve_tesseract_command


class TesseractResolutionTests(unittest.TestCase):
    def test_loads_user_configuration_when_parent_process_environment_is_stale(self) -> None:
        executable = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        tessdata = r"C:\Users\Student\AppData\Local\Tesseract-OCR\tessdata"
        with (
            patch.dict(os.environ) as environment,
            patch(
                "textbook_engine.ocr._windows_user_environment",
                return_value={
                    "TESSERACT_CMD": executable,
                    "TESSDATA_PREFIX": tessdata,
                },
            ),
            patch("textbook_engine.ocr.shutil.which", return_value=executable) as which,
        ):
            environment.pop("TESSERACT_CMD", None)
            environment.pop("TESSDATA_PREFIX", None)

            resolved = resolve_tesseract_command()

            self.assertEqual(resolved, executable)
            self.assertEqual(environment["TESSDATA_PREFIX"], tessdata)
            which.assert_called_once_with(executable)

    def test_process_environment_overrides_user_configuration(self) -> None:
        process_executable = r"D:\Tools\tesseract.exe"
        user_executable = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        with (
            patch.dict(os.environ, {"TESSERACT_CMD": process_executable}),
            patch(
                "textbook_engine.ocr._windows_user_environment",
                return_value={"TESSERACT_CMD": user_executable},
            ),
            patch(
                "textbook_engine.ocr.shutil.which",
                return_value=process_executable,
            ) as which,
        ):
            self.assertEqual(resolve_tesseract_command(), process_executable)
            which.assert_called_once_with(process_executable)

    def test_falls_back_to_path_when_no_explicit_configuration_exists(self) -> None:
        with (
            patch.dict(os.environ) as environment,
            patch(
                "textbook_engine.ocr._windows_user_environment",
                return_value={},
            ),
            patch(
                "textbook_engine.ocr.shutil.which",
                return_value="tesseract",
            ) as which,
        ):
            environment.pop("TESSERACT_CMD", None)

            self.assertEqual(resolve_tesseract_command(), "tesseract")
            which.assert_called_once_with("tesseract")


if __name__ == "__main__":
    unittest.main()
