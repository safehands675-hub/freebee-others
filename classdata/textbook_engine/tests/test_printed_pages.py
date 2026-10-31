from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from textbook_engine.core import ContentEngineError
from textbook_engine.printed_pages import index_printed_pages


TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
)


class PrintedPageIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.package = Path(self.temporary_directory.name)
        (self.package / "extracted").mkdir()
        Image.new("RGB", (993, 1404), "white").save(
            self.package / "extracted" / "page_0011.png"
        )
        self.manifest_path = self.package / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "schema_version": "1.0",
            "source_id": "0123456789abcdef",
            "source": {"media_type": "pdf"},
            "assets": [{
                "kind": "pdf_page",
                "page": 11,
                "image_path": "extracted/page_0011.png",
            }],
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _ocr_result(
        text: str = "",
        confidence: float = 92.0,
        left: int = 894,
        top: int = 134,
        width: int = 21,
        height: int = 14,
        extra: tuple[tuple[str, float, int, int, int, int], ...] = (),
    ) -> SimpleNamespace:
        rows = []
        if text:
            rows.append(
                f"5\t1\t1\t1\t1\t1\t{left}\t{top}\t{width}\t{height}\t{confidence}\t{text}"
            )
        rows.extend(
            f"5\t1\t1\t1\t1\t1\t{x}\t{y}\t{w}\t{h}\t{conf}\t{value}"
            for value, conf, x, y, w, h in extra
        )
        return SimpleNamespace(
            returncode=0,
            stdout=(TSV_HEADER + "".join(f"{row}\n" for row in rows)).encode("utf-8"),
            stderr=b"",
        )

    def _set_pages(self, page_numbers: list[int]) -> None:
        assets = []
        for page_number in page_numbers:
            image_path = f"extracted/page_{page_number:04d}.png"
            Image.new("RGB", (993, 1404), "white").save(self.package / image_path)
            assets.append({
                "kind": "pdf_page",
                "page": page_number,
                "image_path": image_path,
            })
        self.manifest_path.write_text(json.dumps({
            "schema_version": "1.0",
            "source_id": "0123456789abcdef",
            "source": {"media_type": "pdf"},
            "assets": assets,
        }), encoding="utf-8")

    def test_indexes_number_from_page_corner_and_writes_mapping(self) -> None:
        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages.subprocess.run",
                return_value=self._ocr_result("3"),
            ) as run,
        ):
            result = index_printed_pages(self.manifest_path)

        index = json.loads((self.package / "printed_page_index.json").read_text(encoding="utf-8"))
        self.assertEqual(result["matched_pages"], 1)
        self.assertEqual(result["needs_review_pages"], 0)
        self.assertEqual(len(run.call_args_list), 1)
        self.assertEqual(index["pages"][0]["pdf_page"], 11)
        self.assertEqual(index["pages"][0]["printed_page"], 3)
        self.assertEqual(index["pages"][0]["status"], "matched")

    def test_conflicting_corner_numbers_are_flagged_not_auto_mapped(self) -> None:
        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages.subprocess.run",
                return_value=self._ocr_result(
                    "3",
                    extra=(("4", 92.0, 99, 134, 21, 14),),
                ),
            ),
        ):
            result = index_printed_pages(self.manifest_path)

        index = json.loads((self.package / "printed_page_index.json").read_text(encoding="utf-8"))
        self.assertEqual(result["matched_pages"], 0)
        self.assertEqual(result["needs_review_pages"], 1)
        self.assertEqual(index["pages"][0]["status"], "ambiguous")
        self.assertEqual(
            {candidate["printed_page"] for candidate in index["pages"][0]["candidates"]},
            {3, 4},
        )

    def test_footer_digits_outside_badge_are_not_accepted(self) -> None:
        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages.subprocess.run",
                return_value=self._ocr_result("4", left=700, top=134, width=17, height=13),
            ),
        ):
            result = index_printed_pages(self.manifest_path)

        index = json.loads((self.package / "printed_page_index.json").read_text(encoding="utf-8"))
        self.assertEqual(result["matched_pages"], 0)
        self.assertEqual(index["pages"][0]["status"], "needs_review")

    def test_unreadable_badge_uses_repeated_page_sequence(self) -> None:
        self._set_pages([18, 60, 80, 11, 19])
        ocr_results = []
        for text, confidence in (
            ("10", 96.8),
            ("52", 70.3),
            ("72", 87.8),
            ("", 0),
            ("", 0),
        ):
            ocr_results.append(self._ocr_result(text, confidence))

        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages.subprocess.run",
                side_effect=ocr_results,
            ),
        ):
            result = index_printed_pages(self.manifest_path)

        index = json.loads((self.package / "printed_page_index.json").read_text(encoding="utf-8"))
        page_11 = next(page for page in index["pages"] if page["pdf_page"] == 11)
        page_19 = next(page for page in index["pages"] if page["pdf_page"] == 19)
        self.assertEqual(result["matched_pages"], 5)
        self.assertEqual(index["sequence_offset"], -8)
        self.assertEqual(page_11["printed_page"], 3)
        self.assertEqual(page_11["status"], "matched")
        self.assertEqual(page_11["verification_method"], "page_sequence")
        self.assertEqual(page_19["printed_page"], 11)
        self.assertEqual(page_19["verification_method"], "page_sequence")

    def test_missing_tesseract_fails_with_install_guidance(self) -> None:
        with patch("textbook_engine.ocr.shutil.which", return_value=None):
            with self.assertRaisesRegex(ContentEngineError, "Install Tesseract OCR"):
                index_printed_pages(self.manifest_path)

    def test_ocr_timeout_fails_with_page_context(self) -> None:
        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages.subprocess.run",
                side_effect=subprocess.TimeoutExpired("tesseract.exe", 30),
            ),
        ):
            with self.assertRaisesRegex(ContentEngineError, "timed out.*page_0011"):
                index_printed_pages(self.manifest_path)


if __name__ == "__main__":
    unittest.main()
