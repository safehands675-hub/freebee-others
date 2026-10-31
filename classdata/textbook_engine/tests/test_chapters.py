from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pymupdf
from PIL import Image

from textbook_engine.chapters import (
    _chapter_completion,
    _chapter_span,
    _chapter_span_from_index,
    _markers_in_text,
    _split_chunks,
    process_chapter,
)
from textbook_engine.core import ContentEngineError
from textbook_engine.pipeline import process_and_load_pdf_pages


class ChapterProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output_dir = self.root / "processed"
        self.source = self.root / "Class_01_English_Medium_Science.pdf"
        self._make_pdf()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _mock_image_extraction(*args: object, **kwargs: object) -> dict[str, int]:
        return {"unique_images": 0, "page_placements": 0}

    @staticmethod
    def _mock_page_index(*args: object, **kwargs: object) -> dict[str, int]:
        manifest_path = Path(args[0])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = manifest_path.parent / "printed_page_index.json"
        index_path.write_text(
            json.dumps({
                "schema_version": 1,
                "source_id": manifest["source_id"],
                "pages": [
                    {
                        "pdf_page": asset["page"],
                        "printed_page": None,
                        "confidence": None,
                        "status": "needs_review",
                    }
                    for asset in manifest["assets"]
                    if asset.get("kind") == "pdf_page"
                ],
            }),
            encoding="utf-8",
        )
        return {"matched_pages": 0, "needs_review_pages": len(manifest["assets"])}

    def _process_chapter(self, chapter_number: int, **kwargs: Any) -> dict[str, Any]:
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng", "tel", "hin", "san", "urd"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value="",
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ),
        ):
            return process_chapter(
                self.source,
                chapter_number,
                self.output_dir,
                data_dir=self.root / "data",
                **kwargs,
            )

    def _make_pdf(self) -> None:
        document = pymupdf.open()
        first = document.new_page()
        first.insert_text((72, 72), "UNIT - 1\nPLANTS\nA plant grows from a seed.")
        image_buffer = BytesIO()
        Image.new("RGB", (40, 30), "green").save(image_buffer, format="PNG")
        image_data = image_buffer.getvalue()
        first.insert_image(pymupdf.Rect(72, 120, 172, 195), stream=image_data)
        second = document.new_page()
        second.insert_text((72, 72), "Plants need water and sunlight.")
        second.insert_image(pymupdf.Rect(72, 120, 172, 195), stream=image_data)
        third = document.new_page()
        third.insert_text((72, 72), "UNIT - 2\nANIMALS\nAnimals need food.")
        document.save(self.source)
        document.close()

    @staticmethod
    def _completion(
        messages: list[dict[str, Any]],
        model: str,
        api_key: str,
    ) -> str:
        return json.dumps({
            "chapter_title": "Plants",
            "introduction": "This lesson introduces plants.",
            "learning_objectives": ["Describe what plants need."],
            "sections": [{
                "title": "Growing plants",
                "explanation": "Plants grow from seeds and need water and sunlight.",
                "definitions": [{"term": "seed", "meaning": "A part that can grow into a plant."}],
                "examples": ["A seed can grow into a plant."],
                "important_facts": ["Plants need water and sunlight."],
                "formulas": [],
                "questions": ["What do plants need to grow?"],
            }],
        })

    def test_processes_only_the_selected_unit_and_writes_valid_json_and_images(self) -> None:
        with patch(
            "textbook_engine.pipeline.process_and_load_pdf_pages",
            wraps=process_and_load_pdf_pages,
        ) as load_pages:
            result = self._process_chapter(
                1,
                completion=self._completion,
            )
        load_pages.assert_called_once()

        chapter_dir = Path(result["output"])
        chapter = json.loads((chapter_dir / "chapter.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["boundary_marker"], "unit")
        self.assertEqual(chapter["chapter"]["title"], "PLANTS")
        self.assertIn("Plants need water and sunlight.", chapter["original_text"])
        self.assertNotIn("Animals need food.", chapter["original_text"])
        self.assertEqual(chapter["student_questions"], ["What do plants need to grow?"])
        self.assertTrue(chapter["images"])
        embedded_images = [
            image for image in chapter["images"] if image["kind"] == "embedded_image"
        ]
        self.assertEqual(len(embedded_images), 1)
        self.assertEqual(len(embedded_images[0]["related_text"]), 2)
        for image in chapter["images"]:
            self.assertTrue((chapter_dir / image["path"]).is_file())
            self.assertEqual(image["visual_understanding"], "not_performed")
        self.assertTrue((chapter_dir / "extracted_text.txt").is_file())
        self.assertTrue((chapter_dir / "processing_report.json").is_file())

    def test_repeated_runs_update_the_same_output_files(self) -> None:
        first = self._process_chapter(
            1,
            completion=self._completion,
        )
        second = self._process_chapter(
            1,
            completion=self._completion,
        )

        self.assertEqual(first["output"], second["output"])
        chapter_dir = Path(first["output"])
        self.assertEqual(len(list((chapter_dir / "images").glob("*.png"))), first["images_extracted"])

    def test_missing_end_boundary_fails_instead_of_using_remaining_pdf(self) -> None:
        with self.assertRaisesRegex(ContentEngineError, "could not identify where it ends"):
            _chapter_span(["CHAPTER 1\nTopic text", "more topic text"], 1)

    def test_contents_page_is_not_used_as_the_chapter_start(self) -> None:
        start, end, kind, title = _chapter_span(
            [
                "CONTENTS\nCHAPTER 1\nCHAPTER 2",
                "CHAPTER 1\nFIRST TOPIC\nLesson content.",
                "CHAPTER 2\nNEXT TOPIC\nOther content.",
            ],
            1,
        )
        self.assertEqual((start, end, kind, title), (1, 2, "chapter", "FIRST TOPIC"))

    def test_numbered_textbook_chapter_headings_are_recognized(self) -> None:
        pages = [
            "INDEX\n1 FOOD COMPONENTS 1\n2 ACIDS - BASES 10",
            "1 FOOD COMPONENTS\nFood supplies the energy we need.",
            "More food components content.",
            "2 ACIDS - BASES\nWe use a lot of substances.",
            (
                "lesson we try to learn about some changes. "
                "Appearance of the Full Moon 7"
            ),
        ]

        self.assertEqual(
            _markers_in_text(pages[1])[0],
            {"kind": "chapter", "number": 1, "title": "FOOD COMPONENTS"},
        )
        self.assertEqual(
            _chapter_span(pages, 1),
            (1, 3, "chapter", "FOOD COMPONENTS"),
        )
        self.assertFalse(
            any(
                marker["kind"] == "lesson" and marker["number"] == 7
                for marker in _markers_in_text(pages[4])
            )
        )
        self.assertEqual(_markers_in_text("Table - 3\nColumn A   Column B"), [])

    def test_gemini_chapter_index_uses_physical_pdf_boundaries(self) -> None:
        pages = ["Chapter one content.", "Chapter two content.", "Appendix."]
        index_json = json.dumps({
            "chapters": [
                {
                    "number": 1,
                    "title": "First chapter",
                    "start_pdf_page": 1,
                    "marker_kind": "chapter",
                },
                {
                    "number": 2,
                    "title": "Second chapter",
                    "start_pdf_page": 2,
                    "marker_kind": "chapter",
                },
            ],
        })

        self.assertEqual(
            _chapter_span_from_index(index_json, pages, 1),
            (0, 1, "chapter", "First chapter"),
        )

    def test_gemini_chapter_index_rejects_duplicate_boundaries(self) -> None:
        index_json = json.dumps({
            "chapters": [
                {
                    "number": 1,
                    "title": "First chapter",
                    "start_pdf_page": 1,
                    "marker_kind": "chapter",
                },
                {
                    "number": 1,
                    "title": "Duplicate chapter",
                    "start_pdf_page": 2,
                    "marker_kind": "chapter",
                },
            ],
        })

        with self.assertRaisesRegex(ContentEngineError, "invalid or duplicate"):
            _chapter_span_from_index(index_json, ["Page 1", "Page 2"], 1)

    def test_layout_split_unit_headings_are_recognized(self) -> None:
        first_page = (
            "UNIT                              UNIT - 1\n"
            "PLANT LIFE   1\n"
            "UNIT PLANT LIFE\n"
        )
        wrapped_heading = "Unit The Town Mouse and\nthe Country Mouse 1\n"
        second_start = "UNIT\nGOING TO SCHOOL   2\n"
        third_start = "UNIT\nAMMUS SCHOOL   3\n"
        self.assertEqual(_markers_in_text(first_page)[0]["number"], 1)
        start, end, kind, title = _chapter_span(
            [
                wrapped_heading,
                "Story content.",
                second_start,
                "School content.",
                third_start,
            ],
            1,
        )
        self.assertEqual(
            (start, end, kind, title),
            (0, 2, "unit", "The Town Mouse and the Country Mouse"),
        )
        self.assertEqual(_markers_in_text(first_page)[0]["number"], 1)

    def test_unit_headings_with_number_and_title_on_separate_lines_are_recognized(self) -> None:
        pages = [
            "Unit The Town Mouse and\nthe Country Mouse 1\n",
            "First unit content.",
            "Unit\nPuru, the Brave\n3\n",
            "Third unit content.",
            "Unit\n4 Tenali Paints a Horse\n",
        ]

        self.assertEqual(
            _chapter_span(pages, 3),
            (2, 4, "unit", "Puru, the Brave"),
        )

    def test_inline_chapter_title_before_marker_is_recognized(self) -> None:
        pages = [
            "CONTENTS\nForce 1\nFriction 2\n",
            (
                "Force                                             Chapter     1\n"
                "Activity-1\nClassify each action as a push or pull.\n"
            ),
            "More force lesson content and examples.\n",
            "Chapter       Friction     2\nFriction content.\n",
        ]

        self.assertEqual(
            _markers_in_text(pages[1])[0],
            {"kind": "chapter", "number": 1, "title": "Force"},
        )
        self.assertEqual(
            _chapter_span(pages, 1),
            (1, 3, "chapter", "Force"),
        )

    def test_chunking_preserves_the_exact_source_text(self) -> None:
        text = ("First paragraph.\n\n" + ("word " * 2_500) + "\nLast paragraph.")
        chunks = _split_chunks(text, maximum=1_000)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk) <= 1_000 for chunk in chunks))

    def test_chapter_gemini_call_uses_selected_model_and_api_key(self) -> None:
        with patch(
            "textbook_engine.chapters.complete_chapter_chunk",
            return_value="{}",
        ) as gemini_call:
            _chapter_completion([], "test-model", "test-key")

        gemini_call.assert_called_once_with([], "test-model", "test-key")

    def test_malformed_gemini_json_is_rejected_before_output_is_written(self) -> None:
        with self.assertRaisesRegex(ContentEngineError, "malformed JSON"):
            self._process_chapter(
                1,
                completion=lambda messages, model, api_key: "not JSON",
            )
        self.assertFalse((self.output_dir / "class_1").exists())

    def test_missing_gemini_key_is_reported_before_processing(self) -> None:
        with (
            patch(
                "textbook_engine.chapters._load_gemini_api_key",
                side_effect=ContentEngineError("Gemini API key is missing."),
            ),
            self.assertRaisesRegex(ContentEngineError, "Gemini API key is missing"),
        ):
            self._process_chapter(1)

    def test_saved_chapter_records_gemini_model_and_prompt_provenance(self) -> None:
        result = self._process_chapter(
            1,
            model="gemini-test",
            completion=self._completion,
        )

        chapter_dir = Path(result["output"])
        chapter = json.loads((chapter_dir / "chapter.json").read_text(encoding="utf-8"))
        report = json.loads(
            (chapter_dir / "processing_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(chapter["generation_provenance"]["provider"], "google_gemini")
        self.assertEqual(chapter["generation_provenance"]["model_id"], "gemini-test")
        self.assertEqual(chapter["generation_provenance"]["prompt_version"], "chapter-understanding-v1")
        self.assertFalse(chapter["generation_provenance"]["human_reviewed"])
        self.assertEqual(report["generation_model"], "gemini-test")


if __name__ == "__main__":
    unittest.main()
