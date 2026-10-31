from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pymupdf

from textbook_engine.book_pipeline import (
    index_book_printed_pages,
    ocr_book_page_text,
    process_book,
)
from textbook_engine.book_store import FileSystemBookStore
from textbook_engine.chapters import _generate_chapter_content
from textbook_engine.core import ContentEngineError
from textbook_engine.pipeline import PIPELINE_VERSION, process_and_load_pdf_pages


class BookIngestionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.books_dir = self.root / "books"
        self.source = self.root / "Class_04_Science.pdf"
        self._make_pdf()
        self.store = FileSystemBookStore(self.books_dir)

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

    def _process_book(self, source: Path | None = None, **kwargs: Any) -> dict[str, Any]:
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
            return process_book(
                source or self.source,
                books_dir=self.books_dir,
                data_dir=self.root / "data",
                **kwargs,
            )

    def _make_pdf(self, *, include_chapters: bool = True) -> None:
        document = pymupdf.open()
        contents = document.new_page()
        contents.insert_text(
            (72, 72),
            "CONTENTS\nUNIT 1\nUNIT 2"
            if include_chapters
            else "CONTENTS\nNo chapter headings are listed.",
        )
        first = document.new_page()
        first.insert_text(
            (72, 72),
            "UNIT - 1\nPLANTS\nA plant grows from a seed."
            if include_chapters
            else "A plant grows from a seed.",
        )
        second = document.new_page()
        second.insert_text((72, 72), "Plants need water and sunlight.")
        last = document.new_page()
        if include_chapters:
            last.insert_text((72, 72), "UNIT - 2\nANIMALS\nAnimals need food.")
        else:
            last.insert_text((72, 72), "More content without chapter headings.")
        document.save(self.source)
        document.close()

    @staticmethod
    def _completion(
        messages: list[dict[str, Any]],
        model: str,
        api_key: str,
    ) -> str:
        return json.dumps({
            "chapter_title": None,
            "introduction": "A source-grounded lesson overview.",
            "learning_objectives": ["Describe the lesson topic."],
            "sections": [{
                "title": "Main idea",
                "explanation": "A short source-grounded explanation.",
                "definitions": [],
                "examples": [],
                "important_facts": ["The source includes a key fact."],
                "formulas": [],
                "questions": ["What does the source explain?"],
            }],
        })

    def test_publishes_all_pages_and_does_not_guess_printed_page_numbers(self) -> None:
        with patch(
            "textbook_engine.pipeline.process_and_load_pdf_pages",
            wraps=process_and_load_pdf_pages,
        ) as load_pages:
            result = self._process_book(
                language="English",
                model="gemini-test",
                completion=self._completion,
            )
        load_pages.assert_called_once()

        manifest = self.store.read_manifest(result["book_id"], result["version_id"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["physical_pdf_pages"], 4)
        self.assertEqual(result["printed_page_numbers"], "unknown_until_ocr")
        self.assertEqual(
            [chapter["chapter_id"] for chapter in manifest["chapters"]],
            ["front_matter", "unit_01", "unit_02"],
        )
        self.assertEqual(manifest["source_pdf"]["physical_page_count"], 4)
        self.assertEqual(
            manifest["source_pdf"]["processing_pipeline_version"],
            PIPELINE_VERSION,
        )
        all_pages = []
        for chapter_entry in manifest["chapters"]:
            chapter = self.store.read_chapter(
                result["book_id"],
                result["version_id"],
                chapter_entry["chapter_id"],
            )
            all_pages.extend(chapter["pages"])
        self.assertEqual(
            [page["source_provenance"]["source_pdf_page_number_1_indexed"] for page in all_pages],
            [1, 2, 3, 4],
        )
        self.assertEqual(result["page_images_rendered"], 4)
        for page in all_pages:
            image_reference = page["visuals"]["page_image"]
            self.assertTrue(
                self.store.asset_path(
                    result["book_id"],
                    result["version_id"],
                    image_reference,
                ).is_file()
            )
        self.assertTrue(all(
            page["source_provenance"]["printed_page_number"] is None
            for page in all_pages
        ))
        self.assertEqual(all_pages[1]["source_text"]["extracted_text_raw_ordered"].splitlines()[0], "UNIT - 1")
        self.assertEqual(
            all_pages[1]["source_provenance"]["text_extraction_method"],
            "native_pdf",
        )
        source_id = hashlib.sha256(self.source.read_bytes()).hexdigest()[:16]
        source_manifest = json.loads(
            (self.root / "data" / "sources" / source_id / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(source_manifest["processing"]["status"], "complete")
        self.assertEqual(
            source_manifest["assets"][1]["text_path"],
            "extracted/page_0002.txt",
        )
        self.assertEqual(
            self.store.read_chapter(
                result["book_id"],
                result["version_id"],
                "unit_02",
            )["generation_provenance"]["model_id"],
            "gemini-test",
        )

    def test_repeat_run_reuses_published_version_without_repeating_generation(self) -> None:
        with patch(
            "textbook_engine.book_pipeline._generate_chapter_content",
            wraps=_generate_chapter_content,
        ) as generation:
            first = self._process_book(
                model="gemini-test",
                completion=self._completion,
            )
            self.assertEqual(generation.call_count, 2)
            second = self._process_book(
                model="gemini-test",
                completion=self._completion,
            )

        self.assertEqual(second["status"], "already_published")
        self.assertEqual(first["version_id"], second["version_id"])
        self.assertEqual(generation.call_count, 2)

    def test_renamed_identical_pdf_reuses_published_version(self) -> None:
        renamed_source = self.root / "renamed_textbook.pdf"
        renamed_source.write_bytes(self.source.read_bytes())
        first = self._process_book(
            class_number=4,
            subject="Science",
            language="English",
            model="gemini-test",
            completion=self._completion,
        )
        with patch(
            "textbook_engine.book_pipeline._generate_chapter_content",
            wraps=_generate_chapter_content,
        ) as generation:
            renamed = self._process_book(
                renamed_source,
                class_number=4,
                subject="Science",
                language="English",
                model="gemini-test",
                completion=self._completion,
            )

        self.assertEqual(renamed["status"], "already_published")
        self.assertEqual(first["book_id"], renamed["book_id"])
        self.assertEqual(first["version_id"], renamed["version_id"])
        generation.assert_not_called()

    def test_source_without_detectable_chapters_is_not_published(self) -> None:
        self._make_pdf(include_chapters=False)
        with self.assertRaisesRegex(ContentEngineError, "explicit"):
            self._process_book(
                completion=self._completion,
            )
        self.assertFalse(any(self.books_dir.rglob("book.json")))

    def test_printed_page_ocr_writes_versioned_rebuildable_sidecar(self) -> None:
        result = self._process_book(
            model="gemini-test",
            completion=self._completion,
        )
        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages._ocr_footer",
                side_effect=lambda _exe, image: [{
                    "printed_page": int(image.stem[-4:]),
                    "confidence": 92.0,
                    "corner": "left",
                }],
            ) as ocr,
        ):
            indexed = index_book_printed_pages(
                result["book_id"],
                result["version_id"],
                books_dir=self.root / "ocr_indexes",
                store=self.store,
            )

        self.assertEqual(indexed["status"], "ok")
        self.assertEqual(indexed["matched_pages"], 4)
        self.assertEqual(ocr.call_count, 4)
        index_path = Path(indexed["index"])
        self.assertTrue(index_path.samefile(
            self.root
            / "ocr_indexes"
            / "_indexes"
            / result["book_id"]
            / result["version_id"]
            / "printed_page_index.json",
        ))
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        manifest = self.store.read_manifest(result["book_id"], result["version_id"])
        self.assertEqual(index_data["source_sha256"], manifest["source_pdf"]["sha256"])
        self.assertEqual(index_data["pages"][2]["printed_page"], 3)

    def test_book_page_ocr_recovers_expected_script_and_rebuilds_legacy_cache(self) -> None:
        self.source = self.root / "Class_04_First_Language_Telugu.pdf"
        self._make_pdf()
        result = self._process_book(
            model="gemini-test",
            completion=self._completion,
        )
        manifest = self.store.read_manifest(result["book_id"], result["version_id"])
        source_id = hashlib.sha256(
            f"{result['book_id']}:{result['version_id']}".encode("utf-8")
        ).hexdigest()[:16]
        cache_path = (
            self.books_dir
            / "_indexes"
            / result["book_id"]
            / result["version_id"]
            / "page_text_ocr.json"
        )
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(
            json.dumps({
                "schema_version": 1,
                "source_id": source_id,
                "source_sha256": manifest["source_pdf"]["sha256"],
                "pages": {
                    "2": {
                        "text": "Cached Latin text from the previous OCR policy.",
                        "method": "native_pdf",
                        "needs_review": False,
                    },
                },
            }),
            encoding="utf-8",
        )
        recognized_text = ("\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 " * 6).strip()
        with (
            patch(
                "textbook_engine.ocr.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.ocr.resolve_tesseract_languages",
                return_value={"eng", "tel"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value=recognized_text,
            ) as page_ocr,
        ):
            pages = ocr_book_page_text(
                result["book_id"],
                result["version_id"],
                books_dir=self.books_dir,
                store=self.store,
            )

        self.assertEqual(pages[2], recognized_text)
        self.assertEqual(page_ocr.call_count, 4)
        refreshed_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(refreshed_cache["schema_version"], 2)

    def test_explicit_book_language_is_used_for_lazy_ocr(self) -> None:
        result = self._process_book(
            language="Telugu",
            model="gemini-test",
            completion=self._completion,
        )
        manifest = self.store.read_manifest(result["book_id"], result["version_id"])
        self.assertEqual(manifest["source_pdf"]["ocr_language"], "tel")
        recognized_text = ("\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 " * 8).strip()

        with (
            patch(
                "textbook_engine.ocr.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.ocr.resolve_tesseract_languages",
                return_value={"tel"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value=recognized_text,
            ) as page_ocr,
        ):
            pages = ocr_book_page_text(
                result["book_id"],
                result["version_id"],
                pdf_page=2,
                books_dir=self.books_dir,
                store=self.store,
            )

        self.assertEqual(pages[2], recognized_text)
        page_ocr.assert_called_once_with("tesseract.exe", unittest.mock.ANY, "tel")

    def test_process_book_rejects_unsupported_explicit_ocr_language(self) -> None:
        with self.assertRaisesRegex(ContentEngineError, "Unsupported OCR language"):
            self._process_book(
                language="Klingon",
                model="gemini-test",
                completion=self._completion,
            )
