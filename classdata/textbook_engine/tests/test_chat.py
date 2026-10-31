from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any
import urllib.request
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from textbook_engine.chat import (
    ChatError,
    ChatService,
    ContentCatalog,
    LocalChatServer,
    _call_groq,
    _grade_prompt,
)
from textbook_engine.book_store import FileSystemBookStore
from textbook_engine.pipeline import PIPELINE_VERSION


class ChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        self.books = [
            (1, "0123456789abcdef", "Class_01_English_Medium_English.pdf", [
                (1, "A tree has green leaves. Birds can sit in a tree."),
                (2, "A flower can grow near a tree."),
            ]),
            (1, "fedcba9876543210", "Class_01_English_Medium_Science.pdf", [
                (1, "A volcano can release hot lava."),
            ]),
            (1, "aaaaaaaaaaaaaaaa", "Class_01_English_Medium_First_Language_(Telugu).pdf", [
                (11, "\uf0bb" * 12),
            ]),
            (1, "bbbbbbbbbbbbbbbb", "Class_01_English_Medium_Second_Language_(Telugu).pdf", [
                (11, "\uf0fc" * 12),
            ]),
            (2, "1111111111111111", "Class_02_English_Medium_English.pdf", [
                (1, "Plants need water and sunlight to grow."),
            ]),
        ]
        summary_books: list[dict[str, Any]] = []
        for class_number, source_id, filename, pages in self.books:
            package_dir = self.data_root / "books" / source_id
            (package_dir / "text").mkdir(parents=True)
            (package_dir / "pages").mkdir()
            assets = []
            for page_number, text in pages:
                text_path = f"text/page_{page_number:04d}.txt"
                image_path = f"pages/page_{page_number:04d}.png"
                (package_dir / text_path).write_text(text, encoding="utf-8")
                Image.new("RGB", (80, 60), "white").save(package_dir / image_path)
                assets.append({
                    "kind": "pdf_page",
                    "page": page_number,
                    "text_path": text_path,
                    "image_path": image_path,
                })
            manifest_path = package_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"assets": assets}), encoding="utf-8")
            printed_page_records = [
                {
                    "pdf_page": page_number,
                    "printed_page": (
                        3 if source_id == "aaaaaaaaaaaaaaaa" and page_number == 11
                        else page_number
                    ),
                    "confidence": 92.0,
                    "status": "matched",
                }
                for page_number, _ in pages
            ]
            (package_dir / "printed_page_index.json").write_text(
                json.dumps({
                    "source_id": source_id,
                    "pages": printed_page_records,
                }),
                encoding="utf-8",
            )
            summary_books.append({
                "class": class_number,
                "source_id": source_id,
                "filename": filename,
                "manifest": str(manifest_path),
                "text_characters": 100,
                "private_use_characters": 0,
            })

        self.summary_path = self.data_root / "summary.json"
        self.summary_path.write_text(
            json.dumps({"books": summary_books}),
            encoding="utf-8",
        )
        self.prepared_book_dir = self.data_root / "prepared_books" / "class_01" / "english"
        (self.prepared_book_dir / "chapters").mkdir(parents=True)
        (self.prepared_book_dir / "images").mkdir()
        (self.prepared_book_dir / "book.json").write_text(
            json.dumps({
                "schema_version": 1,
                "book_id": "prepared_english",
                "class": 1,
                "subject": "English",
                "title": "English (Prepared)",
                "chapters": [{
                    "chapter_id": "prepared_unit_01",
                    "number": 1,
                    "file": "chapters/unit_01.json",
                }],
            }),
            encoding="utf-8",
        )
        Image.new("RGB", (80, 60), "blue").save(
            self.prepared_book_dir / "images" / "bird.png"
        )
        (self.prepared_book_dir / "chapters" / "unit_01.json").write_text(
            json.dumps({
                "schema_version": 1,
                "book_id": "prepared_english",
                "chapter_id": "prepared_unit_01",
                "number": 1,
                "title": "A Small Bird",
                "summary": "A bird can fly.",
                "sections": [{
                    "section_id": "birds",
                    "heading": "Birds",
                    "source_text": "A bird has wings. It can fly.",
                    "explanation": "Birds use their wings to fly.",
                    "key_terms": ["wings"],
                    "key_facts": ["Birds can fly."],
                    "questions": [{
                        "question": "What does a bird use to fly?",
                        "answer": "It uses its wings.",
                        "image_ids": ["bird_img001"],
                    }],
                    "image_ids": ["bird_img001"],
                }],
                "images": [{
                    "image_id": "bird_img001",
                    "file": "images/bird.png",
                    "kind": "page",
                    "description": "A small blue bird with spread wings.",
                    "labels": ["bird"],
                    "related_section_ids": ["birds"],
                    "visual_understanding": "described",
                    "source_pdf_page": 11,
                }],
            }),
            encoding="utf-8",
        )
        self.catalog = ContentCatalog(self.data_root, self.summary_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_current_source_package(
        data_root: Path,
        filename: str,
        class_number: int,
        source_bytes: bytes,
        *,
        pipeline_version: str = PIPELINE_VERSION,
        include_image: bool = True,
    ) -> tuple[str, Path]:
        checksum = hashlib.sha256(source_bytes).hexdigest()
        source_id = checksum[:16]
        package_dir = data_root / "sources" / source_id
        (package_dir / "extracted").mkdir(parents=True, exist_ok=True)
        imported_pdf = package_dir / "source.pdf"
        imported_pdf.write_bytes(source_bytes)
        text_path = package_dir / "extracted" / "page_0001.txt"
        text_path.write_text("A seed can grow into a plant.", encoding="utf-8")
        image_path = package_dir / "extracted" / "page_0001.png"
        if include_image:
            Image.new("RGB", (80, 60), "white").save(image_path)
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "schema_version": "1.0",
                "source_id": source_id,
                "source": {
                    "filename": filename,
                    "sha256": checksum,
                    "imported_path": "source.pdf",
                    "original_path": str(imported_pdf),
                    "media_type": "pdf",
                    "class_number": class_number,
                    "pdf_page_count": 1,
                    "ocr_language": "eng",
                },
                "assets": [{
                    "kind": "pdf_page",
                    "page": 1,
                    "text_path": "extracted/page_0001.txt",
                    "image_path": "extracted/page_0001.png",
                }],
                "processing": {
                    "status": "complete",
                    "pipeline_version": pipeline_version,
                    "result": {
                        "status": "complete",
                        "source_id": source_id,
                        "pdf_pages": 1,
                        "ocr_language": "eng",
                        "pages_needing_ocr_review": 0,
                    },
                },
            }),
            encoding="utf-8",
        )
        (package_dir / "printed_page_index.json").write_text(
            json.dumps({
                "schema_version": 1,
                "source_id": source_id,
                "source_sha256": checksum,
                "pages": [{
                    "pdf_page": 1,
                    "printed_page": 1,
                    "confidence": 92.0,
                    "status": "matched",
                }],
            }),
            encoding="utf-8",
        )
        return source_id, manifest_path

    def test_duplicate_manual_book_registration_is_ignored(self) -> None:
        duplicate_root = self.data_root / "verification" / "sources" / "english"
        duplicate_root.mkdir(parents=True)
        (duplicate_root / "images").mkdir(exist_ok=True)
        (duplicate_root / "chapters").mkdir(exist_ok=True)
        shutil.copy2(self.prepared_book_dir / "book.json", duplicate_root / "book.json")
        shutil.copy2(self.prepared_book_dir / "chapters" / "unit_01.json", duplicate_root / "chapters" / "unit_01.json")
        shutil.copy2(self.prepared_book_dir / "images" / "bird.png", duplicate_root / "images" / "bird.png")
        catalog = ContentCatalog(self.data_root, self.summary_path)
        self.assertTrue(any(book.title == "English (Prepared)" for book in catalog.books.values()))
        self.assertEqual(len(catalog.prepared_chapters), 1)

    def _write_page_json_book(self) -> Path:
        book_root = self.data_root / "prepared_books" / "class_05" / "Maths"
        (book_root / "images").mkdir(parents=True)
        Image.new("RGB", (80, 60), "white").save(book_root / "images" / "page_001.png")

        def page_record(printed_page: int, pdf_page: int, image: str | None) -> dict[str, Any]:
            return {
                "schema_version": "2.0",
                "source_provenance": {
                    "printed_page_number": printed_page,
                    "source_pdf_page_number_1_indexed": pdf_page,
                },
                "page_identity": {
                    "page_heading": f"Printed page {printed_page}: Bigger numbers",
                },
                "source_text": {
                    "extracted_text_normalized_for_search": f"Page {printed_page} teaches place value.",
                    "extracted_text_raw_ordered": f"Page {printed_page} teaches place value.",
                },
                "page_explanation": {
                    "source_grounded_summary": f"AI page description for printed page {printed_page}.",
                    "concepts_and_teaching_notes": ["Compare the place values from right to left."],
                },
                "questions_and_activities": {
                    "source_extracted_questions_and_activities": [
                        f"Which digit is in the tens place on page {printed_page}?"
                    ],
                    "detected_question_or_instruction_lines": [],
                },
                "tables_and_structured_data": {"extracted_table_text": []},
                "visuals": {
                    "visual_source_image": image,
                    "visual_items": [{"description": "A place-value chart."}],
                    "visual_description_status": "AI description; verify against the page image.",
                },
                "quality_control": {"image_review_status": "not human reviewed"},
            }

        chapter_path = book_root / "chapter_01_bigger_numbers.json"
        chapter_path.write_text(
            json.dumps({
                "schema_version": "2.0",
                "source_book": {
                    "title": "Mathematics",
                    "class": 5,
                    "board": "Telangana",
                    "language": "English",
                    "source_file": "class5-maths.pdf",
                },
                "chapter": {
                    "number": 1,
                    "title": "Bigger numbers",
                    "printed_page_range": {"start": 1, "end": 2},
                    "page_count": 2,
                },
                "chapter_usage_note": "Use extracted text and supplied descriptions as evidence.",
                "pages": [
                    page_record(1, 10, "images/page_001.png"),
                    page_record(2, 11, "images/page_002.png"),
                ],
            }),
            encoding="utf-8",
        )
        return chapter_path

    def test_page_json_schema_feeds_page_text_description_image_and_citation(self) -> None:
        self._write_page_json_book()
        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        self.assertEqual(book.class_number, 5)
        self.assertTrue(catalog.catalog_response()["classes"][4]["books"][0]["page_lookup_available"])
        self.assertEqual(catalog.resolve_printed_page(book, 1), 10)

        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The page teaches place value."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "page": 1,
            "question": "Explain this page",
        })

        self.assertEqual(response["sources"][0]["page"], 1)
        self.assertEqual(response["sources"][0]["pdf_page"], 10)
        self.assertEqual(response["sources"][0]["page_number_type"], "printed")
        self.assertFalse(response["visual_context_used"])
        self.assertEqual(response["provider"], "groq")
        self.assertIsInstance(captured["content"], str)
        user_text = captured["content"]
        self.assertIn("Page 1 teaches place value.", user_text)
        self.assertIn("AI page description for printed page 1.", user_text)
        self.assertIn("A place-value chart.", user_text)
        self.assertIn("not human reviewed", user_text)

    def test_page_json_unit_named_files_appear_in_catalog(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
        chapter_data["source_book"]["title"] = "My English World"
        chapter_data["source_book"]["source_file"] = "class5-english.pdf"
        english_root = self.data_root / "prepared_books" / "class_05" / "English"
        english_root.mkdir(parents=True)
        chapter_path.unlink()
        (english_root / "unit_01_lets_be_friends.json").write_text(
            json.dumps(chapter_data),
            encoding="utf-8",
        )

        catalog = ContentCatalog(self.data_root, self.summary_path)
        english_book = next(
            (
                book
                for book in catalog.books.values()
                if book.class_number == 5 and book.title == "My English World (English)"
            ),
            None,
        )

        self.assertIsNotNone(english_book)
        self.assertTrue(english_book.page_json)
        self.assertEqual(
            catalog.resolve_printed_page(english_book, 1),
            10,
        )
        class_books = catalog.catalog_response()["classes"][4]["books"]
        self.assertTrue(any(book["title"] == "My English World (English)" for book in class_books))

    def test_chapter_search_index_routes_to_page_records_and_ignores_legacy_packages(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
        english_root = self.data_root / "prepared_books" / "class_05" / "English"
        english_root.mkdir(parents=True)
        page_texts = [
            "What is the place you see in the picture? The page shows a home courtyard.",
            "Once upon a time, there was a small boy called Hari. He teased his classmates.",
        ]
        page_names = [
            "011_printed001_pdf011_preparatory_page.json",
            "012_printed002_pdf012_lets_be_friends.json",
        ]
        pages = chapter_data["pages"]
        page_numbers = [(1, 11), (2, 12)]
        for page, page_text, page_name, (printed_page, pdf_page) in zip(
            pages,
            page_texts,
            page_names,
            page_numbers,
        ):
            page["source_provenance"]["source_pdf_filename"] = "class5-english.pdf"
            page["source_provenance"]["language"] = "English"
            page["source_provenance"]["board"] = "Telangana"
            page["source_provenance"]["printed_page_number"] = printed_page
            page["source_provenance"]["source_pdf_page_number_1_indexed"] = pdf_page
            page["source_text"]["extracted_text_normalized_for_search"] = page_text
            page["source_text"]["extracted_text_raw_ordered"] = page_text
            page["visuals"]["visual_source_image"] = None
            page["visuals"]["visual_items"] = []
            (english_root / page_name).write_text(json.dumps(page), encoding="utf-8")

        legacy_chapter = dict(chapter_data)
        legacy_chapter["source_book"] = dict(chapter_data["source_book"])
        legacy_chapter["source_book"]["title"] = "Legacy duplicate that must not load"
        legacy_chapter["pages"] = [dict(pages[0])]
        legacy_chapter["pages"][0]["source_text"] = {
            "extracted_text_normalized_for_search": "LEGACY PACKAGE TEXT"
        }
        (english_root / "chapter_01_legacy.json").write_text(
            json.dumps(legacy_chapter),
            encoding="utf-8",
        )

        index_path = english_root / "Class5_English_AI_PAGESEARCH_Index.json"
        index_path.write_text(
            json.dumps({
                "schema_version": "1.0",
                "source_book": "My English World — Class V",
                "chapters": [
                    {
                        "chapter_id": "unit_1",
                        "title": "Unit 1 — Let's Be Friends",
                        "printed_pages": {"from": 1, "to": 2},
                        "pdf_pages": {"from": 11, "to": 12},
                        "what_this_section_contains": (
                            "Preparatory picture activities followed by the story of Hari "
                            "learning how teasing affects classmates."
                        ),
                        "search_for": ["Hari", "teasing", "friendship"],
                    },
                ],
            }),
            encoding="utf-8",
        )

        catalog = ContentCatalog(self.data_root, self.summary_path)
        english_book = next(
            book for book in catalog.books.values()
            if book.class_number == 5 and book.title == "My English World (English)"
        )
        chapter = catalog.prepared_chapters[english_book.source_id][1]["data"]
        self.assertEqual(
            len(chapter["sections"]),
            2,
        )
        self.assertIn("teasing affects classmates", chapter["summary"])
        self.assertEqual(chapter["search_terms"], ["Hari", "teasing", "friendship"])
        self.assertEqual(catalog.resolve_printed_page(english_book, 1), 11)

        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "Chapter 1 introduces the picture activity and the story of Hari."

        ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": english_book.source_id,
            "question": "Explain chapter 1",
        })
        prompt = captured["content"]
        self.assertIn(page_texts[0], prompt)
        self.assertIn(page_texts[1], prompt)
        self.assertIn("Index chapter overview", prompt)
        self.assertIn("teasing affects classmates", prompt)
        self.assertNotIn("LEGACY PACKAGE TEXT", prompt)
        self.assertFalse(any(
            book.title.startswith("Legacy duplicate")
            for book in catalog.books.values()
        ))

    def test_chapter_explanation_uses_index_summary_and_only_relevant_page_records(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
        english_root = self.data_root / "prepared_books" / "class_05" / "English"
        english_root.mkdir(parents=True)
        template = chapter_data["pages"][0]
        unique_page = 13
        for printed_page in range(1, 21):
            page = json.loads(json.dumps(template))
            pdf_page = printed_page + 10
            text = (
                f"PAGE {printed_page} CHAPTER DETAIL. "
                "Ordinary chapter content about reading and classroom activities."
            )
            if printed_page == 12:
                text = (
                    "Grammar: We are the articles. I come before singular nouns "
                    "beginning with consonant sounds. Use an before nouns beginning "
                    "with vowel sounds. Use the when a person, place, or thing has "
                    "already been mentioned, or when it is unique. Examples: a "
                    "monster crab; the crab; the sun. Pick more examples of a, an, "
                    "and the from the story."
                )
            if printed_page == unique_page:
                text += " The Crystal Comet scene is the story's turning point."
            page["source_provenance"].update({
                "source_pdf_filename": "class5-english.pdf",
                "language": "English",
                "board": "Telangana",
                "printed_page_number": printed_page,
                "source_pdf_page_number_1_indexed": pdf_page,
            })
            page["source_text"]["extracted_text_raw_ordered"] = text
            page["source_text"]["extracted_text_normalized_for_search"] = text
            page["visuals"]["visual_source_image"] = None
            page["visuals"]["visual_items"] = []
            filename = (
                f"{pdf_page:03d}_printed{printed_page:03d}_pdf{pdf_page:03d}_unit_1.json"
            )
            (english_root / filename).write_text(json.dumps(page), encoding="utf-8")

        (english_root / "Class5_English_AI_PAGESEARCH_Index.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "source_book": "My English World — Class V",
                "chapters": [{
                    "chapter_id": "unit_1",
                    "title": "Unit 1 — The Crystal Comet",
                    "printed_pages": {"from": 1, "to": 20},
                    "pdf_pages": {"from": 11, "to": 30},
                    "what_this_section_contains": (
                        "A story about a Crystal Comet and the choices made by its characters. "
                        "The grammar lesson introduces the articles a, an and the."
                    ),
                    "search_for": ["Crystal Comet", "turning point", "grammar", "articles"],
                }],
            }),
            encoding="utf-8",
        )

        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(
            book for book in catalog.books.values()
            if book.class_number == 5 and book.title == "My English World (English)"
        )
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The Crystal Comet scene changes the story."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "Explain chapter 1 focusing on the Crystal Comet scene",
        })
        prompt = captured["content"]
        self.assertNotIn("Index chapter overview", prompt)
        self.assertIn("Requested chapter focus: comet, crystal, scene", prompt)
        self.assertIn(f"PAGE {unique_page} CHAPTER DETAIL", prompt)
        self.assertLess(len(prompt), 7_000)
        self.assertEqual([source["page"] for source in response["sources"]], [unique_page])

        service = ChatService(catalog, api_key="test-key", completion=completion)
        for question in (
            "grammer chpater 1 explain",
            "explain chptr one grammar",
            "first grammar lesson explain",
            "articles in first unit explain",
        ):
            with self.subTest(question=question):
                response = service.respond({
                    "class": 5,
                    "source_id": book.source_id,
                    "question": question,
                })
                typo_prompt = captured["content"]
                self.assertIn("We are the articles", typo_prompt)
                self.assertIn("Use an before nouns beginning with vowel sounds", typo_prompt)
                self.assertIn("Printed page 12", typo_prompt)
                self.assertIn("Requested chapter focus:", typo_prompt)
                self.assertNotIn("Index chapter overview", typo_prompt)
                self.assertNotIn("Preparatory picture activities", typo_prompt)
                self.assertIn("Do not give a whole-unit overview", typo_prompt)
                self.assertNotIn("PAGE 13 CHAPTER DETAIL", typo_prompt)
                self.assertEqual(
                    [source["page"] for source in response["sources"]],
                    [12],
                )

    def test_visual_question_retrieves_page_from_page_json_visual_description(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter_data = json.loads(chapter_path.read_text(encoding="utf-8"))
        chapter_data["pages"][0]["source_text"]["extracted_text_normalized_for_search"] = (
            "Look at the picture and answer the question."
        )
        chapter_data["pages"][0]["source_text"]["extracted_text_raw_ordered"] = (
            "Look at the picture and answer the question."
        )
        chapter_data["pages"][0]["visuals"]["visual_items"] = [{
            "description_detailed": "A red kite is caught in a tree above the courtyard.",
            "image_grounded_answers": {"what_happened": "The kite is caught in a tree."},
        }]
        chapter_data["pages"][1]["source_text"]["extracted_text_normalized_for_search"] = (
            "A page about mountain habitats and the animals that live there."
        )
        chapter_data["pages"][1]["source_text"]["extracted_text_raw_ordered"] = (
            "A page about mountain habitats and the animals that live there."
        )
        chapter_path.write_text(json.dumps(chapter_data), encoding="utf-8")

        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The red kite is caught in a tree."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "What happened to the red kite in the illustration?",
        })
        message_content = captured["content"]
        prompt = (
            "\n".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict)
            )
            if isinstance(message_content, list)
            else message_content
        )
        self.assertIn("A red kite is caught in a tree above the courtyard.", prompt)
        self.assertEqual([source["page"] for source in response["sources"]], [1])

    def test_page_json_sidecar_visual_items_are_used_for_image_questions(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        chapter["pages"][0]["visuals"]["visual_source_image"] = "images/missing_page_001.png"
        chapter["pages"][0]["source_text"]["extracted_text_normalized_for_search"] += (
            "\n1. What is the place you see in the picture?"
        )
        chapter["pages"][0]["source_text"]["extracted_text_raw_ordered"] += (
            "\n1. What is the place you see in the picture?"
        )
        chapter_path.write_text(json.dumps(chapter), encoding="utf-8")

        visual_item = {
            "visual_id": "page_001_main_illustration",
            "kind": "illustration",
            "title": "Village courtyard",
            "description_detailed": "A home courtyard with women and children doing household work.",
            "image_grounded_answers": {
                "place": "A home courtyard, likely in a village or residential setting.",
                "things": "A clothesline, bucket, and household vessels.",
                "people": "Several women and children.",
                "actions": "Some people are washing or handling household vessels.",
            },
        }
        sidecar = json.loads(json.dumps(chapter["pages"][0]))
        sidecar["source_provenance"]["source_pdf_filename"] = "class5-maths.pdf"
        sidecar["page_explanation"]["source_grounded_summary"] = (
            "A detailed description of the illustrated home courtyard, clothesline, "
            "women, children, bucket, and household work."
        )
        sidecar["visuals"]["visual_items"] = [visual_item]
        sidecar["visuals"]["visual_source_image"] = "images/missing_page_001.png"
        sidecar["visuals"]["visual_description_status"] = "Detailed image description supplied."
        sidecar_path = chapter_path.parent / "010_printed001_pdf010_page_description.json"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["messages"] = messages
            return "The picture shows a home courtyard."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "What is the place you see in the picture?",
        })

        user_content = captured["messages"][-1]["content"]
        prompt = (
            user_content[0]["text"]
            if isinstance(user_content, list)
            else user_content
        )
        self.assertIn("A home courtyard, likely in a village or residential setting.", prompt)
        self.assertIn("description rather than direct image inspection", prompt)
        self.assertNotIn("cannot tell you exactly what place", prompt)
        self.assertFalse(response["visual_context_used"])
        self.assertTrue(any("No image is available" in warning for warning in response["warnings"]))
        self.assertEqual(response["sources"][0]["page"], 1)

    def test_page_json_page_no_phrase_resolves_requested_printed_page(self) -> None:
        self._write_page_json_book()
        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "Page 1 contains the Bigger numbers cricket table."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "explain page no 1 table",
        })

        self.assertEqual(response["sources"][0]["page"], 1)
        self.assertEqual(response["sources"][0]["pdf_page"], 10)
        prompt = captured["content"]
        if isinstance(prompt, list):
            prompt = prompt[0]["text"]
        self.assertIn("Page 1 teaches place value.", prompt)
        self.assertNotIn("Page 2 teaches place value.", prompt)
        self.assertIn("PDF text extraction may merge cells", prompt)

    def test_page_json_page_number_and_page_no_period_are_recognized(self) -> None:
        self._write_page_json_book()
        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)

        for question in ("Explain page number 2", "Explain page no. 2"):
            with self.subTest(question=question):
                response = ChatService(
                    catalog,
                    api_key="test-key",
                    completion=lambda *_: "Page 2 explanation.",
                ).respond({
                    "class": 5,
                    "source_id": book.source_id,
                    "question": question,
                })
                self.assertEqual(response["sources"][0]["page"], 2)
                self.assertEqual(response["sources"][0]["pdf_page"], 11)

    def test_page_json_missing_image_keeps_text_available_and_reports_uncertainty(self) -> None:
        self._write_page_json_book()
        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The text discusses place value."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "page": 2,
            "question": "What pictures are on page 2?",
        })

        self.assertEqual(response["sources"][0]["page"], 2)
        self.assertEqual(response["sources"][0]["pdf_page"], 11)
        self.assertFalse(response["visual_context_used"])
        self.assertTrue(any("No image is available" in warning for warning in response["warnings"]))
        self.assertIn("AI page description for printed page 2.", captured["content"])
        self.assertIn("supplied page descriptions as supplementary evidence", captured["content"])

    def test_page_json_question_sends_only_the_most_relevant_page(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        relevant_page_text = (
            "This unrelated introduction is not needed to answer the question.\n"
            "The teacher wrote a table on the board.\n"
            "Name of the bowler Country Wickets taken\n"
            "Muralidharan Sri Lanka 1347\n"
            "Shane Warne Australia 1001\n"
            "Anil Kumble India 956\n"
            "Kapil Dev India 687\n"
            "Walsh West Indies 519\n"
            "Who has taken the highest number of wickets?\n"
            "This unrelated closing paragraph should also be omitted."
        )
        chapter["pages"][0]["source_text"]["extracted_text_normalized_for_search"] = (
            relevant_page_text
        )
        chapter["pages"][0]["source_text"]["extracted_text_raw_ordered"] = (
            relevant_page_text
        )
        chapter["pages"][1]["source_text"]["extracted_text_normalized_for_search"] = (
            "Irrelevant place-value material. " * 300
        )
        chapter["pages"][1]["source_text"]["extracted_text_raw_ordered"] = (
            "Irrelevant place-value material. " * 300
        )
        chapter_path.write_text(json.dumps(chapter), encoding="utf-8")

        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "Muralidharan took the most wickets."

        service = ChatService(catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "Who has taken the highest number of wickets?",
        })

        prompt = captured["content"]
        self.assertIn("Muralidharan", prompt)
        self.assertIn("1347", prompt)
        self.assertIn("Shane Warne Australia 1001", prompt)
        self.assertNotIn("How many more wickets", prompt)
        self.assertNotIn("unrelated introduction", prompt)
        self.assertNotIn("unrelated closing paragraph", prompt)
        self.assertNotIn("Irrelevant place-value material.", prompt)
        self.assertLess(len(prompt), 2_000)
        self.assertEqual(
            [(source["page"], source["pdf_page"]) for source in response["sources"]],
            [(1, 10)],
        )

        variant_response = service.respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "who has taken highest wickets answers",
        })
        variant_prompt = captured["content"]
        self.assertIn("Muralidharan", variant_prompt)
        self.assertIn("1347", variant_prompt)
        self.assertNotIn("Irrelevant place-value material.", variant_prompt)
        self.assertLess(len(variant_prompt), 2_000)
        self.assertEqual(
            [source["page"] for source in variant_response["sources"]],
            [1],
        )

    def test_page_json_question_bounds_conversation_history(self) -> None:
        self._write_page_json_book()
        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["messages"] = messages
            return "The page teaches place value."

        history = [
            {"role": "user", "content": "Earlier context. " * 100},
            {"role": "assistant", "content": "Earlier answer. " * 100},
            {"role": "user", "content": "Recent context. " * 100},
            {"role": "assistant", "content": "Recent answer. " * 100},
            {"role": "user", "content": "Older extra message. " * 100},
            {"role": "assistant", "content": "Old answer. " * 100},
        ]
        ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "Explain the page",
            "history": history,
        })

        history_messages = captured["messages"][1:-1]
        self.assertLessEqual(len(history_messages), 4)
        self.assertLessEqual(
            sum(len(message["content"]) for message in history_messages),
            800,
        )

    def test_page_json_chapter_request_uses_bounded_page_excerpts(self) -> None:
        chapter_path = self._write_page_json_book()
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        for page in chapter["pages"]:
            page_number = page["source_provenance"]["printed_page_number"]
            page_text = (
                f"Opening idea for page {page_number}.\n"
                + "MIDDLE-ONLY-FILLER. " * 500
                + f"\nClosing idea for page {page_number}."
            )
            page["source_text"]["extracted_text_normalized_for_search"] = page_text
            page["source_text"]["extracted_text_raw_ordered"] = page_text
        chapter_path.write_text(json.dumps(chapter), encoding="utf-8")

        catalog = ContentCatalog(self.data_root, self.summary_path)
        book = next(book for book in catalog.books.values() if book.page_json)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["messages"] = messages
            return "This chapter introduces place value."

        response = ChatService(catalog, api_key="test-key", completion=completion).respond({
            "class": 5,
            "source_id": book.source_id,
            "question": "Explain chapter 1",
        })

        prompt = captured["messages"][-1]["content"]
        self.assertLess(len(prompt), 5_000)
        self.assertIn("Opening idea for page 1.", prompt)
        self.assertIn("Closing idea for page 1.", prompt)
        self.assertIn("Opening idea for page 2.", prompt)
        self.assertIn("Closing idea for page 2.", prompt)
        self.assertIn("Some page text may be omitted", prompt)
        self.assertNotIn("MIDDLE-ONLY-FILLER. " * 100, prompt)
        self.assertFalse(response["visual_context_used"])
        self.assertEqual(
            [source["page"] for source in response["sources"]],
            [1, 2],
        )

    def test_imported_chapter_request_bounds_evidence_and_cites_included_pages(self) -> None:
        book = self.catalog.books["0123456789abcdef"]
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        package_dir = book.manifest_path.parent
        first_asset, second_asset = manifest["assets"]
        (package_dir / first_asset["text_path"]).write_text(
            "Chapter 1\nPLANTS\n" + ("Plants grow from seeds. " * 4_000),
            encoding="utf-8",
        )
        (package_dir / second_asset["text_path"]).write_text(
            "Chapter 2\nANIMALS\nAnimals need food.",
            encoding="utf-8",
        )
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "Plants grow from seeds."

        response = ChatService(
            self.catalog,
            api_key="test-key",
            completion=completion,
        ).respond({
            "class": 1,
            "source_id": book.source_id,
            "question": "Explain chapter 1",
        })

        self.assertLess(len(captured["content"]), 15_000)
        self.assertIn("bounded excerpt", captured["content"])
        self.assertIn("under 250 words", captured["content"])
        self.assertEqual(
            [source["pdf_page"] for source in response["sources"]],
            [1],
        )

    def test_catalog_filters_by_class_and_retrieves_relevant_book_page(self) -> None:
        results = self.catalog.retrieve(1, "tree leaves")
        self.assertEqual(results[0][0].source_id, "0123456789abcdef")
        self.assertEqual(results[0][0].page, 1)
        self.assertTrue(self.catalog.catalog_response()["classes"][0]["available"])
        self.assertFalse(self.catalog.catalog_response()["classes"][2]["available"])

    def test_persistent_page_index_is_reused_after_catalog_restart(self) -> None:
        self.catalog.retrieve(1, "tree leaves")
        index_path = self.data_root / "search" / "page_index.sqlite3"
        self.assertTrue(index_path.is_file())
        restarted_catalog = ContentCatalog(self.data_root, self.summary_path)
        package_text = (
            self.data_root
            / "books"
            / "0123456789abcdef"
            / "text"
            / "page_0001.txt"
        )
        original_read_text = Path.read_text

        def reject_cached_page_file(path: Path, *args: Any, **kwargs: Any) -> str:
            if path.resolve() == package_text.resolve():
                raise AssertionError("A current persistent index should avoid rereading page text.")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", reject_cached_page_file):
            results = restarted_catalog.retrieve(1, "tree leaves")
        self.assertEqual(results[0][0].page, 1)

    def test_persistent_page_index_refreshes_changed_page_text(self) -> None:
        self.catalog.retrieve(1, "tree leaves")
        package_text = (
            self.data_root
            / "books"
            / "0123456789abcdef"
            / "text"
            / "page_0001.txt"
        )
        package_text.write_text(
            "A comet travels through the night sky.",
            encoding="utf-8",
        )
        restarted_catalog = ContentCatalog(self.data_root, self.summary_path)

        results = restarted_catalog.retrieve(1, "comet")

        self.assertEqual(results[0][0].page, 1)
        self.assertIn("comet", results[0][0].text.casefold())

    def test_lazy_page_index_preserves_page_metadata_and_printed_numbers(self) -> None:
        manifest_path = (
            self.data_root
            / "books"
            / "0123456789abcdef"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["assets"][0]["text_extraction_method"] = "tesseract_ocr"
        manifest["assets"][0]["needs_ocr_review"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        page_text_path = manifest_path.parent / "text" / "page_0001.txt"
        page_text_path.write_text(
            "LESSON 1: Seeds\nSeeds grow into plants.",
            encoding="utf-8",
        )

        document = next(
            item
            for item in self.catalog.documents_for_class(1)
            if item.source_id == "0123456789abcdef" and item.page == 1
        )

        self.assertEqual(document.printed_page, 1)
        self.assertEqual(document.heading, "LESSON 1: Seeds")
        self.assertTrue(document.chapter.startswith("lesson_01:"))
        self.assertIn("seeds", document.keywords)
        self.assertEqual(document.ocr_text, document.text)
        self.assertEqual(document.text_extraction_method, "tesseract_ocr")
        self.assertTrue(document.needs_ocr_review)
        results = self.catalog.retrieve(1, "lesson seeds")
        self.assertEqual(results[0][0].page, 1)

    def test_catalog_lists_unprocessed_corpus_books_and_class_nine(self) -> None:
        source_root = self.root / "Telangana_SCERT_Textbooks"
        filenames = (
            "Class_01_English_Medium_English_SSC_Common.pdf",
            "Class_02_English_Medium_Science_SSC_Common.pdf",
            "Class_09_Telugu_Medium_English_SSC_Common.pdf",
        )
        for filename in filenames:
            path = source_root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )
        response = catalog.catalog_response()
        class_one = response["classes"][0]
        class_two = response["classes"][1]
        class_nine = response["classes"][8]

        self.assertEqual(class_one["corpus_books"][0]["title"], "English (English Medium · SSC Common)")
        self.assertFalse(class_one["corpus_books"][0]["processed"])
        self.assertEqual(class_two["corpus_books"][0]["title"], "Science (English Medium · SSC Common)")
        self.assertFalse(class_two["corpus_books"][0]["processed"])
        self.assertTrue(class_two["available"])
        self.assertFalse(class_nine["available"])
        self.assertFalse(class_nine["chat_supported"])
        self.assertEqual(class_nine["corpus_books"][0]["title"], "English (Telugu Medium · SSC Common)")

    def test_catalog_does_not_mark_legacy_pipeline_pdf_ready_from_summary(self) -> None:
        source_root = self.root / "Telangana_SCERT_Textbooks"
        filename = "Class_02_English_Medium_Science_SSC_Common.pdf"
        source_path = (
            source_root
            / "Class_02"
            / "English_Medium"
            / filename
        )
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"selected source PDF")
        source_id, manifest_path = self._write_current_source_package(
            self.data_root,
            filename,
            2,
            source_path.read_bytes(),
            pipeline_version="1.3",
        )
        self.summary_path.write_text(
            json.dumps({
                "books": [{
                    "class": 2,
                    "source_id": source_id,
                    "filename": filename,
                    "manifest": str(manifest_path),
                }],
            }),
            encoding="utf-8",
        )

        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )

        self.assertNotIn(source_id, catalog.books)
        self.assertFalse(catalog.discovered_sources_by_class[2][0].processed)

    def test_first_message_processes_selected_unprocessed_textbook(self) -> None:
        source_root = self.root / "Telangana_SCERT_Textbooks"
        source_path = (
            source_root
            / "Class_02"
            / "English_Medium"
            / "Class_02_English_Medium_Science_SSC_Common.pdf"
        )
        source_path.parent.mkdir(parents=True)
        source_path.touch()
        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )
        selected_source = catalog.discovered_sources_by_class[2][0]
        processed_source_id = ""

        def process_selected_pdf(path: Path, data_root: Path) -> dict[str, Any]:
            nonlocal processed_source_id
            processed_source_id, _manifest_path = self._write_current_source_package(
                data_root,
                path.name,
                2,
                path.read_bytes(),
            )
            return {"status": "complete", "source_id": processed_source_id}

        completion = lambda *_: "This book is ready to study."
        service = ChatService(catalog, api_key="test-key", completion=completion)
        with patch(
            "textbook_engine.pipeline.process_pdf",
            side_effect=process_selected_pdf,
        ) as process_pdf_mock:
            response = service.respond({
                "class": 2,
                "source_id": selected_source.source_key,
                "question": "Explain this book",
            })

        process_pdf_mock.assert_called_once_with(source_path.resolve(), self.data_root.resolve())
        self.assertIn(processed_source_id, service.catalog.books)
        self.assertEqual(response["answer"], "This book is ready to study.")

    def test_process_selected_textbook_returns_processing_report(self) -> None:
        source_root = self.root / "Telangana_SCERT_Textbooks"
        source_path = (
            source_root
            / "Class_02"
            / "English_Medium"
            / "Class_02_English_Medium_Science_SSC_Common.pdf"
        )
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"selected source PDF")
        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )
        selected_source = catalog.discovered_sources_by_class[2][0]
        processed_source_id = ""

        def process_selected_pdf(path: Path, data_root: Path) -> dict[str, Any]:
            nonlocal processed_source_id
            processed_source_id, manifest_path = self._write_current_source_package(
                data_root,
                path.name,
                2,
                path.read_bytes(),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["processing"]["result"]["warnings"] = [
                "Printed-page indexing was unavailable."
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return {
                "status": "complete",
                "source_id": processed_source_id,
                "warnings": ["Printed-page indexing was unavailable."],
            }

        service = ChatService(catalog, api_key="test-key")
        with patch(
            "textbook_engine.pipeline.process_pdf",
            side_effect=process_selected_pdf,
        ) as process_pdf_mock:
            server = LocalChatServer(("127.0.0.1", 0), catalog, service)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/process",
                    data=json.dumps({
                        "class": 2,
                        "source_id": selected_source.source_key,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as result:
                    response = json.loads(result.read().decode("utf-8"))
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/catalog",
                    timeout=5,
                ) as result:
                    refreshed_catalog = json.loads(result.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        process_pdf_mock.assert_called_once_with(
            source_path.resolve(),
            self.data_root.resolve(),
        )
        self.assertEqual(response["source_id"], processed_source_id)
        self.assertEqual(
            response["processing"]["warnings"],
            ["Printed-page indexing was unavailable."],
        )
        class_two = next(
            item for item in refreshed_catalog["classes"] if item["class"] == 2
        )
        self.assertTrue(
            any(book["source_id"] == processed_source_id for book in class_two["books"])
        )
        self.assertTrue(
            next(
                item
                for item in class_two["corpus_books"]
                if item["filename"] == selected_source.filename
            )["processed"]
        )

    def test_process_selected_textbook_rejects_unknown_source(self) -> None:
        service = ChatService(self.catalog, api_key="test-key")
        with self.assertRaisesRegex(ChatError, "not available in the source catalog"):
            service.process_discovered_source(2, "unknown-source")

    def test_processing_registers_valid_package_when_refreshed_catalog_misses_it(self) -> None:
        source_root = self.root / "Telangana_SCERT_Textbooks"
        source_path = (
            source_root
            / "Class_02"
            / "English_Medium"
            / "Class_02_English_Medium_Science_SSC_Common.pdf"
        )
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"selected source PDF")
        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )
        selected_source = catalog.discovered_sources_by_class[2][0]
        stale_catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )
        source_id = ""

        def process_selected_pdf(path: Path, data_root: Path) -> dict[str, Any]:
            nonlocal source_id
            source_id, _manifest_path = self._write_current_source_package(
                data_root,
                path.name,
                2,
                path.read_bytes(),
            )
            return {"status": "complete", "source_id": source_id}

        service = ChatService(catalog, api_key="test-key")
        with (
            patch(
                "textbook_engine.pipeline.process_pdf",
                side_effect=process_selected_pdf,
            ),
            patch(
                "textbook_engine.chat.ContentCatalog",
                return_value=stale_catalog,
            ),
        ):
            result = service.process_discovered_source(
                2,
                selected_source.source_key,
            )

        self.assertEqual(result["source_id"], source_id)
        self.assertEqual(service.catalog.books[source_id].class_number, 2)
        self.assertTrue(
            next(
                source
                for source in service.catalog.discovered_sources_by_class[2]
                if source.source_key == selected_source.source_key
            ).processed
        )

    def test_explain_chapter_uses_selected_books_complete_chapter_text(self) -> None:
        source_id = "0123456789abcdef"
        book = self.catalog.books[source_id]
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        package_dir = book.manifest_path.parent
        (package_dir / manifest["assets"][0]["text_path"]).write_text(
            "UNIT - 1\nPLANTS\nPlants grow from seeds.",
            encoding="utf-8",
        )
        (package_dir / manifest["assets"][1]["text_path"]).write_text(
            "Plants need water and sunlight.",
            encoding="utf-8",
        )
        third_page_text = "text/page_0003.txt"
        third_page_image = "pages/page_0003.png"
        (package_dir / third_page_text).write_text(
            "UNIT - 2\nANIMALS\nAnimals need food.",
            encoding="utf-8",
        )
        Image.new("RGB", (80, 60), "white").save(package_dir / third_page_image)
        manifest["assets"].append({
            "kind": "pdf_page",
            "page": 3,
            "text_path": third_page_text,
            "image_path": third_page_image,
        })
        book.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (package_dir / "printed_page_index.json").unlink()
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], model: str, api_key: str) -> str:
            captured["messages"] = messages
            return "Plants grow from seeds and need water and sunlight."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "source_id": source_id,
            "question": "hii can u explain 1 chapter?",
        })

        prompt = captured["messages"][-1]["content"]
        self.assertIn("chapter 1", prompt.casefold())
        self.assertIn("PLANTS", prompt)
        self.assertIn("Plants grow from seeds.", prompt)
        self.assertIn("Plants need water and sunlight.", prompt)
        self.assertIn("Selected textbook: English", prompt)
        self.assertIn("do not ask which subject or textbook", prompt.casefold())
        self.assertIn("Do not ask the student to provide the book name", prompt)
        self.assertEqual(response["answer"], "Plants grow from seeds and need water and sunlight.")
        self.assertEqual([source["pdf_page"] for source in response["sources"]], [1, 2])

    def test_unrecognized_chapter_layout_uses_and_caches_gemini_index(self) -> None:
        source_id = "0123456789abcdef"
        book = self.catalog.books[source_id]
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        package_dir = book.manifest_path.parent
        for asset, text in zip(
            manifest["assets"],
            ["First lesson text.", "Second lesson text."],
            strict=True,
        ):
            (package_dir / asset["text_path"]).write_text(text, encoding="utf-8")
        book.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        chapter_index = json.dumps({
            "chapters": [
                {
                    "number": 1,
                    "title": "First lesson",
                    "start_pdf_page": 1,
                    "marker_kind": "lesson",
                },
                {
                    "number": 2,
                    "title": "Second lesson",
                    "start_pdf_page": 2,
                    "marker_kind": "lesson",
                },
            ],
        })

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            return "A grounded chapter explanation."

        service = ChatService(
            self.catalog,
            api_key="groq-test-key",
            gemini_api_key="gemini-test-key",
            completion=completion,
        )
        with patch(
            "textbook_engine.gemini.complete_chapter_index",
            return_value=chapter_index,
        ) as index_completion:
            first = service.respond({
                "class": 1,
                "source_id": source_id,
                "question": "Explain chapter 1",
            })
            second = service.respond({
                "class": 1,
                "source_id": source_id,
                "question": "Explain chapter 2",
            })

        self.assertEqual(
            [source["pdf_page"] for source in first["sources"]],
            [1],
        )
        self.assertEqual(
            [source["pdf_page"] for source in second["sources"]],
            [2],
        )
        index_completion.assert_called_once()

    def test_prepared_chapter_sends_question_linked_image_to_vision_model(self) -> None:
        prepared_book = next(book for book in self.catalog.books.values() if book.prepared)
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "A bird uses its wings to fly."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "source_id": prepared_book.source_id,
            "question": "What can you see in the bird picture in chapter 1?",
        })

        user_content = captured["content"]
        self.assertIsInstance(user_content, list)
        self.assertIn("Birds use their wings to fly.", user_content[0]["text"])
        self.assertIn("A small blue bird with spread wings.", user_content[0]["text"])
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(
            user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual([image["image_id"] for image in response["images"]], ["bird_img001"])
        self.assertEqual(response["sources"][0]["pdf_page"], 11)
        self.assertTrue(response["visual_context_used"])

    def test_prepared_chapter_selects_the_image_linked_to_a_question(self) -> None:
        prepared_book = next(book for book in self.catalog.books.values() if book.prepared)
        service = ChatService(
            self.catalog,
            api_key="test-key",
            completion=lambda *_: "It uses its wings.",
        )

        response = service.respond({
            "class": 1,
            "source_id": prepared_book.source_id,
            "question": "What does a bird use to fly?",
        })

        self.assertEqual(response["images"][0]["image_id"], "bird_img001")
        self.assertEqual([source["pdf_page"] for source in response["sources"]], [11])
        self.assertTrue(
            self.catalog.prepared_image_path(prepared_book.source_id, "bird_img001").is_file()
        )
        prepared_entry = next(
            item
            for group in self.catalog.catalog_response()["classes"]
            for item in group["books"]
            if item["source_id"] == prepared_book.source_id
        )
        self.assertTrue(prepared_entry["prepared"])

    def test_prepared_chapter_rejects_image_path_traversal(self) -> None:
        chapter_path = self.prepared_book_dir / "chapters" / "unit_01.json"
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        chapter["images"][0]["file"] = "../outside.png"
        chapter_path.write_text(json.dumps(chapter), encoding="utf-8")

        with self.assertRaisesRegex(ChatError, "outside the local data directory"):
            ContentCatalog(self.data_root, self.summary_path)

    def test_catalog_discovers_completed_engine_processed_uploads(self) -> None:
        filename = "Class_02_English_Medium_Mathematics_SSC_Common.pdf"
        source_id, manifest_path = self._write_current_source_package(
            self.data_root,
            filename,
            2,
            b"matching pdf bytes",
        )

        catalog = ContentCatalog(self.data_root, self.summary_path)

        self.assertIn(source_id, catalog.books)
        self.assertEqual(catalog.books[source_id].class_number, 2)
        self.assertEqual(
            catalog.books[source_id].title,
            "Mathematics (English Medium · SSC Common)",
        )
        results = catalog.retrieve(2, "seed plant", source_id=source_id)
        self.assertEqual(results[0][0].source_id, source_id)
        self.assertTrue(
            any(
                book["source_id"] == source_id
                for book in catalog.catalog_response()["classes"][1]["books"]
            )
        )
        book = catalog.books[source_id]

        def create_printed_page_index(_index_manifest_path: Path) -> None:
            book.printed_pages_path.write_text(
                json.dumps({
                    "source_id": source_id,
                    "source_sha256": book.source_sha256,
                    "pages": [{
                        "printed_page": 1,
                        "pdf_page": 1,
                        "status": "matched",
                    }],
                }),
                encoding="utf-8",
            )

        with patch(
            "textbook_engine.printed_pages.index_printed_pages",
            side_effect=create_printed_page_index,
        ) as indexer:
            self.assertEqual(catalog.resolve_printed_page(book, 1), 1)

        indexer.assert_not_called()

    def test_catalog_rejects_old_pipeline_manifest_as_not_ready(self) -> None:
        filename = "Class_02_English_Medium_Mathematics_SSC_Common.pdf"
        source_bytes = b"old version source"
        source_id, manifest_path = self._write_current_source_package(
            self.data_root,
            filename,
            2,
            source_bytes,
            pipeline_version="1.1",
        )
        source_root = self.root / "Telangana_SCERT_Textbooks"
        (source_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (source_root / filename).write_bytes(source_bytes)

        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )

        self.assertTrue(manifest_path.is_file())
        self.assertNotIn(source_id, catalog.books)
        self.assertFalse(catalog.discovered_sources_by_class[2][0].processed)

    def test_catalog_rejects_current_version_manifest_with_missing_assets(self) -> None:
        filename = "Class_02_English_Medium_Mathematics_SSC_Common.pdf"
        source_bytes = b"missing image source"
        source_id, _manifest_path = self._write_current_source_package(
            self.data_root,
            filename,
            2,
            source_bytes,
            include_image=False,
        )
        source_root = self.root / "Telangana_SCERT_Textbooks"
        (source_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (source_root / filename).write_bytes(source_bytes)

        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )

        self.assertNotIn(source_id, catalog.books)
        self.assertFalse(catalog.discovered_sources_by_class[2][0].processed)

    def test_catalog_detects_same_filename_with_changed_pdf_content(self) -> None:
        filename = "Class_02_English_Medium_Mathematics_SSC_Common.pdf"
        old_source_id, _manifest_path = self._write_current_source_package(
            self.data_root,
            filename,
            2,
            b"previous content",
        )
        source_root = self.root / "Telangana_SCERT_Textbooks"
        (source_root / filename).parent.mkdir(parents=True, exist_ok=True)
        (source_root / filename).write_bytes(b"updated content")

        catalog = ContentCatalog(
            self.data_root,
            self.summary_path,
            source_pdf_root=source_root,
        )

        self.assertNotIn(old_source_id, catalog.books)
        self.assertFalse(catalog.discovered_sources_by_class[2][0].processed)

    def test_explicit_page_requires_a_book_and_selects_requested_page(self) -> None:
        with self.assertRaisesRegex(ChatError, "Choose a specific book"):
            self.catalog.retrieve(1, "tree", page_number=1)
        results = self.catalog.retrieve(
            1,
            "no matching words",
            source_id="0123456789abcdef",
            page_number=2,
        )
        self.assertEqual([(doc.page, score) for doc, score in results], [(2, 1_000_000.0)])

    def test_missing_printed_page_index_fails_instead_of_guessing_pdf_page(self) -> None:
        book = self.catalog.books["0123456789abcdef"]
        book.printed_pages_path.unlink()
        service = ChatService(self.catalog, api_key="test-key", completion=lambda *_: "unused")
        with self.assertRaisesRegex(ChatError, "Printed page numbers have not been indexed"):
            service.respond({
                "class": 1,
                "source_id": book.source_id,
                "page": 1,
                "question": "Explain this page",
            })

    def test_page_reference_in_question_resolves_named_book_and_only_that_page(self) -> None:
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["messages"] = messages
            return "Here is the explanation."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "question": "Explain First Language Telugu page number 3",
        })

        self.assertEqual(len(response["sources"]), 1)
        self.assertEqual(
            response["sources"][0]["book"],
            "First Language (Telugu) (English Medium)",
        )
        self.assertEqual(response["sources"][0]["page"], 3)
        self.assertEqual(response["sources"][0]["pdf_page"], 11)
        self.assertTrue(response["visual_context_used"])
        self.assertEqual(captured["messages"][-1]["content"][1]["type"], "image_url")

    def test_ambiguous_telugu_page_request_asks_for_book_selection(self) -> None:
        service = ChatService(self.catalog, api_key="test-key", completion=lambda *_: "unused")
        with self.assertRaisesRegex(ChatError, "multiple Class 1 books"):
            service.respond({
                "class": 1,
                "question": "Explain Telugu page number 3",
            })

    def test_explicit_page_with_unreadable_text_uses_its_page_image(self) -> None:
        document = next(
            item for item in self.catalog.documents_for_class(1)
            if item.title == "First Language (Telugu) (English Medium)"
            and item.page == 11
        )
        self.assertEqual(document.text, "")
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], model: str, api_key: str) -> str:
            captured["user_content"] = messages[-1]["content"]
            return "I can explain what is visible on this page."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "question": "Explain First Language Telugu page 3",
        })

        self.assertTrue(response["visual_context_used"])
        self.assertEqual(response["sources"][0]["page"], 3)
        self.assertEqual(response["sources"][0]["pdf_page"], 11)
        self.assertIsInstance(captured["user_content"], list)
        self.assertEqual(captured["user_content"][1]["type"], "image_url")
        self.assertIn(
            "Do not ask the student to supply the title, topic, or words",
            captured["user_content"][0]["text"],
        )

    def test_explicit_page_with_readable_text_uses_text_only_completion(self) -> None:
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["user_content"] = messages[-1]["content"]
            return "I looked at the selected page."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "source_id": "0123456789abcdef",
            "page": 1,
            "question": "Explain page number 1",
        })

        self.assertFalse(response["visual_context_used"])
        self.assertEqual(response["provider"], "groq")
        self.assertIsInstance(captured["user_content"], str)
        self.assertIn("A tree has green leaves", captured["user_content"])

    def test_grade_guidance_changes_by_class_and_missing_evidence_is_explicit(self) -> None:
        self.assertIn("very short sentences", _grade_prompt(1, True))
        self.assertIn("subject vocabulary", _grade_prompt(8, True))
        self.assertIn("matching textbook passage was not found", _grade_prompt(4, False))

    def test_picture_question_sends_page_image_and_returns_source_citation(self) -> None:
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], model: str, api_key: str) -> str:
            captured["messages"] = messages
            captured["model"] = model
            captured["api_key"] = api_key
            return "I can see a tree on this page."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "source_id": "0123456789abcdef",
            "page": 1,
            "question": "What can you see in this picture?",
        })
        user_content = captured["messages"][-1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(response["visual_context_used"])
        self.assertEqual(response["sources"][0]["page"], 1)
        self.assertEqual(response["answer"], "I can see a tree on this page.")
        self.assertEqual(captured["api_key"], "test-key")

    def test_text_question_uses_cached_answer_on_identical_request(self) -> None:
        calls = 0

        def completion(_messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            nonlocal calls
            calls += 1
            return "A tree has green leaves."

        service = ChatService(
            self.catalog,
            api_key="test-key",
            completion=completion,
        )
        request = {
            "class": 1,
            "source_id": "0123456789abcdef",
            "question": "What does a tree have?",
        }

        first = service.respond(request)
        second = service.respond(request)

        self.assertEqual(calls, 1)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["answer"], second["answer"])

    def test_insufficient_evidence_retrieves_more_pages_once(self) -> None:
        book = self.catalog.books["0123456789abcdef"]
        package_dir = book.manifest_path.parent
        manifest = json.loads(book.manifest_path.read_text(encoding="utf-8"))
        manifest["assets"] = []
        for page_number in range(1, 7):
            text_relative = f"text/expanded_{page_number:04d}.txt"
            image_relative = f"pages/expanded_{page_number:04d}.png"
            (package_dir / "text").mkdir(exist_ok=True)
            (package_dir / "pages").mkdir(exist_ok=True)
            (package_dir / text_relative).write_text(
                f"Comet evidence on page {page_number}. A comet crosses the sky.",
                encoding="utf-8",
            )
            Image.new("RGB", (80, 60), "white").save(package_dir / image_relative)
            manifest["assets"].append({
                "kind": "pdf_page",
                "page": page_number,
                "text_path": text_relative,
                "image_path": image_relative,
            })
        book.manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        calls: list[str] = []

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            calls.append(str(messages[-1]["content"]))
            if len(calls) == 1:
                return "[NEED_MORE_TEXTBOOK_EVIDENCE] The first pages do not explain this."
            return "The textbook says a comet crosses the sky."

        catalog = ContentCatalog(self.data_root, self.summary_path)
        response = ChatService(
            catalog,
            api_key="test-key",
            completion=completion,
        ).respond({
            "class": 1,
            "source_id": book.source_id,
            "question": "comet",
        })

        self.assertEqual(len(calls), 2)
        self.assertTrue(response["retrieval_expanded"])
        self.assertEqual(len(response["sources"]), 6)
        self.assertIn("Additional relevant textbook evidence", calls[1])
        self.assertNotIn("[NEED_MORE_TEXTBOOK_EVIDENCE]", response["answer"])

    def test_visual_question_uses_gemini_without_requiring_groq_key(self) -> None:
        captured: dict[str, Any] = {}

        def groq_completion(*_args: Any) -> str:
            self.fail("Visual page questions should not be sent to Groq.")

        def vision_completion(
            messages: list[dict[str, Any]],
            model: str,
            api_key: str,
        ) -> str:
            captured["messages"] = messages
            captured["model"] = model
            captured["api_key"] = api_key
            return "The picture shows a tree."

        service = ChatService(
            self.catalog,
            api_key="",
            gemini_api_key="gemini-test-key",
            gemini_model="gemini-test-model",
            completion=groq_completion,
            vision_completion=vision_completion,
        )
        response = service.respond({
            "class": 1,
            "source_id": "0123456789abcdef",
            "page": 1,
            "question": "What can you see in this picture?",
        })

        self.assertEqual(response["provider"], "gemini")
        self.assertEqual(response["model"], "gemini-test-model")
        self.assertEqual(response["answer"], "The picture shows a tree.")
        self.assertTrue(response["visual_context_used"])
        self.assertEqual(captured["model"], "gemini-test-model")
        self.assertEqual(captured["api_key"], "gemini-test-key")
        self.assertIsInstance(captured["messages"][-1]["content"], list)

    def test_visual_question_reports_missing_gemini_key_without_groq_fallback(self) -> None:
        service = ChatService(
            self.catalog,
            api_key="groq-test-key",
            gemini_api_key="",
        )
        with self.assertRaisesRegex(ChatError, "Gemini API key is missing"):
            service.respond({
                "class": 1,
                "source_id": "0123456789abcdef",
                "page": 1,
                "question": "What can you see in this picture?",
            })

    def test_follow_up_plural_picture_question_keeps_exact_page_image_context(self) -> None:
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The page shows a bus and a car."

        book = self.catalog.books["0123456789abcdef"]
        service = ChatService(
            self.catalog,
            api_key="test-key",
            completion=completion,
        )
        response = service.respond({
            "class": 1,
            "source_id": book.source_id,
            "page": 1,
            "question": "Explain this song.",
            "history": [
                {"role": "user", "content": "Explain this song."},
                {"role": "assistant", "content": "It is a song about a bus and car."},
            ],
        })

        follow_up = service.respond({
            "class": 1,
            "source_id": book.source_id,
            "question": "Youyes i like waht pictures are there ins ide",
            "history": [
                {
                    "role": "user",
                    "content": "Explain this song.",
                    "page_context": response["page_context"],
                },
                {"role": "assistant", "content": response["answer"]},
            ],
        })

        self.assertEqual(
            [(source["page"], source["pdf_page"]) for source in follow_up["sources"]],
            [(1, 1)],
        )
        self.assertEqual(follow_up["page_context"], response["page_context"])
        self.assertTrue(follow_up["visual_context_used"])
        self.assertIsInstance(captured["content"], list)
        self.assertEqual(captured["content"][1]["type"], "image_url")
        self.assertIn("The attached image is that exact printed page", captured["content"][0]["text"])
        self.assertIn("Do not ask the student to supply the title", captured["content"][0]["text"])

    def test_follow_up_picture_question_recovers_page_from_prior_question(self) -> None:
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The selected page shows a bus and a car."

        book = self.catalog.books["0123456789abcdef"]
        response = ChatService(
            self.catalog,
            api_key="test-key",
            completion=completion,
        ).respond({
            "class": 1,
            "source_id": book.source_id,
            "question": "Youyes, what pictures are there inside?",
            "history": [
                {"role": "user", "content": "Explain page number 1."},
                {"role": "assistant", "content": "The page is a song about vehicles."},
            ],
        })

        self.assertEqual(
            [(source["page"], source["pdf_page"]) for source in response["sources"]],
            [(1, 1)],
        )
        self.assertTrue(response["visual_context_used"])
        self.assertIsInstance(captured["content"], list)
        self.assertEqual(captured["content"][1]["type"], "image_url")

    def test_missing_key_and_invalid_class_fail_explicitly(self) -> None:
        service = ChatService(self.catalog, api_key="", completion=lambda *_: "unused")
        with self.assertRaisesRegex(ChatError, "Groq API key is missing"):
            service.respond({"class": 1, "question": "What is a tree?"})
        with self.assertRaisesRegex(ChatError, "Choose a class from 1 to 8"):
            service.respond({"class": 9, "question": "What is a tree?"})

    def test_groq_request_uses_server_side_bearer_key(self) -> None:
        response = BytesIO(
            b'{"choices":[{"message":{"content":"A tree is a plant."}}]}'
        )
        with patch("textbook_engine.chat.urllib.request.urlopen", return_value=response) as urlopen:
            answer = _call_groq([], "test-model", "unit-test-key")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-key")
        self.assertEqual(
            request.get_header("User-agent"),
            "learning-content-engine/0.1.0",
        )
        self.assertEqual(answer, "A tree is a plant.")

    def test_groq_truncated_response_is_reported_instead_of_returned_as_complete(self) -> None:
        response = BytesIO(
            b'{"choices":[{"message":{"content":"An unfinished answer"},"finish_reason":"length"}]}'
        )
        with patch("textbook_engine.chat.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ChatError, "before finishing its answer"):
                _call_groq([], "test-model", "unit-test-key")

    def test_groq_auth_error_shows_provider_detail_without_echoing_secret(self) -> None:
        response_body = json.dumps({
            "error": {
                "message": "Invalid API Key: gsk_sensitiveexample",
            }
        }).encode("utf-8")
        error = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            BytesIO(response_body),
        )
        with patch("textbook_engine.chat.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ChatError) as raised:
                _call_groq([], "test-model", "gsk_sensitiveexample")

        message = str(raised.exception)
        self.assertIn("HTTP 401", message)
        self.assertIn("Invalid API Key", message)
        self.assertNotIn("gsk_sensitiveexample", message)

    def test_picture_question_without_page_image_does_not_claim_visual_context(self) -> None:
        document = self.catalog.documents_for_class(1)[0]
        document.page_image.unlink()
        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], model: str, api_key: str) -> str:
            captured["user_message"] = messages[-1]["content"]
            return "Please choose a page image."

        service = ChatService(self.catalog, api_key="test-key", completion=completion)
        response = service.respond({
            "class": 1,
            "source_id": document.source_id,
            "page": document.page,
            "question": "What can you see in this picture?",
        })
        self.assertFalse(response["visual_context_used"])
        self.assertIn("No textbook page image was selected", response["warnings"][0])
        self.assertIn("No page image is attached", captured["user_message"])

    def test_versioned_filesystem_package_loads_into_catalog_with_pdf_citations(self) -> None:
        books_root = self.data_root / "books"
        store = FileSystemBookStore(books_root)
        image_source = self.root / "versioned_page.png"
        Image.new("RGB", (80, 60), "green").save(image_source)
        book_id = "class_01_science_demo"
        version_id = "v_abc123"
        source_filename = "Class_01_Science.pdf"
        manifest = {
            "schema_version": "1.0",
            "book_id": book_id,
            "version_id": version_id,
            "class": 1,
            "subject": "Science",
            "language": "English",
            "title": "Science Demo",
            "source_pdf": {
                "filename": source_filename,
                "sha256": "a" * 64,
                "physical_page_count": 2,
            },
            "chapters": [{
                "chapter_id": "unit_01",
                "title": "Plants",
                "chapter_number": 1,
                "pdf_pages": {"from": 1, "to": 2},
                "page_count": 2,
            }],
        }
        chapter = {
            "schema_version": "1.0",
            "book_id": book_id,
            "version_id": version_id,
            "chapter_id": "unit_01",
            "number": 1,
            "title": "Plants",
            "overview": "Plants need water and sunlight.",
            "pages": [
                {
                    "page_id": f"pdf_{page_number:04d}",
                    "source_provenance": {
                        "source_pdf_filename": source_filename,
                        "source_pdf_page_number_1_indexed": page_number,
                        "printed_page_number": None,
                        "text_extraction_method": "native_pdf_text",
                    },
                    "page_heading": f"Plant facts, page {page_number}",
                    "source_text": {
                        "extracted_text_raw_ordered": (
                            "A plant grows from a seed."
                            if page_number == 1
                            else "Plants."
                        ),
                        "extracted_text_normalized_for_search": (
                            "A plant grows from a seed."
                            if page_number == 1
                            else "Plants."
                        ),
                    },
                    "visuals": {
                        "page_image": f"images/pages/pdf_page_{page_number:04d}.png",
                        "items": [],
                    },
                }
                for page_number in (1, 2)
            ],
        }
        store.publish(
            book_id,
            version_id,
            manifest,
            {"unit_01": chapter},
            assets={
                "images/pages/pdf_page_0001.png": image_source,
                "images/pages/pdf_page_0002.png": image_source,
            },
        )

        with patch.object(store, "list_packages", wraps=store.list_packages) as list_packages:
            catalog = ContentCatalog(
                self.data_root,
                self.summary_path,
                book_store=store,
            )
        list_packages.assert_called_once_with()
        book = next(
            book for book in catalog.books.values()
            if book.filename == source_filename
        )
        self.assertEqual(book.page_number_type, "pdf")
        context, sources, images, image_paths, _has_undescribed_images = catalog.prepared_page_context(
            book,
            2,
        )
        self.assertIn("Plants.", context)
        self.assertEqual(sources[0]["page"], 2)
        self.assertEqual(sources[0]["pdf_page"], 2)
        self.assertEqual(sources[0]["page_number_type"], "pdf")
        self.assertEqual(images[0]["pdf_page"], 2)
        self.assertEqual(image_paths[0].read_bytes(), image_source.read_bytes())

        captured: dict[str, Any] = {}

        def completion(messages: list[dict[str, Any]], _model: str, _api_key: str) -> str:
            captured["content"] = messages[-1]["content"]
            return "The selected textbook page is available."

        with (
            patch(
                "textbook_engine.printed_pages.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.printed_pages._ocr_footer",
                side_effect=lambda _exe, image: [{
                    "printed_page": 40 + int(image.stem[-4:]),
                    "confidence": 92.0,
                    "corner": "left",
                }],
            ) as ocr,
            patch(
                "textbook_engine.ocr.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.ocr.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value="Plants need water and sunlight.",
            ) as page_ocr,
        ):
            response = ChatService(
                catalog,
                api_key="test-key",
                completion=completion,
            ).respond({
                "class": 1,
                "source_id": book.source_id,
                "page": 42,
                "question": "What can you see in this picture?",
            })
        self.assertEqual(ocr.call_count, 2)
        page_ocr.assert_called_once()
        self.assertTrue(book.printed_pages_path.is_file())
        self.assertEqual(catalog.printed_page_map_for(book), {41: 1, 42: 2})
        self.assertEqual(response["sources"][0]["page"], 42)
        self.assertEqual(response["sources"][0]["pdf_page"], 2)
        self.assertTrue(response["visual_context_used"])
        self.assertIsInstance(captured["content"], list)
        self.assertEqual(captured["content"][1]["type"], "image_url")
        self.assertIn(
            "Plants need water and sunlight.",
            captured["content"][0]["text"],
        )
        self.assertEqual(
            catalog.quality_warning_for(book),
            None,
        )
        restarted_catalog = ContentCatalog(self.data_root, self.summary_path)
        restarted_book = next(
            candidate
            for candidate in restarted_catalog.books.values()
            if candidate.filename == source_filename
        )
        self.assertEqual(
            restarted_catalog.resolve_printed_page(restarted_book, 42),
            2,
        )
        with patch("textbook_engine.pipeline._ocr_page") as repeated_ocr:
            restarted_catalog.prepare_book_text(restarted_book, pdf_page=2)
        repeated_ocr.assert_not_called()
        recovered_section = next(
            section
            for chapter in restarted_catalog.prepared_chapters[restarted_book.source_id].values()
            for section in chapter["data"]["sections"]
            if section["pdf_page_number"] == 2
        )
        self.assertEqual(
            recovered_section["source_text"],
            "Plants need water and sunlight.",
        )


if __name__ == "__main__":
    unittest.main()
