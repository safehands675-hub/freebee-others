from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from textbook_engine.book_store import FileSystemBookStore
from textbook_engine.core import ContentEngineError


class FileSystemBookStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = FileSystemBookStore(self.root)
        self.manifest, self.chapters = self._package()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _package() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        manifest = {
            "schema_version": "1.0",
            "title": "Plants",
            "class": 4,
            "subject": "Science",
            "source_pdf": {
                "filename": "Class_04_Science.pdf",
                "sha256": "a" * 64,
                "physical_page_count": 1,
            },
            "chapters": [{
                "chapter_id": "unit_01",
                "title": "Plants",
                "pdf_pages": {"from": 1, "to": 1},
                "page_count": 1,
            }],
        }
        chapter = {
            "schema_version": "1.0",
            "book_id": "class_04_science_plants",
            "version_id": "v001",
            "chapter_id": "unit_01",
            "pages": [{
                "page_id": "pdf_0001",
                "source_provenance": {
                    "source_pdf_filename": "Class_04_Science.pdf",
                    "source_pdf_page_number_1_indexed": 1,
                    "printed_page_number": None,
                    "text_extraction_method": "native_pdf_text",
                },
                "source_text": {
                    "extracted_text_raw_ordered": "Plants need water.",
                    "extracted_text_normalized_for_search": "Plants need water.",
                },
            }],
        }
        return manifest, {"unit_01": chapter}

    def test_publishes_and_reads_a_versioned_page_record_package(self) -> None:
        package_path = self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )

        self.assertTrue((package_path / "book.json").is_file())
        self.assertTrue((package_path / "chapters" / "unit_01.json").is_file())
        saved_manifest = self.store.read_manifest("class_04_science_plants", "v001")
        self.assertEqual(saved_manifest["chapters"][0]["file"], "chapters/unit_01.json")
        self.assertEqual(
            self.store.read_chapter(
                "class_04_science_plants",
                "v001",
                "unit_01",
            ),
            self.chapters["unit_01"],
        )

    def test_republishing_identical_content_is_idempotent(self) -> None:
        first = self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )
        second = self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )

        self.assertEqual(first, second)

    def test_lists_published_packages_without_exposing_partial_directories(self) -> None:
        self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )
        (self.root / "class_04_science_plants" / "versions" / "v002").mkdir()

        self.assertEqual(
            self.store.list_packages(),
            [("class_04_science_plants", "v001")],
        )

    def test_published_version_cannot_be_overwritten(self) -> None:
        self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )
        changed_chapters = {
            "unit_01": {
                **self.chapters["unit_01"],
                "pages": [{
                    **self.chapters["unit_01"]["pages"][0],
                    "source_text": {
                        "extracted_text_raw_ordered": "Different source.",
                        "extracted_text_normalized_for_search": "Different source.",
                    },
                }],
            }
        }

        with self.assertRaisesRegex(ContentEngineError, "new version ID"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                changed_chapters,
            )

    def test_rejects_path_traversal_and_printed_page_guesses(self) -> None:
        with self.assertRaisesRegex(ContentEngineError, "Book ID"):
            self.store.publish("../outside", "v001", self.manifest, self.chapters)

        chapter = self.chapters["unit_01"]
        invalid_chapter = {
            **chapter,
            "pages": [{
                **chapter["pages"][0],
                "source_provenance": {
                    **chapter["pages"][0]["source_provenance"],
                    "printed_page_number": 0,
                },
            }],
        }
        with self.assertRaisesRegex(ContentEngineError, "invalid printed page"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                {"unit_01": invalid_chapter},
            )

    def test_requires_manifest_and_chapter_page_coverage_to_match(self) -> None:
        with self.assertRaisesRegex(ContentEngineError, "must match"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                {"unexpected": {}},
            )

    def test_requires_complete_physical_page_coverage(self) -> None:
        manifest = {
            **self.manifest,
            "source_pdf": {
                **self.manifest["source_pdf"],
                "physical_page_count": 2,
            },
        }
        with self.assertRaisesRegex(ContentEngineError, "every physical PDF page"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                manifest,
                self.chapters,
            )

    def test_detects_tampered_chapter_content_on_read(self) -> None:
        package_path = self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            self.chapters,
        )
        chapter_path = package_path / "chapters" / "unit_01.json"
        chapter_path.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(ContentEngineError, "integrity check"):
            self.store.read_chapter(
                "class_04_science_plants",
                "v001",
                "unit_01",
            )

    def test_publishes_and_integrity_checks_referenced_image_assets(self) -> None:
        image_source = self.root / "page.png"
        Image.new("RGB", (20, 20), "white").save(image_source)
        chapter = self.chapters["unit_01"]
        chapter_with_image = {
            **chapter,
            "pages": [{
                **chapter["pages"][0],
                "visuals": {"page_image": "images/pages/pdf_page_0001.png"},
            }],
        }
        package_path = self.store.publish(
            "class_04_science_plants",
            "v001",
            self.manifest,
            {"unit_01": chapter_with_image},
            assets={"images/pages/pdf_page_0001.png": image_source},
        )

        stored_image = self.store.asset_path(
            "class_04_science_plants",
            "v001",
            "images/pages/pdf_page_0001.png",
        )
        self.assertTrue(stored_image.is_file())
        self.assertEqual(
            stored_image.read_bytes(),
            image_source.read_bytes(),
        )
        stored_image.write_bytes(b"tampered")
        with self.assertRaisesRegex(ContentEngineError, "integrity check"):
            self.store.asset_path(
                "class_04_science_plants",
                "v001",
                "images/pages/pdf_page_0001.png",
            )
        self.assertTrue((package_path / "book.json").is_file())

    def test_rejects_unsafe_and_unreferenced_image_assets(self) -> None:
        image_source = self.root / "page.png"
        image_source.write_bytes(b"page")
        with self.assertRaisesRegex(ContentEngineError, "safe paths"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                self.chapters,
                assets={"images/../outside.png": image_source},
            )
        with self.assertRaisesRegex(ContentEngineError, "must be referenced"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                self.chapters,
                assets={"images/pages/page.png": image_source},
            )

    def test_does_not_publish_when_json_serialization_fails(self) -> None:
        chapter = {
            **self.chapters["unit_01"],
            "unsupported": object(),
        }
        with self.assertRaisesRegex(ContentEngineError, "non-JSON"):
            self.store.publish(
                "class_04_science_plants",
                "v001",
                self.manifest,
                {"unit_01": chapter},
            )

        self.assertFalse(
            (self.root / "class_04_science_plants" / "versions" / "v001").exists()
        )
