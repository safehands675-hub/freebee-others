from __future__ import annotations

import json
import hashlib
import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from textbook_engine.core import ContentEngineError
from textbook_engine.core import ingest_source
from textbook_engine.benchmark import benchmark_processed_pdfs
from textbook_engine.cli import main as cli_main
from textbook_engine.page_index import PersistentPageIndex
from textbook_engine.pipeline import (
    PIPELINE_VERSION,
    ensure_all_pdf_pages,
    is_current_pdf_package,
    process_pdf,
)


class PdfProcessingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "data"
        self.input_dir = self.root / "inputs"
        self.input_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _make_pdf(self, name: str, text: str, page_count: int = 1) -> Path:
        source = self.input_dir / name
        document = pymupdf.open()
        for page_number in range(page_count):
            page = document.new_page()
            if text:
                page.insert_text((72, 72), f"{text} Page {page_number + 1}.")
        document.save(source)
        document.close()
        return source

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
                "source_sha256": manifest["source"]["sha256"],
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
        return {"matched_pages": 1, "needs_review_pages": 0}

    def _process_pdf(self, source: Path, **kwargs: object) -> dict[str, object]:
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
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ),
        ):
            return process_pdf(source, self.data_dir, **kwargs)

    def test_one_call_processes_and_catalogs_a_pdf_idempotently(self) -> None:
        source = self._make_pdf(
            "new-lesson.pdf",
            "UNIT - 1\nClass 4 English lesson. A tree grows from a seed in the soil.",
        )
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ) as tesseract,
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ) as extract_images,
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ) as printed_index,
        ):
            first = process_pdf(source, self.data_dir)
            second = process_pdf(source, self.data_dir)

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["class_number"], 4)
        self.assertEqual(first["pdf_pages"], 1)
        self.assertEqual(tesseract.call_count, 0)
        extract_images.assert_called_once()
        printed_index.assert_called_once()
        manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["processing"]["status"], "complete")
        self.assertEqual(
            manifest["processing"]["pipeline_version"],
            PIPELINE_VERSION,
        )
        self.assertEqual(manifest["source"]["class_number"], 4)
        self.assertEqual(
            manifest["assets"][0]["text_extraction_method"],
            "native_pdf",
        )
        index = PersistentPageIndex(
            self.data_dir / "search" / "page_index.sqlite3"
        )
        indexed_pages = index.search("tree seed", 4, [first["source_id"]], 1)
        self.assertEqual(len(indexed_pages), 1)
        self.assertEqual(indexed_pages[0].page, 1)
        self.assertEqual(indexed_pages[0].heading, "UNIT - 1")
        self.assertTrue(indexed_pages[0].chapter.startswith("unit_01:"))
        self.assertEqual(indexed_pages[0].extraction_method, "native_pdf")
        self.assertEqual(indexed_pages[0].ocr_text, "")
        self.assertIsNotNone(indexed_pages[0].image_path)
        benchmark = benchmark_processed_pdfs(self.data_dir)
        self.assertEqual(benchmark["corpus_summary"]["expected_physical_pages"], 1)
        self.assertEqual(benchmark["corpus_summary"]["indexed_pages"], 1)
        self.assertIsNone(benchmark["corpus_summary"]["cer"])
        self.assertEqual(
            benchmark["measurements"]["cer_wer"],
            "not measured: no page text ground truth",
        )

        page_text = Path(first["manifest"]).parent.joinpath(
            manifest["assets"][0]["text_path"]
        ).read_text(encoding="utf-8")
        ground_truth_path = self.root / "ground_truth.json"
        ground_truth_path.write_text(
            json.dumps({
                "documents": {
                    first["source_id"]: {
                        "pages": {"1": page_text},
                        "chapter_start_pdf_pages": [1],
                    },
                },
            }),
            encoding="utf-8",
        )
        labeled_benchmark = benchmark_processed_pdfs(
            self.data_dir,
            ground_truth_path=ground_truth_path,
        )
        self.assertEqual(labeled_benchmark["corpus_summary"]["cer"], 0)
        self.assertEqual(labeled_benchmark["corpus_summary"]["wer"], 0)
        self.assertEqual(labeled_benchmark["corpus_summary"]["ground_truth_documents"], 1)
        self.assertEqual(
            labeled_benchmark["corpus_summary"]["chapter_boundary_precision"],
            1,
        )
        self.assertEqual(
            labeled_benchmark["corpus_summary"]["chapter_boundary_recall"],
            1,
        )

    def test_old_pipeline_version_is_reprocessed(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        initial = self._process_pdf(source)
        manifest_path = Path(initial["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["processing"]["pipeline_version"] = "1.1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        refreshed = self._process_pdf(source)

        self.assertEqual(refreshed["status"], "complete")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["processing"]["pipeline_version"],
            PIPELINE_VERSION,
        )
        self.assertEqual(saved["processing"]["status"], "complete")

    def test_completed_package_accepts_page_assets_and_index_in_partial_import_order(self) -> None:
        source = self._make_pdf(
            "Class_04_English.pdf",
            "A plant grows from a seed.",
            page_count=3,
        )
        result = self._process_pdf(source)
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["assets"] = [
            manifest["assets"][2],
            manifest["assets"][0],
            manifest["assets"][1],
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        printed_index_path = manifest_path.parent / "printed_page_index.json"
        printed_index = json.loads(printed_index_path.read_text(encoding="utf-8"))
        printed_index["pages"] = [
            printed_index["pages"][2],
            printed_index["pages"][0],
            printed_index["pages"][1],
        ]
        printed_index_path.write_text(json.dumps(printed_index), encoding="utf-8")

        self.assertTrue(is_current_pdf_package(manifest_path))

    def test_benchmark_distinguishes_nonempty_but_unusable_page_text(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        result = self._process_pdf(source)
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        page_text_path = manifest_path.parent / manifest["assets"][0]["text_path"]
        page_text_path.write_text("\uf0bb" * 100, encoding="utf-8")

        benchmark = benchmark_processed_pdfs(self.data_dir)

        self.assertEqual(benchmark["corpus_summary"]["pages_with_text"], 1)
        self.assertEqual(benchmark["corpus_summary"]["pages_with_usable_text"], 0)
        self.assertEqual(benchmark["corpus_summary"]["text_coverage"], 1)
        self.assertEqual(benchmark["corpus_summary"]["usable_text_coverage"], 0)
        self.assertFalse(benchmark["documents"][0]["pages"][0]["has_usable_text"])

    def test_persistent_index_ignores_printed_pages_for_another_source_hash(self) -> None:
        source = self._make_pdf(
            "Class_04_English.pdf",
            "UNIT - 1\nA plant grows from a seed.",
        )
        result = self._process_pdf(source)
        manifest_path = Path(result["manifest"])
        printed_path = manifest_path.parent / "printed_page_index.json"
        printed_path.write_text(
            json.dumps({
                "schema_version": 1,
                "source_id": result["source_id"],
                "source_sha256": "0" * 64,
                "pages": [{
                    "pdf_page": 1,
                    "printed_page": 999,
                    "confidence": 99,
                    "status": "matched",
                }],
            }),
            encoding="utf-8",
        )

        index = PersistentPageIndex(self.data_dir / "search" / "page_index.sqlite3")
        index.index_processed_package(manifest_path, 4, "English")
        indexed_pages = index.search("plant seed", 4, [result["source_id"]], 1)

        self.assertEqual(len(indexed_pages), 1)
        self.assertIsNone(indexed_pages[0].printed_page)

    def test_explicit_ocr_language_is_honored_for_imported_pdfs(self) -> None:
        source = self._make_pdf(
            "Class_04_Science.pdf",
            "This long native text is not in the requested Telugu script. " * 3,
        )
        recognized = ("\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 " * 12).strip()
        with patch(
            "textbook_engine.pipeline._ocr_page",
            return_value=recognized,
        ) as page_ocr:
            result = self._process_pdf(source, language="Telugu")

        page_ocr.assert_called_once_with("tesseract.exe", unittest.mock.ANY, "tel")
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["ocr_language"], "tel")
        self.assertEqual(manifest["processing"]["result"]["ocr_language"], "tel")

    def test_shorter_readable_ocr_replaces_long_corrupted_native_text(self) -> None:
        corrupted_text = (
            "The lesson begins here "
            + "\ufffd" * 24
            + " and continues with unreadable symbols."
        )
        source = self._make_pdf("Class_04_Mathematics.pdf", corrupted_text)
        recognized_text = "Math lesson: count the objects shown."
        original_page_extraction = ensure_all_pdf_pages

        def extract_then_corrupt_page_text(
            source_path: Path,
            manifest_path: Path,
            *,
            dpi: int = 120,
        ) -> int:
            pages_added = original_page_extraction(
                source_path,
                manifest_path,
                dpi=dpi,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            page_asset = next(
                asset
                for asset in manifest["assets"]
                if asset.get("kind") == "pdf_page"
            )
            (manifest_path.parent / page_asset["text_path"]).write_text(
                corrupted_text,
                encoding="utf-8",
            )
            return pages_added

        with patch(
            "textbook_engine.pipeline._ocr_page",
            return_value=recognized_text,
        ) as page_ocr:
            with patch(
                "textbook_engine.pipeline.ensure_all_pdf_pages",
                side_effect=extract_then_corrupt_page_text,
            ):
                result = self._process_pdf(source)

        page_ocr.assert_called_once()
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        asset = manifest["assets"][0]
        text_path = Path(result["manifest"]).parent / asset["text_path"]
        processed_text = text_path.read_text(encoding="utf-8")
        self.assertEqual(asset["text_extraction_method"], "tesseract_ocr")
        self.assertFalse(asset["needs_ocr_review"])
        self.assertEqual(processed_text.strip(), recognized_text)
        self.assertEqual(result["pages_needing_ocr_review"], 0)

    def test_native_text_pdf_completes_when_tesseract_is_unavailable(self) -> None:
        source = self._make_pdf(
            "Class_04_English.pdf",
            "A plant grows from a seed. Sunlight and water help it grow.",
        )
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
        ):
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ocr_status"], "not_needed")
        self.assertEqual(result["pages_ocr_processed"], 0)
        self.assertTrue(Path(result["manifest"]).is_file())
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["processing"]["status"], "complete")
        self.assertEqual(
            manifest["assets"][0]["text_extraction_method"],
            "native_pdf",
        )

    def test_missing_tesseract_does_not_block_sparse_pdf_or_page_index(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "Hi")
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
        ):
            result = process_pdf(source, self.data_dir)

        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(
            (manifest_path.parent / "printed_page_index.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ocr_status"], "partial")
        self.assertEqual(result["pages_without_ocr_language"], 1)
        self.assertEqual(manifest["processing"]["status"], "complete")
        self.assertTrue(manifest["assets"][0]["needs_ocr_review"])
        self.assertEqual(index["indexing_status"], "unavailable")
        self.assertEqual(index["pages"][0]["status"], "needs_review")
        self.assertTrue(is_current_pdf_package(manifest_path))

    def test_ocr_page_failure_preserves_native_text_and_completes_package(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "Hi", page_count=2)
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                side_effect=[
                    ContentEngineError("OCR page failed."),
                    "This page was recovered by OCR with readable English text.",
                ],
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
            result = process_pdf(source, self.data_dir)

        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        first_asset, second_asset = manifest["assets"]
        first_text = (Path(result["manifest"]).parent / first_asset["text_path"]).read_text(
            encoding="utf-8"
        )
        second_text = (Path(result["manifest"]).parent / second_asset["text_path"]).read_text(
            encoding="utf-8"
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ocr_status"], "partial")
        self.assertEqual(result["pages_ocr_processed"], 2)
        self.assertEqual(result["pages_without_ocr_language"], 0)
        self.assertEqual(result["pages_ocr_failed"], 1)
        self.assertIn("OCR failed on 1 page(s).", result["warnings"])
        self.assertEqual(first_asset["text_extraction_method"], "native_pdf_ocr_failed")
        self.assertEqual(first_asset["ocr_error"], "OCR page failed.")
        self.assertTrue(first_asset["needs_ocr_review"])
        self.assertIn("Hi Page 1", first_text)
        self.assertEqual(second_asset["text_extraction_method"], "tesseract_ocr")
        self.assertIn("recovered by OCR", second_text)

    def test_printed_page_ocr_failure_does_not_block_text_search_index(self) -> None:
        source = self._make_pdf(
            "Class_04_English.pdf",
            "A plant grows from a seed. Sunlight and water help it grow.",
        )
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=ContentEngineError("Tesseract is missing."),
            ),
        ):
            result = process_pdf(source, self.data_dir)

        index = PersistentPageIndex(
            self.data_dir / "search" / "page_index.sqlite3"
        )
        pages = index.search("plant seed", 4, [result["source_id"]], 1)
        self.assertEqual(len(pages), 1)
        self.assertIn("plant", pages[0].text)
        self.assertEqual(result["printed_pages_matched"], 0)
        self.assertEqual(
            result["printed_page_index_status"],
            "needs_review",
        )

    def test_process_pdf_cli_passes_explicit_ocr_language(self) -> None:
        source = self.input_dir / "Class_04_Science.pdf"
        output = io.StringIO()
        with (
            patch("textbook_engine.cli.process_pdf", return_value={"status": "complete"}) as call,
            contextlib.redirect_stdout(output),
        ):
            status = cli_main([
                "process-pdf",
                str(source),
                "--language",
                "Telugu",
            ])

        self.assertEqual(status, 0)
        self.assertEqual(call.call_args.kwargs["language"], "Telugu")

    def test_same_filename_with_changed_content_gets_new_source_identity(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        first = self._process_pdf(source)
        first_id = first["source_id"]
        source.unlink()
        source = self._make_pdf("Class_04_English.pdf", "A river flows toward the sea.")

        second = self._process_pdf(source)

        self.assertNotEqual(second["source_id"], first_id)
        self.assertTrue((self.data_dir / "sources" / first_id / "manifest.json").is_file())
        second_manifest = json.loads(
            Path(second["manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            second_manifest["source"]["sha256"][:16],
            hashlib.sha256(source.read_bytes()).hexdigest()[:16],
        )

    def test_renamed_pdf_with_same_content_reuses_source_identity(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        first = self._process_pdf(source)
        renamed = self.input_dir / "Class_04_English_Renamed.pdf"
        shutil.copyfile(source, renamed)

        second = self._process_pdf(renamed)

        self.assertEqual(second["source_id"], first["source_id"])
        manifests = list((self.data_dir / "sources").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].name, Path(first["manifest"]).name)
        manifest = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["filename"], renamed.name)
        self.assertEqual(manifest["source"]["original_path"], str(renamed.resolve()))

    def test_corrupt_manifest_is_quarantined_and_rebuilt(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        first = self._process_pdf(source)
        manifest_path = Path(first["manifest"])
        manifest_path.write_text("{not valid json", encoding="utf-8")

        refreshed = self._process_pdf(source)

        self.assertEqual(refreshed["source_id"], first["source_id"])
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8"))["processing"]["status"],
            "complete",
        )
        quarantined = self.data_dir / "sources" / "_stale"
        self.assertTrue(list(quarantined.glob(f"{first['source_id']}-*/manifest.json")))

    def test_missing_page_assets_are_rebuilt(self) -> None:
        source = self._make_pdf(
            "Class_04_English.pdf",
            "A plant grows from a seed.",
            page_count=2,
        )
        first = self._process_pdf(source)
        manifest_path = Path(first["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        package_dir = manifest_path.parent
        (package_dir / manifest["assets"][0]["text_path"]).unlink()
        (package_dir / manifest["assets"][1]["image_path"]).unlink()

        refreshed = self._process_pdf(source)

        self.assertEqual(refreshed["pdf_pages"], 2)
        repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(all(
            (package_dir / asset[key]).is_file()
            for asset in repaired["assets"]
            if asset.get("kind") == "pdf_page"
            for key in ("text_path", "image_path")
        ))

    def test_failed_refresh_remains_not_ready_and_can_recover(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        first = self._process_pdf(source)
        manifest_path = Path(first["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["processing"]["pipeline_version"] = "1.1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=ContentEngineError("simulated extraction failure"),
            ),
        ):
            with self.assertRaisesRegex(ContentEngineError, "simulated extraction failure"):
                process_pdf(source, self.data_dir)

        failed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(failed_manifest["processing"]["status"], "processing")
        self.assertEqual(self._process_pdf(source)["status"], "complete")

    def test_missing_manifest_in_existing_package_is_quarantined_and_rebuilt(self) -> None:
        source = self._make_pdf("Class_04_English.pdf", "A plant grows from a seed.")
        first = self._process_pdf(source)
        Path(first["manifest"]).unlink()

        refreshed = self._process_pdf(source)

        self.assertEqual(refreshed["source_id"], first["source_id"])
        self.assertTrue(Path(refreshed["manifest"]).is_file())
        self.assertTrue(list(
            (self.data_dir / "sources" / "_stale").glob(
                f"{first['source_id']}-*/extracted/page_0001.txt"
            )
        ))

    def test_scanned_or_sparse_page_uses_ocr_to_recover_class_metadata(self) -> None:
        source = self._make_pdf("upload.pdf", "Hi")
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value="Class V. Plants make food using sunlight and water.",
            ) as page_ocr,
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ),
        ):
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["class_number"], 5)
        self.assertEqual(result["pages_ocr_processed"], 1)
        page_ocr.assert_called_once()
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        page = manifest["assets"][0]
        text = (Path(result["manifest"]).parent / page["text_path"]).read_text(encoding="utf-8")
        self.assertIn("Class V", text)
        self.assertEqual(page["text_extraction_method"], "tesseract_ocr")

    def test_filename_metadata_sets_class_subject_medium_and_board(self) -> None:
        source = self._make_pdf(
            "FirstClass_English_Medium_Mathematics_SSC_Common.pdf",
            "Numbers can be grouped and compared in different ways.",
        )
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
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
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["class_number"], 1)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["subject"], "Mathematics")
        self.assertEqual(manifest["source"]["medium"], "English")
        self.assertEqual(manifest["source"]["board"], "SSC")
        self.assertEqual(manifest["source"]["school"], "Common")

    def test_pipeline_completes_a_previously_partial_pdf_import(self) -> None:
        source = self._make_pdf(
            "Class_02_English_Medium_Math.pdf",
            "Class 2. Count numbers and compare groups.",
            page_count=2,
        )
        manifest_path = ingest_source(source, self.data_dir, pages="1", dpi=72)
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value="Class 2. Count numbers and compare groups.",
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
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["pdf_pages"], 2)
        self.assertEqual(result["pages_added_from_partial_import"], 1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([asset["page"] for asset in manifest["assets"]], [1, 2])

    def test_rejects_non_pdf_input(self) -> None:
        source = self.input_dir / "lesson.txt"
        source.write_text("lesson", encoding="utf-8")
        with self.assertRaisesRegex(ContentEngineError, "PDF files only"):
            process_pdf(source, self.data_dir)

    def test_unavailable_script_language_is_not_replaced_with_english_ocr(self) -> None:
        source = self._make_pdf(
            "Class_01_English_Medium_First_Language_(Telugu).pdf",
            "Hi",
        )
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng", "osd"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
            ) as page_ocr,
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ),
        ):
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["class_number"], 1)
        self.assertEqual(result["ocr_language"], "tel")
        self.assertFalse(result["ocr_language_available"])
        self.assertEqual(result["pages_without_ocr_language"], 1)
        page_ocr.assert_not_called()
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        asset = manifest["assets"][0]
        self.assertEqual(asset["text_extraction_method"], "native_pdf_language_ocr_unavailable")
        self.assertTrue(asset["needs_ocr_review"])
        text = (Path(result["manifest"]).parent / asset["text_path"]).read_text(encoding="utf-8")
        self.assertIn("Hi Page 1", text)

    def test_native_text_missing_the_book_script_is_replaced_with_language_ocr(self) -> None:
        source = self._make_pdf(
            "Class_01_English_Medium_First_Language_(Telugu).pdf",
            "This is Latin text that does not match the language of this Telugu textbook. " * 3,
        )
        recognized_text = ("\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 " * 12).strip()
        with (
            patch(
                "textbook_engine.pipeline.resolve_tesseract_command",
                return_value="tesseract.exe",
            ),
            patch(
                "textbook_engine.pipeline.resolve_tesseract_languages",
                return_value={"eng", "tel"},
            ),
            patch(
                "textbook_engine.pipeline._ocr_page",
                return_value=recognized_text,
            ) as page_ocr,
            patch(
                "textbook_engine.pipeline.extract_pdf_images",
                side_effect=self._mock_image_extraction,
            ),
            patch(
                "textbook_engine.pipeline.index_printed_pages",
                side_effect=self._mock_page_index,
            ),
        ):
            result = process_pdf(source, self.data_dir)

        self.assertEqual(result["pages_ocr_processed"], 1)
        page_ocr.assert_called_once_with(
            "tesseract.exe",
            unittest.mock.ANY,
            "tel",
        )
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        asset = manifest["assets"][0]
        text = (Path(result["manifest"]).parent / asset["text_path"]).read_text(encoding="utf-8")
        self.assertEqual(text.strip(), recognized_text)
        self.assertEqual(asset["text_extraction_method"], "tesseract_ocr")
        self.assertFalse(asset["needs_ocr_review"])


if __name__ == "__main__":
    unittest.main()
