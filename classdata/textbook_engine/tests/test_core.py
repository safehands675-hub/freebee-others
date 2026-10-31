from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image, features

from textbook_engine.core import (
    ContentEngineError,
    extract_pdf_images,
    ingest_source,
    render_annotation_preview,
    run_detection,
    validate_annotation_file,
)


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "data"
        self.source_dir = self.root / "inputs"
        self.source_dir.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _make_pdf(self) -> Path:
        path = self.source_dir / "lesson.pdf"
        document = pymupdf.open()
        first = document.new_page()
        first.insert_text((72, 72), "A tree has leaves.")
        second = document.new_page()
        second.insert_text((72, 72), "A bird can fly.")
        document.save(path)
        document.close()
        return path

    def test_ingests_selected_pdf_pages_and_extracts_text(self) -> None:
        source = self._make_pdf()
        manifest_path = ingest_source(source, self.data_dir, pages="2", dpi=72)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([asset["page"] for asset in manifest["assets"]], [2])
        page_asset = manifest["assets"][0]
        text_path = manifest_path.parent / page_asset["text_path"]
        self.assertIn("bird", text_path.read_text(encoding="utf-8"))
        self.assertTrue((manifest_path.parent / page_asset["image_path"]).is_file())
        self.assertEqual(source.read_bytes()[:4], b"%PDF")

    def test_extracts_embedded_pdf_images_and_page_placements_idempotently(self) -> None:
        source = self._make_pdf()
        embedded_source = self.source_dir / "embedded.png"
        Image.new("RGB", (24, 16), "blue").save(embedded_source)
        document = pymupdf.open(source)
        page = document.load_page(0)
        xref = page.insert_image(pymupdf.Rect(72, 100, 120, 132), filename=str(embedded_source))
        page.insert_image(pymupdf.Rect(150, 100, 198, 132), xref=xref)
        document.load_page(1).insert_image(
            pymupdf.Rect(72, 100, 120, 132),
            filename=str(embedded_source),
        )
        document.saveIncr()
        document.close()

        manifest_path = ingest_source(source, self.data_dir, pages="1", dpi=72)
        first_result = extract_pdf_images(manifest_path, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_asset = manifest["embedded_images"][0]
        extracted_path = manifest_path.parent / image_asset["path"]
        self.assertTrue(extracted_path.is_file())
        with Image.open(extracted_path) as extracted:
            self.assertEqual(extracted.size, (24, 16))
        self.assertEqual(len(image_asset["placements"]), 2)
        page_image_refs = manifest["assets"][0]["embedded_images"]
        self.assertEqual(len(page_image_refs), 2)
        self.assertTrue(all(ref["image_id"] == image_asset["image_id"] for ref in page_image_refs))

        second_result = extract_pdf_images(manifest_path, self.data_dir)
        reread = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(first_result["unique_images"], 1)
        self.assertEqual(first_result["page_placements"], 2)
        self.assertEqual(second_result["unique_images"], 1)
        self.assertEqual(len(reread["embedded_images"]), 1)

    @unittest.skipUnless(features.check("jpg_2000"), "Pillow was built without JPEG2000 support.")
    def test_normalizes_jpeg2000_pdf_images_to_png(self) -> None:
        source = self._make_pdf()
        embedded_source = self.source_dir / "embedded.jp2"
        Image.new("RGB", (28, 18), (200, 40, 10)).save(
            embedded_source,
            format="JPEG2000",
        )
        document = pymupdf.open(source)
        document.load_page(0).insert_image(
            pymupdf.Rect(72, 100, 120, 132),
            filename=str(embedded_source),
        )
        document.saveIncr()
        document.close()

        manifest_path = ingest_source(source, self.data_dir, pages="1", dpi=72)
        extract_pdf_images(manifest_path, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_asset = manifest["embedded_images"][0]
        self.assertEqual(image_asset["source_format"], "jpx")
        self.assertEqual(image_asset["format"], "png")
        self.assertEqual(image_asset["extraction_method"], "decoded_to_png")
        with Image.open(manifest_path.parent / image_asset["path"]) as extracted:
            extracted.load()
            self.assertEqual(extracted.size, (28, 18))

    def test_ingests_utf8_text_and_structured_json(self) -> None:
        text_source = self.source_dir / "lesson.txt"
        text_source.write_text("Learning content", encoding="utf-8")
        text_manifest = ingest_source(text_source, self.data_dir)
        text_data = json.loads(text_manifest.read_text(encoding="utf-8"))
        text_path = text_manifest.parent / text_data["assets"][0]["text_path"]
        self.assertEqual(text_path.read_text(encoding="utf-8"), "Learning content")

        json_source = self.source_dir / "activity.json"
        json_source.write_text('{"prompt": "Find a tree"}', encoding="utf-8")
        json_manifest = ingest_source(json_source, self.data_dir)
        json_data = json.loads(json_manifest.read_text(encoding="utf-8"))
        content_path = json_manifest.parent / json_data["assets"][0]["content_path"]
        self.assertEqual(json.loads(content_path.read_text(encoding="utf-8")), {"prompt": "Find a tree"})

    def test_ingests_raster_image_and_renders_approved_annotation(self) -> None:
        source = self.source_dir / "picture.png"
        Image.new("RGB", (100, 80), "white").save(source)
        manifest_path = ingest_source(source, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset = manifest["assets"][0]
        annotation_path = manifest_path.parent / asset["annotation_path"]
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        record["annotations"] = [{
            "object_id": "object_001",
            "label": "tree",
            "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.5},
            "confidence": 0.95,
            "status": "approved",
            "verified_by": "test-reviewer",
        }]
        annotation_path.write_text(json.dumps(record), encoding="utf-8")
        validate_annotation_file(annotation_path, self.data_dir)
        preview_path = self.data_dir / "previews" / "picture.png"
        render_annotation_preview(annotation_path, preview_path, self.data_dir)
        self.assertTrue(preview_path.is_file())
        with Image.open(preview_path) as preview:
            self.assertEqual(preview.size, (100, 80))

    def test_rejects_invalid_box(self) -> None:
        source = self.source_dir / "picture.png"
        Image.new("RGB", (100, 80), "white").save(source)
        manifest_path = ingest_source(source, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        annotation_path = manifest_path.parent / manifest["assets"][0]["annotation_path"]
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        record["annotations"] = [{
            "object_id": "object_001",
            "label": "tree",
            "bounding_box": {"x": 0.8, "y": 0.2, "width": 0.4, "height": 0.5},
            "status": "needs_review",
        }]
        annotation_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ContentEngineError):
            validate_annotation_file(annotation_path, self.data_dir)

    def test_approved_annotation_requires_reviewer(self) -> None:
        source = self.source_dir / "picture.png"
        Image.new("RGB", (100, 80), "white").save(source)
        manifest_path = ingest_source(source, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        annotation_path = manifest_path.parent / manifest["assets"][0]["annotation_path"]
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        record["annotations"] = [{
            "object_id": "object_001",
            "label": "tree",
            "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.5},
            "status": "approved",
        }]
        annotation_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ContentEngineError):
            validate_annotation_file(annotation_path, self.data_dir)

    def test_detection_converts_boxes_to_normalized_candidate_annotations(self) -> None:
        source = self.source_dir / "picture.png"
        Image.new("RGB", (100, 80), "white").save(source)
        manifest_path = ingest_source(source, self.data_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        annotation_path = manifest_path.parent / manifest["assets"][0]["annotation_path"]

        class FakeBoxes:
            xyxy = [[10, 8, 50, 48]]
            conf = [0.9]
            cls = [0]

        class FakePrediction:
            boxes = FakeBoxes()
            names = {0: "tree"}

        class FakeDetector:
            def __init__(self, weights: str) -> None:
                self.weights = weights

            def predict(self, image_path: str, conf: float) -> FakePrediction:
                self.assertion_target.assertTrue(Path(image_path).is_file())
                self.assertion_target.assertEqual(conf, 0.25)
                return FakePrediction()

        FakeDetector.assertion_target = self
        result = run_detection(
            annotation_path,
            self.data_dir,
            detector_factory=FakeDetector,
        )
        record = validate_annotation_file(annotation_path, self.data_dir)
        self.assertEqual(result["detections_added"], 1)
        self.assertEqual(record["annotations"][0]["label"], "tree")
        self.assertEqual(
            record["annotations"][0]["bounding_box"],
            {"x": 0.1, "y": 0.1, "width": 0.4, "height": 0.5},
        )
        self.assertEqual(record["annotations"][0]["status"], "needs_review")
        run_detection(
            annotation_path,
            self.data_dir,
            detector_factory=FakeDetector,
        )
        reread = validate_annotation_file(annotation_path, self.data_dir)
        self.assertEqual(len(reread["annotations"]), 1)

    def test_rejects_svg_scripts_and_accepts_basic_svg(self) -> None:
        safe_svg = self.source_dir / "diagram.svg"
        safe_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">'
            "<title>Simple diagram</title><rect width=\"100\" height=\"50\" /></svg>",
            encoding="utf-8",
        )
        manifest_path = ingest_source(safe_svg, self.data_dir)
        asset = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"][0]
        self.assertEqual(asset["svg"]["title"], "Simple diagram")
        self.assertFalse(asset["svg"]["raster_preview_available"])

        unsafe_svg = self.source_dir / "unsafe.svg"
        unsafe_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            encoding="utf-8",
        )
        with self.assertRaises(ContentEngineError):
            ingest_source(unsafe_svg, self.data_dir)

        unsafe_css = self.source_dir / "unsafe_css.svg"
        unsafe_css.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><style>'
            "rect { fill: url(https://example.invalid/paint); }"
            "</style><rect width=\"10\" height=\"10\" /></svg>",
            encoding="utf-8",
        )
        with self.assertRaises(ContentEngineError):
            ingest_source(unsafe_css, self.data_dir)

        unsupported_svg = self.source_dir / "unsupported.svg"
        unsupported_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><image href="#picture" /></svg>',
            encoding="utf-8",
        )
        with self.assertRaises(ContentEngineError):
            ingest_source(unsupported_svg, self.data_dir)

    def test_rejects_out_of_range_pdf_pages(self) -> None:
        with self.assertRaises(ContentEngineError):
            ingest_source(self._make_pdf(), self.data_dir, pages="3")


if __name__ == "__main__":
    unittest.main()
