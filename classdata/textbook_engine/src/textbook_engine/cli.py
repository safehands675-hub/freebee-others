from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .benchmark import benchmark_processed_pdfs
from .book_pipeline import DEFAULT_BOOKS_DIR, index_book_printed_pages, process_book
from .chapters import DEFAULT_OUTPUT_DIR, process_chapter
from .core import (
    ContentEngineError,
    extract_pdf_images,
    ingest_source,
    render_annotation_preview,
    run_detection,
    validate_annotation_file,
)
from .printed_pages import index_printed_pages
from .pipeline import process_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-engine",
        description="Import educational content and validate reviewed image annotations.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Import a PDF, image, SVG, text, or JSON file.")
    ingest.add_argument("source", type=Path, help="Source file; left unchanged.")
    ingest.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ingest.add_argument(
        "--pages",
        help="PDF pages, for example 1,3-5. Omit to process every PDF page.",
    )
    ingest.add_argument("--dpi", type=int, default=120, help="PDF render resolution (36-600, default: 120).")

    process = commands.add_parser(
        "process-pdf",
        help="Automatically extract, OCR, index, and catalog an uploaded PDF.",
    )
    process.add_argument("source", type=Path, help="PDF to process; the original is left unchanged.")
    process.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    process.add_argument("--dpi", type=int, default=120, help="PDF render resolution (36-600, default: 120).")
    process.add_argument(
        "--class-number",
        type=int,
        choices=range(1, 9),
        help="Optional class override when the engine cannot infer it from the PDF.",
    )
    process.add_argument(
        "--language",
        help="Optional OCR language override: English, Telugu, Hindi, Sanskrit, or Urdu.",
    )

    chapter = commands.add_parser(
        "process-chapter",
        help="Extract and explain one explicitly selected PDF chapter with Gemini.",
    )
    chapter.add_argument("source", type=Path, help="Textbook PDF; the original is left unchanged.")
    chapter.add_argument("--chapter", type=int, required=True, help="Chapter, unit, or lesson number.")
    chapter.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    chapter.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    chapter.add_argument("--language", help="Optional OCR language override.")
    chapter.add_argument("--model", help="Optional Gemini model override; defaults to GEMINI_MODEL.")

    book = commands.add_parser(
        "process-book",
        help="Extract every PDF page, explain detected chapters with Gemini, and publish a versioned package.",
    )
    book.add_argument("source", type=Path, help="Textbook PDF; the original is left unchanged.")
    book.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    book.add_argument("--books-dir", type=Path, default=DEFAULT_BOOKS_DIR)
    book.add_argument(
        "--class-number",
        type=int,
        choices=range(1, 11),
        help="Optional class override when the engine cannot infer it from the PDF filename.",
    )
    book.add_argument("--subject", help="Optional subject override.")
    book.add_argument("--language", help="Optional language label for the source text.")
    book.add_argument("--model", help="Optional Gemini model override; defaults to GEMINI_MODEL.")

    page_index = commands.add_parser(
        "index-book-pages",
        help="Use local Tesseract OCR to map printed pages in a published book package.",
    )
    page_index.add_argument("book_id", help="Book ID returned by process-book.")
    page_index.add_argument("version_id", help="Version ID returned by process-book.")
    page_index.add_argument("--books-dir", type=Path, default=DEFAULT_BOOKS_DIR)
    page_index.add_argument(
        "--minimum-confidence",
        type=float,
        default=55.0,
        help="Minimum Tesseract confidence for direct page-number matches (0-100).",
    )

    benchmark = commands.add_parser(
        "benchmark-pdfs",
        help="Measure extraction coverage, OCR outcomes, printed-page mapping, and chapter discovery.",
    )
    benchmark.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    benchmark.add_argument(
        "--ground-truth",
        type=Path,
        help="Optional JSON labels for page text and chapter start pages.",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; otherwise print the report.",
    )

    extract_images = commands.add_parser(
        "extract-images",
        help="Extract embedded images from an imported PDF package.",
    )
    extract_images.add_argument("manifest", type=Path, help="Imported PDF manifest.json.")
    extract_images.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    printed_pages = commands.add_parser(
        "index-printed-pages",
        help="Use local Tesseract OCR to map printed textbook pages to PDF pages.",
    )
    printed_pages.add_argument("manifest", type=Path, help="Imported PDF manifest.json.")
    printed_pages.add_argument(
        "--minimum-confidence",
        type=float,
        default=55.0,
        help="Minimum Tesseract OCR confidence for direct page-number matches (0-100).",
    )

    validate = commands.add_parser("validate", help="Validate an annotation JSON and referenced image.")
    validate.add_argument("annotations", type=Path)
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    detect = commands.add_parser("detect", help="Add LibreYOLO candidate object boxes to annotation JSON.")
    detect.add_argument("annotations", type=Path)
    detect.add_argument("--weights", default="LibreYOLO9t.pt")
    detect.add_argument("--confidence", type=float, default=0.25)
    detect.add_argument("--labels", help="Optional comma-separated class names to keep.")
    detect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    preview = commands.add_parser("preview", help="Render a Pillow overlay preview for annotations.")
    preview.add_argument("annotations", type=Path)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            manifest_path = ingest_source(args.source, args.data_dir, args.pages, args.dpi)
            result = {"status": "ok", "manifest": str(manifest_path)}
        elif args.command == "process-pdf":
            result = process_pdf(
                args.source,
                args.data_dir,
                dpi=args.dpi,
                class_number=args.class_number,
                language=args.language,
            )
        elif args.command == "process-chapter":
            result = process_chapter(
                args.source,
                args.chapter,
                args.output_dir,
                data_dir=args.data_dir,
                language=args.language,
                model=args.model,
            )
        elif args.command == "process-book":
            result = process_book(
                args.source,
                books_dir=args.books_dir,
                data_dir=args.data_dir,
                class_number=args.class_number,
                language=args.language,
                subject=args.subject,
                model=args.model,
            )
        elif args.command == "index-book-pages":
            result = index_book_printed_pages(
                args.book_id,
                args.version_id,
                books_dir=args.books_dir,
                minimum_confidence=args.minimum_confidence,
            )
        elif args.command == "benchmark-pdfs":
            result = benchmark_processed_pdfs(
                args.data_dir,
                ground_truth_path=args.ground_truth,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                result = {
                    "status": "complete",
                    "output": str(args.output.resolve()),
                    "corpus_summary": result["corpus_summary"],
                    "measurements": result["measurements"],
                }
        elif args.command == "extract-images":
            result = {
                "status": "ok",
                **extract_pdf_images(args.manifest, args.data_dir),
            }
        elif args.command == "index-printed-pages":
            result = index_printed_pages(
                args.manifest,
                minimum_confidence=args.minimum_confidence,
            )
        elif args.command == "validate":
            record = validate_annotation_file(args.annotations, args.data_dir)
            result = {
                "status": "valid",
                "asset_id": record["asset_id"],
                "annotation_count": len(record["annotations"]),
            }
        elif args.command == "detect":
            labels = (
                {label.strip() for label in args.labels.split(",") if label.strip()}
                if args.labels
                else None
            )
            result = {
                "status": "ok",
                **run_detection(
                    args.annotations,
                    args.data_dir,
                    args.weights,
                    args.confidence,
                    labels,
                ),
            }
        else:
            output_path = render_annotation_preview(args.annotations, args.output, args.data_dir)
            result = {"status": "ok", "preview": str(output_path)}
    except ContentEngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: file operation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
