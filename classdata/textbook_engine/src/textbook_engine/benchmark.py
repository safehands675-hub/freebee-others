from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .book_pipeline import _discover_chapters
from .core import ContentEngineError
from .page_index import _usable_page_text
from .pipeline import is_current_pdf_package


def _edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def _text_error_counts(actual: str, expected: str) -> tuple[int, int, int, int]:
    actual_characters = list(" ".join(actual.split()).casefold())
    expected_characters = list(" ".join(expected.split()).casefold())
    actual_words = " ".join(actual.split()).casefold().split()
    expected_words = " ".join(expected.split()).casefold().split()
    return (
        _edit_distance(actual_characters, expected_characters),
        len(expected_characters),
        _edit_distance(actual_words, expected_words),
        len(expected_words),
    )


def _load_ground_truth(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(f"Could not read benchmark ground truth {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("documents"), dict):
        raise ContentEngineError(
            "Benchmark ground truth must be an object with a 'documents' mapping."
        )
    return value["documents"]


def benchmark_processed_pdfs(
    data_dir: Path,
    *,
    ground_truth_path: Path | None = None,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    source_root = data_dir / "sources"
    if not source_root.is_dir():
        raise ContentEngineError(f"Processed source directory does not exist: {source_root}")
    ground_truth = _load_ground_truth(ground_truth_path)
    records: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    total_expected_pages = 0
    total_indexed_pages = 0
    total_pages_with_text = 0
    total_pages_with_usable_text = 0
    total_pages_needing_ocr = 0
    total_ocr_review_pages = 0
    total_printed_matched = 0
    total_printed_unresolved = 0
    total_cer_edits = 0
    total_cer_characters = 0
    total_wer_edits = 0
    total_wer_words = 0
    chapter_boundary_true_positive = 0
    chapter_boundary_false_positive = 0
    chapter_boundary_false_negative = 0
    chapter_ground_truth_documents = 0

    for manifest_path in sorted(source_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            unresolved.append({
                "manifest": str(manifest_path),
                "failure": f"manifest_read_error: {exc}",
            })
            continue
        if not isinstance(manifest, dict):
            unresolved.append({
                "manifest": str(manifest_path),
                "failure": "manifest_not_object",
            })
            continue
        source_id = manifest.get("source_id")
        source = manifest.get("source")
        processing = manifest.get("processing")
        if not isinstance(source_id, str) or not isinstance(source, dict):
            unresolved.append({
                "manifest": str(manifest_path),
                "failure": "invalid_source_metadata",
            })
            continue
        if not is_current_pdf_package(manifest_path):
            unresolved.append({
                "source_id": source_id,
                "failure": "stale_or_invalid_processed_package",
            })
            continue

        page_count = source["pdf_page_count"]
        assets = [
            asset
            for asset in manifest["assets"]
            if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
        ]
        package_dir = manifest_path.parent
        page_texts: dict[int, str] = {}
        extraction_methods: dict[str, int] = {}
        needs_review = 0
        needs_ocr = 0
        page_records: list[dict[str, Any]] = []
        for asset in assets:
            page_number = asset["page"]
            text = (package_dir / asset["text_path"]).read_text(encoding="utf-8")
            page_texts[page_number] = text
            usable_text = _usable_page_text(text)
            method = asset.get("text_extraction_method", "unknown")
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            needs_review += int(bool(asset.get("needs_ocr_review")))
            needs_ocr += int(method != "native_pdf")
            page_records.append({
                "pdf_page": page_number,
                "text_characters": len(text),
                "has_text": bool(text.strip()),
                "has_usable_text": bool(usable_text.strip()),
                "text_extraction_method": method,
                "needs_ocr_review": bool(asset.get("needs_ocr_review")),
                "has_page_image": (
                    package_dir / asset["image_path"]
                ).is_file(),
            })

        index_path = manifest_path.parent / "printed_page_index.json"
        try:
            printed_index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            unresolved.append({
                "source_id": source_id,
                "failure": f"printed_page_index_read_error: {exc}",
            })
            printed_index = {}
        printed_pages = (
            printed_index.get("pages", [])
            if isinstance(printed_index, dict)
            and printed_index.get("source_id") == source_id
            and printed_index.get("source_sha256") == source.get("sha256")
            and isinstance(printed_index.get("pages"), list)
            else []
        )
        matched_printed_pdf_pages = {
            entry["pdf_page"]
            for entry in printed_pages
            if isinstance(entry, dict)
            and entry.get("status") == "matched"
            and isinstance(entry.get("printed_page"), int)
            and isinstance(entry.get("pdf_page"), int)
            and not isinstance(entry.get("pdf_page"), bool)
        }
        unresolved_printed_pages = max(
            0,
            page_count - len(matched_printed_pdf_pages),
        )
        try:
            chapters = _discover_chapters([
                page_texts.get(page_number, "")
                for page_number in range(1, page_count + 1)
            ])
            detected_boundaries = sorted(
                chapter.start_page + 1
                for chapter in chapters
                if chapter.start_page < page_count
            )
            chapter_detection_error = None
        except ContentEngineError as exc:
            detected_boundaries = []
            chapter_detection_error = str(exc)
            unresolved.append({
                "source_id": source_id,
                "failure": f"chapter_detection_unresolved: {exc}",
            })

        truth = ground_truth.get(source_id)
        text_gold = truth.get("pages", {}) if isinstance(truth, dict) else {}
        if isinstance(text_gold, dict):
            for page_key, expected in text_gold.items():
                try:
                    page_number = int(page_key)
                except (TypeError, ValueError):
                    continue
                actual = page_texts.get(page_number)
                if isinstance(actual, str) and isinstance(expected, str):
                    cer_edits, cer_characters, wer_edits, wer_words = (
                        _text_error_counts(actual, expected)
                    )
                    total_cer_edits += cer_edits
                    total_cer_characters += cer_characters
                    total_wer_edits += wer_edits
                    total_wer_words += wer_words
        expected_boundaries = (
            truth.get("chapter_start_pdf_pages")
            if isinstance(truth, dict)
            else None
        )
        if isinstance(expected_boundaries, list):
            expected_set = {
                page for page in expected_boundaries
                if isinstance(page, int) and not isinstance(page, bool) and page > 0
            }
            detected_set = set(detected_boundaries)
            chapter_boundary_true_positive += len(expected_set & detected_set)
            chapter_boundary_false_positive += len(detected_set - expected_set)
            chapter_boundary_false_negative += len(expected_set - detected_set)
            chapter_ground_truth_documents += 1

        total_expected_pages += page_count
        total_indexed_pages += len(assets)
        total_pages_with_text += sum(bool(text.strip()) for text in page_texts.values())
        total_pages_with_usable_text += sum(
            bool(_usable_page_text(text).strip()) for text in page_texts.values()
        )
        total_pages_needing_ocr += needs_ocr
        total_ocr_review_pages += needs_review
        total_printed_matched += len(matched_printed_pdf_pages)
        total_printed_unresolved += unresolved_printed_pages
        records.append({
            "source_id": source_id,
            "filename": source.get("filename"),
            "source_sha256": source.get("sha256"),
            "pipeline_version": (
                processing.get("pipeline_version")
                if isinstance(processing, dict)
                else None
            ),
            "physical_pdf_pages": page_count,
            "indexed_pages": len(assets),
            "pages_with_text": sum(bool(text.strip()) for text in page_texts.values()),
            "extraction_methods": extraction_methods,
            "pages_needing_ocr": needs_ocr,
            "pages_needing_ocr_review": needs_review,
            "printed_pages_matched": len(matched_printed_pdf_pages),
            "printed_pages_unresolved": unresolved_printed_pages,
            "chapter_boundaries_detected": detected_boundaries,
            "chapter_detection_error": chapter_detection_error,
            "pages": page_records,
        })

    return {
        "benchmark_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "corpus_summary": {
            "documents_measured": len(records),
            "expected_physical_pages": total_expected_pages,
            "indexed_pages": total_indexed_pages,
            "page_coverage": (
                total_indexed_pages / total_expected_pages
                if total_expected_pages
                else None
            ),
            "pages_with_text": total_pages_with_text,
            "pages_with_usable_text": total_pages_with_usable_text,
            "text_coverage": (
                total_pages_with_text / total_expected_pages
                if total_expected_pages
                else None
            ),
            "usable_text_coverage": (
                total_pages_with_usable_text / total_expected_pages
                if total_expected_pages
                else None
            ),
            "pages_needing_ocr": total_pages_needing_ocr,
            "pages_needing_ocr_review": total_ocr_review_pages,
            "printed_pages_matched": total_printed_matched,
            "printed_pages_unresolved": total_printed_unresolved,
            "cer": (
                total_cer_edits / total_cer_characters
                if total_cer_characters
                else None
            ),
            "wer": (
                total_wer_edits / total_wer_words
                if total_wer_words
                else None
            ),
            "chapter_boundary_precision": (
                chapter_boundary_true_positive
                / (chapter_boundary_true_positive + chapter_boundary_false_positive)
                if chapter_ground_truth_documents
                and chapter_boundary_true_positive + chapter_boundary_false_positive
                else None
            ),
            "chapter_boundary_recall": (
                chapter_boundary_true_positive
                / (chapter_boundary_true_positive + chapter_boundary_false_negative)
                if chapter_ground_truth_documents
                and chapter_boundary_true_positive + chapter_boundary_false_negative
                else None
            ),
            "ground_truth_documents": chapter_ground_truth_documents,
            "unresolved_failures": len(unresolved),
        },
        "measurements": {
            "coverage_and_processing_counts": (
                "measured"
                if records
                else "not measured: no current valid processed packages"
            ),
            "usable_text_coverage": (
                "measured"
                if records
                else "not measured: no current valid processed packages"
            ),
            "cer_wer": "measured" if total_cer_characters else "not measured: no page text ground truth",
            "chapter_boundary_accuracy": (
                "measured"
                if chapter_ground_truth_documents
                else "not measured: no labeled chapter boundaries"
            ),
            "visual_semantic_accuracy": "not measured: requires manually labeled questions and images",
        },
        "documents": records,
        "unresolved_failures": unresolved,
    }
