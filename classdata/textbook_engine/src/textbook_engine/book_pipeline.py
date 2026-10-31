from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .book_store import BookStore, FileSystemBookStore
from .chapters import (
    Completion,
    _class_from_filename,
    _generate_chapter_content,
    _is_contents_page,
    _load_gemini_api_key,
    _markers_in_text,
    _slug,
    _subject_from_filename,
    _title_from_page,
)
from .core import ContentEngineError, sha256_file
from .gemini import CHAPTER_PROMPT_VERSION, DEFAULT_GEMINI_MODEL
from .printed_pages import INDEX_FILENAME, index_page_images

DEFAULT_BOOKS_DIR = Path(__file__).resolve().parents[2] / "data" / "books"
DEFAULT_DATA_DIR = DEFAULT_BOOKS_DIR.parent
_MODEL_VERSION = "chapter-package-v3"
PAGE_IMAGE_DPI = 120
PAGE_IMAGE_QUALITY = 78
_PAGE_TEXT_OCR_INDEX = "page_text_ocr.json"
_PAGE_TEXT_OCR_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ChapterSpan:
    chapter_id: str
    number: int | None
    marker_kind: str | None
    title: str
    start_page: int
    end_page: int


def _discover_chapters(page_texts: list[str]) -> list[_ChapterSpan]:
    markers_by_page = [
        [] if _is_contents_page(text, markers) else markers
        for text in page_texts
        for markers in [_markers_in_text(text)]
    ]
    marker_counts = {
        kind: sum(
            marker["kind"] == kind
            for markers in markers_by_page
            for marker in markers
        )
        for kind in ("chapter", "unit", "lesson")
    }
    marker_kind = max(
        marker_counts,
        key=lambda kind: (marker_counts[kind], ("chapter", "unit", "lesson").index(kind)),
    )
    if marker_counts[marker_kind] == 0:
        raise ContentEngineError(
            "Could not confidently identify chapter boundaries. The PDF needs explicit "
            "CHAPTER, UNIT, or LESSON headings."
        )

    candidates: list[tuple[int, dict[str, Any]]] = [
        (page_number, marker)
        for page_number, markers in enumerate(markers_by_page)
        for marker in markers
        if marker["kind"] == marker_kind
    ]
    candidates.sort(key=lambda item: item[0])
    unique_candidates: list[tuple[int, dict[str, Any]]] = []
    seen_numbers: set[int] = set()
    seen_pages: set[int] = set()
    for page_number, marker in candidates:
        number = marker["number"]
        if number < 1:
            raise ContentEngineError(
                f"Detected an invalid {marker_kind.upper()} number on physical PDF "
                f"page {page_number + 1}."
            )
        if page_number in seen_pages:
            raise ContentEngineError(
                f"Multiple {marker_kind.upper()} headings were found on physical PDF page "
                f"{page_number + 1}; review the source before processing."
            )
        if number in seen_numbers:
            raise ContentEngineError(
                f"Duplicate {marker_kind.upper()} {number} headings were found; "
                "review chapter boundaries before processing."
            )
        seen_pages.add(page_number)
        seen_numbers.add(number)
        unique_candidates.append((page_number, marker))

    spans: list[_ChapterSpan] = []
    if unique_candidates[0][0] > 0:
        spans.append(_ChapterSpan(
            "front_matter",
            None,
            None,
            "Front matter",
            0,
            unique_candidates[0][0],
        ))
    for index, (start_page, marker) in enumerate(unique_candidates):
        end_page = (
            unique_candidates[index + 1][0]
            if index + 1 < len(unique_candidates)
            else len(page_texts)
        )
        title = marker["title"] or _title_from_page(page_texts[start_page])
        number = marker["number"]
        spans.append(_ChapterSpan(
            f"{marker_kind}_{number:02d}",
            number,
            marker_kind,
            title or f"{marker_kind.title()} {number}",
            start_page,
            end_page,
        ))
    return spans


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _page_record(
    source: Path,
    page_number: int,
    raw_text: str,
    image_path: str,
    text_extraction_method: str,
    needs_ocr_review: bool,
) -> dict[str, Any]:
    text_status = (
        "empty_native_text"
        if not raw_text.strip()
        else (
            "ocr_extracted"
            if text_extraction_method == "tesseract_ocr"
            else "extracted"
        )
    )
    quality_notes = (
        ["Text extraction needs review."]
        if needs_ocr_review
        else []
    )
    return {
        "page_id": f"pdf_{page_number:04d}",
        "source_provenance": {
            "source_pdf_filename": source.name,
            "source_pdf_page_number_1_indexed": page_number,
            "printed_page_number": None,
            "text_extraction_method": text_extraction_method,
        },
        "page_heading": None,
        "source_text": {
            "extracted_text_raw_ordered": raw_text,
            "extracted_text_normalized_for_search": _normalized_text(raw_text),
        },
        "questions_and_activities": [],
        "tables": [],
        "page_explanation": None,
        "visuals": {"page_image": image_path, "items": []},
        "quality_control": {
            "text_status": text_status,
            "text_quality_notes": quality_notes,
            "visual_status": "not_processed",
            "review_status": "not_human_reviewed",
        },
    }


def process_book(
    source: Path,
    *,
    books_dir: Path = DEFAULT_BOOKS_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    class_number: int | None = None,
    language: str | None = None,
    subject: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    completion: Completion | None = None,
    store: BookStore | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if source.suffix.casefold() != ".pdf":
        raise ContentEngineError("Book processing accepts PDF files only.")
    if not source.is_file():
        raise ContentEngineError(f"Source PDF does not exist: {source}")
    resolved_class = class_number if class_number is not None else _class_from_filename(source)
    if (
        isinstance(resolved_class, bool)
        or not isinstance(resolved_class, int)
        or not 1 <= resolved_class <= 10
    ):
        raise ContentEngineError(
            "A class number from 1 to 10 is required; provide --class-number when it "
            "cannot be inferred from the PDF filename."
        )
    resolved_subject = (subject or _subject_from_filename(source)).strip()
    if not resolved_subject:
        raise ContentEngineError("Book subject must not be empty.")
    source_language = language.strip() if language and language.strip() else None
    from .pipeline import PIPELINE_VERSION, _ocr_language

    ocr_language = _ocr_language(source.name, source_language)
    model = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    if not model:
        raise ContentEngineError("Gemini model ID must not be empty.")

    source_hash = sha256_file(source)
    version_identity = {
        "source_sha256": source_hash,
        "class": resolved_class,
        "subject": resolved_subject,
        "language": source_language,
        "ocr_language": ocr_language,
        "pdf_pipeline_version": PIPELINE_VERSION,
        "model_id": model,
        "prompt_version": CHAPTER_PROMPT_VERSION,
        "package_version": _MODEL_VERSION,
    }
    subject_slug = _slug(resolved_subject)[:80].rstrip("_") or "unknown_subject"
    book_id = f"class_{resolved_class:02d}_{subject_slug}_{source_hash[:16]}"
    version_fingerprint = hashlib.sha256(
        json.dumps(version_identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    version_id = f"v_{version_fingerprint[:24]}"
    book_store = store or FileSystemBookStore(books_dir)

    from .pipeline import process_and_load_pdf_pages

    canonical_manifest, processed_pages = process_and_load_pdf_pages(
        source,
        data_dir,
        dpi=PAGE_IMAGE_DPI,
        class_number=resolved_class if resolved_class <= 8 else None,
        language=ocr_language,
    )
    if book_store.contains(book_id, version_id):
        manifest = book_store.read_manifest(book_id, version_id)
        return {
            "status": "already_published",
            "book_id": book_id,
            "version_id": version_id,
            "package": str(book_store.package_path(book_id, version_id)),
            "chapters": len(manifest["chapters"]),
            "physical_pdf_pages": manifest["source_pdf"]["physical_page_count"],
        }
    page_texts = [page["text"] for page in processed_pages]
    if not page_texts:
        raise ContentEngineError("The PDF has no physical pages.")
    canonical_source = canonical_manifest.get("source")
    canonical_language = (
        canonical_source.get("ocr_language")
        if isinstance(canonical_source, dict)
        else None
    )
    if not isinstance(canonical_language, str):
        raise ContentEngineError(
            "The processed PDF manifest does not contain its OCR language."
        )
    spans = _discover_chapters(page_texts)
    if completion is None:
        api_key = api_key.strip() if api_key and api_key.strip() else _load_gemini_api_key()

    with tempfile.TemporaryDirectory(prefix="book-page-assets-") as staging_name:
        staging_directory = Path(staging_name)
        image_assets: dict[str, Path] = {}
        for page in processed_pages:
            page_number = page["page"]
            asset_name = f"images/pages/pdf_page_{page_number:04d}.jpg"
            staging_path = staging_directory / f"pdf_page_{page_number:04d}.jpg"
            try:
                with Image.open(page["image_path"]) as image:
                    image.convert("RGB").save(
                        staging_path,
                        format="JPEG",
                        quality=PAGE_IMAGE_QUALITY,
                        optimize=True,
                    )
            except (OSError, Image.DecompressionBombError) as exc:
                raise ContentEngineError(
                    f"Could not prepare processed PDF page image {page_number}: {exc}"
                ) from exc
            image_assets[asset_name] = staging_path

        if sha256_file(source) != source_hash:
            raise ContentEngineError(
                "The source PDF changed during extraction; no book version was published."
            )
        return _publish_book_package(
            source=source,
            source_hash=source_hash,
            book_id=book_id,
            version_id=version_id,
            class_number=resolved_class,
            subject=resolved_subject,
            language=source_language,
            ocr_language=canonical_language,
            pdf_pipeline_version=PIPELINE_VERSION,
            model=model,
            api_key=api_key,
            completion=completion,
            spans=spans,
            page_texts=page_texts,
            page_metadata=processed_pages,
            image_assets=image_assets,
            store=book_store,
        )


def index_book_printed_pages(
    book_id: str,
    version_id: str,
    *,
    books_dir: Path = DEFAULT_BOOKS_DIR,
    minimum_confidence: float = 55.0,
    store: BookStore | None = None,
) -> dict[str, Any]:
    book_store = store if store is not None else FileSystemBookStore(books_dir)
    manifest = book_store.read_manifest(book_id, version_id)
    source_id = hashlib.sha256(
        f"{book_id}:{version_id}".encode("utf-8")
    ).hexdigest()[:16]
    source_pdf = manifest.get("source_pdf")
    chapters = manifest.get("chapters")
    if not isinstance(source_pdf, dict) or not isinstance(chapters, list):
        raise ContentEngineError(
            f"Book manifest is missing source or chapter metadata: {book_id}/{version_id}."
        )
    image_pages: list[tuple[int, Path]] = []
    for chapter_entry in chapters:
        if not isinstance(chapter_entry, dict) or not isinstance(
            chapter_entry.get("chapter_id"),
            str,
        ):
            raise ContentEngineError(
                f"Book manifest has invalid chapter metadata: {book_id}/{version_id}."
            )
        chapter = book_store.read_chapter(
            book_id,
            version_id,
            chapter_entry["chapter_id"],
        )
        pages = chapter.get("pages")
        if not isinstance(pages, list):
            raise ContentEngineError(
                f"Book chapter {chapter_entry['chapter_id']} has no page records."
            )
        for page in pages:
            if not isinstance(page, dict):
                raise ContentEngineError(
                    f"Book chapter {chapter_entry['chapter_id']} contains an invalid page."
                )
            provenance = page.get("source_provenance")
            visuals = page.get("visuals")
            pdf_page = (
                provenance.get("source_pdf_page_number_1_indexed")
                if isinstance(provenance, dict)
                else None
            )
            image_reference = visuals.get("page_image") if isinstance(visuals, dict) else None
            if not isinstance(image_reference, str):
                raise ContentEngineError(
                    f"Book page {pdf_page} has no rendered page image for OCR."
                )
            image_pages.append((
                pdf_page,
                book_store.asset_path(book_id, version_id, image_reference),
            ))
    image_pages.sort(key=lambda item: item[0])
    index_path = (
        Path(books_dir).expanduser().resolve()
        / "_indexes"
        / book_id
        / version_id
        / INDEX_FILENAME
    )
    return index_page_images(
        image_pages,
        output_path=index_path,
        source_id=source_id,
        source_manifest="book.json",
        source_sha256=source_pdf.get("sha256"),
        minimum_confidence=minimum_confidence,
    )


def ocr_book_page_text(
    book_id: str,
    version_id: str,
    *,
    pdf_page: int | None = None,
    books_dir: Path = DEFAULT_BOOKS_DIR,
    store: BookStore | None = None,
) -> dict[int, str]:
    if pdf_page is not None and (
        isinstance(pdf_page, bool)
        or not isinstance(pdf_page, int)
        or pdf_page < 1
    ):
        raise ContentEngineError("Physical PDF page must be a positive integer.")
    book_store = store if store is not None else FileSystemBookStore(books_dir)
    manifest = book_store.read_manifest(book_id, version_id)
    source_pdf = manifest.get("source_pdf")
    chapters = manifest.get("chapters")
    if (
        not isinstance(source_pdf, dict)
        or not isinstance(source_pdf.get("sha256"), str)
        or not isinstance(chapters, list)
    ):
        raise ContentEngineError(
            f"Book manifest is missing source or chapter metadata: {book_id}/{version_id}."
        )
    source_hash = source_pdf["sha256"]
    source_id = hashlib.sha256(f"{book_id}:{version_id}".encode("utf-8")).hexdigest()[:16]
    index_path = (
        Path(books_dir).expanduser().resolve()
        / "_indexes"
        / book_id
        / version_id
        / _PAGE_TEXT_OCR_INDEX
    )

    with _PAGE_TEXT_OCR_LOCK:
        cache: dict[str, Any] = {
            "schema_version": 2,
            "source_id": source_id,
            "source_sha256": source_hash,
            "pages": {},
        }
        if index_path.is_file():
            try:
                cached_value = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContentEngineError(
                    f"Could not read OCR text cache {index_path}: {exc}"
                ) from exc
            if (
                not isinstance(cached_value, dict)
                or cached_value.get("source_id") != source_id
                or cached_value.get("source_sha256") != source_hash
                or not isinstance(cached_value.get("pages"), dict)
            ):
                raise ContentEngineError(
                    f"OCR text cache does not match book version {book_id}/{version_id}."
                )
            if cached_value.get("schema_version") == 2:
                cache = cached_value
            elif cached_value.get("schema_version") != 1:
                raise ContentEngineError(
                    f"OCR text cache does not match book version {book_id}/{version_id}."
                )

        pages_cache: dict[str, Any] = cache["pages"]
        page_records: dict[int, dict[str, Any]] = {}
        for chapter_entry in chapters:
            if not isinstance(chapter_entry, dict) or not isinstance(
                chapter_entry.get("chapter_id"),
                str,
            ):
                raise ContentEngineError(
                    f"Book manifest has invalid chapter metadata: {book_id}/{version_id}."
                )
            chapter = book_store.read_chapter(
                book_id,
                version_id,
                chapter_entry["chapter_id"],
            )
            pages = chapter.get("pages")
            if not isinstance(pages, list):
                raise ContentEngineError(
                    f"Book chapter {chapter_entry['chapter_id']} has no page records."
                )
            for page in pages:
                if not isinstance(page, dict):
                    raise ContentEngineError(
                        f"Book chapter {chapter_entry['chapter_id']} has an invalid page."
                    )
                provenance = page.get("source_provenance")
                page_number = (
                    provenance.get("source_pdf_page_number_1_indexed")
                    if isinstance(provenance, dict)
                    else None
                )
                if isinstance(page_number, bool) or not isinstance(page_number, int):
                    raise ContentEngineError(
                        f"Book chapter {chapter_entry['chapter_id']} has an invalid PDF page."
                    )
                if pdf_page is not None and page_number != pdf_page:
                    continue
                page_records[page_number] = page
        if pdf_page is not None and pdf_page not in page_records:
            raise ContentEngineError(
                f"PDF page {pdf_page} is not present in book {book_id}/{version_id}."
            )

        from .pipeline import (
            _has_expected_script,
            _ocr_language,
            _ocr_page,
            _requires_ocr,
        )
        from .ocr import resolve_tesseract_command, resolve_tesseract_languages

        explicit_language = source_pdf.get("ocr_language", manifest.get("language"))
        language = _ocr_language(
            str(source_pdf.get("filename", "")),
            explicit_language if isinstance(explicit_language, str) else None,
        )
        selected_pages: dict[int, dict[str, Any]] = {}
        for page_number, page in page_records.items():
            cached_page = pages_cache.get(str(page_number))
            if isinstance(cached_page, dict) and isinstance(cached_page.get("text"), str):
                continue
            source_text = page.get("source_text")
            native_text = (
                source_text.get("extracted_text_raw_ordered", "")
                if isinstance(source_text, dict)
                else ""
            )
            if not isinstance(native_text, str):
                native_text = ""
            if _requires_ocr(native_text, language):
                selected_pages[page_number] = page

        if selected_pages:
            executable = resolve_tesseract_command()
            available_languages = resolve_tesseract_languages(executable)
            if language not in available_languages:
                raise ContentEngineError(
                    f"OCR language '{language}' is not installed for {source_pdf.get('filename')}. "
                    f"Install the matching Tesseract traineddata and retry."
                )
            changed = False
            for page_number, page in selected_pages.items():
                visuals = page.get("visuals")
                image_reference = (
                    visuals.get("page_image")
                    if isinstance(visuals, dict)
                    else None
                )
                if not isinstance(image_reference, str):
                    raise ContentEngineError(
                        f"PDF page {page_number} has no page image for OCR."
                    )
                image_path = book_store.asset_path(
                    book_id,
                    version_id,
                    image_reference,
                )
                source_text = page.get("source_text")
                native_text = (
                    source_text.get("extracted_text_raw_ordered", "")
                    if isinstance(source_text, dict)
                    else ""
                )
                if not isinstance(native_text, str):
                    native_text = ""
                recognized_text = _ocr_page(executable, image_path, language)
                recognized_script = _has_expected_script(recognized_text, language)
                native_script = _has_expected_script(native_text, language)
                result_text = (
                    recognized_text
                    if len(recognized_text) > len(native_text.strip())
                    or (not native_script and recognized_script)
                    else native_text
                )
                pages_cache[str(page_number)] = {
                    "text": result_text,
                    "method": (
                        "tesseract_ocr"
                        if result_text == recognized_text and recognized_text
                        else "native_pdf"
                    ),
                    "needs_review": _requires_ocr(result_text, language),
                }
                changed = True
            if changed:
                temporary_path: Path | None = None
                try:
                    index_path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=index_path.parent,
                        suffix=".tmp",
                        delete=False,
                    ) as temporary:
                        json.dump(cache, temporary, ensure_ascii=False, indent=2)
                        temporary.write("\n")
                        temporary_path = Path(temporary.name)
                    temporary_path.replace(index_path)
                except OSError as exc:
                    raise ContentEngineError(
                        f"Could not save OCR text cache {index_path}: {exc}"
                    ) from exc
                finally:
                    if temporary_path is not None and temporary_path.exists():
                        temporary_path.unlink()

        results: dict[int, str] = {}
        for page_number, page in page_records.items():
            cached_page = pages_cache.get(str(page_number))
            if isinstance(cached_page, dict) and isinstance(cached_page.get("text"), str):
                results[page_number] = cached_page["text"]
            else:
                source_text = page.get("source_text")
                native_text = (
                    source_text.get("extracted_text_raw_ordered", "")
                    if isinstance(source_text, dict)
                    else ""
                )
                results[page_number] = native_text if isinstance(native_text, str) else ""
        return results


def _publish_book_package(
    *,
    source: Path,
    source_hash: str,
    book_id: str,
    version_id: str,
    class_number: int,
    subject: str,
    language: str | None,
    ocr_language: str,
    pdf_pipeline_version: str,
    model: str,
    api_key: str | None,
    completion: Completion | None,
    spans: list[_ChapterSpan],
    page_texts: list[str],
    page_metadata: list[dict[str, Any]],
    image_assets: dict[str, Path],
    store: BookStore,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    generation_provenance = {
        "provider": "google_gemini",
        "api": "interactions",
        "model_id": model,
        "prompt_version": CHAPTER_PROMPT_VERSION,
        "generated_at": now,
        "review_status": "not_human_reviewed",
    }
    chapter_records: dict[str, dict[str, Any]] = {}
    manifest_chapters: list[dict[str, Any]] = []
    for span in spans:
        pages = [
            _page_record(
                source,
                index + 1,
                page_texts[index],
                f"images/pages/pdf_page_{index + 1:04d}.jpg",
                page_metadata[index]["text_extraction_method"],
                page_metadata[index]["needs_ocr_review"],
            )
            for index in range(span.start_page, span.end_page)
        ]
        if span.marker_kind is not None:
            chapter_text = "\n\n".join(
                page_texts[index].strip()
                for index in range(span.start_page, span.end_page)
                if page_texts[index].strip()
            )
            generated = _generate_chapter_content(
                source,
                span.number or 0,
                span.marker_kind,
                span.title,
                chapter_text,
                model,
                api_key=api_key,
                completion=completion,
            )
            chapter_title = generated["chapter_title"] or span.title
            chapter_record = {
                "schema_version": "1.0",
                "book_id": book_id,
                "version_id": version_id,
                "chapter_id": span.chapter_id,
                "number": span.number,
                "marker_kind": span.marker_kind,
                "title": chapter_title,
                "overview": generated["introduction"],
                "learning_objectives": generated["learning_objectives"],
                "sections": [
                    {
                        key: value
                        for key, value in section.items()
                        if key != "source_text"
                    }
                    for section in generated["sections"]
                ],
                "student_questions": generated["student_questions"],
                "generation_provenance": generation_provenance,
                "pages": pages,
            }
        else:
            chapter_title = span.title
            chapter_record = {
                "schema_version": "1.0",
                "book_id": book_id,
                "version_id": version_id,
                "chapter_id": span.chapter_id,
                "number": None,
                "marker_kind": None,
                "title": chapter_title,
                "overview": None,
                "learning_objectives": [],
                "sections": [],
                "student_questions": [],
                "pages": pages,
            }
        chapter_records[span.chapter_id] = chapter_record
        manifest_chapters.append({
            "chapter_id": span.chapter_id,
            "title": chapter_title,
            "chapter_number": span.number,
            "pdf_pages": {
                "from": span.start_page + 1,
                "to": span.end_page,
            },
            "page_count": len(pages),
        })

    package_manifest = {
        "schema_version": "1.0",
        "book_id": book_id,
        "version_id": version_id,
        "class": class_number,
        "subject": subject,
        "language": language,
        "title": source.stem,
        "source_pdf": {
            "filename": source.name,
            "sha256": source_hash,
            "source_path": str(source),
            "physical_page_count": len(page_texts),
            "ocr_language": ocr_language,
            "processing_pipeline_version": pdf_pipeline_version,
        },
        "page_numbering": {
            "printed_page_numbers": "unknown_until_ocr",
            "physical_pdf_pages": {"from": 1, "to": len(page_texts)},
        },
        "page_rendering": {
            "format": "jpeg",
            "dpi": PAGE_IMAGE_DPI,
            "quality": PAGE_IMAGE_QUALITY,
        },
        "generation_provenance": generation_provenance,
        "chapters": manifest_chapters,
    }
    if sha256_file(source) != source_hash:
        raise ContentEngineError(
            "The source PDF changed during generation; no book version was published."
        )
    package_path = store.publish(
        book_id,
        version_id,
        package_manifest,
        chapter_records,
        assets=image_assets,
    )
    return {
        "status": "complete",
        "book_id": book_id,
        "version_id": version_id,
        "package": str(package_path),
        "chapters": len(chapter_records),
        "physical_pdf_pages": len(page_texts),
        "printed_page_numbers": "unknown_until_ocr",
        "generation_model": model,
        "page_images_rendered": len(image_assets),
        "page_image_dpi": PAGE_IMAGE_DPI,
    }
