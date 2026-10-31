from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from .core import (
    ContentEngineError,
    ensure_all_pdf_pages,
    extract_pdf_images,
    ingest_source,
    sha256_file,
)
from .ocr import resolve_tesseract_command, resolve_tesseract_languages
from .page_index import PageIndexError, PersistentPageIndex
from .printed_pages import index_printed_pages
from .source_naming import class_number_from_text, parse_source_filename


PIPELINE_VERSION = "1.4"
MINIMUM_NATIVE_TEXT_LENGTH = 30
MAX_CLASS_DETECTION_PAGES = 10
SOURCE_OCR_LANGUAGES = {
    "telugu": "tel",
    "hindi": "hin",
    "sanskrit": "san",
    "urdu": "urd",
}
EXPECTED_SCRIPT_RANGES = {
    "tel": (0x0C00, 0x0C7F),
    "hin": (0x0900, 0x097F),
    "san": (0x0900, 0x097F),
    "urd": (0x0600, 0x06FF),
}
OCR_LANGUAGE_ALIASES = {
    "english": "eng",
    "englishmedium": "eng",
    "eng": "eng",
    "telugu": "tel",
    "telugumedium": "tel",
    "tel": "tel",
    "hindi": "hin",
    "hin": "hin",
    "sanskrit": "san",
    "san": "san",
    "urdu": "urd",
    "urd": "urd",
}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(manifest, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_unavailable_printed_page_index(
    manifest_path: Path,
    manifest: dict[str, Any],
    reason: str,
) -> dict[str, int]:
    source = manifest.get("source")
    source_id = manifest.get("source_id")
    if not isinstance(source, dict) or not isinstance(source_id, str):
        raise ContentEngineError(
            "Cannot save printed-page indexing status without valid source metadata."
        )
    pages = [
        {
            "pdf_page": asset["page"],
            "printed_page": None,
            "status": "needs_review",
            "processing_note": reason,
        }
        for asset in manifest.get("assets", [])
        if (
            isinstance(asset, dict)
            and asset.get("kind") == "pdf_page"
            and isinstance(asset.get("page"), int)
            and not isinstance(asset.get("page"), bool)
        )
    ]
    index = {
        "schema_version": 1,
        "source_id": source_id,
        "source_manifest": manifest_path.name,
        "source_sha256": source.get("sha256"),
        "ocr_engine": "tesseract",
        "sequence_offset": None,
        "indexing_status": "unavailable",
        "indexing_error": reason,
        "pages": pages,
    }
    index_path = manifest_path.parent / "printed_page_index.json"
    try:
        index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ContentEngineError(
            f"Could not save unavailable printed-page index: {exc}"
        ) from exc
    return {"matched_pages": 0, "needs_review_pages": len(pages)}


def _class_from_text(text: str) -> int | None:
    return class_number_from_text(text)


def _ocr_language(filename: str, language: str | None = None) -> str:
    if language is not None and language.strip():
        language_key = re.sub(r"[^a-z0-9]", "", language.casefold())
        resolved = OCR_LANGUAGE_ALIASES.get(language_key)
        if resolved is None:
            raise ContentEngineError(
                f"Unsupported OCR language {language!r}. Use English, Telugu, "
                "Hindi, Sanskrit, or Urdu."
            )
        return resolved
    normalized = filename.casefold()
    for language_name, language_code in SOURCE_OCR_LANGUAGES.items():
        if language_name in normalized:
            return language_code
    return "eng"


def _has_expected_script(text: str, language: str) -> bool:
    script_range = EXPECTED_SCRIPT_RANGES.get(language)
    if script_range is None:
        return True
    minimum, maximum = script_range
    return sum(minimum <= ord(character) <= maximum for character in text) >= 3


def _broken_character_count(text: str) -> int:
    return sum(
        unicodedata.category(character) == "Co" or character == "\ufffd"
        for character in text
    )


def _has_broken_mapping(text: str) -> bool:
    return _broken_character_count(text) / max(1, len(text)) > 0.01


def _ocr_text_is_better(native_text: str, recognized_text: str, language: str) -> bool:
    recognized_text = recognized_text.strip()
    if not recognized_text:
        return False
    native_expected_script = _has_expected_script(native_text, language)
    recognized_expected_script = _has_expected_script(recognized_text, language)
    if recognized_expected_script and not native_expected_script:
        return True
    if _has_broken_mapping(native_text):
        native_broken = _broken_character_count(native_text)
        recognized_broken = _broken_character_count(recognized_text)
        if (
            recognized_expected_script
            and recognized_broken < native_broken
            and recognized_broken / len(recognized_text) < native_broken / len(native_text)
        ):
            return True
    return len(recognized_text) > len(native_text.strip())


def _requires_ocr(text: str, language: str = "eng") -> bool:
    if len(text.strip()) < MINIMUM_NATIVE_TEXT_LENGTH:
        return True
    return _has_broken_mapping(text) or not _has_expected_script(text, language)


def _ocr_page(executable: str, image_path: Path, language: str) -> str:
    try:
        result = subprocess.run(
            [executable, str(image_path), "stdout", "--psm", "3", "-l", language],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContentEngineError(
            f"Tesseract OCR timed out on {image_path.name} after 60 seconds."
        ) from exc
    except OSError as exc:
        raise ContentEngineError(f"Could not start Tesseract OCR: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContentEngineError(
            f"Tesseract failed on {image_path.name}: {detail or 'unknown OCR error'}"
        )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _infer_class(manifest: dict[str, Any], package_dir: Path) -> int | None:
    source = manifest.get("source", {})
    filename = source.get("filename", "")
    class_number = _class_from_text(filename)
    if class_number is not None:
        return class_number

    for asset in manifest.get("assets", []):
        if not isinstance(asset, dict) or asset.get("kind") != "pdf_page":
            continue
        page_number = asset.get("page")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        if page_number > MAX_CLASS_DETECTION_PAGES:
            continue
        text_path_value = asset.get("text_path")
        if not isinstance(text_path_value, str):
            continue
        try:
            text = (package_dir / text_path_value).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        class_number = _class_from_text(text)
        if class_number is not None:
            return class_number
    return None


def _restore_generated_ocr(
    source: Path,
    manifest: dict[str, Any],
    package_dir: Path,
    language: str,
) -> None:
    generated_assets = [
        asset
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict)
        and asset.get("kind") == "pdf_page"
        and asset.get("text_extraction_method") == "tesseract_ocr"
    ]
    if not generated_assets:
        return
    try:
        document = pymupdf.open(source)
    except (pymupdf.FileDataError, ValueError) as exc:
        raise ContentEngineError(f"Could not reopen PDF {source}: {exc}") from exc
    try:
        for asset in generated_assets:
            page_number = asset.get("page")
            text_path_value = asset.get("text_path")
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or not 1 <= page_number <= document.page_count
                or not isinstance(text_path_value, str)
            ):
                raise ContentEngineError("A previously OCR-processed PDF page has invalid metadata.")
            text_path = (package_dir / text_path_value).resolve()
            try:
                text_path.relative_to(package_dir.resolve())
            except ValueError as exc:
                raise ContentEngineError(
                    "A previously OCR-processed text path escapes its source package."
                ) from exc
            native_text = document.load_page(page_number - 1).get_text("text", sort=True)
            try:
                text_path.write_text(native_text, encoding="utf-8")
            except OSError as exc:
                raise ContentEngineError(
                    f"Could not restore native PDF text for page {page_number}: {exc}"
                ) from exc
            asset["text_characters"] = len(native_text)
            asset["text_extraction_method"] = "native_pdf"
            asset["needs_ocr_review"] = _requires_ocr(native_text, language)
    finally:
        document.close()


def _existing_image_result(manifest: dict[str, Any], package_dir: Path) -> dict[str, int] | None:
    images = manifest.get("embedded_images")
    if not isinstance(images, list):
        return None
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("path"), str):
            return None
        image_path = (package_dir / image["path"]).resolve()
        try:
            image_path.relative_to(package_dir.resolve())
        except ValueError:
            return None
        if not image_path.is_file():
            return None
    placements = sum(
        len(image.get("placements", []))
        for image in images
        if isinstance(image, dict) and isinstance(image.get("placements"), list)
    )
    return {"unique_images": len(images), "page_placements": placements}


def _existing_page_index_result(
    manifest_path: Path,
    source_id: str,
    manifest: dict[str, Any],
) -> dict[str, int] | None:
    index_path = manifest_path.parent / "printed_page_index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(index, dict) or index.get("source_id") != source_id:
        return None
    page_numbers = {
        asset.get("page")
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
    }
    index_pages = index.get("pages")
    if not isinstance(index_pages, list) or {
        entry.get("pdf_page")
        for entry in index_pages
        if isinstance(entry, dict)
    } != page_numbers:
        return None
    return {
        "matched_pages": sum(
            isinstance(entry, dict) and entry.get("status") == "matched"
            for entry in index_pages
        ),
        "needs_review_pages": sum(
            not isinstance(entry, dict) or entry.get("status") != "matched"
            for entry in index_pages
        ),
    }


def _package_asset_path(package_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        path = (package_dir / value).resolve()
        path.relative_to(package_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def _pdf_page_assets_are_valid(
    manifest: dict[str, Any],
    package_dir: Path,
    page_count: int,
) -> bool:
    if not _pdf_page_asset_subset_is_valid(manifest, package_dir, page_count):
        return False
    page_numbers = [
        asset["page"]
        for asset in manifest["assets"]
        if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
    ]
    return len(page_numbers) == page_count


def _pdf_page_asset_subset_is_valid(
    manifest: dict[str, Any],
    package_dir: Path,
    page_count: int,
) -> bool:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        return False
    page_assets = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
    ]
    page_numbers = [asset.get("page") for asset in page_assets]
    if any(
        isinstance(page, bool)
        or not isinstance(page, int)
        or not 1 <= page <= page_count
        for page in page_numbers
    ) or len(set(page_numbers)) != len(page_numbers):
        return False
    for asset in page_assets:
        text_path = _package_asset_path(package_dir, asset.get("text_path"))
        image_path = _package_asset_path(package_dir, asset.get("image_path"))
        if text_path is None or image_path is None:
            return False
        try:
            text_path.read_text(encoding="utf-8")
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnicodeDecodeError, ValueError, Image.DecompressionBombError):
            return False
    return True


def _printed_page_index_is_valid(
    index_path: Path,
    source_id: str,
    page_count: int,
) -> bool:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != 1
        or index.get("source_id") != source_id
    ):
        return False
    pages = index.get("pages")
    if not isinstance(pages, list) or len(pages) != page_count:
        return False
    seen_pdf_pages: set[int] = set()
    for entry in pages:
        if (
            not isinstance(entry, dict)
            or isinstance(entry.get("pdf_page"), bool)
            or not isinstance(entry.get("pdf_page"), int)
            or not 1 <= entry["pdf_page"] <= page_count
            or entry["pdf_page"] in seen_pdf_pages
            or entry.get("status") not in {"matched", "needs_review"}
        ):
            return False
        seen_pdf_pages.add(entry["pdf_page"])
        printed_page = entry.get("printed_page")
        if printed_page is not None and (
            isinstance(printed_page, bool)
            or not isinstance(printed_page, int)
            or printed_page < 1
        ):
            return False
    return seen_pdf_pages == set(range(1, page_count + 1))


def is_current_pdf_package(
    manifest_path: Path,
    *,
    expected_source_hash: str | None = None,
) -> bool:
    """Return whether an imported PDF package is safe to use without reprocessing."""
    manifest_path = manifest_path.expanduser().resolve()
    package_dir = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    source_id = manifest.get("source_id")
    source = manifest.get("source")
    processing = manifest.get("processing")
    if (
        not isinstance(source_id, str)
        or package_dir.name != source_id
        or not isinstance(source, dict)
        or not isinstance(processing, dict)
        or processing.get("status") != "complete"
        or processing.get("pipeline_version") != PIPELINE_VERSION
    ):
        return False
    source_hash = source.get("sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
        or source_hash[:16] != source_id
        or (expected_source_hash is not None and source_hash != expected_source_hash)
        or source.get("media_type") != "pdf"
    ):
        return False
    result = processing.get("result")
    page_count = source.get("pdf_page_count")
    if (
        not isinstance(result, dict)
        or result.get("status") != "complete"
        or result.get("source_id") != source_id
        or not isinstance(result.get("ocr_language"), str)
        or result.get("ocr_language") != source.get("ocr_language")
        or result.get("ocr_language") not in set(OCR_LANGUAGE_ALIASES.values())
        or isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
        or result.get("pdf_pages") != page_count
    ):
        return False

    imported_path = _package_asset_path(package_dir, source.get("imported_path"))
    if imported_path is None or not imported_path.is_file():
        return False
    try:
        if sha256_file(imported_path) != source_hash:
            return False
    except OSError:
        return False
    if not _pdf_page_assets_are_valid(manifest, package_dir, page_count):
        return False
    if not _printed_page_index_is_valid(
        manifest_path.parent / "printed_page_index.json",
        source_id,
        page_count,
    ):
        return False

    embedded_images = manifest.get("embedded_images", [])
    if not isinstance(embedded_images, list):
        return False
    for image in embedded_images:
        if not isinstance(image, dict):
            return False
        image_path = _package_asset_path(package_dir, image.get("path"))
        if image_path is None or not image_path.is_file():
            return False
    return True


def _quarantine_source_package(package_dir: Path) -> None:
    quarantine_root = package_dir.parent / "_stale"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{package_dir.name}-{uuid.uuid4().hex[:8]}"
    try:
        package_dir.rename(destination)
    except OSError as exc:
        raise ContentEngineError(
            f"Could not preserve stale PDF package {package_dir}: {exc}"
        ) from exc


def process_pdf(
    source: Path,
    data_dir: Path,
    dpi: int = 120,
    class_number: int | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if source.suffix.lower() != ".pdf":
        raise ContentEngineError("The automatic content pipeline accepts PDF files only.")
    if class_number is not None and (
        isinstance(class_number, bool)
        or not isinstance(class_number, int)
        or not 1 <= class_number <= 8
    ):
        raise ContentEngineError("Class number must be a whole number from 1 to 8.")

    if not source.is_file():
        raise ContentEngineError(f"Source file does not exist: {source}")
    ocr_language = _ocr_language(source.name, language)
    source_id = sha256_file(source)[:16]
    manifest_path = data_dir / "sources" / source_id / "manifest.json"
    package_dir = manifest_path.parent
    if package_dir.exists():
        if not manifest_path.is_file():
            _quarantine_source_package(package_dir)
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                _quarantine_source_package(package_dir)
            else:
                if not isinstance(manifest, dict):
                    _quarantine_source_package(package_dir)
                else:
                    stored_source = manifest.get("source")
                    imported_path = (
                        _package_asset_path(package_dir, stored_source.get("imported_path"))
                        if isinstance(stored_source, dict)
                        else None
                    )
                    source_hash = (
                        stored_source.get("sha256")
                        if isinstance(stored_source, dict)
                        else None
                    )
                    valid_source_copy = False
                    if (
                        manifest.get("source_id") == source_id
                        and isinstance(source_hash, str)
                        and source_hash == sha256_file(source)
                        and stored_source.get("media_type") == "pdf"
                        and imported_path is not None
                        and imported_path.is_file()
                    ):
                        try:
                            valid_source_copy = sha256_file(imported_path) == source_hash
                        except OSError:
                            valid_source_copy = False
                    if not valid_source_copy:
                        _quarantine_source_package(package_dir)
                    elif is_current_pdf_package(
                        manifest_path,
                        expected_source_hash=source_hash,
                    ) and manifest["processing"]["result"].get(
                        "ocr_language"
                    ) == ocr_language:
                        source_metadata = manifest["source"]
                        source_name = parse_source_filename(source.name)
                        source_metadata["filename"] = source.name
                        source_metadata["original_path"] = str(source)
                        requested_class = class_number or _class_from_text(source.name)
                        result = manifest["processing"]["result"]
                        if requested_class is not None:
                            source_metadata["class_number"] = requested_class
                            result["class_number"] = requested_class
                        for key, value in (
                            ("subject", source_name.subject),
                            ("medium", source_name.medium),
                            ("board", source_name.board),
                            ("school", source_name.school),
                        ):
                            if value:
                                source_metadata[key] = value
                            else:
                                source_metadata.pop(key, None)
                        _write_manifest(manifest_path, manifest)
                        return result


                    else:
                        manifest["processing"] = {
                            "status": "processing",
                            "pipeline_version": PIPELINE_VERSION,
                        }
                        _write_manifest(manifest_path, manifest)
                        stored_page_count = manifest["source"].get("pdf_page_count")
                        page_count = (
                            stored_page_count
                            if isinstance(stored_page_count, int)
                            and not isinstance(stored_page_count, bool)
                            else 0
                        )
                        if not _pdf_page_asset_subset_is_valid(
                            manifest,
                            package_dir,
                            page_count,
                        ):
                            manifest["assets"] = [
                                asset
                                for asset in manifest.get("assets", [])
                                if not (
                                    isinstance(asset, dict)
                                    and asset.get("kind") == "pdf_page"
                                )
                            ] if isinstance(manifest.get("assets"), list) else []
                            manifest["embedded_images"] = []
                        _write_manifest(manifest_path, manifest)
    if not manifest_path.is_file():
        manifest_path = ingest_source(source, data_dir, dpi=dpi)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["processing"] = {
        "status": "processing",
        "pipeline_version": PIPELINE_VERSION,
    }
    _write_manifest(manifest_path, manifest)

    package_dir = manifest_path.parent
    pages_added = ensure_all_pdf_pages(source, manifest_path, dpi=dpi)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_metadata = manifest.get("source")
    if not isinstance(source_metadata, dict):
        raise ContentEngineError("Imported PDF manifest has invalid source metadata.")
    filename = source.name
    source_metadata["filename"] = filename
    source_metadata["original_path"] = str(source)
    source_metadata["ocr_language"] = ocr_language
    _restore_generated_ocr(source, manifest, package_dir, ocr_language)
    assets = [
        asset
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
    ]
    pages_needing_ocr: set[int] = set()
    for asset in assets:
        text_path_value = asset.get("text_path")
        if not isinstance(text_path_value, str):
            raise ContentEngineError(
                f"PDF page {asset.get('page')} is missing its extracted text path."
            )
        text_path = (package_dir / text_path_value).resolve()
        if not text_path.is_relative_to(package_dir.resolve()):
            raise ContentEngineError("A PDF text asset points outside its source package.")
        try:
            if _requires_ocr(text_path.read_text(encoding="utf-8"), ocr_language):
                pages_needing_ocr.add(asset["page"])
        except (OSError, UnicodeDecodeError) as exc:
            raise ContentEngineError(
                f"Could not read extracted text for PDF page {asset.get('page')}: {exc}"
            ) from exc

    executable: str | None = None
    available_languages: set[str] = set()
    ocr_setup_error: str | None = None
    if pages_needing_ocr:
        try:
            executable = resolve_tesseract_command()
            available_languages = resolve_tesseract_languages(executable)
        except ContentEngineError as exc:
            ocr_setup_error = str(exc)
    ocr_language_available = ocr_language in available_languages
    pages_requiring_ocr = 0
    pages_without_ocr_language = 0
    pages_ocr_failed = 0
    unreadable_pages = 0
    ocr_failures: list[dict[str, Any]] = []
    for asset in manifest.get("assets", []):
        if not isinstance(asset, dict) or asset.get("kind") != "pdf_page":
            continue
        text_path_value = asset.get("text_path")
        image_path_value = asset.get("image_path")
        if not isinstance(text_path_value, str) or not isinstance(image_path_value, str):
            raise ContentEngineError(
                f"PDF page {asset.get('page')} is missing extracted text or image paths."
            )
        text_path = (package_dir / text_path_value).resolve()
        image_path = (package_dir / image_path_value).resolve()
        if not text_path.is_relative_to(package_dir.resolve()) or not image_path.is_relative_to(package_dir.resolve()):
            raise ContentEngineError("A PDF page asset points outside its source package.")
        try:
            native_text = text_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ContentEngineError(
                f"Could not read extracted text for PDF page {asset.get('page')}: {exc}"
            ) from exc

        needs_ocr = _requires_ocr(native_text, ocr_language)
        if needs_ocr and executable is not None and ocr_language_available:
            pages_requiring_ocr += 1
            try:
                recognized_text = _ocr_page(executable, image_path, ocr_language)
            except ContentEngineError as exc:
                pages_ocr_failed += 1
                asset["text_extraction_method"] = "native_pdf_ocr_failed"
                asset["ocr_error"] = str(exc)
                ocr_failures.append({
                    "page": asset.get("page"),
                    "error": str(exc),
                })
            else:
                if _ocr_text_is_better(native_text, recognized_text, ocr_language):
                    text_path.write_text(recognized_text + "\n", encoding="utf-8")
                    asset["text_characters"] = len(recognized_text)
                    asset["text_extraction_method"] = "tesseract_ocr"
                    asset.pop("ocr_error", None)
                else:
                    asset["text_extraction_method"] = "native_pdf"
        elif needs_ocr:
            pages_without_ocr_language += 1
            asset["text_extraction_method"] = (
                "native_pdf_ocr_unavailable"
                if executable is None
                else "native_pdf_language_ocr_unavailable"
            )
            if ocr_setup_error is not None:
                asset["ocr_error"] = ocr_setup_error
        else:
            asset["text_extraction_method"] = "native_pdf"

        final_text = text_path.read_text(encoding="utf-8")
        asset["needs_ocr_review"] = _requires_ocr(final_text, ocr_language)
        if asset["needs_ocr_review"]:
            unreadable_pages += 1

    _write_manifest(manifest_path, manifest)
    image_result = _existing_image_result(manifest, package_dir)
    if image_result is None:
        image_result = extract_pdf_images(manifest_path, data_dir)
    page_index_result = _existing_page_index_result(manifest_path, source_id, manifest)
    if page_index_result is None:
        try:
            page_index_result = index_printed_pages(manifest_path)
        except ContentEngineError as exc:
            page_index_result = _write_unavailable_printed_page_index(
                manifest_path,
                manifest,
                str(exc),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if class_number is None:
        class_number = _infer_class(manifest, package_dir)
    source_metadata = manifest["source"]
    source_metadata["class_number"] = class_number
    source_metadata["ocr_language"] = ocr_language
    source_name = parse_source_filename(filename)
    source_metadata["subject"] = source_name.subject
    if source_name.medium:
        source_metadata["medium"] = source_name.medium
    if source_name.board:
        source_metadata["board"] = source_name.board
    if source_name.school:
        source_metadata["school"] = source_name.school
    result = {
        "status": "complete",
        "source_id": source_id,
        "manifest": str(manifest_path),
        "class_number": class_number,
        "pdf_pages": len(manifest.get("assets", [])),
        "pages_added_from_partial_import": pages_added,
        "pages_ocr_processed": pages_requiring_ocr,
        "pages_without_ocr_language": pages_without_ocr_language,
        "pages_ocr_failed": pages_ocr_failed,
        "pages_needing_ocr_review": unreadable_pages,
        "ocr_language": ocr_language,
        "ocr_language_available": ocr_language_available,
        "ocr_status": (
            "not_needed"
            if not pages_requiring_ocr and not pages_without_ocr_language
            and not pages_ocr_failed
            else "partial"
            if pages_without_ocr_language or pages_ocr_failed
            else "complete"
        ),
        "ocr_setup_error": ocr_setup_error,
        "ocr_failures": ocr_failures,
        "unique_embedded_images": image_result["unique_images"],
        "image_placements": image_result["page_placements"],
        "printed_pages_matched": page_index_result["matched_pages"],
        "printed_pages_needing_review": page_index_result["needs_review_pages"],
        "printed_page_index_status": (
            "complete"
            if page_index_result["matched_pages"]
            else "needs_review"
        ),
        "warnings": [
            *(
                [f"OCR was not available for {pages_without_ocr_language} page(s)."]
                if pages_without_ocr_language
                else []
            ),
            *(
                [f"OCR failed on {pages_ocr_failed} page(s)."]
                if pages_ocr_failed
                else []
            ),
            *(
                ["Printed-page indexing was unavailable; search by question or PDF page instead."]
                if page_index_result["matched_pages"] == 0
                else []
            ),
        ],
    }
    manifest["processing"] = {
        "status": "complete",
        "pipeline_version": PIPELINE_VERSION,
        "result": result,
    }
    _write_manifest(manifest_path, manifest)
    if isinstance(class_number, int) and not isinstance(class_number, bool) and 1 <= class_number <= 8:
        try:
            PersistentPageIndex(
                data_dir / "search" / "page_index.sqlite3"
            ).index_processed_package(
                manifest_path,
                class_number,
                source_name.display_title,
            )
        except PageIndexError as exc:
            manifest["processing"] = {
                "status": "failed",
                "pipeline_version": PIPELINE_VERSION,
                "error": str(exc),
            }
            _write_manifest(manifest_path, manifest)
            raise ContentEngineError(str(exc)) from exc
    return result


def process_and_load_pdf_pages(
    source: Path,
    data_dir: Path,
    *,
    dpi: int = 120,
    class_number: int | None = None,
    language: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = source.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not source.is_file():
        raise ContentEngineError(f"Source file does not exist: {source}")
    expected_hash = sha256_file(source)
    result = process_pdf(
        source,
        data_dir,
        dpi=dpi,
        class_number=class_number,
        language=language,
    )
    if result.get("status") != "complete":
        raise ContentEngineError(f"PDF processing did not complete for {source.name}.")
    if sha256_file(source) != expected_hash:
        raise ContentEngineError(
            "The source PDF changed while its processed page data was being loaded."
        )

    manifest_path_value = result.get("manifest")
    if not isinstance(manifest_path_value, str):
        raise ContentEngineError("PDF processing returned no source manifest path.")
    manifest_path = Path(manifest_path_value).resolve()
    try:
        manifest_path.relative_to(data_dir)
    except ValueError as exc:
        raise ContentEngineError(
            "Processed PDF manifest is outside the data directory."
        ) from exc
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(
            f"Could not read processed PDF manifest {manifest_path}: {exc}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or not is_current_pdf_package(
            manifest_path,
            expected_source_hash=expected_hash,
        )
    ):
        raise ContentEngineError(
            f"Processed PDF package is incomplete or stale for {source.name}."
        )

    source_metadata = manifest.get("source")
    page_count = (
        source_metadata.get("pdf_page_count")
        if isinstance(source_metadata, dict)
        else None
    )
    assets = manifest.get("assets")
    if (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or page_count < 1
        or not isinstance(assets, list)
    ):
        raise ContentEngineError(
            f"Processed PDF manifest has invalid page metadata for {source.name}."
        )

    package_dir = manifest_path.parent
    page_assets = sorted(
        (
            asset
            for asset in assets
            if isinstance(asset, dict) and asset.get("kind") == "pdf_page"
        ),
        key=lambda asset: asset.get("page", 0),
    )
    if [asset.get("page") for asset in page_assets] != list(
        range(1, page_count + 1)
    ):
        raise ContentEngineError(
            f"Processed PDF package does not contain every page for {source.name}."
        )

    pages: list[dict[str, Any]] = []
    for asset in page_assets:
        text_path = _package_asset_path(package_dir, asset.get("text_path"))
        image_path = _package_asset_path(package_dir, asset.get("image_path"))
        if text_path is None or image_path is None:
            raise ContentEngineError(
                f"Processed PDF page {asset['page']} has an unsafe asset path."
            )
        try:
            text = text_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ContentEngineError(
                f"Could not read processed text for PDF page {asset['page']}: {exc}"
            ) from exc
        if not image_path.is_file():
            raise ContentEngineError(
                f"Processed image for PDF page {asset['page']} is missing."
            )
        pages.append({
            "page": asset["page"],
            "text": text,
            "text_extraction_method": asset.get(
                "text_extraction_method",
                "native_pdf",
            ),
            "needs_ocr_review": bool(asset.get("needs_ocr_review")),
            "image_path": image_path,
        })
    return manifest, pages
