from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Protocol

from .core import ContentEngineError

_PACKAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


class BookStore(Protocol):
    def list_packages(self) -> list[tuple[str, str]]: ...

    def package_path(self, book_id: str, version_id: str) -> Path: ...

    def contains(self, book_id: str, version_id: str) -> bool: ...

    def asset_path(self, book_id: str, version_id: str, asset_name: str) -> Path: ...

    def publish(
        self,
        book_id: str,
        version_id: str,
        manifest: Mapping[str, Any],
        chapters: Mapping[str, Mapping[str, Any]],
        *,
        assets: Mapping[str, Path] | None = None,
    ) -> Path: ...

    def read_manifest(self, book_id: str, version_id: str) -> dict[str, Any]: ...

    def read_chapter(
        self,
        book_id: str,
        version_id: str,
        chapter_id: str,
    ) -> dict[str, Any]: ...


class FileSystemBookStore:
    """Immutable versioned book packages stored as JSON under a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def list_packages(self) -> list[tuple[str, str]]:
        if not self.root.is_dir():
            return []
        packages = []
        for manifest_path in self.root.glob("*/versions/*/book.json"):
            book_id = manifest_path.parents[2].name
            version_id = manifest_path.parent.name
            package_path = self._version_path(book_id, version_id)
            if (package_path / "book.json").is_file():
                packages.append((book_id, version_id))
        return sorted(packages)

    def publish(
        self,
        book_id: str,
        version_id: str,
        manifest: Mapping[str, Any],
        chapters: Mapping[str, Mapping[str, Any]],
        *,
        assets: Mapping[str, Path] | None = None,
    ) -> Path:
        _validate_id(book_id, "book")
        _validate_id(version_id, "version")
        if not isinstance(manifest, Mapping):
            raise ContentEngineError("Book manifest must be a JSON object.")
        if manifest.get("schema_version") != "1.0":
            raise ContentEngineError("Book manifest schema_version must be '1.0'.")
        if manifest.get("book_id") not in (None, book_id):
            raise ContentEngineError("Book manifest book_id does not match the package ID.")
        if manifest.get("version_id") not in (None, version_id):
            raise ContentEngineError("Book manifest version_id does not match the package version.")
        if not chapters:
            raise ContentEngineError("A book package must contain at least one chapter.")

        source_pdf = manifest.get("source_pdf")
        if not isinstance(source_pdf, Mapping):
            raise ContentEngineError("Book manifest requires source_pdf provenance.")
        source_filename = source_pdf.get("filename")
        source_checksum = source_pdf.get("sha256")
        page_count = source_pdf.get("physical_page_count")
        if not isinstance(source_filename, str) or not source_filename.strip():
            raise ContentEngineError("Book manifest source_pdf filename is required.")
        if not isinstance(source_checksum, str) or re.fullmatch(r"[0-9a-f]{64}", source_checksum) is None:
            raise ContentEngineError("Book manifest requires a lowercase SHA-256 source checksum.")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 1:
            raise ContentEngineError("Book manifest physical_page_count must be a positive integer.")

        manifest_chapters = manifest.get("chapters")
        if not isinstance(manifest_chapters, list):
            raise ContentEngineError("Book manifest chapters must be a list.")
        expected_chapter_ids: list[str] = []
        for entry in manifest_chapters:
            if not isinstance(entry, Mapping):
                raise ContentEngineError("Book manifest chapter entries must be objects.")
            chapter_id = entry.get("chapter_id")
            _validate_id(chapter_id, "chapter")
            expected_chapter_ids.append(chapter_id)
        if len(set(expected_chapter_ids)) != len(expected_chapter_ids):
            raise ContentEngineError("Book manifest contains duplicate chapter IDs.")
        if set(expected_chapter_ids) != set(chapters):
            raise ContentEngineError(
                "Book manifest chapter IDs must match the supplied chapter records."
            )

        records: dict[str, dict[str, Any]] = {}
        package_page_numbers: set[int] = set()
        package_page_ids: set[str] = set()
        for chapter_id, chapter in chapters.items():
            _validate_id(chapter_id, "chapter")
            if not isinstance(chapter, Mapping):
                raise ContentEngineError(f"Chapter {chapter_id} must be a JSON object.")
            if chapter.get("schema_version") != "1.0":
                raise ContentEngineError(
                    f"Chapter {chapter_id} schema_version must be '1.0'."
                )
            if chapter.get("book_id") != book_id or chapter.get("version_id") != version_id:
                raise ContentEngineError(
                    f"Chapter {chapter_id} book/version metadata does not match its package."
                )
            if chapter.get("chapter_id") != chapter_id:
                raise ContentEngineError(
                    f"Chapter record ID does not match manifest chapter {chapter_id}."
                )
            pages = chapter.get("pages")
            if not isinstance(pages, list) or not pages:
                raise ContentEngineError(f"Chapter {chapter_id} must contain page records.")
            page_ids, page_numbers = _validate_pages(
                chapter_id,
                pages,
                source_filename,
            )
            if package_page_ids.intersection(page_ids):
                raise ContentEngineError("Book package contains duplicate page IDs.")
            if package_page_numbers.intersection(page_numbers):
                raise ContentEngineError(
                    "A physical PDF page is assigned to more than one chapter."
                )
            package_page_ids.update(page_ids)
            package_page_numbers.update(page_numbers)
            records[chapter_id] = dict(chapter)

        expected_page_numbers = set(range(1, page_count + 1))
        if package_page_numbers != expected_page_numbers:
            raise ContentEngineError(
                "Book package pages must cover every physical PDF page exactly once."
            )
        previous_range_end = 0
        for entry in manifest_chapters:
            chapter_id = entry["chapter_id"]
            chapter_pages = records[chapter_id]["pages"]
            page_numbers = sorted(
                page["source_provenance"]["source_pdf_page_number_1_indexed"]
                for page in chapter_pages
            )
            page_range = entry.get("pdf_pages")
            if not isinstance(page_range, Mapping):
                raise ContentEngineError(
                    f"Book manifest chapter {chapter_id} requires a physical PDF page range."
                )
            page_start = page_range.get("from")
            page_end = page_range.get("to")
            if (
                isinstance(page_start, bool)
                or not isinstance(page_start, int)
                or isinstance(page_end, bool)
                or not isinstance(page_end, int)
                or page_start < 1
                or page_end < page_start
                or page_numbers != list(range(page_start, page_end + 1))
            ):
                raise ContentEngineError(
                    f"Book manifest physical PDF range does not match chapter {chapter_id} pages."
                )
            if page_start <= previous_range_end:
                raise ContentEngineError(
                    "Book manifest chapter ranges must be ordered and non-overlapping."
                )
            previous_range_end = page_end
            declared_page_count = entry.get("page_count")
            if (
                isinstance(declared_page_count, bool)
                or not isinstance(declared_page_count, int)
                or declared_page_count != len(chapter_pages)
            ):
                raise ContentEngineError(
                    f"Book manifest page_count does not match chapter {chapter_id} pages."
                )

        serialized_chapters = {
            f"chapters/{chapter_id}.json": _serialize(record)
            for chapter_id, record in records.items()
        }
        source_assets = assets or {}
        normalized_assets: dict[str, Path] = {}
        for asset_name, asset_path in source_assets.items():
            relative_path = _validate_asset_name(asset_name)
            if relative_path in serialized_chapters or relative_path == "book.json":
                raise ContentEngineError(
                    f"Book package asset path is reserved: {relative_path}."
                )
            source_path = Path(asset_path).expanduser().resolve()
            if not source_path.is_file():
                raise ContentEngineError(
                    f"Book package asset does not exist: {source_path}."
                )
            normalized_assets[relative_path] = source_path
        referenced_assets: set[str] = set()
        for chapter in records.values():
            for page in chapter["pages"]:
                visuals = page.get("visuals")
                if not isinstance(visuals, Mapping):
                    continue
                image_reference = visuals.get("page_image")
                if image_reference in (None, ""):
                    continue
                relative_path = _validate_asset_name(image_reference)
                if relative_path not in normalized_assets:
                    raise ContentEngineError(
                        f"Page record references a missing package asset: {relative_path}."
                    )
                referenced_assets.add(relative_path)
        if referenced_assets != set(normalized_assets):
            raise ContentEngineError(
                "Every page image asset must be referenced by a page record."
            )
        package_manifest = dict(manifest)
        package_manifest["book_id"] = book_id
        package_manifest["version_id"] = version_id
        package_manifest["package_files"] = {
            relative_path: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for relative_path, content in serialized_chapters.items()
        }
        for relative_path, source_path in normalized_assets.items():
            package_manifest["package_files"][relative_path] = _sha256_file(source_path)
        package_manifest["chapters"] = [
            {
                **dict(entry),
                "file": f"chapters/{entry['chapter_id']}.json",
            }
            for entry in manifest_chapters
        ]
        package = {
            "book.json": _serialize(package_manifest),
            **serialized_chapters,
        }
        version_root = self._version_path(book_id, version_id)
        version_root.parent.mkdir(parents=True, exist_ok=True)
        if version_root.exists():
            self._assert_same_package(version_root, package, normalized_assets)
            return version_root

        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{version_id}-", dir=version_root.parent)
        )
        try:
            for relative_path, content in package.items():
                target = staging_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            for relative_path, source_path in normalized_assets.items():
                target = staging_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
            try:
                staging_root.rename(version_root)
            except OSError:
                if not version_root.exists():
                    raise
                self._assert_same_package(version_root, package, normalized_assets)
            return version_root
        except OSError as exc:
            raise ContentEngineError(
                f"Could not publish book package {book_id}/{version_id}: {exc}"
            ) from exc
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)

    def read_manifest(self, book_id: str, version_id: str) -> dict[str, Any]:
        path = self._version_path(book_id, version_id) / "book.json"
        manifest = _read_json(path)
        if manifest.get("book_id") != book_id or manifest.get("version_id") != version_id:
            raise ContentEngineError("Book manifest identity does not match its version path.")
        return manifest

    def contains(self, book_id: str, version_id: str) -> bool:
        package_path = self.package_path(book_id, version_id)
        manifest_path = package_path / "book.json"
        if package_path.exists() and not manifest_path.is_file():
            raise ContentEngineError(
                "A partial package already occupies this version path; review it before retrying."
            )
        return manifest_path.is_file()

    def package_path(self, book_id: str, version_id: str) -> Path:
        return self._version_path(book_id, version_id)

    def asset_path(self, book_id: str, version_id: str, asset_name: str) -> Path:
        relative_path = _validate_asset_name(asset_name)
        manifest = self.read_manifest(book_id, version_id)
        checksums = manifest.get("package_files")
        expected_checksum = (
            checksums.get(relative_path)
            if isinstance(checksums, dict)
            else None
        )
        if not isinstance(expected_checksum, str):
            raise ContentEngineError(
                f"Book manifest does not contain asset {relative_path}."
            )
        path = self._version_path(book_id, version_id) / relative_path
        if not path.is_file() or _sha256_file(path) != expected_checksum:
            raise ContentEngineError(
                f"Book package asset {relative_path} failed its integrity check."
            )
        return path

    def read_chapter(
        self,
        book_id: str,
        version_id: str,
        chapter_id: str,
    ) -> dict[str, Any]:
        _validate_id(chapter_id, "chapter")
        manifest = self.read_manifest(book_id, version_id)
        chapter_ids = {
            entry.get("chapter_id")
            for entry in manifest.get("chapters", [])
            if isinstance(entry, dict)
        }
        if chapter_id not in chapter_ids:
            raise ContentEngineError(
                f"Chapter {chapter_id} is not listed in book {book_id}/{version_id}."
            )
        relative_path = f"chapters/{chapter_id}.json"
        checksums = manifest.get("package_files")
        expected_checksum = (
            checksums.get(relative_path)
            if isinstance(checksums, dict)
            else None
        )
        if not isinstance(expected_checksum, str):
            raise ContentEngineError(
                f"Book manifest has no integrity checksum for chapter {chapter_id}."
            )
        path = self._version_path(book_id, version_id) / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ContentEngineError(
                f"Could not read book package file {path.name}: {exc}"
            ) from exc
        actual_checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_checksum != expected_checksum:
            raise ContentEngineError(
                f"Book package chapter {chapter_id} failed its integrity check."
            )
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ContentEngineError(
                f"Could not read book package file {path.name}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContentEngineError(
                f"Book package file {path.name} must contain a JSON object."
            )
        return value

    def _version_path(self, book_id: str, version_id: str) -> Path:
        _validate_id(book_id, "book")
        _validate_id(version_id, "version")
        path = (self.root / book_id / "versions" / version_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ContentEngineError("Book package path escapes the configured storage root.") from exc
        return path

    @staticmethod
    def _assert_same_package(
        version_root: Path,
        package: Mapping[str, str],
        assets: Mapping[str, Path],
    ) -> None:
        if not version_root.is_dir():
            raise ContentEngineError("A published book version exists but is not a directory.")
        existing_files = {
            path.relative_to(version_root).as_posix()
            for path in version_root.rglob("*")
            if path.is_file()
        }
        if existing_files != set(package) | set(assets):
            raise ContentEngineError(
                "This book version is already published with different package files. "
                "Publish changed content under a new version ID."
            )
        for relative_path, expected in package.items():
            try:
                existing = (version_root / relative_path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ContentEngineError(
                    f"Could not verify existing book package file {relative_path}: {exc}"
                ) from exc
            if existing != expected:
                raise ContentEngineError(
                    "This book version is already published with different content. "
                    "Publish changed content under a new version ID."
                )
        for relative_path, expected_path in assets.items():
            if _sha256_file(version_root / relative_path) != _sha256_file(expected_path):
                raise ContentEngineError(
                    "This book version is already published with different content. "
                    "Publish changed content under a new version ID."
                )


def _validate_id(value: Any, label: str) -> None:
    if not isinstance(value, str) or _PACKAGE_ID.fullmatch(value) is None:
        raise ContentEngineError(
            f"{label.capitalize()} ID must be 1-128 letters, digits, underscores, or hyphens."
        )


def _validate_pages(
    chapter_id: str,
    pages: list[Any],
    expected_source_filename: str,
) -> tuple[set[str], set[int]]:
    page_ids: set[str] = set()
    pdf_pages: set[int] = set()
    previous_pdf_page = 0
    for page in pages:
        if not isinstance(page, Mapping):
            raise ContentEngineError(f"Chapter {chapter_id} page records must be objects.")
        page_id = page.get("page_id")
        _validate_id(page_id, "page")
        if page_id in page_ids:
            raise ContentEngineError(f"Chapter {chapter_id} contains duplicate page IDs.")
        page_ids.add(page_id)
        provenance = page.get("source_provenance")
        if not isinstance(provenance, Mapping):
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} requires source provenance."
            )
        source_filename = provenance.get("source_pdf_filename")
        if not isinstance(source_filename, str) or not source_filename.strip():
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} requires a source PDF filename."
            )
        if source_filename != expected_source_filename:
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} source filename does not match the book."
            )
        pdf_page = provenance.get("source_pdf_page_number_1_indexed")
        if isinstance(pdf_page, bool) or not isinstance(pdf_page, int) or pdf_page < 1:
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} requires a positive physical PDF page."
            )
        if pdf_page in pdf_pages:
            raise ContentEngineError(
                f"Chapter {chapter_id} contains duplicate physical PDF pages."
            )
        if pdf_page <= previous_pdf_page:
            raise ContentEngineError(
                f"Chapter {chapter_id} page records are not in physical PDF order."
            )
        previous_pdf_page = pdf_page
        pdf_pages.add(pdf_page)
        extraction_method = provenance.get("text_extraction_method")
        if not isinstance(extraction_method, str) or not extraction_method.strip():
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} requires a text extraction method."
            )
        printed_page = provenance.get("printed_page_number")
        if printed_page is not None and (
            isinstance(printed_page, bool)
            or not isinstance(printed_page, int)
            or printed_page < 1
        ):
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} has an invalid printed page number."
            )
        source_text = page.get("source_text")
        if not isinstance(source_text, Mapping):
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} requires source text fields."
            )
        if not all(
            isinstance(source_text.get(field), str)
            for field in (
                "extracted_text_raw_ordered",
                "extracted_text_normalized_for_search",
            )
        ):
            raise ContentEngineError(
                f"Chapter {chapter_id} page {page_id} source text must contain raw and normalized text."
            )
    return page_ids, pdf_pages


def _serialize(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    except (TypeError, ValueError) as exc:
        raise ContentEngineError(f"Book package contains non-JSON data: {exc}") from exc


def _validate_asset_name(value: Any) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ContentEngineError("Book package asset paths must use relative POSIX paths.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] != "images"
    ):
        raise ContentEngineError(
            "Book package assets must use safe paths beneath the images/ directory."
        )
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContentEngineError(f"Could not read package asset {path.name}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(f"Could not read book package file {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContentEngineError(f"Book package file {path.name} must contain a JSON object.")
    return value
