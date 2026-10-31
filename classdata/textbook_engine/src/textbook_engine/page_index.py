from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from .core import ContentEngineError


@dataclass(frozen=True)
class IndexedPage:
    source_id: str
    class_number: int
    title: str
    page: int
    text: str
    image_path: Path | None
    score: float
    printed_page: int | None = None
    chapter: str | None = None
    heading: str | None = None
    keywords: str = ""
    ocr_text: str = ""
    extraction_method: str = "unknown"
    needs_ocr_review: bool = False


class PageIndexError(Exception):
    pass


def _usable_page_text(text: str) -> str:
    if not text:
        return ""
    private_characters = sum(
        unicodedata.category(character) == "Co" for character in text
    )
    return "" if private_characters / len(text) > 0.01 else text


def package_page_signature(
    manifest_path: Path,
    page_assets: list[tuple[int, Path, Path | None]],
    class_number: int,
    title: str,
) -> str:
    import hashlib

    signature = hashlib.sha256(manifest_path.read_bytes())
    signature.update(f"{class_number}:{title}".encode("utf-8"))
    for page_number, text_path, image_path in page_assets:
        text_stat = text_path.stat()
        image_stat = image_path.stat() if image_path is not None else None
        signature.update(
            f"{page_number}:{text_stat.st_size}:{text_stat.st_mtime_ns}:"
            f"{image_stat.st_size if image_stat else -1}:"
            f"{image_stat.st_mtime_ns if image_stat else -1}".encode("ascii")
        )
    printed_index_path = manifest_path.parent / "printed_page_index.json"
    if printed_index_path.is_file():
        index_stat = printed_index_path.stat()
        signature.update(
            f"printed:{index_stat.st_size}:{index_stat.st_mtime_ns}".encode("ascii")
        )
    return signature.hexdigest()


class PersistentPageIndex:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_books (
                    source_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    page_count INTEGER NOT NULL
                )
                """
            )
            existing_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(indexed_pages)"
                ).fetchall()
            }
            required_columns = {
                "source_id", "class_number", "title", "page", "heading",
                "chapter", "keywords", "text", "ocr_text",
                "extraction_method", "needs_ocr_review", "printed_page",
                "image_path",
            }
            if existing_columns and not required_columns.issubset(existing_columns):
                connection.execute("DROP TABLE indexed_pages")
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS indexed_pages
                USING fts5(
                    source_id UNINDEXED,
                    class_number UNINDEXED,
                    title,
                    page UNINDEXED,
                    heading,
                    chapter,
                    keywords,
                    text,
                    ocr_text UNINDEXED,
                    extraction_method UNINDEXED,
                    needs_ocr_review UNINDEXED,
                    printed_page UNINDEXED,
                    image_path UNINDEXED,
                    tokenize = 'unicode61'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def is_current(
        self,
        source_id: str,
        signature: str,
        page_count: int,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT signature, page_count,
                    (SELECT COUNT(*) FROM indexed_pages
                     WHERE indexed_pages.source_id = indexed_books.source_id) AS indexed_count
                FROM indexed_books
                WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
            return (
                row is not None
                and row["signature"] == signature
                and row["page_count"] == page_count
                and row["indexed_count"] == page_count
            )

    def get_cached_response(self, cache_key: str) -> dict[str, object] | None:
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload, created_at FROM answer_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            connection.execute(
                "DELETE FROM answer_cache WHERE created_at < ?",
                (now - 30 * 24 * 60 * 60,),
            )
        if row is None or row["created_at"] < now - 30 * 24 * 60 * 60:
            return None
        value = json.loads(row["payload"])
        if not isinstance(value, dict):
            raise ValueError("Cached chat response is not a JSON object.")
        return value

    def cache_response(self, cache_key: str, payload: dict[str, object]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        now = int(time.time())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO answer_cache(cache_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (cache_key, serialized, now),
            )
            connection.execute(
                """
                DELETE FROM answer_cache
                WHERE cache_key IN (
                    SELECT cache_key FROM answer_cache
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET 5000
                )
                """
            )

    def replace_book(
        self,
        source_id: str,
        class_number: int,
        title: str,
        signature: str,
        pages: list[tuple[int, str, Path | None]],
        *,
        page_metadata: dict[int, dict[str, object]] | None = None,
    ) -> None:
        page_metadata = page_metadata or {}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM indexed_pages WHERE source_id = ?",
                (source_id,),
            )
            connection.executemany(
                """
                INSERT INTO indexed_pages (
                    source_id, class_number, title, page, heading, chapter,
                    keywords, text, ocr_text, extraction_method,
                    needs_ocr_review, printed_page, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        source_id,
                        class_number,
                        title,
                        page_number,
                        str(page_metadata.get(page_number, {}).get("heading") or ""),
                        str(page_metadata.get(page_number, {}).get("chapter") or ""),
                        str(page_metadata.get(page_number, {}).get("keywords") or ""),
                        text,
                        str(page_metadata.get(page_number, {}).get("ocr_text") or ""),
                        str(
                            page_metadata.get(page_number, {}).get(
                                "extraction_method",
                                "unknown",
                            )
                        ),
                        int(bool(
                            page_metadata.get(page_number, {}).get(
                                "needs_ocr_review",
                                False,
                            )
                        )),
                        page_metadata.get(page_number, {}).get("printed_page"),
                        str(image_path) if image_path is not None else "",
                    )
                    for page_number, text, image_path in pages
                ],
            )
            connection.execute(
                """
                INSERT INTO indexed_books(source_id, signature, page_count)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    signature = excluded.signature,
                    page_count = excluded.page_count
                """,
                (source_id, signature, len(pages)),
            )

    def index_processed_package(
        self,
        manifest_path: Path,
        class_number: int,
        title: str,
    ) -> None:
        package_dir = manifest_path.parent.resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageIndexError(f"Could not read the processed PDF manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise PageIndexError("Processed PDF manifest must contain an object.")
        source_id = manifest.get("source_id")
        source = manifest.get("source")
        assets = manifest.get("assets")
        if (
            not isinstance(source_id, str)
            or not isinstance(source, dict)
            or not isinstance(assets, list)
        ):
            raise PageIndexError("Processed PDF manifest has invalid page metadata.")
        source_sha256 = source.get("sha256")

        page_assets: list[tuple[int, Path, Path | None]] = []
        pages: list[tuple[int, str, Path | None]] = []
        metadata: dict[int, dict[str, object]] = {}
        for asset in assets:
            if not isinstance(asset, dict) or asset.get("kind") != "pdf_page":
                continue
            page_number = asset.get("page")
            text_value = asset.get("text_path")
            image_value = asset.get("image_path")
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
                or not isinstance(text_value, str)
                or not isinstance(image_value, str)
            ):
                raise PageIndexError("Processed PDF manifest has an invalid page asset.")
            text_path = (package_dir / text_value).resolve()
            image_path = (package_dir / image_value).resolve()
            try:
                text_path.relative_to(package_dir)
                image_path.relative_to(package_dir)
                text = text_path.read_text(encoding="utf-8")
            except (ValueError, OSError, UnicodeDecodeError) as exc:
                raise PageIndexError(
                    f"Could not index processed PDF page {page_number}: {exc}"
                ) from exc
            text = _usable_page_text(text)
            available_image = image_path if image_path.is_file() else None
            page_assets.append((page_number, text_path, available_image))
            pages.append((page_number, text, available_image))
            method = asset.get("text_extraction_method", "unknown")
            tokens = Counter(re.findall(r"[\w]+", text, flags=re.UNICODE))
            metadata[page_number] = {
                "heading": next(
                    (line.strip()[:200] for line in text.splitlines() if line.strip()),
                    "",
                ),
                "keywords": " ".join(
                    item[0] for item in tokens.most_common(24)
                ),
                "ocr_text": text if method == "tesseract_ocr" else "",
                "extraction_method": method,
                "needs_ocr_review": bool(asset.get("needs_ocr_review")),
            }
        if not pages:
            raise PageIndexError("Processed PDF contains no pages to index.")
        page_count = manifest.get("source", {}).get("pdf_page_count")
        chapter_by_page: dict[int, str] = {}
        if isinstance(page_count, int) and not isinstance(page_count, bool):
            page_texts = [next(
                (text for number, text, _ in pages if number == page_number),
                "",
            ) for page_number in range(1, page_count + 1)]
            try:
                from .book_pipeline import _discover_chapters

                chapter_spans = _discover_chapters(page_texts)
            except ContentEngineError:
                chapter_spans = []
            for span in chapter_spans:
                chapter_label = f"{span.chapter_id}: {span.title}"
                for page_number in range(span.start_page + 1, span.end_page + 1):
                    chapter_by_page[page_number] = chapter_label
        printed_page_by_pdf: dict[int, int] = {}
        try:
            printed_index = json.loads(
                (package_dir / "printed_page_index.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            printed_index = {}
        if (
            isinstance(printed_index, dict)
            and printed_index.get("source_id") == source_id
            and (
                not isinstance(source_sha256, str)
                or printed_index.get("source_sha256") == source_sha256
            )
            and isinstance(printed_index.get("pages"), list)
        ):
            candidates: dict[int, set[int]] = {}
            for entry in printed_index["pages"]:
                if (
                    not isinstance(entry, dict)
                    or entry.get("status") != "matched"
                    or isinstance(entry.get("pdf_page"), bool)
                    or not isinstance(entry.get("pdf_page"), int)
                    or isinstance(entry.get("printed_page"), bool)
                    or not isinstance(entry.get("printed_page"), int)
                ):
                    continue
                candidates.setdefault(entry["pdf_page"], set()).add(
                    entry["printed_page"]
                )
            printed_page_by_pdf = {
                pdf_page: next(iter(printed_pages))
                for pdf_page, printed_pages in candidates.items()
                if len(printed_pages) == 1
            }
        for page_number, page_metadata in metadata.items():
            page_metadata["chapter"] = chapter_by_page.get(page_number, "")
            page_metadata["printed_page"] = printed_page_by_pdf.get(page_number)
        try:
            signature = package_page_signature(
                manifest_path,
                page_assets,
                class_number,
                title,
            )
            self.replace_book(
                source_id,
                class_number,
                title,
                signature,
                pages,
                page_metadata=metadata,
            )
        except (OSError, sqlite3.Error) as exc:
            raise PageIndexError(f"Could not build the persistent page index: {exc}") from exc

    @staticmethod
    def _match_expression(query: str) -> str:
        tokens = Counter(
            token.casefold()
            for token in re.findall(r"[\w]+", query, flags=re.UNICODE)
        )
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(
        self,
        query: str,
        class_number: int,
        source_ids: list[str],
        limit: int,
    ) -> list[IndexedPage]:
        match_expression = self._match_expression(query)
        if not match_expression or not source_ids or limit < 1:
            return []
        placeholders = ",".join("?" for _ in source_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT source_id, class_number, title, page, heading, chapter,
                    keywords, text, ocr_text, extraction_method,
                    needs_ocr_review, printed_page, image_path,
                    bm25(
                        indexed_pages, 0.0, 0.0, 1.0, 0.0, 2.0, 1.5,
                        1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ) AS rank
                FROM indexed_pages
                WHERE indexed_pages MATCH ?
                    AND class_number = ?
                    AND source_id IN ({placeholders})
                ORDER BY rank, CAST(page AS INTEGER), source_id
                LIMIT ?
                """,
                (match_expression, class_number, *source_ids, limit),
            ).fetchall()
        return [
            IndexedPage(
                source_id=row["source_id"],
                class_number=row["class_number"],
                title=row["title"],
                page=int(row["page"]),
                text=row["text"],
                image_path=Path(row["image_path"]) if row["image_path"] else None,
                score=max(0.0, -float(row["rank"])),
                printed_page=(
                    int(row["printed_page"])
                    if row["printed_page"] is not None
                    else None
                ),
                chapter=row["chapter"] or None,
                heading=row["heading"] or None,
                keywords=row["keywords"] or "",
                ocr_text=row["ocr_text"] or "",
                extraction_method=row["extraction_method"] or "unknown",
                needs_ocr_review=bool(row["needs_ocr_review"]),
            )
            for row in rows
        ]

    def get_page(self, source_id: str, page_number: int) -> IndexedPage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_id, class_number, title, page, heading, chapter,
                    keywords, text, ocr_text, extraction_method,
                    needs_ocr_review, printed_page, image_path
                FROM indexed_pages
                WHERE source_id = ? AND CAST(page AS INTEGER) = ?
                LIMIT 1
                """,
                (source_id, page_number),
            ).fetchone()
        if row is None:
            return None
        return IndexedPage(
            source_id=row["source_id"],
            class_number=row["class_number"],
            title=row["title"],
            page=int(row["page"]),
            text=row["text"],
            image_path=Path(row["image_path"]) if row["image_path"] else None,
            score=1_000_000.0,
            printed_page=(
                int(row["printed_page"])
                if row["printed_page"] is not None
                else None
            ),
            chapter=row["chapter"] or None,
            heading=row["heading"] or None,
            keywords=row["keywords"] or "",
            ocr_text=row["ocr_text"] or "",
            extraction_method=row["extraction_method"] or "unknown",
            needs_ocr_review=bool(row["needs_ocr_review"]),
        )

    def pages_for_class(self, class_number: int, source_ids: list[str]) -> list[IndexedPage]:
        if not source_ids:
            return []
        placeholders = ",".join("?" for _ in source_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT source_id, class_number, title, page, heading, chapter,
                    keywords, text, ocr_text, extraction_method,
                    needs_ocr_review, printed_page, image_path
                FROM indexed_pages
                WHERE class_number = ? AND source_id IN ({placeholders})
                ORDER BY source_id, CAST(page AS INTEGER)
                """,
                (class_number, *source_ids),
            ).fetchall()
        return [
            IndexedPage(
                source_id=row["source_id"],
                class_number=row["class_number"],
                title=row["title"],
                page=int(row["page"]),
                text=row["text"],
                image_path=Path(row["image_path"]) if row["image_path"] else None,
                score=0.0,
                printed_page=(
                    int(row["printed_page"])
                    if row["printed_page"] is not None
                    else None
                ),
                chapter=row["chapter"] or None,
                heading=row["heading"] or None,
                keywords=row["keywords"] or "",
                ocr_text=row["ocr_text"] or "",
                extraction_method=row["extraction_method"] or "unknown",
                needs_ocr_review=bool(row["needs_ocr_review"]),
            )
            for row in rows
        ]
