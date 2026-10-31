from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pymupdf
from PIL import Image

from .chat import (
    ChatError,
    load_local_env,
)
from .core import ContentEngineError, sha256_file
from .gemini import (
    CHAPTER_PROMPT_VERSION,
    DEFAULT_GEMINI_MODEL,
    complete_chapter_chunk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
MAX_CHUNK_CHARACTERS = 9_000
IMAGE_DPI = 96
CHAPTER_MARKER = re.compile(
    r"\b(?P<kind>chapter|unit|lesson)\s*(?:no\.?\s*)?[-.:]?\s*"
    r"(?P<number>\d{1,2}|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:[:.)\-–—]\s*(?P<title>.*))?$",
    flags=re.IGNORECASE,
)
TITLED_UNIT_MARKER = re.compile(
    r"^\s*(?P<kind>chapter|unit|lesson)\s+"
    r"(?P<title>[A-Z][A-Z0-9'’&,. \-]{2,}?)\s+"
    r"(?P<number>\d{1,2}|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*$",
    flags=re.IGNORECASE,
)
INLINE_TITLED_MARKER = re.compile(
    r"^\s*(?P<title>[A-Z][A-Za-z0-9'’&,. \-]{1,40}?)\s+"
    r"(?P<kind>chapter|unit|lesson)\s*(?:no\.?\s*)?[-.:]?\s*"
    r"(?P<number>\d{1,2}|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*$",
    flags=re.IGNORECASE,
)
UNIT_TITLE_LINE = re.compile(
    r"^\s*(?P<title>[A-Z][A-Z0-9'’&,. \-]{2,}?)\s+"
    r"(?P<number>\d{1,2}|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*$",
    flags=re.IGNORECASE,
)
NUMBERED_CHAPTER_PREFIX = re.compile(
    r"^\s*(?P<number>\d{1,2})[.)]?\s+"
    r"(?P<title>[A-Za-z][A-Za-z0-9'’&,. \-]*)\s*$"
)
NUMBERED_CHAPTER_SUFFIX = re.compile(
    r"^\s*(?P<title>[A-Za-z][A-Za-z0-9'’&,. \-]*?)\s+"
    r"(?P<number>\d{1,2})\s*$"
)
NUMBERED_CHAPTER_MIDDLE = re.compile(
    r"^\s*(?P<prefix>[A-Za-z][A-Za-z0-9'’&,. \-]*?)\s+"
    r"(?P<number>\d{1,2})\s*[-.)]?\s+"
    r"(?P<suffix>[A-Za-z][A-Za-z0-9'’&,. \-]*)\s*$"
)
NON_CHAPTER_TITLE_TERMS = {"activity", "class", "table"}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
ROMAN_NUMBERS = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}
GENERIC_HEADING = re.compile(
    r"^(?:contents|table of contents|index|scert(?:,?\s+telangana)?|"
    r"government['’]s gift for students['’] progress)$",
    flags=re.IGNORECASE,
)

Completion = Callable[[list[dict[str, Any]], str, str], str]


def _load_gemini_api_key() -> str:
    try:
        load_local_env(PROJECT_ROOT)
    except ChatError as exc:
        raise ContentEngineError(str(exc)) from exc
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.casefold() == "your_google_gemini_api_key":
        raise ContentEngineError(
            "Gemini API key is missing. Add GEMINI_API_KEY=your_key to the project .env file."
        )
    return api_key


def _chapter_completion(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
) -> str:
    return complete_chapter_chunk(messages, model, api_key)


def _number_value(value: str) -> int | None:
    normalized = value.casefold()
    if normalized.isdecimal():
        return int(normalized)
    if normalized in NUMBER_WORDS:
        return NUMBER_WORDS[normalized]
    return ROMAN_NUMBERS.get(normalized)


def _numbered_chapter_marker(text: str) -> dict[str, Any] | None:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    heading_lines = lines[:4]
    if not heading_lines or re.search(
        r"\b(?:chapter|unit|lesson)\b",
        " ".join(heading_lines[:3]),
        flags=re.IGNORECASE,
    ):
        return None
    for line_count in range(1, min(2, len(heading_lines)) + 1):
        candidate = " ".join(heading_lines[:line_count])
        match = NUMBERED_CHAPTER_PREFIX.fullmatch(candidate)
        if match is not None:
            title = match.group("title").strip(" -")
            number = int(match.group("number"))
        else:
            match = NUMBERED_CHAPTER_SUFFIX.fullmatch(candidate)
            if match is not None:
                title = match.group("title").strip(" -")
                number = int(match.group("number"))
            else:
                match = NUMBERED_CHAPTER_MIDDLE.fullmatch(candidate)
                if match is None:
                    continue
                title = (
                    f"{match.group('prefix').strip(' -')} "
                    f"{match.group('suffix').strip(' -')}"
                )
                number = int(match.group("number"))
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", title)
        if (
            words
            and len(words) <= 6
            and len(title) <= 100
            and not NON_CHAPTER_TITLE_TERMS.intersection(
                word.casefold() for word in words
            )
            and not title.endswith((".", "!", "?"))
            and not re.search(r"\b(?:chapter|unit|lesson)\s*$", title, flags=re.IGNORECASE)
        ):
            return {"kind": "chapter", "number": number, "title": title}
    return None


def _markers_in_text(text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    lines = text.splitlines()
    numbered_marker = _numbered_chapter_marker(text)
    if numbered_marker is not None:
        markers.append(numbered_marker)
    for line_index, line in enumerate(lines):
        titled_match = TITLED_UNIT_MARKER.fullmatch(line)
        if (
            titled_match is None
            and line_index + 1 < len(lines)
            and re.match(r"^\s*(?:chapter|unit|lesson)\s+\S", line, flags=re.IGNORECASE)
        ):
            titled_match = TITLED_UNIT_MARKER.fullmatch(
                f"{line.strip()} {lines[line_index + 1].strip()}"
            )
        inline_match = (
            INLINE_TITLED_MARKER.fullmatch(line)
            if titled_match is None
            else None
        )
        if (
            titled_match is not None
            and not line.lstrip().split(None, 1)[0][:1].isupper()
        ):
            titled_match = None
        if (
            titled_match is not None
            and titled_match.group("title").strip(" -").casefold()
            in {"chapter", "unit", "lesson"}
        ):
            titled_match = None
        if titled_match is None and inline_match is not None:
            number = _number_value(inline_match.group("number"))
            if number is not None:
                markers.append({
                    "kind": inline_match.group("kind").casefold(),
                    "number": number,
                    "title": inline_match.group("title").strip() or None,
                })
            continue
        matches = (
            [titled_match]
            if titled_match is not None
            else [
                match
                for match in CHAPTER_MARKER.finditer(line)
                if not line[:match.start()].strip()
                or re.fullmatch(
                    r"(?:chapter|unit|lesson)",
                    line[:match.start()].strip(),
                    flags=re.IGNORECASE,
                )
            ]
        )
        if (
            not matches
            and line_index <= 2
            and line.strip().casefold() in {"chapter", "unit", "lesson"}
            and line_index + 1 < len(lines)
        ):
            kind = line.strip().casefold()
            next_line = lines[line_index + 1].strip()
            titled_match = UNIT_TITLE_LINE.fullmatch(next_line)
            numbered_match = NUMBERED_CHAPTER_PREFIX.fullmatch(next_line)
            if titled_match is not None:
                number = _number_value(titled_match.group("number"))
                title = titled_match.group("title").strip()
            elif numbered_match is not None:
                number = _number_value(numbered_match.group("number"))
                title = numbered_match.group("title").strip()
            elif (
                line_index + 2 < len(lines)
                and re.fullmatch(r"\d{1,2}", lines[line_index + 2].strip())
            ):
                number = int(lines[line_index + 2].strip())
                title = next_line
            else:
                number = None
                title = ""
            if number is not None and title:
                markers.append({
                    "kind": kind,
                    "number": number,
                    "title": title,
                })
            continue
        if len(matches) > 1:
            matches = matches[-1:]
        for match in matches:
            if match is None:
                continue
            number = _number_value(match.group("number"))
            if number is None:
                continue
            markers.append({
                "kind": match.group("kind").casefold(),
                "number": number,
                "title": (match.groupdict().get("title") or "").strip() or None,
            })
    return markers


def _is_contents_page(text: str, markers: list[dict[str, Any]]) -> bool:
    if re.search(
        r"\b(?:table\s+of\s+)?contents\b|\bindex\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    return len({(marker["kind"], marker["number"]) for marker in markers}) > 1


def _title_from_page(text: str) -> str | None:
    candidates: list[str] = []
    for line in text.splitlines():
        candidate = " ".join(line.split()).strip(" \t.:;,-–—")
        candidate = re.sub(r"\s+\d{1,3}$", "", candidate).strip()
        if (
            len(candidate) < 3
            or GENERIC_HEADING.fullmatch(candidate)
            or CHAPTER_MARKER.fullmatch(candidate)
            or re.match(r"^(?:chapter|unit|lesson)\b", candidate, flags=re.IGNORECASE)
            or re.fullmatch(r"(?:\d+|[ivxlcdm]+)", candidate, flags=re.IGNORECASE)
        ):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    uppercase = [
        candidate
        for candidate in candidates
        if any(char.isalpha() for char in candidate)
        and candidate == candidate.upper()
        and len(candidate.split()) >= 2
    ]
    return uppercase[0] if uppercase else (candidates[0] if candidates else None)


def _chapter_span(
    pages: list[str],
    chapter_number: int,
) -> tuple[int, int, str, str | None]:
    page_markers = [_markers_in_text(text) for text in pages]
    candidates: list[tuple[int, dict[str, Any]]] = []
    for page_index, markers in enumerate(page_markers):
        if _is_contents_page(pages[page_index], markers):
            continue
        candidates.extend((page_index, marker) for marker in markers)
    marker_counts = {
        candidate: sum(
            marker["kind"] == candidate
            for _, marker in candidates
        )
        for candidate in ("chapter", "unit", "lesson")
    }
    kind = max(
        marker_counts,
        key=lambda candidate: (
            marker_counts[candidate],
            ("chapter", "unit", "lesson").index(candidate),
        ),
    )
    if marker_counts[kind] == 0:
        kind = None
    if kind is None:
        raise ContentEngineError(
            "Could not confidently identify chapter boundaries. The PDF needs explicit "
            "CHAPTER, UNIT, or LESSON headings."
        )

    candidates = [
        (page_index, marker)
        for page_index, marker in candidates
        if marker["kind"] == kind
    ]

    starts = [
        (page_index, marker)
        for page_index, marker in candidates
        if marker["number"] == chapter_number
    ]
    if not starts:
        raise ContentEngineError(
            f"Could not find a reliable {kind.upper()} {chapter_number} heading in the PDF."
        )
    start_page, start_marker = starts[0]
    end_candidates = [
        page_index
        for page_index, marker in candidates
        if page_index > start_page and marker["number"] > chapter_number
    ]
    if not end_candidates:
        raise ContentEngineError(
            f"Found {kind.upper()} {chapter_number}, but could not identify where it ends. "
            "The next numbered chapter/unit/lesson heading is missing."
        )
    end_page = min(end_candidates)
    title = start_marker["title"] or _title_from_page(pages[start_page])
    return start_page, end_page, kind, title


def _chapter_index_evidence(pages: list[str]) -> str:
    contents_pages = [
        f"PDF page {page_number}:\n{text[:2_500]}"
        for page_number, text in enumerate(pages[:40], start=1)
        if re.search(r"\b(?:contents|index)\b", text, flags=re.IGNORECASE)
    ]
    heading_samples: list[str] = []
    for page_number, text in enumerate(pages, start=1):
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        if lines:
            heading_samples.append(
                f"PDF page {page_number}: "
                + " | ".join(line[:140] for line in lines[:4])
            )
    return (
        "Contents/index excerpts (page numbers here are physical PDF pages):\n"
        + ("\n\n".join(contents_pages) or "No explicit contents/index page was found.")
        + "\n\nTop-of-page text samples (physical PDF page numbers):\n"
        + "\n".join(heading_samples)
    )[:120_000]


def _chapter_span_from_index(
    index_json: str,
    pages: list[str],
    chapter_number: int,
) -> tuple[int, int, str, str]:
    try:
        index_data = json.loads(index_json)
    except json.JSONDecodeError as exc:
        raise ContentEngineError(
            "Gemini returned malformed chapter-index JSON."
        ) from exc
    entries = index_data.get("chapters") if isinstance(index_data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ContentEngineError("Gemini did not identify any textbook chapters.")

    chapters: list[dict[str, Any]] = []
    seen_numbers: set[int] = set()
    seen_pages: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContentEngineError("Gemini returned an invalid chapter-index entry.")
        number = entry.get("number")
        title = entry.get("title")
        start_page = entry.get("start_pdf_page")
        marker_kind = entry.get("marker_kind")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            or not isinstance(title, str)
            or not title.strip()
            or isinstance(start_page, bool)
            or not isinstance(start_page, int)
            or not 1 <= start_page <= len(pages)
            or marker_kind not in {"chapter", "unit", "lesson"}
            or number in seen_numbers
            or start_page in seen_pages
        ):
            raise ContentEngineError(
                "Gemini returned invalid or duplicate chapter numbers or PDF pages."
            )
        seen_numbers.add(number)
        seen_pages.add(start_page)
        chapters.append({
            "number": number,
            "title": title.strip(),
            "start_pdf_page": start_page,
            "marker_kind": marker_kind,
        })
    chapters.sort(key=lambda entry: entry["start_pdf_page"])

    requested = next(
        (entry for entry in chapters if entry["number"] == chapter_number),
        None,
    )
    if requested is None:
        raise ContentEngineError(
            f"Gemini could not locate chapter {chapter_number} in the textbook index."
        )
    start_page = requested["start_pdf_page"] - 1
    next_page = next(
        (
            entry["start_pdf_page"] - 1
            for entry in chapters
            if entry["start_pdf_page"] > requested["start_pdf_page"]
        ),
        len(pages),
    )
    if next_page <= start_page:
        raise ContentEngineError(
            f"Gemini returned an invalid page range for chapter {chapter_number}."
        )
    return (
        start_page,
        next_page,
        requested["marker_kind"],
        requested["title"],
    )


def _split_chunks(text: str, maximum: int = MAX_CHUNK_CHARACTERS) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start:
                end = boundary
        chunks.append(text[start:end])
        start = end
    return chunks


def _nearby_text(page_text: str, maximum: int = 500) -> str | None:
    normalized = " ".join(page_text.split())
    return normalized[:maximum] or None


def _page_has_vector_visual(page: pymupdf.Page) -> bool:
    page_area = max(1.0, page.rect.width * page.rect.height)
    return any(
        drawing["rect"].get_area() >= page_area * 0.02
        for drawing in page.get_drawings()
        if drawing.get("rect") is not None
    )


def _extract_chapter_images(
    document: pymupdf.Document,
    page_indexes: range,
    staging_dir: Path,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    image_number = 0
    visual_number = 0
    embedded_by_digest: dict[str, dict[str, Any]] = {}
    scale = IMAGE_DPI / 72
    for page_index in page_indexes:
        page = document.load_page(page_index)
        page_text = page.get_text("text", sort=True)
        context = _nearby_text(page_text)
        image_blocks = [
            block
            for block in page.get_text("dict")["blocks"]
            if block.get("type") == 1 and isinstance(block.get("image"), bytes)
        ]
        has_vector_visual = _page_has_vector_visual(page)
        if image_blocks or has_vector_visual:
            visual_number += 1
            visual_path = staging_dir / f"page_visual_{visual_number:03d}.png"
            try:
                page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).save(visual_path)
            except (RuntimeError, ValueError, OSError) as exc:
                raise ContentEngineError(
                    f"Could not render a chapter visual from the PDF: {exc}"
                ) from exc
            images.append({
                "id": f"visual_{visual_number:03d}",
                "path": f"images/{visual_path.name}",
                "kind": "page_visual",
                "description": None,
                "visual_understanding": "not_performed",
                "related_text": [context] if context else [],
            })

        for block in image_blocks:
            image_bytes = block["image"]
            digest = hashlib.sha256(image_bytes).hexdigest()
            existing_image = embedded_by_digest.get(digest)
            if existing_image is not None:
                if context and context not in existing_image["related_text"]:
                    existing_image["related_text"].append(context)
                continue
            image_number += 1
            image_path = staging_dir / f"image_{image_number:03d}.png"
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    image.convert("RGBA" if "A" in image.getbands() else "RGB").save(image_path)
            except (OSError, SyntaxError, ValueError) as exc:
                raise ContentEngineError(
                    f"Could not decode an embedded image in the selected chapter: {exc}"
                ) from exc
            image_record = {
                "id": f"image_{image_number:03d}",
                "path": f"images/{image_path.name}",
                "kind": "embedded_image",
                "description": None,
                "visual_understanding": "not_performed",
                "related_text": [context] if context else [],
            }
            images.append(image_record)
            embedded_by_digest[digest] = image_record
    return images


def _string_list(value: Any, field: str, chunk_number: int) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContentEngineError(
            f"Gemini returned malformed chapter JSON: {field} must be a list of strings "
            f"(chunk {chunk_number})."
        )
    return [item.strip() for item in value if item.strip()]


def _decode_chunk_response(response: str, chunk_number: int) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ContentEngineError(
            f"Gemini returned malformed JSON for chapter chunk {chunk_number}."
        ) from exc
    if not isinstance(value, dict):
        raise ContentEngineError(
            f"Gemini returned malformed chapter JSON for chunk {chunk_number}; expected an object."
        )
    title = value.get("chapter_title")
    introduction = value.get("introduction")
    if title is not None and not isinstance(title, str):
        raise ContentEngineError("Gemini returned malformed chapter JSON: chapter_title must be text or null.")
    if introduction is not None and not isinstance(introduction, str):
        raise ContentEngineError("Gemini returned malformed chapter JSON: introduction must be text or null.")
    objectives = _string_list(value.get("learning_objectives", []), "learning_objectives", chunk_number)
    sections = value.get("sections", [])
    if not isinstance(sections, list):
        raise ContentEngineError(
            f"Gemini returned malformed chapter JSON: sections must be a list (chunk {chunk_number})."
        )
    validated_sections: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            raise ContentEngineError(
                f"Gemini returned malformed chapter JSON: section must be an object (chunk {chunk_number})."
            )
        section_title = section.get("title")
        explanation = section.get("explanation")
        if section_title is not None and not isinstance(section_title, str):
            raise ContentEngineError("Gemini returned malformed chapter JSON: section title must be text or null.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ContentEngineError(
                f"Gemini returned malformed chapter JSON: section explanation is missing (chunk {chunk_number})."
            )
        definitions = section.get("definitions", [])
        if not isinstance(definitions, list) or any(
            not isinstance(definition, dict)
            or not isinstance(definition.get("term"), str)
            or not isinstance(definition.get("meaning"), str)
            for definition in definitions
        ):
            raise ContentEngineError(
                f"Gemini returned malformed chapter JSON: definitions are invalid (chunk {chunk_number})."
            )
        validated_sections.append({
            "title": section_title.strip() if section_title else None,
            "explanation": explanation.strip(),
            "definitions": [
                {"term": item["term"].strip(), "meaning": item["meaning"].strip()}
                for item in definitions
            ],
            "examples": _string_list(section.get("examples", []), "examples", chunk_number),
            "important_facts": _string_list(
                section.get("important_facts", []), "important_facts", chunk_number
            ),
            "formulas": _string_list(section.get("formulas", []), "formulas", chunk_number),
            "questions": _string_list(section.get("questions", []), "questions", chunk_number),
        })
    return {
        "chapter_title": title.strip() if title else None,
        "introduction": introduction.strip() if introduction else None,
        "learning_objectives": objectives,
        "sections": validated_sections,
    }


def _subject_from_filename(source: Path) -> str:
    subject = re.sub(r"(?i)^class[_\s-]*\d+[_\s-]*", "", source.stem)
    subject = re.sub(r"[_\s-]+", " ", subject).strip()
    return subject or source.stem


def _class_from_filename(source: Path) -> int | None:
    match = re.search(r"(?i)class[_\s-]*0*(\d+)", source.stem)
    return int(match.group(1)) if match else None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "unknown_subject"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        json.loads(serialized)
        path.write_text(serialized, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ContentEngineError(f"Could not write valid JSON to {path}: {exc}") from exc


def _save_chapter_output(
    source: Path,
    chapter_output: Path,
    chapter_number: int,
    start_page: int,
    end_page: int,
    marker_kind: str,
    chapter_title: str | None,
    subject: str,
    class_number: int | None,
    source_text: str,
    model: str,
    chunks: list[str],
    introduction: str | None,
    objectives: list[str],
    sections: list[dict[str, Any]],
    questions: list[str],
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    try:
        document = pymupdf.open(source)
    except (pymupdf.FileDataError, ValueError) as exc:
        raise ContentEngineError(f"Could not reopen PDF {source}: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="chapter-images-") as staging_name:
        staging_dir = Path(staging_name)
        try:
            if document.needs_pass:
                raise ContentEngineError("Password-protected PDFs are not supported.")
            image_records = _extract_chapter_images(
                document,
                range(start_page, end_page),
                staging_dir,
            )
        finally:
            document.close()

        chapter_json = {
            "schema_version": "1.0",
            "chapter": {
                "number": chapter_number,
                "title": chapter_title,
                "subject": subject,
                "class": class_number,
                "introduction": introduction,
                "learning_objectives": objectives,
            },
            "source": {
                "filename": source.name,
                "sha256": source_hash,
                "chapter_marker": marker_kind,
            },
            "generation_provenance": {
                "provider": "google_gemini",
                "api": "interactions",
                "model_id": model,
                "prompt_version": CHAPTER_PROMPT_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "human_reviewed": False,
            },
            "original_text": source_text,
            "sections": sections,
            "student_questions": questions,
            "images": image_records,
            "warnings": [
                "Image descriptions and labels were not generated; no vision model was used."
            ] if image_records else [],
        }
        for record in image_records:
            if not (staging_dir / Path(record["path"]).name).is_file():
                raise ContentEngineError(
                    "A chapter JSON image reference does not point to an extracted image."
                )
        serialized = json.dumps(chapter_json, ensure_ascii=False, indent=2) + "\n"
        try:
            validated_json = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ContentEngineError(f"Generated chapter JSON is invalid: {exc}") from exc

        image_directory = chapter_output / "images"
        image_directory.mkdir(parents=True, exist_ok=True)
        for record in image_records:
            image_name = Path(record["path"]).name
            shutil.copy2(staging_dir / image_name, image_directory / image_name)
        chapter_output.mkdir(parents=True, exist_ok=True)
        (chapter_output / "extracted_text.txt").write_text(
            source_text + "\n",
            encoding="utf-8",
        )
        (chapter_output / "chapter.json").write_text(serialized, encoding="utf-8")
        report = {
            "status": "complete",
            "source": source.name,
            "source_sha256": source_hash,
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "boundary_marker": marker_kind,
            "content_pages_processed": end_page - start_page,
            "text_characters": len(source_text),
            "text_chunks_sent_to_gemini": len(chunks),
            "generation_model": model,
            "images_extracted": len(image_records),
            "image_understanding": "not_performed",
            "output": str(chapter_output),
        }
        _write_json(chapter_output / "processing_report.json", report)
        if not isinstance(validated_json, dict):
            raise ContentEngineError("Generated chapter JSON must be an object.")
        missing_images = [
            record["path"]
            for record in validated_json["images"]
            if not (chapter_output / record["path"]).is_file()
        ]
        if missing_images:
            raise ContentEngineError(
                "Generated chapter JSON references missing image files: "
                + ", ".join(missing_images)
            )
    return report


def _generate_chapter_content(
    source: Path,
    chapter_number: int,
    marker_kind: str,
    detected_title: str | None,
    source_text: str,
    model: str,
    *,
    api_key: str | None = None,
    completion: Completion | None = None,
) -> dict[str, Any]:
    chunks = _split_chunks(source_text)
    if not chunks:
        raise ContentEngineError(
            f"{marker_kind.upper()} {chapter_number} has no extractable text."
        )
    if completion is None:
        api_key = api_key.strip() if api_key and api_key.strip() else _load_gemini_api_key()
    api_completion = completion or _chapter_completion
    chapter_title = detected_title
    introduction: str | None = None
    objectives: list[str] = []
    sections: list[dict[str, Any]] = []
    questions: list[str] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful textbook learning-content extractor. Use only the "
                    "provided source text. Preserve all concepts and do not invent facts, "
                    "labels, examples, objectives, or formulas. Use empty strings for "
                    "unavailable text and empty arrays for unavailable lists. Return one "
                    "valid JSON object with keys "
                    "chapter_title, introduction, learning_objectives, and sections. "
                    "Each section must have title, explanation, definitions "
                    "([{term,meaning}]), examples, important_facts, formulas, and questions. "
                    "Explanations should be clear and student-friendly while preserving the "
                    "textbook meaning. Use plain text, not Markdown. Do not claim to have seen "
                    "or understood images."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Book subject: {_subject_from_filename(source)}\n"
                    f"Requested chapter/unit/lesson: {chapter_number}\n"
                    f"Detected heading: {detected_title or 'not reliably available'}\n"
                    f"Source text chunk {chunk_number} of {len(chunks)}:\n{chunk}"
                ),
            },
        ]
        response = api_completion(messages, model, api_key or "")
        result = _decode_chunk_response(response, chunk_number)
        chapter_title = chapter_title or result["chapter_title"]
        introduction = introduction or result["introduction"]
        for objective in result["learning_objectives"]:
            if objective not in objectives:
                objectives.append(objective)
        for section in result["sections"]:
            section["source_text"] = chunk
            sections.append(section)
            for question in section["questions"]:
                if question not in questions:
                    questions.append(question)
    if not sections:
        raise ContentEngineError(
            f"Gemini did not return any explained sections for {marker_kind.upper()} {chapter_number}."
        )
    return {
        "chapter_title": chapter_title,
        "introduction": introduction,
        "learning_objectives": objectives,
        "sections": sections,
        "student_questions": questions,
        "chunks": chunks,
    }


def process_chapter(
    source: Path,
    chapter_number: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    language: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    completion: Completion | None = None,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source.suffix.casefold() != ".pdf":
        raise ContentEngineError("Chapter processing accepts PDF files only.")
    if not source.is_file():
        raise ContentEngineError(f"Source PDF does not exist: {source}")
    if isinstance(chapter_number, bool) or not isinstance(chapter_number, int) or chapter_number < 1:
        raise ContentEngineError("Chapter number must be a positive whole number.")
    model = (model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()

    from .pipeline import process_and_load_pdf_pages

    _, processed_pages = process_and_load_pdf_pages(
        source,
        data_dir,
        language=language,
    )
    page_texts = [page["text"] for page in processed_pages]
    start_page, end_page, marker_kind, detected_title = _chapter_span(
        page_texts,
        chapter_number,
    )
    source_text = "\n\n".join(
        text.strip() for text in page_texts[start_page:end_page] if text.strip()
    )
    if not source_text:
        raise ContentEngineError(
            f"{marker_kind.upper()} {chapter_number} has no extractable text."
        )

    generated = _generate_chapter_content(
        source,
        chapter_number,
        marker_kind,
        detected_title,
        source_text,
        model,
        api_key=api_key,
        completion=completion,
    )

    class_number = _class_from_filename(source)
    subject = _subject_from_filename(source)
    chapter_output = (
        output_dir / f"class_{class_number}"
        if class_number is not None
        else output_dir / "class_unknown"
    ) / _slug(subject) / f"chapter_{chapter_number}"
    return _save_chapter_output(
        source,
        chapter_output,
        chapter_number,
        start_page,
        end_page,
        marker_kind,
        generated["chapter_title"],
        subject,
        class_number,
        source_text,
        model,
        generated["chunks"],
        generated["introduction"],
        generated["learning_objectives"],
        generated["sections"],
        generated["student_questions"],
    )
