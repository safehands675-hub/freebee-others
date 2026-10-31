from __future__ import annotations

import csv
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .core import ContentEngineError
from .ocr import resolve_tesseract_command


INDEX_FILENAME = "printed_page_index.json"
PAGE_NUMBER_WHITELIST = "0123456789"
MINIMUM_PRINTED_PAGE = 1
MAXIMUM_PRINTED_PAGE = 999
MINIMUM_SEQUENCE_ANCHORS = 3
BADGE_CENTER_Y_RANGE = (0.35, 0.58)
BADGE_SIDE_RANGES = {
    "left": (0.025, 0.20),
    "right": (0.775, 0.95),
}


def _ocr_footer(
    executable: str,
    image_path: Path,
) -> list[dict[str, Any]]:
    try:
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            width, height = image.size
            crop = image.crop((0, int(height * 0.78), width, height))
            image_data = io.BytesIO()
            crop.save(image_data, format="PNG")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ContentEngineError(
            f"Could not read page image {image_path.name}: {exc}"
        ) from exc

    try:
        result = subprocess.run(
            [
                executable,
                "stdin",
                "stdout",
                "--psm",
                "11",
                "-c",
                f"tessedit_char_whitelist={PAGE_NUMBER_WHITELIST}",
                "tsv",
            ],
            input=image_data.getvalue(),
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContentEngineError(
            f"Tesseract OCR timed out on the printed-page badge in {image_path.name}."
        ) from exc
    except OSError as exc:
        raise ContentEngineError(f"Could not start Tesseract OCR: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContentEngineError(
            f"Tesseract failed on {image_path.name}: {detail or 'unknown OCR error'}"
        )

    try:
        rows = csv.DictReader(
            io.StringIO(result.stdout.decode("utf-8")),
            delimiter="\t",
        )
        candidates: list[dict[str, Any]] = []
        for row in rows:
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf", "-1"))
                left = int(row.get("left", "-1"))
                top = int(row.get("top", "-1"))
                box_width = int(row.get("width", "-1"))
                box_height = int(row.get("height", "-1"))
            except ValueError:
                continue
            if row.get("level") != "5" or not text.isdecimal():
                continue
            center_x = (left + box_width / 2) / crop.width
            center_y = (top + box_height / 2) / crop.height
            if not BADGE_CENTER_Y_RANGE[0] <= center_y <= BADGE_CENTER_Y_RANGE[1]:
                continue
            side = next(
                (
                    badge_side
                    for badge_side, (minimum_x, maximum_x) in BADGE_SIDE_RANGES.items()
                    if minimum_x <= center_x <= maximum_x
                ),
                None,
            )
            if side is None:
                continue
            printed_page = int(text)
            if not MINIMUM_PRINTED_PAGE <= printed_page <= MAXIMUM_PRINTED_PAGE:
                continue
            candidates.append({
                "printed_page": printed_page,
                "confidence": round(confidence, 2),
                "corner": side,
            })
        return candidates
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ContentEngineError(
            f"Tesseract returned invalid OCR data for {image_path.name}: {exc}"
        ) from exc


def index_printed_pages(
    manifest_path: Path,
    minimum_confidence: float = 55.0,
) -> dict[str, Any]:
    if not 0 <= minimum_confidence <= 100:
        raise ContentEngineError("Minimum OCR confidence must be between 0 and 100.")
    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(
            f"Could not read manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("assets"), list):
        raise ContentEngineError("Manifest must contain an assets list.")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("media_type") != "pdf":
        raise ContentEngineError("Printed page indexing requires an imported PDF manifest.")

    image_pages: list[tuple[int, Path]] = []
    package_root = manifest_path.parent.resolve()
    for asset in manifest["assets"]:
        if not isinstance(asset, dict) or asset.get("kind") != "pdf_page":
            continue
        pdf_page = asset.get("page")
        image_value = asset.get("image_path")
        if (
            isinstance(pdf_page, bool)
            or not isinstance(pdf_page, int)
            or pdf_page < 1
            or not isinstance(image_value, str)
        ):
            continue
        image_path = (package_root / image_value).resolve()
        try:
            image_path.relative_to(package_root)
        except ValueError as exc:
            raise ContentEngineError(
                f"Page image path escapes the imported source package: {image_value}"
            ) from exc
        if not image_path.is_file():
            raise ContentEngineError(f"Page image is missing: {image_value}")
        image_pages.append((pdf_page, image_path))

    return index_page_images(
        image_pages,
        output_path=package_root / INDEX_FILENAME,
        source_id=manifest.get("source_id"),
        source_manifest=manifest_path.name,
        source_sha256=source.get("sha256"),
        minimum_confidence=minimum_confidence,
    )


def index_page_images(
    image_pages: list[tuple[int, Path]],
    *,
    output_path: Path,
    source_id: str,
    source_manifest: str,
    source_sha256: str | None = None,
    minimum_confidence: float = 55.0,
) -> dict[str, Any]:
    if not 0 <= minimum_confidence <= 100:
        raise ContentEngineError("Minimum OCR confidence must be between 0 and 100.")
    if not isinstance(source_id, str) or not source_id:
        raise ContentEngineError("Printed-page indexing requires a source ID.")
    if not image_pages:
        raise ContentEngineError("No page images were supplied for printed-page indexing.")
    page_numbers = [page_number for page_number, _ in image_pages]
    if (
        any(
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            for page_number in page_numbers
        )
        or len(page_numbers) != len(set(page_numbers))
    ):
        raise ContentEngineError(
            "Printed-page indexing requires unique positive physical PDF page numbers."
        )

    executable = resolve_tesseract_command()
    pages: list[dict[str, Any]] = []
    for pdf_page, image_path_value in image_pages:
        image_path = Path(image_path_value).expanduser().resolve()
        if not image_path.is_file():
            raise ContentEngineError(f"Page image is missing: {image_path}")
        candidates = _ocr_footer(executable, image_path)
        strong_numbers = {
            candidate["printed_page"]
            for candidate in candidates
            if candidate["confidence"] >= minimum_confidence
        }
        pages.append({
            "pdf_page": pdf_page,
            "printed_page": None,
            "confidence": max(
                (candidate["confidence"] for candidate in candidates),
                default=None,
            ),
            "status": "needs_review",
            "candidates": candidates,
            "strong_numbers": sorted(strong_numbers),
        })

    offsets: dict[int, set[int]] = {}
    for page in pages:
        strong_numbers = page.pop("strong_numbers")
        if len(strong_numbers) == 1:
            offset = strong_numbers[0] - page["pdf_page"]
            offsets.setdefault(offset, set()).add(page["pdf_page"])
    ranked_offsets = sorted(offsets.items(), key=lambda item: len(item[1]), reverse=True)
    sequence_offset: int | None = None
    if (
        ranked_offsets
        and len(ranked_offsets[0][1]) >= MINIMUM_SEQUENCE_ANCHORS
        and (
            len(ranked_offsets) == 1
            or len(ranked_offsets[0][1]) > len(ranked_offsets[1][1])
        )
    ):
        sequence_offset = ranked_offsets[0][0]

    for page in pages:
        candidates = page.pop("candidates")
        strong_candidates = [
            candidate
            for candidate in candidates
            if candidate["confidence"] >= minimum_confidence
        ]
        strong_numbers = {candidate["printed_page"] for candidate in strong_candidates}
        if len(strong_numbers) == 1:
            best = max(strong_candidates, key=lambda candidate: candidate["confidence"])
            page.update({
                "printed_page": best["printed_page"],
                "confidence": best["confidence"],
                "status": "matched",
                "verification_method": "ocr",
            })
        elif len(strong_numbers) > 1:
            page.update({
                "confidence": max(candidate["confidence"] for candidate in strong_candidates),
                "status": "ambiguous",
                "candidates": strong_candidates,
            })
        elif sequence_offset is not None and page["pdf_page"] + sequence_offset >= 1:
            page.update({
                "printed_page": page["pdf_page"] + sequence_offset,
                "status": "matched",
                "verification_method": "page_sequence",
            })
        elif candidates:
            page.update({
                "confidence": max(candidate["confidence"] for candidate in candidates),
                "candidates": candidates,
            })
        else:
            page.pop("confidence")

    index = {
        "schema_version": 1,
        "source_id": source_id,
        "source_manifest": source_manifest,
        "source_sha256": source_sha256,
        "ocr_engine": "tesseract",
        "minimum_confidence": minimum_confidence,
        "sequence_offset": sequence_offset,
        "pages": pages,
    }
    index_path = output_path.expanduser().resolve()
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
            json.dump(index, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(index_path)
    except OSError as exc:
        raise ContentEngineError(
            f"Could not save printed-page index {index_path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "status": "ok",
        "index": str(index_path),
        "pdf_pages": len(pages),
        "matched_pages": sum(page["status"] == "matched" for page in pages),
        "needs_review_pages": sum(page["status"] != "matched" for page in pages),
    }