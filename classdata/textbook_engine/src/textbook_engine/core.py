from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ANNOTATION_SCHEMA_VERSION = "1.0"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf", ".svg", ".txt", ".json"}
ANNOTATION_STATUSES = {"needs_review", "approved", "rejected"}
SVG_NS = "{http://www.w3.org/2000/svg}"
SAFE_SVG_ELEMENTS = {
    "svg", "title", "desc", "g", "rect", "circle", "ellipse", "line",
    "polyline", "polygon", "path", "text", "tspan", "defs", "clippath",
    "mask", "lineargradient", "radialgradient", "stop", "use",
}


class ContentEngineError(Exception):
    """An actionable error while importing or validating learning content."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(data, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _parse_pages(value: str | None, page_count: int) -> list[int]:
    if value is None:
        return list(range(1, page_count + 1))

    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ContentEngineError("Page selection contains an empty item.")
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise ContentEngineError(
                f"Invalid page selection {part!r}; use values such as 1,3-5."
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > page_count:
            raise ContentEngineError(
                f"Page range {part!r} is outside the PDF's 1-{page_count} range."
            )
        pages.update(range(start, end + 1))
    return sorted(pages)


def _safe_svg_metadata(source_path: Path) -> dict[str, Any]:
    raw = source_path.read_bytes()
    lowered = raw.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ContentEngineError("SVG files with DOCTYPE or ENTITY declarations are not accepted.")
    if b"<?xml-stylesheet" in lowered:
        raise ContentEngineError("SVG external stylesheet processing instructions are not accepted.")

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ContentEngineError(f"Invalid SVG XML: {exc}") from exc

    if root.tag not in {f"{SVG_NS}svg", "svg"}:
        raise ContentEngineError("The XML root element is not SVG.")

    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1].lower()
        if local_name not in SAFE_SVG_ELEMENTS:
            raise ContentEngineError(f"SVG element <{local_name}> is outside the supported safe subset.")
        for name, value in element.attrib.items():
            local_attribute = name.rsplit("}", 1)[-1].lower()
            if local_attribute.startswith("on"):
                raise ContentEngineError("SVG event-handler attributes are not accepted.")
            if local_attribute in {"href", "src"} and value.strip() and not value.strip().startswith("#"):
                raise ContentEngineError("SVG external and embedded resource references are not accepted.")
            if "javascript:" in value.lower() or "expression(" in value.lower():
                raise ContentEngineError("SVG script expressions are not accepted.")
            for url in re.findall(r"url\((.*?)\)", value, flags=re.IGNORECASE):
                if not url.strip(" \t\r\n'\"").startswith("#"):
                    raise ContentEngineError("SVG external CSS/resource references are not accepted.")
        if local_name == "style" and element.text:
            style = element.text.lower()
            if "@import" in style:
                raise ContentEngineError("SVG CSS imports are not accepted.")
            for url in re.findall(r"url\((.*?)\)", style, flags=re.IGNORECASE):
                if not url.strip(" \t\r\n'\"").startswith("#"):
                    raise ContentEngineError("SVG external CSS/resource references are not accepted.")

    def dimension(name: str) -> float | None:
        value = root.attrib.get(name, "").strip()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:px)?", value)
        return float(match.group(1)) if match else None

    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    title = next(
        (element.text.strip() for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "title" and element.text and element.text.strip()),
        None,
    )
    description = next(
        (element.text.strip() for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "desc" and element.text and element.text.strip()),
        None,
    )
    return {
        "width": dimension("width"),
        "height": dimension("height"),
        "view_box": view_box,
        "title": title,
        "description": description,
        "raster_preview_available": False,
    }


def _base_manifest(source_id: str, source_path: Path, checksum: str, package_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_id": source_id,
        "source": {
            "filename": source_path.name,
            "sha256": checksum,
            "imported_path": str((package_dir / f"source{source_path.suffix.lower()}").relative_to(package_dir)).replace("\\", "/"),
            "original_path": str(source_path.resolve()),
            "media_type": source_path.suffix.lower().lstrip("."),
        },
        "assets": [],
    }


def _new_annotation_record(
    asset_id: str,
    source_id: str,
    source_type: str,
    source_path: str,
    checksum: str,
    image_path: str,
    width: int,
    height: int,
    page_number: int | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "source_id": source_id,
        "source_type": source_type,
        "source_path": source_path,
        "sha256": checksum,
    }
    if page_number is not None:
        source["page"] = page_number
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "asset_id": asset_id,
        "source": source,
        "image": {"path": image_path, "width": width, "height": height},
        "annotations": [],
    }


def ingest_source(
    source: Path,
    data_dir: Path,
    pages: str | None = None,
    dpi: int = 120,
) -> Path:
    source = source.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not source.is_file():
        raise ContentEngineError(f"Source file does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ContentEngineError(
            f"Unsupported file type {source.suffix!r}. Supported types: {supported}."
        )
    if dpi < 36 or dpi > 600:
        raise ContentEngineError("DPI must be between 36 and 600.")
    if pages is not None and source.suffix.lower() != ".pdf":
        raise ContentEngineError("--pages can only be used with a PDF source.")

    checksum = sha256_file(source)
    source_id = checksum[:16]
    package_dir = data_dir / "sources" / source_id
    manifest_path = package_dir / "manifest.json"
    if manifest_path.exists():
        raise ContentEngineError(
            f"This source is already imported at {manifest_path}. "
            "Use that package or remove it explicitly before re-importing."
        )

    package_dir.mkdir(parents=True, exist_ok=False)
    imported_source = package_dir / f"source{source.suffix.lower()}"
    try:
        shutil.copyfile(source, imported_source)
        manifest = _base_manifest(source_id, source, checksum, package_dir)
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            _ingest_pdf(source, package_dir, manifest, pages, dpi, checksum)
        elif suffix in SUPPORTED_IMAGE_SUFFIXES:
            _ingest_raster(source, package_dir, manifest, source_id, checksum)
        elif suffix == ".svg":
            _ingest_svg(source, package_dir, manifest, source_id)
        elif suffix == ".txt":
            _ingest_text(source, package_dir, manifest, source_id, checksum)
        elif suffix == ".json":
            _ingest_json(source, package_dir, manifest, source_id)
        _write_json(manifest_path, manifest)
    except Exception:
        shutil.rmtree(package_dir)
        raise
    return manifest_path


def extract_pdf_images(manifest_path: Path, data_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not _within_directory(manifest_path, data_dir):
        raise ContentEngineError("Manifest must be located inside the data directory.")
    manifest = load_json(manifest_path)
    if manifest.get("source", {}).get("media_type") != "pdf":
        raise ContentEngineError("Embedded image extraction requires an imported PDF.")

    package_dir = manifest_path.parent
    imported_path = package_dir / manifest["source"]["imported_path"]
    if not _within_directory(imported_path, data_dir):
        raise ContentEngineError("Imported PDF path escapes the data directory.")
    if not imported_path.is_file():
        raise ContentEngineError(f"Imported PDF does not exist: {imported_path}")

    extracted_images: dict[str, dict[str, Any]] = {}
    total_placements = 0
    selected_pages = [
        asset.get("page")
        for asset in manifest.get("assets", [])
        if asset.get("kind") == "pdf_page"
    ]
    if not selected_pages or any(
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
        for page_number in selected_pages
    ):
        raise ContentEngineError("PDF manifest does not contain valid imported page assets.")
    selected_pages.sort()
    try:
        document = pymupdf.open(imported_path)
    except (pymupdf.FileDataError, ValueError) as exc:
        raise ContentEngineError(f"Could not open imported PDF {imported_path}: {exc}") from exc
    try:
        if document.needs_pass:
            raise ContentEngineError("Password-protected PDFs are not supported.")
        if selected_pages[-1] > document.page_count:
            raise ContentEngineError("PDF manifest references a page outside the source document.")
        for page_number in selected_pages:
            page = document.load_page(page_number - 1)
            page_images: list[dict[str, Any]] = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 1:
                    continue
                image_bytes = block.get("image")
                if not isinstance(image_bytes, bytes) or not image_bytes:
                    raise ContentEngineError(
                        f"PDF image block {block.get('number')} on page {page_number} "
                        "did not contain extractable image data."
                    )
                source_format = str(block.get("ext", "bin")).lower()
                extraction_method = "embedded_stream"
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        image_width, image_height = image.size
                        if source_format in {"jpx", "jp2"}:
                            normalized = image.convert(
                                "RGBA" if "A" in image.getbands() else "RGB"
                            )
                            output = io.BytesIO()
                            normalized.save(output, format="PNG")
                            image_bytes = output.getvalue()
                            extension = "png"
                            extraction_method = "decoded_to_png"
                        else:
                            extension = source_format
                            if not re.fullmatch(r"[a-z0-9]{1,10}", extension):
                                extension = "bin"
                except (OSError, SyntaxError, ValueError):
                    rect = pymupdf.Rect(block["bbox"])
                    if rect.width <= 0 or rect.height <= 0:
                        raise ContentEngineError(
                            f"PDF image block {block.get('number')} on page {page_number} "
                            "has invalid placement bounds."
                        )
                    scale = min(
                        max(
                            float(block["width"]) / rect.width,
                            float(block["height"]) / rect.height,
                            1.0,
                        ),
                        4.0,
                    )
                    try:
                        fallback_pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(scale, scale),
                            clip=rect,
                            alpha=False,
                        )
                        image_bytes = fallback_pixmap.tobytes("png")
                    except (RuntimeError, ValueError, OSError) as exc:
                        raise ContentEngineError(
                            f"Could not decode or render PDF image block "
                            f"{block.get('number')} on page {page_number}: {exc}"
                        ) from exc
                    source_format = source_format or "unknown"
                    extension = "png"
                    extraction_method = "page_render_fallback"
                    image_width, image_height = fallback_pixmap.width, fallback_pixmap.height
                image_digest = hashlib.sha256(image_bytes).hexdigest()
                image_id = f"pdf_image_{image_digest[:16]}"
                if image_id not in extracted_images:
                    image_rel = f"extracted/images/{image_id}.{extension}"
                    image_path = package_dir / image_rel
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(image_bytes)
                    extracted_images[image_id] = {
                        "image_id": image_id,
                        "kind": "embedded_pdf_image",
                        "path": image_rel,
                        "width": image_width,
                        "height": image_height,
                        "sha256": image_digest,
                        "format": extension,
                        "source_format": source_format,
                        "extraction_method": extraction_method,
                        "placements": [],
                    }

                placement = {
                    "page": page_number,
                    "page_rect": [float(value) for value in block["bbox"]],
                }
                extracted_images[image_id]["placements"].append(placement)
                page_images.append({
                    "image_id": image_id,
                    "page_rect": placement["page_rect"],
                })
                total_placements += 1
            page_asset = next(
                (
                    asset for asset in manifest.get("assets", [])
                    if asset.get("kind") == "pdf_page" and asset.get("page") == page_number
                ),
                None,
            )
            if page_asset is not None:
                page_asset["embedded_images"] = page_images
    finally:
        document.close()

    manifest["embedded_images"] = [
        extracted_images[image_id] for image_id in sorted(extracted_images)
    ]
    _write_json(manifest_path, manifest)
    return {
        "source_id": manifest["source_id"],
        "unique_images": len(extracted_images),
        "page_placements": total_placements,
        "manifest": str(manifest_path),
    }


def _ingest_pdf(
    source: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    page_selection: str | None,
    dpi: int,
    checksum: str,
) -> None:
    try:
        document = pymupdf.open(source)
    except (pymupdf.FileDataError, ValueError) as exc:
        raise ContentEngineError(f"Could not open PDF {source}: {exc}") from exc
    try:
        if document.needs_pass:
            raise ContentEngineError("Password-protected PDFs are not supported.")
        selected_pages = _parse_pages(page_selection, document.page_count)
        scale = dpi / 72
        manifest["source"]["pdf_page_count"] = document.page_count
        manifest["source"]["pdf_metadata"] = {
            key: value for key, value in document.metadata.items() if value
        }
        for page_number in selected_pages:
            page = document.load_page(page_number - 1)
            asset_id = f"page_{page_number:04d}"
            image_rel = f"extracted/{asset_id}.png"
            text_rel = f"extracted/{asset_id}.txt"
            annotation_rel = f"annotations/{asset_id}.json"
            image_path = package_dir / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            pixmap.save(image_path)
            extracted_text = page.get_text("text", sort=True)
            _write_text(package_dir / text_rel, extracted_text)
            with Image.open(image_path) as image:
                width, height = image.size
            record = _new_annotation_record(
                asset_id=asset_id,
                source_id=manifest["source_id"],
                source_type="pdf",
                source_path=manifest["source"]["original_path"],
                checksum=checksum,
                image_path=image_rel,
                width=width,
                height=height,
                page_number=page_number,
            )
            record["source"]["page_rect"] = [
                float(value) for value in page.rect
            ]
            _write_json(package_dir / annotation_rel, record)
            manifest["assets"].append({
                "asset_id": asset_id,
                "kind": "pdf_page",
                "page": page_number,
                "image_path": image_rel,
                "text_path": text_rel,
                "annotation_path": annotation_rel,
                "width": width,
                "height": height,
                "text_characters": len(extracted_text),
                "needs_ocr_review": not bool(extracted_text.strip()),
            })
    finally:
        document.close()


def ensure_all_pdf_pages(source: Path, manifest_path: Path, dpi: int = 120) -> int:
    source = source.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if source.suffix.lower() != ".pdf":
        raise ContentEngineError("Complete PDF page extraction requires a PDF source.")
    if dpi < 36 or dpi > 600:
        raise ContentEngineError("DPI must be between 36 and 600.")
    manifest = load_json(manifest_path)
    checksum = sha256_file(source)
    if manifest.get("source_id") != checksum[:16]:
        raise ContentEngineError(
            f"Source PDF does not match the imported package {manifest_path.parent.name}."
        )
    source_metadata = manifest.get("source")
    if not isinstance(source_metadata, dict) or source_metadata.get("media_type") != "pdf":
        raise ContentEngineError("The imported source package is not a PDF.")
    try:
        document = pymupdf.open(source)
    except (pymupdf.FileDataError, ValueError) as exc:
        raise ContentEngineError(f"Could not open PDF {source}: {exc}") from exc
    try:
        if document.needs_pass:
            raise ContentEngineError("Password-protected PDFs are not supported.")
        page_count = document.page_count
    finally:
        document.close()

    imported_pages = {
        asset.get("page")
        for asset in manifest.get("assets", [])
        if isinstance(asset, dict)
        and asset.get("kind") == "pdf_page"
        and isinstance(asset.get("page"), int)
        and not isinstance(asset.get("page"), bool)
        and 1 <= asset["page"] <= page_count
    }
    missing_pages = [
        page_number
        for page_number in range(1, page_count + 1)
        if page_number not in imported_pages
    ]
    if missing_pages:
        _ingest_pdf(
            source,
            manifest_path.parent,
            manifest,
            ",".join(str(page_number) for page_number in missing_pages),
            dpi,
            checksum,
        )
        _write_json(manifest_path, manifest)
    return len(missing_pages)


def _ingest_raster(
    source: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    source_id: str,
    checksum: str,
) -> None:
    try:
        with Image.open(source) as opened:
            opened.verify()
        with Image.open(source) as opened:
            width, height = opened.size
            image_format = opened.format
            color_mode = opened.mode
    except (OSError, Image.DecompressionBombError) as exc:
        raise ContentEngineError(f"Could not read raster image {source}: {exc}") from exc
    asset_id = f"image_{source_id[:8]}"
    image_rel = f"source{source.suffix.lower()}"
    annotation_rel = f"annotations/{asset_id}.json"
    record = _new_annotation_record(
        asset_id=asset_id,
        source_id=source_id,
        source_type="raster_image",
        source_path=manifest["source"]["original_path"],
        checksum=checksum,
        image_path=image_rel,
        width=width,
        height=height,
    )
    _write_json(package_dir / annotation_rel, record)
    manifest["assets"].append({
        "asset_id": asset_id,
        "kind": "raster_image",
        "image_path": image_rel,
        "annotation_path": annotation_rel,
        "width": width,
        "height": height,
        "image_format": image_format,
        "color_mode": color_mode,
    })


def _ingest_svg(
    source: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    source_id: str,
) -> None:
    svg_info = _safe_svg_metadata(source)
    manifest["assets"].append({
        "asset_id": f"svg_{source_id[:8]}",
        "kind": "svg",
        "path": f"source{source.suffix.lower()}",
        "annotation_supported": False,
        "svg": svg_info,
    })


def _ingest_text(
    source: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    source_id: str,
    checksum: str,
) -> None:
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ContentEngineError(f"Text source must be UTF-8 encoded: {source}") from exc
    extracted_path = package_dir / "extracted" / "content.txt"
    _write_text(extracted_path, text)
    manifest["assets"].append({
        "asset_id": f"text_{source_id[:8]}",
        "kind": "text",
        "text_path": "extracted/content.txt",
        "characters": len(text),
    })


def _ingest_json(
    source: Path,
    package_dir: Path,
    manifest: dict[str, Any],
    source_id: str,
) -> None:
    try:
        parsed = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(f"Invalid UTF-8 JSON source {source}: {exc}") from exc
    extracted_path = package_dir / "extracted" / "content.json"
    _write_json(extracted_path, parsed)
    manifest["assets"].append({
        "asset_id": f"json_{source_id[:8]}",
        "kind": "structured_json",
        "content_path": "extracted/content.json",
    })


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContentEngineError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContentEngineError(f"Expected a JSON object in {path}.")
    return data


def validate_annotation_file(annotation_path: Path, data_dir: Path) -> dict[str, Any]:
    annotation_path = annotation_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not _within_directory(annotation_path, data_dir):
        raise ContentEngineError("Annotation JSON must be located inside the data directory.")
    record = load_json(annotation_path)
    if record.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ContentEngineError(
            f"Unsupported annotation schema version: {record.get('schema_version')!r}."
        )
    asset_id = record.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ContentEngineError("Annotation JSON must have a non-empty asset_id.")
    image = record.get("image")
    if not isinstance(image, dict):
        raise ContentEngineError("Annotation JSON must contain an image object.")
    image_path_value = image.get("path")
    if not isinstance(image_path_value, str) or not image_path_value:
        raise ContentEngineError("Image path must be a non-empty relative path.")
    image_path = (annotation_path.parent.parent / image_path_value).resolve()
    if not _within_directory(image_path, data_dir):
        raise ContentEngineError("Image path escapes the data directory.")
    if not image_path.is_file():
        raise ContentEngineError(f"Referenced image does not exist: {image_path}")
    try:
        with Image.open(image_path) as opened:
            actual_width, actual_height = opened.size
    except (OSError, Image.DecompressionBombError) as exc:
        raise ContentEngineError(f"Referenced image cannot be opened: {exc}") from exc
    width, height = image.get("width"), image.get("height")
    if width != actual_width or height != actual_height:
        raise ContentEngineError(
            f"Recorded dimensions {width}x{height} do not match image "
            f"dimensions {actual_width}x{actual_height}."
        )
    annotations = record.get("annotations")
    if not isinstance(annotations, list):
        raise ContentEngineError("annotations must be a JSON array.")
    seen_ids: set[str] = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ContentEngineError(f"Annotation at index {index} must be an object.")
        object_id = annotation.get("object_id")
        if not isinstance(object_id, str) or not object_id.strip():
            raise ContentEngineError(f"Annotation at index {index} needs an object_id.")
        if object_id in seen_ids:
            raise ContentEngineError(f"Duplicate object_id: {object_id}")
        seen_ids.add(object_id)
        label = annotation.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ContentEngineError(f"Annotation {object_id} needs a label.")
        status = annotation.get("status", "needs_review")
        if not isinstance(status, str) or status not in ANNOTATION_STATUSES:
            raise ContentEngineError(f"Annotation {object_id} has invalid status {status!r}.")
        if status == "approved" and not (
            isinstance(annotation.get("verified_by"), str)
            and annotation["verified_by"].strip()
        ):
            raise ContentEngineError(
                f"Approved annotation {object_id} must include verified_by."
            )
        box = annotation.get("bounding_box")
        if not isinstance(box, dict):
            raise ContentEngineError(f"Annotation {object_id} needs a bounding_box.")
        values = [box.get(key) for key in ("x", "y", "width", "height")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ContentEngineError(f"Annotation {object_id} has non-numeric box coordinates.")
        x, y, box_width, box_height = values
        if not all(math.isfinite(value) for value in values):
            raise ContentEngineError(f"Annotation {object_id} has non-finite box coordinates.")
        if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > 1 or y + box_height > 1:
            raise ContentEngineError(f"Annotation {object_id} bounding box must fit within [0, 1].")
        confidence = annotation.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ContentEngineError(f"Annotation {object_id} confidence must be between 0 and 1.")
    return record


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        name = names.get(class_id, names.get(str(class_id), str(class_id)))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        name = names[class_id]
    else:
        name = str(class_id)
    return str(name)


def run_detection(
    annotation_path: Path,
    data_dir: Path,
    weights: str = "LibreYOLO9t.pt",
    confidence_threshold: float = 0.25,
    labels: set[str] | None = None,
    detector_factory: Any = None,
) -> dict[str, Any]:
    annotation_path = annotation_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    record = validate_annotation_file(annotation_path, data_dir)
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ContentEngineError("Confidence threshold must be between 0 and 1.")
    if not weights.strip():
        raise ContentEngineError("Model weights must be a non-empty path or model name.")

    if detector_factory is None:
        try:
            from libreyolo import LibreYOLO
        except ImportError as exc:
            raise ContentEngineError(
                "LibreYOLO is not installed. Install the optional detector with "
                "'python -m pip install -e .[detection]'."
            ) from exc
        detector_factory = LibreYOLO

    image_path = (annotation_path.parent.parent / record["image"]["path"]).resolve()
    model = detector_factory(weights)
    prediction = model.predict(str(image_path), conf=confidence_threshold)

    boxes = getattr(prediction, "boxes", None)
    if boxes is None:
        raise ContentEngineError("LibreYOLO returned no box results for the image.")
    coordinates = getattr(boxes, "xyxy", None)
    confidences = getattr(boxes, "conf", None)
    class_ids = getattr(boxes, "cls", None)
    if coordinates is None or confidences is None or class_ids is None:
        raise ContentEngineError("LibreYOLO result is missing boxes, confidence, or class IDs.")
    coordinates_list = _as_list(coordinates)
    confidences_list = _as_list(confidences)
    class_ids_list = _as_list(class_ids)
    if not (len(coordinates_list) == len(confidences_list) == len(class_ids_list)):
        raise ContentEngineError("LibreYOLO returned mismatched detection arrays.")

    names = getattr(prediction, "names", {})
    width = record["image"]["width"]
    height = record["image"]["height"]
    requested_labels = {label.casefold() for label in labels} if labels else None
    try:
        model_version = importlib.metadata.version("libreyolo")
    except importlib.metadata.PackageNotFoundError:
        model_version = "unknown"

    existing = [
        item for item in record["annotations"]
        if not (
            item.get("origin") == "model"
            and item.get("model", {}).get("name") == "LibreYOLO"
            and item.get("model", {}).get("weights") == weights
            and item.get("status", "needs_review") == "needs_review"
        )
    ]
    detections: list[dict[str, Any]] = []
    existing_ids = {item["object_id"] for item in existing}
    for index, (raw_box, raw_confidence, raw_class_id) in enumerate(
        zip(coordinates_list, confidences_list, class_ids_list, strict=True),
        start=1,
    ):
        if len(raw_box) != 4:
            raise ContentEngineError(f"LibreYOLO detection {index} has an invalid box.")
        try:
            left, top, right, bottom = (float(value) for value in raw_box)
            score = float(raw_confidence)
            class_id = int(raw_class_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContentEngineError(f"LibreYOLO detection {index} has invalid values.") from exc
        if not all(math.isfinite(value) for value in (left, top, right, bottom, score)):
            raise ContentEngineError(f"LibreYOLO detection {index} contains non-finite values.")
        label = _class_name(names, class_id)
        if requested_labels is not None and label.casefold() not in requested_labels:
            continue
        if not 0 <= score <= 1:
            raise ContentEngineError(f"LibreYOLO detection {index} has invalid confidence {score}.")
        left = min(max(left, 0.0), float(width))
        right = min(max(right, 0.0), float(width))
        top = min(max(top, 0.0), float(height))
        bottom = min(max(bottom, 0.0), float(height))
        if right <= left or bottom <= top:
            continue
        normalized_box = {
            "x": left / width,
            "y": top / height,
            "width": (right - left) / width,
            "height": (bottom - top) / height,
        }
        detection_signature = (
            f"{weights}\0{label}\0"
            f"{normalized_box['x']:.6f}\0{normalized_box['y']:.6f}\0"
            f"{normalized_box['width']:.6f}\0{normalized_box['height']:.6f}"
        )
        object_id = "det_" + hashlib.sha256(detection_signature.encode("utf-8")).hexdigest()[:16]
        if object_id in existing_ids:
            continue
        existing_ids.add(object_id)
        detections.append({
            "object_id": object_id,
            "label": label,
            "bounding_box": normalized_box,
            "confidence": score,
            "status": "needs_review",
            "verified_by": None,
            "origin": "model",
            "model": {
                "name": "LibreYOLO",
                "version": model_version,
                "weights": weights,
            },
        })

    record["annotations"] = existing + detections
    _write_json(annotation_path, record)
    return {
        "asset_id": record["asset_id"],
        "detections_added": len(detections),
        "annotations_preserved": len(existing),
        "weights": weights,
        "model_version": model_version,
    }


def render_annotation_preview(
    annotation_path: Path,
    output_path: Path,
    data_dir: Path,
) -> Path:
    annotation_path = annotation_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    if not _within_directory(output_path, data_dir):
        raise ContentEngineError("Preview output must be written inside the data directory.")
    record = validate_annotation_file(annotation_path, data_dir)
    image_path = (annotation_path.parent.parent / record["image"]["path"]).resolve()
    with Image.open(image_path) as opened:
        canvas = opened.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    image_width, image_height = canvas.size
    try:
        font = ImageFont.truetype("arial.ttf", max(12, image_height // 45))
    except OSError:
        font = ImageFont.load_default()
    for annotation in record["annotations"]:
        if annotation.get("status", "needs_review") == "rejected":
            continue
        box = annotation["bounding_box"]
        left = round(box["x"] * image_width)
        top = round(box["y"] * image_height)
        right = round((box["x"] + box["width"]) * image_width)
        bottom = round((box["y"] + box["height"]) * image_height)
        color = "#20a464" if annotation.get("status") == "approved" else "#e24b35"
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, image_width // 500))
        label = annotation["label"]
        label_y = max(0, top - font.size - 2)
        text_box = draw.textbbox((left, label_y), label, font=font)
        padding = 3
        draw.rectangle(
            (
                text_box[0] - padding,
                max(0, text_box[1] - padding),
                text_box[2] + padding,
                text_box[3] + padding,
            ),
            fill=color,
        )
        draw.text((left, label_y), label, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path
