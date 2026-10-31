from __future__ import annotations

import base64
import binascii
import re
from typing import Any

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

from .core import ContentEngineError


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
CHAPTER_PROMPT_VERSION = "chapter-understanding-v1"
CHAPTER_CHUNK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapter_title": {"type": "string"},
        "introduction": {"type": "string"},
        "learning_objectives": {
            "type": "array",
            "items": {"type": "string"},
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                    "definitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "meaning": {"type": "string"},
                            },
                            "required": ["term", "meaning"],
                        },
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "important_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "formulas": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "explanation",
                    "definitions",
                    "examples",
                    "important_facts",
                    "formulas",
                    "questions",
                ],
            },
        },
    },
    "required": [
        "chapter_title",
        "introduction",
        "learning_objectives",
        "sections",
    ],
}
CHAPTER_INDEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "start_pdf_page": {"type": "integer"},
                    "marker_kind": {
                        "type": "string",
                        "enum": ["chapter", "unit", "lesson"],
                    },
                },
                "required": ["number", "title", "start_pdf_page", "marker_kind"],
            },
        },
    },
    "required": ["chapters"],
}


def _redact_api_key(message: str, api_key: str) -> str:
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = re.sub(r"(?i)\bAIza[0-9A-Za-z_-]{20,}\b", "[redacted]", message)
    return " ".join(message.split())[:300]


def _transport_error(operation: str, api_key: str, error: Exception) -> ContentEngineError:
    detail = _redact_api_key(str(error), api_key)
    detail_text = f": {detail}" if detail else ""
    return ContentEngineError(
        f"Gemini {operation} request could not be completed{detail_text}"
    )


def complete_chapter_chunk(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
) -> str:
    system_instructions = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ]
    user_inputs = [
        message["content"]
        for message in messages
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    if not system_instructions or not user_inputs:
        raise ContentEngineError(
            "Gemini chapter request requires system instructions and user content."
        )
    if not api_key.strip():
        raise ContentEngineError(
            "Gemini API key is missing. Set GEMINI_API_KEY in the project .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )
    try:
        interaction = client.interactions.create(
            model=model,
            input="\n\n".join(user_inputs),
            system_instruction="\n\n".join(system_instructions),
            generation_config={
                "max_output_tokens": 4_000,
                "temperature": 0.2,
                "thinking_level": "low",
            },
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": CHAPTER_CHUNK_SCHEMA,
            },
            store=False,
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        code_text = f" (HTTP {code})" if isinstance(code, int) else ""
        detail = _redact_api_key(str(exc), api_key)
        detail_text = f": {detail}" if detail else ""
        raise ContentEngineError(
            f"Gemini API request failed{code_text}{detail_text}"
        ) from exc
    except (httpx.HTTPError, TimeoutError, OSError) as exc:
        raise _transport_error("chapter", api_key, exc) from exc
    finally:
        client.close()

    output = interaction.output_text
    if not isinstance(output, str) or not output.strip():
        raise ContentEngineError("Gemini returned an empty chapter response.")
    return output


def complete_chapter_index(
    evidence: str,
    model: str,
    api_key: str,
) -> str:
    if not evidence.strip():
        raise ContentEngineError(
            "Gemini chapter indexing requires extracted page-heading evidence."
        )
    if not api_key.strip():
        raise ContentEngineError(
            "Gemini API key is missing. Set GEMINI_API_KEY in the project .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )
    try:
        interaction = client.interactions.create(
            model=model,
            input=evidence,
            system_instruction=(
                "You are locating top-level chapter, unit, or lesson boundaries "
                "in a textbook. Read the contents/index and the page-heading samples. "
                "Return only headings supported by the supplied evidence. Each start "
                "must be the 1-based physical PDF page where that top-level heading "
                "appears, not its printed page number. Do not treat subsections, "
                "activities, tables, examples, or question numbers as chapters. Do "
                "not guess missing numbers or page locations. Return all supported "
                "top-level headings in reading order."
            ),
            generation_config={
                "max_output_tokens": 2_500,
                "temperature": 0.1,
                "thinking_level": "low",
            },
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": CHAPTER_INDEX_SCHEMA,
            },
            store=False,
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        code_text = f" (HTTP {code})" if isinstance(code, int) else ""
        detail = _redact_api_key(str(exc), api_key)
        detail_text = f": {detail}" if detail else ""
        raise ContentEngineError(
            f"Gemini chapter-index request failed{code_text}{detail_text}"
        ) from exc
    except (httpx.HTTPError, TimeoutError, OSError) as exc:
        raise _transport_error("chapter-index", api_key, exc) from exc
    finally:
        client.close()

    output = interaction.output_text
    if not isinstance(output, str) or not output.strip():
        raise ContentEngineError("Gemini returned an empty chapter index.")
    return output


def complete_visual_question(
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
) -> str:
    system_instructions = [
        message["content"]
        for message in messages
        if message.get("role") == "system" and isinstance(message.get("content"), str)
    ]
    input_parts: list[dict[str, str]] = []
    user_text_found = False
    image_found = False
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            input_parts.append({
                "type": "text",
                "text": f"{role.title()}: {content}",
            })
            user_text_found = user_text_found or role == "user"
            continue
        if role != "user" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                input_parts.append({"type": "text", "text": part["text"]})
                user_text_found = True
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                data_url = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(data_url, str) or not data_url.startswith("data:"):
                    raise ContentEngineError(
                        "Gemini visual requests require a local inline page image."
                    )
                header, separator, encoded_data = data_url.partition(",")
                mime_type = header[5:].split(";", 1)[0]
                if (
                    not separator
                    or not mime_type.startswith("image/")
                    or ";base64" not in header
                ):
                    raise ContentEngineError(
                        "The selected page image has an unsupported data format."
                    )
                try:
                    image_bytes = base64.b64decode(encoded_data, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ContentEngineError(
                        "The selected page image could not be decoded."
                    ) from exc
                if not image_bytes:
                    raise ContentEngineError("The selected page image is empty.")
                input_parts.append({
                    "type": "image",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                    "mime_type": mime_type,
                })
                image_found = True
    if not system_instructions or not user_text_found or not image_found:
        raise ContentEngineError(
            "Gemini visual request requires system instructions, a question, and a page image."
        )
    if not api_key.strip():
        raise ContentEngineError(
            "Gemini API key is missing. Set GEMINI_API_KEY in the project .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )
    try:
        interaction = client.interactions.create(
            model=model,
            input=input_parts,
            system_instruction="\n\n".join(system_instructions),
            generation_config={
                "max_output_tokens": 400,
                "temperature": 0.3,
                "thinking_level": "low",
            },
            store=False,
        )
    except APIError as exc:
        code = getattr(exc, "code", None)
        code_text = f" (HTTP {code})" if isinstance(code, int) else ""
        detail = _redact_api_key(str(exc), api_key)
        detail_text = f": {detail}" if detail else ""
        raise ContentEngineError(
            f"Gemini visual request failed{code_text}{detail_text}"
        ) from exc
    except (httpx.HTTPError, TimeoutError, OSError) as exc:
        raise _transport_error("visual", api_key, exc) from exc
    finally:
        client.close()

    output = interaction.output_text
    if not isinstance(output, str) or not output.strip():
        raise ContentEngineError("Gemini returned an empty visual answer.")
    return output.strip()
