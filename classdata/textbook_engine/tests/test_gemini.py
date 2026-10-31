from __future__ import annotations

import base64
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from textbook_engine.core import ContentEngineError
from textbook_engine import gemini
from textbook_engine.gemini import (
    CHAPTER_CHUNK_SCHEMA,
    CHAPTER_INDEX_SCHEMA,
    DEFAULT_GEMINI_MODEL,
    complete_chapter_chunk,
    complete_chapter_chunk,
    complete_chapter_index,
    complete_visual_question,
)


class GeminiChapterCompletionTests(unittest.TestCase):
    def test_uses_interactions_api_with_configured_model_and_json_schema(self) -> None:
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(
                    return_value=SimpleNamespace(output_text='{"sections": []}')
                )
            ),
            close=Mock(),
        )
        with patch("textbook_engine.gemini.genai.Client", return_value=client) as factory:
            response = complete_chapter_chunk(
                [
                    {"role": "system", "content": "Use only source text."},
                    {"role": "user", "content": "Explain plants."},
                ],
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        self.assertEqual(response, '{"sections": []}')
        factory.assert_called_once_with(
            api_key="test-api-key",
            http_options=gemini.types.HttpOptions(timeout=60_000),
        )
        client.interactions.create.assert_called_once_with(
            model=DEFAULT_GEMINI_MODEL,
            input="Explain plants.",
            system_instruction="Use only source text.",
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
        client.close.assert_called_once_with()

    def test_builds_structured_chapter_index_from_page_evidence(self) -> None:
        evidence = "PDF page 10: CONTENTS\nPDF page 11: 1 FOOD COMPONENTS"
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(output_text='{"chapters": []}'))
            ),
            close=Mock(),
        )
        with patch("textbook_engine.gemini.genai.Client", return_value=client) as factory:
            response = complete_chapter_index(
                evidence,
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        self.assertEqual(response, '{"chapters": []}')
        factory.assert_called_once_with(
            api_key="test-api-key",
            http_options=gemini.types.HttpOptions(timeout=60_000),
        )
        request = client.interactions.create.call_args.kwargs
        self.assertEqual(request["input"], evidence)
        self.assertIn("physical PDF page", request["system_instruction"])
        self.assertEqual(
            request["response_format"],
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": CHAPTER_INDEX_SCHEMA,
            },
        )
        client.close.assert_called_once_with()

    def test_rejects_missing_prompt_parts_without_creating_client(self) -> None:
        with (
            patch("textbook_engine.gemini.genai.Client") as factory,
            self.assertRaisesRegex(ContentEngineError, "system instructions and user content"),
        ):
            complete_chapter_chunk([], DEFAULT_GEMINI_MODEL, "test-api-key")

        factory.assert_not_called()

    def test_rejects_empty_api_key_without_creating_client(self) -> None:
        with (
            patch("textbook_engine.gemini.genai.Client") as factory,
            self.assertRaisesRegex(ContentEngineError, "GEMINI_API_KEY"),
        ):
            complete_chapter_chunk(
                [
                    {"role": "system", "content": "Instructions."},
                    {"role": "user", "content": "Text."},
                ],
                DEFAULT_GEMINI_MODEL,
                " ",
            )

        factory.assert_not_called()

    def test_rejects_empty_model_response(self) -> None:
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(output_text=" "))
            ),
            close=Mock(),
        )
        with (
            patch("textbook_engine.gemini.genai.Client", return_value=client),
            self.assertRaisesRegex(ContentEngineError, "empty chapter response"),
        ):
            complete_chapter_chunk(
                [
                    {"role": "system", "content": "Instructions."},
                    {"role": "user", "content": "Text."},
                ],
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        client.close.assert_called_once_with()

    def test_chapter_index_transport_failure_is_reported_as_content_error(self) -> None:
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(side_effect=httpx.ConnectError("network offline"))
            ),
            close=Mock(),
        )
        with (
            patch("textbook_engine.gemini.genai.Client", return_value=client),
            self.assertRaisesRegex(
                ContentEngineError,
                "Gemini chapter-index request could not be completed: network offline",
            ),
        ):
            complete_chapter_index(
                "PDF page 10: CONTENTS",
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        client.close.assert_called_once_with()

    def test_chapter_transport_failure_is_reported_as_content_error(self) -> None:
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(side_effect=httpx.ConnectError("network offline"))
            ),
            close=Mock(),
        )
        with (
            patch("textbook_engine.gemini.genai.Client", return_value=client),
            self.assertRaisesRegex(
                ContentEngineError,
                "Gemini chapter request could not be completed: network offline",
            ),
        ):
            complete_chapter_chunk(
                [
                    {"role": "system", "content": "Use only source text."},
                    {"role": "user", "content": "Explain plants."},
                ],
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        client.close.assert_called_once_with()

    def test_visual_transport_failure_is_reported_as_content_error(self) -> None:
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(side_effect=httpx.ConnectError("network offline"))
            ),
            close=Mock(),
        )
        messages = [
            {"role": "system", "content": "Answer from evidence."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is shown?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,c2FtcGxl"
                        },
                    },
                ],
            },
        ]
        with (
            patch("textbook_engine.gemini.genai.Client", return_value=client),
            self.assertRaisesRegex(
                ContentEngineError,
                "Gemini visual request could not be completed: network offline",
            ),
        ):
            complete_visual_question(
                messages,
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        client.close.assert_called_once_with()

    def test_sends_inline_page_image_and_question_to_gemini(self) -> None:
        image_bytes = b"small test image"
        image_data = base64.b64encode(image_bytes).decode("ascii")
        client = SimpleNamespace(
            interactions=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(output_text="A bus and a car.")),
            ),
            close=Mock(),
        )
        messages = [
            {"role": "system", "content": "Answer as a kind school tutor."},
            {"role": "user", "content": "Explain page 23."},
            {"role": "assistant", "content": "It is a song about vehicles."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What pictures are there?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            },
        ]
        with patch("textbook_engine.gemini.genai.Client", return_value=client) as factory:
            answer = complete_visual_question(
                messages,
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        self.assertEqual(answer, "A bus and a car.")
        factory.assert_called_once_with(
            api_key="test-api-key",
            http_options=gemini.types.HttpOptions(timeout=60_000),
        )
        client.interactions.create.assert_called_once_with(
            model=DEFAULT_GEMINI_MODEL,
            input=[
                {"type": "text", "text": "User: Explain page 23."},
                {"type": "text", "text": "Assistant: It is a song about vehicles."},
                {"type": "text", "text": "What pictures are there?"},
                {
                    "type": "image",
                    "data": image_data,
                    "mime_type": "image/jpeg",
                },
            ],
            system_instruction="Answer as a kind school tutor.",
            generation_config={
                "max_output_tokens": 400,
                "temperature": 0.3,
                "thinking_level": "low",
            },
            store=False,
        )
        client.close.assert_called_once_with()

    def test_rejects_visual_request_without_image(self) -> None:
        with (
            patch("textbook_engine.gemini.genai.Client") as factory,
            self.assertRaisesRegex(ContentEngineError, "question, and a page image"),
        ):
            complete_visual_question(
                [
                    {"role": "system", "content": "Answer from evidence."},
                    {"role": "user", "content": "What is in this picture?"},
                ],
                DEFAULT_GEMINI_MODEL,
                "test-api-key",
            )

        factory.assert_not_called()

    def test_rejects_missing_visual_api_key(self) -> None:
        with (
            patch("textbook_engine.gemini.genai.Client") as factory,
            self.assertRaisesRegex(ContentEngineError, "GEMINI_API_KEY"),
        ):
            complete_visual_question(
                [
                    {"role": "system", "content": "Answer from the image."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is shown?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,c2FtcGxl"
                                },
                            },
                        ],
                    },
                ],
                DEFAULT_GEMINI_MODEL,
                "",
            )

        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
