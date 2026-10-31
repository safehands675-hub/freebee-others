from __future__ import annotations

import unittest

from textbook_engine.source_naming import (
    class_number_from_text,
    parse_source_filename,
)


class SourceNamingTests(unittest.TestCase):
    def test_parses_class_number_and_subject_from_short_filename(self) -> None:
        parsed = parse_source_filename("FirstClass_English.pdf")

        self.assertEqual(parsed.class_number, 1)
        self.assertEqual(parsed.subject, "English")
        self.assertIsNone(parsed.medium)
        self.assertEqual(parsed.display_title, "English")

    def test_parses_canonical_medium_subject_and_common_board_suffix(self) -> None:
        parsed = parse_source_filename(
            "Class_06_Telugu_Medium_First_Language_(Telugu)_SSC_Common.pdf"
        )

        self.assertEqual(parsed.class_number, 6)
        self.assertEqual(parsed.subject, "First Language (Telugu)")
        self.assertEqual(parsed.medium, "Telugu")
        self.assertEqual(parsed.board, "SSC")
        self.assertEqual(parsed.school, "Common")
        self.assertEqual(
            parsed.display_title,
            "First Language (Telugu) (Telugu Medium · SSC Common)",
        )

    def test_parses_school_specific_board_suffix(self) -> None:
        parsed = parse_source_filename(
            "Class_08_English_Medium_Science_CBSE_Green_Valley_School.pdf"
        )

        self.assertEqual(parsed.class_number, 8)
        self.assertEqual(parsed.subject, "Science")
        self.assertEqual(parsed.board, "CBSE")
        self.assertEqual(parsed.school, "Green Valley School")

    def test_class_number_parser_supports_digits_words_ordinals_and_roman(self) -> None:
        for value, expected in (
            ("Class 1", 1),
            ("FirstClass", 1),
            ("Class 2", 2),
            ("second class", 2),
            ("Class IV", 4),
            ("Grade VIII", 8),
            ("Class_09", 9),
            ("NinthClass", 9),
            ("Class IX", 9),
        ):
            with self.subTest(value=value):
                self.assertEqual(class_number_from_text(value), expected)

    def test_class_number_parser_does_not_infer_from_unrelated_digits(self) -> None:
        self.assertIsNone(class_number_from_text("English workbook 2026"))


if __name__ == "__main__":
    unittest.main()
