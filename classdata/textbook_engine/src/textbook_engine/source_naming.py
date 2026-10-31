from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_CLASS_VALUE_PATTERN = (
    r"0*[1-9]|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|viii|vii|vi|iv|ix|v|iii|ii|i"
)
_CLASS_PATTERN = re.compile(
    rf"\b(?:class|grade|standard|std\.?)[\s_-]*(?:no\.?[\s_-]*)?"
    rf"(?P<class_value>{_CLASS_VALUE_PATTERN})(?!\d)"
    rf"|\b(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)"
    r"[\s_-]*class(?=$|[_\W])",
    flags=re.IGNORECASE,
)
_CLASS_NUMBERS = {
    "first": 1,
    "1st": 1,
    "i": 1,
    "second": 2,
    "2nd": 2,
    "ii": 2,
    "third": 3,
    "3rd": 3,
    "iii": 3,
    "fourth": 4,
    "4th": 4,
    "iv": 4,
    "fifth": 5,
    "5th": 5,
    "v": 5,
    "sixth": 6,
    "6th": 6,
    "vi": 6,
    "seventh": 7,
    "7th": 7,
    "vii": 7,
    "eighth": 8,
    "8th": 8,
    "viii": 8,
    "ninth": 9,
    "9th": 9,
    "ix": 9,
}
_MEDIUM_PATTERN = re.compile(
    r"^(?P<medium>english|telugu)_medium(?:_|$)",
    flags=re.IGNORECASE,
)
_BOARD_PATTERN = re.compile(
    r"_(?P<board>ssc|cbse|icse)_(?P<qualifier>common|[a-z0-9][a-z0-9_-]*)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceName:
    class_number: int | None
    subject: str
    medium: str | None
    board: str | None
    school: str | None

    @property
    def display_title(self) -> str:
        title = self.subject
        details = [
            f"{self.medium} Medium" if self.medium else None,
            f"{self.board} {self.school}" if self.board and self.school else self.board,
        ]
        details = [detail for detail in details if detail]
        return f"{title} ({' · '.join(details)})" if details else title


def class_number_from_text(value: str) -> int | None:
    match = _CLASS_PATTERN.search(value)
    if match is None:
        return None
    class_value = (match.group("class_value") or match.group("ordinal")).casefold()
    if class_value.isdecimal():
        return int(class_value)
    return _CLASS_NUMBERS.get(class_value)


def parse_source_filename(filename: str | Path) -> SourceName:
    stem = Path(filename).stem
    class_match = _CLASS_PATTERN.search(stem)
    class_number = class_number_from_text(stem)
    remainder = stem[class_match.end():] if class_match else stem
    remainder = remainder.lstrip("_- ")

    medium_match = _MEDIUM_PATTERN.match(remainder)
    medium = medium_match.group("medium").title() if medium_match else None
    if medium_match:
        remainder = remainder[medium_match.end():]

    board_match = _BOARD_PATTERN.search(remainder)
    board: str | None = None
    school: str | None = None
    if board_match:
        board = board_match.group("board").upper()
        qualifier = board_match.group("qualifier")
        remainder = remainder[:board_match.start()]
        if qualifier.casefold() != "common":
            school = qualifier.replace("_", " ").replace("-", " ").title()
        else:
            school = "Common"

    subject = re.sub(r"[_-]+", " ", remainder).strip(" .-_")
    subject = re.sub(r"\s+", " ", subject)
    subject = re.sub(r"\s*\(\s*", " (", subject)
    subject = re.sub(r"\s*\)", ")", subject)
    subject = subject.title() if subject else Path(filename).stem
    return SourceName(
        class_number=class_number,
        subject=subject,
        medium=medium,
        board=board,
        school=school,
    )
