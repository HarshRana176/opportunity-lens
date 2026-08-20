"""
Characterizes app.extractor.parse_resume_date: supported formats,
"Present"-equivalent handling, and rejection of unsupported text.
"""
import datetime as dt

import pytest

from app.extractor import _PRESENT_VALUES, parse_resume_date


LOCKED_PRESENT_VALUES = {
    "present",
    "current",
    "now",
    "ongoing",
    "till date",
    "to date",
}


def test_present_values_set_is_unchanged():
    # Locks the exact vocabulary treated as "ongoing employment" so a
    # future edit to this set is a visible, deliberate diff.
    assert _PRESENT_VALUES == LOCKED_PRESENT_VALUES


@pytest.mark.parametrize(
    "date_text, expected_date, expected_precision",
    [
        ("15 Aug 2026", dt.date(2026, 8, 15), "day"),
        ("15 August 2026", dt.date(2026, 8, 15), "day"),
        ("Aug 2026", dt.date(2026, 8, 1), "month"),
        ("August 2026", dt.date(2026, 8, 1), "month"),
        ("2026-08", dt.date(2026, 8, 1), "month"),
        ("2026", dt.date(2026, 1, 1), "year"),
    ],
)
def test_supported_date_formats(date_text, expected_date, expected_precision):
    result = parse_resume_date(date_text)

    assert result["date"] == expected_date
    assert result["precision"] == expected_precision


@pytest.mark.parametrize(
    "present_text",
    [
        "Present",
        "present",
        "PRESENT",
        "  Present  ",
        "Current",
        "current",
        "Now",
        "Ongoing",
        "Till Date",
        "To Date",
    ],
)
def test_present_equivalent_values_resolve_to_today(frozen_today, present_text):
    result = parse_resume_date(present_text)

    assert result["date"] == frozen_today
    assert result["precision"] == "month"


@pytest.mark.parametrize(
    "invalid_text",
    [
        "",
        "   ",
        "not a date",
        "13/2026",
        "August",
        "2026-13",
        "32 Aug 2026",
    ],
)
def test_unsupported_date_text_raises_value_error(invalid_text):
    with pytest.raises(ValueError, match="Unsupported resume date format"):
        parse_resume_date(invalid_text)
