from datetime import date, datetime

from app.schemas import EmploymentPeriod


# DATE PARSING


_PRESENT_VALUES = {
    "present",
    "current",
    "now",
    "ongoing",
    "till date",
    "to date",
}


def parse_resume_date(date_text: str):

    date_text = date_text.strip()

    if date_text.lower() in _PRESENT_VALUES:

        today = date.today()

        return {
            "date": today,
            "precision": "month"
        }

    formats = [

        ("%d %b %Y", "day"),
        ("%d %B %Y", "day"),

        ("%b %Y", "month"),
        ("%B %Y", "month"),

        ("%Y-%m", "month"),

        ("%Y", "year"),
    ]

    for fmt, precision in formats:

        try:

            parsed = datetime.strptime(
                date_text,
                fmt
            )

            return {
                "date": parsed.date(),
                "precision": precision
            }

        except ValueError:
            continue

    raise ValueError(
        f"Unsupported resume date format: '{date_text}'"
    )


# EXPERIENCE CALCULATION


def date_to_month_index(parsed_date):

    return (
        parsed_date.year * 12
        + (parsed_date.month - 1)
    )


def calculate_total_experience(
    employment_history: list[EmploymentPeriod]
):
    """
    Total distinct calendar months of employment, using INCLUSIVE
    month/year granularity: a period's start month and end month both
    count (May 2025 -> May 2025 is 1 month, May 2025 -> Jun 2025 is 2).

    Overlapping and back-to-back periods (e.g. Jan-Mar and Apr-Jun, with
    no gap between them) must not double-count or under-count shared or
    adjacent calendar months, so total experience is computed as the
    size of the UNION of each period's inclusive month range, not the
    sum of each period's raw length.

    A period whose dates cannot be parsed (including a missing/empty
    end_date -- which must NOT be assumed to mean "Present") is skipped
    rather than raising, and does not contribute to the total.
    """

    intervals = []

    for employment in employment_history:

        try:

            start = parse_resume_date(
                employment.start_date
            )

            end = parse_resume_date(
                employment.end_date
            )

            start_month = date_to_month_index(
                start["date"]
            )

            end_month = date_to_month_index(
                end["date"]
            )

            if end_month < start_month:
                continue

            # Both ends are inclusive.
            intervals.append([
                start_month,
                end_month
            ])

        except ValueError:

            # Ignore an employment entry whose
            # dates cannot be parsed.
            continue

    if not intervals:

        return {
            "months": 0,
            "years": 0.0
        }

    # Sort intervals by start date.
    intervals.sort()

    # Merge overlapping AND adjacent (no-gap) employment periods so the
    # total below counts each calendar month once, as a union of ranges.
    merged = [intervals[0]]

    for start, end in intervals[1:]:

        previous_end = merged[-1][1]

        if start <= previous_end + 1:

            merged[-1][1] = max(
                previous_end,
                end
            )

        else:

            merged.append([
                start,
                end
            ])

    # Both ends of each merged range are inclusive, so a single-month
    # range (start == end) contributes 1, not 0.
    total_months = sum(
        end - start + 1
        for start, end in merged
    )

    return {
        "months": total_months,
        "years": round(
            total_months / 12,
            2
        )
    }
