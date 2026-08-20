"""
Characterizes app.experience.calculate_total_experience for single,
non-overlapping employment periods.

Approved domain decision: employment dates are month/year granularity,
so duration uses INCLUSIVE calendar-month semantics (May 2025 -> May
2025 == 1 month; May 2025 -> Jun 2025 == 2 months, etc).

This file previously carried a `TestCurrentDurationBehavior` class that
locked in the pre-fix exclusive-end output (undercounting every period
by one month) as a refactor safety net, plus 4 of the cases below as
`xfail(strict=True)`. Both are now retired: the fix has landed, so the
"current" (buggy) baseline no longer describes reality, and the
intended-semantics cases below pass for real instead of being expected
failures. See app/experience.py::calculate_total_experience for the
implementation.
"""
from app.extractor import EmploymentPeriod, calculate_total_experience


def _period(company, start_date, end_date):
    return EmploymentPeriod(company=company, start_date=start_date, end_date=end_date)


def _month_range_inclusive(start_year, start_month, end_year, end_month):
    """Ground-truth set of (year, month) tuples, both ends inclusive."""
    months = set()
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.add((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return months


def _expected_inclusive_months(start_year, start_month, end_year, end_month):
    return len(_month_range_inclusive(start_year, start_month, end_year, end_month))


class TestDurationSemantics:
    def test_single_month_period_is_one_month(self):
        result = calculate_total_experience(
            [_period("Acme", "May 2025", "May 2025")]
        )
        assert result["months"] == _expected_inclusive_months(2025, 5, 2025, 5)
        assert result["months"] == 1
        assert result["years"] == 0.08

    def test_two_month_period_is_two_months(self):
        result = calculate_total_experience(
            [_period("Acme", "May 2025", "Jun 2025")]
        )
        assert result["months"] == _expected_inclusive_months(2025, 5, 2025, 6)
        assert result["months"] == 2

    def test_four_month_period_is_four_months(self):
        result = calculate_total_experience(
            [_period("Acme", "May 2025", "Aug 2025")]
        )
        assert result["months"] == _expected_inclusive_months(2025, 5, 2025, 8)
        assert result["months"] == 4
        assert result["years"] == 0.33

    def test_present_period_is_inclusive_of_the_current_month(self, frozen_today):
        # frozen_today = 2030-01-15 (see conftest.FROZEN_TODAY)
        result = calculate_total_experience(
            [_period("Acme", "May 2025", "Present")]
        )
        assert result["months"] == _expected_inclusive_months(2025, 5, 2030, 1)

    def test_reversed_dates_contribute_nothing(self):
        # Not affected by inclusive/exclusive semantics -- an end date
        # before the start date must not invent a duration.
        result = calculate_total_experience(
            [_period("Acme", "Jun 2025", "Jan 2025")]
        )
        assert result == {"months": 0, "years": 0.0}

    def test_incomplete_end_date_is_not_assumed_to_be_present(self):
        # An empty end_date is NOT one of the "Present"-equivalent values
        # (see test_dates.LOCKED_PRESENT_VALUES) and does not match any
        # supported date format, so parse_resume_date raises ValueError
        # and calculate_total_experience must skip the entry rather than
        # treating it as ongoing.
        result = calculate_total_experience(
            [_period("Acme", "May 2025", "")]
        )
        assert result == {"months": 0, "years": 0.0}

    def test_unparseable_dates_do_not_invent_a_duration(self):
        result = calculate_total_experience(
            [_period("Ghost Co", "not a date", "also not a date")]
        )
        assert result == {"months": 0, "years": 0.0}

    def test_empty_history_is_zero(self):
        assert calculate_total_experience([]) == {"months": 0, "years": 0.0}
