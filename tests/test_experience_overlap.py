"""
Characterizes app.experience.calculate_total_experience for MULTIPLE
employment periods: merging of overlapping/adjacent periods, and totals,
under the approved inclusive calendar-month semantics.

Total experience is the size of the UNION of each period's inclusive
month range -- overlapping and back-to-back (no-gap) periods must each
have their shared/adjacent months counted once, not per-period.

This file previously carried a `TestCurrentOverlapBehavior` class that
locked in the pre-fix exclusive-end output as a refactor safety net,
plus 5 of the cases below as `xfail(strict=True)`. Both are now retired
now that the fix has landed -- see app/experience.py::calculate_total_experience.

Only the final `months` total is asserted (not the internal merged-
interval representation), so these tests constrain behavior without
constraining implementation.
"""
from app.extractor import EmploymentPeriod, calculate_total_experience


def _period(company, start_date, end_date):
    return EmploymentPeriod(company=company, start_date=start_date, end_date=end_date)


def _month_range_inclusive(start_year, start_month, end_year, end_month):
    months = set()
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.add((y, m))
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return months


def _expected_inclusive_months(*periods):
    """periods: (start_year, start_month, end_year, end_month) tuples."""
    all_months = set()
    for sy, sm, ey, em in periods:
        all_months |= _month_range_inclusive(sy, sm, ey, em)
    return len(all_months)


CONTAINED = [
    ("Big Co", "Jan 2025", "Dec 2025"),
    ("Side Gig", "Mar 2025", "Jun 2025"),
]
PARTIAL_OVERLAP = [
    ("Acme", "Jan 2025", "Jun 2025"),
    ("Beta", "Apr 2025", "Sep 2025"),
]
TOUCHING_BOUNDARY = [
    ("Acme", "Jan 2025", "Mar 2025"),
    ("Beta", "Mar 2025", "Jun 2025"),
]
CONSECUTIVE_DISJOINT = [
    ("Acme", "Jan 2025", "Mar 2025"),
    ("Beta", "Apr 2025", "Jun 2025"),
]
DISJOINT_WITH_GAP = [
    ("Acme", "Jan 2025", "Feb 2025"),
    ("Beta", "Jun 2025", "Aug 2025"),
]


def _periods(rows):
    return [_period(*row) for row in rows]


class TestOverlapSemantics:
    def test_contained_period_is_absorbed(self):
        result = calculate_total_experience(_periods(CONTAINED))
        assert result["months"] == _expected_inclusive_months((2025, 1, 2025, 12))
        assert result["months"] == 12

    def test_partial_overlap_merges(self):
        result = calculate_total_experience(_periods(PARTIAL_OVERLAP))
        assert result["months"] == _expected_inclusive_months((2025, 1, 2025, 9))
        assert result["months"] == 9

    def test_touching_boundary_merges(self):
        # Both periods share March 2025 as a literal start/end value.
        result = calculate_total_experience(_periods(TOUCHING_BOUNDARY))
        assert result["months"] == _expected_inclusive_months((2025, 1, 2025, 6))
        assert result["months"] == 6

    def test_consecutive_non_touching_periods_cover_all_months(self):
        # Mar 2025 end, Apr 2025 start: no shared month, but no gap
        # either -- together they cover Jan..Jun with nothing missing.
        result = calculate_total_experience(_periods(CONSECUTIVE_DISJOINT))
        assert result["months"] == _expected_inclusive_months(
            (2025, 1, 2025, 3), (2025, 4, 2025, 6)
        )
        assert result["months"] == 6

    def test_disjoint_periods_with_a_gap_sum_separately(self):
        result = calculate_total_experience(_periods(DISJOINT_WITH_GAP))
        assert result["months"] == _expected_inclusive_months(
            (2025, 1, 2025, 2), (2025, 6, 2025, 8)
        )
        assert result["months"] == 5

    def test_mixed_parseable_and_unparseable_entries_skips_the_bad_one(self):
        periods = [
            _period("Acme", "May 2025", "Aug 2025"),
            _period("Ghost Co", "not a date", "also not a date"),
        ]
        result = calculate_total_experience(periods)
        # Only the parseable entry contributes; matches the single-period
        # case in test_experience.py::test_four_month_period_is_four_months.
        assert result["months"] == 4

    def test_all_unparseable_entries_is_zero(self):
        periods = [_period("Ghost Co", "not a date", "also not a date")]
        assert calculate_total_experience(periods) == {"months": 0, "years": 0.0}

    def test_result_is_independent_of_input_order(self):
        forward = calculate_total_experience(_periods(DISJOINT_WITH_GAP))
        backward = calculate_total_experience(_periods(list(reversed(DISJOINT_WITH_GAP))))
        assert forward == backward
