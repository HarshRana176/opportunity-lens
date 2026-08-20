"""
Characterizes app.experience.calculate_period_interval -- the Task 5
per-position helper.

Deliberately asserts consistency WITH calculate_total_experience rather
than restating its math independently: the point of adding this helper
was that per-period durations and the aggregate total must never
disagree about what a month means. calculate_total_experience itself is
unchanged by Task 5 and remains covered by tests/test_experience.py and
tests/test_experience_overlap.py.
"""
import pytest

from app.experience import calculate_period_interval, calculate_total_experience
from app.schemas import EmploymentPeriod


def _period(start_date, end_date, company="Acme", role=None):
    return EmploymentPeriod(
        company=company, role=role, start_date=start_date, end_date=end_date
    )


class TestInclusiveDuration:
    @pytest.mark.parametrize(
        "start, end, expected_months",
        [
            ("May 2025", "May 2025", 1),
            ("May 2025", "Jun 2025", 2),
            ("May 2025", "Aug 2025", 4),
            ("Jan 2025", "Dec 2025", 12),
            ("Dec 2024", "Jan 2025", 2),
        ],
    )
    def test_duration_is_inclusive_of_both_end_months(self, start, end, expected_months):
        result = calculate_period_interval(_period(start, end))
        assert result["duration_months"] == expected_months

    def test_single_period_duration_matches_the_aggregate_calculation(self):
        period = _period("May 2025", "Aug 2025")

        interval = calculate_period_interval(period)
        aggregate = calculate_total_experience([period])

        assert interval["duration_months"] == aggregate["months"]

    def test_non_overlapping_periods_sum_to_the_aggregate(self):
        periods = [_period("Jan 2025", "Mar 2025"), _period("Jun 2025", "Aug 2025")]

        per_period = [calculate_period_interval(p)["duration_months"] for p in periods]
        aggregate = calculate_total_experience(periods)

        assert sum(per_period) == aggregate["months"]


class TestMonthIndices:
    def test_indices_are_populated_for_parseable_dates(self):
        result = calculate_period_interval(_period("May 2025", "Aug 2025"))

        assert result["start_month_index"] is not None
        assert result["end_month_index"] is not None
        assert result["end_month_index"] > result["start_month_index"]

    def test_indices_are_monotonic_across_a_year_boundary(self):
        earlier = calculate_period_interval(_period("Dec 2024", "Dec 2024"))
        later = calculate_period_interval(_period("Jan 2025", "Jan 2025"))

        assert later["start_month_index"] == earlier["start_month_index"] + 1


class TestUninterpretableDates:
    @pytest.mark.parametrize(
        "start, end",
        [
            ("not a date", "also not a date"),
            ("May 2025", ""),
            ("", "Aug 2025"),
            ("May 2025", "a few months"),
        ],
    )
    def test_unparseable_dates_yield_none_without_raising(self, start, end):
        result = calculate_period_interval(_period(start, end))

        assert result["duration_months"] is None
        assert result["start_month_index"] is None
        assert result["end_month_index"] is None

    def test_empty_end_date_is_not_treated_as_present(self):
        # Same rule the aggregate calculation enforces: a missing end
        # date must never be assumed to mean ongoing employment.
        result = calculate_period_interval(_period("May 2025", ""))

        assert result["is_current"] is False
        assert result["duration_months"] is None

    def test_reversed_dates_yield_no_duration(self):
        result = calculate_period_interval(_period("Jun 2025", "Jan 2025"))

        assert result["duration_months"] is None

    def test_reversed_dates_still_expose_their_parsed_indices(self):
        # The dates parsed fine; it is their ORDER that is
        # uninterpretable. Keeping the indices preserves information the
        # caller may want to report, while duration stays None rather
        # than negative.
        result = calculate_period_interval(_period("Jun 2025", "Jan 2025"))

        assert result["start_month_index"] is not None
        assert result["end_month_index"] is not None


class TestIsCurrent:
    @pytest.mark.parametrize(
        "end_text", ["Present", "present", "  Present  ", "Current", "Ongoing", "To Date"]
    )
    def test_present_equivalents_set_is_current(self, frozen_today, end_text):
        result = calculate_period_interval(_period("Jan 2025", end_text))
        assert result["is_current"] is True

    def test_a_concrete_end_date_is_not_current(self):
        result = calculate_period_interval(_period("Jan 2025", "Mar 2025"))
        assert result["is_current"] is False

    def test_present_period_duration_is_inclusive_of_the_current_month(self, frozen_today):
        # frozen_today = 2030-01-15 (see conftest.FROZEN_TODAY)
        result = calculate_period_interval(_period("Nov 2029", "Present"))

        # Nov 2029, Dec 2029, Jan 2030 -> 3 inclusive months
        assert result["duration_months"] == 3
