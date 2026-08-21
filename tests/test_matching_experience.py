"""
Characterizes app.matching.match_experience: candidate total experience
(months) compared against a job's ExperienceRequirement.
"""
import pytest

from app.matching import match_experience
from app.schemas import CandidateProfile, EducationRequirement, ExperienceRequirement, JobProfile


def _candidate(months, **overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        total_experience_months=months,
        total_experience_years=round(months / 12, 2),
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(experience, **overrides):
    defaults = dict(
        title="Engineer",
        experience=experience,
        education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


class TestUnspecifiedRequirement:
    def test_unspecified_requirement_is_unknown_not_pass(self):
        candidate = _candidate(0)
        job = _job(ExperienceRequirement(is_specified=False))

        evidence = match_experience(candidate, job)

        assert evidence.status == "unknown"

    def test_unspecified_requirement_with_a_well_qualified_candidate_is_still_unknown(self):
        # A generous candidate must not turn "not stated" into a pass.
        candidate = _candidate(120)
        job = _job(ExperienceRequirement(is_specified=False))

        evidence = match_experience(candidate, job)

        assert evidence.status == "unknown"


class TestExactAndBoundaryMinimum:
    def test_exactly_at_the_minimum_passes(self):
        candidate = _candidate(36)
        job = _job(ExperienceRequirement(min_months=36, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "pass"

    def test_above_the_minimum_passes(self):
        candidate = _candidate(48)
        job = _job(ExperienceRequirement(min_months=36, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "pass"

    def test_below_the_minimum_fails_with_shortfall(self):
        candidate = _candidate(29)
        job = _job(ExperienceRequirement(min_months=36, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "fail"
        assert evidence.shortfall_months == 7

    def test_zero_experience_candidate_fails_a_real_minimum(self):
        candidate = _candidate(0)
        job = _job(ExperienceRequirement(min_months=12, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "fail"
        assert evidence.shortfall_months == 12


class TestRangeRequirement:
    def test_within_range_passes(self):
        candidate = _candidate(48)
        job = _job(ExperienceRequirement(min_months=36, max_months=60, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "pass"

    def test_below_range_fails(self):
        candidate = _candidate(24)
        job = _job(ExperienceRequirement(min_months=36, max_months=60, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "fail"
        assert evidence.shortfall_months == 12

    def test_above_range_is_partial_not_fail(self):
        candidate = _candidate(72)
        job = _job(ExperienceRequirement(min_months=36, max_months=60, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "partial"
        assert evidence.surplus_months == 12


class TestMaxOnlyRequirement:
    def test_at_the_maximum_passes(self):
        candidate = _candidate(60)
        job = _job(ExperienceRequirement(max_months=60, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "pass"

    def test_above_the_maximum_is_partial(self):
        candidate = _candidate(72)
        job = _job(ExperienceRequirement(max_months=60, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "partial"
        assert evidence.surplus_months == 12


class TestContradictoryRequirement:
    def test_min_greater_than_max_is_unknown(self):
        candidate = _candidate(48)
        job = _job(ExperienceRequirement(min_months=60, max_months=36, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.status == "unknown"

    def test_contradictory_requirement_is_unknown_regardless_of_candidate_experience(self):
        for months in (0, 12, 48, 200):
            candidate = _candidate(months)
            job = _job(ExperienceRequirement(min_months=60, max_months=36, is_specified=True))

            evidence = match_experience(candidate, job)

            assert evidence.status == "unknown"


class TestMonthsYearsUnitAgreement:
    def test_candidate_months_field_is_reported_verbatim(self):
        candidate = _candidate(29)
        job = _job(ExperienceRequirement(min_months=36, is_specified=True))

        evidence = match_experience(candidate, job)

        assert evidence.candidate_months == 29

    def test_requirement_is_preserved_on_the_evidence(self):
        requirement = ExperienceRequirement(min_months=36, raw_text="3+ years", is_specified=True)
        candidate = _candidate(48)
        job = _job(requirement)

        evidence = match_experience(candidate, job)

        assert evidence.requirement is requirement


class TestReasonIsPopulated:
    @pytest.mark.parametrize(
        "requirement, months",
        [
            (ExperienceRequirement(is_specified=False), 0),
            (ExperienceRequirement(min_months=36, is_specified=True), 12),
            (ExperienceRequirement(min_months=36, is_specified=True), 48),
            (ExperienceRequirement(max_months=36, is_specified=True), 48),
            (ExperienceRequirement(min_months=60, max_months=36, is_specified=True), 0),
        ],
    )
    def test_every_status_includes_a_non_empty_reason(self, requirement, months):
        evidence = match_experience(_candidate(months), _job(requirement))

        assert evidence.reason
        assert isinstance(evidence.reason, str)
