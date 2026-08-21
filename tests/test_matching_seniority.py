"""
Characterizes app.matching.match_seniority: candidate Seniority
compared against a job's required Seniority, both the existing ordinal
enum. No title normalization is exercised or introduced here.
"""
import pytest

from app.matching import match_seniority
from app.schemas import CandidateProfile, EducationRequirement, ExperienceRequirement, JobProfile, Seniority


def _candidate(seniority, **overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        seniority=seniority,
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(seniority, **overrides):
    defaults = dict(
        title="Engineer",
        seniority=seniority,
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


class TestUnknownSeniority:
    def test_candidate_seniority_unknown_yields_unknown(self):
        candidate = _candidate(None)
        job = _job(Seniority.SENIOR)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "unknown"
        assert evidence.level_gap is None

    def test_job_seniority_unknown_yields_unknown(self):
        candidate = _candidate(Seniority.SENIOR)
        job = _job(None)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "unknown"

    def test_both_unknown_yields_unknown(self):
        candidate = _candidate(None)
        job = _job(None)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "unknown"


class TestExactAndAboveMatch:
    def test_exact_match_passes(self):
        candidate = _candidate(Seniority.SENIOR)
        job = _job(Seniority.SENIOR)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "pass"
        assert evidence.level_gap == 0

    def test_candidate_above_requirement_passes(self):
        candidate = _candidate(Seniority.PRINCIPAL)
        job = _job(Seniority.MID)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "pass"
        assert evidence.level_gap == 3


class TestBelowRequirement:
    def test_one_level_below_is_partial(self):
        candidate = _candidate(Seniority.MID)
        job = _job(Seniority.SENIOR)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "partial"
        assert evidence.level_gap == -1

    def test_two_levels_below_fails(self):
        candidate = _candidate(Seniority.JUNIOR)
        job = _job(Seniority.SENIOR)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "fail"
        assert evidence.level_gap == -2

    def test_maximally_below_fails(self):
        candidate = _candidate(Seniority.INTERN)
        job = _job(Seniority.PRINCIPAL)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "fail"
        assert evidence.level_gap == -5


class TestOnceSeniorNowJuniorScenario:
    def test_a_formerly_senior_candidate_now_at_a_lower_level_is_judged_on_current_seniority(self):
        # CandidateProfile.seniority already reflects the CURRENT role
        # (Task 5 design) -- match_seniority simply compares whatever
        # it is given; this test documents that a "downgraded" current
        # seniority still yields the correct partial/fail verdict.
        candidate = _candidate(Seniority.INTERN)
        job = _job(Seniority.SENIOR)

        evidence = match_seniority(candidate, job)

        assert evidence.status == "fail"
        assert evidence.candidate == Seniority.INTERN


class TestReasonIsPopulated:
    @pytest.mark.parametrize(
        "candidate_seniority, job_seniority",
        [
            (None, Seniority.SENIOR),
            (Seniority.SENIOR, None),
            (Seniority.SENIOR, Seniority.SENIOR),
            (Seniority.MID, Seniority.SENIOR),
            (Seniority.JUNIOR, Seniority.SENIOR),
        ],
    )
    def test_every_status_includes_a_non_empty_reason(self, candidate_seniority, job_seniority):
        evidence = match_seniority(_candidate(candidate_seniority), _job(job_seniority))

        assert evidence.reason
        assert isinstance(evidence.reason, str)
