"""
Characterizes app.matching's skill matching: skills_match() and
match_skills(). No LLM, no I/O -- purely deterministic, offline.
"""
import pytest

from app.matching import match_skills, skills_match
from app.schemas import CandidateProfile, CandidateSkill, JobProfile, SkillRequirement


def _candidate_skill(raw, match_key, canonical=None, category=None, resolution="taxonomy"):
    return CandidateSkill(
        raw=raw, match_key=match_key, canonical=canonical, category=category, resolution=resolution
    )


def _job_skill(raw, match_key, canonical=None, category=None, resolution="taxonomy",
               requirement_level="required"):
    return SkillRequirement(
        raw=raw, match_key=match_key, canonical=canonical, category=category,
        resolution=resolution, requirement_level=requirement_level,
    )


def _candidate(skills=None, **overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        skills=skills or [],
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(required=None, preferred=None, **overrides):
    from app.schemas import EducationRequirement, ExperienceRequirement

    defaults = dict(
        title="Engineer",
        required_skills=required or [],
        preferred_skills=preferred or [],
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


class TestSkillsMatchExactAndCanonical:
    def test_identical_canonical_matches(self):
        c = _candidate_skill("Python", "python", canonical="python")
        j = _job_skill("Python", "python", canonical="python")

        matched, matched_on = skills_match(c, j)

        assert matched is True
        assert matched_on == "canonical"

    def test_alias_bridges_via_shared_canonical(self):
        # Résumé says "Postgres", JD says "PostgreSQL" -- different raw
        # and match_key, same canonical.
        c = _candidate_skill("Postgres", "postgres", canonical="postgresql")
        j = _job_skill("PostgreSQL", "postgresql", canonical="postgresql")

        matched, matched_on = skills_match(c, j)

        assert c.match_key != j.match_key
        assert matched is True
        assert matched_on == "canonical"

    def test_unrelated_canonicals_do_not_match(self):
        c = _candidate_skill("Python", "python", canonical="python")
        j = _job_skill("Docker", "docker", canonical="docker")

        matched, matched_on = skills_match(c, j)

        assert matched is False
        assert matched_on is None


class TestKafkaUnresolvedMatching:
    """The Task 5/6 regression guard: an unresolved skill on both sides
    must still match via match_key -- never require canonical."""

    def test_kafka_unresolved_both_sides_matches_via_match_key(self):
        c = _candidate_skill("Kafka", "kafka", canonical=None, resolution="unresolved")
        j = _job_skill("Kafka", "kafka", canonical=None, resolution="unresolved")

        matched, matched_on = skills_match(c, j)

        assert matched is True
        assert matched_on == "match_key"

    def test_kafka_llm_enriched_candidate_vs_unresolved_job_still_matches(self):
        # Candidate side went through enrich_unresolved_skills (category
        # filled, canonical still None); job side never got enriched.
        c = _candidate_skill("Kafka", "kafka", canonical=None, category="tools", resolution="llm")
        j = _job_skill("Kafka", "kafka", canonical=None, resolution="unresolved")

        matched, matched_on = skills_match(c, j)

        assert matched is True
        assert matched_on == "match_key"

    def test_different_unresolved_skills_do_not_match(self):
        c = _candidate_skill("Kafka", "kafka", resolution="unresolved")
        j = _job_skill("Terraform", "terraform", resolution="unresolved")

        matched, _ = skills_match(c, j)

        assert matched is False


class TestCCppCSharpDistinction:
    """Full 3x3 grid: C/C++/C# must never cross-match, in either role."""

    @pytest.mark.parametrize(
        "candidate_canonical, job_canonical, should_match",
        [
            ("c", "c", True),
            ("c++", "c++", True),
            ("c#", "c#", True),
            ("c", "c++", False),
            ("c", "c#", False),
            ("c++", "c", False),
            ("c++", "c#", False),
            ("c#", "c", False),
            ("c#", "c++", False),
        ],
    )
    def test_c_family_grid(self, candidate_canonical, job_canonical, should_match):
        c = _candidate_skill(candidate_canonical, candidate_canonical, canonical=candidate_canonical)
        j = _job_skill(job_canonical, job_canonical, canonical=job_canonical)

        matched, _ = skills_match(c, j)

        assert matched is should_match


class TestStaleTaxonomySkew:
    """
    Documents the D1-approved token-set intersection rule's exact
    coverage: it recovers the stale-skew case where the STALE record's
    raw text happened to be the CANONICAL spelling, but cannot recover
    the case where the stale record's raw text was an ALIAS spelling --
    see app.matching's module docstring for the full proof. Both cases
    are made explicit here rather than left as an unstated gap.
    """

    def test_stale_record_matches_when_its_raw_text_was_the_canonical_spelling(self):
        # Candidate persisted "PostgreSQL" before the taxonomy existed
        # -> unresolved, match_key="postgresql". Job parsed later with
        # "Postgres" -> resolved, canonical="postgresql".
        stale_candidate = _candidate_skill(
            "PostgreSQL", "postgresql", canonical=None, resolution="unresolved"
        )
        fresh_job = _job_skill("Postgres", "postgres", canonical="postgresql")

        matched, matched_on = skills_match(stale_candidate, fresh_job)

        assert matched is True
        assert matched_on == "match_key"

    def test_stale_record_does_not_match_when_its_raw_text_was_an_alias_spelling(self):
        # KNOWN, BOUNDED LIMITATION: candidate persisted "Postgres"
        # before the taxonomy existed -> unresolved, match_key=
        # "postgres". Job parsed later with "PostgreSQL" -> resolved,
        # canonical="postgresql". No pure comparison rule can recover
        # this without re-normalizing the stale record.
        stale_candidate = _candidate_skill(
            "Postgres", "postgres", canonical=None, resolution="unresolved"
        )
        fresh_job = _job_skill("PostgreSQL", "postgresql", canonical="postgresql")

        matched, matched_on = skills_match(stale_candidate, fresh_job)

        assert matched is False
        assert matched_on is None


class TestMatchSkillsRequiredVsPreferred:
    def test_required_and_preferred_are_reported_separately(self):
        candidate = _candidate(skills=[_candidate_skill("Python", "python", canonical="python")])
        job = _job(
            required=[_job_skill("Python", "python", canonical="python")],
            preferred=[_job_skill("Docker", "docker", canonical="docker")],
        )

        evidence = match_skills(candidate, job)

        assert evidence.required[0].status == "pass"
        assert evidence.preferred[0].status == "fail"
        assert evidence.matched_required == 1
        assert evidence.total_required == 1
        assert evidence.matched_preferred == 0
        assert evidence.total_preferred == 1

    def test_missing_required_skill_is_visible_as_fail(self):
        candidate = _candidate(skills=[])
        job = _job(required=[_job_skill("Python", "python", canonical="python")])

        evidence = match_skills(candidate, job)

        assert evidence.required[0].status == "fail"
        assert evidence.required[0].matched_candidate_skill is None

    def test_missing_preferred_skill_is_visible_but_distinct_from_required(self):
        candidate = _candidate(skills=[])
        job = _job(preferred=[_job_skill("Docker", "docker", canonical="docker")])

        evidence = match_skills(candidate, job)

        assert evidence.preferred[0].status == "fail"
        assert evidence.total_required == 0

    def test_unmatched_candidate_skills_are_retained(self):
        candidate = _candidate(
            skills=[
                _candidate_skill("Python", "python", canonical="python"),
                _candidate_skill("Rust", "rust", canonical="rust"),
            ]
        )
        job = _job(required=[_job_skill("Python", "python", canonical="python")])

        evidence = match_skills(candidate, job)

        assert [s.raw for s in evidence.unmatched_candidate_skills] == ["Rust"]

    def test_empty_job_skills_yields_no_matches_and_all_candidate_skills_unmatched(self):
        candidate = _candidate(skills=[_candidate_skill("Python", "python", canonical="python")])
        job = _job()

        evidence = match_skills(candidate, job)

        assert evidence.required == []
        assert evidence.preferred == []
        assert len(evidence.unmatched_candidate_skills) == 1

    def test_empty_candidate_skills_fails_every_requirement(self):
        candidate = _candidate(skills=[])
        job = _job(required=[_job_skill("Python", "python", canonical="python")])

        evidence = match_skills(candidate, job)

        assert evidence.matched_required == 0
        assert evidence.unmatched_candidate_skills == []

    def test_matched_on_reports_canonical_for_a_clean_taxonomy_match(self):
        candidate = _candidate(skills=[_candidate_skill("Python", "python", canonical="python")])
        job = _job(required=[_job_skill("Python", "python", canonical="python")])

        evidence = match_skills(candidate, job)

        assert evidence.required[0].matched_on == "canonical"

    def test_matched_on_reports_match_key_for_an_unresolved_match(self):
        candidate = _candidate(skills=[_candidate_skill("Kafka", "kafka", resolution="unresolved")])
        job = _job(required=[_job_skill("Kafka", "kafka", resolution="unresolved")])

        evidence = match_skills(candidate, job)

        assert evidence.required[0].matched_on == "match_key"
