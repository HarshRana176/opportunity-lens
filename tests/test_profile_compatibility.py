"""
Verifies that a CandidateProfile (Task 5) and a JobProfile (Task 4) are
actually comparable -- the premise the whole two-sided design rests on.

These tests do NOT implement matching (explicitly out of scope for Task
5). They assert only that the DATA on both sides supports the
comparisons a future matcher will need: shared canonical identity,
shared fallback identity for unresolved skills, a shared duration unit,
and shared ordinal enums. If any of these break, matching becomes
impossible without a schema rewrite -- which is precisely what Task 5
exists to prevent.

The tiny helpers below (_skills_match, _experience_satisfied) are local
to this file on purpose: they express the INTENDED comparison rules so
the data can be checked against them, without shipping matching logic
into app/.
"""
import pytest

from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    NormalizedSkill,
    Seniority,
    SkillRequirement,
)
from app.skills import skill_identity


def _candidate_skill(raw, match_key, canonical=None, category=None, resolution="taxonomy"):
    return CandidateSkill(
        raw=raw,
        match_key=match_key,
        canonical=canonical,
        category=category,
        resolution=resolution,
    )


def _job_skill(raw, match_key, canonical=None, category=None, resolution="taxonomy",
               requirement_level="required"):
    return SkillRequirement(
        raw=raw,
        match_key=match_key,
        canonical=canonical,
        category=category,
        resolution=resolution,
        requirement_level=requirement_level,
    )


def _skills_match(candidate_skill, job_skill) -> bool:
    """
    The intended matching rule: two skills refer to the same technology
    when their curated canonical names are equal, or -- when either side
    is unresolved -- when their match_keys are equal.
    """
    if candidate_skill.canonical is not None and job_skill.canonical is not None:
        return candidate_skill.canonical == job_skill.canonical
    return candidate_skill.match_key == job_skill.match_key


def _experience_satisfied(candidate: CandidateProfile, requirement: ExperienceRequirement) -> bool:
    """Intended rule: an unspecified requirement constrains nothing."""
    if not requirement.is_specified:
        return True
    months = candidate.total_experience_months
    if requirement.min_months is not None and months < requirement.min_months:
        return False
    if requirement.max_months is not None and months > requirement.max_months:
        return False
    return True


def _candidate(**overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        seniority=Seniority.SENIOR,
        current_role="Senior Backend Engineer",
        skills=[],
        total_experience_months=60,
        total_experience_years=5.0,
        employment_history=[],
        education=None,
        raw_text="resume text",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(**overrides):
    defaults = dict(
        title="Senior Backend Engineer",
        seniority=Seniority.SENIOR,
        required_skills=[],
        preferred_skills=[],
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        responsibilities=[],
        raw_text="job text",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


class TestSharedSkillRepresentation:
    def test_both_sides_inherit_the_same_normalized_skill_base(self):
        assert issubclass(CandidateSkill, NormalizedSkill)
        assert issubclass(SkillRequirement, NormalizedSkill)

    def test_both_sides_expose_the_same_identity_fields(self):
        shared = {"raw", "match_key", "canonical", "category", "resolution"}
        assert shared <= set(CandidateSkill.model_fields)
        assert shared <= set(SkillRequirement.model_fields)

    def test_the_shared_identity_rule_applies_to_both_sides(self):
        candidate = _candidate_skill("Python", "python", canonical="python")
        job = _job_skill("Python", "python", canonical="python")

        assert skill_identity(candidate) == skill_identity(job)

    def test_candidate_skill_has_no_requirement_level(self):
        # required/preferred is a property of a job's demand, not of a
        # candidate's ability.
        assert "requirement_level" not in CandidateSkill.model_fields
        assert "requirement_level" in SkillRequirement.model_fields


class TestCanonicalMatching:
    def test_identical_canonical_names_match(self):
        candidate = _candidate_skill("Python", "python", canonical="python")
        job = _job_skill("Python", "python", canonical="python")

        assert _skills_match(candidate, job) is True

    def test_different_spellings_match_via_canonical(self):
        # The alias bridge: résumé says "Postgres", JD says "PostgreSQL".
        # Different raw, different match_key, SAME canonical.
        candidate = _candidate_skill("Postgres", "postgres", canonical="postgresql")
        job = _job_skill("PostgreSQL", "postgresql", canonical="postgresql")

        assert candidate.match_key != job.match_key
        assert _skills_match(candidate, job) is True

    def test_distinct_technologies_do_not_match(self):
        candidate = _candidate_skill("Python", "python", canonical="python")
        job = _job_skill("Docker", "docker", canonical="docker")

        assert _skills_match(candidate, job) is False

    @pytest.mark.parametrize(
        "candidate_canonical, job_canonical, should_match",
        [
            ("c", "c", True),
            ("c++", "c++", True),
            ("c#", "c#", True),
            ("c", "c++", False),
            ("c", "c#", False),
            ("c++", "c#", False),
        ],
    )
    def test_c_cpp_csharp_never_cross_match(
        self, candidate_canonical, job_canonical, should_match
    ):
        candidate = _candidate_skill(
            candidate_canonical, candidate_canonical, canonical=candidate_canonical
        )
        job = _job_skill(job_canonical, job_canonical, canonical=job_canonical)

        assert _skills_match(candidate, job) is should_match


class TestUnresolvedSkillMatching:
    def test_unresolved_on_both_sides_matches_via_match_key(self):
        # Neither side has a taxonomy entry -- but a JD requiring Kafka
        # still matches a candidate who has Kafka.
        candidate = _candidate_skill("Kafka", "kafka", resolution="unresolved")
        job = _job_skill("Kafka", "kafka", resolution="unresolved")

        assert _skills_match(candidate, job) is True

    def test_unresolved_candidate_matches_enriched_job_skill(self):
        # Asymmetric resolution (one side got LLM-enriched, the other
        # did not) must still match -- canonical is None on both, so the
        # match_key fallback carries it.
        candidate = _candidate_skill("Kafka", "kafka", resolution="unresolved")
        job = _job_skill("Kafka", "kafka", category="tools", resolution="llm")

        assert _skills_match(candidate, job) is True

    def test_casing_differences_do_not_prevent_an_unresolved_match(self):
        # match_key is lowercased, so résumé "KAFKA" vs JD "Kafka" match.
        candidate = _candidate_skill("KAFKA", "kafka", resolution="unresolved")
        job = _job_skill("Kafka", "kafka", resolution="unresolved")

        assert _skills_match(candidate, job) is True

    def test_different_unresolved_skills_do_not_match(self):
        candidate = _candidate_skill("Kafka", "kafka", resolution="unresolved")
        job = _job_skill("Terraform", "terraform", resolution="unresolved")

        assert _skills_match(candidate, job) is False


class TestExperienceCompatibility:
    def test_both_sides_use_months_as_the_shared_unit(self):
        assert "total_experience_months" in CandidateProfile.model_fields
        assert "min_months" in ExperienceRequirement.model_fields
        assert "max_months" in ExperienceRequirement.model_fields

    def test_candidate_meeting_a_minimum_is_satisfied(self):
        candidate = _candidate(total_experience_months=60)
        requirement = ExperienceRequirement(min_months=36, is_specified=True)

        assert _experience_satisfied(candidate, requirement) is True

    def test_candidate_below_a_minimum_is_not_satisfied(self):
        candidate = _candidate(total_experience_months=12)
        requirement = ExperienceRequirement(min_months=36, is_specified=True)

        assert _experience_satisfied(candidate, requirement) is False

    def test_candidate_within_a_range_is_satisfied(self):
        candidate = _candidate(total_experience_months=48)
        requirement = ExperienceRequirement(
            min_months=36, max_months=60, is_specified=True
        )

        assert _experience_satisfied(candidate, requirement) is True

    def test_unspecified_requirement_constrains_nothing(self):
        # The is_specified flag exists precisely so "no requirement
        # stated" is never mistaken for "zero months required".
        candidate = _candidate(total_experience_months=0)
        requirement = ExperienceRequirement(is_specified=False)

        assert _experience_satisfied(candidate, requirement) is True


class TestSeniorityCompatibility:
    def test_both_sides_use_the_same_ordinal_enum(self):
        candidate = _candidate(seniority=Seniority.SENIOR)
        job = _job(seniority=Seniority.MID)

        assert candidate.seniority >= job.seniority

    def test_under_qualified_seniority_compares_correctly(self):
        candidate = _candidate(seniority=Seniority.JUNIOR)
        job = _job(seniority=Seniority.SENIOR)

        assert candidate.seniority < job.seniority

    def test_missing_seniority_on_either_side_is_representable(self):
        candidate = _candidate(seniority=None)
        job = _job(seniority=None)

        assert candidate.seniority is None
        assert job.seniority is None


class TestEndToEndProfilePairing:
    def test_a_realistic_candidate_and_job_compare_across_every_dimension(self):
        candidate = _candidate(
            seniority=Seniority.SENIOR,
            total_experience_months=60,
            skills=[
                _candidate_skill("Python", "python", canonical="python",
                                 category="programming_languages"),
                _candidate_skill("Postgres", "postgres", canonical="postgresql",
                                 category="tools"),
                _candidate_skill("Kafka", "kafka", category="tools", resolution="llm"),
            ],
            employment_history=[
                CandidateEmployment(
                    company="Acme",
                    role="Senior Backend Engineer",
                    start_date="Jan 2021",
                    end_date="Dec 2025",
                    duration_months=60,
                    seniority=Seniority.SENIOR,
                )
            ],
        )
        job = _job(
            seniority=Seniority.MID,
            required_skills=[
                _job_skill("Python", "python", canonical="python",
                           category="programming_languages"),
                # Alias spelling differs from the candidate's.
                _job_skill("PostgreSQL", "postgresql", canonical="postgresql",
                           category="tools"),
            ],
            preferred_skills=[
                _job_skill("Kafka", "kafka", resolution="unresolved",
                           requirement_level="preferred"),
                _job_skill("Kubernetes", "kubernetes", canonical="kubernetes",
                           category="tools", requirement_level="preferred"),
            ],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
        )

        def matched(job_skill):
            return any(_skills_match(cs, job_skill) for cs in candidate.skills)

        # Every required skill matches -- including across an alias.
        assert all(matched(js) for js in job.required_skills)

        # Preferred: Kafka matches via match_key despite being
        # unresolved on the JD side; Kubernetes is genuinely absent.
        matched_preferred = [js.raw for js in job.preferred_skills if matched(js)]
        missing_preferred = [js.raw for js in job.preferred_skills if not matched(js)]
        assert matched_preferred == ["Kafka"]
        assert missing_preferred == ["Kubernetes"]

        assert _experience_satisfied(candidate, job.experience) is True
        assert candidate.seniority >= job.seniority

    def test_education_is_the_one_known_gap(self):
        # Task 5 (D5) deliberately leaves candidate education
        # unpopulated; the JD side already has a structured requirement.
        # This asymmetry is expected and documented, not a defect --
        # résumé education extraction is Task 6.
        candidate = _candidate()
        job = _job(education=EducationRequirement(minimum_level=None))

        assert candidate.education is None
        assert "education" in JobProfile.model_fields
        assert "education" in CandidateProfile.model_fields
