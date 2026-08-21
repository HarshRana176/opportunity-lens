"""
Characterizes app.matching.evaluate_hard_constraints: the FAIL >
UNKNOWN > PASS eligibility gate derived from experience, education, and
required-skills evidence.

The most important test in this file is
TestHardFailureRemainsVisible::test_hard_experience_failure_blocks_eligibility_even_when_everything_else_passes
-- this is the concrete case the whole matching layer exists to
prevent: a candidate must never be marked eligible because of strong
skill/education/seniority signal while a genuine hard requirement
(here, experience) fails.
"""
from app.matching import evaluate_hard_constraints, match_education, match_experience, match_skills
from app.schemas import (
    CandidateProfile,
    CandidateSkill,
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    SkillRequirement,
)


def _skill(raw, canonical=None, resolution="taxonomy", requirement_level="required"):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution=resolution, requirement_level=requirement_level,
    )


def _cskill(raw, canonical=None, resolution="taxonomy"):
    return CandidateSkill(raw=raw, match_key=raw.lower(), canonical=canonical, category=None, resolution=resolution)


def _candidate(**overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        skills=[],
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(**overrides):
    defaults = dict(
        title="Engineer",
        required_skills=[],
        preferred_skills=[],
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _evaluate(candidate, job):
    skill_evidence = match_skills(candidate, job)
    experience_evidence = match_experience(candidate, job)
    education_evidence = match_education(candidate, job)
    return evaluate_hard_constraints(skill_evidence, experience_evidence, education_evidence)


class TestPassCase:
    def test_all_three_constraints_pass_yields_pass_eligibility(self):
        candidate = _candidate(
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
            education=EducationBackground(
                records=[EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                          level=EducationLevel.BACHELORS, resolution="taxonomy")],
                highest_level=EducationLevel.BACHELORS,
            ),
        )
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
        )

        constraints, eligibility = _evaluate(candidate, job)

        assert eligibility == "pass"
        assert all(c.status == "pass" for c in constraints)


class TestFailCase:
    def test_a_single_failing_constraint_fails_eligibility(self):
        candidate = _candidate(total_experience_months=12)
        job = _job(experience=ExperienceRequirement(min_months=36, is_specified=True))

        constraints, eligibility = _evaluate(candidate, job)

        assert eligibility == "fail"


class TestUnknownCase:
    def test_unspecified_experience_with_no_other_constraints_is_unknown_not_pass(self):
        candidate = _candidate()
        job = _job()  # no required skills, no experience, no required education

        constraints, eligibility = _evaluate(candidate, job)

        # experience.is_specified defaults False -> UNKNOWN; required
        # skills empty -> PASS (vacuous); education no minimum -> UNKNOWN.
        # Worst status (UNKNOWN) must win over the vacuous PASS.
        assert eligibility == "unknown"

    def test_unknown_never_silently_becomes_pass(self):
        candidate = _candidate(
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
        )
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
            # candidate has no education section -> UNKNOWN
        )

        constraints, eligibility = _evaluate(candidate, job)

        education_constraint = next(c for c in constraints if c.kind == "education")
        assert education_constraint.status == "unknown"
        assert eligibility == "unknown"


class TestPrecedence:
    def test_fail_beats_unknown(self):
        candidate = _candidate(total_experience_months=12)
        job = _job(
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            # education has no minimum_level -> UNKNOWN
        )

        constraints, eligibility = _evaluate(candidate, job)

        statuses = {c.kind: c.status for c in constraints}
        assert statuses["experience"] == "fail"
        assert statuses["education"] == "unknown"
        assert eligibility == "fail"

    def test_unknown_beats_pass(self):
        candidate = _candidate(
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
        )
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            # education UNKNOWN (no minimum stated)
        )

        constraints, eligibility = _evaluate(candidate, job)

        statuses = {c.kind: c.status for c in constraints}
        assert statuses["required_skills"] == "pass"
        assert statuses["experience"] == "pass"
        assert statuses["education"] == "unknown"
        assert eligibility == "unknown"


class TestOverQualificationDoesNotBlockEligibility:
    def test_experience_partial_status_counts_as_passing_for_eligibility(self):
        candidate = _candidate(total_experience_months=200)
        job = _job(experience=ExperienceRequirement(max_months=60, is_specified=True))

        experience_evidence = match_experience(candidate, job)
        assert experience_evidence.status == "partial"  # confirm the premise

        constraints, eligibility = _evaluate(candidate, job)

        experience_constraint = next(c for c in constraints if c.kind == "experience")
        assert experience_constraint.status == "pass"
        assert eligibility != "fail"


class TestHardConstraintKinds:
    def test_exactly_three_kinds_are_always_present(self):
        candidate = _candidate()
        job = _job()

        constraints, _ = _evaluate(candidate, job)

        assert {c.kind for c in constraints} == {"experience", "education", "required_skills"}

    def test_no_required_skills_is_vacuously_passing_not_unknown(self):
        candidate = _candidate()
        job = _job(required_skills=[])

        constraints, _ = _evaluate(candidate, job)

        skills_constraint = next(c for c in constraints if c.kind == "required_skills")
        assert skills_constraint.status == "pass"

    def test_missing_required_skill_fails_the_constraint(self):
        candidate = _candidate(skills=[])
        job = _job(required_skills=[_skill("Python", canonical="python")])

        constraints, eligibility = _evaluate(candidate, job)

        skills_constraint = next(c for c in constraints if c.kind == "required_skills")
        assert skills_constraint.status == "fail"
        assert eligibility == "fail"

    def test_partial_required_skills_match_still_fails_the_constraint(self):
        # 3 required skills, 2 matched, 1 unmatched -- a partial match is
        # not "close enough": the hard constraint only passes when EVERY
        # required skill is matched, so 2-of-3 must still be FAIL.
        candidate = _candidate(
            skills=[
                _cskill("Python", canonical="python"),
                _cskill("FastAPI", canonical="fastapi"),
            ]
        )
        job = _job(
            required_skills=[
                _skill("Python", canonical="python"),
                _skill("FastAPI", canonical="fastapi"),
                _skill("Kubernetes", canonical="kubernetes"),
            ]
        )

        skill_evidence = match_skills(candidate, job)
        assert skill_evidence.matched_required == 2
        assert skill_evidence.total_required == 3

        constraints, eligibility = _evaluate(candidate, job)

        skills_constraint = next(c for c in constraints if c.kind == "required_skills")
        assert skills_constraint.status == "fail"
        assert eligibility == "fail"


class TestHardFailureRemainsVisible:
    """The central guarantee: a strong match on every OTHER dimension
    must never hide a genuine hard-requirement failure."""

    def test_hard_experience_failure_blocks_eligibility_even_when_everything_else_passes(self):
        candidate = _candidate(
            skills=[
                _cskill("Python", canonical="python"),
                _cskill("FastAPI", canonical="fastapi"),
            ],
            total_experience_months=12,  # far below the requirement
            education=EducationBackground(
                records=[EducationRecord(degree_raw="M.Tech", degree_key="mtech",
                                          level=EducationLevel.MASTERS, resolution="taxonomy")],
                highest_level=EducationLevel.MASTERS,
            ),
        )
        job = _job(
            required_skills=[
                _skill("Python", canonical="python"),
                _skill("FastAPI", canonical="fastapi"),
            ],
            experience=ExperienceRequirement(min_months=60, is_specified=True),  # 5+ years
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
        )

        skill_evidence = match_skills(candidate, job)
        education_evidence = match_education(candidate, job)

        # Confirm every OTHER dimension genuinely passes.
        assert skill_evidence.matched_required == skill_evidence.total_required
        assert education_evidence.status == "pass"

        constraints, eligibility = _evaluate(candidate, job)

        experience_constraint = next(c for c in constraints if c.kind == "experience")
        assert experience_constraint.status == "fail"
        assert eligibility == "fail"
