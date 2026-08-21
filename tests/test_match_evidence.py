"""
Characterizes app.matching.build_match_evidence (full assembly) and
format_match_evidence (deterministic rendering).
"""
from app.matching import build_match_evidence, format_match_evidence
from app.schemas import (
    CandidateProfile,
    CandidateSkill,
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    Seniority,
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


class TestBuildMatchEvidenceAssembly:
    def test_all_four_dimensions_are_populated(self):
        candidate = _candidate()
        job = _job()

        evidence = build_match_evidence(candidate, job)

        assert evidence.skills is not None
        assert evidence.experience is not None
        assert evidence.education is not None
        assert evidence.seniority is not None

    def test_semantic_is_always_none(self):
        candidate = _candidate()
        job = _job()

        evidence = build_match_evidence(candidate, job)

        assert evidence.semantic is None

    def test_no_score_or_weight_field_exists_anywhere(self):
        candidate = _candidate()
        job = _job()

        evidence = build_match_evidence(candidate, job)

        dumped = evidence.model_dump()
        forbidden_substrings = ("score", "weight", "percent")
        found = [
            key
            for key in _flatten_keys(dumped)
            if any(term in key.lower() for term in forbidden_substrings)
        ]
        # semantic.similarity_score is the one declared-but-always-None
        # exception (reserved for a later task); everything else must
        # be free of scoring vocabulary.
        assert found == [] or found == ["semantic"]

    def test_hard_constraints_and_eligibility_are_present(self):
        candidate = _candidate()
        job = _job()

        evidence = build_match_evidence(candidate, job)

        assert len(evidence.hard_constraints) == 3
        assert evidence.eligibility in ("pass", "fail", "unknown", "partial")

    def test_unresolved_notes_collects_every_unknown_dimension(self):
        candidate = _candidate()  # no seniority, no education, no experience specified
        job = _job()

        evidence = build_match_evidence(candidate, job)

        assert len(evidence.unresolved_notes) >= 1
        assert all(isinstance(note, str) and note for note in evidence.unresolved_notes)

    def test_clean_full_match_has_no_unresolved_notes(self):
        candidate = _candidate(
            seniority=Seniority.SENIOR,
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
            education=EducationBackground(
                records=[EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                          level=EducationLevel.BACHELORS, resolution="taxonomy")],
                highest_level=EducationLevel.BACHELORS,
            ),
        )
        job = _job(
            seniority=Seniority.SENIOR,
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
        )

        evidence = build_match_evidence(candidate, job)

        assert evidence.unresolved_notes == []
        assert evidence.eligibility == "pass"


class TestDeterminism:
    def test_same_input_produces_identical_evidence(self):
        candidate = _candidate(
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
        )
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
        )

        first = build_match_evidence(candidate, job)
        second = build_match_evidence(candidate, job)

        assert first.model_dump() == second.model_dump()

    def test_same_input_produces_identical_formatted_output(self):
        candidate = _candidate(total_experience_months=48)
        job = _job(experience=ExperienceRequirement(min_months=36, is_specified=True))

        evidence = build_match_evidence(candidate, job)

        assert format_match_evidence(evidence) == format_match_evidence(evidence)


class TestFormatMatchEvidence:
    def test_output_is_derived_from_structure_not_a_fixed_template(self):
        candidate_a = _candidate(total_experience_months=10)
        candidate_b = _candidate(total_experience_months=100)
        job = _job(experience=ExperienceRequirement(min_months=36, is_specified=True))

        text_a = format_match_evidence(build_match_evidence(candidate_a, job))
        text_b = format_match_evidence(build_match_evidence(candidate_b, job))

        assert text_a != text_b
        assert "10 months" in text_a
        assert "100 months" in text_b

    def test_output_reports_eligibility(self):
        candidate = _candidate(total_experience_months=12)
        job = _job(experience=ExperienceRequirement(min_months=36, is_specified=True))

        text = format_match_evidence(build_match_evidence(candidate, job))

        assert "Eligibility: FAIL" in text

    def test_output_lists_required_skill_names(self):
        candidate = _candidate(skills=[_cskill("Python", canonical="python")])
        job = _job(
            required_skills=[
                _skill("Python", canonical="python"),
                _skill("Kubernetes", canonical="kubernetes"),
            ]
        )

        text = format_match_evidence(build_match_evidence(candidate, job))

        assert "Python: PASS" in text
        assert "Kubernetes: FAIL" in text

    def test_returns_a_plain_string(self):
        evidence = build_match_evidence(_candidate(), _job())

        result = format_match_evidence(evidence)

        assert isinstance(result, str)
        assert len(result) > 0


def _flatten_keys(d, prefix=""):
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            keys.extend(_flatten_keys(v, full_key))
    elif isinstance(d, list):
        for item in d:
            keys.extend(_flatten_keys(item, prefix))
    return keys
