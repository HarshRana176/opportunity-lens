"""
Characterizes app.matching.match_education: candidate EducationBackground
compared against a job's EducationRequirement.
"""
from app.matching import match_education
from app.schemas import (
    CandidateProfile,
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
)


def _record(degree_raw, level=None, resolution="taxonomy", field=None):
    return EducationRecord(
        degree_raw=degree_raw,
        field_of_study_raw=field,
        institution_raw=None,
        completion_raw=None,
        degree_key=degree_raw.lower().replace(" ", "").replace(".", ""),
        level=level,
        resolution=resolution,
    )


def _background(records, highest_level=None):
    return EducationBackground(records=records, highest_level=highest_level)


def _candidate(education=None, **overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        education=education,
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(requirement, **overrides):
    defaults = dict(
        title="Engineer",
        experience=ExperienceRequirement(),
        education=requirement,
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


class TestNoRequirementStated:
    def test_no_minimum_level_is_unknown(self):
        candidate = _candidate(education=None)
        job = _job(EducationRequirement(minimum_level=None, is_required=False))

        evidence = match_education(candidate, job)

        assert evidence.status == "unknown"


class TestPreferredNeverBlocks:
    def test_preferred_education_passes_even_when_candidate_has_none(self):
        candidate = _candidate(education=None)
        job = _job(EducationRequirement(minimum_level=EducationLevel.MASTERS, is_required=False))

        evidence = match_education(candidate, job)

        assert evidence.status == "pass"

    def test_preferred_education_passes_even_when_candidate_is_below_it(self):
        candidate = _candidate(
            education=_background(
                [_record("Class XII", level=EducationLevel.HIGH_SCHOOL)],
                highest_level=EducationLevel.HIGH_SCHOOL,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.MASTERS, is_required=False))

        evidence = match_education(candidate, job)

        assert evidence.status == "pass"


class TestMissingCandidateEducation:
    def test_no_education_section_is_unknown_not_fail(self):
        candidate = _candidate(education=None)
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.status == "unknown"


class TestUnresolvedCandidateEducation:
    def test_all_unresolved_records_is_unknown_not_fail(self):
        candidate = _candidate(
            education=_background(
                [_record("Novel Program", level=None, resolution="unresolved")],
                highest_level=None,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.status == "unknown"

    def test_missing_and_unresolved_are_both_unknown_but_distinguishable_inputs(self):
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        missing = match_education(_candidate(education=None), job)
        unresolved = match_education(
            _candidate(
                education=_background(
                    [_record("Novel Program", level=None, resolution="unresolved")]
                )
            ),
            job,
        )

        assert missing.status == "unknown"
        assert unresolved.status == "unknown"
        # Different reasons -- the two cases remain distinguishable even
        # though both currently resolve to the same status.
        assert missing.reason != unresolved.reason


class TestLevelComparison:
    def test_exact_level_match_passes(self):
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS)],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.status == "pass"

    def test_higher_level_than_required_passes(self):
        candidate = _candidate(
            education=_background(
                [_record("PhD", level=EducationLevel.DOCTORATE)],
                highest_level=EducationLevel.DOCTORATE,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.status == "pass"

    def test_lower_level_than_required_fails(self):
        candidate = _candidate(
            education=_background(
                [_record("Class XII", level=EducationLevel.HIGH_SCHOOL)],
                highest_level=EducationLevel.HIGH_SCHOOL,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.MASTERS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.status == "fail"


class TestMultipleDegreesMatchingRecords:
    def test_only_the_qualifying_degree_is_named_in_matching_records(self):
        candidate = _candidate(
            education=_background(
                [
                    _record("Class XII", level=EducationLevel.HIGH_SCHOOL),
                    _record("B.Tech", level=EducationLevel.BACHELORS),
                ],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert [r.degree_raw for r in evidence.matching_records] == ["B.Tech"]

    def test_multiple_qualifying_degrees_are_all_named(self):
        candidate = _candidate(
            education=_background(
                [
                    _record("B.Tech", level=EducationLevel.BACHELORS),
                    _record("M.Tech", level=EducationLevel.MASTERS),
                ],
                highest_level=EducationLevel.MASTERS,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert {r.degree_raw for r in evidence.matching_records} == {"B.Tech", "M.Tech"}

    def test_no_record_qualifies_when_all_are_below_the_requirement(self):
        candidate = _candidate(
            education=_background(
                [_record("Class XII", level=EducationLevel.HIGH_SCHOOL)],
                highest_level=EducationLevel.HIGH_SCHOOL,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.MASTERS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.matching_records == []


class TestFieldOfStudyIsEvidenceOnly:
    def test_field_overlap_is_populated_when_fields_match(self):
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS, field="Computer Science")],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(
            EducationRequirement(
                minimum_level=EducationLevel.BACHELORS,
                fields_of_study=["Computer Science"],
                is_required=True,
            )
        )

        evidence = match_education(candidate, job)

        assert evidence.field_overlap == ["Computer Science"]

    def test_field_overlap_is_case_insensitive(self):
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS, field="computer science")],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(
            EducationRequirement(
                minimum_level=EducationLevel.BACHELORS,
                fields_of_study=["Computer Science"],
                is_required=True,
            )
        )

        evidence = match_education(candidate, job)

        assert evidence.field_overlap == ["Computer Science"]

    def test_related_field_phrase_is_never_treated_as_a_deterministic_match(self):
        # "related field" is not a field name -- it must not spuriously
        # overlap with anything the candidate has.
        candidate = _candidate(
            education=_background(
                [
                    _record(
                        "B.Tech",
                        level=EducationLevel.BACHELORS,
                        field="Electronics & Communication Engineering",
                    )
                ],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(
            EducationRequirement(
                minimum_level=EducationLevel.BACHELORS,
                fields_of_study=["Computer Science", "related field"],
                is_required=True,
            )
        )

        evidence = match_education(candidate, job)

        assert evidence.field_overlap == []

    def test_no_field_overlap_does_not_affect_status(self):
        # A genuinely mismatched field must not turn a level PASS into
        # a FAIL -- field is evidence-only.
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS, field="Mechanical Engineering")],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(
            EducationRequirement(
                minimum_level=EducationLevel.BACHELORS,
                fields_of_study=["Computer Science"],
                is_required=True,
            )
        )

        evidence = match_education(candidate, job)

        assert evidence.status == "pass"
        assert evidence.field_overlap == []

    def test_field_match_assessable_is_always_false(self):
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS, field="Computer Science")],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(
            EducationRequirement(
                minimum_level=EducationLevel.BACHELORS,
                fields_of_study=["Computer Science"],
                is_required=True,
            )
        )

        evidence = match_education(candidate, job)

        assert evidence.field_match_assessable is False

    def test_no_fields_stated_on_the_job_yields_empty_overlap(self):
        candidate = _candidate(
            education=_background(
                [_record("B.Tech", level=EducationLevel.BACHELORS, field="Computer Science")],
                highest_level=EducationLevel.BACHELORS,
            )
        )
        job = _job(EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True))

        evidence = match_education(candidate, job)

        assert evidence.field_overlap == []
