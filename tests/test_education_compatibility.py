"""
Verifies that CandidateProfile.education (Task 6) and
JobProfile.education (Task 4) are actually comparable -- closing the
one gap tests/test_profile_compatibility.py documented and deferred.

Like that file, this does NOT implement matching. It defines the
comparison rule locally (_education_satisfied) and asserts the DATA
supports it: ordinal level comparison, multiple candidate degrees,
missing/unresolved candidate education, and "no requirement stated"
all need to be distinguishable without guessing -- this file proves
they are.
"""
from app.schemas import (
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EducationRequirement,
)


def _record(degree_raw, level=None, resolution="taxonomy", field=None):
    return EducationRecord(
        degree_raw=degree_raw,
        field_of_study_raw=field,
        institution_raw=None,
        completion_raw=None,
        degree_key=degree_raw.lower().replace(" ", ""),
        level=level,
        resolution=resolution,
    )


def _background(records, highest_level=None, raw_text=None):
    return EducationBackground(records=records, highest_level=highest_level, raw_text=raw_text)


def _education_satisfied(
    candidate_education: EducationBackground | None,
    requirement: EducationRequirement,
) -> bool:
    """
    The intended rule: an unspecified/not-required requirement
    constrains nothing; otherwise the candidate's highest resolved
    level must be at least the requirement's minimum.
    """
    if requirement.minimum_level is None or not requirement.is_required:
        return True
    if candidate_education is None or candidate_education.highest_level is None:
        return False
    return candidate_education.highest_level >= requirement.minimum_level


class TestEducationLevelOrdinalComparison:
    def test_candidate_meeting_the_minimum_is_satisfied(self):
        candidate = _background(
            [_record("B. Tech", level=EducationLevel.BACHELORS)],
            highest_level=EducationLevel.BACHELORS,
        )
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS, is_required=True
        )

        assert _education_satisfied(candidate, requirement) is True

    def test_candidate_exceeding_the_minimum_is_satisfied(self):
        candidate = _background(
            [_record("PhD", level=EducationLevel.DOCTORATE)],
            highest_level=EducationLevel.DOCTORATE,
        )
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS, is_required=True
        )

        assert _education_satisfied(candidate, requirement) is True

    def test_candidate_below_the_minimum_is_not_satisfied(self):
        candidate = _background(
            [_record("Class XII", level=EducationLevel.HIGH_SCHOOL)],
            highest_level=EducationLevel.HIGH_SCHOOL,
        )
        requirement = EducationRequirement(
            minimum_level=EducationLevel.MASTERS, is_required=True
        )

        assert _education_satisfied(candidate, requirement) is False

    def test_incompatible_levels_across_the_full_ordinal_range(self):
        levels = list(EducationLevel)
        for i, candidate_level in enumerate(levels):
            for j, required_level in enumerate(levels):
                candidate = _background(
                    [_record("Degree", level=candidate_level)], highest_level=candidate_level
                )
                requirement = EducationRequirement(minimum_level=required_level, is_required=True)

                expected = candidate_level >= required_level
                assert _education_satisfied(candidate, requirement) is expected


class TestNoRequirementStated:
    def test_missing_minimum_level_constrains_nothing(self):
        candidate = _background([], highest_level=None)
        requirement = EducationRequirement(minimum_level=None, is_required=False)

        assert _education_satisfied(candidate, requirement) is True

    def test_preferred_not_required_constrains_nothing_even_if_unmet(self):
        candidate = _background(
            [_record("Class XII", level=EducationLevel.HIGH_SCHOOL)],
            highest_level=EducationLevel.HIGH_SCHOOL,
        )
        requirement = EducationRequirement(
            minimum_level=EducationLevel.MASTERS, is_required=False
        )

        assert _education_satisfied(candidate, requirement) is True


class TestMissingCandidateEducation:
    def test_no_education_section_fails_a_real_requirement(self):
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS, is_required=True
        )

        assert _education_satisfied(None, requirement) is False

    def test_no_education_section_is_fine_when_nothing_is_required(self):
        requirement = EducationRequirement(minimum_level=None, is_required=False)

        assert _education_satisfied(None, requirement) is True

    def test_unresolved_education_is_distinct_from_missing_education(self):
        # Candidate HAS an education section, but nothing in it
        # canonicalized -- this must be representable distinctly from
        # candidate.education being None outright.
        unresolved_candidate = _background(
            [_record("Some Unrecognized Diploma", level=None, resolution="unresolved")],
            highest_level=None,
        )

        assert unresolved_candidate is not None
        assert len(unresolved_candidate.records) == 1
        assert unresolved_candidate.highest_level is None

        # Both "missing" and "unresolved" currently fail a hard
        # requirement (neither supplies a usable level), but they are
        # NOT the same state -- a matcher can tell "no information was
        # ever found" apart from "information was found but could not
        # be interpreted" and react differently (e.g. flag for review).
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS, is_required=True
        )
        assert _education_satisfied(unresolved_candidate, requirement) is False
        assert _education_satisfied(None, requirement) is False


class TestMultipleDegreesCompatibility:
    def test_only_the_qualifying_degree_among_several_needs_to_match(self):
        # A candidate with a bachelor's AND a master's satisfies a
        # bachelor's requirement via highest_level, without needing to
        # know WHICH record qualified (record-level selection is left
        # to the future matcher; see EducationBackground's docstring).
        candidate = _background(
            [
                _record("B.Tech", level=EducationLevel.BACHELORS, field="ECE"),
                _record("M.Tech", level=EducationLevel.MASTERS, field="Computer Science"),
            ],
            highest_level=EducationLevel.MASTERS,
        )
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS, is_required=True
        )

        assert _education_satisfied(candidate, requirement) is True

    def test_field_of_study_data_is_available_on_every_record_for_future_field_matching(self):
        candidate = _background(
            [
                _record("B.Tech", level=EducationLevel.BACHELORS, field="Electronics"),
                _record("M.Tech", level=EducationLevel.MASTERS, field="Computer Science"),
            ],
            highest_level=EducationLevel.MASTERS,
        )

        fields = {r.field_of_study_raw for r in candidate.records}
        assert fields == {"Electronics", "Computer Science"}

    def test_job_side_multiple_acceptable_fields_data_already_exists(self):
        # JD's fields_of_study is already a list (Task 4) -- no
        # candidate-side change was needed to make "matches one of
        # several acceptable fields" representable.
        requirement = EducationRequirement(
            minimum_level=EducationLevel.BACHELORS,
            fields_of_study=["Computer Science", "related field"],
            is_required=True,
        )

        assert requirement.fields_of_study == ["Computer Science", "related field"]


class TestSchemaSymmetry:
    def test_both_sides_expose_an_ordinal_education_level(self):
        assert "highest_level" in EducationBackground.model_fields
        assert "minimum_level" in EducationRequirement.model_fields

    def test_both_sides_expose_field_of_study_information(self):
        assert "field_of_study_raw" in EducationRecord.model_fields
        assert "fields_of_study" in EducationRequirement.model_fields
