"""
Characterizes app.requirements: deterministic interpretation of
experience, education, and seniority phrases extracted (verbatim) from
a job description. No LLM involved anywhere in this file.
"""
import pytest

from app.requirements import (
    derive_seniority,
    parse_education_requirement,
    parse_experience_requirement,
)
from app.schemas import EducationLevel, Seniority


class TestParseExperienceRequirement:
    @pytest.mark.parametrize(
        "text, expected_min, expected_max",
        [
            ("3+ years", 36, None),
            ("10+ years of experience", 120, None),
            ("3 - 5 years", 36, 60),
            ("2 to 4 years", 24, 48),
            ("2-3 years of relevant experience", 24, 36),
            ("minimum 2 years", 24, None),
            ("minimum of 2 years", 24, None),
            ("at least 4 years", 48, None),
            ("min. 3 years", 36, None),
            ("up to 5 years", None, 60),
            ("maximum 5 years", None, 60),
            ("over 6 years", 72, None),
            ("more than 6 years", 72, None),
            ("18 months", 18, 18),
            ("5years", 60, 60),
        ],
    )
    def test_recognized_patterns(self, text, expected_min, expected_max):
        result = parse_experience_requirement(text)
        assert result.min_months == expected_min
        assert result.max_months == expected_max
        assert result.is_specified is True
        assert result.raw_text == text

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "   ",
            "3+",  # no unit -- ambiguous, not guessed
            "a few years",
            "strong communication skills",
            "several years of experience",
        ],
    )
    def test_unspecified_or_unparseable_yields_unspecified(self, text):
        result = parse_experience_requirement(text)
        assert result.min_months is None
        assert result.max_months is None
        assert result.is_specified is False

    def test_missing_text_has_no_raw_text(self):
        result = parse_experience_requirement(None)
        assert result.raw_text is None

    def test_unparseable_but_present_text_preserves_raw_text(self):
        # Distinguishes "nothing was extracted" from "something was
        # extracted but could not be interpreted" -- the caller (which
        # assembles parse_warnings) needs raw_text to report the latter.
        result = parse_experience_requirement("a few years")
        assert result.raw_text == "a few years"
        assert result.is_specified is False

    def test_is_case_insensitive(self):
        result = parse_experience_requirement("MINIMUM 3 YEARS")
        assert result.min_months == 36
        assert result.is_specified is True

    def test_range_pattern_takes_priority_over_bare_pattern(self):
        # Regression guard against a broad bare-number pattern matching
        # only the first number in a range and discarding the second.
        result = parse_experience_requirement("3-5 years")
        assert result.min_months == 36
        assert result.max_months == 60


class TestParseEducationRequirement:
    def test_missing_text_yields_no_requirement(self):
        result = parse_education_requirement(None)
        assert result.minimum_level is None
        assert result.is_required is False
        assert result.fields_of_study == []
        assert result.raw_text is None

    def test_empty_text_yields_no_requirement(self):
        result = parse_education_requirement("")
        assert result.minimum_level is None
        assert result.is_required is False

    @pytest.mark.parametrize(
        "text, expected_level",
        [
            ("PhD required", EducationLevel.DOCTORATE),
            ("Doctorate in a related field", EducationLevel.DOCTORATE),
            ("Master's degree required", EducationLevel.MASTERS),
            ("MBA preferred", EducationLevel.MASTERS),
            ("Bachelor's degree in Computer Science", EducationLevel.BACHELORS),
            ("BSc in Engineering", EducationLevel.BACHELORS),
            ("Associate degree", EducationLevel.ASSOCIATE),
            ("High school diploma required", EducationLevel.HIGH_SCHOOL),
        ],
    )
    def test_recognized_level_terms(self, text, expected_level):
        result = parse_education_requirement(text)
        assert result.minimum_level == expected_level

    def test_no_recognized_level_term_yields_no_level_but_keeps_raw_text(self):
        result = parse_education_requirement("some post-secondary training")
        assert result.minimum_level is None
        assert result.raw_text == "some post-secondary training"
        assert result.is_required is False

    def test_multiple_levels_take_the_lower_as_the_practical_minimum(self):
        result = parse_education_requirement("Bachelor's or Master's degree")
        assert result.minimum_level == EducationLevel.BACHELORS

    def test_required_by_default_when_a_level_is_found(self):
        result = parse_education_requirement("Bachelor's degree required")
        assert result.is_required is True

    @pytest.mark.parametrize(
        "text",
        [
            "Master's degree preferred",
            "Bachelor's degree, a plus",
            "PhD is a bonus",
            "Bachelor's degree or equivalent experience",
        ],
    )
    def test_optional_markers_set_is_required_false(self, text):
        result = parse_education_requirement(text)
        assert result.is_required is False

    def test_or_equivalent_credential_idiom_stays_required(self):
        # "High school diploma or equivalent" means "diploma or GED" --
        # a hard requirement satisfiable by an equivalent credential,
        # NOT an optional/preferred requirement. Must not be confused
        # with "Bachelor's degree or equivalent EXPERIENCE" (optional).
        result = parse_education_requirement("High school diploma or equivalent")
        assert result.is_required is True
        assert result.minimum_level == EducationLevel.HIGH_SCHOOL

    def test_fields_of_study_passed_through_verbatim(self):
        result = parse_education_requirement(
            "Bachelor's in Computer Science",
            fields_of_study=["Computer Science", "related field"],
        )
        assert result.fields_of_study == ["Computer Science", "related field"]

    def test_fields_of_study_defaults_to_empty_list_when_none_derivable(self):
        result = parse_education_requirement("Bachelor's degree required")
        assert result.fields_of_study == []

    def test_is_case_insensitive(self):
        result = parse_education_requirement("BACHELOR'S DEGREE REQUIRED")
        assert result.minimum_level == EducationLevel.BACHELORS

    @pytest.mark.parametrize(
        "text, expected_fields",
        [
            ("Bachelor's in Computer Science or related field", ["Computer Science", "related field"]),
            (
                "Bachelor's degree in Computer Science, Statistics, or a related quantitative field",
                ["Computer Science", "Statistics", "a related quantitative field"],
            ),
            ("PhD in Machine Learning", ["Machine Learning"]),
            ("Master's in Data Science", ["Data Science"]),
            ("Bachelor's degree required", []),
            ("Bachelor's degree or equivalent experience", []),
        ],
    )
    def test_fields_of_study_derived_deterministically_when_not_overridden(
        self, text, expected_fields
    ):
        # No explicit fields_of_study argument passed here -- this is
        # the production code path (app.job_extractor never has an LLM-
        # supplied list to pass; see app.schemas.RawJobRequirementsExtraction's
        # docstring for why). Regex-derived from the same verbatim text
        # that also produces raw_text/minimum_level.
        result = parse_education_requirement(text)
        assert result.fields_of_study == expected_fields

    def test_explicit_fields_of_study_overrides_derivation(self):
        # An explicit argument takes precedence over what the regex
        # would derive from the text itself.
        result = parse_education_requirement(
            "Bachelor's in Computer Science", fields_of_study=["Explicit Override"]
        )
        assert result.fields_of_study == ["Explicit Override"]

    def test_explicit_empty_list_overrides_derivation_too(self):
        # An explicit empty list is a meaningful override (not the same
        # as "not provided"), so it must NOT trigger derivation either.
        result = parse_education_requirement(
            "Bachelor's in Computer Science", fields_of_study=[]
        )
        assert result.fields_of_study == []


class TestDeriveSeniority:
    @pytest.mark.parametrize(
        "title, expected",
        [
            ("Senior Software Engineer", Seniority.SENIOR),
            ("Sr. Data Scientist", Seniority.SENIOR),
            ("Sr Data Scientist", Seniority.SENIOR),
            ("Junior Developer", Seniority.JUNIOR),
            ("Entry-Level Analyst", Seniority.JUNIOR),
            ("Entry Level Analyst", Seniority.JUNIOR),
            ("Software Engineering Intern", Seniority.INTERN),
            ("Marketing Trainee", Seniority.INTERN),
            ("Mid-Level Backend Engineer", Seniority.MID),
            ("Staff Engineer", Seniority.LEAD),
            ("Lead Data Engineer", Seniority.LEAD),
            ("Principal Architect", Seniority.PRINCIPAL),
            ("Head of Engineering", Seniority.PRINCIPAL),
        ],
    )
    def test_recognized_titles(self, title, expected):
        assert derive_seniority(title) == expected

    @pytest.mark.parametrize("title", [None, "", "   ", "Software Engineer", "Data Analyst"])
    def test_titles_with_no_seniority_signal_return_none(self, title):
        assert derive_seniority(title) is None

    def test_multiple_matches_take_the_most_senior(self):
        # "Senior Staff Engineer" matches both "senior" (SENIOR) and
        # "staff" (LEAD) -- the more senior of the two must win.
        assert derive_seniority("Senior Staff Engineer") == Seniority.LEAD

    def test_word_boundary_matching_avoids_false_positives(self):
        # "lead" must not spuriously match inside an unrelated word.
        assert derive_seniority("Leadership Development Program Analyst") is None

    def test_punctuation_does_not_prevent_a_match(self):
        assert derive_seniority("Sr., Backend Engineer") == Seniority.SENIOR
