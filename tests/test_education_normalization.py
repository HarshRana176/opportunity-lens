"""
Characterizes app.education: deterministic normalization of raw
education records extracted (verbatim) from a résumé. No LLM involved
anywhere in this file.

Does NOT touch app.requirements' EDUCATION_LEVEL_TERMS (JD side) --
that table and its 22-entry count are unchanged by Task 6, and this
file includes an explicit guard for that.
"""
import pytest

from app.education import (
    build_education_background,
    compute_degree_key,
    normalize_education_record,
)
from app.schemas import EducationLevel, RawEducationRecord
from app.taxonomy import DEGREE_CANONICAL, EDUCATION_LEVEL_TERMS


def _raw(degree, field=None, institution=None, completion=None):
    return RawEducationRecord(
        degree=degree,
        field_of_study=field,
        institution=institution,
        completion_text=completion,
    )


class TestTaxonomyIsolation:
    """Guards that Task 6 did not touch the Task 4 JD-side table."""

    def test_education_level_terms_is_unchanged(self):
        assert len(EDUCATION_LEVEL_TERMS) == 22

    def test_degree_canonical_is_a_separate_table(self):
        assert DEGREE_CANONICAL is not EDUCATION_LEVEL_TERMS

    def test_degree_canonical_values_are_all_valid_education_levels(self):
        valid_names = set(EducationLevel.__members__)
        assert all(v in valid_names for v in DEGREE_CANONICAL.values())


class TestComputeDegreeKey:
    @pytest.mark.parametrize(
        "text, expected_key",
        [
            ("B. Tech", "btech"),
            ("B.Tech", "btech"),
            ("BTech", "btech"),
            ("B.Tech.", "btech"),
            ("  B. Tech  ", "btech"),
            ("B.TECH", "btech"),
            ("b.tech", "btech"),
            ("Class X", "classx"),
            ("Class XII", "classxii"),
            ("M. Tech", "mtech"),
        ],
    )
    def test_squashes_punctuation_whitespace_and_casing(self, text, expected_key):
        assert compute_degree_key(text) == expected_key

    def test_classx_and_classxii_are_distinct_keys(self):
        # Regression guard: a careless squash could collapse these.
        assert compute_degree_key("Class X") != compute_degree_key("Class XII")


class TestDegreeCanonicalMapping:
    @pytest.mark.parametrize(
        "degree_text, expected_level",
        [
            ("B. Tech", EducationLevel.BACHELORS),
            ("B.Tech", EducationLevel.BACHELORS),
            ("BTech", EducationLevel.BACHELORS),
            ("Bachelor of Technology", EducationLevel.BACHELORS),
            ("B.E.", EducationLevel.BACHELORS),
            ("BE", EducationLevel.BACHELORS),
            ("B.S.", EducationLevel.BACHELORS),
            ("BSc", EducationLevel.BACHELORS),
            ("BCA", EducationLevel.BACHELORS),
            ("M.Tech", EducationLevel.MASTERS),
            ("MTech", EducationLevel.MASTERS),
            ("MSc", EducationLevel.MASTERS),
            ("MBA", EducationLevel.MASTERS),
            ("MCA", EducationLevel.MASTERS),
            ("PhD", EducationLevel.DOCTORATE),
            ("Ph.D.", EducationLevel.DOCTORATE),
            ("Doctorate", EducationLevel.DOCTORATE),
            ("Diploma", EducationLevel.ASSOCIATE),
            ("Class X", EducationLevel.HIGH_SCHOOL),
            ("Class XII", EducationLevel.HIGH_SCHOOL),
            ("GED", EducationLevel.HIGH_SCHOOL),
        ],
    )
    def test_known_degrees_resolve_to_the_correct_level(self, degree_text, expected_level):
        record = normalize_education_record(_raw(degree_text))
        assert record.level == expected_level
        assert record.resolution == "taxonomy"

    def test_unknown_degree_is_retained_as_unresolved_not_dropped(self):
        record = normalize_education_record(_raw("Bachelor of Underwater Basket Weaving"))

        assert record.level is None
        assert record.resolution == "unresolved"
        assert record.degree_raw == "Bachelor of Underwater Basket Weaving"

    def test_degree_key_is_always_populated_even_when_unresolved(self):
        record = normalize_education_record(_raw("Certificate in Something Novel"))

        assert record.degree_key == "certificateinsomethingnovel"
        assert record.resolution == "unresolved"


class TestD6RawPreservation:
    """
    Pins the exact D6 clarification: Class X / Class XII normalize to
    HIGH_SCHOOL for canonical comparison, but degree_raw is NEVER
    rewritten to "High School" or anything else.
    """

    def test_class_x_exact_contract(self):
        record = normalize_education_record(_raw("Class X"))

        assert record.degree_raw == "Class X"
        assert record.degree_key == "classx"
        assert record.level == EducationLevel.HIGH_SCHOOL
        assert record.resolution == "taxonomy"

    def test_class_xii_exact_contract(self):
        record = normalize_education_record(_raw("Class XII"))

        assert record.degree_raw == "Class XII"
        assert record.degree_key == "classxii"
        assert record.level == EducationLevel.HIGH_SCHOOL
        assert record.resolution == "taxonomy"

    def test_class_x_and_class_xii_share_a_level_but_not_a_key(self):
        x = normalize_education_record(_raw("Class X"))
        xii = normalize_education_record(_raw("Class XII"))

        assert x.level == xii.level == EducationLevel.HIGH_SCHOOL
        assert x.degree_key != xii.degree_key

    def test_field_of_study_institution_and_completion_are_preserved_verbatim(self):
        record = normalize_education_record(
            _raw(
                "B. Tech",
                field="Electronics & Communication Engineering",
                institution="Manipal Institute of Technology",
                completion="2026",
            )
        )

        assert record.field_of_study_raw == "Electronics & Communication Engineering"
        assert record.institution_raw == "Manipal Institute of Technology"
        assert record.completion_raw == "2026"

    def test_completion_raw_never_becomes_a_date_even_when_percentage_like(self):
        record = normalize_education_record(_raw("Class XII", completion="92%"))

        assert record.completion_raw == "92%"
        assert isinstance(record.completion_raw, str)

    def test_none_optional_fields_stay_none_not_fabricated(self):
        record = normalize_education_record(_raw("PhD"))

        assert record.field_of_study_raw is None
        assert record.institution_raw is None
        assert record.completion_raw is None


class TestBuildEducationBackground:
    def test_no_records_yields_none_not_an_empty_background(self):
        background, warnings = build_education_background([], None)

        # None means "no education section found" -- distinguishable
        # from an EducationBackground with an empty `records` list,
        # which is never produced by this function.
        assert background is None
        assert warnings == []

    def test_single_record(self):
        background, warnings = build_education_background(
            [_raw("B. Tech", field="Computer Science")], "raw text"
        )

        assert len(background.records) == 1
        assert background.highest_level == EducationLevel.BACHELORS
        assert background.raw_text == "raw text"
        assert warnings == []

    def test_multiple_records_are_all_preserved(self):
        raws = [
            _raw("B. Tech", field="Electronics & Communication Engineering",
                 institution="Manipal Institute of Technology", completion="2026"),
            _raw("Class XII", institution="Delhi Public School, Greater Noida", completion="92%"),
            _raw("Class X", institution="Delhi Public School, Greater Noida", completion="90.6%"),
        ]

        background, warnings = build_education_background(raws, "raw text")

        assert len(background.records) == 3
        assert [r.degree_raw for r in background.records] == ["B. Tech", "Class XII", "Class X"]
        assert warnings == []

    def test_highest_level_is_the_maximum_across_resolved_records(self):
        raws = [_raw("Class X"), _raw("Class XII"), _raw("B. Tech")]

        background, _ = build_education_background(raws, None)

        assert background.highest_level == EducationLevel.BACHELORS

    def test_highest_level_is_none_when_no_record_resolves(self):
        raws = [_raw("Certificate in Novel Field A"), _raw("Certificate in Novel Field B")]

        background, warnings = build_education_background(raws, None)

        assert background.highest_level is None
        assert len(background.records) == 2
        assert len(warnings) == 2

    def test_mixed_resolved_and_unresolved_records_all_survive(self):
        raws = [_raw("B. Tech"), _raw("Certificate in Something Unusual")]

        background, warnings = build_education_background(raws, None)

        assert len(background.records) == 2
        assert background.highest_level == EducationLevel.BACHELORS
        assert len(warnings) == 1
        assert "Something Unusual" in warnings[0]

    def test_unresolved_record_warning_names_the_degree(self):
        _, warnings = build_education_background(
            [_raw("Bachelor of Underwater Basket Weaving")], None
        )

        assert len(warnings) == 1
        assert "Bachelor of Underwater Basket Weaving" in warnings[0]

    def test_resolved_records_produce_no_warnings(self):
        _, warnings = build_education_background([_raw("PhD"), _raw("MBA")], None)

        assert warnings == []


class TestAdversarialDegreeForms:
    """
    Review-fix regression suite: compute_degree_key previously only
    stripped '.' and whitespace, so common possessive phrasings
    ("Bachelor's"), parenthetical qualifiers ("(Hons)"), and comma/
    slash-separated equivalents ("B.A./B.S.") silently failed to
    resolve, even though an unpunctuated equivalent form (e.g.
    "Bachelors") was already in DEGREE_CANONICAL.

    Every expected level below is HAND-WRITTEN, not derived from
    DEGREE_CANONICAL or compute_degree_key -- these must fail if the
    normalization logic regresses, not merely restate whatever it
    currently does.
    """

    @pytest.mark.parametrize(
        "degree_text, expected_level",
        [
            ("Bachelor's", EducationLevel.BACHELORS),
            ("Bachelor's Degree", EducationLevel.BACHELORS),
            ("Master's", EducationLevel.MASTERS),
            ("Master's Degree", EducationLevel.MASTERS),
            ("Associate's", EducationLevel.ASSOCIATE),
            ("B.Sc. (Hons.)", EducationLevel.BACHELORS),
            ("B.A./B.S.", EducationLevel.BACHELORS),
            ("M.S., Computer Science", EducationLevel.MASTERS),
            ("B.Tech (Hons)", EducationLevel.BACHELORS),
        ],
    )
    def test_adversarial_forms_resolve_to_the_correct_level(self, degree_text, expected_level):
        record = normalize_education_record(_raw(degree_text))

        assert record.level == expected_level
        assert record.resolution == "taxonomy"

    @pytest.mark.parametrize(
        "degree_text",
        [
            "Bachelor's",
            "Bachelor's Degree",
            "Master's",
            "Master's Degree",
            "Associate's",
            "B.Sc. (Hons.)",
            "B.A./B.S.",
            "M.S., Computer Science",
            "B.Tech (Hons)",
        ],
    )
    def test_adversarial_forms_preserve_degree_raw_exactly(self, degree_text):
        # The whole point of fixing normalization: degree_raw must stay
        # byte-for-byte identical to the input regardless of how much
        # smarter the level-resolution logic gets.
        record = normalize_education_record(_raw(degree_text))

        assert record.degree_raw == degree_text

    def test_parenthetical_qualifier_does_not_change_degree_raw_or_key_identity(self):
        # "(Hons)" is dropped only for LEVEL LOOKUP purposes; degree_raw
        # is untouched, and the plain form without the qualifier still
        # produces the same resolved level (though not necessarily the
        # same degree_key, since degree_key is a whole-string identity).
        with_hons = normalize_education_record(_raw("B.Tech (Hons)"))
        without_hons = normalize_education_record(_raw("B.Tech"))

        assert with_hons.degree_raw == "B.Tech (Hons)"
        assert with_hons.level == without_hons.level == EducationLevel.BACHELORS

    def test_slash_separated_equivalent_degrees_resolve_via_the_first_segment(self):
        record = normalize_education_record(_raw("B.A./B.S."))

        assert record.level == EducationLevel.BACHELORS
        assert record.resolution == "taxonomy"

    def test_comma_separated_degree_and_field_resolves_via_the_degree_segment(self):
        record = normalize_education_record(_raw("M.S., Computer Science"))

        assert record.level == EducationLevel.MASTERS

    def test_genuinely_unresolvable_comma_separated_text_stays_unresolved(self):
        # Neither segment is a real degree -- must not resolve, and
        # must not raise.
        record = normalize_education_record(_raw("Novel Program A, Novel Program B"))

        assert record.level is None
        assert record.resolution == "unresolved"
        assert record.degree_raw == "Novel Program A, Novel Program B"


class TestEmptyStringRawFieldPreservation:
    """
    Regression guard: some LLM responses return "" rather than null for
    an omitted optional field. Raw preservation means passing that
    value through EXACTLY as given -- never coercing "" to None, and
    never coercing None to "".
    """

    def test_empty_string_field_of_study_is_preserved_not_converted_to_none(self):
        raw = RawEducationRecord(degree="Class XII", field_of_study="", institution=None)

        record = normalize_education_record(raw)

        assert record.field_of_study_raw == ""
        assert record.field_of_study_raw is not None

    def test_empty_string_institution_is_preserved_not_converted_to_none(self):
        raw = RawEducationRecord(degree="Class XII", institution="")

        record = normalize_education_record(raw)

        assert record.institution_raw == ""

    def test_empty_string_completion_text_is_preserved_not_converted_to_none(self):
        raw = RawEducationRecord(degree="Class XII", completion_text="")

        record = normalize_education_record(raw)

        assert record.completion_raw == ""

    def test_none_field_of_study_stays_none_not_coerced_to_empty_string(self):
        # The inverse direction: this module must not paper over the
        # distinction the LLM itself already made.
        raw = RawEducationRecord(degree="Class XII", field_of_study=None)

        record = normalize_education_record(raw)

        assert record.field_of_study_raw is None
