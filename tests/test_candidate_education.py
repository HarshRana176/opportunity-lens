"""
Characterizes education handling inside
app.candidate_extractor.build_candidate_profile.

The global `_stub_education_extraction_chain` autouse fixture
(tests/conftest.py) defaults every test's education chain to an empty
extraction; tests in this file override
`candidate_extractor.education_extraction_chain` per-test to exercise
real education scenarios, following the same monkeypatch pattern
tests/test_candidate_extractor.py already uses for
`extraction_chain`/`batch_skill_classifier_chain`.

No real PDF, Ollama, or Postgres access anywhere in this file.
"""
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.candidate_extractor as candidate_extractor
import app.services as services
from app.database import Base
from app.schemas import (
    CandidateProfile,
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EmploymentPeriod,
    RawEducationExtraction,
    RawEducationRecord,
    RawResumeExtraction,
)


def _raw_resume(candidate_name="Jane Doe", employment_history=None, skills=None):
    return RawResumeExtraction(
        candidate_name=candidate_name,
        employment_history=employment_history or [],
        skills=skills or [],
    )


def _period(company, role, start_date, end_date):
    return EmploymentPeriod(
        company=company, role=role, start_date=start_date, end_date=end_date
    )


def _edu_record(degree, field=None, institution=None, completion=None):
    return RawEducationRecord(
        degree=degree, field_of_study=field, institution=institution, completion_text=completion
    )


def _stub(monkeypatch, raw_resume=None, education_records=None, education_raises=False,
          text="Jane Doe resume text"):
    """
    Same shape as test_candidate_extractor.py's _stub, plus control
    over the education chain (which that file's helper deliberately
    leaves at the conftest default of "no education").
    """
    monkeypatch.setattr(candidate_extractor, "extract_text_from_pdf", lambda path: text)
    monkeypatch.setattr(
        candidate_extractor,
        "extraction_chain",
        Mock(invoke=Mock(return_value=raw_resume or _raw_resume())),
    )

    if education_raises:
        education_chain = Mock()
        education_chain.invoke.side_effect = RuntimeError("ollama unreachable")
    else:
        education_chain = Mock(
            invoke=Mock(
                return_value=RawEducationExtraction(education=education_records or [])
            )
        )
    monkeypatch.setattr(candidate_extractor, "education_extraction_chain", education_chain)

    return education_chain


class TestSingleEducationRecord:
    def test_single_degree_is_populated_on_the_profile(self, monkeypatch):
        _stub(monkeypatch, education_records=[_edu_record("B. Tech", field="Computer Science")])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education is not None
        assert len(profile.education.records) == 1
        assert profile.education.records[0].degree_raw == "B. Tech"
        assert profile.education.highest_level == EducationLevel.BACHELORS


class TestMultipleEducationRecords:
    def test_all_records_are_preserved_not_reduced_to_one(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[
                _edu_record(
                    "B. Tech",
                    field="Electronics & Communication Engineering",
                    institution="Manipal Institute of Technology",
                    completion="2026",
                ),
                _edu_record(
                    "Class XII",
                    institution="Delhi Public School, Greater Noida",
                    completion="92%",
                ),
                _edu_record(
                    "Class X",
                    institution="Delhi Public School, Greater Noida",
                    completion="90.6%",
                ),
            ],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.education.records) == 3
        assert [r.degree_raw for r in profile.education.records] == [
            "B. Tech",
            "Class XII",
            "Class X",
        ]

    def test_highest_level_reflects_the_best_record(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[_edu_record("Class X"), _edu_record("Class XII"), _edu_record("B. Tech")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education.highest_level == EducationLevel.BACHELORS

    def test_a_masters_and_a_bachelors_both_survive(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[
                _edu_record("B.Tech", field="Electronics & Communication Engineering"),
                _edu_record("M.Tech", field="Computer Science"),
            ],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.education.records) == 2
        assert profile.education.highest_level == EducationLevel.MASTERS
        fields = {r.field_of_study_raw for r in profile.education.records}
        assert fields == {"Electronics & Communication Engineering", "Computer Science"}


class TestMissingEducation:
    def test_no_education_section_yields_none(self, monkeypatch):
        # The conftest autouse default already does this; asserted
        # explicitly here for this file's own regression coverage.
        _stub(monkeypatch, education_records=[])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education is None

    def test_extraction_failure_yields_none_plus_a_warning_not_a_crash(self, monkeypatch):
        _stub(monkeypatch, education_raises=True)

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education is None
        assert any("education" in w.lower() for w in profile.parse_warnings)

    def test_extraction_failure_does_not_prevent_the_rest_of_the_profile(self, monkeypatch):
        _stub(
            monkeypatch,
            raw_resume=_raw_resume(
                candidate_name="Jane Doe",
                skills=["Python"],
                employment_history=[_period("Acme", "Engineer", "May 2025", "Aug 2025")],
            ),
            education_raises=True,
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.candidate_name == "Jane Doe"
        assert len(profile.skills) == 1
        assert profile.total_experience_months == 4
        assert profile.education is None


class TestAmbiguousAndPartialEducation:
    def test_unrecognized_degree_is_retained_not_dropped(self, monkeypatch):
        _stub(monkeypatch, education_records=[_edu_record("Diplôme d'Ingénieur")])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.education.records) == 1
        assert profile.education.records[0].degree_raw == "Diplôme d'Ingénieur"
        assert profile.education.records[0].resolution == "unresolved"
        assert profile.education.highest_level is None

    def test_unresolved_degree_adds_a_parse_warning(self, monkeypatch):
        _stub(monkeypatch, education_records=[_edu_record("Bachelor of Underwater Basket Weaving")])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert any("Underwater Basket Weaving" in w for w in profile.parse_warnings)

    def test_partial_record_with_no_field_or_institution(self, monkeypatch):
        _stub(monkeypatch, education_records=[_edu_record("MBA")])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        record = profile.education.records[0]
        assert record.field_of_study_raw is None
        assert record.institution_raw is None
        assert record.level == EducationLevel.MASTERS

    def test_mixed_resolved_and_unresolved_records_both_survive(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[_edu_record("B.Tech"), _edu_record("Certificate in Novel Field")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.education.records) == 2
        assert profile.education.highest_level == EducationLevel.BACHELORS
        assert any(r.resolution == "unresolved" for r in profile.education.records)


class TestRawWordingPreservation:
    def test_class_x_and_class_xii_keep_their_exact_wording(self, monkeypatch):
        # The D6 contract, end to end through the candidate builder.
        _stub(
            monkeypatch,
            education_records=[_edu_record("Class X"), _edu_record("Class XII")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        by_degree = {r.degree_raw: r for r in profile.education.records}
        assert by_degree["Class X"].degree_key == "classx"
        assert by_degree["Class X"].level == EducationLevel.HIGH_SCHOOL
        assert by_degree["Class XII"].degree_key == "classxii"
        assert by_degree["Class XII"].level == EducationLevel.HIGH_SCHOOL
        # Neither degree_raw was rewritten to "High School".
        assert "High School" not in by_degree

    def test_completion_text_is_preserved_verbatim_and_never_becomes_a_date(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[_edu_record("Class XII", completion="92%")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        record = profile.education.records[0]
        assert record.completion_raw == "92%"
        assert isinstance(record.completion_raw, str)

    def test_no_completion_text_stays_none_not_fabricated(self, monkeypatch):
        _stub(monkeypatch, education_records=[_edu_record("PhD")])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education.records[0].completion_raw is None

    def test_institution_and_field_are_preserved_exactly(self, monkeypatch):
        _stub(
            monkeypatch,
            education_records=[
                _edu_record(
                    "B. Tech",
                    field="Electronics & Communication Engineering",
                    institution="Manipal Institute of Technology",
                )
            ],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        record = profile.education.records[0]
        assert record.field_of_study_raw == "Electronics & Communication Engineering"
        assert record.institution_raw == "Manipal Institute of Technology"


class TestNoHallucination:
    def test_empty_extraction_never_invents_a_record(self, monkeypatch):
        _stub(monkeypatch, education_records=[])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education is None

    def test_education_chain_is_never_called_with_candidate_skills_or_employment_data(
        self, monkeypatch
    ):
        # Confirms the education chain is invoked with the résumé text,
        # not with any already-extracted structured data (which would
        # imply results feeding back into re-extraction).
        chain = _stub(
            monkeypatch,
            education_records=[_edu_record("PhD")],
            text="THE FULL RESUME TEXT",
        )

        candidate_extractor.build_candidate_profile("fake.pdf")

        chain.invoke.assert_called_once_with({"resume_text": "THE FULL RESUME TEXT"})


class TestExistingCandidateProfileBehaviorUnaffected:
    """
    Regression guard: Task 6 must not change skills/employment/
    experience/seniority behavior established in Task 5.
    """

    def test_skills_and_employment_are_unaffected_by_education(self, monkeypatch):
        _stub(
            monkeypatch,
            raw_resume=_raw_resume(
                skills=["Python", "Docker"],
                employment_history=[_period("Acme", "Senior Engineer", "May 2025", "Aug 2025")],
            ),
            education_records=[_edu_record("B. Tech")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert {s.raw for s in profile.skills} == {"Python", "Docker"}
        assert profile.total_experience_months == 4
        assert profile.current_role == "Senior Engineer"

    def test_clean_profile_with_education_still_has_no_unrelated_warnings(self, monkeypatch):
        _stub(
            monkeypatch,
            raw_resume=_raw_resume(
                skills=["Python"],
                employment_history=[_period("Acme", "Engineer", "May 2025", "Aug 2025")],
            ),
            education_records=[_edu_record("B. Tech")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.parse_warnings == []


@pytest.fixture
def db_session():
    # Same pattern as tests/test_candidate_services.py's fixture of the
    # same name (StaticPool -- see that file for why it's required).
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _profile_with_education(education):
    return CandidateProfile(
        candidate_name="Jane Doe",
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
        education=education,
    )


class TestEducationDatabaseRoundTrip:
    """
    app.services.create_candidate_profile/get_candidate_profile and the
    education JSON column were not modified for Task 6 (the existing
    `.model_dump() if profile.education else None` call already handles
    the reshaped EducationBackground correctly) -- these tests exist to
    prove that empirically rather than merely by inspection.
    """

    def test_multiple_records_round_trip_with_full_fidelity(self, db_session, monkeypatch):
        education = EducationBackground(
            records=[
                EducationRecord(
                    degree_raw="B. Tech",
                    field_of_study_raw="Electronics & Communication Engineering",
                    institution_raw="Manipal Institute of Technology",
                    completion_raw="2026",
                    degree_key="btech",
                    level=EducationLevel.BACHELORS,
                    resolution="taxonomy",
                ),
                EducationRecord(
                    degree_raw="Class XII",
                    field_of_study_raw=None,
                    institution_raw="Delhi Public School, Greater Noida",
                    completion_raw="92%",
                    degree_key="classxii",
                    level=EducationLevel.HIGH_SCHOOL,
                    resolution="taxonomy",
                ),
            ],
            highest_level=EducationLevel.BACHELORS,
            raw_text="resume text",
        )
        monkeypatch.setattr(
            services, "build_candidate_profile", lambda path: _profile_with_education(education)
        )

        created = services.create_candidate_profile(db_session, "fake.pdf")
        fetched = services.get_candidate_profile(db_session, created.id)
        rehydrated = EducationBackground(**fetched.education)

        assert len(rehydrated.records) == 2
        assert rehydrated.records[0].degree_raw == "B. Tech"
        assert rehydrated.records[1].degree_raw == "Class XII"
        assert rehydrated.records[1].completion_raw == "92%"

    def test_highest_level_rehydrates_into_the_ordinal_enum(self, db_session, monkeypatch):
        education = EducationBackground(
            records=[
                EducationRecord(
                    degree_raw="PhD",
                    degree_key="phd",
                    level=EducationLevel.DOCTORATE,
                    resolution="taxonomy",
                )
            ],
            highest_level=EducationLevel.DOCTORATE,
        )
        monkeypatch.setattr(
            services, "build_candidate_profile", lambda path: _profile_with_education(education)
        )

        created = services.create_candidate_profile(db_session, "fake.pdf")
        fetched = services.get_candidate_profile(db_session, created.id)
        rehydrated = EducationBackground(**fetched.education)

        assert rehydrated.highest_level == EducationLevel.DOCTORATE
        assert rehydrated.highest_level >= EducationLevel.BACHELORS

    def test_unresolved_record_round_trips_with_level_none(self, db_session, monkeypatch):
        education = EducationBackground(
            records=[
                EducationRecord(
                    degree_raw="Bachelor of Underwater Basket Weaving",
                    degree_key="bachelorofunderwaterbasketweaving",
                    level=None,
                    resolution="unresolved",
                )
            ],
            highest_level=None,
        )
        monkeypatch.setattr(
            services, "build_candidate_profile", lambda path: _profile_with_education(education)
        )

        created = services.create_candidate_profile(db_session, "fake.pdf")
        fetched = services.get_candidate_profile(db_session, created.id)
        rehydrated = EducationBackground(**fetched.education)

        assert rehydrated.records[0].resolution == "unresolved"
        assert rehydrated.records[0].level is None
        assert rehydrated.highest_level is None

    def test_none_education_persists_as_null(self, db_session, monkeypatch):
        monkeypatch.setattr(
            services, "build_candidate_profile", lambda path: _profile_with_education(None)
        )

        created = services.create_candidate_profile(db_session, "fake.pdf")
        fetched = services.get_candidate_profile(db_session, created.id)

        assert fetched.education is None
