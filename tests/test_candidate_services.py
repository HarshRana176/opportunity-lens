"""
Characterizes the Task 5 service layer and database round-trip:
app.services.create_candidate_profile / get_candidate_profile, and the
app.models.CandidateProfile table.

Task 5 adds NO HTTP endpoint (approved decision D4), so these are
service-level tests only -- there is deliberately no TestClient here.

Uses in-memory SQLite via StaticPool (same rationale as
tests/test_api.py's db_session fixture) and stubs
app.services.build_candidate_profile, so nothing here needs real
Postgres, Ollama, or a real PDF.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services as services
from app import models
from app.database import Base
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateSkill,
    Seniority,
)


@pytest.fixture
def db_session():
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


def _profile(**overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        seniority=Seniority.SENIOR,
        current_role="Senior Backend Engineer",
        skills=[
            CandidateSkill(
                raw="Python",
                match_key="python",
                canonical="python",
                category="programming_languages",
                resolution="taxonomy",
            ),
            CandidateSkill(
                raw="Kafka",
                match_key="kafka",
                canonical=None,
                category="tools",
                resolution="llm",
            ),
        ],
        total_experience_months=48,
        total_experience_years=4.0,
        employment_history=[
            CandidateEmployment(
                company="Acme",
                role="Senior Backend Engineer",
                start_date="Jan 2022",
                end_date="Dec 2025",
                start_month_index=24264,
                end_month_index=24311,
                duration_months=48,
                seniority=Seniority.SENIOR,
                is_current=False,
            )
        ],
        education=None,
        raw_text="FULL RESUME TEXT",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


class TestCreateCandidateProfile:
    def test_persists_and_returns_the_record(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())

        record = services.create_candidate_profile(db_session, "fake.pdf")

        assert record.id is not None
        assert record.candidate_name == "Jane Doe"
        assert record.total_experience_months == 48
        assert record.current_role == "Senior Backend Engineer"

    def test_seniority_is_stored_as_a_plain_integer(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())

        record = services.create_candidate_profile(db_session, "fake.pdf")

        assert record.seniority == int(Seniority.SENIOR)

    def test_none_seniority_persists_as_null(self, db_session, monkeypatch):
        monkeypatch.setattr(
            services,
            "build_candidate_profile",
            lambda path: _profile(seniority=None, current_role=None),
        )

        record = services.create_candidate_profile(db_session, "fake.pdf")

        assert record.seniority is None
        assert record.current_role is None

    def test_resume_id_lineage_is_stored_when_provided(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())

        record = services.create_candidate_profile(db_session, "fake.pdf", resume_id=7)

        assert record.resume_id == 7

    def test_resume_id_defaults_to_none(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())

        record = services.create_candidate_profile(db_session, "fake.pdf")

        assert record.resume_id is None

    def test_extraction_failure_propagates_and_persists_nothing(
        self, db_session, monkeypatch
    ):
        def _raise(path):
            raise ValueError("Could not extract text from the PDF.")

        monkeypatch.setattr(services, "build_candidate_profile", _raise)

        with pytest.raises(ValueError):
            services.create_candidate_profile(db_session, "fake.pdf")

        assert db_session.query(models.CandidateProfile).count() == 0

    def test_commit_failure_rolls_back(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())

        def _raise_on_commit():
            raise RuntimeError("simulated database failure")

        monkeypatch.setattr(db_session, "commit", _raise_on_commit)

        with pytest.raises(RuntimeError):
            services.create_candidate_profile(db_session, "fake.pdf")


class TestGetCandidateProfile:
    def test_returns_none_when_absent(self, db_session):
        assert services.get_candidate_profile(db_session, 999) is None

    def test_returns_the_matching_record(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.candidate_name == "Jane Doe"


class TestDatabaseRoundTrip:
    def test_skills_survive_the_round_trip_with_full_fidelity(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)
        by_raw = {s["raw"]: s for s in fetched.skills}

        assert by_raw["Python"]["canonical"] == "python"
        assert by_raw["Python"]["resolution"] == "taxonomy"
        # The unresolved-but-enriched skill keeps canonical=None -- the
        # never-invent-canonical rule survives persistence.
        assert by_raw["Kafka"]["canonical"] is None
        assert by_raw["Kafka"]["category"] == "tools"
        assert by_raw["Kafka"]["resolution"] == "llm"

    def test_skills_rehydrate_into_candidate_skill_objects(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)
        rehydrated = [CandidateSkill(**s) for s in fetched.skills]

        assert {s.canonical for s in rehydrated} == {"python", None}

    def test_seniority_rehydrates_into_the_ordinal_enum(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)
        seniority = Seniority(fetched.seniority)

        assert seniority == Seniority.SENIOR
        assert seniority >= Seniority.MID  # ordinal comparison still works

    def test_employment_history_survives_the_round_trip(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)
        entry = fetched.employment_history[0]

        assert entry["company"] == "Acme"
        assert entry["start_date"] == "Jan 2022"
        assert entry["duration_months"] == 48

    def test_raw_text_survives_the_round_trip(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)

        assert fetched.raw_text == "FULL RESUME TEXT"

    def test_education_persists_as_null_in_task_5(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)

        assert fetched.education is None


class TestResumeTableUnaffected:
    """Task 5 must not disturb the frozen Resume contract."""

    def test_resume_table_columns_are_unchanged(self):
        assert [c.name for c in models.Resume.__table__.columns] == [
            "id",
            "candidate_name",
            "technical_stack",
            "employment_history",
            "total_experience_months",
            "total_experience_years",
        ]

    def test_candidate_profiles_is_a_separate_table(self):
        assert models.CandidateProfile.__tablename__ == "candidate_profiles"
        assert models.Resume.__tablename__ == "resumes"
