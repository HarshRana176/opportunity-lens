"""
Characterizes app.services: the storage -> extraction -> persistence
orchestration, listing/pagination, lookup, and failure cleanup.

Uses an in-memory SQLite database (via the `db_session` fixture) rather
than real PostgreSQL, and monkeypatches `app.services.extract_resume`
rather than calling the real LLM -- both to keep these tests hermetic.
`app.storage.get_settings` is monkeypatched so uploads land under
tmp_path rather than the real uploads/ directory.
"""
import io
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.services as services
import app.storage as storage_module
from app import models
from app.database import Base
from app.schemas import EmploymentPeriod, ResumeExtraction, TechnicalStack
from app.storage import InvalidUploadError


@pytest.fixture
def db_session():
    # StaticPool: force every checkout to share one connection so the
    # tables created below are visible regardless of which thread later
    # queries run on. See test_api.py's db_session fixture for why this
    # matters there; applied here too for the same robustness even
    # though these tests happen to call services functions directly on
    # a single thread today.
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


@pytest.fixture
def fake_upload_settings(tmp_path, monkeypatch):
    settings = SimpleNamespace(upload_dir=str(tmp_path), max_upload_bytes=10_000_000)
    monkeypatch.setattr(storage_module, "get_settings", lambda: settings)
    return settings


def _fake_extraction():
    return ResumeExtraction(
        candidate_name="Jane Doe",
        technical_stack=TechnicalStack(programming_languages=["Python"]),
        employment_history=[
            EmploymentPeriod(company="Acme", start_date="May 2025", end_date="Aug 2025")
        ],
        total_experience_months=4,
        total_experience_years=0.33,
    )


class TestCreateResumeFromUpload:
    def test_persists_and_returns_the_record(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())

        resume = services.create_resume_from_upload(
            db_session, io.BytesIO(b"%PDF-1.4\nbody"), "resume.pdf"
        )

        assert resume.id is not None
        assert resume.candidate_name == "Jane Doe"
        assert resume.technical_stack["programming_languages"] == ["Python"]
        assert resume.total_experience_months == 4
        assert resume.total_experience_years == 0.33

        # Actually persisted, not just returned in-memory.
        fetched = services.get_resume(db_session, resume.id)
        assert fetched is not None
        assert fetched.candidate_name == "Jane Doe"

    def test_calls_extract_resume_with_the_stored_file_path(
        self, db_session, fake_upload_settings, tmp_path, monkeypatch
    ):
        captured_paths = []

        def _capture(path):
            captured_paths.append(path)
            return _fake_extraction()

        monkeypatch.setattr(services, "extract_resume", _capture)

        services.create_resume_from_upload(
            db_session, io.BytesIO(b"%PDF-1.4\nbody"), "resume.pdf"
        )

        assert len(captured_paths) == 1
        assert str(tmp_path) in captured_paths[0]

    def test_rejects_non_pdf_content_before_extraction_is_attempted(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        invoked = []
        monkeypatch.setattr(
            services, "extract_resume", lambda path: invoked.append(path)
        )

        with pytest.raises(InvalidUploadError):
            services.create_resume_from_upload(
                db_session, io.BytesIO(b"not a pdf"), "resume.pdf"
            )

        assert invoked == []
        assert db_session.query(models.Resume).count() == 0

    def test_cleans_up_the_stored_file_on_extraction_failure(
        self, db_session, fake_upload_settings, tmp_path, monkeypatch
    ):
        def _raise(path):
            raise ValueError("Could not extract text from the PDF.")

        monkeypatch.setattr(services, "extract_resume", _raise)

        with pytest.raises(ValueError):
            services.create_resume_from_upload(
                db_session, io.BytesIO(b"%PDF-1.4\nbody"), "resume.pdf"
            )

        assert list(tmp_path.iterdir()) == []
        assert db_session.query(models.Resume).count() == 0

    def test_cleans_up_the_stored_file_on_persistence_failure(
        self, db_session, fake_upload_settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())

        def _raise_on_commit():
            raise RuntimeError("simulated database failure")

        monkeypatch.setattr(db_session, "commit", _raise_on_commit)

        with pytest.raises(RuntimeError):
            services.create_resume_from_upload(
                db_session, io.BytesIO(b"%PDF-1.4\nbody"), "resume.pdf"
            )

        assert list(tmp_path.iterdir()) == []


class TestListResumes:
    def _seed(self, db_session, count):
        for i in range(count):
            db_session.add(
                models.Resume(
                    candidate_name=f"Person {i}",
                    technical_stack={
                        "programming_languages": [],
                        "frameworks": [],
                        "tools": [],
                    },
                    employment_history=[],
                    total_experience_months=0,
                    total_experience_years=0.0,
                )
            )
        db_session.commit()

    def test_orders_by_id(self, db_session):
        self._seed(db_session, 3)

        results = services.list_resumes(db_session, limit=10, offset=0)

        assert [r.candidate_name for r in results] == ["Person 0", "Person 1", "Person 2"]

    def test_pagination_boundaries(self, db_session):
        self._seed(db_session, 5)

        page1 = services.list_resumes(db_session, limit=2, offset=0)
        page2 = services.list_resumes(db_session, limit=2, offset=2)
        page3 = services.list_resumes(db_session, limit=2, offset=4)
        past_the_end = services.list_resumes(db_session, limit=2, offset=10)

        assert [r.candidate_name for r in page1] == ["Person 0", "Person 1"]
        assert [r.candidate_name for r in page2] == ["Person 2", "Person 3"]
        assert [r.candidate_name for r in page3] == ["Person 4"]
        assert past_the_end == []

    def test_empty_table_returns_empty_list(self, db_session):
        assert services.list_resumes(db_session, limit=10, offset=0) == []


class TestGetResume:
    def test_returns_none_when_absent(self, db_session):
        assert services.get_resume(db_session, 999) is None

    def test_returns_the_matching_record(self, db_session):
        resume = models.Resume(
            candidate_name="Jane Doe",
            technical_stack={"programming_languages": [], "frameworks": [], "tools": []},
            employment_history=[],
            total_experience_months=0,
            total_experience_years=0.0,
        )
        db_session.add(resume)
        db_session.commit()

        fetched = services.get_resume(db_session, resume.id)

        assert fetched is not None
        assert fetched.id == resume.id
        assert fetched.candidate_name == "Jane Doe"
