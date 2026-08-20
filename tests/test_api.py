"""
Characterizes the HTTP layer in app.main: status codes, response
shapes, pagination, and /health liveness.

`TestClient(main.app)` is constructed WITHOUT the `with` context-manager
form deliberately -- Starlette only fires ASGI lifespan (startup/
shutdown) events when the client is used as a context manager
(empirically verified: a bare `TestClient(app)` never runs `lifespan`).
Since this app's lifespan calls `Base.metadata.create_all(bind=
get_engine())` against the REAL configured database, entering it here
would require real PostgreSQL. Avoiding `with` keeps these tests
hermetic; routes still work identically because `get_db` is overridden
below, so no route ever touches `get_engine()` either.

An in-memory SQLite database backs `get_db` for every test in this
file, and `app.services.extract_resume` is monkeypatched per test (the
route calls `services.create_resume_from_upload`, which calls the bare
name `extract_resume` inside app.services -- patching the attribute on
the `app.services` module, which `app.main` also holds a reference to,
is sufficient; see the equivalent pattern already established for
`extractor.pymupdf` in Task 2).
"""
import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main
import app.services as services
import app.storage as storage_module
from app.database import Base, get_db
from app.schemas import EmploymentPeriod, ResumeExtraction, TechnicalStack


@pytest.fixture
def db_session():
    # StaticPool -- not just check_same_thread=False -- is required
    # here: routes run via Starlette's threadpool (a consequence of the
    # required async def -> def change for blocking extraction work),
    # so the query connection is on a different thread than the one
    # that ran create_all() below. SQLite's default per-thread pooling
    # would hand that query a fresh, table-less in-memory database;
    # StaticPool forces every checkout to reuse the same connection
    # regardless of thread.
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
def client(db_session):
    def _override_get_db():
        yield db_session

    main.app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()


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


PDF_BYTES = b"%PDF-1.4\nfake pdf body"

# The exact key set of the ORIGINAL hand-built response dicts in
# app/main.py before Task 3 (both GET routes returned this shape).
# Pinned here so response_model can never silently drop or rename a
# field without a test catching it.
ORIGINAL_RESUME_KEYS = {
    "id",
    "candidate_name",
    "technical_stack",
    "employment_history",
    "total_experience_months",
    "total_experience_years",
}


def test_lifespan_is_not_triggered_by_a_bare_testclient():
    # Guards the assumption the rest of this file's `client` fixture
    # relies on: if this ever starts failing, it means a future
    # Starlette/FastAPI upgrade changed TestClient's lifespan behavior,
    # and every other test in this file needs to be revisited for an
    # unwanted real-database dependency.
    bare_client = TestClient(main.app)
    response = bare_client.get("/")
    assert response.status_code == 200


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Resume Parser API is running"}


class TestHealthEndpoint:
    def test_returns_200_without_touching_the_database_or_ollama(self, client):
        # No db override needed to prove this -- if /health touched the
        # database or Ollama, this test's environment (no live Postgres
        # connection wired for a bare route call, no Ollama call
        # permitted anywhere in this suite) would surface it.
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestUploadResume:
    def test_rejects_non_pdf_content_type(self, client):
        response = client.post(
            "/resumes",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_rejects_pdf_content_type_with_non_pdf_bytes(
        self, client, fake_upload_settings
    ):
        # Spoofed content-type header; app.storage's magic-byte check
        # must still catch it.
        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(b"not really a pdf"), "application/pdf")},
        )
        assert response.status_code == 400

    def test_rejects_oversized_upload(self, client, tmp_path, monkeypatch):
        settings = SimpleNamespace(upload_dir=str(tmp_path), max_upload_bytes=5)
        monkeypatch.setattr(storage_module, "get_settings", lambda: settings)

        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert response.status_code == 413

    def test_returns_422_when_extraction_finds_no_text(
        self, client, fake_upload_settings, monkeypatch
    ):
        def _raise(path):
            raise ValueError("Could not extract text from the PDF.")

        monkeypatch.setattr(services, "extract_resume", _raise)

        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert response.status_code == 422

    def test_returns_503_when_ollama_is_unreachable(
        self, client, fake_upload_settings, monkeypatch
    ):
        def _raise(path):
            # The exact exception langchain_ollama's httpx client raises
            # on connection failure, confirmed empirically by pointing
            # ChatOllama at a closed port (see main.py's comment).
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(services, "extract_resume", _raise)

        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert response.status_code == 503

    def test_returns_503_and_does_not_leak_raw_completion_when_llm_output_is_unparseable(
        self, client, fake_upload_settings, monkeypatch
    ):
        # OutputParserException is a ValueError subclass (confirmed
        # empirically -- see langchain_core.exceptions), raised by
        # PydanticOutputParser when the LLM's structured output fails
        # schema validation. This must be distinguished from the plain
        # ValueError extract_resume() raises for "no extractable text"
        # (422, see test above): it is an extraction-service failure,
        # not a malformed-PDF case, and its message contains the raw
        # LLM completion text, which must not reach the client.
        raw_completion = "SENSITIVE_RAW_COMPLETION_MARKER_12345"

        def _raise(path):
            raise OutputParserException(
                f"Failed to parse Foo from completion {raw_completion}. "
                f"Got: 1 validation error"
            )

        monkeypatch.setattr(services, "extract_resume", _raise)

        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 503
        assert raw_completion not in response.text

    def test_successful_upload_returns_200_with_id_and_extraction_data(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())

        response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body.keys() == ORIGINAL_RESUME_KEYS  # D1: id, plus everything already returned
        assert isinstance(body["id"], int)
        assert body["candidate_name"] == "Jane Doe"
        assert body["technical_stack"]["programming_languages"] == ["Python"]
        assert body["total_experience_months"] == 4
        assert body["total_experience_years"] == 0.33

    def test_successful_upload_is_actually_persisted(
        self, client, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())

        post_response = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        resume_id = post_response.json()["id"]

        get_response = client.get(f"/resumes/{resume_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == resume_id
        assert get_response.json()["candidate_name"] == "Jane Doe"


class TestGetResumeById:
    def test_returns_404_for_an_unknown_id(self, client):
        response = client.get("/resumes/999999")
        assert response.status_code == 404

    def test_response_shape_matches_the_original_keys(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())
        resume_id = client.post(
            "/resumes",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        ).json()["id"]

        response = client.get(f"/resumes/{resume_id}")

        assert response.status_code == 200
        assert response.json().keys() == ORIGINAL_RESUME_KEYS


class TestListResumesEndpoint:
    def _upload_n(self, client, monkeypatch, count):
        monkeypatch.setattr(services, "extract_resume", lambda path: _fake_extraction())
        ids = []
        for _ in range(count):
            resp = client.post(
                "/resumes",
                files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
            )
            ids.append(resp.json()["id"])
        return ids

    def test_returns_a_list_with_the_original_keys(
        self, client, fake_upload_settings, monkeypatch
    ):
        self._upload_n(client, monkeypatch, 1)

        response = client.get("/resumes")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0].keys() == ORIGINAL_RESUME_KEYS

    def test_pagination_limit_and_offset(self, client, fake_upload_settings, monkeypatch):
        ids = self._upload_n(client, monkeypatch, 5)

        page1 = client.get("/resumes", params={"limit": 2, "offset": 0}).json()
        page2 = client.get("/resumes", params={"limit": 2, "offset": 2}).json()

        assert [r["id"] for r in page1] == ids[0:2]
        assert [r["id"] for r in page2] == ids[2:4]

    def test_default_pagination_returns_up_to_20(
        self, client, fake_upload_settings, monkeypatch
    ):
        self._upload_n(client, monkeypatch, 3)

        response = client.get("/resumes")

        assert len(response.json()) == 3

    def test_limit_above_the_maximum_is_rejected(self, client):
        response = client.get("/resumes", params={"limit": 1000})
        assert response.status_code == 422

    def test_negative_offset_is_rejected(self, client):
        response = client.get("/resumes", params={"offset": -1})
        assert response.status_code == 422

    def test_empty_table_returns_an_empty_list(self, client):
        response = client.get("/resumes")
        assert response.status_code == 200
        assert response.json() == []
