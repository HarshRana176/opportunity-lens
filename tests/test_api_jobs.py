"""
Characterizes the HTTP layer for Task 4's JD endpoints in app.main:
POST /jobs, GET /jobs, GET /jobs/{job_id}.

Mirrors tests/test_api.py's pattern exactly (separate file so the
résumé API tests stay untouched): an in-memory SQLite database via
StaticPool backs `get_db`, and `app.services.extract_job` is
monkeypatched per test (the route calls
`services.create_job_from_text`, which calls the bare name
`extract_job` inside app.services -- patching that module attribute is
sufficient, same reasoning as `services.extract_resume` in
test_api.py). No real Ollama/Postgres access anywhere in this file.
"""
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main
import app.services as services
from app.database import Base, get_db
from app.schemas import (
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    Seniority,
    SkillRequirement,
)


@pytest.fixture
def db_session():
    # See test_api.py's db_session fixture for why StaticPool is
    # required (routes run in Starlette's threadpool).
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


def _fake_profile(**overrides):
    defaults = dict(
        title="Senior Backend Engineer",
        seniority=Seniority.SENIOR,
        required_skills=[
            SkillRequirement(
                raw="Python",
                match_key="python",
                canonical="python",
                category="programming_languages",
                resolution="taxonomy",
                requirement_level="required",
            )
        ],
        preferred_skills=[],
        experience=ExperienceRequirement(
            min_months=60, max_months=None, raw_text="5+ years", is_specified=True
        ),
        education=EducationRequirement(
            minimum_level=None, fields_of_study=[], raw_text=None, is_required=False
        ),
        responsibilities=["Build things"],
        raw_text="Senior Backend Engineer JD text",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


# The full key set JobResponse exposes -- pinned so response_model can
# never silently drop or rename a field without a test catching it.
JOB_RESPONSE_KEYS = {
    "id",
    "title",
    "seniority",
    "required_skills",
    "preferred_skills",
    "experience",
    "education",
    "responsibilities",
    "raw_text",
    "parse_warnings",
}


class TestCreateJob:
    def test_returns_422_for_empty_job_text(self, client, monkeypatch):
        def _raise(text):
            raise ValueError("Could not extract information: job description text is empty.")

        monkeypatch.setattr(services, "extract_job", _raise)

        response = client.post("/jobs", json={"job_text": ""})
        assert response.status_code == 422

    def test_returns_422_when_job_text_field_is_missing(self, client):
        # FastAPI's own request-body validation (JobCreateRequest
        # requires job_text) -- fires before the route body ever runs.
        response = client.post("/jobs", json={})
        assert response.status_code == 422

    def test_returns_503_when_ollama_is_unreachable(self, client, monkeypatch):
        def _raise(text):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(services, "extract_job", _raise)

        response = client.post("/jobs", json={"job_text": "some JD text"})
        assert response.status_code == 503

    def test_returns_503_and_does_not_leak_raw_completion_when_llm_output_is_unparseable(
        self, client, monkeypatch
    ):
        raw_completion = "SENSITIVE_RAW_COMPLETION_MARKER_67890"

        def _raise(text):
            raise OutputParserException(
                f"Failed to parse RawJobCoreExtraction from completion {raw_completion}."
            )

        monkeypatch.setattr(services, "extract_job", _raise)

        response = client.post("/jobs", json={"job_text": "some JD text"})

        assert response.status_code == 503
        assert raw_completion not in response.text

    def test_successful_creation_returns_200_with_id_and_profile_data(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(services, "extract_job", lambda text: _fake_profile())

        response = client.post("/jobs", json={"job_text": "some JD text"})

        assert response.status_code == 200
        body = response.json()
        assert body.keys() == JOB_RESPONSE_KEYS
        assert isinstance(body["id"], int)
        assert body["title"] == "Senior Backend Engineer"
        assert body["seniority"] == int(Seniority.SENIOR)
        assert body["required_skills"][0]["canonical"] == "python"
        assert body["experience"]["min_months"] == 60

    def test_successful_creation_is_actually_persisted(self, client, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _fake_profile())

        post_response = client.post("/jobs", json={"job_text": "some JD text"})
        job_id = post_response.json()["id"]

        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == job_id
        assert get_response.json()["title"] == "Senior Backend Engineer"

    def test_job_with_no_seniority_persists_null(self, client, monkeypatch):
        monkeypatch.setattr(
            services, "extract_job", lambda text: _fake_profile(seniority=None)
        )

        response = client.post("/jobs", json={"job_text": "some JD text"})

        assert response.status_code == 200
        assert response.json()["seniority"] is None

    def test_job_with_parse_warnings_persists_them(self, client, monkeypatch):
        monkeypatch.setattr(
            services,
            "extract_job",
            lambda text: _fake_profile(parse_warnings=["Could not interpret X"]),
        )

        response = client.post("/jobs", json={"job_text": "some JD text"})

        assert response.status_code == 200
        assert response.json()["parse_warnings"] == ["Could not interpret X"]


class TestGetJobById:
    def test_returns_404_for_an_unknown_id(self, client):
        response = client.get("/jobs/999999")
        assert response.status_code == 404

    def test_response_shape_matches_the_documented_keys(self, client, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _fake_profile())
        job_id = client.post("/jobs", json={"job_text": "some JD text"}).json()["id"]

        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json().keys() == JOB_RESPONSE_KEYS


class TestListJobsEndpoint:
    def _create_n(self, client, monkeypatch, count):
        monkeypatch.setattr(services, "extract_job", lambda text: _fake_profile())
        ids = []
        for _ in range(count):
            resp = client.post("/jobs", json={"job_text": "some JD text"})
            ids.append(resp.json()["id"])
        return ids

    def test_returns_a_list_with_the_documented_keys(self, client, monkeypatch):
        self._create_n(client, monkeypatch, 1)

        response = client.get("/jobs")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0].keys() == JOB_RESPONSE_KEYS

    def test_pagination_limit_and_offset(self, client, monkeypatch):
        ids = self._create_n(client, monkeypatch, 5)

        page1 = client.get("/jobs", params={"limit": 2, "offset": 0}).json()
        page2 = client.get("/jobs", params={"limit": 2, "offset": 2}).json()

        assert [j["id"] for j in page1] == ids[0:2]
        assert [j["id"] for j in page2] == ids[2:4]

    def test_default_pagination_returns_up_to_20(self, client, monkeypatch):
        self._create_n(client, monkeypatch, 3)

        response = client.get("/jobs")

        assert len(response.json()) == 3

    def test_limit_above_the_maximum_is_rejected(self, client):
        response = client.get("/jobs", params={"limit": 1000})
        assert response.status_code == 422

    def test_negative_offset_is_rejected(self, client):
        response = client.get("/jobs", params={"offset": -1})
        assert response.status_code == 422

    def test_empty_table_returns_an_empty_list(self, client):
        response = client.get("/jobs")
        assert response.status_code == 200
        assert response.json() == []


class TestResumeRoutesUnaffectedByTask4:
    """Guards that adding /jobs did not disturb the résumé routes."""

    def test_root_endpoint_still_works(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint_still_works(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_resumes_endpoint_still_reachable(self, client):
        response = client.get("/resumes")
        assert response.status_code == 200
        assert response.json() == []
