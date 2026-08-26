"""
Tests for the product-facing workflow: POST /job-matches
    resume PDF -> CandidateProfile (incl. projects) -> ranked list of
    EVERY persisted job, scored via the frozen app.matching/app.scoring
    path, sorted deterministically.

Fully offline: no real Postgres, Ollama, or PDF. Mirrors the fixture
patterns already established in tests/test_api.py, tests/
test_api_jobs.py, and tests/test_match_orchestration.py.
"""
import io
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main
import app.services as services
from app import storage as storage_module
from app.database import Base, get_db
from app.embeddings import FakeEmbeddingProvider
from app.schemas import (
    CandidateProfile,
    CandidateProject,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    ProjectDepthClassification,
    SkillRequirement,
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
def fake_upload_settings(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    settings = SimpleNamespace(upload_dir=tmp_dir, max_upload_bytes=10_000_000)
    monkeypatch.setattr(storage_module, "get_settings", lambda: settings)
    return settings


class _StubDepthChain:
    def __init__(self, depth="substantive"):
        self.depth = depth
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload["project_text"])
        return ProjectDepthClassification(depth=self.depth)


PDF_BYTES = b"%PDF-1.4\nfake pdf body"


def _skill(raw, canonical=None):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy" if canonical else "unresolved", requirement_level="required",
    )


def _candidate_profile(**overrides):
    defaults = dict(
        candidate_name="Jane Doe", seniority=None, current_role=None,
        skills=[CandidateSkill(raw="Python", match_key="python", canonical="python",
                                category=None, resolution="taxonomy")],
        total_experience_months=0, total_experience_years=0.0,
        employment_history=[],
        projects=[
            CandidateProject(
                title="Payments API",
                description="Designed and built a payment processing API and debugged "
                             "a race condition in concurrent updates.",
                technologies=["Python"], role="solo",
                outcome_text="Reduced failed transactions by 30%.",
            )
        ],
        education=None, raw_text="FULL RESUME TEXT", parse_warnings=[],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job_profile(**overrides):
    defaults = dict(
        title="Backend Engineer", seniority=None,
        required_skills=[_skill("Python", canonical="python")],
        preferred_skills=[], experience=ExperienceRequirement(),
        education=EducationRequirement(), responsibilities=["Build APIs."],
        raw_text="FULL JOB TEXT", parse_warnings=[],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _upload_resume(client):
    return client.post(
        "/job-matches", files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
    )


def _create_job(client, monkeypatch, **job_overrides):
    monkeypatch.setattr(services, "extract_job", lambda text: _job_profile(**job_overrides))
    response = client.post("/jobs", json={"job_text": "some job description"})
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------- 1/2/3. end-to-end + multiple jobs + sorting


class TestResumeToRankedJobs:
    def test_resume_produces_ranked_jobs_without_needing_ids(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        weak_job = _create_job(client, monkeypatch, title="Weak Match",
                                required_skills=[_skill("Go", canonical="go")])
        strong_job = _create_job(client, monkeypatch, title="Strong Match",
                                  required_skills=[_skill("Python", canonical="python")])

        response = _upload_resume(client)

        assert response.status_code == 200
        body = response.json()
        assert "candidate_profile_id" in body
        assert len(body["matches"]) == 2
        job_ids = {m["job_id"] for m in body["matches"]}
        assert job_ids == {weak_job["id"], strong_job["id"]}

    def test_multiple_jobs_are_all_evaluated(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        for i in range(5):
            _create_job(client, monkeypatch, title=f"Job {i}")

        response = _upload_resume(client)

        assert len(response.json()["matches"]) == 5

    def test_results_are_sorted_by_score_descending(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        weak_job = _create_job(client, monkeypatch, title="Weak Match",
                                required_skills=[_skill("Go", canonical="go")])
        strong_job = _create_job(client, monkeypatch, title="Strong Match",
                                  required_skills=[_skill("Python", canonical="python")])

        body = _upload_resume(client).json()

        scores = [m["result"]["overall_score"] for m in body["matches"]]
        assert scores == sorted(scores, reverse=True)
        assert body["matches"][0]["job_id"] == strong_job["id"]
        assert body["matches"][-1]["job_id"] == weak_job["id"]


# ---------------------------------------------------------------- 4. deterministic tie-breaking


class TestDeterministicTieBreak:
    def test_tied_scores_broken_by_job_id_ascending(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        # Identical scoring inputs -> identical overall_score; only job_id differs.
        first_job = _create_job(client, monkeypatch, title="Tied A",
                                 required_skills=[_skill("Python", canonical="python")])
        second_job = _create_job(client, monkeypatch, title="Tied B",
                                  required_skills=[_skill("Python", canonical="python")])
        assert first_job["id"] < second_job["id"]

        body = _upload_resume(client).json()

        assert body["matches"][0]["result"]["overall_score"] == body["matches"][1]["result"]["overall_score"]
        assert body["matches"][0]["job_id"] == first_job["id"]
        assert body["matches"][1]["job_id"] == second_job["id"]

    def test_tie_break_is_deterministic_across_repeated_calls(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        job_a = _create_job(client, monkeypatch, title="A", required_skills=[_skill("Python", canonical="python")])
        job_b = _create_job(client, monkeypatch, title="B", required_skills=[_skill("Python", canonical="python")])

        first = _upload_resume(client).json()
        second = _upload_resume(client).json()

        order_first = [m["job_id"] for m in first["matches"] if m["job_id"] in (job_a["id"], job_b["id"])]
        order_second = [m["job_id"] for m in second["matches"] if m["job_id"] in (job_a["id"], job_b["id"])]
        assert order_first == order_second == [job_a["id"], job_b["id"]]


# ---------------------------------------------------------------- 5. empty job corpus


class TestEmptyJobCorpus:
    def test_no_jobs_returns_empty_matches_not_an_error(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())

        response = _upload_resume(client)

        assert response.status_code == 200
        assert response.json()["matches"] == []


# ---------------------------------------------------------------- 6. invalid PDF


class TestInvalidInput:
    def test_rejects_non_pdf_content_type(self, client):
        response = client.post(
            "/job-matches", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_extraction_failure_is_mapped_to_422(self, client, fake_upload_settings, monkeypatch):
        def _raise(path):
            raise ValueError("Could not extract text from the PDF.")

        monkeypatch.setattr(services, "build_candidate_profile", _raise)

        response = _upload_resume(client)

        assert response.status_code == 422


# ---------------------------------------------------------------- 7/8. project weight behavior


class TestProjectEvidenceWeight:
    def test_weight_omitted_preserves_legacy_five_component_result(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        _create_job(client, monkeypatch, title="Job")

        body = _upload_resume(client).json()

        result = body["matches"][0]["result"]
        assert result["weights_version"] == "v1"
        assert len(result["components"]) == 5
        assert result["evidence"]["project_evidence"] is None

    def test_weight_zero_also_preserves_legacy_behavior(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        _create_job(client, monkeypatch, title="Job")

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
            data={"project_evidence_weight": "0"},
        )

        result = response.json()["matches"][0]["result"]
        assert len(result["components"]) == 5

    def test_positive_weight_enables_project_evidence_on_every_job(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        _create_job(client, monkeypatch, title="Job A")
        _create_job(client, monkeypatch, title="Job B")

        # Inject fast, offline stand-ins instead of real Ollama.
        monkeypatch.setattr(
            services, "_PROJECT_EVIDENCE_PROVIDER", FakeEmbeddingProvider()
        )
        import app.project_relevance as project_relevance_module
        monkeypatch.setattr(project_relevance_module, "project_depth_chain", _StubDepthChain("substantive"))

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
            data={"project_evidence_weight": "2.0"},
        )

        body = response.json()
        assert len(body["matches"]) == 2
        for match in body["matches"]:
            result = match["result"]
            assert result["weights_version"] == "v1+project_evidence"
            names = [c["name"] for c in result["components"]]
            assert "project_evidence" in names
            assert result["evidence"]["project_evidence"] is not None


# ---------------------------------------------------------------- 9. projects survive persistence/reload


class TestProjectsPersist:
    def test_candidate_profile_projects_are_persisted_and_reused_for_every_job(
        self, client, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        _create_job(client, monkeypatch, title="Job")

        body = _upload_resume(client).json()
        candidate_id = body["candidate_profile_id"]

        stored = services.get_candidate_profile(db_session, candidate_id)
        assert len(stored.projects) == 1
        assert stored.projects[0]["title"] == "Payments API"


# ---------------------------------------------------------------- 10. eligibility unaffected


class TestEligibilityUnaffected:
    def test_eligibility_identical_with_weight_0_and_positive(
        self, client, fake_upload_settings, monkeypatch
    ):
        # Zero experience, JD requires 24 months -> ineligible regardless of projects.
        monkeypatch.setattr(
            services, "build_candidate_profile",
            lambda path: _candidate_profile(total_experience_months=0),
        )
        _create_job(client, monkeypatch, title="Job",
                    experience=ExperienceRequirement(min_months=24, is_specified=True))
        monkeypatch.setattr(services, "_PROJECT_EVIDENCE_PROVIDER", FakeEmbeddingProvider())
        import app.project_relevance as project_relevance_module
        monkeypatch.setattr(project_relevance_module, "project_depth_chain", _StubDepthChain("substantive"))

        legacy = _upload_resume(client).json()["matches"][0]["result"]

        # Second, independent upload+search with weight>0 against the same job set.
        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
            data={"project_evidence_weight": "5.0"},
        )
        with_projects = response.json()["matches"][0]["result"]

        assert legacy["evidence"]["eligibility"] == with_projects["evidence"]["eligibility"] == "fail"
        assert legacy["evidence"]["experience"]["status"] == with_projects["evidence"]["experience"]["status"] == "fail"


# ---------------------------------------------------------------- 11. JSON serializable / 12. backward compatible


class TestSerializationAndBackwardCompatibility:
    def test_response_is_valid_json_with_expected_shape(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        _create_job(client, monkeypatch, title="Job")

        response = _upload_resume(client)

        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        match = body["matches"][0]
        # Exact key set, still pinned -- online-discovery provenance
        # (source/job_url/company/location) is now part of the contract
        # so a caller can link back to the original posting. These are
        # NULL for a job created from caller-supplied text via POST /jobs.
        assert set(match.keys()) == {
            "job_id", "job_title", "result", "source", "job_url", "company", "location",
        }
        assert match["source"] is None
        assert match["job_url"] is None
        assert set(match["result"].keys()) == {"evidence", "weights_version", "overall_score", "components"}

    def test_existing_match_endpoint_still_works_unchanged(
        self, client, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        job = _create_job(client, monkeypatch, title="Job")
        candidate_resp = client.post(
            "/candidate-profiles", files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        response = client.post(
            "/match", json={"candidate_profile_id": candidate_resp.json()["id"], "job_id": job["id"]},
        )

        assert response.status_code == 200
        assert len(response.json()["components"]) == 5

    def test_existing_resumes_endpoint_still_works_unchanged(self, client, fake_upload_settings, monkeypatch):
        from app.schemas import EmploymentPeriod, ResumeExtraction, TechnicalStack

        monkeypatch.setattr(
            services, "extract_resume",
            lambda path: ResumeExtraction(
                candidate_name="Jane Doe", technical_stack=TechnicalStack(programming_languages=["Python"]),
                employment_history=[EmploymentPeriod(company="Acme", start_date="Jan 2024", end_date="Present")],
                total_experience_months=6, total_experience_years=0.5,
            ),
        )

        response = client.post(
            "/resumes", files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["candidate_name"] == "Jane Doe"
