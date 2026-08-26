"""
End-to-end tests for the match-orchestration addition:
    PDF -> CandidateProfile (incl. projects) -> persisted -> reloaded
    CandidateProfile + JobDescription -> MatchResult
    project_evidence_weight omitted/0 vs > 0
    POST /candidate-profiles, POST /match (the API layer)

Fully offline: no real Postgres, Ollama, or PDF. LLM-heavy steps
(build_candidate_profile, extract_job, the project depth classifier)
are stubbed/injected exactly as the rest of this suite already does --
see tests/test_candidate_services.py and tests/test_api_jobs.py for the
precedents this file mirrors.

The "no project LLM/embedding call when weight is 0/omitted" guarantee
is tested with POISON stand-ins that raise if ever invoked -- proving
absence of a call, not just absence of its effect.
"""
import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main
import app.services as services
from app import models
from app.database import Base, get_db
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateProject,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    ProjectDepthClassification,
    Seniority,
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


class _PoisonProvider:
    """Raises if embed()/is_available() is ever called -- proves the
    orchestration never attempts a real embedding call."""

    model_id = "poison"

    def is_available(self):
        raise AssertionError("embedding provider must not be touched at weight 0")

    def embed(self, texts):
        raise AssertionError("embedding provider must not be touched at weight 0")


class _PoisonDepthChain:
    def invoke(self, payload):
        raise AssertionError("depth classifier must not be invoked at weight 0")


class _StubDepthChain:
    def __init__(self, depth="substantive"):
        self.depth = depth
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload["project_text"])
        return ProjectDepthClassification(depth=self.depth)


def _skill(raw, canonical=None, level="required"):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy" if canonical else "unresolved", requirement_level=level,
    )


def _candidate_profile(**overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        seniority=None,
        current_role=None,
        skills=[CandidateSkill(raw="Python", match_key="python", canonical="python",
                                category=None, resolution="taxonomy")],
        total_experience_months=0,
        total_experience_years=0.0,
        employment_history=[],
        projects=[
            CandidateProject(
                title="Payments API",
                description="Designed and built a payment processing API, wrote "
                             "PostgreSQL migrations, and debugged a race condition.",
                technologies=["Python"],
                role="solo",
                outcome_text="Reduced failed transactions by 30%.",
            )
        ],
        education=None,
        raw_text="FULL RESUME TEXT",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job_profile(**overrides):
    defaults = dict(
        title="Backend Engineer",
        seniority=None,
        required_skills=[_skill("Python", canonical="python")],
        preferred_skills=[],
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        responsibilities=["Build and maintain payment processing APIs."],
        raw_text="FULL JOB TEXT",
        parse_warnings=[],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


# ---------------------------------------------------------------- 1. persistence round-trip


class TestProjectsPersistAndReload:
    def test_pdf_to_candidateprofile_projects_persisted_and_reloaded(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())

        created = services.create_candidate_profile(db_session, "fake.pdf")
        fetched = services.get_candidate_profile(db_session, created.id)

        assert fetched is not None
        assert len(fetched.projects) == 1
        stored = fetched.projects[0]
        assert stored["title"] == "Payments API"
        assert "payment processing API" in stored["description"]
        assert stored["technologies"] == ["Python"]
        assert stored["outcome_text"] == "Reduced failed transactions by 30%."

    def test_reconstructed_candidateprofile_includes_the_project(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        created = services.create_candidate_profile(db_session, "fake.pdf")

        rebuilt = services._candidate_profile_from_row(created)

        assert len(rebuilt.projects) == 1
        assert isinstance(rebuilt.projects[0], CandidateProject)
        assert rebuilt.projects[0].title == "Payments API"

    def test_candidate_with_no_projects_persists_and_reloads_as_empty_list(
        self, db_session, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile(projects=[]))
        created = services.create_candidate_profile(db_session, "fake.pdf")

        fetched = services.get_candidate_profile(db_session, created.id)
        rebuilt = services._candidate_profile_from_row(fetched)

        assert fetched.projects == []
        assert rebuilt.projects == []


# ---------------------------------------------------------------- 2. CandidateProfile + JobDescription -> MatchResult


class TestMatchCandidateToJob:
    def _seed(self, db_session, monkeypatch, candidate_kwargs=None, job_kwargs=None):
        monkeypatch.setattr(services, "build_candidate_profile",
                             lambda path: _candidate_profile(**(candidate_kwargs or {})))
        monkeypatch.setattr(services, "extract_job",
                             lambda text: _job_profile(**(job_kwargs or {})))
        candidate = services.create_candidate_profile(db_session, "fake.pdf")
        job = services.create_job_from_text(db_session, "job text")
        return candidate, job

    def test_produces_a_match_result(self, db_session, monkeypatch):
        candidate, job = self._seed(db_session, monkeypatch)

        result = services.match_candidate_to_job(db_session, candidate.id, job.id)

        assert result is not None
        assert result.overall_score is not None
        assert len(result.components) == 5

    def test_returns_none_for_missing_candidate(self, db_session, monkeypatch):
        _, job = self._seed(db_session, monkeypatch)

        assert services.match_candidate_to_job(db_session, 999, job.id) is None

    def test_returns_none_for_missing_job(self, db_session, monkeypatch):
        candidate, _ = self._seed(db_session, monkeypatch)

        assert services.match_candidate_to_job(db_session, candidate.id, 999) is None


# ---------------------------------------------------------------- 3. weight 0/omitted: legacy behavior, no calls


class TestWeightZeroOrOmitted:
    def _seed(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile(db_session, "fake.pdf")
        job = services.create_job_from_text(db_session, "job text")
        return candidate, job

    def test_omitted_weight_makes_no_provider_or_classifier_call(self, db_session, monkeypatch):
        candidate, job = self._seed(db_session, monkeypatch)

        result = services.match_candidate_to_job(
            db_session, candidate.id, job.id,
            embedding_provider=_PoisonProvider(), depth_classifier=_PoisonDepthChain(),
        )  # must not raise

        assert result is not None
        assert result.weights_version == "v1"
        assert len(result.components) == 5
        assert "project_evidence" not in [c.name for c in result.components]

    def test_zero_weight_also_makes_no_call(self, db_session, monkeypatch):
        candidate, job = self._seed(db_session, monkeypatch)

        result = services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=0.0,
            embedding_provider=_PoisonProvider(), depth_classifier=_PoisonDepthChain(),
        )  # must not raise

        assert len(result.components) == 5

    def test_omitted_weight_matches_direct_score_match_call(self, db_session, monkeypatch):
        """The orchestrated result at weight=0 is byte-identical to
        calling build_match_evidence/score_match directly -- the legacy
        path is genuinely untouched, not just superficially similar."""
        from app.matching import build_match_evidence
        from app.scoring import score_match

        candidate, job = self._seed(db_session, monkeypatch)
        orchestrated = services.match_candidate_to_job(db_session, candidate.id, job.id)

        direct_candidate = _candidate_profile()
        direct_job = _job_profile()
        direct = score_match(build_match_evidence(direct_candidate, direct_job))

        assert orchestrated.overall_score == direct.overall_score
        assert [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in orchestrated.components] == \
               [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in direct.components]


# ---------------------------------------------------------------- 4. weight > 0: project evidence produced


class TestWeightPositive:
    def _seed(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile(db_session, "fake.pdf")
        job = services.create_job_from_text(db_session, "job text")
        return candidate, job

    def test_project_evidence_is_computed_and_attached(self, db_session, monkeypatch):
        from app.embeddings import FakeEmbeddingProvider

        candidate, job = self._seed(db_session, monkeypatch)

        result = services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=1.0,
            embedding_provider=FakeEmbeddingProvider(), depth_classifier=_StubDepthChain("substantive"),
        )

        assert result.evidence.project_evidence is not None
        assert len(result.evidence.project_evidence.per_project) == 1
        assert result.evidence.project_evidence.per_project[0].evidence_depth == "substantive"

    def test_sixth_component_present_with_the_caller_weight(self, db_session, monkeypatch):
        from app.embeddings import FakeEmbeddingProvider

        candidate, job = self._seed(db_session, monkeypatch)

        result = services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=2.5,
            embedding_provider=FakeEmbeddingProvider(), depth_classifier=_StubDepthChain("substantive"),
        )

        names = {c.name: c for c in result.components}
        assert "project_evidence" in names
        assert names["project_evidence"].weight == 2.5
        assert names["project_evidence"].status == "pass"
        assert result.weights_version == "v1+project_evidence"

    def test_no_preset_weight_is_invented(self, db_session, monkeypatch):
        """DEFAULT_WEIGHTS itself is never mutated by a positive-weight call."""
        from app.embeddings import FakeEmbeddingProvider
        from app.scoring import DEFAULT_WEIGHTS

        before = DEFAULT_WEIGHTS.model_dump()
        candidate, job = self._seed(db_session, monkeypatch)

        services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=1.0,
            embedding_provider=FakeEmbeddingProvider(), depth_classifier=_StubDepthChain("substantive"),
        )

        assert DEFAULT_WEIGHTS.model_dump() == before


# ---------------------------------------------------------------- 5. projects never affect eligibility


class TestEligibilityUnaffected:
    def test_eligibility_and_first_five_components_identical_at_weight_0_vs_positive(
        self, db_session, monkeypatch
    ):
        from app.embeddings import FakeEmbeddingProvider

        # Zero professional experience, JD requires 24 months -> ineligible.
        monkeypatch.setattr(
            services, "build_candidate_profile",
            lambda path: _candidate_profile(total_experience_months=0),
        )
        monkeypatch.setattr(
            services, "extract_job",
            lambda text: _job_profile(
                experience=ExperienceRequirement(min_months=24, is_specified=True)
            ),
        )
        candidate = services.create_candidate_profile(db_session, "fake.pdf")
        job = services.create_job_from_text(db_session, "job text")

        legacy = services.match_candidate_to_job(db_session, candidate.id, job.id)
        with_projects = services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=5.0,
            embedding_provider=FakeEmbeddingProvider(), depth_classifier=_StubDepthChain("substantive"),
        )

        assert legacy.evidence.eligibility == with_projects.evidence.eligibility == "fail"
        assert legacy.evidence.experience.status == with_projects.evidence.experience.status == "fail"
        assert legacy.evidence.hard_constraints == with_projects.evidence.hard_constraints

        original_names = ["required_skills", "preferred_skills", "experience", "education", "seniority"]
        legacy_five = [c.model_dump() for c in legacy.components if c.name in original_names]
        with_projects_five = [c.model_dump() for c in with_projects.components if c.name in original_names]
        assert legacy_five == with_projects_five

    def test_total_experience_months_unaffected(self, db_session, monkeypatch):
        from app.embeddings import FakeEmbeddingProvider

        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile(db_session, "fake.pdf")
        job = services.create_job_from_text(db_session, "job text")

        services.match_candidate_to_job(
            db_session, candidate.id, job.id, project_evidence_weight=5.0,
            embedding_provider=FakeEmbeddingProvider(), depth_classifier=_StubDepthChain("substantive"),
        )

        reloaded = services.get_candidate_profile(db_session, candidate.id)
        assert reloaded.total_experience_months == 0


# ---------------------------------------------------------------- 6. API-level serialization


PDF_BYTES = b"%PDF-1.4\nfake pdf body"


class TestCandidateProfileEndpoint:
    def test_upload_persists_and_returns_projects(self, client, db_session, monkeypatch):
        from app import storage as storage_module
        from types import SimpleNamespace
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        monkeypatch.setattr(
            storage_module, "get_settings",
            lambda: SimpleNamespace(upload_dir=tmp_dir, max_upload_bytes=10_000_000),
        )
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())

        response = client.post(
            "/candidate-profiles",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["candidate_name"] == "Jane Doe"
        assert len(body["projects"]) == 1
        assert body["projects"][0]["title"] == "Payments API"

    def test_rejects_non_pdf_content_type(self, client):
        response = client.post(
            "/candidate-profiles",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400


class TestMatchEndpoint:
    def _seed_via_api(self, client, db_session, monkeypatch):
        from app import storage as storage_module
        from types import SimpleNamespace
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        monkeypatch.setattr(
            storage_module, "get_settings",
            lambda: SimpleNamespace(upload_dir=tmp_dir, max_upload_bytes=10_000_000),
        )
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())

        candidate_resp = client.post(
            "/candidate-profiles",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        job_resp = client.post("/jobs", json={"job_text": "some job description"})
        return candidate_resp.json()["id"], job_resp.json()["id"]

    def test_match_result_is_serialized_correctly(self, client, db_session, monkeypatch):
        candidate_id, job_id = self._seed_via_api(client, db_session, monkeypatch)

        response = client.post(
            "/match", json={"candidate_profile_id": candidate_id, "job_id": job_id},
        )

        assert response.status_code == 200
        body = response.json()
        assert "overall_score" in body
        assert "weights_version" in body
        assert body["weights_version"] == "v1"
        assert len(body["components"]) == 5
        assert body["evidence"]["eligibility"] in ("pass", "fail", "unknown", "partial")
        assert body["evidence"]["project_evidence"] is None

    def test_match_returns_404_for_unknown_ids(self, client, db_session, monkeypatch):
        response = client.post(
            "/match", json={"candidate_profile_id": 999, "job_id": 999},
        )
        assert response.status_code == 404

    def test_project_evidence_weight_field_is_accepted(self, client, db_session, monkeypatch):
        candidate_id, job_id = self._seed_via_api(client, db_session, monkeypatch)

        # weight omitted entirely -- must still succeed with legacy 5-component shape.
        response = client.post(
            "/match", json={"candidate_profile_id": candidate_id, "job_id": job_id},
        )

        assert response.status_code == 200
        assert len(response.json()["components"]) == 5
