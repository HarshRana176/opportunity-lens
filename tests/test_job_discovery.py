"""
Tests for the online job-discovery layer:
    JobSource (protocol + StaticJobSource) / AdzunaJobSource (HTTP client)
    -> app.services.ingest_external_listing / discover_and_persist_jobs
    -> the existing JD pipeline -> matching -> ranking
    -> POST /job-matches with online discovery.

Fully offline: the real Adzuna API is NEVER called here. The HTTP layer
is exercised through an injected fake httpx-shaped client, and the
end-to-end flow through app.job_sources.StaticJobSource -- the same
FakeEmbeddingProvider-style discipline the semantic layer already uses.
"""
import io
import tempfile
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as main
import app.services as services
from app import models
from app import storage as storage_module
from app.adzuna import AdzunaJobSource
from app.database import Base, get_db
from app.job_sources import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    JobSourceAuthError,
    JobSourceNotConfigured,
    JobSourceRateLimited,
    JobSourceUnavailable,
    StaticJobSource,
    clamp_limit,
)
from app.schemas import (
    CandidateProfile,
    CandidateProject,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    ExternalJobListing,
    JobProfile,
    SkillRequirement,
)

# APP_ID is Adzuna's PUBLIC application identifier; APP_KEY is the
# SECRET. They are held to different standards throughout this file --
# see TestCredentialHandling for the exact contract.
PUBLIC_APP_ID = "test-app-id-public"
SECRET_APP_KEY = "test-app-key-do-not-leak"

# Backwards-compatible aliases for the tests written before the
# public/secret distinction was made explicit.
SECRET_ID = PUBLIC_APP_ID
SECRET_KEY = SECRET_APP_KEY

# A realistic Adzuna redirect_url: Adzuna itself appends
# utm_medium=api&utm_source=<app_id> as provider attribution, exactly as
# in its documented response examples and as confirmed against the live
# API. Used to prove the app_id is tolerated there and the URL is not
# stripped.
ADZUNA_STYLE_URL = (
    "https://www.adzuna.in/land/ad/5846723676"
    f"?se=abc123&utm_medium=api&utm_source={PUBLIC_APP_ID}&v=F83D2913"
)

PDF_BYTES = b"%PDF-1.4\nfake pdf body"


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


class _FakeHTTPClient:
    """httpx-shaped stand-in. Records the params it was called with so a
    test can assert credentials were sent correctly WITHOUT the real
    network, and can simulate any status/transport failure."""

    def __init__(self, pages=None, status_code=200, raises=None, json_body=None):
        self.pages = pages if pages is not None else []
        self.status_code = status_code
        self.raises = raises
        self.json_body = json_body
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        if self.raises is not None:
            raise self.raises

        if self.json_body is not None:
            body = self.json_body
        else:
            index = len(self.calls) - 1
            body = self.pages[index] if index < len(self.pages) else {"results": []}

        return _FakeResponse(self.status_code, body)


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _raw_job(job_id="1", title="Backend Engineer", description="Build Python APIs.",
             url="https://example.test/job/1", company="Acme", location="London"):
    return {
        "id": job_id, "title": title, "description": description, "redirect_url": url,
        "company": {"display_name": company}, "location": {"display_name": location},
        "created": "2026-08-20T09:00:00Z",
    }


def _listing(external_job_id="ext-1", title="Backend Engineer",
             description="Build and maintain Python APIs using FastAPI.",
             job_url="https://example.test/job/1", source="static"):
    return ExternalJobListing(
        source=source, external_job_id=external_job_id, title=title,
        description=description, job_url=job_url, company="Acme",
        location="London", posted_at="2026-08-20T09:00:00Z",
    )


def _skill(raw, canonical=None):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy" if canonical else "unresolved", requirement_level="required",
    )


def _candidate_profile(**overrides):
    defaults = dict(
        candidate_name="Jane Doe", seniority=None, current_role="Backend Engineer",
        skills=[CandidateSkill(raw="Python", match_key="python", canonical="python",
                                category=None, resolution="taxonomy")],
        total_experience_months=0, total_experience_years=0.0, employment_history=[],
        projects=[CandidateProject(
            title="Payments API",
            description="Designed and built a payment processing API and debugged a race condition.",
            technologies=["Python"], role="solo", outcome_text="Cut failures 30%.")],
        education=None, raw_text="FULL RESUME TEXT", parse_warnings=[],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job_profile(**overrides):
    defaults = dict(
        title="Backend Engineer", seniority=None,
        required_skills=[_skill("Python", canonical="python")], preferred_skills=[],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        responsibilities=["Build APIs."], raw_text="FULL JOB TEXT", parse_warnings=[],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


# ============================================================ 1. normalized jobs


class TestAdzunaNormalization:
    def test_returns_normalized_listings(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job()]}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        listings = source.search(what="python", limit=5)

        assert len(listings) == 1
        listing = listings[0]
        assert listing.source == "adzuna"
        assert listing.external_job_id == "1"
        assert listing.title == "Backend Engineer"
        assert listing.company == "Acme"
        assert listing.location == "London"
        assert listing.posted_at == "2026-08-20T09:00:00Z"

    def test_job_url_is_preserved(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url="https://adzuna.test/ad/999")]}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        listings = source.search(what="python")

        assert listings[0].job_url == "https://adzuna.test/ad/999"

    def test_missing_optional_fields_become_none_not_invented(self):
        raw = {"id": "7", "title": "Dev", "description": "Work.",
                "redirect_url": "https://example.test/7"}
        http = _FakeHTTPClient(pages=[{"results": [raw]}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        listing = source.search(what="dev")[0]

        assert listing.company is None
        assert listing.location is None
        assert listing.posted_at is None

    def test_malformed_listing_is_skipped_not_fatal(self):
        good = _raw_job(job_id="1")
        malformed = {"id": "2", "title": "No URL"}  # missing description/redirect_url
        http = _FakeHTTPClient(pages=[{"results": [malformed, good]}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        listings = source.search(what="x", limit=10)

        assert [l.external_job_id for l in listings] == ["1"]


# ============================================================ 2/3. auth + failure handling


class TestAdzunaFailureHandling:
    def test_missing_credentials_is_not_configured(self, monkeypatch):
        monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
        monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
        source = AdzunaJobSource(client=_FakeHTTPClient())

        assert source.is_configured() is False
        with pytest.raises(JobSourceNotConfigured):
            source.search(what="python")

    def test_blank_credentials_are_treated_as_missing(self):
        source = AdzunaJobSource(app_id="   ", app_key="", client=_FakeHTTPClient())
        assert source.is_configured() is False

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_rejection(self, status):
        http = _FakeHTTPClient(status_code=status, json_body={})
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        with pytest.raises(JobSourceAuthError):
            source.search(what="python")

    def test_rate_limit(self):
        http = _FakeHTTPClient(status_code=429, json_body={})
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        with pytest.raises(JobSourceRateLimited):
            source.search(what="python")

    def test_timeout(self):
        http = _FakeHTTPClient(raises=httpx.TimeoutException("too slow"))
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        with pytest.raises(JobSourceUnavailable):
            source.search(what="python")

    def test_server_error(self):
        http = _FakeHTTPClient(status_code=503, json_body={})
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        with pytest.raises(JobSourceUnavailable):
            source.search(what="python")

    def test_non_json_response(self):
        http = _FakeHTTPClient(json_body=ValueError("not json"))
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        with pytest.raises(JobSourceUnavailable):
            source.search(what="python")

    def test_empty_results(self):
        http = _FakeHTTPClient(pages=[{"results": []}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        assert source.search(what="python") == []


# ============================================================ 18. credential handling
#
# THE CONTRACT (verified against the live Adzuna API):
#
#   ADZUNA_APP_KEY is the SECRET. It must NEVER appear in an API
#   response, in persisted job data, in a log line, in a redirect/apply
#   URL, or in committed source.
#
#   ADZUNA_APP_ID is a PUBLIC application identifier. Adzuna itself
#   appends it to every redirect_url it returns
#   (utm_medium=api&utm_source=<app_id>) as provider attribution. That
#   is normal, documented provider behaviour, so the app_id MAY appear
#   inside an Adzuna-supplied redirect_url -- and the URL is passed
#   through unmodified so the user can follow the real application link.
#
# An earlier version of this file asserted that NEITHER value could ever
# appear anywhere. That was wrong, and it passed only because the test
# fixture's redirect_url happened to omit the attribution parameter the
# real API always includes -- i.e. it gave false assurance. These tests
# now distinguish the two explicitly.


class TestCredentialHandling:
    # ---- the SECRET: never, anywhere ----

    @pytest.mark.parametrize("status,exc_type", [
        (401, JobSourceAuthError), (429, JobSourceRateLimited), (503, JobSourceUnavailable),
    ])
    def test_error_messages_never_contain_the_secret_key(self, status, exc_type):
        http = _FakeHTTPClient(status_code=status, json_body={})
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        with pytest.raises(exc_type) as excinfo:
            source.search(what="python")

        assert SECRET_APP_KEY not in str(excinfo.value)

    def test_transport_error_message_never_contains_the_secret_key(self):
        # httpx errors can embed the full request URL (which carries the key).
        leaky = httpx.ConnectError(
            f"failed connecting to https://api.adzuna.test/?app_key={SECRET_APP_KEY}"
        )
        http = _FakeHTTPClient(raises=leaky)
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        with pytest.raises(JobSourceUnavailable) as excinfo:
            source.search(what="python")

        assert SECRET_APP_KEY not in str(excinfo.value)

    def test_secret_key_never_reaches_a_listing_even_with_an_attribution_url(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        listing = source.search(what="python")[0]

        assert SECRET_APP_KEY not in listing.model_dump_json()

    def test_secret_key_never_reaches_persisted_job_data(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        listing = _listing(job_url=ADZUNA_STYLE_URL)

        job, _created = services.ingest_external_listing(db_session, listing)

        persisted = " ".join(
            str(v) for v in (job.job_url, job.raw_text, job.title, job.company, job.location)
        )
        assert SECRET_APP_KEY not in persisted

    def test_secret_key_never_reaches_the_api_response(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        monkeypatch.setattr(
            services, "_default_job_source",
            lambda: AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http),
        )

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        assert SECRET_APP_KEY not in response.text

    # ---- the PUBLIC identifier: allowed only where Adzuna put it ----

    def test_app_id_is_allowed_inside_an_adzuna_supplied_redirect_url(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        listing = source.search(what="python")[0]

        # Present, and present specifically as Adzuna's attribution param.
        assert f"utm_source={PUBLIC_APP_ID}" in listing.job_url

    def test_redirect_url_is_passed_through_completely_unmodified(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        listing = source.search(what="python")[0]

        # Byte-for-byte: nothing stripped, rewritten, or re-ordered, so the
        # user can follow the real application link.
        assert listing.job_url == ADZUNA_STYLE_URL

    def test_apply_url_survives_persistence_and_the_api_response(
        self, client, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        monkeypatch.setattr(
            services, "_default_job_source",
            lambda: AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http),
        )

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.json()["matches"][0]["job_url"] == ADZUNA_STYLE_URL

    def test_app_id_does_not_appear_outside_the_redirect_url(self):
        """The app_id is tolerated in Adzuna's own URL -- it must not be
        copied into any other field of the listing."""
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(url=ADZUNA_STYLE_URL)]}])
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        listing = source.search(what="python")[0]

        others = listing.model_copy(update={"job_url": ""}).model_dump_json()
        assert PUBLIC_APP_ID not in others

    # ---- outbound request ----

    def test_credentials_are_sent_as_params_not_embedded_in_our_request_url(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job()]}])
        source = AdzunaJobSource(app_id=PUBLIC_APP_ID, app_key=SECRET_APP_KEY, client=http)

        source.search(what="python")

        url, params = http.calls[0]
        assert PUBLIC_APP_ID not in url and SECRET_APP_KEY not in url
        assert params["app_id"] == PUBLIC_APP_ID and params["app_key"] == SECRET_APP_KEY

    def test_no_credential_is_hardcoded_in_the_source_module(self):
        """Guards against a key ever being committed into app/adzuna.py."""
        import pathlib

        text = pathlib.Path(services.__file__).parent.joinpath("adzuna.py").read_text(encoding="utf-8")
        assert "os.getenv" in text
        assert SECRET_APP_KEY not in text and PUBLIC_APP_ID not in text


# ============================================================ 8. bounded results


class TestBoundedSearch:
    def test_limit_is_respected(self):
        page = {"results": [_raw_job(job_id=str(i)) for i in range(50)]}
        http = _FakeHTTPClient(pages=[page])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        assert len(source.search(what="x", limit=3)) == 3

    def test_limit_is_clamped_to_max(self):
        assert clamp_limit(10_000) == MAX_SEARCH_LIMIT
        assert clamp_limit(None) == DEFAULT_SEARCH_LIMIT
        assert clamp_limit(0) == 1
        assert clamp_limit(-5) == 1

    def test_requested_page_size_never_exceeds_provider_max(self):
        page = {"results": [_raw_job(job_id=str(i)) for i in range(50)]}
        http = _FakeHTTPClient(pages=[page, page])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        source.search(what="x", limit=MAX_SEARCH_LIMIT)

        for _url, params in http.calls:
            assert params["results_per_page"] <= 50

    def test_pagination_stops_on_short_page(self):
        http = _FakeHTTPClient(pages=[{"results": [_raw_job(job_id="1")]}])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        listings = source.search(what="x", limit=MAX_SEARCH_LIMIT)

        assert len(listings) == 1
        assert len(http.calls) == 1  # did not keep paging into the void

    def test_provider_level_duplicate_ids_are_dropped(self):
        dup = {"results": [_raw_job(job_id="1"), _raw_job(job_id="1")]}
        http = _FakeHTTPClient(pages=[dup])
        source = AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http)

        assert len(source.search(what="x", limit=10)) == 1


# ============================================================ 4/5/6. ingestion + dedupe


class TestIngestion:
    def _stub_extraction(self, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())

    def test_listing_is_ingested_through_the_existing_pipeline(self, db_session, monkeypatch):
        self._stub_extraction(monkeypatch)

        job, created = services.ingest_external_listing(db_session, _listing())

        assert created is True
        assert job.id is not None
        assert job.source == "static"
        assert job.external_job_id == "ext-1"
        assert job.job_url == "https://example.test/job/1"
        assert job.company == "Acme"
        assert job.location == "London"
        assert job.posted_at == "2026-08-20T09:00:00Z"

    def test_duplicate_listing_is_reused_not_reingested(self, db_session, monkeypatch):
        calls = []

        def _counting_extract(text):
            calls.append(text)
            return _job_profile()

        monkeypatch.setattr(services, "extract_job", _counting_extract)

        first, created_first = services.ingest_external_listing(db_session, _listing())
        second, created_second = services.ingest_external_listing(db_session, _listing())

        assert created_first is True and created_second is False
        assert first.id == second.id
        assert len(calls) == 1  # the LLM extractor ran only once
        assert db_session.query(models.JobDescription).count() == 1

    def test_different_listings_are_both_ingested(self, db_session, monkeypatch):
        self._stub_extraction(monkeypatch)

        services.ingest_external_listing(db_session, _listing(external_job_id="a"))
        services.ingest_external_listing(db_session, _listing(external_job_id="b"))

        assert db_session.query(models.JobDescription).count() == 2

    def test_same_id_from_different_source_is_not_a_duplicate(self, db_session, monkeypatch):
        self._stub_extraction(monkeypatch)

        services.ingest_external_listing(db_session, _listing(external_job_id="1", source="static"))
        services.ingest_external_listing(db_session, _listing(external_job_id="1", source="other"))

        assert db_session.query(models.JobDescription).count() == 2

    def test_job_text_contains_provider_content_only(self):
        text = services.build_job_text_from_listing(_listing())

        assert "Backend Engineer" in text
        assert "Acme" in text
        assert "Build and maintain Python APIs" in text

    def test_post_jobs_path_leaves_source_columns_null(self, db_session, monkeypatch):
        self._stub_extraction(monkeypatch)

        job = services.create_job_from_text(db_session, "some job text")

        assert job.source is None and job.external_job_id is None and job.job_url is None


# ============================================================ 7/17. discovery orchestration


class TestDiscoverAndPersist:
    def test_reports_ok_and_counts(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        source = StaticJobSource([_listing(external_job_id="a"), _listing(external_job_id="b")])

        report = services.discover_and_persist_jobs(db_session, source, "python", None, 10)

        assert report.status == "ok"
        assert report.fetched == 2
        assert report.newly_ingested == 2
        assert report.reused_existing == 0
        assert report.failed_to_ingest == 0

    def test_unconfigured_source_is_reported_not_raised(self, db_session):
        source = StaticJobSource([_listing()], configured=False)

        report = services.discover_and_persist_jobs(db_session, source, "python", None, 10)

        assert report.status == "not_configured"
        assert report.detail

    def test_provider_failure_is_reported_not_raised(self, db_session):
        source = StaticJobSource([], raises=JobSourceRateLimited("rate limited"))

        report = services.discover_and_persist_jobs(db_session, source, "python", None, 10)

        assert report.status == "failed"
        assert "rate limited" in report.detail

    def test_empty_query_skips_the_search(self, db_session):
        source = StaticJobSource([_listing()])

        report = services.discover_and_persist_jobs(db_session, source, "", None, 10)

        assert report.status == "not_configured"
        assert source.search_calls == []  # provider was never called

    def test_one_bad_job_does_not_destroy_the_search(self, db_session, monkeypatch):
        def _flaky(text):
            if "Broken" in text:
                raise RuntimeError("extraction blew up")
            return _job_profile()

        monkeypatch.setattr(services, "extract_job", _flaky)
        source = StaticJobSource([
            _listing(external_job_id="ok-1"),
            # Title still matches the "engineer" query (so the source
            # returns it) but triggers the flaky extractor above.
            _listing(external_job_id="bad", title="Broken Engineer"),
            _listing(external_job_id="ok-2"),
        ])

        report = services.discover_and_persist_jobs(db_session, source, "engineer", None, 10)

        assert report.status == "ok"
        assert report.fetched == 3
        assert report.newly_ingested == 2
        assert report.failed_to_ingest == 1
        assert db_session.query(models.JobDescription).count() == 2

    def test_repeat_search_reuses_existing(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        source = StaticJobSource([_listing(external_job_id="a")])

        services.discover_and_persist_jobs(db_session, source, "python", None, 10)
        second = services.discover_and_persist_jobs(db_session, source, "python", None, 10)

        assert second.newly_ingested == 0
        assert second.reused_existing == 1

    def test_touched_job_ids_collects_both_created_and_reused(self, db_session, monkeypatch):
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        source = StaticJobSource([_listing(external_job_id="a"), _listing(external_job_id="b")])

        first_ids: list[int] = []
        services.discover_and_persist_jobs(db_session, source, "python", None, 10, touched_job_ids=first_ids)
        assert len(first_ids) == 2

        second_ids: list[int] = []
        services.discover_and_persist_jobs(db_session, source, "python", None, 10, touched_job_ids=second_ids)

        # Same two listings searched again -> reused, but still reported.
        assert second_ids == first_ids

    def test_touched_job_ids_omitted_is_unaffected(self, db_session, monkeypatch):
        """Every existing caller that omits touched_job_ids sees identical behavior."""
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        source = StaticJobSource([_listing(external_job_id="a")])

        report = services.discover_and_persist_jobs(db_session, source, "python", None, 10)

        assert report.status == "ok"
        assert report.newly_ingested == 1


# ============================================================ query derivation


class TestQueryDerivation:
    def test_uses_current_role_when_present(self):
        query = services.derive_job_search_query(_candidate_profile(current_role="Data Engineer"))
        assert query == "Data Engineer"

    def test_falls_back_to_skills(self):
        query = services.derive_job_search_query(_candidate_profile(current_role=None))
        assert "Python" in query

    def test_empty_when_nothing_available(self):
        query = services.derive_job_search_query(
            _candidate_profile(current_role=None, skills=[])
        )
        assert query == ""


# ============================================================ 9-16, 19. end-to-end product flow


class TestEndToEndProductFlow:
    def _setup(self, monkeypatch, listings=None):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        return StaticJobSource(listings if listings is not None else [
            _listing(external_job_id="a", job_url="https://example.test/a"),
            _listing(external_job_id="b", job_url="https://example.test/b"),
        ])

    def _search(self, client, db_session, source, **form):
        # Inject the source by patching the module-level default resolver.
        return services.create_candidate_profile_and_search_jobs(
            db_session, io.BytesIO(PDF_BYTES), "resume.pdf", job_source=source, **form
        )

    def test_resume_to_online_search_to_ranked_matches(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        candidate, results, discovery = self._search(None, db_session, source)

        assert candidate.id is not None
        assert discovery.status == "ok"
        assert discovery.newly_ingested == 2
        assert len(results) == 2  # both discovered jobs were scored

    def test_multiple_jobs_scored_and_ranked_descending(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        _c, results, _d = self._search(None, db_session, source)

        scores = [r.overall_score for _job, r in results]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_tie_break_by_job_id(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        _c, results, _d = self._search(None, db_session, source)

        # Identical job profiles -> identical scores -> job_id ascending.
        assert results[0][1].overall_score == results[1][1].overall_score
        assert results[0][0].id < results[1][0].id

    def test_source_urls_survive_to_the_results(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        _c, results, _d = self._search(None, db_session, source)

        urls = {job.job_url for job, _r in results}
        assert urls == {"https://example.test/a", "https://example.test/b"}

    def test_search_online_false_skips_discovery(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        _c, results, discovery = self._search(None, db_session, source, search_online=False)

        assert discovery.status == "not_requested"
        assert source.search_calls == []
        assert results == []

    def test_caller_supplied_query_overrides_derived(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        self._search(None, db_session, source, what="machine learning", where="Berlin")

        assert source.search_calls[0][0] == "machine learning"
        assert source.search_calls[0][1] == "Berlin"

    def test_limit_is_passed_through_bounded(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        self._search(None, db_session, source, limit=99_999)

        assert source.search_calls[0][2] == MAX_SEARCH_LIMIT

    def test_project_weight_zero_preserves_legacy_five_components(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        source = self._setup(monkeypatch)

        _c, results, _d = self._search(None, db_session, source, project_evidence_weight=0)

        for _job, result in results:
            assert result.weights_version == "v1"
            assert len(result.components) == 5
            assert result.evidence.project_evidence is None

    def test_positive_project_weight_enables_project_evidence(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        from app.embeddings import FakeEmbeddingProvider
        from app.schemas import ProjectDepthClassification

        class _StubDepth:
            def invoke(self, payload):
                return ProjectDepthClassification(depth="substantive")

        source = self._setup(monkeypatch)
        import app.project_relevance as pr
        monkeypatch.setattr(pr, "project_depth_chain", _StubDepth())

        _c, results, _d = self._search(
            None, db_session, source, project_evidence_weight=2.0,
            embedding_provider=FakeEmbeddingProvider(),
        )

        for _job, result in results:
            assert result.weights_version == "v1+project_evidence"
            assert "project_evidence" in [c.name for c in result.components]
            assert result.evidence.project_evidence is not None

    def test_projects_cannot_affect_eligibility(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        from app.embeddings import FakeEmbeddingProvider
        from app.schemas import ProjectDepthClassification

        class _StubDepth:
            def invoke(self, payload):
                return ProjectDepthClassification(depth="substantive")

        monkeypatch.setattr(services, "build_candidate_profile",
                             lambda path: _candidate_profile(total_experience_months=0))
        monkeypatch.setattr(
            services, "extract_job",
            lambda text: _job_profile(
                experience=ExperienceRequirement(min_months=24, is_specified=True)),
        )
        import app.project_relevance as pr
        monkeypatch.setattr(pr, "project_depth_chain", _StubDepth())
        source = StaticJobSource([_listing(external_job_id="a")])

        _c, results, _d = self._search(
            None, db_session, source, project_evidence_weight=5.0,
            embedding_provider=FakeEmbeddingProvider(),
        )

        # A fresher with strong projects still fails the experience gate.
        for _job, result in results:
            assert result.evidence.eligibility == "fail"
            assert result.evidence.experience.status == "fail"

    def test_unconfigured_source_degrades_to_database_only(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        # A pre-existing, database-only job (no online provenance).
        services.create_job_from_text(db_session, "existing job text")
        source = StaticJobSource([_listing()], configured=False)

        _c, results, discovery = self._search(None, db_session, source)

        assert discovery.status == "not_configured"
        assert len(results) == 1  # the database-backed job still matched
        assert results[0][0].source is None

    def test_successful_discovery_excludes_unrelated_persisted_jobs(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        """
        The scoping fix: a job persisted by an EARLIER, unrelated search
        (different candidate, different query) must not be silently
        scored/ranked alongside jobs discovered by THIS search, once
        online discovery actually succeeds.
        """
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        stale_job = services.create_job_from_text(db_session, "an unrelated, previously-persisted job")
        source = StaticJobSource([_listing(external_job_id="fresh")])

        _c, results, discovery = self._search(None, db_session, source)

        assert discovery.status == "ok"
        result_ids = {job.id for job, _r in results}
        assert stale_job.id not in result_ids
        assert len(results) == 1

    def test_successful_discovery_with_no_listings_scores_nothing(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        """
        A successful search that genuinely finds zero online listings
        must report matches=[] rather than silently falling back to the
        full historical job table.
        """
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        services.create_job_from_text(db_session, "an unrelated, previously-persisted job")
        source = StaticJobSource([])  # configured, but nothing to return

        _c, results, discovery = self._search(None, db_session, source)

        assert discovery.status == "ok"
        assert discovery.fetched == 0
        assert results == []


class TestSearchJobsForCandidateScoping:
    """Direct coverage of search_jobs_for_candidate's job_ids parameter,
    independent of the online-discovery orchestration above."""

    def test_job_ids_none_scores_every_persisted_job(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile_from_upload(db_session, io.BytesIO(PDF_BYTES), "resume.pdf")
        services.create_job_from_text(db_session, "job one")
        services.create_job_from_text(db_session, "job two")

        results = services.search_jobs_for_candidate(db_session, candidate.id)

        assert len(results) == 2

    def test_job_ids_scopes_to_exactly_those_jobs(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile_from_upload(db_session, io.BytesIO(PDF_BYTES), "resume.pdf")
        keep = services.create_job_from_text(db_session, "job to keep")
        services.create_job_from_text(db_session, "job to exclude")

        results = services.search_jobs_for_candidate(db_session, candidate.id, job_ids=[keep.id])

        assert [job.id for job, _r in results] == [keep.id]

    def test_job_ids_empty_list_scores_nothing(
        self, db_session, fake_upload_settings, monkeypatch
    ):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        candidate = services.create_candidate_profile_from_upload(db_session, io.BytesIO(PDF_BYTES), "resume.pdf")
        services.create_job_from_text(db_session, "job one")

        results = services.search_jobs_for_candidate(db_session, candidate.id, job_ids=[])

        assert results == []


# ============================================================ API layer


class TestJobMatchesEndpointWithDiscovery:
    def test_discovery_report_is_serialized(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        monkeypatch.setattr(
            services, "_default_job_source",
            lambda: StaticJobSource([_listing(external_job_id="a", job_url="https://example.test/a")]),
        )

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["discovery"]["status"] == "ok"
        assert body["discovery"]["source"] == "static"
        assert body["discovery"]["newly_ingested"] == 1
        assert body["matches"][0]["job_url"] == "https://example.test/a"
        assert body["matches"][0]["source"] == "static"

    def test_missing_credentials_still_returns_200(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(
            services, "_default_job_source",
            lambda: StaticJobSource([_listing()], configured=False),
        )

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["discovery"]["status"] == "not_configured"

    def test_response_never_contains_credentials(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        monkeypatch.setattr(services, "extract_job", lambda text: _job_profile())
        http = _FakeHTTPClient(status_code=401, json_body={})
        monkeypatch.setattr(
            services, "_default_job_source",
            lambda: AdzunaJobSource(app_id=SECRET_ID, app_key=SECRET_KEY, client=http),
        )

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )

        assert response.status_code == 200
        assert response.json()["discovery"]["status"] == "failed"
        assert SECRET_ID not in response.text
        assert SECRET_KEY not in response.text

    def test_search_online_false_via_form(self, client, fake_upload_settings, monkeypatch):
        monkeypatch.setattr(services, "build_candidate_profile", lambda path: _candidate_profile())
        source = StaticJobSource([_listing()])
        monkeypatch.setattr(services, "_default_job_source", lambda: source)

        response = client.post(
            "/job-matches",
            files={"file": ("resume.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
            data={"search_online": "false"},
        )

        assert response.json()["discovery"]["status"] == "not_requested"
        assert source.search_calls == []
