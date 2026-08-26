"""
Online job-source abstraction.

Defines WHAT a job source must do, the error vocabulary every source
reports failures through, and one fully offline deterministic
implementation (StaticJobSource). Deliberately contains NO HTTP or
provider-specific code: the real Adzuna client lives in app.adzuna,
exactly the way app.embeddings holds the EmbeddingProvider Protocol +
FakeEmbeddingProvider while app.ollama_embeddings holds the real
network-backed one.

This module is the seam that lets the whole discovery -> ingestion ->
matching flow be implemented, tested, and reviewed without an API key
and without a single network call.


WHAT A JOB SOURCE IS AND IS NOT
--------------------------------------------------------------------------
A JobSource DISCOVERS job listings and returns them as
app.schemas.ExternalJobListing. That is its entire responsibility.

It does NOT, and must never:
  - extract structured requirements/skills/responsibilities (that is
    app.job_extractor.extract_job, reached via
    app.services.ingest_external_listing)
  - normalize skills (app.skills)
  - parse experience/education requirements (app.requirements)
  - match, score, rank, or evaluate eligibility (app.matching,
    app.scoring, app.project_relevance)

Adding a second provider therefore means implementing this Protocol and
nothing else -- no matching, scoring, or extraction code changes.


CREDENTIALS
--------------------------------------------------------------------------
A source that needs credentials reads them from the ENVIRONMENT, in its
own constructor, and reports is_configured() == False when they are
absent.

A provider's SECRET (an API key/secret token) must never appear in an
ExternalJobListing, in a JobSourceError message, in a log line, in
persisted job data, or in an API response. The error types below are
deliberately message-only (no request/URL echo) so a raised error can
never carry a secret by accident.

A provider's PUBLIC application identifier is a different matter, and
this contract deliberately does not forbid it: some providers embed
their own identifier in the URLs they return as attribution tracking
(Adzuna appends `utm_source=<app_id>` to every redirect_url -- see
app.adzuna). `job_url` is passed through UNMODIFIED so the user can
follow the real application link, so such an identifier can legitimately
appear there. Sources must not invent, add, or strip such parameters.
"""
from typing import Optional, Protocol, Sequence

from app.schemas import ExternalJobListing

# Bounded by default so a misconfigured caller can never trigger an
# unbounded crawl. Every job fetched costs TWO Ollama calls downstream
# (app.job_extractor runs two chains per job), so these stay small on
# purpose -- see app.services.discover_and_persist_jobs.
DEFAULT_SEARCH_LIMIT = 10

MAX_SEARCH_LIMIT = 50


class JobSourceError(RuntimeError):
    """
    Base for every job-source failure.

    Callers (app.services) catch this ONE type and degrade gracefully:
    a discovery failure never fails the whole /job-matches request, it
    is reported in app.schemas.JobDiscoveryReport instead. Messages
    must stay credential-free.
    """


class JobSourceNotConfigured(JobSourceError):
    """Required credentials/config are absent from the environment."""


class JobSourceAuthError(JobSourceError):
    """The provider rejected the credentials (HTTP 401/403)."""


class JobSourceRateLimited(JobSourceError):
    """The provider rate-limited this client (HTTP 429)."""


class JobSourceUnavailable(JobSourceError):
    """Timeout, connection failure, 5xx, or an unreadable response."""


class JobSource(Protocol):
    """
    The contract app.services.discover_and_persist_jobs depends on.
    Structural (typing.Protocol), not an ABC, so a provider -- or a
    test stub -- satisfies it by shape without importing or
    subclassing anything from here. Same rationale as
    app.embeddings.EmbeddingProvider.

    is_configured() must never raise; it answers "does this source have
    what it needs to run at all" (typically: are its credentials
    present) and must not make a network call to decide.

    search() MAY raise, but only a JobSourceError subclass -- every
    provider is responsible for translating its own transport
    exceptions into that vocabulary, so callers never need to know
    which HTTP library a provider uses.
    """

    @property
    def source_name(self) -> str: ...

    def is_configured(self) -> bool: ...

    def search(
        self,
        what: str,
        where: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[ExternalJobListing]: ...


def clamp_limit(limit: Optional[int]) -> int:
    """
    Shared bound every source applies to a caller-supplied limit.

    None/absent -> DEFAULT_SEARCH_LIMIT. Anything above MAX_SEARCH_LIMIT
    is clamped down rather than rejected: a caller asking for more than
    the cap gets the cap, never an unbounded fetch and never an error.
    Values below 1 clamp to 1 -- a search that fetches nothing is a
    configuration mistake, not a meaningful request.
    """
    if limit is None:
        return DEFAULT_SEARCH_LIMIT
    return max(1, min(int(limit), MAX_SEARCH_LIMIT))


class StaticJobSource:
    """
    A real, deterministic, fully offline JobSource backed by a fixed
    list of listings -- not a mock with canned network behavior.

    Exists so the entire discovery -> ingestion -> dedupe -> matching
    -> ranking flow can be exercised end to end with no API key and no
    network, the same role FakeEmbeddingProvider plays for the semantic
    layer. `configured=False` simulates absent credentials, and
    `raises` simulates a provider failure, without needing a real
    outage.

    Filtering is deliberately trivial (a case-insensitive substring
    test over title/description) and is NOT a matching or ranking
    mechanism -- real relevance is decided downstream by the frozen
    app.matching/app.scoring path, never here.
    """

    def __init__(
        self,
        listings: Sequence[ExternalJobListing] = (),
        source_name: str = "static",
        configured: bool = True,
        raises: Optional[JobSourceError] = None,
    ):
        self._listings = list(listings)
        self._source_name = source_name
        self._configured = configured
        self._raises = raises
        self.search_calls: list[tuple[str, Optional[str], int]] = []

    @property
    def source_name(self) -> str:
        return self._source_name

    def is_configured(self) -> bool:
        return self._configured

    def search(
        self,
        what: str,
        where: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[ExternalJobListing]:
        limit = clamp_limit(limit)
        self.search_calls.append((what, where, limit))

        if self._raises is not None:
            raise self._raises

        needle = (what or "").strip().lower()
        if not needle:
            selected = self._listings
        else:
            selected = [
                listing
                for listing in self._listings
                if needle in listing.title.lower() or needle in listing.description.lower()
            ] or self._listings

        return selected[:limit]
