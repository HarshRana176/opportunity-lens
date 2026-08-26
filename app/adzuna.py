"""
Adzuna job-source provider.

A THIN implementation of the app.job_sources.JobSource Protocol backed
by Adzuna's official public job-search API. It adds no new abstraction:
the error vocabulary, the result bound, and the normalized listing
shape all already exist in app.job_sources / app.schemas and are reused
untouched.

This module is the ONLY place in the codebase that talks to a job
provider. Everything above it is provider-agnostic by construction:
swapping this class for app.job_sources.StaticJobSource changes nothing
but where the listings come from.


WHY ADZUNA
--------------------------------------------------------------------------
It publishes an official, documented, keyed HTTP API for job search
(https://developer.adzuna.com/overview, endpoint docs at
https://developer.adzuna.com/docs/search) with a free developer tier --
so using it needs no scraping and violates no site's terms, unlike
LinkedIn/Indeed, which prohibit scraping. It covers 18 countries and
returns structured listings carrying exactly the fields
app.schemas.ExternalJobListing needs, including a stable per-listing id
(for deduplication) and a redirect_url (the source URL shown to the
user).


KNOWN, DOCUMENTED LIMITATION -- DESCRIPTIONS ARE SNIPPETS
--------------------------------------------------------------------------
Adzuna's search response returns a TRUNCATED description, not the full
posting: its own docs state "We currently only provide a snipped of the
job description in the response." This is a real constraint on match
quality, not a detail to paper over -- the downstream
app.job_extractor.extract_job runs against that snippet, so a JD whose
experience/education requirements appear only in the untruncated body
will parse as UNSPECIFIED.

That degrades SAFELY rather than silently: app.requirements produces
is_specified=False / minimum_level=None, app.matching turns those into
UNKNOWN (never FAIL -- see app.matching.match_experience/
match_education), and app.scoring scores UNKNOWN at 0.5 rather than 0.
A snippet-sourced job is therefore scored on the evidence that is
actually present, and a missing requirement never masquerades as a
satisfied one. Callers wanting full text must fetch the posting from
redirect_url themselves; this module never fabricates description text
it did not receive.


CREDENTIALS -- APP_KEY IS THE SECRET, APP_ID IS NOT
--------------------------------------------------------------------------
ADZUNA_APP_ID and ADZUNA_APP_KEY are read from the ENVIRONMENT at
construction, never hardcoded. is_configured() reports their absence
without raising and without a network call. Both are sent as query
parameters because Adzuna's API requires that; they are passed via
httpx's `params` argument, never string-formatted into a URL that could
end up in an exception message or a log, and no JobSourceError raised
here ever includes the request URL or any parameter value.

The two are NOT equally sensitive, and this module treats them
differently on purpose:

  ADZUNA_APP_KEY is the SECRET. It must never appear in an API
  response, in persisted job data, in a log line, in a redirect/apply
  URL, or in committed source. Nothing in this codebase ever writes it
  anywhere but the outbound request's params.

  ADZUNA_APP_ID is a PUBLIC application identifier, and Adzuna itself
  publishes it back to us: every `redirect_url` Adzuna returns carries
  `utm_medium=api&utm_source=<app_id>` as provider attribution
  tracking, exactly as shown in Adzuna's own documented response
  examples. That URL is therefore stored on JobDescription.job_url and
  returned to callers WITH the parameter intact.

  The redirect_url is deliberately NOT stripped or rewritten: it is the
  real application link a user follows to apply, Adzuna generates the
  attribution itself, and removing it would both break the link's
  provenance and work against the attribution Adzuna's terms require
  (see README.md's "Attribution" section). An app_id alone grants no
  API access -- Adzuna requires app_id AND app_key on every request.

tests/test_job_discovery.py::TestCredentialHandling pins both halves of
this: app_key absent everywhere, app_id permitted only inside an
Adzuna-supplied redirect_url.


NO I/O AT CONSTRUCTION
--------------------------------------------------------------------------
Building this provider only reads environment variables and constructs
an httpx.Client (which does not connect). Importing or instantiating it
is therefore safe in an offline test suite, at module import time, or
on a machine with no credentials -- exactly the guarantee
app.ollama_embeddings makes for the same reason.
"""
import os
from typing import Optional

import httpx

from app.job_sources import (
    DEFAULT_SEARCH_LIMIT,
    JobSourceAuthError,
    JobSourceNotConfigured,
    JobSourceRateLimited,
    JobSourceUnavailable,
    clamp_limit,
)
from app.schemas import ExternalJobListing

SOURCE_NAME = "adzuna"

API_ROOT = "https://api.adzuna.com/v1/api/jobs"

DEFAULT_COUNTRY = "gb"

DEFAULT_TIMEOUT_SECONDS = 15.0

# Adzuna's documented per-page maximum. Requesting more in one call is
# not an error upstream, it is simply capped -- we page instead.
MAX_RESULTS_PER_PAGE = 50

_APP_ID_ENV_VAR = "ADZUNA_APP_ID"
_APP_KEY_ENV_VAR = "ADZUNA_APP_KEY"
_COUNTRY_ENV_VAR = "ADZUNA_COUNTRY"
_TIMEOUT_ENV_VAR = "ADZUNA_TIMEOUT_SECONDS"


class AdzunaJobSource:
    """
    Satisfies app.job_sources.JobSource structurally (the Protocol is
    not subclassed -- it never needs to be).

    Configuration precedence: explicit constructor argument, then the
    corresponding environment variable, then the module default. An
    unparseable ADZUNA_TIMEOUT_SECONDS falls back to the default rather
    than raising, since a malformed timeout must not prevent the
    application from starting.
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None,
        country: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[object] = None,
    ):
        self._app_id = app_id if app_id is not None else os.getenv(_APP_ID_ENV_VAR)
        self._app_key = app_key if app_key is not None else os.getenv(_APP_KEY_ENV_VAR)
        self._country = (country or os.getenv(_COUNTRY_ENV_VAR) or DEFAULT_COUNTRY).strip().lower()

        if timeout is not None:
            self._timeout = timeout
        else:
            raw_timeout = os.getenv(_TIMEOUT_ENV_VAR)
            try:
                self._timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
            except ValueError:
                self._timeout = DEFAULT_TIMEOUT_SECONDS

        # Builds an httpx client; performs no network I/O. `client` is
        # an injection point for tests -- there is no other way to
        # exercise this class without live credentials.
        self._client = client if client is not None else httpx.Client(timeout=self._timeout)

    @property
    def source_name(self) -> str:
        return SOURCE_NAME

    @property
    def country(self) -> str:
        return self._country

    def is_configured(self) -> bool:
        """
        True only when BOTH credentials are present and non-blank.
        Never raises, never makes a network call -- callers use this to
        decide whether to attempt discovery at all.
        """
        return bool(self._app_id and self._app_id.strip() and self._app_key and self._app_key.strip())

    def _request_page(self, page: int, params: dict) -> dict:
        """
        One page request. Translates every transport/HTTP failure into
        the app.job_sources error vocabulary. Deliberately never
        includes the URL, the params, or any credential in an exception
        message.
        """
        url = f"{API_ROOT}/{self._country}/search/{page}"

        try:
            response = self._client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise JobSourceUnavailable(
                f"Adzuna request timed out after {self._timeout}s."
            ) from exc
        except httpx.HTTPError as exc:
            # Bare type name only -- an httpx error's str() can embed the
            # full request URL, which carries app_id/app_key.
            raise JobSourceUnavailable(
                f"Adzuna request failed ({type(exc).__name__})."
            ) from exc

        status = response.status_code

        if status in (401, 403):
            raise JobSourceAuthError(
                "Adzuna rejected the configured credentials "
                f"(HTTP {status}). Check ADZUNA_APP_ID / ADZUNA_APP_KEY."
            )
        if status == 429:
            raise JobSourceRateLimited("Adzuna rate limit reached (HTTP 429).")
        if status >= 400:
            raise JobSourceUnavailable(f"Adzuna returned HTTP {status}.")

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 -- any decode failure is "unusable response"
            raise JobSourceUnavailable("Adzuna returned a non-JSON response.") from exc

        if not isinstance(payload, dict):
            raise JobSourceUnavailable("Adzuna returned an unexpected response shape.")

        return payload

    def _to_listing(self, raw: object) -> Optional[ExternalJobListing]:
        """
        Convert ONE raw Adzuna result into an ExternalJobListing, or
        None if it is unusable.

        A malformed/incomplete listing is SKIPPED, never guessed at and
        never allowed to abort the whole page: id, title, description,
        and redirect_url are the minimum needed to identify, display,
        and match a job, so a listing missing any of them is dropped
        with no substitute value invented. Optional fields
        (company/location/created) are simply left None when absent --
        Adzuna genuinely omits them for some listings.
        """
        if not isinstance(raw, dict):
            return None

        external_id = raw.get("id")
        title = raw.get("title")
        description = raw.get("description")
        job_url = raw.get("redirect_url")

        if not (external_id and title and description and job_url):
            return None

        company = None
        raw_company = raw.get("company")
        if isinstance(raw_company, dict):
            company = raw_company.get("display_name") or None

        location = None
        raw_location = raw.get("location")
        if isinstance(raw_location, dict):
            location = raw_location.get("display_name") or None

        created = raw.get("created")

        return ExternalJobListing(
            source=SOURCE_NAME,
            external_job_id=str(external_id),
            title=str(title),
            description=str(description),
            job_url=str(job_url),
            company=str(company) if company else None,
            location=str(location) if location else None,
            posted_at=str(created) if created else None,
        )

    def search(
        self,
        what: str,
        where: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[ExternalJobListing]:
        """
        Search Adzuna for up to `limit` listings (bounded by
        app.job_sources.MAX_SEARCH_LIMIT).

        Pages only as far as needed: one request when limit <=
        MAX_RESULTS_PER_PAGE, otherwise sequential page requests --
        never concurrent -- stopping as soon as `limit` listings are
        collected or the provider returns a short/empty page. Results
        are returned in provider order; ordering/relevance is decided
        downstream by the frozen matching path, not here.
        """
        if not self.is_configured():
            raise JobSourceNotConfigured(
                f"Adzuna is not configured: set {_APP_ID_ENV_VAR} and "
                f"{_APP_KEY_ENV_VAR} in the environment."
            )

        limit = clamp_limit(limit)

        listings: list[ExternalJobListing] = []
        seen_ids: set[str] = set()
        page = 1

        while len(listings) < limit:
            remaining = limit - len(listings)
            params = {
                "app_id": self._app_id,
                "app_key": self._app_key,
                "results_per_page": min(remaining, MAX_RESULTS_PER_PAGE),
                "what": what,
                "content-type": "application/json",
            }
            if where:
                params["where"] = where

            payload = self._request_page(page, params)
            results = payload.get("results")
            if not isinstance(results, list) or not results:
                break

            for raw in results:
                listing = self._to_listing(raw)
                if listing is None:
                    continue
                # Provider-level dedupe: the same ad can surface twice
                # across pages. Persistence-level dedupe still happens
                # separately in app.services.ingest_external_listing.
                if listing.external_job_id in seen_ids:
                    continue
                seen_ids.add(listing.external_job_id)
                listings.append(listing)
                if len(listings) >= limit:
                    break

            if len(results) < params["results_per_page"]:
                # Short page -> provider has nothing more to give.
                break

            page += 1

        return listings
