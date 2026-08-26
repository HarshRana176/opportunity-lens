"""
Application-layer orchestration for resume upload, listing, and lookup,
and (Task 4) job-description creation, listing, and lookup.

Business logic (PDF extraction, experience calculation, skill
categorization; JD parsing, requirement interpretation) stays in
app.extractor / app.experience / app.skills / app.job_extractor /
app.requirements; this module only coordinates storage/extraction and
persistence, and owns failure cleanup so routes in app.main can stay
thin HTTP adapters.
"""
from sqlalchemy.orm import Session

from app import models
from app.candidate_extractor import build_candidate_profile
from app.embeddings import CachingEmbeddingProvider
from app.extractor import extract_resume
from app.job_extractor import extract_job
from app.matching import build_match_evidence
from app.ollama_embeddings import OllamaEmbeddingProvider
from app.project_relevance import attach_project_evidence
from app.job_sources import DEFAULT_SEARCH_LIMIT, JobSourceError, clamp_limit
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateProject,
    CandidateSkill,
    EducationBackground,
    EducationRequirement,
    ExperienceRequirement,
    ExternalJobListing,
    JobDiscoveryReport,
    JobProfile,
    MatchResult,
    Seniority,
    SkillRequirement,
)
from app.scoring import DEFAULT_WEIGHTS, score_match
from app.storage import cleanup, save_upload

# Match-orchestration addition. Constructed once, at module import time
# -- app.ollama_embeddings.OllamaEmbeddingProvider's docstring
# guarantees "NO I/O AT CONSTRUCTION" (it only builds an httpx client;
# every real network call happens in is_available()/embed(), on actual
# use), so this is safe to build even when Ollama is not running and
# is never touched unless a caller supplies project_evidence_weight > 0.
# Wrapped in CachingEmbeddingProvider so repeated match requests reuse
# embeddings for the same job/candidate text instead of re-embedding
# on every call.
_PROJECT_EVIDENCE_PROVIDER = CachingEmbeddingProvider(OllamaEmbeddingProvider())

# Online job-discovery addition. Built LAZILY (on first use) rather than
# at import time: constructing it is I/O-free, but it opens an httpx
# client that a deployment never doing online discovery would never
# use. Cached after the first build so repeated searches reuse one
# connection pool. app.adzuna is imported inside the function for the
# same reason -- importing app.services must not require the job-source
# provider to be importable at all.
_JOB_SOURCE = None


def _default_job_source():
    global _JOB_SOURCE
    if _JOB_SOURCE is None:
        from app.adzuna import AdzunaJobSource

        _JOB_SOURCE = AdzunaJobSource()
    return _JOB_SOURCE


def create_resume_from_upload(
    db: Session,
    file_obj,
    original_filename: str | None,
) -> models.Resume:
    """
    Store the uploaded file, run it through the extraction pipeline,
    and persist the result.

    On any failure after the file has been stored -- extraction error
    or a database error -- the stored file is removed, the session is
    rolled back, and the exception is re-raised for the caller to map
    to an HTTP response. A failure during storage itself (bad size,
    bad content) has already cleaned up after itself in app.storage
    and is simply re-raised here.
    """
    final_path, _sanitized_original = save_upload(file_obj, original_filename)

    try:
        result = extract_resume(str(final_path))

        resume = models.Resume(
            candidate_name=result.candidate_name,
            technical_stack=result.technical_stack.model_dump(),
            employment_history=[
                item.model_dump() for item in result.employment_history
            ],
            total_experience_months=result.total_experience_months,
            total_experience_years=result.total_experience_years,
        )

        db.add(resume)
        db.commit()
        db.refresh(resume)

        return resume

    except Exception:
        db.rollback()
        cleanup(final_path)
        raise


def list_resumes(db: Session, limit: int, offset: int) -> list[models.Resume]:
    return (
        db.query(models.Resume)
        .order_by(models.Resume.id)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_resume(db: Session, resume_id: int) -> models.Resume | None:
    return (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id)
        .first()
    )


# ---------------------------------------------------------------------------
# Task 4 additions below (Job Description parsing).
# ---------------------------------------------------------------------------


def create_job_from_text(
    db: Session,
    job_text: str,
    listing: ExternalJobListing | None = None,
) -> models.JobDescription:
    """
    Run job_text through the JD extraction pipeline and persist the
    result. Unlike create_resume_from_upload, there is no file to store
    or clean up (D1: text-only JSON input) -- extract_job() either
    raises (propagated to the caller to map to an HTTP response) or
    returns a complete JobProfile; only the database write itself needs
    a rollback-on-failure guard.

    `listing` (online job-discovery addition) is OPTIONAL provenance
    for a job that came from an external JobSource: when supplied, its
    source/external_job_id/job_url/company/location/posted_at are
    recorded on the row so the job can be deduplicated on repeat
    searches and its source URL can be shown to the user. It does NOT
    affect extraction, matching, or scoring in any way -- extract_job
    sees only `job_text`, exactly as it does for a caller-supplied JD.
    Omitting it (POST /jobs) leaves all six columns NULL and behaves
    exactly as this function always has.
    """
    profile = extract_job(job_text)

    job = models.JobDescription(
        title=profile.title,
        seniority=int(profile.seniority) if profile.seniority is not None else None,
        required_skills=[skill.model_dump() for skill in profile.required_skills],
        preferred_skills=[skill.model_dump() for skill in profile.preferred_skills],
        experience=profile.experience.model_dump(),
        education=profile.education.model_dump(),
        responsibilities=profile.responsibilities,
        raw_text=profile.raw_text,
        parse_warnings=profile.parse_warnings,
        source=listing.source if listing else None,
        external_job_id=listing.external_job_id if listing else None,
        job_url=listing.job_url if listing else None,
        company=listing.company if listing else None,
        location=listing.location if listing else None,
        posted_at=listing.posted_at if listing else None,
    )

    try:
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise


def list_jobs(db: Session, limit: int, offset: int) -> list[models.JobDescription]:
    return (
        db.query(models.JobDescription)
        .order_by(models.JobDescription.id)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_job(db: Session, job_id: int) -> models.JobDescription | None:
    return (
        db.query(models.JobDescription)
        .filter(models.JobDescription.id == job_id)
        .first()
    )


# ---------------------------------------------------------------------------
# Task 5 additions below (Candidate Profile).
#
# Internal service layer only -- no HTTP route exists for these yet
# (approved decision D4). Nothing outside the application consumes a
# CandidateProfile until a matching engine does, so app/main.py is
# deliberately untouched by Task 5.
# ---------------------------------------------------------------------------


def create_candidate_profile(
    db: Session,
    pdf_path: str,
    resume_id: int | None = None,
) -> models.CandidateProfile:
    """
    Build a CandidateProfile from a résumé PDF already on disk and
    persist it.

    Unlike create_resume_from_upload, this does not store or clean up a
    file -- it reads a path the caller already owns (typically one
    produced by an earlier résumé upload), so the file's lifecycle
    stays with whoever created it. build_candidate_profile() either
    raises (propagated for the caller to map) or returns a complete
    profile; only the database write needs a rollback guard.
    """
    profile = build_candidate_profile(pdf_path)

    candidate = models.CandidateProfile(
        resume_id=resume_id,
        candidate_name=profile.candidate_name,
        seniority=int(profile.seniority) if profile.seniority is not None else None,
        current_role=profile.current_role,
        skills=[skill.model_dump() for skill in profile.skills],
        total_experience_months=profile.total_experience_months,
        total_experience_years=profile.total_experience_years,
        employment_history=[item.model_dump() for item in profile.employment_history],
        projects=[item.model_dump() for item in profile.projects],
        education=profile.education.model_dump() if profile.education else None,
        raw_text=profile.raw_text,
        parse_warnings=profile.parse_warnings,
    )

    try:
        db.add(candidate)
        db.commit()
        db.refresh(candidate)
        return candidate
    except Exception:
        db.rollback()
        raise


def get_candidate_profile(
    db: Session, candidate_id: int
) -> models.CandidateProfile | None:
    return (
        db.query(models.CandidateProfile)
        .filter(models.CandidateProfile.id == candidate_id)
        .first()
    )


def create_candidate_profile_from_upload(
    db: Session,
    file_obj,
    original_filename: str | None,
) -> models.CandidateProfile:
    """
    Upload-handling counterpart to create_resume_from_upload, for the
    match-orchestration POST /candidate-profiles route. Mirrors that
    function's structure (save -> build+persist -> cleanup-on-failure)
    but calls build_candidate_profile (Task 5+, includes projects) via
    create_candidate_profile instead of extract_resume, and persists to
    CandidateProfile instead of Resume.

    create_candidate_profile already rolls back its own DB attempt
    internally on a commit failure (see above) -- this wrapper's only
    added responsibility is removing the stored file when ANYTHING
    after save_upload fails, including a build_candidate_profile
    (extraction) failure that never touched the database at all.
    """
    final_path, _sanitized_original = save_upload(file_obj, original_filename)

    try:
        return create_candidate_profile(db, str(final_path))
    except Exception:
        cleanup(final_path)
        raise


# ---------------------------------------------------------------------------
# Match-orchestration additions below. Wires the existing, frozen
# app.matching/app.scoring/app.project_relevance modules together for
# the first time -- none of THEM are modified here. This module only
# reconstructs their inputs from persisted rows and calls them in
# order; app.matching.build_match_evidence, app.matching.
# evaluate_hard_constraints, and app.scoring's five original components
# are untouched by anything below.
# ---------------------------------------------------------------------------


def _job_profile_from_row(row: models.JobDescription) -> JobProfile:
    """
    Rebuilds the real app.schemas.JobProfile from a persisted
    JobDescription row. Safe: every JSON column was originally written
    via .model_dump() on these exact schema types (see
    create_job_from_text above), so model_validate reconstructs each
    nested object exactly as it was before persistence -- no
    reinterpretation, no re-parsing of raw text.
    """
    return JobProfile(
        title=row.title,
        seniority=Seniority(row.seniority) if row.seniority is not None else None,
        required_skills=[
            SkillRequirement.model_validate(s) for s in (row.required_skills or [])
        ],
        preferred_skills=[
            SkillRequirement.model_validate(s) for s in (row.preferred_skills or [])
        ],
        experience=(
            ExperienceRequirement.model_validate(row.experience)
            if row.experience else ExperienceRequirement()
        ),
        education=(
            EducationRequirement.model_validate(row.education)
            if row.education else EducationRequirement()
        ),
        responsibilities=row.responsibilities or [],
        raw_text=row.raw_text,
        parse_warnings=row.parse_warnings or [],
    )


def _candidate_profile_from_row(row: models.CandidateProfile) -> CandidateProfile:
    """
    Rebuilds the real app.schemas.CandidateProfile from a persisted
    CandidateProfile row -- the counterpart to _job_profile_from_row
    above, including the projects column create_candidate_profile now
    persists.
    """
    return CandidateProfile(
        candidate_name=row.candidate_name,
        seniority=Seniority(row.seniority) if row.seniority is not None else None,
        current_role=row.current_role,
        skills=[CandidateSkill.model_validate(s) for s in (row.skills or [])],
        total_experience_months=row.total_experience_months or 0,
        total_experience_years=row.total_experience_years or 0.0,
        employment_history=[
            CandidateEmployment.model_validate(e) for e in (row.employment_history or [])
        ],
        projects=[CandidateProject.model_validate(p) for p in (row.projects or [])],
        education=(
            EducationBackground.model_validate(row.education)
            if row.education else None
        ),
        raw_text=row.raw_text,
        parse_warnings=row.parse_warnings or [],
    )


def match_candidate_to_job(
    db: Session,
    candidate_profile_id: int,
    job_id: int,
    project_evidence_weight: float | None = None,
    embedding_provider=None,
    depth_classifier=None,
) -> MatchResult | None:
    """
    Loads a persisted CandidateProfile and JobDescription, reconstructs
    the real app.schemas types, and produces a MatchResult via the
    frozen app.matching/app.scoring path.

    Returns None if either id does not exist -- mirrors get_resume/
    get_job/get_candidate_profile's not-found contract; app.main maps
    that to a 404, never a 500.

    project_evidence_weight is OPTIONAL and CALLER-SUPPLIED ONLY (see
    app.schemas.MatchRequest):
      - None or 0 (not > 0): app.project_relevance.
        compute_project_evidence is NEVER called -- no LLM depth-
        classification call, no embedding call for any project -- and
        scoring uses app.scoring.DEFAULT_WEIGHTS completely unchanged,
        producing the same 5-component result as the pre-match-
        orchestration path, byte-for-byte.
      - > 0: project evidence IS computed (real Ollama depth
        classification + real embedding similarity, unless a stub
        `embedding_provider`/`depth_classifier` is supplied -- see
        those parameters' docstring below) and attached to the
        evidence BEFORE scoring. Scoring then uses a MatchWeights that
        copies DEFAULT_WEIGHTS' five original values verbatim plus this
        one caller-supplied weight, under a distinct `version` string
        ("v1+project_evidence") so the result is traceable back to "v1
        plus an explicit project-evidence weight" rather than being
        mistaken for pure v1. No preset/default nonzero weight is
        invented or shipped here -- the number always comes from the
        caller.

    embedding_provider/depth_classifier are optional injection points,
    identical in spirit to app.project_relevance.compute_project_
    evidence's own depth_classifier parameter -- production code never
    supplies them (the module-level _PROJECT_EVIDENCE_PROVIDER and the
    real app.llm.project_depth_chain are used), but tests can supply a
    FakeEmbeddingProvider/stub chain to exercise the weight > 0 path
    without any real Ollama/network call.

    app.matching.build_match_evidence and app.matching.
    evaluate_hard_constraints are called exactly as they always are,
    with no wrapper or modification -- eligibility, required/preferred
    skills, experience, education, and seniority are decided before
    project_evidence_weight is ever consulted, and nothing below this
    line can change that decision.
    """
    candidate_row = get_candidate_profile(db, candidate_profile_id)
    job_row = get_job(db, job_id)
    if candidate_row is None or job_row is None:
        return None

    candidate = _candidate_profile_from_row(candidate_row)
    job = _job_profile_from_row(job_row)
    return _score_candidate_against_job(
        candidate, job, project_evidence_weight, embedding_provider, depth_classifier
    )


def _score_candidate_against_job(
    candidate: CandidateProfile,
    job: JobProfile,
    project_evidence_weight: float | None,
    embedding_provider,
    depth_classifier,
) -> MatchResult:
    """
    Shared single-pair scoring step, extracted so match_candidate_to_job
    (the low-level primitive) and search_jobs_for_candidate (the
    product-facing ranked-search workflow) can never let the
    project_evidence_weight contract drift apart between the two call
    sites -- see match_candidate_to_job's docstring for the full,
    authoritative contract this implements. Pure composition of
    already-frozen functions: app.matching.build_match_evidence,
    app.project_relevance.attach_project_evidence (only when weight >
    0), and app.scoring.score_match. Nothing here re-implements or
    re-tunes any of them.
    """
    evidence = build_match_evidence(candidate, job)

    if project_evidence_weight is not None and project_evidence_weight > 0:
        provider = embedding_provider if embedding_provider is not None else _PROJECT_EVIDENCE_PROVIDER
        evidence = attach_project_evidence(evidence, candidate, job, provider, depth_classifier)
        weights = DEFAULT_WEIGHTS.model_copy(update={
            "version": "v1+project_evidence",
            "project_evidence": project_evidence_weight,
        })
    else:
        weights = DEFAULT_WEIGHTS

    return score_match(evidence, weights)


def search_jobs_for_candidate(
    db: Session,
    candidate_profile_id: int,
    project_evidence_weight: float | None = None,
    embedding_provider=None,
    depth_classifier=None,
) -> list[tuple[models.JobDescription, MatchResult]] | None:
    """
    Product-facing ranked job search: matches ONE persisted candidate
    against EVERY persisted job, via the exact same frozen app.matching/
    app.scoring path and the exact same project_evidence_weight
    contract as match_candidate_to_job (both call the shared
    _score_candidate_against_job above, so the two can never disagree
    about what weight=0/omitted or weight>0 means).

    Returns None if candidate_profile_id does not exist (app.main maps
    that to a 404). An empty job corpus is NOT an error -- returns []
    so callers can distinguish "no such candidate" (None) from
    "candidate exists, nothing to match against yet" ([]).

    Results are sorted by overall_score DESCENDING, ties broken by
    job_id ASCENDING -- a fixed, deterministic tie-breaker, never a
    learned or new ranking mechanism.

    At project_evidence_weight 0/omitted, this makes exactly as many
    LLM/embedding calls as match_candidate_to_job does at weight 0:
    none -- _score_candidate_against_job never reaches
    attach_project_evidence in that case, for any job. At weight > 0,
    every match reuses the same _PROJECT_EVIDENCE_PROVIDER (a
    CachingEmbeddingProvider) unless a caller injects their own, so
    repeated job/candidate text is embedded at most once regardless of
    how many jobs are scored -- and this loop is a plain, sequential
    for-loop: no threading, no async fan-out, no uncontrolled
    parallelism.
    """
    candidate_row = get_candidate_profile(db, candidate_profile_id)
    if candidate_row is None:
        return None

    candidate = _candidate_profile_from_row(candidate_row)

    job_rows = (
        db.query(models.JobDescription)
        .order_by(models.JobDescription.id)
        .all()
    )

    results: list[tuple[models.JobDescription, MatchResult]] = []
    for job_row in job_rows:
        job = _job_profile_from_row(job_row)
        result = _score_candidate_against_job(
            candidate, job, project_evidence_weight, embedding_provider, depth_classifier
        )
        results.append((job_row, result))

    results.sort(key=lambda pair: (-pair[1].overall_score, pair[0].id))
    return results


# ---------------------------------------------------------------------------
# Online job-discovery orchestration. Discovers listings via a pluggable
# app.job_sources.JobSource, converts each through the EXISTING, unmodified
# app.job_extractor.extract_job path (via create_job_from_text above), and
# deduplicates against what is already persisted. Contains no extraction,
# matching, scoring, or eligibility logic of its own.
# ---------------------------------------------------------------------------


MAX_QUERY_SKILLS = 3


def derive_job_search_query(candidate: CandidateProfile) -> str:
    """
    Build the online-search keyword string for a candidate.

    DELIBERATELY TRIVIAL AND EXPLAINABLE, and deliberately NOT a second
    matching system: it picks the candidate's current role if the
    résumé states one, else the first few skill names, else "". It
    never scores, ranks, weights, or filters anything -- real relevance
    is decided downstream by the frozen app.matching/app.scoring path
    against the JD's own parsed requirements. Changing this function
    changes only WHICH jobs are fetched for consideration, never how
    any fetched job is judged.

    Returns "" when the résumé yields neither a role nor a skill; the
    caller reports that as a skipped search rather than firing an
    unbounded empty query at the provider.
    """
    if candidate.current_role and candidate.current_role.strip():
        return candidate.current_role.strip()

    skill_names = [s.raw.strip() for s in candidate.skills if s.raw and s.raw.strip()]
    if skill_names:
        return " ".join(skill_names[:MAX_QUERY_SKILLS])

    return ""


def find_job_by_source_id(
    db: Session, source: str, external_job_id: str
) -> models.JobDescription | None:
    """
    The deduplication lookup: (source, external_job_id) is a listing's
    stable identity across repeat searches. A hit means this exact
    listing was already extracted and persisted, so it is REUSED --
    saving both an external API round trip's worth of work and, more
    importantly, the two Ollama calls app.job_extractor would otherwise
    spend re-parsing text it has already parsed.
    """
    return (
        db.query(models.JobDescription)
        .filter(
            models.JobDescription.source == source,
            models.JobDescription.external_job_id == external_job_id,
        )
        .first()
    )


def build_job_text_from_listing(listing: ExternalJobListing) -> str:
    """
    Compose the text handed to the existing app.job_extractor.
    extract_job.

    Includes the title and (when present) company/location as labelled
    lines above the provider's description, because extract_job's first
    chain is asked to copy a title verbatim and a bare snippet often
    does not restate one. Nothing is invented: every line is text the
    provider actually returned, and no requirement, skill, or
    responsibility is synthesized here -- extract_job does all of that,
    unchanged, from this text.

    NOTE: for providers that return only a description SNIPPET (Adzuna
    does -- see app.adzuna's module docstring), this text is
    correspondingly partial, and requirements present only in the full
    posting will parse as unspecified -> UNKNOWN downstream, never as
    satisfied.
    """
    lines = [f"Job Title: {listing.title}"]
    if listing.company:
        lines.append(f"Company: {listing.company}")
    if listing.location:
        lines.append(f"Location: {listing.location}")
    lines.append("")
    lines.append(listing.description)
    return "\n".join(lines)


def ingest_external_listing(
    db: Session, listing: ExternalJobListing
) -> tuple[models.JobDescription, bool]:
    """
    Convert ONE external listing into a persisted JobDescription,
    reusing an already-ingested row when this listing has been seen
    before.

    Returns (job_row, was_newly_created). Raises whatever
    create_job_from_text raises (an extraction/LLM failure) -- the
    per-listing caller, discover_and_persist_jobs, isolates that so one
    bad listing never aborts a whole search.
    """
    existing = find_job_by_source_id(db, listing.source, listing.external_job_id)
    if existing is not None:
        return existing, False

    job = create_job_from_text(db, build_job_text_from_listing(listing), listing=listing)
    return job, True


def discover_and_persist_jobs(
    db: Session,
    source,
    what: str,
    where: str | None = None,
    limit: int | None = None,
) -> JobDiscoveryReport:
    """
    Run one bounded online search and ingest its results.

    Never raises for a provider problem: an unconfigured source, an
    auth rejection, a rate limit, a timeout, or an upstream error all
    resolve to a JobDiscoveryReport with status "not_configured"/
    "failed" and a credential-free detail message, so /job-matches
    still succeeds against whatever jobs are already persisted. This
    mirrors the UNKNOWN-never-FAIL discipline the matching layer
    already applies to missing signals.

    Per-listing failures are isolated the same way: a listing whose
    extraction fails (malformed text, Ollama unavailable mid-run)
    increments failed_to_ingest and the loop continues, so one bad job
    cannot destroy the entire search. Ingestion is sequential -- no
    threads, no async fan-out -- because each new listing costs two
    Ollama calls and uncontrolled concurrency against a local model is
    exactly what this design avoids.
    """
    report = JobDiscoveryReport(
        source=getattr(source, "source_name", None),
        query=what or None,
        location=where,
    )

    if not what or not what.strip():
        report.status = "not_configured"
        report.detail = (
            "No search query could be derived from the résumé (no current role "
            "and no skills), so no online search was attempted."
        )
        return report

    try:
        if not source.is_configured():
            report.status = "not_configured"
            report.detail = (
                "The job source is not configured; set its credentials in the "
                "environment to enable online job discovery."
            )
            return report

        listings = source.search(what=what, where=where, limit=clamp_limit(limit))

    except JobSourceError as exc:
        report.status = "failed"
        report.detail = str(exc)
        return report
    except Exception as exc:  # noqa: BLE001 -- a provider must never break the request
        report.status = "failed"
        report.detail = f"Job source failed unexpectedly ({type(exc).__name__})."
        return report

    report.status = "ok"
    report.fetched = len(listings)

    for listing in listings:
        try:
            _job, created = ingest_external_listing(db, listing)
        except Exception:  # noqa: BLE001 -- one bad listing must not abort the search
            report.failed_to_ingest += 1
            continue
        if created:
            report.newly_ingested += 1
        else:
            report.reused_existing += 1

    return report


def create_candidate_profile_and_search_jobs(
    db: Session,
    file_obj,
    original_filename: str | None,
    project_evidence_weight: float | None = None,
    embedding_provider=None,
    depth_classifier=None,
    job_source=None,
    search_online: bool = True,
    what: str | None = None,
    where: str | None = None,
    limit: int | None = None,
) -> tuple[models.CandidateProfile, list[tuple[models.JobDescription, MatchResult]], JobDiscoveryReport]:
    """
    The product-facing end-to-end workflow for POST /job-matches:
    upload -> persist a CandidateProfile (via
    create_candidate_profile_from_upload, so projects persist and an
    upload/extraction failure is cleaned up exactly as it already is)
    -> DISCOVER jobs online via a JobSource and persist them ->
    rank the candidate against every persisted job (via
    search_jobs_for_candidate).

    Online discovery runs FIRST so newly discovered jobs are part of
    the pool the candidate is then ranked against. It is best-effort by
    construction: discover_and_persist_jobs never raises, so a missing
    API key or a provider outage degrades this to exactly the previous
    database-only behavior, with the reason reported in the returned
    JobDiscoveryReport rather than as a failed request.

    `what` overrides the derived search query when a caller supplies
    one; otherwise derive_job_search_query builds it from the résumé.
    `job_source` defaults to the module-level Adzuna provider and is an
    injection point for tests.

    Pure composition -- no new extraction, persistence, matching, or
    scoring logic lives here.
    """
    candidate = create_candidate_profile_from_upload(db, file_obj, original_filename)

    if search_online:
        source = job_source if job_source is not None else _default_job_source()
        profile = _candidate_profile_from_row(candidate)
        query = what.strip() if what and what.strip() else derive_job_search_query(profile)
        discovery = discover_and_persist_jobs(db, source, query, where, limit)
    else:
        discovery = JobDiscoveryReport(
            status="not_requested",
            detail="Online job discovery was not requested for this search.",
        )

    results = search_jobs_for_candidate(
        db, candidate.id, project_evidence_weight, embedding_provider, depth_classifier
    )
    return candidate, results, discovery
