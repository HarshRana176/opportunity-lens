"""
Semantic orchestration across employment records (Task 8B-2b-i).

Compares each of a candidate's employment positions against ONE job's
responsibilities, then aggregates the per-position similarities into a
single SemanticEvidence.

    CandidateProfile.employment_history[*].responsibilities
                              |
                              v
              (one call per position, via the FROZEN
               app.semantic.compute_semantic_evidence)
                              ^
                              |
                 JobProfile.responsibilities

app.semantic and app.embeddings are FROZEN interfaces: this module
CALLS compute_semantic_evidence once per usable position and never
reimplements cosine, provider handling, or the UNKNOWN contract. Every
failure mode -- no provider, provider unavailable, provider raises,
empty text, zero-magnitude vector -- is already handled there and is
simply carried through here.


WHY ONE CALL PER POSITION IS NOT WASTEFUL
--------------------------------------------------------------------------
compute_semantic_evidence embeds both texts on every call, so N
positions means the job text is submitted N times. Wrapping the
provider in the frozen CachingEmbeddingProvider collapses that to ONE
actual embedding of the job text (its cache is content-addressed and
model-scoped), which is exactly the case that class was written for.
Callers are strongly encouraged to pass a cached provider; correctness
does not depend on it, only efficiency.


AGGREGATION
--------------------------------------------------------------------------
Headline `similarity_score` is the MAX across usable positions --
"has this candidate actually done this kind of work, in any role?".
Max is deliberately not a mean: averaging punishes a long career, so a
candidate whose most recent role is a perfect match would score worse
merely for having held unrelated jobs earlier.

`weighted_mean_score` is recorded as SECONDARY, informational evidence
(duration-weighted, so a four-year role counts for more than a
two-month one). It is not the headline and nothing consumes it for
scoring today -- Task 8A does not read semantic evidence at all. It is
deliberately NOT tuned here; choosing how (or whether) to blend max and
weighted mean is Task 8C's calibration work, which needs labelled data
this task does not have.

Both aggregates are order-independent: max is order-independent by
definition, and the weighted mean is a sum of (score * weight) divided
by a sum of weights, neither of which depends on iteration order.
`per_employment` preserves the candidate's ORIGINAL employment_history
order -- never sorted by score, so the evidence reads in résumé order
and ties never reshuffle between runs.

Positions with no responsibility bullets are SKIPPED, not scored 0.0,
and are still listed in per_employment with a skipped_reason. Scoring
them zero would treat "this résumé didn't bullet that job" as "this
job was irrelevant", which is exactly the missing-signal-as-negative-
signal error the UNKNOWN contract exists to prevent.
"""
from typing import Optional

from app.embeddings import EmbeddingProvider
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    EmploymentSimilarity,
    JobProfile,
    MatchEvidence,
    SemanticEvidence,
)
from app.semantic import COSINE_METHOD, compute_semantic_evidence

MAX_EMBED_CHARS = 8000

AGGREGATION_MAX = "max"


def _normalize_bullet(bullet: str) -> str:
    """Collapse internal whitespace runs so formatting noise from the
    PDF never changes the embedded text (or the cache key)."""
    return " ".join(bullet.split())


def _join_bullets(bullets: list[str]) -> tuple[str, bool]:
    """
    Join responsibility bullets into one embeddable text, capped at
    MAX_EMBED_CHARS.

    Truncation is at a BULLET BOUNDARY and is reported (the returned
    bool), never silent: a half-sentence would be embedded as though it
    were the candidate's actual claim. If even the first bullet exceeds
    the cap it is kept whole rather than cut mid-word -- one oversized
    bullet is still a real, complete statement, and the cap exists to
    bound request size, not to enforce a hard character budget.
    """
    cleaned = [_normalize_bullet(b) for b in bullets]
    cleaned = [b for b in cleaned if b]

    if not cleaned:
        return "", False

    kept: list[str] = []
    length = 0
    truncated = False

    for bullet in cleaned:
        # +1 for the newline separator, except before the first bullet.
        projected = length + len(bullet) + (1 if kept else 0)
        if kept and projected > MAX_EMBED_CHARS:
            truncated = True
            break
        kept.append(bullet)
        length = projected

    return "\n".join(kept), truncated


def build_candidate_employment_text(employment: CandidateEmployment) -> tuple[str, bool]:
    """
    The candidate-side semantic text for ONE position.

    Reads employment.responsibilities and NOTHING ELSE. Company, role,
    dates, duration, seniority, and is_current are deliberately absent:
    every one of them is already owned by a structured matching
    dimension (skills/experience/education/seniority), and including
    them here would double-count that signal inside the semantic score.
    """
    return _join_bullets(employment.responsibilities)


def build_job_text(job: JobProfile) -> tuple[str, bool]:
    """
    The job-side semantic text.

    Reads job.responsibilities and NOTHING ELSE -- not title (that is
    the seniority dimension), not required/preferred skills (the skills
    dimension), not the experience or education requirements (their own
    dimensions), and not raw_text (which re-encodes all four).
    """
    return _join_bullets(job.responsibilities)


def _unknown_evidence(
    model_id: Optional[str],
    reason: str,
    per_employment: Optional[list[EmploymentSimilarity]] = None,
) -> SemanticEvidence:
    return SemanticEvidence(
        similarity_score=None,
        method=COSINE_METHOD,
        status="unknown",
        model_id=model_id,
        reason=reason,
        per_employment=per_employment or [],
        aggregation=None,
        weighted_mean_score=None,
    )


def _weighted_mean(
    scored: list[tuple[EmploymentSimilarity, Optional[int]]],
) -> Optional[float]:
    """
    Duration-weighted mean over the positions that were actually
    scored.

    Deterministic handling of missing/zero duration: a position whose
    duration_months is None or <= 0 (unparseable résumé dates, or a
    role shorter than the inclusive-month granularity) still
    participates, with weight 1 -- dropping it would silently discard a
    real comparison, and weighting it 0 would do the same thing less
    visibly. When NO position carries usable duration data this
    degenerates to a plain arithmetic mean, which is the correct
    fallback rather than an error.

    Returns None only when nothing was scored at all.
    """
    if not scored:
        return None

    total_weight = 0.0
    total = 0.0
    for similarity, duration_months in scored:
        weight = float(duration_months) if duration_months and duration_months > 0 else 1.0
        total += similarity.similarity_score * weight
        total_weight += weight

    if total_weight <= 0:
        return None

    return total / total_weight


def compute_employment_semantic_evidence(
    candidate: CandidateProfile,
    job: JobProfile,
    provider: Optional[EmbeddingProvider],
) -> SemanticEvidence:
    """
    Compare every candidate position against the job's
    responsibilities and aggregate into one SemanticEvidence.

    UNKNOWN (never 0.0, never "fail") whenever the comparison could not
    be made: the job states no responsibilities, no position has any
    responsibility bullets, or the provider could not produce a usable
    embedding for any position. per_employment is still populated in
    those cases wherever there is something to report, so the evidence
    explains itself.
    """
    model_id = getattr(provider, "model_id", None) if provider is not None else None

    job_text, job_truncated = build_job_text(job)

    if not job_text:
        return _unknown_evidence(
            model_id,
            "The job description lists no responsibilities to compare against.",
        )

    per_employment: list[EmploymentSimilarity] = []
    # (evidence entry, duration) for positions that produced a real score.
    scored: list[tuple[EmploymentSimilarity, Optional[int]]] = []

    # Original employment_history order is preserved deliberately -- see
    # this module's docstring. Never sorted.
    for employment in candidate.employment_history:
        candidate_text, candidate_truncated = build_candidate_employment_text(employment)

        if not candidate_text:
            per_employment.append(
                EmploymentSimilarity(
                    company=employment.company,
                    role=employment.role,
                    similarity_score=None,
                    status="unknown",
                    skipped_reason="This position has no responsibility bullets to compare.",
                )
            )
            continue

        evidence = compute_semantic_evidence(candidate_text, job_text, provider)

        entry = EmploymentSimilarity(
            company=employment.company,
            role=employment.role,
            similarity_score=evidence.similarity_score,
            status=evidence.status,
            skipped_reason=None if evidence.status == "pass" else evidence.reason,
            truncated=candidate_truncated or job_truncated,
        )
        per_employment.append(entry)

        if evidence.status == "pass" and evidence.similarity_score is not None:
            scored.append((entry, employment.duration_months))

    if not per_employment:
        return _unknown_evidence(
            model_id,
            "The candidate has no employment history to compare.",
        )

    if not scored:
        return _unknown_evidence(
            model_id,
            (
                "No employment position could be compared semantically "
                "(no responsibility bullets, or embeddings were unavailable)."
            ),
            per_employment,
        )

    # max() over scores, not over entries: order-independent, and never
    # dependent on how ties happen to be ordered.
    headline = max(entry.similarity_score for entry, _ in scored)
    weighted_mean = _weighted_mean(scored)

    return SemanticEvidence(
        similarity_score=headline,
        method=COSINE_METHOD,
        status="pass",
        model_id=model_id,
        reason=(
            f"Best match {headline:.4f} across {len(scored)} of "
            f"{len(per_employment)} position(s), embedded with {model_id}."
        ),
        per_employment=per_employment,
        aggregation=AGGREGATION_MAX,
        weighted_mean_score=weighted_mean,
    )


def attach_employment_semantic_evidence(
    evidence: MatchEvidence,
    candidate: CandidateProfile,
    job: JobProfile,
    provider: Optional[EmbeddingProvider],
) -> MatchEvidence:
    """
    Returns a COPY of `evidence` carrying aggregated semantic evidence.
    The input MatchEvidence is never mutated, and only the `semantic`
    field differs between input and output -- every structured
    dimension is carried across untouched, so app.scoring.score_match
    produces an identical score either way.

    This is the profile-level counterpart to the frozen
    app.semantic.attach_semantic_evidence (which takes two raw texts);
    that function is unchanged and still available for single-pair use.
    """
    semantic = compute_employment_semantic_evidence(candidate, job, provider)
    return evidence.model_copy(update={"semantic": semantic})
