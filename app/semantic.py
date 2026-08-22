"""
Semantic similarity evidence (Task 8B-1).

Turns two texts plus an EmbeddingProvider into a SemanticEvidence, and
ATTACHES that evidence beside an existing MatchEvidence -- returning a
copy, never mutating the original.


WHY THIS IS A SEPARATE FUNCTION AND NOT PART OF build_match_evidence
--------------------------------------------------------------------------
app.matching.build_match_evidence is documented and tested as pure:
"no LLM calls, no I/O, no database, no network". Producing an embedding
is I/O by definition. Folding embedding into build_match_evidence would
destroy that guarantee for every existing caller -- including callers
that only want structured matching and have no embedding provider at
all -- and would make the structured pipeline fail, hang, or vary with
network conditions.

So the split is deliberate and permanent:

    build_match_evidence(candidate, job)      -> pure, offline, semantic=None
    attach_semantic_evidence(evidence, ...)   -> explicitly I/O, returns a COPY

Everything structured is decided before this module runs, and this
module cannot change any of it. tests/test_semantic.py::
TestStructuredEvidenceIsUnchanged and ::TestScoringIsUnaffected pin
both halves.


RELATIONSHIP TO TASK 8A SCORING
--------------------------------------------------------------------------
app.scoring.score_match reads exactly five dimensions (required_skills,
preferred_skills, experience, education, seniority) and never reads
`semantic`. Attaching semantic evidence therefore cannot change
weights_version, overall_score, or any ScoreComponent -- Task 8A stays
frozen and is not modified by 8B-1. Giving semantic similarity any
weight in the score is Task 8C, and is deliberately NOT done here:
inventing a weight without calibration data is exactly the black-box
outcome this design is avoiding.


WHAT IS DELIBERATELY NOT DECIDED HERE
--------------------------------------------------------------------------
This module takes TEXTS, not a CandidateProfile/JobProfile. Choosing
WHICH text represents a candidate and a job -- the double-counting
question, and the fact that the résumé side currently has no work-
narrative text at all (CandidateEmployment carries only company, role,
and dates) -- is Task 8B-2's decision, and encoding a premature answer
here would quietly foreclose it. The similarity mechanism is complete
and testable without it.
"""
from math import sqrt
from typing import Optional, Sequence

from app.embeddings import EmbeddingProvider
from app.schemas import MatchEvidence, SemanticEvidence

COSINE_METHOD = "cosine"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """
    Pure-Python cosine similarity. Returns None -- never 0.0 -- when
    similarity is UNDEFINED rather than merely low:

      - either vector is empty, or the two differ in dimension (not
        comparable at all)
      - either vector has zero magnitude (a zero vector has no
        direction, so the angle between it and anything is undefined)

    None is the "cannot be computed" signal that becomes UNKNOWN
    upstream; 0.0 is a real, meaningful similarity value ("unrelated")
    and must never be used to mean "no answer".

    numpy is deliberately not used: it is not a dependency of this
    project, and plain Python float arithmetic over a fixed-order list
    is both sufficient and exactly reproducible.

    The result is clamped to [-1.0, 1.0]. Floating-point accumulation
    can land a hair outside that range for identical vectors (e.g.
    1.0000000000000002), and a "similarity" above 1 is not meaningful
    to any downstream consumer.
    """
    if len(a) == 0 or len(a) != len(b):
        return None

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return None

    similarity = dot / (sqrt(norm_a) * sqrt(norm_b))
    return max(-1.0, min(1.0, similarity))


def _unknown(model_id: Optional[str], reason: str) -> SemanticEvidence:
    return SemanticEvidence(
        similarity_score=None,
        method=COSINE_METHOD,
        status="unknown",
        model_id=model_id,
        reason=reason,
    )


def compute_semantic_evidence(
    candidate_text: str,
    job_text: str,
    provider: Optional[EmbeddingProvider],
) -> SemanticEvidence:
    """
    Produces SemanticEvidence for one pair of texts.

    Every failure mode resolves to status="unknown" with
    similarity_score=None and a human-readable reason -- no provider,
    provider reports unavailable, provider raises, either text empty,
    wrong number of vectors returned, or a degenerate/zero vector. None
    of these is "fail", and none produces a similarity of 0.0: a signal
    that could not be measured is not a measurement of dissimilarity.

    This function never raises for provider problems. It is the
    boundary at which embedding I/O stops being able to affect the rest
    of the pipeline.
    """
    model_id = getattr(provider, "model_id", None) if provider is not None else None

    if provider is None:
        return _unknown(None, "No embedding provider was supplied.")

    try:
        if not provider.is_available():
            return _unknown(
                model_id,
                f"Embedding provider ({model_id}) reported itself unavailable.",
            )
    except Exception as exc:  # noqa: BLE001 -- availability must never propagate
        return _unknown(
            model_id,
            f"Embedding provider availability check failed: {type(exc).__name__}: {exc}",
        )

    if not candidate_text or not candidate_text.strip():
        return _unknown(model_id, "Candidate text is empty; nothing to embed.")

    if not job_text or not job_text.strip():
        return _unknown(model_id, "Job text is empty; nothing to embed.")

    try:
        vectors = provider.embed([candidate_text, job_text])
    except Exception as exc:  # noqa: BLE001 -- see module docstring
        return _unknown(
            model_id,
            f"Embedding provider failed: {type(exc).__name__}: {exc}",
        )

    if vectors is None or len(vectors) != 2:
        returned = "None" if vectors is None else str(len(vectors))
        return _unknown(
            model_id,
            f"Embedding provider returned {returned} vectors for 2 texts.",
        )

    similarity = cosine_similarity(vectors[0], vectors[1])

    if similarity is None:
        return _unknown(
            model_id,
            "Similarity is undefined for these embeddings (empty, "
            "mismatched, or zero-magnitude vector).",
        )

    return SemanticEvidence(
        similarity_score=similarity,
        method=COSINE_METHOD,
        status="pass",
        model_id=model_id,
        reason=(
            f"Cosine similarity {similarity:.4f} between candidate and job "
            f"text, embedded with {model_id}."
        ),
    )


def attach_semantic_evidence(
    evidence: MatchEvidence,
    candidate_text: str,
    job_text: str,
    provider: Optional[EmbeddingProvider],
) -> MatchEvidence:
    """
    Returns a COPY of `evidence` carrying semantic evidence. The input
    MatchEvidence is never mutated -- callers holding the structural
    result keep exactly what build_match_evidence gave them, which is
    what makes it safe to attach semantic evidence to evidence that is
    also being scored or persisted elsewhere.

    Only the `semantic` field differs between input and output. Every
    structured dimension (skills, experience, education, seniority,
    hard_constraints, eligibility, unresolved_notes) is carried across
    untouched, so app.scoring.score_match produces an identical score
    either way.
    """
    semantic = compute_semantic_evidence(candidate_text, job_text, provider)
    return evidence.model_copy(update={"semantic": semantic})
