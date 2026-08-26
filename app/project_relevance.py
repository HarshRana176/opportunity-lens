"""
Project-relevance evidence layer (Phase 4).

Compares a candidate's CandidateProfile.projects against ONE job and
produces app.schemas.ProjectEvidence -- structured evidence, attached
BESIDE an existing MatchEvidence, never folded into it. Structurally
mirrors app.semantic_match (the employment-narrative semantic layer,
Task 8B-2b-i) on purpose: that module already solved "attach richer
narrative evidence without touching eligibility or scoring", and this
is the same problem for a different field.


WHY THIS IS NOT JUST COSINE SIMILARITY
--------------------------------------------------------------------------
A cosine-similarity score alone cannot distinguish "candidate followed
a tutorial that happens to share the JD's vocabulary" from "candidate
built something substantive on the exact required stack" -- both can
embed close to job text. This module therefore reports THREE
independent signals per project, never collapsed into one number here:

    1. technology_overlap  -- deterministic (Python only): does this
                               project actually name any of the JD's
                               required/preferred technologies.
    2. evidence_depth       -- title_only (deterministic: no text to
                               judge) / tutorial_or_basic / substantive
                               (the one genuine reading-comprehension
                               judgment, via app.llm.project_depth_chain,
                               called ONLY when there is non-empty text).
    3. similarity_score     -- the SAME frozen app.semantic.
                               compute_semantic_evidence used for
                               employment narratives, reused verbatim.

Combining these three into a single relevance tier (the way
evaluation/PROJECT_RUBRIC.md's frozen human rubric does, for a
SEPARATE, frozen 80-candidate evaluation corpus) is explicitly NOT done
here -- see the "NOT YET SCORED" section below.


WHY THIS CAN NEVER AFFECT ELIGIBILITY OR overall_score
--------------------------------------------------------------------------
Exactly the same guarantee as app.semantic_match, for the exact same
structural reason: app.scoring.score_match reads only
MatchEvidence.{skills, experience, education, seniority, hard_
constraints} -- it does not import this module and has no reference to
`project_evidence` anywhere in its source. app.matching.build_match_
evidence is untouched (not imported here, not modified by this Phase);
MatchEvidence.project_evidence has a `None` default, so eligibility and
every hard_constraint are decided before this module ever runs and are
never read by it. attach_project_evidence() returns a COPY with only
`.project_evidence` populated -- every other field is carried across
byte-identical, the same contract app.semantic.attach_semantic_evidence
and app.semantic_match.attach_employment_semantic_evidence already make
(see tests/test_semantic_match.py::TestScoringUnaffected /
TestAttachment for the precedent this mirrors).

This module also never reads total_experience_months, employment_
history, education, or seniority -- it reads ONLY candidate.projects
and job.required_skills/preferred_skills/responsibilities. A fresher
with zero professional experience and strong projects therefore gets
exactly the same project evidence a senior candidate with the same
projects would get: nothing here special-cases, boosts, or penalizes
based on experience, because nothing here ever looks at it.


NOT YET SCORED (deliberate, per Phase 4's approved constraint)
--------------------------------------------------------------------------
This module produces EVIDENCE only. Whether/how project_evidence should
ever influence overall_score is explicitly deferred to a future,
separately-approved change to app.scoring -- inventing a weight or a
combination rule here, without calibration against real outcomes, is
exactly the black-box risk app.semantic's docstring already flags for
`.semantic`, and Phase 4 was explicitly scoped to avoid it.


LEAKAGE / CIRCULARITY
--------------------------------------------------------------------------
Nothing in this module imports from, or reads any file under,
`evaluation/`. It has no knowledge of evaluation/PROJECT_RUBRIC.md's
specific wording, no knowledge of any candidate in the frozen 80-
candidate Phase 2 corpus, and no knowledge of any label, hidden
stratum, or blind key. app.llm.project_depth_chain's prompt was written
from general software-engineering judgment, not from reading that
corpus's labels, and must never be tuned against them (see that
chain's docstring in app.llm). Evaluating this module against the
frozen Phase 2 labels is a separate, later, explicitly-approved step
that reads this module's OUTPUT from the outside -- this module itself
never reads evaluation artifacts.
"""
from typing import Optional

from app.embeddings import EmbeddingProvider
from app.llm import project_depth_chain
from app.schemas import (
    CandidateProfile,
    CandidateProject,
    JobProfile,
    MatchEvidence,
    ProjectEvidence,
    ProjectEvidenceDepth,
    ProjectRelevanceSignal,
    ProjectTechnologyOverlap,
)
from app.semantic import COSINE_METHOD, compute_semantic_evidence
from app.semantic_match import MAX_EMBED_CHARS, build_job_text

EVIDENCE_DEPTH_METHOD_VERSION = "v1"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_candidate_project_text(project: CandidateProject) -> tuple[str, bool]:
    """
    The candidate-side text for ONE project: description + outcome_text
    only, joined, capped at MAX_EMBED_CHARS (the same cap
    app.semantic_match uses for employment text, reused for
    consistency).

    Deliberately excludes title (a title alone is not evidence of work
    -- this is exactly the deterministic "title_only" case elsewhere in
    this module) and technologies (a named technology is not evidence
    of what was actually done with it; crediting it here would let
    keyword-stuffing a technologies list substitute for real narrative,
    which is precisely the failure mode a title-only/tutorial-vs-
    substantive distinction exists to catch).

    Truncation semantics deliberately mirror app.semantic_match.
    _join_bullets exactly (part-boundary truncation, first part kept
    whole even if oversized, truncation reported not silent) -- kept as
    a small local copy rather than importing that private helper across
    a module boundary.
    """
    parts = [project.description, project.outcome_text]
    cleaned = [_normalize_text(p) for p in parts if p and p.strip()]

    if not cleaned:
        return "", False

    kept: list[str] = []
    length = 0
    truncated = False

    for part in cleaned:
        projected = length + len(part) + (1 if kept else 0)
        if kept and projected > MAX_EMBED_CHARS:
            truncated = True
            break
        kept.append(part)
        length = projected

    return "\n".join(kept), truncated


def project_technology_overlap(
    project: CandidateProject, job: JobProfile
) -> ProjectTechnologyOverlap:
    """
    Deterministic, Python-only comparison of this project's named
    technologies against the job's required/preferred skills.

    Case-insensitive comparison against each requirement's match_key
    (already lowercased) and canonical (when resolved). Deliberately
    does NOT go through app.skills or app.matching.skills_match: this
    is a relevance-only signal over CandidateProject.technologies
    (plain strings, per that schema's own docstring), never a change to
    SkillEvidence or eligibility.
    """
    tokens = {t.strip().lower() for t in project.technologies if t and t.strip()}

    def _matches(requirement) -> bool:
        return requirement.match_key in tokens or (
            requirement.canonical is not None and requirement.canonical in tokens
        )

    return ProjectTechnologyOverlap(
        matched_required=[r.raw for r in job.required_skills if _matches(r)],
        matched_preferred=[r.raw for r in job.preferred_skills if _matches(r)],
        total_required=len(job.required_skills),
        total_preferred=len(job.preferred_skills),
    )


def classify_evidence_depth(
    project: CandidateProject, classifier=None
) -> ProjectEvidenceDepth:
    """
    Classifies ONE project's own narrative depth, independent of any
    job.

    "title_only" is decided deterministically in Python -- when there
    is no description/outcome text at all (the same text
    build_candidate_project_text would embed), there is nothing for an
    LLM to judge, so none is asked. Only when real text exists is
    project_depth_chain invoked, to distinguish tutorial_or_basic from
    substantive -- the one distinction that genuinely requires reading
    comprehension.

    `classifier` is an optional injection point (defaults to the real
    app.llm.project_depth_chain) so callers -- tests included -- can
    supply a fast, offline stand-in without any Ollama/network
    dependency; production code never needs to pass it.

    Fail-safe like every other LLM-backed step in this codebase
    (_extract_education, _extract_projects, classify_unknown_skill):
    any exception from the classifier is caught and never propagates.
    Unlike those extraction steps (which fail to "nothing"), this is a
    binary classification with no neutral value, so failure resolves to
    the WEAKER conclusion, "tutorial_or_basic" -- this module must never
    claim "substantive" evidence it could not actually verify.
    """
    text, _ = build_candidate_project_text(project)

    if not text:
        return "title_only"

    chain = classifier if classifier is not None else project_depth_chain

    try:
        result = chain.invoke({"project_text": text})
        return result.depth
    except Exception:
        return "tutorial_or_basic"


def compute_project_evidence(
    candidate: CandidateProfile,
    job: JobProfile,
    provider: Optional[EmbeddingProvider],
    depth_classifier=None,
) -> ProjectEvidence:
    """
    Compares every candidate project against the job and aggregates
    into one ProjectEvidence.

    technology_overlap and evidence_depth are computed for EVERY
    project regardless of whether the job states any responsibilities
    to embed against -- unlike app.semantic_match.
    compute_employment_semantic_evidence (which returns immediately,
    with no per_employment entries at all, when the job has no
    responsibilities), because those two signals do not depend on
    job.responsibilities at all: technology_overlap only needs
    job.required_skills/preferred_skills, and evidence_depth only needs
    the project's own text. Only similarity_score/similarity_status
    depend on job_text, and only that piece is UNKNOWN/skipped when it
    is missing.

    UNKNOWN (never 0.0, never "fail") whenever no project could be
    compared semantically -- the candidate has no projects, no project
    has usable text, or the job has no responsibilities/the provider
    could not produce a usable embedding for any project. per_project
    is still populated wherever there is something to report (a
    candidate with projects but no job responsibilities still gets
    technology_overlap/evidence_depth per project), so the evidence
    explains itself even in the UNKNOWN case.
    """
    model_id = getattr(provider, "model_id", None) if provider is not None else None

    if not candidate.projects:
        return ProjectEvidence(
            per_project=[],
            best_similarity_score=None,
            method=COSINE_METHOD,
            model_id=model_id,
            status="unknown",
            reason="The candidate has no projects to compare.",
            evidence_depth_method_version=EVIDENCE_DEPTH_METHOD_VERSION,
        )

    job_text, _job_truncated = build_job_text(job)

    per_project: list[ProjectRelevanceSignal] = []
    scored_scores: list[float] = []

    # Original projects order is preserved deliberately, mirroring
    # app.semantic_match -- never sorted by score.
    for project in candidate.projects:
        technology_overlap = project_technology_overlap(project, job)
        evidence_depth = classify_evidence_depth(project, depth_classifier)

        project_text, _project_truncated = build_candidate_project_text(project)

        if not project_text:
            per_project.append(
                ProjectRelevanceSignal(
                    title=project.title,
                    technology_overlap=technology_overlap,
                    evidence_depth=evidence_depth,
                    similarity_score=None,
                    similarity_status="unknown",
                    skipped_reason="This project has no description/outcome text to compare.",
                )
            )
            continue

        if not job_text:
            per_project.append(
                ProjectRelevanceSignal(
                    title=project.title,
                    technology_overlap=technology_overlap,
                    evidence_depth=evidence_depth,
                    similarity_score=None,
                    similarity_status="unknown",
                    skipped_reason="The job description lists no responsibilities to compare against.",
                )
            )
            continue

        semantic = compute_semantic_evidence(project_text, job_text, provider)

        per_project.append(
            ProjectRelevanceSignal(
                title=project.title,
                technology_overlap=technology_overlap,
                evidence_depth=evidence_depth,
                similarity_score=semantic.similarity_score,
                similarity_status=semantic.status,
                skipped_reason=None if semantic.status == "pass" else semantic.reason,
            )
        )

        if semantic.status == "pass" and semantic.similarity_score is not None:
            scored_scores.append(semantic.similarity_score)

    if not scored_scores:
        return ProjectEvidence(
            per_project=per_project,
            best_similarity_score=None,
            method=COSINE_METHOD,
            model_id=model_id,
            status="unknown",
            reason=(
                "No project could be compared semantically (no usable "
                "text, no job responsibilities, or embeddings were "
                "unavailable)."
            ),
            evidence_depth_method_version=EVIDENCE_DEPTH_METHOD_VERSION,
        )

    best = max(scored_scores)
    return ProjectEvidence(
        per_project=per_project,
        best_similarity_score=best,
        method=COSINE_METHOD,
        model_id=model_id,
        status="pass",
        reason=(
            f"Best match {best:.4f} across {len(scored_scores)} of "
            f"{len(per_project)} project(s), embedded with {model_id}."
        ),
        evidence_depth_method_version=EVIDENCE_DEPTH_METHOD_VERSION,
    )


def attach_project_evidence(
    evidence: MatchEvidence,
    candidate: CandidateProfile,
    job: JobProfile,
    provider: Optional[EmbeddingProvider],
    depth_classifier=None,
) -> MatchEvidence:
    """
    Returns a COPY of `evidence` carrying project evidence. The input
    MatchEvidence is never mutated, and only `.project_evidence`
    differs between input and output -- every structured dimension
    (skills, experience, education, seniority, hard_constraints,
    eligibility, semantic, unresolved_notes) is carried across
    untouched, so app.scoring.score_match produces an identical score
    either way. Mirrors app.semantic_match.
    attach_employment_semantic_evidence exactly.
    """
    project_evidence = compute_project_evidence(candidate, job, provider, depth_classifier)
    return evidence.model_copy(update={"project_evidence": project_evidence})
