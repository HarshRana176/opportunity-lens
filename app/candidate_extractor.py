"""
Candidate profile construction -- the résumé-side counterpart to
app.job_extractor, producing the canonical CandidateProfile a future
matching engine compares against a JobProfile.

This is a SEPARATE extraction path from app.extractor.extract_resume(),
deliberately (approved Task 5 decision D1-A). It reuses the same shared
parts:

    app.pdf         -- PDF -> raw text (external boundary: PyMuPDF)
    app.llm         -- the SAME résumé extraction chain (external: Ollama)
    app.skills      -- shared normalization + batch enrichment
    app.experience  -- shared deterministic date/duration primitives
    app.requirements-- shared deterministic seniority derivation
    app.schemas     -- shared contracts

...but it consumes RawResumeExtraction.skills DIRECTLY rather than
ResumeExtraction.technical_stack. That distinction is the entire reason
this module exists: build_technical_stack() drops any skill the
per-skill LLM classifier labels "exclude", which silently destroys real
technologies (empirically, "Kafka" among them), and those skills are
unrecoverable from ResumeExtraction afterwards. A candidate profile
must never lose a skill a job might require, so this path normalizes
the raw extracted skills and uses the never-delete batch enrichment
(app.skills.enrich_unresolved_skills) instead.

Deliberately does NOT import from app.extractor or app.job_extractor:
the three pipelines share vocabulary and primitives, never orchestration.
"""
from app.experience import calculate_period_interval, calculate_total_experience
from app.llm import extraction_chain
from app.pdf import extract_text_from_pdf
from app.requirements import derive_seniority
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateSkill,
    EmploymentPeriod,
)
from app.skills import enrich_unresolved_skills, normalize_skill, skill_identity


def _normalize_candidate_skills(
    raw_skills: list[str],
) -> tuple[list[CandidateSkill], list[str]]:
    """
    Turn the raw extracted skill strings into deduplicated
    CandidateSkill entries.

    Uses normalize_skill WITHOUT extra_excluded: only the curated
    résumé-side EXCLUDED_TECHNOLOGIES applies here. JD_EXCLUDED_TERMS is
    deliberately NOT applied (it is JD-only by approved design), so a
    résumé listing e.g. "Agile" yields a retained unresolved skill
    rather than a dropped one -- consistent with never silently losing
    candidate information.

    Unknown skills go through the batched, never-delete enrichment path
    (enrich_unresolved_skills), never through classify_unknown_skill:
    the latter deletes on an "exclude" verdict, which is precisely the
    data loss this module exists to avoid.
    """
    normalized = []

    for raw_skill in raw_skills:
        skill = normalize_skill(raw_skill)
        if skill is None:
            # Blank, or an explicitly curated non-technology term. This
            # is the ONLY place a candidate skill is dropped, and it is
            # never the LLM's decision.
            continue
        normalized.append(skill)

    enriched, warnings = enrich_unresolved_skills(normalized)

    # Dedupe via the shared identity rule; first occurrence wins so the
    # résumé's own casing/spelling is what survives in `raw`.
    deduped: dict[str, CandidateSkill] = {}
    for skill in enriched:
        identity = skill_identity(skill)
        if identity not in deduped:
            deduped[identity] = CandidateSkill(**skill.model_dump())

    return list(deduped.values()), warnings


def _build_employment_history(
    employment_history: list[EmploymentPeriod],
) -> tuple[list[CandidateEmployment], list[str]]:
    """
    Normalize each position into a CandidateEmployment, keeping the
    verbatim company/role/date strings alongside the derived month
    indices, inclusive duration, per-role seniority, and is_current
    flag. A position whose dates cannot be parsed keeps its verbatim
    strings with None derived values -- it is never dropped.
    """
    entries = []
    warnings = []

    for period in employment_history:
        interval = calculate_period_interval(period)

        if interval["duration_months"] is None:
            warnings.append(
                f"Could not interpret employment dates for "
                f"{period.company!r}: {period.start_date!r} to {period.end_date!r}"
            )

        entries.append(
            CandidateEmployment(
                company=period.company,
                role=period.role,
                start_date=period.start_date,
                end_date=period.end_date,
                start_month_index=interval["start_month_index"],
                end_month_index=interval["end_month_index"],
                duration_months=interval["duration_months"],
                seniority=derive_seniority(period.role),
                is_current=interval["is_current"],
            )
        )

    return entries, warnings


def _select_latest_position(
    employment_history: list[CandidateEmployment],
) -> CandidateEmployment | None:
    """
    Pick the position that represents the candidate NOW.

    A position marked is_current wins over any past position; among
    several, the one that started most recently. With no current
    position, the most recently started parseable position is used.
    Positions with unparseable dates are only considered as a last
    resort (first listed), since résumés conventionally lead with the
    most recent role.

    Profile-level seniority comes from this position specifically, NOT
    from the maximum seniority ever held: a candidate who was once a
    Senior Engineer but is currently an intern is currently an intern.
    """
    if not employment_history:
        return None

    current = [e for e in employment_history if e.is_current]
    candidates = current or employment_history

    dated = [e for e in candidates if e.start_month_index is not None]
    if dated:
        return max(dated, key=lambda e: e.start_month_index)

    return candidates[0]


def build_candidate_profile(pdf_path: str) -> CandidateProfile:

    # PDF -> text

    full_text = extract_text_from_pdf(pdf_path)

    if not full_text.strip():
        raise ValueError("Could not extract text from the PDF.")

    # LLM extracts facts -- the SAME chain app.extractor uses, so the
    # extraction behavior itself is identical; only what we do with the
    # result differs (raw skills preserved, not run through the lossy
    # technical_stack categorization).

    raw_result = extraction_chain.invoke({"resume_text": full_text})

    # Python normalizes and interprets everything below.

    skills, warnings = _normalize_candidate_skills(raw_result.skills)

    employment_history, employment_warnings = _build_employment_history(
        raw_result.employment_history
    )
    warnings.extend(employment_warnings)

    # Total experience uses the existing, unchanged aggregate
    # calculation -- inclusive-month semantics, union of overlapping
    # periods -- so a CandidateProfile and a Resume built from the same
    # PDF always agree on total experience.
    experience = calculate_total_experience(raw_result.employment_history)

    latest = _select_latest_position(employment_history)

    return CandidateProfile(
        candidate_name=raw_result.candidate_name,
        seniority=latest.seniority if latest else None,
        current_role=latest.role if latest else None,
        skills=skills,
        total_experience_months=experience["months"],
        total_experience_years=experience["years"],
        employment_history=employment_history,
        # Task 5 (D5): résumé education extraction is deferred to Task 6.
        education=None,
        raw_text=full_text,
        parse_warnings=warnings,
    )
