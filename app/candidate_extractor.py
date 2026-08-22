"""
Candidate profile construction -- the résumé-side counterpart to
app.job_extractor, producing the canonical CandidateProfile a future
matching engine compares against a JobProfile.

This is a SEPARATE extraction path from app.extractor.extract_resume(),
deliberately (approved Task 5 decision D1-A). It reuses the same shared
parts:

    app.pdf         -- PDF -> raw text (external boundary: PyMuPDF)
    app.llm         -- the SAME résumé extraction chain, PLUS a separate
                       education-only chain (Task 6) (external: Ollama)
    app.skills      -- shared normalization + batch enrichment
    app.experience  -- shared deterministic date/duration primitives
    app.requirements-- shared deterministic seniority derivation
    app.education   -- deterministic degree normalization (Task 6)
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
from app.education import build_education_background
from app.experience import calculate_period_interval, calculate_total_experience
from app.llm import (
    education_extraction_chain,
    extraction_chain,
    work_narrative_extraction_chain,
)
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


def _extract_education(full_text: str) -> tuple:
    """
    Invoke the education extraction chain and normalize its result.

    Mirrors enrich_unresolved_skills's failure-safety contract: any
    exception from the chain (Ollama unreachable, malformed output) is
    caught here so a failure to extract education never fails
    CandidateProfile construction as a whole -- the rest of the profile
    (skills, employment, experience) must still be produced. On
    failure, education comes back None (indistinguishable, deliberately,
    from "no education section" -- both mean "no education information
    is available for this profile") plus a warning explaining why.
    """
    try:
        raw_education = education_extraction_chain.invoke({"resume_text": full_text})
    except Exception:
        return None, ["Education extraction failed; no education information is available."]

    return build_education_background(raw_education.education, full_text)


def _attach_work_narrative(
    employment_history: list[CandidateEmployment],
    full_text: str,
) -> list[str]:
    """
    Populate each CandidateEmployment.responsibilities in place from a
    SEPARATE narrative extraction call, and return any warnings.

    Match-back is by EXACT company string, the same never-delete rule
    app.skills uses to reattach batch-classified skills: the résumé
    chain decides which positions exist, and this step may only add
    bullets to positions that already exist. A narrative entry naming a
    company with no corresponding employment record is recorded as a
    warning and otherwise ignored -- it NEVER creates or invents a
    position, because the LLM naming a company is not evidence that the
    candidate held a job there.

    When several positions share one company (a promotion within the
    same employer), the bullets are attached to the FIRST such record
    rather than duplicated across all of them: attributing the same
    work to two positions would double-count that text in every later
    per-employment semantic comparison.

    Failure-safety mirrors _extract_education: any exception from the
    chain (Ollama unreachable, malformed output) leaves every
    responsibilities list empty and returns a warning, so a narrative
    failure can never fail CandidateProfile construction as a whole.
    The try covers reading `.positions` off the result, not just the
    invoke call -- a chain that returns successfully but hands back
    something of the wrong shape is just as much a narrative failure,
    and must not surface as an AttributeError from profile
    construction.
    """
    try:
        raw_narrative = work_narrative_extraction_chain.invoke({"resume_text": full_text})
        positions = list(raw_narrative.positions)
    except Exception:
        return [
            "Work narrative extraction failed; no responsibility bullets are "
            "available for this profile."
        ]

    warnings: list[str] = []

    for entry in positions:
        bullets = [b.strip() for b in entry.responsibilities if b and b.strip()]
        if not bullets:
            continue

        match = next(
            (e for e in employment_history if e.company == entry.company),
            None,
        )

        if match is None:
            warnings.append(
                f"Work narrative mentioned company {entry.company!r}, which does "
                f"not match any extracted employment record; its bullets were "
                f"not attached."
            )
            continue

        match.responsibilities.extend(bullets)

    return warnings


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

    # Responsibility bullets are extracted by a SEPARATE call (Task
    # 8B-2a) and attached to the employment records built above -- see
    # work_narrative_extraction_chain's docstring in app.llm for why it
    # is not folded into raw_result. This runs AFTER employment history
    # exists precisely so it can only annotate positions, never define
    # them.
    warnings.extend(_attach_work_narrative(employment_history, full_text))

    latest = _select_latest_position(employment_history)

    # Education is extracted via a SEPARATE LLM call (Task 6), never
    # folded into raw_result above -- see education_extraction_chain's
    # docstring in app.llm for why.
    education, education_warnings = _extract_education(full_text)
    warnings.extend(education_warnings)

    return CandidateProfile(
        candidate_name=raw_result.candidate_name,
        seniority=latest.seniority if latest else None,
        current_role=latest.role if latest else None,
        skills=skills,
        total_experience_months=experience["months"],
        total_experience_years=experience["years"],
        employment_history=employment_history,
        education=education,
        raw_text=full_text,
        parse_warnings=warnings,
    )
