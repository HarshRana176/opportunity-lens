"""
High-level job-description extraction pipeline orchestration.

extract_job() wires together, for JD parsing, the analogue of what
app.extractor.extract_resume() does for résumés:

    app.llm            -- two focused LLM calls (external boundary: Ollama)
    app.skills          -- deterministic + LLM-fallback skill normalization
    app.requirements    -- deterministic experience/education/seniority parsing
    app.schemas         -- Pydantic contracts shared across all of the above

Deliberately does NOT import anything from app.extractor (résumé
pipeline) or vice versa: the two pipelines share vocabulary (taxonomy,
schemas, the month unit, the LLM client) but not orchestration, per the
Task 4 design -- résumé parsing must not change as a side effect of
adding JD parsing.
"""
from app.llm import job_core_extraction_chain, job_requirements_extraction_chain
from app.requirements import derive_seniority, parse_education_requirement, parse_experience_requirement
from app.schemas import JobProfile, RawJobExtraction, SkillRequirement
from app.skills import enrich_unresolved_skills, normalize_skill
from app.taxonomy import JD_EXCLUDED_TERMS


def _dedupe_preferring_required(skill_requirements: list[SkillRequirement]) -> list[SkillRequirement]:
    """
    Collapse duplicate skill mentions (e.g. the same technology listed
    under both a Requirements and a Nice-to-have section) to one entry,
    identified by canonical name when known, else by match_key. When a
    skill appears at both levels, "required" wins -- a JD stating a
    skill is required anywhere makes it required overall.
    """
    best_by_identity: dict[str, SkillRequirement] = {}

    for skill in skill_requirements:
        identity = skill.canonical or skill.match_key
        existing = best_by_identity.get(identity)

        if existing is None:
            best_by_identity[identity] = skill
        elif existing.requirement_level == "preferred" and skill.requirement_level == "required":
            best_by_identity[identity] = skill

    return list(best_by_identity.values())


def _normalize_and_enrich_skill_mentions(
    skill_mentions,
) -> tuple[list[SkillRequirement], list[str]]:
    """
    Turn RawSkillMention entries into deduplicated SkillRequirement
    entries (required_skills/preferred_skills combined, not yet split),
    normalizing against the taxonomy and enriching unresolved skills
    with one batched LLM call. Returns (skill_requirements, warnings).
    """
    levels: list[str] = []
    normalized_skills = []

    for mention in skill_mentions:
        normalized = normalize_skill(mention.name, extra_excluded=JD_EXCLUDED_TERMS)
        if normalized is None:
            # Not a technology at all (résumé-side EXCLUDED_TECHNOLOGIES
            # or JD_EXCLUDED_TERMS) -- dropped entirely, not retained
            # even as unresolved. This is the one place a skill mention
            # is discarded, and it is never the LLM's decision.
            continue
        levels.append(mention.level)
        normalized_skills.append(normalized)

    enriched_skills, warnings = enrich_unresolved_skills(normalized_skills)

    skill_requirements = [
        SkillRequirement(**enriched.model_dump(), requirement_level=level)
        for level, enriched in zip(levels, enriched_skills)
    ]

    return _dedupe_preferring_required(skill_requirements), warnings


def extract_job(job_text: str) -> JobProfile:

    if not job_text.strip():
        raise ValueError("Could not extract information: job description text is empty.")

    # LLM extracts facts -- two focused calls, not one; see
    # app.schemas.RawJobCoreExtraction/RawJobRequirementsExtraction for
    # why splitting this way was empirically necessary.

    core = job_core_extraction_chain.invoke({"job_text": job_text})
    reqs = job_requirements_extraction_chain.invoke({"job_text": job_text})

    raw_result = RawJobExtraction(
        title=core.title,
        responsibilities=core.responsibilities,
        skill_mentions=core.skill_mentions,
        experience_text=reqs.experience_text,
        education_text=reqs.education_text,
    )

    # Python normalizes, enriches, and interprets everything else.

    skill_requirements, warnings = _normalize_and_enrich_skill_mentions(
        raw_result.skill_mentions
    )

    required_skills = [s for s in skill_requirements if s.requirement_level == "required"]
    preferred_skills = [s for s in skill_requirements if s.requirement_level == "preferred"]

    experience = parse_experience_requirement(raw_result.experience_text)
    education = parse_education_requirement(raw_result.education_text)
    seniority = derive_seniority(raw_result.title)

    if experience.raw_text and not experience.is_specified:
        warnings.append(
            f"Could not interpret experience requirement: {experience.raw_text!r}"
        )

    if education.raw_text and education.minimum_level is None:
        warnings.append(
            f"Could not interpret education requirement: {education.raw_text!r}"
        )

    return JobProfile(
        title=raw_result.title,
        seniority=seniority,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        experience=experience,
        education=education,
        responsibilities=raw_result.responsibilities,
        raw_text=job_text,
        parse_warnings=warnings,
    )
