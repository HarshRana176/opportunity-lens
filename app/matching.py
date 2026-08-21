"""
Deterministic candidate <-> job matching engine.

Pure Python: no LLM calls, no I/O, no database, no network. Consumes
only app.schemas types (CandidateProfile, JobProfile, and their sub-
schemas) built by app.candidate_extractor / app.job_extractor -- this
module does NOT import from either extractor, or from app.extractor.

Produces structured MatchEvidence, never a score. Weighting/scoring is
explicitly later work (a separate scoring task) that will consume this
evidence; app.matching's only job is to answer, per dimension:

    - which required/preferred skills match, and on what basis
    - does the candidate meet the experience requirement
    - does the candidate's education satisfy the job's requirement
    - does seniority align
    - which of the above are HARD requirements, and do they pass

Every decision resolves to one of four states (app.schemas.MatchStatus):
PASS, FAIL, UNKNOWN, or PARTIAL -- never a plain boolean. UNKNOWN is
distinct from FAIL specifically so that missing information is never
silently treated as ineligibility (nor silently treated as a pass).


SKILL IDENTITY -- WHY THIS MODULE DOES NOT USE app.skills.skill_identity
--------------------------------------------------------------------------
app.skills.skill_identity() returns a single scalar (canonical, falling
back to match_key) and is used by app.job_extractor / app.candidate_
extractor as a DICT KEY for deduplicating skills WITHIN one profile.
That scalar rule was considered and rejected for cross-profile matching
(Task 7 planning, decision D1) for a proven reason: comparing two
skills' identity() outputs for equality misses a real case --

    candidate: "PostgreSQL", persisted BEFORE "postgresql" existed in
               SKILL_CANONICAL -> match_key="postgresql", canonical=None
    job:       "Postgres", parsed AFTER the taxonomy gained the alias
               -> match_key="postgres", canonical="postgresql"

skill_identity(candidate) == "postgresql" (its match_key, since
canonical is None); skill_identity(job) == "postgresql" (its
canonical). These happen to be equal here, but the general rule
"compare canonical if both resolved, else compare match_key" (the
skill-matching prototype used during Task 5/6 testing) returns False
for this exact pair, because it never compares a canonical against a
match_key.

skills_match() below instead compares the SETS {canonical, match_key}
(dropping None) for intersection. This was proven, exhaustively, to
introduce zero false positives and zero false negatives across all
2016 pairs of the current 64-entry SKILL_CANONICAL taxonomy (Task 7
planning verification), and correctly recovers the stale-taxonomy case
above. It does NOT replace skill_identity(): a set intersection is not
an equivalence relation (it is not transitive -- {x} intersects {x,y}
intersects {y}, but {x} does not intersect {y}), so it cannot serve as
a hash key for deduplication the way skill_identity's scalar output
can. The two rules serve different operations and are kept separate on
purpose: skill_identity dedupes skills WITHIN one profile;
skills_match() compares skills ACROSS two profiles.

KNOWN, BOUNDED LIMITATION: this only recovers HALF of the stale-
taxonomy scenario. If the STALE record's raw text was itself an ALIAS
spelling (e.g. candidate persisted "Postgres" -> match_key="postgres",
canonical=None -- before the taxonomy existed -- against a freshly
parsed job "PostgreSQL" -> match_key="postgresql", canonical=
"postgresql"), the sets {"postgres"} and {"postgresql", "postgresql"}
do not intersect, and no pure comparison rule can recover this: the
stale record's JSON contains no link to a canonical that did not exist
when it was persisted. Fixing this fully requires re-normalizing
persisted skills against the current taxonomy, which is a persistence/
freshness concern out of scope for Task 7. See
tests/test_matching_skills.py::TestStaleTaxonomySkew for both cases
made explicit.
"""
from app.schemas import (
    CandidateProfile,
    CandidateSkill,
    EducationEvidence,
    ExperienceEvidence,
    HardConstraint,
    JobProfile,
    MatchEvidence,
    MatchStatus,
    SeniorityEvidence,
    SkillEvidence,
    SkillMatch,
    SkillRequirement,
)


# SKILLS


def _skill_tokens(skill) -> set[str]:
    return {token for token in (skill.canonical, skill.match_key) if token}


def skills_match(
    candidate_skill: CandidateSkill, job_skill: SkillRequirement
) -> tuple[bool, str | None]:
    """
    Returns (matched, matched_on).

    matched_on == "canonical" only when BOTH sides independently
    resolved to the same curated canonical name -- the strongest
    possible evidence. Any other intersection (including the stale-
    taxonomy repair case, and the case where both sides are simply
    unresolved with equal match_key -- e.g. "Kafka") is reported as
    "match_key": weaker evidence, but still a genuine match. matched_on
    is None only when matched is False.

    See this module's docstring for the full rationale and the token-
    set intersection proof.
    """
    candidate_tokens = _skill_tokens(candidate_skill)
    job_tokens = _skill_tokens(job_skill)

    if not (candidate_tokens & job_tokens):
        return False, None

    if (
        candidate_skill.canonical is not None
        and job_skill.canonical is not None
        and candidate_skill.canonical == job_skill.canonical
    ):
        return True, "canonical"

    return True, "match_key"


def _match_requirement(
    candidate_skills: list[CandidateSkill], requirement: SkillRequirement
) -> SkillMatch:
    for candidate_skill in candidate_skills:
        matched, matched_on = skills_match(candidate_skill, requirement)
        if matched:
            return SkillMatch(
                requirement=requirement,
                matched_candidate_skill=candidate_skill,
                matched_on=matched_on,
                status="pass",
            )

    return SkillMatch(
        requirement=requirement,
        matched_candidate_skill=None,
        matched_on=None,
        status="fail",
    )


def match_skills(candidate: CandidateProfile, job: JobProfile) -> SkillEvidence:
    """
    Compares every required and preferred job skill against the
    candidate's skills via skills_match(). Required and preferred are
    evaluated and reported separately -- a missing preferred skill is
    never conflated with a missing required skill.

    unmatched_candidate_skills lists every candidate skill that did not
    correspond to ANY job requirement (required or preferred) -- kept
    as evidence rather than discarded, since it is exactly what a
    future resume-tailoring step would need.
    """
    required_matches = [
        _match_requirement(candidate.skills, requirement)
        for requirement in job.required_skills
    ]
    preferred_matches = [
        _match_requirement(candidate.skills, requirement)
        for requirement in job.preferred_skills
    ]

    all_requirements = job.required_skills + job.preferred_skills
    unmatched_candidate_skills = [
        candidate_skill
        for candidate_skill in candidate.skills
        if not any(
            skills_match(candidate_skill, requirement)[0]
            for requirement in all_requirements
        )
    ]

    return SkillEvidence(
        required=required_matches,
        preferred=preferred_matches,
        matched_required=sum(1 for m in required_matches if m.status == "pass"),
        total_required=len(required_matches),
        matched_preferred=sum(1 for m in preferred_matches if m.status == "pass"),
        total_preferred=len(preferred_matches),
        unmatched_candidate_skills=unmatched_candidate_skills,
    )


# EXPERIENCE


def match_experience(candidate: CandidateProfile, job: JobProfile) -> ExperienceEvidence:
    """
    Compares candidate.total_experience_months against
    job.experience -- both already the same unit (months), established
    by app.experience / app.requirements; no conversion happens here.

    Approved decisions:
      - unspecified requirement -> UNKNOWN (never PASS: "no requirement
        stated" must not be conflated with "requirement satisfied").
      - contradictory requirement (min > max) -> UNKNOWN.
      - below minimum -> FAIL, with shortfall_months.
      - above maximum -> PARTIAL (over-qualification is not a failure).
      - otherwise -> PASS.
    """
    requirement = job.experience
    months = candidate.total_experience_months

    if not requirement.is_specified:
        return ExperienceEvidence(
            requirement=requirement,
            candidate_months=months,
            status="unknown",
            reason="No experience requirement was specified in the job description.",
        )

    if (
        requirement.min_months is not None
        and requirement.max_months is not None
        and requirement.min_months > requirement.max_months
    ):
        return ExperienceEvidence(
            requirement=requirement,
            candidate_months=months,
            status="unknown",
            reason=(
                f"Experience requirement is contradictory: minimum "
                f"({requirement.min_months} months) exceeds maximum "
                f"({requirement.max_months} months)."
            ),
        )

    if requirement.min_months is not None and months < requirement.min_months:
        shortfall = requirement.min_months - months
        return ExperienceEvidence(
            requirement=requirement,
            candidate_months=months,
            status="fail",
            shortfall_months=shortfall,
            reason=(
                f"Requires at least {requirement.min_months} months; candidate "
                f"has {months} months (shortfall {shortfall} months)."
            ),
        )

    if requirement.max_months is not None and months > requirement.max_months:
        surplus = months - requirement.max_months
        return ExperienceEvidence(
            requirement=requirement,
            candidate_months=months,
            status="partial",
            surplus_months=surplus,
            reason=(
                f"Candidate exceeds the stated maximum of "
                f"{requirement.max_months} months by {surplus} months "
                f"(over-qualified)."
            ),
        )

    return ExperienceEvidence(
        requirement=requirement,
        candidate_months=months,
        status="pass",
        reason=f"Candidate has {months} months, satisfying the requirement.",
    )


# EDUCATION


def match_education(candidate: CandidateProfile, job: JobProfile) -> EducationEvidence:
    """
    Compares the candidate's EducationBackground against the job's
    EducationRequirement using the shared ordinal EducationLevel.

    Approved decisions:
      - no level stated on the job -> UNKNOWN.
      - job marks education preferred (not required) -> PASS (soft;
        never a hard blocker, regardless of candidate data).
      - candidate has no education section at all -> UNKNOWN (not
        FAIL: "no section found" is not evidence of "no degree").
      - candidate has education records but NONE resolved to a level
        -> UNKNOWN (not FAIL: an unresolved degree is not an absent
        one -- see app.education's never-delete principle).
      - highest resolved level >= required level -> PASS, with
        matching_records naming which degree(s) qualified.
      - highest resolved level < required level -> FAIL.

    field_overlap is computed as case-insensitive exact-string overlap
    between the candidate's recorded fields of study and the job's
    acceptable fields_of_study, purely as INFORMATIONAL evidence.
    field_match_assessable is always False: there is no field-of-study
    taxonomy today, and job-side phrases such as "related field" are
    not a field name that can be deterministically compared against
    anything -- field_overlap NEVER affects `status`.
    """
    requirement = job.education
    background = candidate.education
    highest = background.highest_level if background else None

    matching_records = []
    if background and requirement.minimum_level is not None:
        matching_records = [
            record
            for record in background.records
            if record.level is not None and record.level >= requirement.minimum_level
        ]

    field_overlap = []
    if background and requirement.fields_of_study:
        candidate_fields_lower = {
            record.field_of_study_raw.strip().lower()
            for record in background.records
            if record.field_of_study_raw and record.field_of_study_raw.strip()
        }
        field_overlap = [
            field
            for field in requirement.fields_of_study
            if field.strip().lower() in candidate_fields_lower
        ]

    if requirement.minimum_level is None:
        status = "unknown"
        reason = "No education level requirement was specified in the job description."
    elif not requirement.is_required:
        status = "pass"
        reason = (
            f"Education (minimum {requirement.minimum_level.name}) is preferred, "
            f"not required; treated as satisfied."
        )
    elif background is None:
        status = "unknown"
        reason = "No education information is available for this candidate."
    elif highest is None:
        status = "unknown"
        reason = (
            "Candidate has education records, but none could be resolved to a "
            "recognized education level."
        )
    elif highest >= requirement.minimum_level:
        status = "pass"
        reason = (
            f"Candidate's highest recognized education level ({highest.name}) "
            f"meets the requirement ({requirement.minimum_level.name})."
        )
    else:
        status = "fail"
        reason = (
            f"Candidate's highest recognized education level ({highest.name}) "
            f"is below the requirement ({requirement.minimum_level.name})."
        )

    return EducationEvidence(
        requirement=requirement,
        candidate_highest_level=highest,
        matching_records=matching_records,
        status=status,
        field_overlap=field_overlap,
        field_match_assessable=False,
        reason=reason,
    )


# SENIORITY


def match_seniority(candidate: CandidateProfile, job: JobProfile) -> SeniorityEvidence:
    """
    Ordinal comparison of the existing Seniority enum on both sides. No
    title normalization is introduced -- seniority is already derived
    deterministically (app.requirements.derive_seniority) by Tasks 4/5.

    Seniority is NEVER a hard constraint (see evaluate_hard_constraints):
    its derivation from a title is comparatively weak signal, so it
    contributes only soft evidence.

      - either side unknown -> UNKNOWN.
      - candidate >= required -> PASS (exceeding is not a failure).
      - candidate exactly one level below required -> PARTIAL.
      - candidate two or more levels below required -> FAIL.
    """
    required = job.seniority
    candidate_seniority = candidate.seniority

    if required is None or candidate_seniority is None:
        return SeniorityEvidence(
            required=required,
            candidate=candidate_seniority,
            level_gap=None,
            status="unknown",
            reason="Seniority could not be compared because it is unknown on at least one side.",
        )

    gap = candidate_seniority - required

    if gap >= 0:
        status = "pass"
        reason = (
            f"Candidate seniority ({candidate_seniority.name}) meets or exceeds "
            f"the required level ({required.name})."
        )
    elif gap == -1:
        status = "partial"
        reason = (
            f"Candidate seniority ({candidate_seniority.name}) is one level "
            f"below the required level ({required.name})."
        )
    else:
        status = "fail"
        reason = (
            f"Candidate seniority ({candidate_seniority.name}) is "
            f"{abs(gap)} levels below the required level ({required.name})."
        )

    return SeniorityEvidence(
        required=required,
        candidate=candidate_seniority,
        level_gap=gap,
        status=status,
        reason=reason,
    )


# HARD CONSTRAINTS


_STATUS_PRECEDENCE = {"fail": 0, "unknown": 1, "partial": 2, "pass": 3}


def evaluate_hard_constraints(
    skill_evidence: SkillEvidence,
    experience_evidence: ExperienceEvidence,
    education_evidence: EducationEvidence,
) -> tuple[list[HardConstraint], MatchStatus]:
    """
    Derives the hard-eligibility gate from three dimensions -- explicit
    experience requirements, required (not merely preferred) education,
    and required skills. These are the only hard constraints today;
    seniority is never one (see match_seniority), and there is
    currently no location/salary/work-authorization data on either
    profile to gate on.

    All three kinds are always present in the returned list (a fixed,
    complete shape for callers), with their status already encoding
    "not applicable" appropriately -- e.g. a job with no required
    skills reports "required_skills": PASS (vacuously satisfied, not
    unknown: the absence of any required skill is a known fact, not
    missing information).

    eligibility is the WORST status across the three, using the
    precedence fail > unknown > pass. UNKNOWN never silently resolves
    to PASS: eligibility is only PASS when every hard constraint is
    positively known to pass. Experience's own PARTIAL status
    (over-qualification) is remapped to a PASS-equivalent for the hard-
    constraint entry specifically -- exceeding a stated maximum is not
    a reason to block eligibility, even though ExperienceEvidence
    itself (used for soft signal elsewhere) still reports "partial".
    """
    if skill_evidence.total_required == 0:
        skills_status: MatchStatus = "pass"
        skills_reason = "The job specifies no required skills; the requirement is trivially satisfied."
    elif skill_evidence.matched_required == skill_evidence.total_required:
        skills_status = "pass"
        skills_reason = f"All {skill_evidence.total_required} required skill(s) are matched."
    else:
        skills_status = "fail"
        missing = skill_evidence.total_required - skill_evidence.matched_required
        skills_reason = (
            f"{missing} of {skill_evidence.total_required} required skill(s) "
            f"are not matched."
        )

    experience_status: MatchStatus = (
        "pass" if experience_evidence.status == "partial" else experience_evidence.status
    )

    constraints = [
        HardConstraint(kind="required_skills", status=skills_status, reason=skills_reason),
        HardConstraint(kind="experience", status=experience_status, reason=experience_evidence.reason),
        HardConstraint(kind="education", status=education_evidence.status, reason=education_evidence.reason),
    ]

    worst = min(constraints, key=lambda c: _STATUS_PRECEDENCE[c.status])

    return constraints, worst.status


# ASSEMBLY


def build_match_evidence(candidate: CandidateProfile, job: JobProfile) -> MatchEvidence:
    """
    Orchestrates all four matching dimensions plus hard-constraint
    evaluation into one MatchEvidence. Produces no score -- see this
    module's docstring. semantic is always None (reserved for a later
    embeddings task).
    """
    skill_evidence = match_skills(candidate, job)
    experience_evidence = match_experience(candidate, job)
    education_evidence = match_education(candidate, job)
    seniority_evidence = match_seniority(candidate, job)

    hard_constraints, eligibility = evaluate_hard_constraints(
        skill_evidence, experience_evidence, education_evidence
    )

    unresolved_notes = [
        evidence.reason
        for evidence in (experience_evidence, education_evidence, seniority_evidence)
        if evidence.status == "unknown"
    ]

    return MatchEvidence(
        skills=skill_evidence,
        experience=experience_evidence,
        education=education_evidence,
        seniority=seniority_evidence,
        hard_constraints=hard_constraints,
        eligibility=eligibility,
        semantic=None,
        unresolved_notes=unresolved_notes,
    )


def format_match_evidence(evidence: MatchEvidence) -> str:
    """
    Deterministic, human-readable rendering of a MatchEvidence, built
    entirely from its structured fields -- never a hardcoded template
    filled with example values. Same input always produces the exact
    same output. Intended as a debugging/inspection aid, not a final
    UI string (no localization, no styling beyond plain text).
    """
    lines: list[str] = []

    lines.append("Required skills:")
    if evidence.skills.required:
        for match in evidence.skills.required:
            mark = "PASS" if match.status == "pass" else "FAIL"
            lines.append(f"  {match.requirement.raw}: {mark}")
    else:
        lines.append("  (none specified)")

    lines.append("Preferred skills:")
    if evidence.skills.preferred:
        for match in evidence.skills.preferred:
            mark = "PASS" if match.status == "pass" else "FAIL"
            lines.append(f"  {match.requirement.raw}: {mark}")
    else:
        lines.append("  (none specified)")

    lines.append("")
    lines.append("Experience:")
    if evidence.experience.requirement.min_months is not None:
        lines.append(f"  Required: {evidence.experience.requirement.min_months} months")
    lines.append(f"  Candidate: {evidence.experience.candidate_months} months")
    lines.append(f"  Status: {evidence.experience.status.upper()}")

    lines.append("")
    lines.append("Education:")
    if evidence.education.requirement.minimum_level is not None:
        lines.append(f"  Required: {evidence.education.requirement.minimum_level.name}")
    if evidence.education.candidate_highest_level is not None:
        lines.append(f"  Candidate: {evidence.education.candidate_highest_level.name}")
    lines.append(f"  Status: {evidence.education.status.upper()}")

    lines.append("")
    lines.append("Seniority:")
    if evidence.seniority.required is not None:
        lines.append(f"  Required: {evidence.seniority.required.name}")
    if evidence.seniority.candidate is not None:
        lines.append(f"  Candidate: {evidence.seniority.candidate.name}")
    lines.append(f"  Status: {evidence.seniority.status.upper()}")

    lines.append("")
    lines.append(f"Eligibility: {evidence.eligibility.upper()}")

    return "\n".join(lines)
