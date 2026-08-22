from enum import IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmploymentPeriod(BaseModel):
    company: str = Field(
        description="Employer/company name exactly as written."
    )

    role: Optional[str] = Field(
        default=None,
        description="Job title exactly as written."
    )

    start_date: str = Field(
        description="Employment start date exactly as written."
    )

    end_date: str = Field(
        description=(
            "Employment end date exactly as written. "
            "If ongoing, return exactly 'Present'."
        )
    )


class RawResumeExtraction(BaseModel):
    candidate_name: str = Field(
        description="Candidate's full name exactly as written."
    )

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete technical technologies explicitly "
            "mentioned in the resume. Return as a flat list."
        )
    )


class TechnicalStack(BaseModel):
    programming_languages: list[str] = Field(
        default_factory=list
    )

    frameworks: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )


class ResumeExtraction(BaseModel):
    candidate_name: str

    technical_stack: TechnicalStack

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    total_experience_months: int

    total_experience_years: float


class SkillCategory(BaseModel):

    category: Literal[
        "programming_language",
        "framework",
        "tool",
        "exclude"
    ]


# ---------------------------------------------------------------------------
# Task 4 additions below (Job Description parsing). Everything above this
# line is the résumé-side contract and is unchanged by Task 4.
# ---------------------------------------------------------------------------


class EducationLevel(IntEnum):
    """Ordinal so "bachelor's or higher" is a plain >= comparison later."""

    HIGH_SCHOOL = 1
    ASSOCIATE = 2
    BACHELORS = 3
    MASTERS = 4
    DOCTORATE = 5


class Seniority(IntEnum):
    """Ordinal job-level, derived from a title. Same >= comparison rationale."""

    INTERN = 1
    JUNIOR = 2
    MID = 3
    SENIOR = 4
    LEAD = 5
    PRINCIPAL = 6


class ExperienceRequirement(BaseModel):
    """
    A JD's experience requirement, in months (matching Resume's
    total_experience_months unit exactly). Produced deterministically by
    app.requirements.parse_experience_requirement from an LLM-extracted
    verbatim phrase -- never computed by the LLM itself.

    is_specified distinguishes "no requirement was stated" from "0
    months required": both leave min_months/max_months as None, but
    only a genuinely unspecified requirement should be excluded from
    future experience-based scoring rather than treated as satisfied
    by any candidate.
    """

    min_months: Optional[int] = None

    max_months: Optional[int] = None

    raw_text: Optional[str] = None

    is_specified: bool = False


class EducationRequirement(BaseModel):
    """
    A JD's education requirement. minimum_level is the deterministic
    mapping (via app.taxonomy.EDUCATION_LEVEL_TERMS) of an LLM-extracted
    verbatim education phrase to an ordinal EducationLevel.
    fields_of_study is extracted verbatim by the LLM, not parsed by
    regex, for the same reason skills are LLM-extracted: identifying
    field names in free text is a language-understanding task.
    """

    minimum_level: Optional[EducationLevel] = None

    fields_of_study: list[str] = Field(default_factory=list)

    raw_text: Optional[str] = None

    is_required: bool = False


class NormalizedSkill(BaseModel):
    """
    A skill mention resolved (to whatever extent possible) against the
    curated taxonomy. Produced by app.skills.normalize_skill /
    enrich_unresolved_skills -- never by the LLM directly assigning a
    canonical name.

    match_key is ALWAYS populated (lowercase + whitespace-collapsed
    only -- no punctuation stripping, so "c", "c++", "c#" stay distinct
    identities). It is the fallback matching handle for a skill that
    could not be canonicalized, so an unresolved skill is still
    matchable by exact match_key equality even with no taxonomy entry.

    canonical/category are None when resolution == "unresolved": the
    skill is retained in raw form, never discarded, per the Task 4 D6
    design (the LLM has no authority to delete a skill).
    """

    raw: str

    match_key: str

    canonical: Optional[str] = None

    category: Optional[Literal["programming_languages", "frameworks", "tools"]] = None

    resolution: Literal["taxonomy", "llm", "unresolved"]


class SkillRequirement(NormalizedSkill):
    """A NormalizedSkill plus whether the JD marked it required or preferred."""

    requirement_level: Literal["required", "preferred"]


class BatchSkillItem(BaseModel):
    """One entry of a batched unknown-skill classification response."""

    name: str = Field(
        description="The technology name, copied EXACTLY as given -- do not rename, reformat, or correct."
    )

    category: Literal[
        "programming_language",
        "framework",
        "tool",
        "exclude",
    ]


class BatchSkillClassification(BaseModel):
    """
    Response shape for classifying multiple unknown skills in one LLM
    call (see app.llm.batch_skill_classifier_chain and
    app.skills.enrich_unresolved_skills). Advisory only: a name in
    `items` is matched back to the input by exact string equality,
    never by position, and an `exclude` verdict never deletes the
    corresponding skill -- see app.skills for the never-delete rule.
    """

    items: list[BatchSkillItem] = Field(default_factory=list)


class RawSkillMention(BaseModel):
    """One skill mention as the LLM found it in the JD -- verbatim name,
    plus whether the JD presents it as required or preferred. This
    required/preferred label IS a judgment call (it depends on phrasing
    and section headings), so it is the one thing the LLM is trusted to
    label directly here; identity/category resolution still happens
    afterward, deterministically, via app.skills."""

    name: str = Field(
        description="The technology/skill name exactly as written in the JD."
    )

    level: Literal["required", "preferred"] = Field(
        description=(
            "\"required\" if the JD presents this as a must-have "
            "(e.g. under a Requirements/Must Have section, or phrased "
            "as required/mandatory). \"preferred\" if presented as "
            "nice-to-have/bonus/a plus."
        )
    )


class RawJobCoreExtraction(BaseModel):
    """
    LLM-facing extraction contract, call 1 of 2: title, responsibilities,
    skill mentions. Split from experience/education extraction
    (RawJobRequirementsExtraction below) into its own call deliberately
    -- empirically, on this repo's 3B local model, asking for title +
    responsibilities + skills + experience + education in ONE structured
    call caused experience_text/education_text to come back null with
    high frequency (reproducible across schema-field-order and prompt-
    wording variants), while asking for just title/responsibilities/
    skills reliably returned all fields correctly and consistently
    across repeated runs. Two focused calls instead of one large call.
    """

    title: str = Field(
        description="Job title exactly as written."
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="Responsibility/duty bullet points, copied verbatim as separate list items."
    )

    skill_mentions: list[RawSkillMention] = Field(
        default_factory=list,
        description=(
            "Every concrete technology/skill explicitly mentioned, each "
            "labeled required or preferred. Do not invent skills not "
            "present in the text."
        )
    )


class RawJobRequirementsExtraction(BaseModel):
    """
    LLM-facing extraction contract, call 2 of 2: experience and education
    requirements only. See RawJobCoreExtraction's docstring for why this
    is a separate call.

    Deliberately does NOT include a field-of-study field: empirically,
    on this repo's 3B local model, adding a third field here (even
    lightly emphasized) made education_text unreliable again -- its
    mere presence in the schema was enough to destabilize the other two
    fields, independent of prompt wording (confirmed by testing the
    identical 2-field schema/prompt both with and without a third
    field). app.requirements.parse_education_requirement instead
    derives fields of study deterministically via regex against the
    (reliably-extracted) education_text -- consistent with how
    app.experience already turns a verbatim date phrase into a
    structured value without further LLM involvement.
    """

    experience_text: Optional[str] = Field(
        default=None,
        description=(
            "The experience requirement phrase exactly as written "
            "(e.g. '3+ years', '2-4 years of relevant experience'). "
            "Null if no experience requirement is mentioned. Do not "
            "calculate or convert this to a number."
        )
    )

    education_text: Optional[str] = Field(
        default=None,
        description=(
            "The education requirement phrase exactly as written "
            "(e.g. \"Bachelor's degree in Computer Science or related "
            "field\"). Null if no education requirement is mentioned."
        )
    )


class RawJobExtraction(BaseModel):
    """
    Merged internal representation combining RawJobCoreExtraction and
    RawJobRequirementsExtraction into the single verbatim-extraction
    shape app.job_extractor works from. Not itself an LLM response
    model -- assembled in app.job_extractor from the two chain calls.
    """

    title: str

    responsibilities: list[str] = Field(default_factory=list)

    skill_mentions: list[RawSkillMention] = Field(default_factory=list)

    experience_text: Optional[str] = None

    education_text: Optional[str] = None


class JobCreateRequest(BaseModel):
    """Request body for POST /jobs -- text-only JD input (Task 4 D1)."""

    job_text: str


class JobProfile(BaseModel):
    """
    Final structured result of JD parsing (app.job_extractor.extract_job).
    required_skills/preferred_skills are flat lists of SkillRequirement
    (not nested by category like TechnicalStack) so a future matcher can
    iterate requirements directly and explain per-requirement matches
    ("matched Python (required); missing Kubernetes (preferred)").
    """

    title: str

    seniority: Optional[Seniority] = None

    required_skills: list[SkillRequirement] = Field(default_factory=list)

    preferred_skills: list[SkillRequirement] = Field(default_factory=list)

    experience: ExperienceRequirement

    education: EducationRequirement

    responsibilities: list[str] = Field(default_factory=list)

    raw_text: str

    parse_warnings: list[str] = Field(default_factory=list)


class ResumeResponse(BaseModel):
    """API response shape for a persisted resume (POST and GET alike)."""

    model_config = {"from_attributes": True}

    id: int

    candidate_name: str

    technical_stack: TechnicalStack

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    total_experience_months: int

    total_experience_years: float


class JobResponse(BaseModel):
    """API response shape for a persisted job description (POST and GET alike)."""

    model_config = {"from_attributes": True}

    id: int

    title: str

    seniority: Optional[Seniority] = None

    required_skills: list[SkillRequirement] = Field(default_factory=list)

    preferred_skills: list[SkillRequirement] = Field(default_factory=list)

    experience: ExperienceRequirement

    education: EducationRequirement

    responsibilities: list[str] = Field(default_factory=list)

    raw_text: str

    parse_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 5 additions below (Candidate Profile). Everything above this line --
# the résumé contract, the JD contract, and the shared vocabulary -- is
# unchanged by Task 5.
# ---------------------------------------------------------------------------


class CandidateSkill(NormalizedSkill):
    """
    A candidate's skill, using the SAME normalized representation as the
    JD side's SkillRequirement (both inherit NormalizedSkill) so the two
    can be compared directly on `canonical`, or on `match_key` when
    either side is unresolved.

    Deliberately has no `requirement_level`: required-vs-preferred is a
    property of a job's demand, not of a candidate's ability.
    """


class CandidateEmployment(BaseModel):
    """
    One position from a résumé, carrying the verbatim strings AND their
    deterministic normalization side by side.

    company/role/start_date/end_date are preserved exactly as the résumé
    wrote them and are never discarded, even when the dates cannot be
    parsed -- in that case the derived fields are None rather than the
    entry being dropped or the dates being invented.

    responsibilities (Task 8B-2a) is the position's verbatim
    responsibility/achievement bullets, populated by a SEPARATE
    extraction chain after this record already exists -- it never
    affects which positions exist, nor any date/duration/seniority
    value. It defaults to an empty list, so every profile built before
    8B-2a (and every profile whose narrative extraction found nothing
    or failed) remains valid and unchanged.

    It exists to give the semantic-similarity dimension a candidate-
    side text that is NOT already covered by structured matching: the
    work actually performed, as opposed to skills, dates, education,
    or title.
    """

    company: str

    role: Optional[str] = None

    start_date: str

    end_date: str

    start_month_index: Optional[int] = None

    end_month_index: Optional[int] = None

    duration_months: Optional[int] = None

    seniority: Optional[Seniority] = None

    is_current: bool = False

    responsibilities: list[str] = Field(default_factory=list)


class RawEducationRecord(BaseModel):
    """
    One education entry as the LLM found it in a résumé -- verbatim
    only. Mirrors the discipline of RawResumeExtraction/RawSkillMention:
    extraction copies text; app.education does all interpretation.
    """

    degree: str = Field(
        description="The degree/qualification exactly as written (e.g. 'B. Tech', 'Class XII', 'MBA')."
    )

    field_of_study: Optional[str] = Field(
        default=None,
        description="Field/major exactly as written. Null if not mentioned."
    )

    institution: Optional[str] = Field(
        default=None,
        description="School/college/university name exactly as written. Null if not mentioned."
    )

    completion_text: Optional[str] = Field(
        default=None,
        description=(
            "Graduation/completion year, date, or status exactly as "
            "written (e.g. '2026', 'May 2024', 'In Progress'). Null if "
            "not mentioned. Do not calculate or infer a date."
        )
    )


class RawEducationExtraction(BaseModel):
    """
    LLM-facing extraction contract for the Task 6 education chain
    (app.llm.education_extraction_chain), consumed only by
    app.candidate_extractor -- NOT by app.extractor.extract_resume(),
    which remains frozen. Never invents an entry; an empty list means
    no education section was found.
    """

    education: list[RawEducationRecord] = Field(
        default_factory=list,
        description="Every education entry in the résumé, copied verbatim. Empty if none is present."
    )


class RawEmploymentNarrative(BaseModel):
    """
    The responsibility/achievement bullets the LLM found under ONE
    position, verbatim. Same discipline as RawEducationRecord: the
    chain copies text, app.candidate_extractor does all interpretation
    and all matching-back.

    company exists solely so the bullets can be matched back to an
    employment record that was ALREADY extracted by the frozen résumé
    chain -- it never creates an employment record. See
    app.candidate_extractor._attach_work_narrative.
    """

    company: str = Field(
        description="Employer/company name exactly as written, so these bullets can be matched to the right position."
    )

    role: Optional[str] = Field(
        default=None,
        description="Job title exactly as written. Null if not mentioned."
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "Each responsibility/achievement bullet under this position, "
            "copied verbatim as a separate list item. Do not summarize, "
            "merge, rewrite, or invent bullets."
        )
    )


class RawWorkNarrativeExtraction(BaseModel):
    """
    LLM-facing extraction contract for the Task 8B-2a work-narrative
    chain (app.llm.work_narrative_extraction_chain), consumed only by
    app.candidate_extractor -- NOT by app.extractor.extract_resume(),
    and NOT folded into RawResumeExtraction.

    Kept as a SEPARATE chain for the reason documented on
    education_extraction_chain: adding fields to RawResumeExtraction
    empirically collapsed skill extraction on this repo's model
    (31 skills -> 0, reproducibly). An empty list means no
    responsibility bullets were found, which is a normal outcome for
    résumés that list only company/role/dates.
    """

    positions: list[RawEmploymentNarrative] = Field(
        default_factory=list,
        description="Every position that has responsibility bullets. Empty if none is present."
    )


class EducationRecord(BaseModel):
    """
    One candidate education entry, normalized -- the raw/normalized/
    canonical split applied to education, mirroring NormalizedSkill.

    degree_raw/field_of_study_raw/institution_raw/completion_raw are
    ALWAYS preserved exactly as extracted, even when degree_key cannot
    be resolved to a level: normalization failing must never delete or
    rewrite the original résumé wording (e.g. "Class X" is never
    rewritten to "High School" even though it resolves to
    EducationLevel.HIGH_SCHOOL -- see app.taxonomy.DEGREE_CANONICAL).

    completion_raw is deliberately a plain string, never a date: résumé
    completion text is sometimes a year, sometimes a percentage/CGPA
    line, sometimes "In Progress" -- inventing a date from ambiguous
    text is exactly the fabrication this schema exists to prevent.
    """

    degree_raw: str

    field_of_study_raw: Optional[str] = None

    institution_raw: Optional[str] = None

    completion_raw: Optional[str] = None

    degree_key: str

    level: Optional[EducationLevel] = None

    resolution: Literal["taxonomy", "unresolved"]


class EducationBackground(BaseModel):
    """
    A candidate's full education background -- ALL records, not just
    the highest one (a candidate may list a B.Tech and prior schooling;
    reducing that to one value would discard information a future
    matcher needs, e.g. to check a specific field of study).

    highest_level is a derived convenience (the max EducationLevel
    across records whose level resolved), never a replacement for
    `records`. It is None when no record resolved to a level, which is
    different from CandidateProfile.education being None outright (no
    education section was found at all) -- see
    app.education.build_education_background for exactly how these two
    "no information" states are kept distinct.
    """

    records: list[EducationRecord] = Field(default_factory=list)

    highest_level: Optional[EducationLevel] = None

    raw_text: Optional[str] = None


class CandidateProfile(BaseModel):
    """
    Canonical, matchable representation of a candidate -- the résumé-side
    counterpart to JobProfile, and the layer a future matching engine
    compares against it.

    Built by app.candidate_extractor.build_candidate_profile from the
    RAW extracted skills (RawResumeExtraction.skills), never from
    ResumeExtraction.technical_stack: that transformation is lossy (it
    drops anything the per-skill LLM classifier labels "exclude",
    which silently destroys real technologies -- "Kafka" among them),
    and a candidate profile must never lose a skill a job might require.
    """

    candidate_name: str

    seniority: Optional[Seniority] = None

    current_role: Optional[str] = None

    skills: list[CandidateSkill] = Field(default_factory=list)

    total_experience_months: int

    total_experience_years: float

    employment_history: list[CandidateEmployment] = Field(default_factory=list)

    education: Optional[EducationBackground] = None

    raw_text: str

    parse_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 7 additions below (Matching Engine). Everything above this line --
# the résumé contract, the JD contract, and the CandidateProfile/JobProfile
# contracts -- is unchanged by Task 7. app.matching consumes these schemas
# but does not alter them.
# ---------------------------------------------------------------------------


MatchStatus = Literal["pass", "fail", "unknown", "partial"]
"""
Every matching decision resolves to exactly one of these four states:

    pass    -- the requirement is satisfied by available evidence.
    fail    -- the requirement is not satisfied by available evidence.
    unknown -- there is not enough information to decide (a requirement
               was not stated, or candidate data could not be resolved).
               This is NOT the same as "fail": missing information must
               never be silently treated as ineligibility, and must
               never be silently treated as a pass either.
    partial -- the requirement is directionally satisfied but not
               cleanly (e.g. the candidate exceeds a stated maximum, or
               is one seniority level below what was asked for).

A plain boolean cannot represent "unknown", which is exactly the
distinction this matching layer exists to preserve -- see
app.matching's module docstring for how these are combined.
"""


class SkillMatch(BaseModel):
    """
    One job skill requirement (required or preferred) compared against
    the candidate's skills.

    Skill matching is always binary -- status is "pass" when
    app.matching.skills_match found a corresponding candidate skill,
    else "fail". There is no "unknown" skill match: a skill mention
    either has supporting evidence in the candidate's profile or it
    does not.
    """

    requirement: SkillRequirement

    matched_candidate_skill: Optional[CandidateSkill] = None

    matched_on: Optional[Literal["canonical", "match_key"]] = None

    status: MatchStatus


class SkillEvidence(BaseModel):
    """
    Full skill-matching evidence for one candidate/job pair. Required
    and preferred requirements are kept separate (a missing preferred
    skill is never treated the same as a missing required skill).

    unmatched_candidate_skills retains every candidate skill that did
    not correspond to any job requirement -- kept as evidence (e.g. for
    later resume-tailoring), never discarded.
    """

    required: list[SkillMatch] = Field(default_factory=list)

    preferred: list[SkillMatch] = Field(default_factory=list)

    matched_required: int = 0

    total_required: int = 0

    matched_preferred: int = 0

    total_preferred: int = 0

    unmatched_candidate_skills: list[CandidateSkill] = Field(default_factory=list)


class ExperienceEvidence(BaseModel):
    """
    Candidate total experience compared against a job's
    ExperienceRequirement. candidate_months and requirement use the
    SAME unit (months) already established by app.experience /
    app.requirements -- no conversion happens here.
    """

    requirement: ExperienceRequirement

    candidate_months: int

    status: MatchStatus

    shortfall_months: Optional[int] = None

    surplus_months: Optional[int] = None

    reason: str


class EducationEvidence(BaseModel):
    """
    Candidate education compared against a job's EducationRequirement.

    matching_records names WHICH of the candidate's (possibly several)
    education records satisfied the level requirement -- never reduced
    to a single "does the candidate qualify" boolean without evidence.

    field_overlap and field_match_assessable exist because field-of-
    study compatibility is NOT decidable deterministically today (no
    field-of-study taxonomy exists, and job-side phrases like "related
    field" are not a field name to compare against) -- see
    app.matching.match_education. field_overlap is informational only
    and NEVER affects `status`.
    """

    requirement: EducationRequirement

    candidate_highest_level: Optional[EducationLevel] = None

    matching_records: list[EducationRecord] = Field(default_factory=list)

    status: MatchStatus

    field_overlap: list[str] = Field(default_factory=list)

    field_match_assessable: bool = False

    reason: str


class SeniorityEvidence(BaseModel):
    """
    Candidate seniority compared against a job's required seniority,
    both the existing ordinal Seniority enum -- no title normalization
    is introduced here. level_gap = candidate - required (positive
    means the candidate exceeds the requirement).
    """

    required: Optional[Seniority] = None

    candidate: Optional[Seniority] = None

    level_gap: Optional[int] = None

    status: MatchStatus

    reason: str


class HardConstraint(BaseModel):
    """
    One hard-eligibility check. Only three kinds exist today --
    experience, education (when the job marks it required), and
    required skills -- see app.matching.evaluate_hard_constraints for
    why these three and not others (e.g. location/salary/work
    authorization: no such data exists on either profile yet).
    """

    kind: Literal["experience", "education", "required_skills"]

    status: MatchStatus

    reason: str


class EmploymentSimilarity(BaseModel):
    """
    One CandidateEmployment compared against the job's responsibilities
    (Task 8B-2b-i). Produced by app.semantic_match, one per employment
    record, in the candidate's original employment_history order --
    never sorted, so the evidence always reads in résumé order.

    company/role are carried purely so the evidence is legible ("which
    job scored this?"); they are NOT part of the text that was
    embedded. Only CandidateEmployment.responsibilities is embedded --
    see app.semantic_match.build_candidate_employment_text.

    similarity_score is None whenever this position could not be
    compared, with skipped_reason saying why (no bullets, or the
    provider could not produce a usable embedding). A position that
    could not be scored is NEVER recorded as 0.0: that would be a
    measurement of dissimilarity rather than an absence of one, and
    would silently drag any aggregate down.
    """

    company: str

    role: Optional[str] = None

    similarity_score: Optional[float] = None

    status: MatchStatus = "unknown"

    skipped_reason: Optional[str] = None

    truncated: bool = False


class SemanticEvidence(BaseModel):
    """
    The semantic-similarity dimension. Reserved (and always None) in
    Task 7; POPULATED as of Task 8B-1 -- but only by
    app.semantic.attach_semantic_evidence, never by
    app.matching.build_match_evidence, which remains pure/offline and
    still always returns semantic=None. See app.semantic for why the
    two are deliberately separate functions.

    Task 8B-1 extends this class ADDITIVELY (similarity_score and
    method keep their Task 7 names, types, and defaults) so nothing
    that already reads a SemanticEvidence can break.

    status is the same four-state MatchStatus used by every other
    dimension, and carries the availability contract: when embeddings
    cannot be produced (no provider, provider unavailable, provider
    raised, empty text on either side, or a zero/degenerate vector),
    status is "unknown" and similarity_score is None -- NEVER "fail"
    and NEVER a similarity of 0.0. A missing signal is not evidence of
    dissimilarity, exactly as UNKNOWN is not FAIL everywhere else in
    this matching layer.

    model_id identifies which embedding model produced the vectors.
    Embedding floats are not guaranteed bit-identical across model
    versions or runtimes (CPU/GPU, library version), so a persisted
    similarity_score is only interpretable alongside the model_id that
    produced it -- the same reasoning that makes weights_version
    mandatory on MatchResult.
    """

    similarity_score: Optional[float] = None

    method: Optional[str] = None

    status: MatchStatus = "unknown"

    model_id: Optional[str] = None

    reason: str = ""

    # Task 8B-2b-i additions. Additive with defaults: a SemanticEvidence
    # built by the Task 8B-1 single-pair path (app.semantic) simply
    # leaves these empty/None and behaves exactly as it did before.
    per_employment: list[EmploymentSimilarity] = Field(default_factory=list)

    aggregation: Optional[str] = None

    weighted_mean_score: Optional[float] = None


class MatchEvidence(BaseModel):
    """
    Full structured comparison of one CandidateProfile against one
    JobProfile. Produced by app.matching.build_match_evidence.

    Deliberately carries NO score, weight, or percentage anywhere --
    that is later work (a scoring task) which will consume this
    evidence, not something app.matching computes. eligibility is the
    aggregate of `hard_constraints` (worst status wins: fail > unknown
    > pass), and exists specifically so a strong semantic/soft score
    can never later be presented as a good match when a genuine hard
    requirement fails or is unknown.
    """

    skills: SkillEvidence

    experience: ExperienceEvidence

    education: EducationEvidence

    seniority: SeniorityEvidence

    hard_constraints: list[HardConstraint] = Field(default_factory=list)

    eligibility: MatchStatus

    semantic: Optional[SemanticEvidence] = None

    unresolved_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 8A -- deterministic scoring layer. Consumes MatchEvidence (above);
# never produced by app.matching, never touches an LLM/DB/network/clock.
# See app.scoring for the arithmetic; these are the data shapes only.
# ---------------------------------------------------------------------------


class MatchWeights(BaseModel):
    """
    One named, versioned set of scoring weights. `version` is mandatory
    (no default) and non-empty (`min_length=1`) so a MatchResult can
    never be produced from an unversioned -- or blank-versioned --
    weight set; see app.scoring.DEFAULT_WEIGHTS for the one currently
    shipped. Weights are non-negative; a weight of 0 excludes that
    dimension from overall_score without needing a separate "enabled"
    flag. Named fields (not a dict) so the set of scored dimensions is
    fixed and enumerable at the type level, and so there is no
    dict/mapping in the weight model whose iteration order could ever
    be mistaken for something that affects scoring.

    Frozen (immutable after construction): app.scoring.DEFAULT_WEIGHTS
    is a module-level MatchWeights instance reused as score_match's
    default argument across every call site that doesn't supply its
    own weights -- frozen=True turns any accidental mutation of that
    shared instance into an immediate exception instead of silently
    corrupting the process-wide default.
    """

    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)

    required_skills: float = Field(ge=0)

    preferred_skills: float = Field(ge=0)

    experience: float = Field(ge=0)

    education: float = Field(ge=0)

    seniority: float = Field(ge=0)


class ScoreComponent(BaseModel):
    """
    One dimension's contribution to a MatchResult.overall_score.
    raw_value in [0.0, 1.0] is the status-derived quality for this
    dimension alone (see app.scoring._STATUS_RAW_VALUE); contribution =
    raw_value * weight, already computed so callers never need to
    re-derive it (or accidentally re-derive it differently).
    """

    name: Literal[
        "required_skills", "preferred_skills", "experience", "education", "seniority"
    ]

    status: MatchStatus

    weight: float

    raw_value: float

    contribution: float


class MatchResult(BaseModel):
    """
    A scored MatchEvidence. Produced by app.scoring.score_match, which
    is pure arithmetic over an existing MatchEvidence -- no LLM,
    network, timestamp, randomness, or DB access, and no re-evaluation
    of matching logic (evaluate_hard_constraints's required_skills
    verdict is reused verbatim; see app.scoring).

    weights_version is always present (copied from the MatchWeights
    used) so a persisted MatchResult can always be traced back to the
    weight set that produced it, even after DEFAULT_WEIGHTS changes.

    components is a fixed 5-element list in a fixed order (required_
    skills, preferred_skills, experience, education, seniority) --
    always a list literal built in that order, never derived from
    set/dict iteration -- so overall_score and components are stable
    across processes, PYTHONHASHSEED values, and repeated calls for the
    same (evidence, weights) input.
    """

    evidence: MatchEvidence

    weights_version: str

    overall_score: float

    components: list[ScoreComponent] = Field(default_factory=list)
