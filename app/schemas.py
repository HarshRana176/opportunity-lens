from enum import IntEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field


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
