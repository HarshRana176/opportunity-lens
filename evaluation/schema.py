"""
Evaluation fixture contract (Task 8C-1).

Defines the on-disk format for labelled candidate/JD evaluation data and
converts it into the app.schemas types the real pipeline consumes. This
package is EVALUATION ONLY: it imports from app.* but nothing in app/
imports from here, and 8C-1 changes no production code.

Named `evaluation` rather than `eval` so the package never reads as the
`eval()` builtin.


WHY LABELS LIVE IN THE FIXTURE AND ARE NULLABLE
--------------------------------------------------------------------------
`relevance` and `eligible` are supplied by a human and default to None.
A fixture with unlabelled candidates loads fine (so the harness can be
built and tested before labelling), but the metrics layer REFUSES to
score it -- see require_labelled(). That split exists so an unlabelled
or half-labelled dataset can never silently produce a number that looks
like a result.

Labels must be assigned WITHOUT looking at any semantic similarity
score. Deriving a label from the model being evaluated makes the
evaluation circular and its conclusion worthless.
"""
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateSkill,
    EducationBackground,
    EducationLevel,
    EducationRecord,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    Seniority,
    SkillRequirement,
)

# Strata exist so the corpus can be checked for adversarial balance
# rather than accidentally consisting of cases that flatter semantic
# similarity. See evaluation/README-strata in the harness docstring.
Stratum = Literal[
    "obvious_match",
    "obvious_non_match",
    "structured_tied",
    "keyword_stuffed",
    "generic_narrative",
    "career_changer",
    "domain_adjacent_unqualified",
    "ineligible_but_impressive",
    "missing_narrative",
    "ambiguous",
    # v2 corpus additions (evaluation-only; see evaluation/datasets_v2).
    "strong_direct",
    "strong_paraphrased",
    "lexical_decoy",
    "adjacent_transferable",
]

MAX_RELEVANCE = 3


class Labels(BaseModel):
    """
    Human ground truth. NEVER generated, inferred, or modified by code
    in this repository.

    relevance is graded 0-3 ("would a hiring manager shortlist this?"):
        0 = would not consider
        1 = weak, probably reject
        2 = plausible, worth a screen
        3 = strong, clearly shortlist

    eligible is a SEPARATE judgement: does the candidate meet the job's
    hard requirements? It is deliberately not derived from relevance --
    a candidate can be genuinely impressive (high relevance if hired)
    while being hard-ineligible, and conflating the two is how hard
    constraints get quietly eroded.
    """

    relevance: Optional[int] = Field(default=None, ge=0, le=MAX_RELEVANCE)

    eligible: Optional[bool] = None

    labeller: Optional[str] = None

    labelled_on: Optional[str] = None

    note: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        return self.relevance is not None and self.eligible is not None


class FixtureSkill(BaseModel):
    raw: str
    canonical: Optional[str] = None


class FixtureEmployment(BaseModel):
    company: str
    role: Optional[str] = None
    start_date: str
    end_date: str
    duration_months: Optional[int] = None
    seniority: Optional[Seniority] = None
    is_current: bool = False
    responsibilities: list[str] = Field(default_factory=list)


class FixtureEducation(BaseModel):
    degree_raw: str
    level: Optional[EducationLevel] = None
    field_of_study_raw: Optional[str] = None


class FixtureCandidate(BaseModel):
    candidate_id: str
    stratum: Stratum
    candidate_name: str = "Candidate"
    seniority: Optional[Seniority] = None
    skills: list[FixtureSkill] = Field(default_factory=list)
    total_experience_months: int = 0
    employment_history: list[FixtureEmployment] = Field(default_factory=list)
    education: list[FixtureEducation] = Field(default_factory=list)
    labels: Labels = Field(default_factory=Labels)

    def to_profile(self) -> CandidateProfile:
        education = None
        if self.education:
            records = [
                EducationRecord(
                    degree_raw=e.degree_raw,
                    degree_key=e.degree_raw.lower().replace(" ", "_"),
                    level=e.level,
                    resolution="taxonomy" if e.level is not None else "unresolved",
                    field_of_study_raw=e.field_of_study_raw,
                )
                for e in self.education
            ]
            levels = [r.level for r in records if r.level is not None]
            education = EducationBackground(
                records=records, highest_level=max(levels) if levels else None
            )

        return CandidateProfile(
            candidate_name=self.candidate_name,
            seniority=self.seniority,
            current_role=self.employment_history[0].role if self.employment_history else None,
            skills=[
                CandidateSkill(
                    raw=s.raw, match_key=s.raw.lower(), canonical=s.canonical,
                    category=None, resolution="taxonomy" if s.canonical else "unresolved",
                )
                for s in self.skills
            ],
            total_experience_months=self.total_experience_months,
            total_experience_years=round(self.total_experience_months / 12, 2),
            employment_history=[
                CandidateEmployment(
                    company=e.company, role=e.role, start_date=e.start_date,
                    end_date=e.end_date, duration_months=e.duration_months,
                    seniority=e.seniority, is_current=e.is_current,
                    responsibilities=e.responsibilities,
                )
                for e in self.employment_history
            ],
            education=education,
            raw_text="(evaluation fixture; raw_text deliberately unused by matching)",
        )


class FixtureJob(BaseModel):
    title: str
    seniority: Optional[Seniority] = None
    required_skills: list[FixtureSkill] = Field(default_factory=list)
    preferred_skills: list[FixtureSkill] = Field(default_factory=list)
    min_experience_months: Optional[int] = None
    max_experience_months: Optional[int] = None
    minimum_education_level: Optional[EducationLevel] = None
    education_required: bool = True
    responsibilities: list[str] = Field(default_factory=list)

    def to_profile(self) -> JobProfile:
        return JobProfile(
            title=self.title,
            seniority=self.seniority,
            required_skills=[
                SkillRequirement(
                    raw=s.raw, match_key=s.raw.lower(), canonical=s.canonical,
                    category=None, resolution="taxonomy" if s.canonical else "unresolved",
                    requirement_level="required",
                )
                for s in self.required_skills
            ],
            preferred_skills=[
                SkillRequirement(
                    raw=s.raw, match_key=s.raw.lower(), canonical=s.canonical,
                    category=None, resolution="taxonomy" if s.canonical else "unresolved",
                    requirement_level="preferred",
                )
                for s in self.preferred_skills
            ],
            experience=ExperienceRequirement(
                min_months=self.min_experience_months,
                max_months=self.max_experience_months,
                is_specified=self.min_experience_months is not None
                or self.max_experience_months is not None,
            ),
            education=EducationRequirement(
                minimum_level=self.minimum_education_level,
                is_required=self.education_required,
            ),
            responsibilities=self.responsibilities,
            raw_text="(evaluation fixture; raw_text deliberately unused by matching)",
        )


class FixtureCase(BaseModel):
    """
    One job description plus every candidate labelled against it.

    `split` is assigned per JD, not per candidate: candidates for the
    same JD are correlated, so splitting at the candidate level would
    leak dev information into the holdout.
    """

    jd_id: str
    # "validation" added for the v2 corpus's model-selection split, which
    # must stay distinct from both dev (fitting) and holdout (sealed).
    # Dataset.dev/.holdout below are unchanged and do not read it; a
    # validation-split case is simply absent from both until the harness
    # is updated to consume it -- deliberately not done in this change.
    split: Literal["dev", "holdout", "validation"]
    job: FixtureJob
    candidates: list[FixtureCandidate] = Field(default_factory=list)
    note: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        return bool(self.candidates) and all(c.labels.is_labelled for c in self.candidates)

    def unlabelled_ids(self) -> list[str]:
        return [c.candidate_id for c in self.candidates if not c.labels.is_labelled]


class Dataset(BaseModel):
    cases: list[FixtureCase] = Field(default_factory=list)

    @property
    def dev(self) -> list[FixtureCase]:
        return [c for c in self.cases if c.split == "dev"]

    @property
    def holdout(self) -> list[FixtureCase]:
        return [c for c in self.cases if c.split == "holdout"]

    def unlabelled(self) -> dict[str, list[str]]:
        return {
            case.jd_id: case.unlabelled_ids()
            for case in self.cases
            if case.unlabelled_ids()
        }


class UnlabelledDatasetError(RuntimeError):
    """Raised when metrics are requested for data a human has not labelled."""


def require_labelled(dataset: Dataset) -> None:
    """
    Gate between "loads fine" and "may be scored". The harness can be
    built and unit-tested against unlabelled fixtures, but no metric may
    ever be computed from them -- an unlabelled dataset must fail loudly
    rather than quietly producing a number that looks like a result.
    """
    missing = dataset.unlabelled()
    if missing:
        detail = "; ".join(f"{jd}: {', '.join(ids)}" for jd, ids in sorted(missing.items()))
        raise UnlabelledDatasetError(
            f"Refusing to compute metrics: {sum(len(v) for v in missing.values())} "
            f"candidate(s) have no human labels -- {detail}"
        )


def load_dataset(directory: str | Path) -> Dataset:
    """
    Load every *.json in `directory` (sorted by filename, so case order
    is deterministic) into one Dataset. Files whose name starts with an
    underscore are ignored, which is how templates and examples are kept
    out of the real corpus.
    """
    directory = Path(directory)
    cases: list[FixtureCase] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_case in payload if isinstance(payload, list) else [payload]:
            cases.append(FixtureCase.model_validate(raw_case))

    seen: set[str] = set()
    for case in cases:
        if case.jd_id in seen:
            raise ValueError(f"Duplicate jd_id in dataset: {case.jd_id!r}")
        seen.add(case.jd_id)

    return Dataset(cases=cases)
