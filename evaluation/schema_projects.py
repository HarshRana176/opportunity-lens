"""
Phase 2 evaluation fixture contract: project-aware relevance experiment.

Defines the on-disk format for the NEW, SEPARATE project-evidence
corpus and converts it into app.schemas types via the real,
UNMODIFIED app.matching/app.scoring path for eligibility. This module
is EVALUATION ONLY: it imports from app.* but nothing in app/ imports
from here, and nothing in this file changes production code.

Deliberately independent of evaluation/schema.py's Dataset/FixtureCase
(the existing 360-candidate employment-only corpus): different corpus,
different blind key, different rubric, different strata. The two must
never be merged, and this module never reads the existing corpus's
labels, blind key, or sealed holdout. FixtureSkill/FixtureEmployment/
FixtureEducation ARE reused by import below because they are pure,
frozen data-shape helpers with no corpus data of their own -- reusing
them is not touching the existing corpus.


RELEVANCE vs ELIGIBILITY STAYS FULLY SEPARATE
--------------------------------------------------------------------------
This module adds three NEW relevance dimensions on top of the existing
framework -- employment_relevance, project_relevance, combined_relevance
-- and changes NOTHING about eligibility. A project can raise
combined_relevance to 3 while the candidate's total_experience_months
still fails the JD's experience requirement: eligibility for every
candidate in this corpus is computed exactly the way the existing
360-candidate corpus computes it, by the frozen app.matching/
app.scoring path reading total_experience_months/skills/education/
seniority. Nothing in this module ever writes to, derives, or
overrides total_experience_months or any other eligibility-affecting
field from project evidence. Projects can never manufacture
professional experience.


THE COMBINATION RULE IS A HYPOTHESIS, NOT AN ESTABLISHED TRUTH
--------------------------------------------------------------------------
combine_relevance() below implements combined = max(employment, project)
with named status categories. This is the APPROVED EXPERIMENTAL rule
for this phase, not a proven-correct formula. employment_score and
project_score are ALWAYS retained alongside combined_score specifically
so an alternative combination rule can be tested later against the same
labels without relabelling the corpus -- callers must never discard the
sub-scores after computing a combined value.


LABELS ARE NULLABLE, SAME DISCIPLINE AS evaluation/schema.py
--------------------------------------------------------------------------
employment_relevance and project_relevance default to None and must be
supplied by a human, independently of each other and independently of
any model/embedding score. combined_score/combination_status are
NEVER supplied by a human -- they are always DERIVED, deterministically,
from the two human sub-scores via combine_relevance(), after both are
frozen.
"""
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from evaluation.schema import (
    EducationLevel,
    FixtureEducation,
    FixtureEmployment,
    FixtureSkill,
    Seniority,
)

# Strata for the project-aware relevance experiment. Deliberately a
# DIFFERENT literal from evaluation.schema.Stratum -- this is a
# different experiment with different adversarial-balance concerns
# (fresher-vs-experienced, project-quality tiers) than the employment-
# only corpus's strata.
ProjectStratum = Literal[
    "no_employment_strong_project",
    "no_employment_tutorial_project",
    "no_employment_title_only_project",
    "internship_only_weak",
    "internship_only_strong",
    "strong_employment_no_project",
    "weak_employment_strong_project",
    "conflicting_employment_project",
    "multi_project_corroboration",
    "adjacent_technology_project",
]

MAX_RELEVANCE = 3

RelevanceScore = Literal[0, 1, 2, 3]

CombinationStatus = Literal[
    "employment_sufficient",
    "corroborated",
    "corroborated_weak",
    "employment_only_weak",
    "project_compensates",
    "conflicting_unresolved",
    "insufficient_evidence",
]


class EmploymentRelevanceLabel(BaseModel):
    """
    Human ground truth for how well a candidate's EMPLOYMENT narrative
    evidences the JD's responsibilities, on the Phase 2 rubric (see
    evaluation/PROJECT_RUBRIC.md). NEVER generated, inferred, or
    modified by code in this repository -- same discipline as
    evaluation.schema.Labels.
    """

    score: Optional[RelevanceScore] = Field(default=None, ge=0, le=MAX_RELEVANCE)

    labeller: Optional[str] = None

    labelled_on: Optional[str] = None

    note: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        return self.score is not None


class ProjectRelevanceLabel(BaseModel):
    """
    Human ground truth for how well a candidate's PROJECT evidence
    evidences the JD's responsibilities, on the Phase 2 rubric. Same
    discipline as EmploymentRelevanceLabel -- assigned independently,
    without reference to the employment score, without reference to
    any model/embedding output, and without reference to stratum.
    """

    score: Optional[RelevanceScore] = Field(default=None, ge=0, le=MAX_RELEVANCE)

    labeller: Optional[str] = None

    labelled_on: Optional[str] = None

    note: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        return self.score is not None


def combine_relevance(
    employment: Optional[int], project: Optional[int]
) -> tuple[Optional[int], Optional[CombinationStatus]]:
    """
    THE EXPERIMENTAL COMBINATION RULE: combined = max(employment, project).

    This is the hypothesis approved for Phase 2, not an established
    truth -- see the module docstring and evaluation/PROJECT_RUBRIC.md.
    Deterministic Python only; no LLM, no embedding, no reference to
    any prior score.

    Implemented as a direct, exhaustive 16-cell lookup table (every
    (employment, project) pair in {0,1,2,3}x{0,1,2,3}) rather than a
    chain of magnitude-based rules, specifically because an earlier
    draft of this function used a blanket "|employment-project|>=2 is a
    conflict" shortcut that FIRED BEFORE this table and silently
    overrode cells the table already resolves cleanly -- e.g.
    employment=3/project=0 (diff=3) is simply "a strong job, an
    irrelevant hobby project", not a conflict, and must stay
    employment_sufficient/combined=3, not fall through to
    conflicting_unresolved/combined=None. The table below is the sole
    source of truth for every cell; nothing here infers "conflict" from
    score magnitude alone.

    employment=None means "no employment records at all" (the pure
    fresher case), distinct from employment=0 ("has employment
    records, none of them relevant"). Both end up driven entirely by
    project when project is present.

    "conflicting_unresolved" / "conflicting_resolved_favor_stronger"
    remain valid CombinationStatus values, but this function NEVER
    emits them: a genuine content-level contradiction (e.g. a project
    narrative that actively contradicts the employment narrative) is a
    qualitative judgement this function has no access to from two bare
    integers, and inferring it from magnitude alone was exactly the
    bug this docstring describes. Such a status may only be assigned
    manually, by a human reviewer annotating CombinedRelevanceRecord
    directly (recording their reasoning in the label's `note` field) --
    never inferred here.
    """
    if employment is None and project is None:
        return None, None

    if employment is None:
        return (project, "insufficient_evidence") if project == 0 else (project, "project_compensates")

    if project is None:
        if employment >= 2:
            return employment, "employment_sufficient"
        if employment == 1:
            return employment, "employment_only_weak"
        return employment, "insufficient_evidence"

    # Exhaustive table: (employment, project) -> (combined, status).
    table: dict[tuple[int, int], tuple[int, CombinationStatus]] = {
        (3, 0): (3, "employment_sufficient"), (3, 1): (3, "employment_sufficient"),
        (3, 2): (3, "employment_sufficient"), (3, 3): (3, "employment_sufficient"),
        (2, 0): (2, "employment_sufficient"), (2, 1): (2, "employment_sufficient"),
        (2, 2): (2, "corroborated"),          (2, 3): (3, "corroborated"),
        (1, 0): (1, "employment_only_weak"),  (1, 1): (1, "corroborated_weak"),
        (1, 2): (2, "project_compensates"),   (1, 3): (3, "project_compensates"),
        (0, 0): (0, "insufficient_evidence"), (0, 1): (1, "project_compensates"),
        (0, 2): (2, "project_compensates"),   (0, 3): (3, "project_compensates"),
    }
    return table[(employment, project)]


class CombinedRelevanceRecord(BaseModel):
    """
    One candidate's full Phase 2 relevance record: both sub-scores with
    provenance, plus the derived (experimental) combined result.

    compute_combined() must be called explicitly, after both
    employment.score and project.score are frozen -- it is never
    invoked automatically during labelling, so a human's raw sub-scores
    are never silently overwritten by a derived value, and the derived
    value is always reproducible from the two frozen inputs alone.
    """

    employment: EmploymentRelevanceLabel = Field(default_factory=EmploymentRelevanceLabel)

    project: ProjectRelevanceLabel = Field(default_factory=ProjectRelevanceLabel)

    combined_score: Optional[RelevanceScore] = None

    combination_status: Optional[CombinationStatus] = None

    @property
    def is_labelled(self) -> bool:
        return self.employment.is_labelled and self.project.is_labelled

    def compute_combined(self) -> "CombinedRelevanceRecord":
        score, status = combine_relevance(
            self.employment.score if self.employment.is_labelled else None,
            self.project.score if self.project.is_labelled else None,
        )
        return self.model_copy(update={"combined_score": score, "combination_status": status})


class FixtureProject(BaseModel):
    """
    One project on a Phase 2 fixture candidate. Mirrors the shape of
    app.schemas.CandidateProject, for the same reason
    evaluation.schema.FixtureEmployment mirrors CandidateEmployment:
    a fixture-side twin the corpus JSON is built from, converted to the
    real production type by to_profile() below.
    """

    title: str
    description: str
    technologies: list[FixtureSkill] = Field(default_factory=list)
    role: Optional[str] = None
    outcome_text: Optional[str] = None


class FixtureProjectCandidate(BaseModel):
    """
    One candidate in the Phase 2 corpus: employment AND project
    evidence, plus the (nullable, human-supplied) relevance record.
    Eligibility is NOT stored here -- it is computed on demand from
    to_profile() by the same frozen app.matching/app.scoring path the
    existing corpus uses, so it can never drift from how eligibility is
    computed everywhere else in this repository.
    """

    candidate_id: str
    stratum: ProjectStratum
    candidate_name: str = "Candidate"
    seniority: Optional[Seniority] = None
    skills: list[FixtureSkill] = Field(default_factory=list)
    total_experience_months: int = 0
    employment_history: list[FixtureEmployment] = Field(default_factory=list)
    projects: list[FixtureProject] = Field(default_factory=list)
    education: list[FixtureEducation] = Field(default_factory=list)
    relevance: CombinedRelevanceRecord = Field(default_factory=CombinedRelevanceRecord)

    def to_profile(self):
        """
        Builds the real app.schemas.CandidateProfile via the SAME
        conversion evaluation.schema.FixtureCandidate.to_profile()
        uses for skills/experience/education/employment, so this
        corpus's eligibility is computed by the identical, unmodified
        production path. projects are attached as CandidateProject
        objects, exactly mirroring app.candidate_extractor's output
        shape -- never read by app.matching/app.scoring (verified: see
        Phase 2 report).
        """
        from app.schemas import (
            CandidateEmployment,
            CandidateProfile,
            CandidateProject,
            CandidateSkill,
            EducationBackground,
            EducationRecord,
        )

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
            projects=[
                CandidateProject(
                    title=p.title, description=p.description,
                    technologies=[t.raw for t in p.technologies],
                    role=p.role, outcome_text=p.outcome_text,
                )
                for p in self.projects
            ],
            education=education,
            raw_text="(evaluation fixture; raw_text deliberately unused by matching)",
        )


class FixtureProjectJob(BaseModel):
    """Same shape as evaluation.schema.FixtureJob; kept local (not
    imported) so this module has one self-contained on-disk contract."""

    title: str
    seniority: Optional[Seniority] = None
    required_skills: list[FixtureSkill] = Field(default_factory=list)
    preferred_skills: list[FixtureSkill] = Field(default_factory=list)
    min_experience_months: Optional[int] = None
    max_experience_months: Optional[int] = None
    minimum_education_level: Optional[EducationLevel] = None
    education_required: bool = True
    responsibilities: list[str] = Field(default_factory=list)

    def to_profile(self):
        from app.schemas import (
            EducationRequirement,
            ExperienceRequirement,
            JobProfile,
            SkillRequirement,
        )

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


class FixtureProjectCase(BaseModel):
    """One JD plus every candidate labelled against it, for the Phase 2 corpus."""

    jd_id: str
    split: Literal["dev", "holdout"] = "dev"
    job: FixtureProjectJob
    candidates: list[FixtureProjectCandidate] = Field(default_factory=list)
    note: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        return bool(self.candidates) and all(c.relevance.is_labelled for c in self.candidates)

    def unlabelled_ids(self) -> list[str]:
        return [c.candidate_id for c in self.candidates if not c.relevance.is_labelled]


class ProjectDataset(BaseModel):
    cases: list[FixtureProjectCase] = Field(default_factory=list)

    def unlabelled(self) -> dict[str, list[str]]:
        return {
            case.jd_id: case.unlabelled_ids()
            for case in self.cases
            if case.unlabelled_ids()
        }


class UnlabelledProjectDatasetError(RuntimeError):
    """Raised when Phase 2 metrics are requested for data a human has not labelled."""


def require_labelled(dataset: ProjectDataset) -> None:
    missing = dataset.unlabelled()
    if missing:
        detail = "; ".join(f"{jd}: {', '.join(ids)}" for jd, ids in sorted(missing.items()))
        raise UnlabelledProjectDatasetError(
            f"Refusing to compute metrics: {sum(len(v) for v in missing.values())} "
            f"candidate(s) have no human labels -- {detail}"
        )


def load_project_dataset(directory: str | Path) -> ProjectDataset:
    """Loads every *.json in `directory` (sorted by filename). Files
    whose name starts with an underscore are ignored."""
    directory = Path(directory)
    cases: list[FixtureProjectCase] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_case in payload if isinstance(payload, list) else [payload]:
            cases.append(FixtureProjectCase.model_validate(raw_case))

    seen: set[str] = set()
    for case in cases:
        if case.jd_id in seen:
            raise ValueError(f"Duplicate jd_id in Phase 2 dataset: {case.jd_id!r}")
        seen.add(case.jd_id)

    return ProjectDataset(cases=cases)
