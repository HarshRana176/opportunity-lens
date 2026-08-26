"""
Deterministic score layer for MatchEvidence (Task 8A).

Pure Python: no LLM calls, no I/O, no database, no network, no clock,
no randomness. Consumes only app.schemas.MatchEvidence (built by
app.matching.build_match_evidence) and app.schemas.MatchWeights --
this module does NOT import from app.matching and does NOT re-run any
matching decision; it only assigns numbers to decisions matching
already made.

Produces app.schemas.MatchResult: a fixed-order list of per-dimension
ScoreComponents plus one overall_score, a weighted average of each
component's status-derived raw_value. See MatchWeights and
ScoreComponent in app.schemas for the field-level contract.


WHY THIS MODULE REUSES evaluate_hard_constraints's verdict FOR
required_skills INSTEAD OF RECOMPUTING IT
--------------------------------------------------------------------------
app.matching.evaluate_hard_constraints already defines "required_skills
satisfied" (vacuously PASS when a job states no required skills, else
PASS only if every required skill matched, else FAIL) as part of the
eligibility gate. Recomputing that rule here, separately, would let the
two definitions silently drift apart after a future change to one but
not the other. _required_skills_status() below instead reads the
existing HardConstraint entry straight off MatchEvidence.hard_constraints
-- the eligibility gate and the required-skills score component are
therefore always the same fact, by construction, not by two people
remembering to update two functions in sync.

experience and education status are read directly off
MatchEvidence.experience.status / .education.status instead -- NOT the
(remapped) HardConstraint entries for those two dimensions.
evaluate_hard_constraints deliberately remaps ExperienceEvidence's
"partial" (over-qualified) to a pass-equivalent "for the hard-
constraint entry specifically" (see its docstring): that remap encodes
an ELIGIBILITY judgment ("over-qualification must never block
eligibility"), not a QUALITY judgment. For scoring, an over-qualified
candidate is still reported as "partial" (raw_value 0.75, not 1.0) so
overall_score can distinguish an exact experience fit from an
over-qualified one -- a distinction the eligibility gate intentionally
discards but scoring intentionally keeps.

preferred_skills has no existing single-status representation on
MatchEvidence (SkillEvidence only carries matched/total counts, since
a missing preferred skill was never a pass/fail decision the matching
layer needed to make) -- _preferred_skills_status() derives one status
value from those counts, for scoring purposes only.


PHASE 4 ADDITION -- project_evidence AS A SIXTH, OPT-IN COMPONENT
--------------------------------------------------------------------------
MatchWeights.project_evidence defaults to 0.0, and score_match only
builds/scores a "project_evidence" ScoreComponent when a caller
supplies a MatchWeights with this field > 0 -- DEFAULT_WEIGHTS (v1)
never does, so every existing caller's components list and
overall_score are byte-identical to before this addition. This is not
a special case bolted onto the arithmetic: with the component simply
absent from `components`, the existing `sum(contribution)/sum(weight)`
formula already treats it as if it never existed.

_project_evidence_status() reads MatchEvidence.project_evidence (an
app.schemas.ProjectEvidence, populated only by
app.project_relevance.attach_project_evidence -- app.matching.
build_match_evidence always leaves it None) and reduces it to the same
four-state MatchStatus vocabulary every other dimension uses, via the
best (max) evidence_depth across the candidate's projects:

    substantive        -> "pass"     (raw_value 1.0)
    tutorial_or_basic   -> "partial"  (raw_value 0.75)
    title_only          -> "fail"     (raw_value 0.0)
    no projects, or
    project_evidence
    never attached      -> "unknown"  (raw_value 0.5)

The "unknown" branch is load-bearing, not a fallback of convenience: a
candidate with no projects, or one whose project extraction failed, or
one a caller simply never ran attach_project_evidence on, must never
be scored identically to a candidate who HAS a project that was judged
title_only/near-worthless -- exactly the missing-signal-is-not-a-
negative-signal contract every other UNKNOWN in this codebase already
honors (see e.g. app.matching.match_experience, app.semantic).

THIS MAPPING IS A PRODUCTION POLICY DECISION, NOT AN EXPERIMENTALLY
PROVEN CALIBRATION. The Phase 4 evaluation (evaluation/labeling_
projects/, read-only, never imported by this module) measured
best_depth_ordinal against frozen human project_relevance labels and
found it correlates strongly (Spearman ~0.92 on the agreed-only
subset) and separates title_only/tutorial_or_basic/substantive
cleanly -- which is why THIS module maps the three categories to three
different MatchStatus values instead of collapsing them. It did NOT
prove that pass=1.0/partial=0.75/fail=0.0 are the correct numeric
raw_values for this component, and tutorial_or_basic's mapping to
"partial" in particular is a judgment call flagged as worth revisiting
(that evaluation's tutorial_or_basic candidates had a mean human label
of 0.875 out of 3, closer to "weak" than to a 0.75 partial-pass) --
recorded here for whoever next recalibrates this mapping, not silently
smoothed over.


DETERMINISM
--------------------------------------------------------------------------
Every value scoring reads is either an int (a count), a float (a
weight), or a MatchStatus string -- all read via fixed named
attributes, never via dict/set iteration. `components` is always built
as a literal 5-element list in a fixed, hardcoded order; overall_score
is a sum()/division over that list, which iterates a list (order-
preserving), never a set or a dict. There is no dict or set anywhere in
this module. Consequently score_match(evidence, weights) is a pure
function of its two arguments: identical inputs, in the same or a
different process, under any PYTHONHASHSEED, produce a bytewise-
identical MatchResult. See tests/test_scoring.py::TestReproducibility.
"""
from app.schemas import MatchEvidence, MatchResult, MatchStatus, MatchWeights, ScoreComponent

DEFAULT_WEIGHTS_VERSION = "v1"

DEFAULT_WEIGHTS = MatchWeights(
    version=DEFAULT_WEIGHTS_VERSION,
    required_skills=2.0,
    preferred_skills=1.0,
    experience=1.5,
    education=1.0,
    seniority=1.0,
)

# Deliberately not exposed as configurable in Task 8A -- there is no
# stated requirement yet for per-deployment tuning of the status->raw
# mapping itself (as opposed to the per-dimension weights, which
# MatchWeights already makes configurable). Revisit if/when a real need
# appears rather than adding the knob speculatively.
_STATUS_RAW_VALUE: dict[MatchStatus, float] = {
    "pass": 1.0,
    "partial": 0.75,
    "unknown": 0.5,
    "fail": 0.0,
}


def _required_skills_status(evidence: MatchEvidence) -> MatchStatus:
    for constraint in evidence.hard_constraints:
        if constraint.kind == "required_skills":
            return constraint.status
    raise ValueError(
        "evidence.hard_constraints is missing its required_skills entry -- "
        "app.matching.evaluate_hard_constraints always includes all three "
        "kinds, so this indicates a MatchEvidence built outside that path."
    )


def _preferred_skills_status(evidence: MatchEvidence) -> MatchStatus:
    skills = evidence.skills
    if skills.total_preferred == 0:
        return "pass"
    if skills.matched_preferred == skills.total_preferred:
        return "pass"
    if skills.matched_preferred == 0:
        return "fail"
    return "partial"


# Ordinal ordering used only to pick the BEST project when a candidate
# has more than one -- mirrors app.project_relevance's own max-
# aggregation philosophy (a candidate's single strongest piece of
# project evidence, not an average dragged down by unrelated ones).
_PROJECT_DEPTH_ORDER = {"title_only": 0, "tutorial_or_basic": 1, "substantive": 2}

# See this module's docstring ("PHASE 4 ADDITION") for the full
# rationale and the explicit caveat that this is a policy choice, not
# an experimentally-proven calibration.
_PROJECT_DEPTH_STATUS: dict[str, MatchStatus] = {
    "substantive": "pass",
    "tutorial_or_basic": "partial",
    "title_only": "fail",
}


def _project_evidence_status(evidence: MatchEvidence) -> MatchStatus:
    """
    UNKNOWN whenever there is no project evidence to judge -- no
    ProjectEvidence was ever attached (evidence.project_evidence is
    None, e.g. a caller that opted into the project_evidence weight
    without calling app.project_relevance.attach_project_evidence
    first), or the candidate has zero projects. NEVER "fail" in that
    case: absence of evidence is not evidence of weak evidence. See
    this module's "PHASE 4 ADDITION" docstring section.
    """
    project_evidence = evidence.project_evidence
    if project_evidence is None or not project_evidence.per_project:
        return "unknown"

    best_depth = max(
        (signal.evidence_depth for signal in project_evidence.per_project),
        key=lambda depth: _PROJECT_DEPTH_ORDER[depth],
    )
    return _PROJECT_DEPTH_STATUS[best_depth]


def _component(name: str, status: MatchStatus, weight: float) -> ScoreComponent:
    raw_value = _STATUS_RAW_VALUE[status]
    return ScoreComponent(
        name=name,
        status=status,
        weight=weight,
        raw_value=raw_value,
        contribution=raw_value * weight,
    )


def score_match(evidence: MatchEvidence, weights: MatchWeights = DEFAULT_WEIGHTS) -> MatchResult:
    """
    Scores an existing MatchEvidence against a MatchWeights. Does not
    build or alter evidence -- call app.matching.build_match_evidence
    first.

    overall_score is the weighted average of each component's
    raw_value, i.e. sum(raw_value * weight) / sum(weight). Since every
    raw_value is in [0.0, 1.0] and every weight is non-negative
    (MatchWeights enforces `ge=0`), overall_score is always in
    [0.0, 1.0] as a direct consequence of being a convex combination --
    no separate clamping is needed. If every weight is 0 (a degenerate
    weight set with nothing enabled), overall_score is defined as 0.0
    rather than dividing by zero.
    """
    components = [
        _component("required_skills", _required_skills_status(evidence), weights.required_skills),
        _component("preferred_skills", _preferred_skills_status(evidence), weights.preferred_skills),
        _component("experience", evidence.experience.status, weights.experience),
        _component("education", evidence.education.status, weights.education),
        _component("seniority", evidence.seniority.status, weights.seniority),
    ]

    # Phase 4: strictly opt-in. Appended only when a caller explicitly
    # supplies weights.project_evidence > 0 -- DEFAULT_WEIGHTS never
    # does, so the components list and overall_score stay byte-
    # identical to pre-Phase-4 behavior for every existing caller. See
    # this module's "PHASE 4 ADDITION" docstring section.
    if weights.project_evidence > 0:
        components.append(
            _component("project_evidence", _project_evidence_status(evidence), weights.project_evidence)
        )

    total_weight = sum(component.weight for component in components)
    if total_weight > 0:
        overall_score = sum(component.contribution for component in components) / total_weight
    else:
        overall_score = 0.0

    return MatchResult(
        evidence=evidence,
        weights_version=weights.version,
        overall_score=overall_score,
        components=components,
    )
