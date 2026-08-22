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
