"""
Phase 4 tests: the project_evidence 6th scoring component
(app.scoring._project_evidence_status, and score_match's opt-in
inclusion of it).

Constructs ProjectEvidence/ProjectRelevanceSignal directly rather than
via app.project_relevance.compute_project_evidence -- scoring only
ever reads `evidence_depth`, never `similarity_score`, so no embedding
provider or LLM classifier is needed here; this keeps these tests
fast, offline, and focused purely on the scoring-layer contract.

Fixture builders mirror tests/test_scoring.py's _candidate()/_job().
"""
import pytest

from app.matching import build_match_evidence
from app.schemas import (
    CandidateProfile,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    MatchWeights,
    ProjectEvidence,
    ProjectRelevanceSignal,
    ProjectTechnologyOverlap,
    SkillRequirement,
)
from app.scoring import DEFAULT_WEIGHTS, DEFAULT_WEIGHTS_VERSION, score_match

DEFAULT_WEIGHTS_DUMP = DEFAULT_WEIGHTS.model_dump()


def _skill(raw, canonical=None):
    return SkillRequirement(raw=raw, match_key=raw.lower(), canonical=canonical,
                             category=None, resolution="taxonomy" if canonical else "unresolved",
                             requirement_level="required")


def _candidate(**overrides):
    defaults = dict(
        candidate_name="Jane Doe", skills=[], total_experience_months=0,
        total_experience_years=0.0, raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(**overrides):
    defaults = dict(
        title="Engineer", required_skills=[], preferred_skills=[],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _signal(depth, title="P"):
    return ProjectRelevanceSignal(
        title=title, technology_overlap=ProjectTechnologyOverlap(),
        evidence_depth=depth, similarity_score=None, similarity_status="unknown",
    )


def _project_evidence(*depths):
    """depths: evidence_depth values for however many projects."""
    return ProjectEvidence(
        per_project=[_signal(d, title=f"P{i}") for i, d in enumerate(depths)],
        best_similarity_score=None, method=None, model_id=None,
        status="pass" if depths else "unknown", reason="test fixture",
    )


def _weights(project_evidence):
    return MatchWeights(
        version="v-test-project-evidence",
        required_skills=DEFAULT_WEIGHTS.required_skills,
        preferred_skills=DEFAULT_WEIGHTS.preferred_skills,
        experience=DEFAULT_WEIGHTS.experience,
        education=DEFAULT_WEIGHTS.education,
        seniority=DEFAULT_WEIGHTS.seniority,
        project_evidence=project_evidence,
    )


def _names(result):
    return [c.name for c in result.components]


def _component(result, name):
    return next(c for c in result.components if c.name == name)


def _score_outputs(result):
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestDefaultWeightsUnchanged:
    def test_default_weights_five_original_values_are_byte_identical(self):
        assert DEFAULT_WEIGHTS.version == DEFAULT_WEIGHTS_VERSION == "v1"
        assert DEFAULT_WEIGHTS.required_skills == 2.0
        assert DEFAULT_WEIGHTS.preferred_skills == 1.0
        assert DEFAULT_WEIGHTS.experience == 1.5
        assert DEFAULT_WEIGHTS.education == 1.0
        assert DEFAULT_WEIGHTS.seniority == 1.0

    def test_default_weights_project_evidence_defaults_to_zero(self):
        assert DEFAULT_WEIGHTS.project_evidence == 0.0

    def test_default_weights_is_still_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_WEIGHTS.project_evidence = 1.0


class TestWeightZeroIsInert:
    """weight=0 (the DEFAULT_WEIGHTS behavior) must be byte-identical to
    pre-Phase-4 scoring, regardless of what project evidence exists."""

    def test_no_component_added_when_weight_is_zero(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, DEFAULT_WEIGHTS)

        assert "project_evidence" not in _names(result)
        assert len(result.components) == 5

    def test_overall_score_identical_with_and_without_project_evidence_at_weight_zero(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        weak = evidence.model_copy(update={"project_evidence": _project_evidence("title_only")})
        strong = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})
        none_attached = evidence

        a = score_match(weak, DEFAULT_WEIGHTS)
        b = score_match(strong, DEFAULT_WEIGHTS)
        c = score_match(none_attached, DEFAULT_WEIGHTS)

        # The SCORING output (weights_version, overall_score, components)
        # must be identical -- evidence.project_evidence itself is
        # deliberately different across a/b/c (that's the point of this
        # test), so a full model_dump() comparison would wrongly fail on
        # that untouched, unscored passenger field.
        assert _score_outputs(a) == _score_outputs(b) == _score_outputs(c)

    def test_explicit_weights_object_with_project_evidence_zero_also_inert(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, _weights(project_evidence=0.0))

        assert len(result.components) == 5
        assert "project_evidence" not in _names(result)


class TestWeightPositiveIncludesComponent:
    def test_component_added_when_weight_is_positive(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, _weights(project_evidence=1.0))

        assert "project_evidence" in _names(result)
        assert len(result.components) == 6

    def test_weights_version_is_traceable(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, _weights(project_evidence=1.0))

        assert result.weights_version == "v-test-project-evidence"


class TestDepthCategoryMapping:
    def _score(self, depths, weight=1.0):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence(*depths)})
        return score_match(attached, _weights(project_evidence=weight))

    def test_substantive_is_pass(self):
        result = self._score(["substantive"])
        component = _component(result, "project_evidence")
        assert component.status == "pass"
        assert component.raw_value == 1.0

    def test_tutorial_or_basic_is_partial(self):
        result = self._score(["tutorial_or_basic"])
        component = _component(result, "project_evidence")
        assert component.status == "partial"
        assert component.raw_value == 0.75

    def test_title_only_is_fail(self):
        result = self._score(["title_only"])
        component = _component(result, "project_evidence")
        assert component.status == "fail"
        assert component.raw_value == 0.0

    def test_no_projects_is_unknown_not_fail(self):
        result = self._score([])
        component = _component(result, "project_evidence")
        assert component.status == "unknown"
        assert component.raw_value == 0.5

    def test_project_evidence_never_attached_is_unknown_not_fail(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())  # project_evidence stays None

        result = score_match(evidence, _weights(project_evidence=1.0))

        component = _component(result, "project_evidence")
        assert component.status == "unknown"
        assert component.raw_value == 0.5

    def test_never_crashes_on_missing_evidence(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        score_match(evidence, _weights(project_evidence=1.0))  # must not raise

    def test_best_of_multiple_projects_wins(self):
        result = self._score(["title_only", "substantive", "tutorial_or_basic"])
        component = _component(result, "project_evidence")
        assert component.status == "pass"

    def test_weak_projects_alone_do_not_reach_pass(self):
        result = self._score(["title_only", "tutorial_or_basic"])
        component = _component(result, "project_evidence")
        assert component.status == "partial"


class TestContributionArithmetic:
    def test_contribution_equals_raw_value_times_weight(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, _weights(project_evidence=2.0))

        component = _component(result, "project_evidence")
        assert component.weight == 2.0
        assert component.contribution == pytest.approx(1.0 * 2.0)

    def test_positive_weight_can_move_overall_score(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        strong = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})
        weak = evidence.model_copy(update={"project_evidence": _project_evidence("title_only")})

        strong_result = score_match(strong, _weights(project_evidence=3.0))
        weak_result = score_match(weak, _weights(project_evidence=3.0))

        assert strong_result.overall_score > weak_result.overall_score


class TestEligibilityAndFiveComponentsUnaffected:
    """The hard requirement: attaching/scoring project evidence can
    never change eligibility, hard_constraints, or the original five
    components' values -- at weight=0 AND at weight>0."""

    def _pair(self):
        candidate = _candidate(total_experience_months=0)
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=24, is_specified=True),
        )
        return candidate, job

    def test_eligibility_unchanged_at_weight_zero(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, DEFAULT_WEIGHTS)

        assert result.evidence.eligibility == evidence.eligibility == "fail"
        assert result.evidence.hard_constraints == evidence.hard_constraints

    def test_eligibility_unchanged_at_weight_positive_even_with_strong_evidence(self):
        """A fresher with zero experience and 'substantive' project
        evidence must still be INELIGIBLE -- projects never manufacture
        experience or override a hard constraint, regardless of weight."""
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        result = score_match(attached, _weights(project_evidence=5.0))

        assert result.evidence.eligibility == "fail"
        assert result.evidence.experience.status == "fail"
        assert result.evidence.hard_constraints == evidence.hard_constraints

    def test_original_five_components_identical_with_and_without_project_evidence(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        bare_result = score_match(evidence, _weights(project_evidence=1.0))
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})
        attached_result = score_match(attached, _weights(project_evidence=1.0))

        original_names = ["required_skills", "preferred_skills", "experience", "education", "seniority"]
        bare_five = [c for c in bare_result.components if c.name in original_names]
        attached_five = [c for c in attached_result.components if c.name in original_names]

        assert [c.model_dump() for c in bare_five] == [c.model_dump() for c in attached_five]

    def test_total_experience_months_never_touched(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})
        score_match(attached, _weights(project_evidence=5.0))

        # candidate is the same object passed in -- confirm it was never mutated.
        assert candidate.total_experience_months == 0


class TestDeterminism:
    def test_repeated_calls_are_identical(self):
        candidate = _candidate()
        evidence = build_match_evidence(candidate, _job())
        attached = evidence.model_copy(update={"project_evidence": _project_evidence("substantive")})

        first = score_match(attached, _weights(project_evidence=1.0))
        second = score_match(attached, _weights(project_evidence=1.0))

        assert first.model_dump() == second.model_dump()
        assert first.model_dump_json() == second.model_dump_json()
