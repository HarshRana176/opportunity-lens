"""
Phase 4 tests: app.project_relevance (project-relevance evidence layer).

Fully offline -- FakeEmbeddingProvider only, no Ollama, no network, no
model pull. The evidence-depth LLM classification step is exercised via
an injectable stub chain (see _StubDepthChain), never the real
app.llm.project_depth_chain, for the same reason test_semantic_match.py
never calls a real embedding model.
"""
import pytest

from app.embeddings import FakeEmbeddingProvider
from app.matching import build_match_evidence
from app.project_relevance import (
    attach_project_evidence,
    build_candidate_project_text,
    classify_evidence_depth,
    compute_project_evidence,
    project_technology_overlap,
)
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    CandidateProject,
    CandidateSkill,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    ProjectDepthClassification,
    SkillRequirement,
)
from app.scoring import score_match

PAYMENTS_JD = [
    "Build and maintain payment processing APIs.",
    "Work with PostgreSQL to model transaction data.",
]


class _StubDepthChain:
    """Injectable stand-in for app.llm.project_depth_chain."""

    def __init__(self, depth="substantive", raises=False):
        self.depth = depth
        self.raises = raises
        self.calls: list[str] = []

    def invoke(self, payload):
        self.calls.append(payload["project_text"])
        if self.raises:
            raise RuntimeError("boom")
        return ProjectDepthClassification(depth=self.depth)


def _skill(raw, canonical=None, level="required"):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy" if canonical else "unresolved", requirement_level=level,
    )


def _project(title="Payments Sync Tool", description="", technologies=None, outcome_text=None, role=None):
    return CandidateProject(
        title=title, description=description, technologies=technologies or [],
        role=role, outcome_text=outcome_text,
    )


def _candidate(projects=None, employment_history=None, total_experience_months=0, **overrides):
    defaults = dict(
        candidate_name="Jane Doe", skills=[], total_experience_months=total_experience_months,
        total_experience_years=round(total_experience_months / 12, 2), raw_text="resume text",
        employment_history=employment_history or [], projects=projects or [],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(responsibilities=None, required_skills=None, preferred_skills=None, **overrides):
    defaults = dict(
        title="Backend Engineer", required_skills=required_skills or [],
        preferred_skills=preferred_skills or [],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        raw_text="job text",
        responsibilities=list(PAYMENTS_JD) if responsibilities is None else responsibilities,
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _provider():
    return FakeEmbeddingProvider()


# ---------------------------------------------------------------- text


class TestTextConstruction:
    def test_uses_only_description_and_outcome(self):
        project = _project(
            title="SecretProjectTitle",
            description="Built a validation layer for incoming requests.",
            technologies=["FastAPI", "PostgreSQL"],
            outcome_text="Cut error rate by 40%.",
        )

        text, _ = build_candidate_project_text(project)

        assert "Built a validation layer" in text
        assert "Cut error rate by 40%" in text
        for leaked in ["SecretProjectTitle", "FastAPI", "PostgreSQL"]:
            assert leaked not in text

    def test_empty_description_and_outcome_is_empty_text(self):
        project = _project(description="")

        text, truncated = build_candidate_project_text(project)

        assert text == ""
        assert truncated is False

    def test_whitespace_only_description_is_empty_text(self):
        project = _project(description="   \n  ")

        text, _ = build_candidate_project_text(project)

        assert text == ""


# ---------------------------------------------------------------- depth


class TestEvidenceDepth:
    def test_title_only_is_deterministic_no_llm_call(self):
        project = _project(description="")
        classifier = _StubDepthChain(depth="substantive")

        depth = classify_evidence_depth(project, classifier)

        assert depth == "title_only"
        assert classifier.calls == []

    def test_whitespace_description_is_also_title_only(self):
        project = _project(description="   ")
        classifier = _StubDepthChain()

        depth = classify_evidence_depth(project, classifier)

        assert depth == "title_only"
        assert classifier.calls == []

    def test_non_empty_description_invokes_classifier(self):
        project = _project(description="Built a REST API for order tracking.")
        classifier = _StubDepthChain(depth="substantive")

        depth = classify_evidence_depth(project, classifier)

        assert depth == "substantive"
        assert len(classifier.calls) == 1
        assert "Built a REST API" in classifier.calls[0]

    def test_tutorial_classification_is_passed_through(self):
        project = _project(description="Followed an online course to build a to-do app.")
        classifier = _StubDepthChain(depth="tutorial_or_basic")

        depth = classify_evidence_depth(project, classifier)

        assert depth == "tutorial_or_basic"

    def test_classifier_exception_falls_back_to_weaker_conclusion(self):
        project = _project(description="Built something real.")
        classifier = _StubDepthChain(raises=True)

        depth = classify_evidence_depth(project, classifier)

        assert depth == "tutorial_or_basic"

    def test_no_classifier_injected_uses_default_chain_reference(self):
        # Does not invoke the real chain (no description -> deterministic
        # title_only path), just confirms the default wiring doesn't blow up.
        project = _project(description="")

        depth = classify_evidence_depth(project)

        assert depth == "title_only"


# ---------------------------------------------------------------- overlap


class TestTechnologyOverlap:
    def test_matches_by_canonical(self):
        project = _project(technologies=["python", "fastapi"])
        job = _job(required_skills=[_skill("Python", canonical="python")])

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == ["Python"]
        assert overlap.total_required == 1

    def test_matches_by_match_key_when_unresolved(self):
        project = _project(technologies=["Kafka"])
        job = _job(required_skills=[_skill("Kafka", canonical=None)])

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == ["Kafka"]

    def test_case_insensitive(self):
        project = _project(technologies=["PYTHON"])
        job = _job(required_skills=[_skill("python", canonical="python")])

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == ["python"]

    def test_no_overlap_is_empty(self):
        project = _project(technologies=["Go", "MySQL"])
        job = _job(required_skills=[_skill("Python", canonical="python")])

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == []
        assert overlap.total_required == 1

    def test_preferred_tracked_separately_from_required(self):
        project = _project(technologies=["docker"])
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            preferred_skills=[_skill("Docker", canonical="docker")],
        )

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == []
        assert overlap.matched_preferred == ["Docker"]

    def test_no_technologies_named_is_empty_overlap(self):
        project = _project(technologies=[])
        job = _job(required_skills=[_skill("Python", canonical="python")])

        overlap = project_technology_overlap(project, job)

        assert overlap.matched_required == []


# ---------------------------------------------------------------- compute


class TestComputeProjectEvidence:
    def test_title_only_project_is_reported_and_skipped_not_scored_zero(self):
        candidate = _candidate([_project(title="X", description="")])

        evidence = compute_project_evidence(candidate, _job(), _provider(), _StubDepthChain())

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "title_only"
        assert signal.similarity_score is None
        assert signal.similarity_status == "unknown"
        assert signal.skipped_reason

    def test_tutorial_project_gets_tutorial_depth_and_still_a_similarity_score(self):
        candidate = _candidate([
            _project(description="Followed a tutorial to build a payment API clone.")
        ])
        classifier = _StubDepthChain(depth="tutorial_or_basic")

        evidence = compute_project_evidence(candidate, _job(), _provider(), classifier)

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "tutorial_or_basic"
        assert signal.similarity_score is not None

    def test_strong_project_gets_substantive_depth_and_high_overlap(self):
        candidate = _candidate([
            _project(
                description="Designed and built a payment processing API, wrote PostgreSQL "
                            "migrations, and debugged a double-charge race condition.",
                technologies=["Python", "FastAPI", "PostgreSQL"],
                outcome_text="Reduced failed transactions by 30%.",
            )
        ])
        job = _job(required_skills=[
            _skill("Python", canonical="python"), _skill("FastAPI", canonical="fastapi"),
            _skill("PostgreSQL", canonical="postgresql"),
        ])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, job, _provider(), classifier)

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "substantive"
        assert signal.similarity_score is not None
        assert set(signal.technology_overlap.matched_required) == {"Python", "FastAPI", "PostgreSQL"}

    def test_adjacent_technology_case_substantive_but_zero_technology_overlap(self):
        """Real, substantive work on a DIFFERENT stack than the JD requires:
        evidence_depth and technology_overlap must stay independent signals."""
        candidate = _candidate([
            _project(
                description="Designed and built an order-matching API, wrote database "
                            "migrations, and debugged a race condition in concurrent updates.",
                technologies=["Go", "MySQL"],
            )
        ])
        job = _job(required_skills=[
            _skill("Python", canonical="python"), _skill("PostgreSQL", canonical="postgresql"),
        ])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, job, _provider(), classifier)

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "substantive"
        assert signal.technology_overlap.matched_required == []
        assert signal.similarity_score is not None

    def test_no_projects_is_unknown(self):
        candidate = _candidate([])

        evidence = compute_project_evidence(candidate, _job(), _provider(), _StubDepthChain())

        assert evidence.status == "unknown"
        assert evidence.best_similarity_score is None
        assert evidence.per_project == []

    def test_technology_overlap_and_depth_still_computed_when_job_has_no_responsibilities(self):
        candidate = _candidate([
            _project(description="Built a caching layer.", technologies=["Redis"])
        ])
        job = _job(responsibilities=[], required_skills=[_skill("Redis", canonical="redis")])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, job, _provider(), classifier)

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "substantive"
        assert signal.technology_overlap.matched_required == ["Redis"]
        assert signal.similarity_score is None
        assert signal.similarity_status == "unknown"
        assert evidence.status == "unknown"

    def test_best_similarity_is_the_maximum_across_projects(self):
        candidate = _candidate([
            _project(title="Weak", description="Built a mobile game with a friend."),
            _project(title="Strong", description="Built and deployed a payment processing API."),
        ])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, _job(), _provider(), classifier)

        scores = [p.similarity_score for p in evidence.per_project]
        assert evidence.best_similarity_score == max(scores)

    def test_order_is_preserved_not_sorted(self):
        candidate = _candidate([
            _project(title="A", description="Built a game."),
            _project(title="B", description="Built a payment API."),
        ])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, _job(), _provider(), classifier)

        assert [p.title for p in evidence.per_project] == ["A", "B"]

    def test_no_provider_is_unknown_but_depth_and_overlap_still_computed(self):
        candidate = _candidate([
            _project(description="Built a payment API.", technologies=["Python"])
        ])
        job = _job(required_skills=[_skill("Python", canonical="python")])
        classifier = _StubDepthChain(depth="substantive")

        evidence = compute_project_evidence(candidate, job, None, classifier)

        assert evidence.status == "unknown"
        signal = evidence.per_project[0]
        assert signal.evidence_depth == "substantive"
        assert signal.technology_overlap.matched_required == ["Python"]
        assert signal.similarity_score is None

    def test_method_version_is_recorded(self):
        candidate = _candidate([_project(description="Built something.")])

        evidence = compute_project_evidence(candidate, _job(), _provider(), _StubDepthChain())

        assert evidence.evidence_depth_method_version == "v1"


# ---------------------------------------------------------------- fresher


class TestFresherIsUnaffected:
    def test_zero_experience_and_rich_experience_give_identical_project_evidence(self):
        shared_project = _project(
            description="Designed and built a payment API, debugged a race condition.",
            technologies=["Python", "FastAPI"],
            outcome_text="Reduced errors by 30%.",
        )
        job = _job(required_skills=[_skill("Python", canonical="python")])
        classifier_a = _StubDepthChain(depth="substantive")
        classifier_b = _StubDepthChain(depth="substantive")

        fresher = _candidate([shared_project], total_experience_months=0, employment_history=[])
        experienced = _candidate(
            [shared_project], total_experience_months=96,
            employment_history=[CandidateEmployment(
                company="Acme", role="Senior Engineer", start_date="2018", end_date="2026",
                duration_months=96,
            )],
        )

        a = compute_project_evidence(fresher, job, _provider(), classifier_a)
        b = compute_project_evidence(experienced, job, _provider(), classifier_b)

        assert a.model_dump() == b.model_dump()

    def test_fresher_can_reach_substantive_depth_and_full_technology_overlap(self):
        candidate = _candidate(
            [_project(
                description="Designed and built a payment API from scratch, deployed it, "
                            "and debugged a concurrency bug.",
                technologies=["Python", "FastAPI"],
            )],
            total_experience_months=0, employment_history=[],
        )
        job = _job(required_skills=[
            _skill("Python", canonical="python"), _skill("FastAPI", canonical="fastapi"),
        ])

        evidence = compute_project_evidence(candidate, job, _provider(), _StubDepthChain(depth="substantive"))

        signal = evidence.per_project[0]
        assert signal.evidence_depth == "substantive"
        assert len(signal.technology_overlap.matched_required) == 2


# ---------------------------------------------------------------- eligibility/scoring


def _score_outputs(result):
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestScoringAndEligibilityUnaffected:
    def _pair(self, project_kwargs=None):
        project = _project(**(project_kwargs or dict(description="Built a payment API.")))
        candidate = _candidate([project], total_experience_months=0, employment_history=[])
        job = _job(required_skills=[_skill("Python", canonical="python")])
        return candidate, job

    def test_build_match_evidence_leaves_project_evidence_none(self):
        candidate, job = self._pair()

        assert build_match_evidence(candidate, job).project_evidence is None

    def test_attach_returns_a_copy_and_does_not_mutate_source(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        before = evidence.model_dump()

        attached = attach_project_evidence(
            evidence, candidate, job, _provider(), _StubDepthChain(depth="substantive")
        )

        assert evidence.model_dump() == before
        assert evidence.project_evidence is None
        assert attached is not evidence
        assert attached.project_evidence is not None

    def test_every_other_field_is_unchanged(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_project_evidence(
            evidence, candidate, job, _provider(), _StubDepthChain(depth="substantive")
        )

        before, after = evidence.model_dump(), attached.model_dump()
        assert before.pop("project_evidence") is None
        assert after.pop("project_evidence") is not None
        assert before == after

    def test_eligibility_is_unaffected_regardless_of_project_strength(self):
        weak_candidate, job = self._pair(project_kwargs=dict(description=""))
        strong_candidate, _ = self._pair(project_kwargs=dict(
            description="Designed and built a full payment platform, deployed to production.",
            technologies=["Python"],
        ))

        weak_evidence = build_match_evidence(weak_candidate, job)
        strong_evidence = build_match_evidence(strong_candidate, job)

        weak_attached = attach_project_evidence(
            weak_evidence, weak_candidate, job, _provider(), _StubDepthChain(depth="title_only")
        )
        strong_attached = attach_project_evidence(
            strong_evidence, strong_candidate, job, _provider(), _StubDepthChain(depth="substantive")
        )

        assert weak_attached.eligibility == weak_evidence.eligibility
        assert strong_attached.eligibility == strong_evidence.eligibility
        assert weak_attached.hard_constraints == weak_evidence.hard_constraints
        assert strong_attached.hard_constraints == strong_evidence.hard_constraints

    def test_score_is_identical_with_and_without_project_evidence(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        attached = attach_project_evidence(
            evidence, candidate, job, _provider(), _StubDepthChain(depth="substantive")
        )

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(attached))

    def test_strong_and_weak_project_evidence_score_the_same(self):
        weak_candidate, job = self._pair(project_kwargs=dict(description=""))
        strong_candidate, _ = self._pair(project_kwargs=dict(
            description="Designed and built a full payment platform, deployed to production.",
        ))

        weak_attached = attach_project_evidence(
            build_match_evidence(weak_candidate, job), weak_candidate, job,
            _provider(), _StubDepthChain(depth="title_only"),
        )
        strong_attached = attach_project_evidence(
            build_match_evidence(strong_candidate, job), strong_candidate, job,
            _provider(), _StubDepthChain(depth="substantive"),
        )

        assert weak_attached.project_evidence.per_project[0].evidence_depth != \
            strong_attached.project_evidence.per_project[0].evidence_depth
        assert _score_outputs(score_match(weak_attached)) == _score_outputs(score_match(strong_attached))

    def test_no_score_component_reflects_project_evidence(self):
        candidate, job = self._pair()
        attached = attach_project_evidence(
            build_match_evidence(candidate, job), candidate, job,
            _provider(), _StubDepthChain(depth="substantive"),
        )

        names = [c.name for c in score_match(attached).components]
        assert "project" not in names
        assert "project_evidence" not in names
        assert len(names) == 5


# ---------------------------------------------------------------- determinism


class TestDeterminism:
    def test_repeated_calls_are_identical(self):
        candidate = _candidate([
            _project(description="Built a payment API.", technologies=["Python"])
        ])
        job = _job(required_skills=[_skill("Python", canonical="python")])

        first = compute_project_evidence(candidate, job, _provider(), _StubDepthChain(depth="substantive"))
        second = compute_project_evidence(candidate, job, _provider(), _StubDepthChain(depth="substantive"))

        assert first.model_dump() == second.model_dump()
