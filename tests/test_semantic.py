"""
Task 8B-1 tests: app.semantic (cosine similarity, SemanticEvidence
production, attachment beside MatchEvidence).

The invariance classes at the bottom are the important ones: they prove
8B-1 changed nothing about Task 7 structured matching or Task 8A
scoring. Entirely offline -- no Ollama, no network, no model pull.
"""
import pytest

from app.embeddings import CachingEmbeddingProvider, FakeEmbeddingProvider
from app.matching import build_match_evidence
from app.schemas import (
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
from app.scoring import score_match
from app.semantic import (
    attach_semantic_evidence,
    compute_semantic_evidence,
    cosine_similarity,
)

# Two texts about the same kind of work, and one about something else
# entirely. Similarity is always asserted RELATIVELY (similar > unrelated),
# never against a hardcoded threshold -- a threshold would be a magic
# number tuned to FakeEmbeddingProvider and would break the moment a real
# model arrives in 8B-2.
CANDIDATE_BACKEND = (
    "Built and operated backend payment services on distributed "
    "infrastructure, owning reliability and latency for high volume "
    "transaction processing."
)
JOB_BACKEND = (
    "Maintain and scale backend payment services across distributed "
    "infrastructure, improving reliability and latency for high volume "
    "transaction processing."
)
JOB_UNRELATED = (
    "Design seasonal window displays, style mannequins, and coordinate "
    "floral arrangements for retail storefronts."
)


class _RaisingProvider:
    model_id = "raising-v1"

    def is_available(self) -> bool:
        return True

    def embed(self, texts):
        raise RuntimeError("embedding backend exploded")


class _AvailabilityRaisingProvider:
    model_id = "availability-raising-v1"

    def is_available(self) -> bool:
        raise RuntimeError("cannot reach daemon")

    def embed(self, texts):  # pragma: no cover - never reached
        raise AssertionError("embed must not be called")


class _WrongCountProvider:
    model_id = "wrong-count-v1"

    def is_available(self) -> bool:
        return True

    def embed(self, texts):
        return [[1.0, 0.0]]


def _skill(raw, canonical=None, requirement_level="required"):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy", requirement_level=requirement_level,
    )


def _cskill(raw, canonical=None):
    return CandidateSkill(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution="taxonomy",
    )


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


def _rich_pair():
    candidate = _candidate(
        skills=[_cskill("Python", "python"), _cskill("PostgreSQL", "postgresql"),
                _cskill("Kafka"), _cskill("Docker", "docker")],
        total_experience_months=54,
        seniority=Seniority.SENIOR,
        education=EducationBackground(
            records=[EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                      level=EducationLevel.BACHELORS, resolution="taxonomy")],
            highest_level=EducationLevel.BACHELORS,
        ),
    )
    job = _job(
        title="Senior Backend Engineer",
        required_skills=[_skill("Python", "python"), _skill("PostgreSQL", "postgresql")],
        preferred_skills=[_skill("Kubernetes", "kubernetes", "preferred")],
        experience=ExperienceRequirement(min_months=36, is_specified=True),
        education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
        seniority=Seniority.SENIOR,
    )
    return candidate, job


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_are_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_result_is_clamped_to_unit_range(self):
        # A long identical vector accumulates float error that can push
        # the raw quotient just past 1.0.
        vector = [0.1] * 1000
        assert -1.0 <= cosine_similarity(vector, vector) <= 1.0

    def test_zero_vector_is_none_not_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) is None
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) is None
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) is None

    def test_empty_vectors_are_none(self):
        assert cosine_similarity([], []) is None

    def test_dimension_mismatch_is_none(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) is None

    def test_is_symmetric(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert cosine_similarity(a, b) == cosine_similarity(b, a)

    def test_does_not_use_numpy(self):
        import sys
        assert "numpy" not in sys.modules


class TestSimilarAndUnrelatedContexts:
    def test_similar_work_contexts_score_higher_than_unrelated(self):
        provider = FakeEmbeddingProvider()

        similar = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, provider)
        unrelated = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_UNRELATED, provider)

        assert similar.status == "pass"
        assert unrelated.status == "pass"
        assert similar.similarity_score > unrelated.similarity_score

    def test_identical_text_scores_at_the_maximum(self):
        provider = FakeEmbeddingProvider()

        evidence = compute_semantic_evidence(JOB_BACKEND, JOB_BACKEND, provider)

        assert evidence.similarity_score == pytest.approx(1.0)

    def test_score_is_within_unit_range(self):
        provider = FakeEmbeddingProvider()

        evidence = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_UNRELATED, provider)

        assert -1.0 <= evidence.similarity_score <= 1.0

    def test_successful_evidence_records_method_and_model(self):
        provider = FakeEmbeddingProvider()

        evidence = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, provider)

        assert evidence.method == "cosine"
        assert evidence.model_id == provider.model_id
        assert evidence.reason


class TestUnknownNeverZero:
    """
    Every unavailable/failed/degenerate path must be UNKNOWN with
    similarity_score None -- never "fail", never a similarity of 0.0.
    """

    def _assert_unknown(self, evidence):
        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert evidence.reason

    def test_no_provider(self):
        self._assert_unknown(compute_semantic_evidence("a text", "b text", None))

    def test_unavailable_provider(self):
        provider = FakeEmbeddingProvider(available=False)
        self._assert_unknown(compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, provider))

    def test_provider_that_raises_on_embed(self):
        self._assert_unknown(
            compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, _RaisingProvider())
        )

    def test_provider_that_raises_on_availability_check(self):
        self._assert_unknown(
            compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, _AvailabilityRaisingProvider())
        )

    def test_provider_returning_wrong_vector_count(self):
        self._assert_unknown(
            compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, _WrongCountProvider())
        )

    def test_empty_candidate_text(self):
        self._assert_unknown(compute_semantic_evidence("", JOB_BACKEND, FakeEmbeddingProvider()))

    def test_whitespace_only_candidate_text(self):
        self._assert_unknown(compute_semantic_evidence("   \n\t ", JOB_BACKEND, FakeEmbeddingProvider()))

    def test_empty_job_text(self):
        self._assert_unknown(compute_semantic_evidence(CANDIDATE_BACKEND, "", FakeEmbeddingProvider()))

    def test_zero_vector_text(self):
        # Punctuation-only text tokenizes to nothing -> zero vector ->
        # undefined similarity -> UNKNOWN (not 0.0).
        self._assert_unknown(
            compute_semantic_evidence("!!! ???", JOB_BACKEND, FakeEmbeddingProvider())
        )

    def test_unknown_is_never_fail(self):
        evidence = compute_semantic_evidence("a", "b", None)
        assert evidence.status != "fail"


class TestDeterminism:
    def test_same_input_and_provider_gives_identical_evidence(self):
        provider = FakeEmbeddingProvider()

        first = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, provider)
        second = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, provider)

        assert first.model_dump() == second.model_dump()
        assert first.model_dump_json() == second.model_dump_json()

    def test_separate_provider_instances_agree(self):
        first = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider())
        second = compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider())

        assert first.model_dump() == second.model_dump()

    def test_caching_does_not_change_the_result(self):
        uncached = compute_semantic_evidence(
            CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )
        cached_provider = CachingEmbeddingProvider(FakeEmbeddingProvider())
        compute_semantic_evidence(CANDIDATE_BACKEND, JOB_BACKEND, cached_provider)
        second_pass = compute_semantic_evidence(
            CANDIDATE_BACKEND, JOB_BACKEND, cached_provider
        )

        assert second_pass.similarity_score == uncached.similarity_score


class TestAttachReturnsACopy:
    def test_original_evidence_is_not_mutated(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        before = evidence.model_dump()

        attach_semantic_evidence(evidence, CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider())

        assert evidence.model_dump() == before
        assert evidence.semantic is None

    def test_returned_object_is_a_different_instance(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )

        assert attached is not evidence
        assert attached.semantic is not None

    def test_attachment_populates_semantic_evidence(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )

        assert attached.semantic.status == "pass"
        assert attached.semantic.similarity_score is not None

    def test_attachment_with_no_provider_still_returns_evidence(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_semantic_evidence(evidence, CANDIDATE_BACKEND, JOB_BACKEND, None)

        assert attached.semantic.status == "unknown"
        assert attached.eligibility == evidence.eligibility


class TestBuildMatchEvidenceStillReturnsNoneSemantic:
    """
    app.matching.build_match_evidence must remain pure and offline.
    8B-1 must not have made it produce semantic evidence.
    """

    def test_semantic_is_none_for_a_rich_pair(self):
        candidate, job = _rich_pair()
        assert build_match_evidence(candidate, job).semantic is None

    def test_semantic_is_none_for_an_empty_pair(self):
        assert build_match_evidence(_candidate(), _job()).semantic is None


class TestStructuredEvidenceIsUnchanged:
    """
    Attaching semantic evidence must change ONLY the `semantic` field.
    """

    def test_every_structured_field_is_identical_after_attachment(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )

        before = evidence.model_dump()
        after = attached.model_dump()
        assert before.pop("semantic") is None
        assert after.pop("semantic") is not None
        assert before == after

    def test_eligibility_and_hard_constraints_survive_attachment(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_UNRELATED, FakeEmbeddingProvider()
        )

        assert attached.eligibility == evidence.eligibility
        assert attached.hard_constraints == evidence.hard_constraints

    def test_a_wildly_different_similarity_cannot_move_structured_evidence(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        high = attach_semantic_evidence(
            evidence, JOB_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )
        low = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_UNRELATED, FakeEmbeddingProvider()
        )

        assert high.semantic.similarity_score != low.semantic.similarity_score
        high_dump, low_dump = high.model_dump(), low.model_dump()
        high_dump.pop("semantic")
        low_dump.pop("semantic")
        assert high_dump == low_dump


def _score_outputs(result):
    """
    The scoring outputs proper. MatchResult.evidence necessarily
    reflects whichever evidence was passed in (that is the point of
    embedding the evidence in the result), so the invariant 8B-1 must
    hold is that the SCORE -- weights_version, overall_score, and every
    component -- is untouched by semantic attachment.
    """
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestScoringIsUnaffected:
    """
    Task 8A is frozen. score_match reads five structured dimensions and
    never reads `semantic`, so attaching semantic evidence cannot move
    the score.
    """

    def test_score_is_identical_with_and_without_semantic(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        attached = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_BACKEND, FakeEmbeddingProvider()
        )

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(attached))

    def test_high_and_low_similarity_produce_the_same_score(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        high = attach_semantic_evidence(evidence, JOB_BACKEND, JOB_BACKEND, FakeEmbeddingProvider())
        low = attach_semantic_evidence(
            evidence, CANDIDATE_BACKEND, JOB_UNRELATED, FakeEmbeddingProvider()
        )

        assert _score_outputs(score_match(high)) == _score_outputs(score_match(low))

    def test_unknown_semantic_produces_the_same_score(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        unknown = attach_semantic_evidence(evidence, CANDIDATE_BACKEND, JOB_BACKEND, None)

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(unknown))

    def test_no_score_component_is_named_semantic(self):
        candidate, job = _rich_pair()
        attached = attach_semantic_evidence(
            build_match_evidence(candidate, job), CANDIDATE_BACKEND, JOB_BACKEND,
            FakeEmbeddingProvider(),
        )

        names = [c.name for c in score_match(attached).components]
        assert "semantic" not in names
        assert len(names) == 5
