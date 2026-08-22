"""
Task 8B-2b-ii Category B: REAL-MODEL integration tests.

These require a running Ollama daemon AND the nomic-embed-text model.
The whole module is gated by a module-level skipif so a machine without
either simply SKIPS these -- the offline suite must never fail because
a model is not installed. Skipped is NOT passed; run pytest with -rs to
see the skip reason.

Everything here exercises the SAME frozen code paths the offline tests
use (app.semantic.compute_semantic_evidence, app.semantic_match,
app.embeddings.CachingEmbeddingProvider, app.scoring.score_match) --
only the provider differs. That is the point: the layers above are
model-agnostic by construction, and swapping FakeEmbeddingProvider for
a real model changes the numbers and nothing else.

The test that matters most is
TestSemanticQuality::test_paraphrase_with_low_lexical_overlap_scores_high:
it uses texts that mean the same thing while sharing almost no
vocabulary. A bag-of-words vectorizer (FakeEmbeddingProvider) cannot
score that pair highly; a real embedding model can. It is the concrete
evidence that the real model earns its place in the pipeline.
"""
import re

import pytest

from app.embeddings import CachingEmbeddingProvider, FakeEmbeddingProvider
from app.matching import build_match_evidence
from app.ollama_embeddings import DEFAULT_EMBEDDING_MODEL, OllamaEmbeddingProvider
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
    SkillRequirement,
)
from app.scoring import score_match
from app.semantic import compute_semantic_evidence, cosine_similarity
from app.semantic_match import (
    attach_employment_semantic_evidence,
    compute_employment_semantic_evidence,
)

EXPECTED_DIMENSIONS = 768


def _model_is_available() -> bool:
    """
    Cheap one-shot gate. Short timeout so collection never stalls when
    the daemon is down. Deliberately reuses the provider's own
    is_available() -- if that logic is wrong these tests skip rather
    than run, which is why app.ollama_embeddings.is_available() is also
    pinned against stubs in the offline tests/test_ollama_embeddings.py.
    """
    try:
        return OllamaEmbeddingProvider(timeout=2).is_available()
    except Exception:  # pragma: no cover - is_available already swallows
        return False


MODEL_AVAILABLE = _model_is_available()

pytestmark = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason=(
        f"Requires a running Ollama daemon with {DEFAULT_EMBEDDING_MODEL!r} "
        f"installed (ollama pull {DEFAULT_EMBEDDING_MODEL})."
    ),
)


# Texts. JOB_TEXT and PAYMENTS_NEAR_DUPLICATE share heavy vocabulary;
# PARAPHRASE means the same as JOB_TEXT while sharing almost none.
JOB_TEXT = "Maintain and scale backend payment services for high volume transaction processing"
PAYMENTS_NEAR_DUPLICATE = (
    "Built and operated backend payment services handling high volume transaction processing"
)
PARAPHRASE = (
    "Developed and maintained transaction-processing endpoints, improving throughput "
    "of the billing platform"
)
VISION_RESEARCH = (
    "Researched convolutional architectures for semantic segmentation of medical imagery"
)
FLORISTRY = (
    "Designed seasonal window displays and coordinated floral arrangements for weddings"
)


@pytest.fixture(scope="module")
def provider():
    return OllamaEmbeddingProvider()


def _similarity(prov, a, b):
    vectors = prov.embed([a, b])
    return cosine_similarity(vectors[0], vectors[1])


def _words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


class CountingProvider:
    """Wraps a provider and records every text actually sent to it."""

    def __init__(self, inner):
        self._inner = inner
        self.embedded = []

    @property
    def model_id(self):
        return self._inner.model_id

    def is_available(self):
        return self._inner.is_available()

    def embed(self, texts):
        self.embedded.extend(texts)
        return self._inner.embed(texts)


class TestAvailabilityAndIdentity:
    def test_model_is_available(self, provider):
        assert provider.is_available() is True

    def test_model_id_carries_a_real_digest(self, provider):
        model_id = provider.model_id

        assert model_id.startswith(f"{DEFAULT_EMBEDDING_MODEL}@")
        digest = model_id.split("@", 1)[1]
        assert digest != "unknown"
        assert len(digest) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", digest), digest

    def test_uninstalled_model_reports_unavailable_against_live_daemon(self):
        bogus = OllamaEmbeddingProvider(model="definitely-not-installed-xyz")

        assert bogus.is_available() is False

    def test_uninstalled_model_raises_on_embed_against_live_daemon(self):
        bogus = OllamaEmbeddingProvider(model="definitely-not-installed-xyz")

        with pytest.raises(Exception):
            bogus.embed(["text"])

    def test_uninstalled_model_yields_unknown_never_a_fabricated_score(self):
        bogus = OllamaEmbeddingProvider(model="definitely-not-installed-xyz")

        evidence = compute_semantic_evidence(JOB_TEXT, PARAPHRASE, bogus)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert evidence.status != "fail"


class TestRealEmbeddings:
    def test_dimensionality_is_768(self, provider):
        vectors = provider.embed([JOB_TEXT])

        assert len(vectors[0]) == EXPECTED_DIMENSIONS

    def test_dimensionality_is_consistent_across_texts(self, provider):
        vectors = provider.embed([JOB_TEXT, PARAPHRASE, FLORISTRY])

        assert {len(v) for v in vectors} == {EXPECTED_DIMENSIONS}

    def test_vectors_are_non_degenerate(self, provider):
        vector = provider.embed([JOB_TEXT])[0]

        assert any(component != 0.0 for component in vector)

    def test_repeated_embedding_is_reproducible_within_one_call(self, provider):
        first, second = provider.embed([JOB_TEXT, JOB_TEXT])

        assert first == second

    def test_repeated_embedding_is_reproducible_across_calls(self, provider):
        first = provider.embed([JOB_TEXT])[0]
        second = provider.embed([JOB_TEXT])[0]

        assert first == second

    def test_reproducible_across_provider_instances(self):
        first = OllamaEmbeddingProvider().embed([JOB_TEXT])[0]
        second = OllamaEmbeddingProvider().embed([JOB_TEXT])[0]

        assert first == second

    def test_different_texts_give_different_vectors(self, provider):
        a, b = provider.embed([JOB_TEXT, FLORISTRY])

        assert a != b


class TestSemanticQuality:
    """
    Relative ordering only -- never a hardcoded threshold. A threshold
    would encode one model's calibration into the test suite and break
    the moment the model changes.
    """

    def test_near_duplicate_scores_higher_than_unrelated(self, provider):
        near = _similarity(provider, JOB_TEXT, PAYMENTS_NEAR_DUPLICATE)
        unrelated = _similarity(provider, JOB_TEXT, FLORISTRY)

        assert near > unrelated

    def test_paraphrase_with_low_lexical_overlap_scores_high(self, provider):
        """
        THE test that distinguishes a real embedding model from a
        bag-of-words vectorizer: same meaning, almost no shared words.
        """
        shared = _words(JOB_TEXT) & _words(PARAPHRASE)
        meaningful_shared = shared - {"and", "of", "the", "for", "a", "to"}
        assert len(meaningful_shared) <= 3, f"texts share too much vocabulary: {shared}"

        paraphrase = _similarity(provider, JOB_TEXT, PARAPHRASE)
        vision = _similarity(provider, JOB_TEXT, VISION_RESEARCH)
        floristry = _similarity(provider, JOB_TEXT, FLORISTRY)

        assert paraphrase > vision
        assert paraphrase > floristry

    def test_full_ordering_is_sensible(self, provider):
        near = _similarity(provider, JOB_TEXT, PAYMENTS_NEAR_DUPLICATE)
        paraphrase = _similarity(provider, JOB_TEXT, PARAPHRASE)
        unrelated = _similarity(provider, JOB_TEXT, FLORISTRY)

        assert near > paraphrase > unrelated

    def test_identical_text_scores_near_one(self, provider):
        assert _similarity(provider, JOB_TEXT, JOB_TEXT) == pytest.approx(1.0, abs=1e-6)

    def test_scores_stay_in_unit_range(self, provider):
        for text in (PAYMENTS_NEAR_DUPLICATE, PARAPHRASE, VISION_RESEARCH, FLORISTRY):
            score = _similarity(provider, JOB_TEXT, text)
            assert -1.0 <= score <= 1.0


class TestFakeVersusReal:
    """
    Report-only comparison. Asserts ONLY the real model's claim; the
    fake's numbers are recorded for the report without a brittle
    inequality across two differently-calibrated similarity scales.
    """

    def test_real_model_separates_the_paraphrase_from_unrelated(self, provider):
        fake = FakeEmbeddingProvider()

        real_paraphrase = _similarity(provider, JOB_TEXT, PARAPHRASE)
        real_unrelated = _similarity(provider, JOB_TEXT, FLORISTRY)
        fake_paraphrase = _similarity(fake, JOB_TEXT, PARAPHRASE)
        fake_unrelated = _similarity(fake, JOB_TEXT, FLORISTRY)

        # The real model's separation is the assertion; the fake's is
        # printed for comparison only.
        assert real_paraphrase > real_unrelated

        print(
            f"\n  paraphrase/unrelated -- real: {real_paraphrase:.4f}/{real_unrelated:.4f}"
            f"  fake: {fake_paraphrase:.4f}/{fake_unrelated:.4f}"
        )

    def test_real_model_ranks_the_paraphrase_closer_to_the_near_duplicate(self, provider):
        near = _similarity(provider, JOB_TEXT, PAYMENTS_NEAR_DUPLICATE)
        paraphrase = _similarity(provider, JOB_TEXT, PARAPHRASE)
        unrelated = _similarity(provider, JOB_TEXT, FLORISTRY)

        assert abs(paraphrase - near) < abs(paraphrase - unrelated)


def _employment(company, responsibilities, duration_months=24):
    return CandidateEmployment(
        company=company, role="Engineer", start_date="Jan 2020", end_date="Jan 2022",
        duration_months=duration_months, responsibilities=responsibilities,
    )


def _candidate(employment_history, **overrides):
    defaults = dict(
        candidate_name="Jane Doe", skills=[], total_experience_months=24,
        total_experience_years=2.0, raw_text="resume text",
        employment_history=employment_history,
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(**overrides):
    defaults = dict(
        title="Engineer", required_skills=[], preferred_skills=[],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        raw_text="job text", responsibilities=[JOB_TEXT],
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _score_outputs(result):
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestFrozenSemanticLayer:
    def test_real_provider_produces_valid_semantic_evidence(self, provider):
        evidence = compute_semantic_evidence(PAYMENTS_NEAR_DUPLICATE, JOB_TEXT, provider)

        assert evidence.status == "pass"
        assert evidence.similarity_score is not None
        assert -1.0 <= evidence.similarity_score <= 1.0
        assert evidence.method == "cosine"
        assert evidence.model_id == provider.model_id
        assert evidence.reason


class TestOrchestrationWithRealModel:
    def test_per_employment_evidence_in_resume_order(self, provider):
        candidate = _candidate([
            _employment("Acme", [PAYMENTS_NEAR_DUPLICATE]),
            _employment("Globex", [VISION_RESEARCH]),
            _employment("Initech", [FLORISTRY]),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.status == "pass"
        assert [e.company for e in evidence.per_employment] == ["Acme", "Globex", "Initech"]
        assert evidence.aggregation == "max"

    def test_headline_is_the_best_matching_position(self, provider):
        candidate = _candidate([
            _employment("Floral", [FLORISTRY]),
            _employment("Payments", [PAYMENTS_NEAR_DUPLICATE]),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        scores = [e.similarity_score for e in evidence.per_employment]
        assert evidence.similarity_score == max(scores)
        assert evidence.per_employment[1].similarity_score > evidence.per_employment[0].similarity_score

    def test_model_id_propagates_into_aggregated_evidence(self, provider):
        candidate = _candidate([_employment("Acme", [PAYMENTS_NEAR_DUPLICATE])])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.model_id == provider.model_id
        assert evidence.model_id.startswith(f"{DEFAULT_EMBEDDING_MODEL}@")

    def test_job_text_is_embedded_once_across_positions(self, provider):
        counting = CountingProvider(provider)
        cached = CachingEmbeddingProvider(counting)
        candidate = _candidate([
            _employment("A", [PAYMENTS_NEAR_DUPLICATE]),
            _employment("B", [VISION_RESEARCH]),
            _employment("C", [FLORISTRY]),
        ])

        compute_employment_semantic_evidence(candidate, _job(), cached)

        assert counting.embedded.count(JOB_TEXT) == 1

    def test_caching_does_not_change_the_result(self, provider):
        candidate = _candidate([
            _employment("A", [PAYMENTS_NEAR_DUPLICATE]),
            _employment("B", [FLORISTRY]),
        ])

        uncached = compute_employment_semantic_evidence(candidate, _job(), provider)
        cached = compute_employment_semantic_evidence(
            candidate, _job(), CachingEmbeddingProvider(OllamaEmbeddingProvider())
        )

        assert uncached.similarity_score == pytest.approx(cached.similarity_score)

    def test_empty_responsibilities_are_skipped_not_scored_zero(self, provider):
        candidate = _candidate([
            _employment("Empty", []),
            _employment("Real", [PAYMENTS_NEAR_DUPLICATE]),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.per_employment[0].similarity_score is None
        assert evidence.per_employment[1].similarity_score is not None


class TestStructuredEvidenceAndScoringUnchanged:
    def _pair(self):
        candidate = _candidate(
            [_employment("Acme", [PAYMENTS_NEAR_DUPLICATE])],
            skills=[],
        )
        job = _job(
            required_skills=[SkillRequirement(
                raw="Python", match_key="python", canonical="python", category=None,
                resolution="taxonomy", requirement_level="required",
            )],
            experience=ExperienceRequirement(min_months=12, is_specified=True),
        )
        return candidate, job

    def test_only_semantic_differs_after_attachment(self, provider):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_employment_semantic_evidence(evidence, candidate, job, provider)

        before, after = evidence.model_dump(), attached.model_dump()
        assert before.pop("semantic") is None
        assert after.pop("semantic") is not None
        assert before == after

    def test_source_evidence_is_not_mutated(self, provider):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)

        attach_employment_semantic_evidence(evidence, candidate, job, provider)

        assert evidence.semantic is None

    def test_score_match_is_identical_with_and_without_semantic(self, provider):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        attached = attach_employment_semantic_evidence(evidence, candidate, job, provider)

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(attached))

    def test_high_and_low_similarity_produce_the_same_score(self, provider):
        job = _job()
        strong = _candidate([_employment("A", [PAYMENTS_NEAR_DUPLICATE])])
        weak = _candidate([_employment("A", [FLORISTRY])])

        strong_attached = attach_employment_semantic_evidence(
            build_match_evidence(strong, job), strong, job, provider
        )
        weak_attached = attach_employment_semantic_evidence(
            build_match_evidence(weak, job), weak, job, provider
        )

        assert strong_attached.semantic.similarity_score > weak_attached.semantic.similarity_score
        assert _score_outputs(score_match(strong_attached)) == _score_outputs(score_match(weak_attached))

    def test_no_score_component_is_named_semantic(self, provider):
        candidate, job = self._pair()
        attached = attach_employment_semantic_evidence(
            build_match_evidence(candidate, job), candidate, job, provider
        )

        names = [c.name for c in score_match(attached).components]
        assert "semantic" not in names
        assert len(names) == 5
