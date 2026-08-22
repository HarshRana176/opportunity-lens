"""
Task 8B-2b-ii Phase A tests: app.ollama_embeddings (Category A only).

COMPLETELY OFFLINE. Every test injects a stub client, so no test here
contacts an Ollama daemon, requires nomic-embed-text to be pulled, or
touches the network. Tests that genuinely need a live model are
Category B and are deliberately NOT in this file -- they arrive in a
later step, gated so they skip when the model is absent.
"""
import time

import pytest

from app.embeddings import CachingEmbeddingProvider, EmbeddingProvider, cache_key
from app.matching import build_match_evidence
from app.ollama_embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaEmbeddingProvider,
)
from app.schemas import (
    CandidateEmployment,
    CandidateProfile,
    EducationRequirement,
    ExperienceRequirement,
    JobProfile,
)
from app.scoring import score_match
from app.semantic import compute_semantic_evidence
from app.semantic_match import (
    attach_employment_semantic_evidence,
    compute_employment_semantic_evidence,
)


class _Model:
    def __init__(self, model, digest):
        self.model = model
        self.digest = digest


class _ListResponse:
    def __init__(self, models):
        self.models = models


class _EmbedResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class StubClient:
    """
    Stands in for ollama.Client. Records what it was asked for so tests
    can assert the configured model actually reaches the wire, and can
    be configured to fail either call independently.
    """

    def __init__(
        self,
        models=((f"{DEFAULT_EMBEDDING_MODEL}:latest", "abcdef0123456789"),),
        list_error=None,
        embed_error=None,
        vectors=None,
        dimensions=8,
    ):
        self._models = [_Model(name, digest) for name, digest in models]
        self._list_error = list_error
        self._embed_error = embed_error
        self._vectors = vectors
        self._dimensions = dimensions
        self.list_call_count = 0
        self.embed_call_count = 0
        self.embed_models = []
        self.embed_inputs = []

    def list(self):
        self.list_call_count += 1
        if self._list_error is not None:
            raise self._list_error
        return _ListResponse(self._models)

    def embed(self, model, input):
        self.embed_call_count += 1
        self.embed_models.append(model)
        self.embed_inputs.append(list(input))
        if self._embed_error is not None:
            raise self._embed_error
        if self._vectors is not None:
            return _EmbedResponse(self._vectors)
        # Deterministic, distinct-per-text vectors.
        return _EmbedResponse([
            [float((len(t) + i) % 7) + 1.0 for i in range(self._dimensions)]
            for t in input
        ])


def _provider(**kwargs):
    client = kwargs.pop("client", None) or StubClient(**kwargs)
    return OllamaEmbeddingProvider(client=client), client


def ollama_response_error():
    """The real 404 ollama raises when a model is not pulled."""
    import ollama

    return ollama.ResponseError("model not found", 404)


class TestConstruction:
    def test_constructor_performs_no_io(self):
        # A dead host would block/raise if the constructor connected.
        start = time.time()
        provider = OllamaEmbeddingProvider(host="http://127.0.0.1:1", timeout=5)
        elapsed = time.time() - start

        assert provider is not None
        assert elapsed < 1.0

    def test_defaults_to_nomic_embed_text(self):
        provider, _ = _provider()

        assert provider.model == "nomic-embed-text"
        assert DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"

    def test_explicit_model_wins(self):
        provider = OllamaEmbeddingProvider(model="custom-embed", client=StubClient())

        assert provider.model == "custom-embed"

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "env-embed")

        provider = OllamaEmbeddingProvider(client=StubClient())

        assert provider.model == "env-embed"

    def test_explicit_model_beats_environment(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "env-embed")

        provider = OllamaEmbeddingProvider(model="explicit", client=StubClient())

        assert provider.model == "explicit"

    def test_malformed_timeout_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "not-a-number")

        provider = OllamaEmbeddingProvider(client=StubClient())

        assert provider._timeout == DEFAULT_TIMEOUT_SECONDS

    def test_no_database_url_required(self, monkeypatch):
        # Embedding config must not be coupled to DATABASE_URL.
        monkeypatch.delenv("DATABASE_URL", raising=False)

        provider = OllamaEmbeddingProvider(client=StubClient())

        assert provider.model_id


class TestProtocolCompatibility:
    def test_satisfies_the_embedding_provider_protocol(self):
        provider: EmbeddingProvider = OllamaEmbeddingProvider(client=StubClient())

        assert isinstance(provider.model_id, str) and provider.model_id
        assert provider.is_available() is True
        assert len(provider.embed(["hello"])) == 1

    def test_works_inside_the_frozen_caching_provider(self):
        provider, client = _provider()
        cached = CachingEmbeddingProvider(provider)

        first = cached.embed(["shared text"])
        second = cached.embed(["shared text"])

        assert first == second
        assert client.embed_call_count == 1


class TestModelIdAndDigest:
    def test_model_id_uses_first_twelve_digest_chars(self):
        provider, _ = _provider(models=((DEFAULT_EMBEDDING_MODEL, "abcdef0123456789xyz"),))

        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@abcdef012345"

    def test_digest_resolution_is_lazy(self):
        provider, client = _provider()

        assert client.list_call_count == 0
        _ = provider.model_id
        assert client.list_call_count == 1

    def test_successful_digest_is_memoized(self):
        provider, client = _provider()

        first = provider.model_id
        second = provider.model_id

        assert first == second
        assert client.list_call_count == 1

    def test_unknown_digest_fallback_when_list_raises(self):
        provider, _ = _provider(list_error=RuntimeError("daemon down"))

        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@unknown"

    def test_unknown_digest_fallback_when_model_absent(self):
        provider, _ = _provider(models=(("some-other-model", "ffffffffffff"),))

        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@unknown"

    def test_failed_digest_resolution_is_retryable(self):
        client = StubClient(list_error=RuntimeError("daemon down"))
        provider = OllamaEmbeddingProvider(client=client)

        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@unknown"

        # Daemon comes back.
        client._list_error = None
        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@abcdef012345"

    def test_model_id_never_raises(self):
        provider, _ = _provider(list_error=RuntimeError("boom"))

        assert isinstance(provider.model_id, str)

    def test_model_id_is_stable_for_cache_keys(self):
        provider, _ = _provider()

        keys = {cache_key(provider.model_id, "text") for _ in range(5)}

        assert len(keys) == 1


class TestAvailability:
    def test_available_when_daemon_and_model_present(self):
        provider, _ = _provider()

        assert provider.is_available() is True

    def test_unavailable_when_model_absent(self):
        provider, _ = _provider(models=(("qwen2.5:3b", "357c53fb659c"),))

        assert provider.is_available() is False

    def test_unavailable_when_daemon_down(self):
        provider, _ = _provider(list_error=ConnectionError("connect timeout"))

        assert provider.is_available() is False

    def test_unavailable_on_response_error(self):
        provider, _ = _provider(list_error=ollama_response_error())

        assert provider.is_available() is False

    @pytest.mark.parametrize("error", [RuntimeError("x"), ValueError("y"), KeyError("z"), TimeoutError()])
    def test_never_raises_for_any_exception(self, error):
        provider, _ = _provider(list_error=error)

        assert provider.is_available() is False

    def test_tag_normalization_latest_suffix(self):
        provider, _ = _provider(models=((f"{DEFAULT_EMBEDDING_MODEL}:latest", "abcdef012345"),))

        assert provider.is_available() is True

    def test_tag_normalization_untagged(self):
        provider, _ = _provider(models=((DEFAULT_EMBEDDING_MODEL, "abcdef012345"),))

        assert provider.is_available() is True

    def test_successful_probe_is_memoized(self):
        provider, client = _provider()

        for _ in range(5):
            assert provider.is_available() is True

        assert client.list_call_count == 1

    def test_failed_probe_stays_retryable(self):
        client = StubClient(list_error=ConnectionError("down"))
        provider = OllamaEmbeddingProvider(client=client)

        assert provider.is_available() is False
        assert provider.is_available() is False
        assert client.list_call_count == 2  # re-probed, not memoized

        client._list_error = None
        assert provider.is_available() is True

    def test_availability_probe_also_populates_digest(self):
        provider, client = _provider()

        provider.is_available()

        assert provider.model_id == f"{DEFAULT_EMBEDDING_MODEL}@abcdef012345"
        assert client.list_call_count == 1  # no second probe for the digest


class TestEmbed:
    def test_empty_input_returns_empty_without_calling_client(self):
        provider, client = _provider()

        assert provider.embed([]) == []
        assert client.embed_call_count == 0

    def test_returns_one_vector_per_text_in_order(self):
        provider, _ = _provider(dimensions=4)

        vectors = provider.embed(["alpha", "bb", "ccc"])

        assert len(vectors) == 3
        assert all(len(v) == 4 for v in vectors)

    def test_configured_model_reaches_the_client(self):
        provider = OllamaEmbeddingProvider(model="my-embed-model", client=StubClient())
        client = provider._client

        provider.embed(["text"])

        assert client.embed_models == ["my-embed-model"]

    def test_qwen_is_never_substituted(self):
        provider, client = _provider(models=(("qwen2.5:3b", "357c53fb659c"),))

        assert provider.is_available() is False
        provider.embed(["text"])

        assert client.embed_models == [DEFAULT_EMBEDDING_MODEL]
        assert "qwen2.5:3b" not in client.embed_models

    def test_vector_count_mismatch_raises(self):
        provider, _ = _provider(vectors=[[1.0, 2.0]])

        with pytest.raises(ValueError, match="1 embeddings for 2 texts"):
            provider.embed(["a", "b"])

    def test_empty_embeddings_response_raises(self):
        provider, _ = _provider(vectors=[])

        with pytest.raises(ValueError):
            provider.embed(["a"])

    def test_provider_failure_propagates(self):
        provider, _ = _provider(embed_error=RuntimeError("backend exploded"))

        with pytest.raises(RuntimeError, match="backend exploded"):
            provider.embed(["a"])

    def test_no_fabricated_vectors_on_failure(self):
        provider, _ = _provider(embed_error=ConnectionError("down"))

        with pytest.raises(ConnectionError):
            provider.embed(["a"])

    def test_input_texts_are_passed_through_unchanged(self):
        provider, client = _provider()

        provider.embed(["first text", "second text"])

        assert client.embed_inputs == [["first text", "second text"]]


class TestUnknownSemanticsThroughFrozenLayer:
    """
    The frozen app.semantic.compute_semantic_evidence is the ONLY place
    that converts an embedding failure into UNKNOWN. These pin that the
    real provider's failure modes flow through it unchanged.
    """

    def test_provider_exception_becomes_unknown(self):
        provider, _ = _provider(embed_error=RuntimeError("backend exploded"))

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert evidence.reason

    def test_unavailable_provider_becomes_unknown(self):
        provider, _ = _provider(models=(("qwen2.5:3b", "357c53fb659c"),))

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_daemon_down_becomes_unknown(self):
        provider, _ = _provider(
            list_error=ConnectionError("down"), embed_error=ConnectionError("down")
        )

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_unknown_is_never_fail_or_zero(self):
        provider, _ = _provider(embed_error=RuntimeError("boom"))

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status != "fail"
        assert evidence.similarity_score is None

    def test_working_provider_produces_a_score(self):
        provider, _ = _provider()

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status == "pass"
        assert evidence.similarity_score is not None
        assert evidence.model_id == f"{DEFAULT_EMBEDDING_MODEL}@abcdef012345"


def _employment(company, responsibilities, duration_months=24):
    return CandidateEmployment(
        company=company, role="Engineer", start_date="Jan 2020", end_date="Jan 2022",
        duration_months=duration_months, responsibilities=responsibilities,
    )


def _candidate(employment_history):
    return CandidateProfile(
        candidate_name="Jane Doe", skills=[], total_experience_months=24,
        total_experience_years=2.0, raw_text="resume text",
        employment_history=employment_history,
    )


def _job(responsibilities=("Scale payment services",)):
    return JobProfile(
        title="Engineer", required_skills=[], preferred_skills=[],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        raw_text="job text", responsibilities=list(responsibilities),
    )


def _score_outputs(result):
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestOrchestrationWithStubbedProvider:
    """Full 8B-2b-i orchestration driven by the real provider class."""

    def test_per_employment_evidence_is_produced(self):
        provider, _ = _provider()
        candidate = _candidate([
            _employment("Acme", ["Built payment services"]),
            _employment("Globex", ["Researched vision models"]),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.status == "pass"
        assert [e.company for e in evidence.per_employment] == ["Acme", "Globex"]
        assert evidence.aggregation == "max"

    def test_model_id_propagates_into_evidence(self):
        provider, _ = _provider()
        candidate = _candidate([_employment("Acme", ["Built payment services"])])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.model_id == f"{DEFAULT_EMBEDDING_MODEL}@abcdef012345"

    def test_cache_reuse_embeds_job_text_once(self):
        provider, client = _provider()
        cached = CachingEmbeddingProvider(provider)
        candidate = _candidate([
            _employment("A", ["Built payment services"]),
            _employment("B", ["Ran billing pipelines"]),
            _employment("C", ["Wrote reporting jobs"]),
        ])

        compute_employment_semantic_evidence(candidate, _job(), cached)

        embedded = [t for call in client.embed_inputs for t in call]
        assert embedded.count("Scale payment services") == 1

    def test_provider_failure_yields_unknown_not_zero(self):
        provider, _ = _provider(embed_error=RuntimeError("boom"))
        candidate = _candidate([_employment("Acme", ["Built payment services"])])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_structured_evidence_is_unchanged(self):
        provider, _ = _provider()
        candidate = _candidate([_employment("Acme", ["Built payment services"])])
        job = _job()
        evidence = build_match_evidence(candidate, job)

        attached = attach_employment_semantic_evidence(evidence, candidate, job, provider)

        before, after = evidence.model_dump(), attached.model_dump()
        assert before.pop("semantic") is None
        assert after.pop("semantic") is not None
        assert before == after

    def test_score_match_is_unchanged(self):
        provider, _ = _provider()
        candidate = _candidate([_employment("Acme", ["Built payment services"])])
        job = _job()
        evidence = build_match_evidence(candidate, job)
        attached = attach_employment_semantic_evidence(evidence, candidate, job, provider)

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(attached))

    def test_build_match_evidence_still_returns_none_semantic(self):
        candidate = _candidate([_employment("Acme", ["Built payment services"])])

        assert build_match_evidence(candidate, _job()).semantic is None


class TestNoNetworkRequired:
    def test_module_imports_without_a_daemon(self):
        import app.ollama_embeddings as module

        assert module.DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"

    def test_full_flow_against_a_dead_host_is_unknown_not_an_error(self):
        provider = OllamaEmbeddingProvider(host="http://127.0.0.1:1", timeout=1)

        evidence = compute_semantic_evidence("candidate text", "job text", provider)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
