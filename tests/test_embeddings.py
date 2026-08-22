"""
Task 8B-1 tests: app.embeddings (provider protocol, FakeEmbeddingProvider,
content-addressed bounded cache).

Entirely offline -- no Ollama, no network, no model pull.
"""
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from app.embeddings import (
    CachingEmbeddingProvider,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    cache_key,
    tokenize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _RaisingProvider:
    """A provider that reports available but blows up on embed()."""

    model_id = "raising-v1"

    def __init__(self):
        self.embed_call_count = 0

    def is_available(self) -> bool:
        return True

    def embed(self, texts):
        self.embed_call_count += 1
        raise RuntimeError("embedding backend exploded")


class TestProtocolConformance:
    def test_fake_provider_satisfies_the_protocol(self):
        provider: EmbeddingProvider = FakeEmbeddingProvider()

        assert isinstance(provider.model_id, str) and provider.model_id
        assert provider.is_available() is True
        assert len(provider.embed(["hello"])) == 1

    def test_caching_provider_satisfies_the_protocol(self):
        provider: EmbeddingProvider = CachingEmbeddingProvider(FakeEmbeddingProvider())

        assert isinstance(provider.model_id, str) and provider.model_id
        assert provider.is_available() is True
        assert len(provider.embed(["hello"])) == 1


class TestTokenize:
    def test_keeps_technology_punctuation_inside_tokens(self):
        assert tokenize("C++ and C# and Node.js") == ["c++", "and", "c#", "and", "node.js"]

    def test_empty_text_produces_no_tokens(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize("!!! ??? ---") == []

    def test_returns_a_list_not_a_set(self):
        # Repetition must survive: a set would silently drop it and
        # would also introduce iteration-order dependence.
        assert tokenize("python python java") == ["python", "python", "java"]


class TestFakeEmbeddingProvider:
    def test_returns_one_vector_per_text_in_input_order(self):
        provider = FakeEmbeddingProvider(dimensions=64)

        vectors = provider.embed(["alpha", "beta", "gamma"])

        assert len(vectors) == 3
        assert all(len(v) == 64 for v in vectors)
        assert vectors[0] == provider.embed(["alpha"])[0]

    def test_is_deterministic_within_a_process(self):
        provider = FakeEmbeddingProvider()

        first = provider.embed(["backend services at scale"])
        second = provider.embed(["backend services at scale"])

        assert first == second

    def test_two_instances_agree(self):
        a = FakeEmbeddingProvider(dimensions=128)
        b = FakeEmbeddingProvider(dimensions=128)

        assert a.embed(["distributed systems"]) == b.embed(["distributed systems"])

    def test_empty_text_produces_a_zero_vector(self):
        provider = FakeEmbeddingProvider(dimensions=32)

        vector = provider.embed([""])[0]

        assert vector == [0.0] * 32

    def test_different_texts_produce_different_vectors(self):
        provider = FakeEmbeddingProvider()

        vectors = provider.embed(["kubernetes orchestration", "poetry and sonnets"])

        assert vectors[0] != vectors[1]

    def test_unavailable_provider_raises_on_embed(self):
        provider = FakeEmbeddingProvider(available=False)

        assert provider.is_available() is False
        with pytest.raises(RuntimeError):
            provider.embed(["anything"])

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError):
            FakeEmbeddingProvider(dimensions=0)


class TestCacheKey:
    def test_same_model_and_text_gives_same_key(self):
        assert cache_key("m1", "text") == cache_key("m1", "text")

    def test_different_model_gives_different_key(self):
        assert cache_key("m1", "text") != cache_key("m2", "text")

    def test_boundary_ambiguity_is_prevented(self):
        # Without a separator, ("a", "bc") and ("ab", "c") would collide.
        assert cache_key("a", "bc") != cache_key("ab", "c")


class TestCachingEmbeddingProvider:
    def test_repeated_text_is_embedded_only_once(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner)

        first = cached.embed(["shared job text"])
        second = cached.embed(["shared job text"])

        assert first == second
        assert inner.embed_call_count == 1

    def test_cache_hit_returns_identical_vectors(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner)

        uncached = FakeEmbeddingProvider().embed(["payments platform"])[0]
        cached.embed(["payments platform"])
        from_cache = cached.embed(["payments platform"])[0]

        assert from_cache == uncached

    def test_duplicates_within_one_call_are_embedded_once(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner)

        vectors = cached.embed(["same", "same", "same"])

        assert len(vectors) == 3
        assert vectors[0] == vectors[1] == vectors[2]
        assert inner.embedded_texts == ["same"]

    def test_results_keep_caller_order_with_mixed_hits_and_misses(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner)
        cached.embed(["b"])  # prime the cache with one text

        vectors = cached.embed(["a", "b", "c"])

        reference = FakeEmbeddingProvider()
        assert vectors == reference.embed(["a", "b", "c"])

    def test_cache_is_bounded_and_evicts(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner, maxsize=2)

        cached.embed(["one"])
        cached.embed(["two"])
        cached.embed(["three"])

        assert len(cached) <= 2

    def test_all_results_returned_even_when_call_exceeds_maxsize(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner, maxsize=1)

        vectors = cached.embed(["a", "b", "c"])

        reference = FakeEmbeddingProvider()
        assert vectors == reference.embed(["a", "b", "c"])

    def test_provider_failure_is_not_cached_and_propagates(self):
        inner = _RaisingProvider()
        cached = CachingEmbeddingProvider(inner)

        with pytest.raises(RuntimeError):
            cached.embed(["text"])

        assert len(cached) == 0

    def test_unavailability_is_delegated(self):
        cached = CachingEmbeddingProvider(FakeEmbeddingProvider(available=False))

        assert cached.is_available() is False

    def test_rejects_non_positive_maxsize(self):
        with pytest.raises(ValueError):
            CachingEmbeddingProvider(FakeEmbeddingProvider(), maxsize=0)

    def test_caller_mutation_cannot_corrupt_the_cache(self):
        """
        Returned vectors are defensive copies: the cache hands out its
        own storage to nobody. Without this, one in-place edit by any
        consumer would silently change every later cache hit.
        """
        inner = FakeEmbeddingProvider(dimensions=8)
        cached = CachingEmbeddingProvider(inner)
        pristine = FakeEmbeddingProvider(dimensions=8).embed(["tok"])[0]

        # 1. caller mutates the returned vector
        returned = cached.embed(["tok"])[0]
        returned[0] = 999.0

        # 2. the cached vector is unchanged
        assert cached._cache[cache_key(cached.model_id, "tok")] == pristine

        # 3. the next cache hit returns the original vector
        assert cached.embed(["tok"])[0] == pristine
        assert inner.embed_call_count == 1

    def test_duplicates_in_one_call_do_not_share_one_list(self):
        cached = CachingEmbeddingProvider(FakeEmbeddingProvider(dimensions=8))

        first, second = cached.embed(["dup", "dup"])
        first[0] = 999.0

        assert second[0] != 999.0


class TestHashSeedIndependence:
    """
    FakeEmbeddingProvider derives vectors from sha256, never Python's
    PYTHONHASHSEED-randomized built-in hash(). Verified with real
    subprocesses, since PYTHONHASHSEED only takes effect at interpreter
    startup and cannot be simulated in-process.
    """

    _SCRIPT = dedent(
        """
        from app.embeddings import FakeEmbeddingProvider
        from app.semantic import cosine_similarity

        provider = FakeEmbeddingProvider(dimensions=64)
        a, b = provider.embed([
            "built backend payment services on distributed infrastructure",
            "maintain payment platform services and backend infrastructure",
        ])
        print(a)
        print(b)
        print(repr(cosine_similarity(a, b)))
        """
    )

    def _run(self, hashseed):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        if hashseed is None:
            env.pop("PYTHONHASHSEED", None)
        else:
            env["PYTHONHASHSEED"] = str(hashseed)

        completed = subprocess.run(
            [sys.executable, "-c", self._SCRIPT],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    def test_vectors_and_similarity_identical_across_hash_seeds(self):
        outputs = {
            "seed_0": self._run(0),
            "seed_1": self._run(1),
            "seed_98765": self._run(98765),
            "random_a": self._run(None),
            "random_b": self._run(None),
        }

        assert len(set(outputs.values())) == 1, (
            f"embedding output differs across PYTHONHASHSEED runs: {outputs}"
        )


class TestNoNetworkDependency:
    def test_modules_import_without_ollama_or_network_libraries(self):
        """
        app.embeddings and app.semantic must not drag in the LLM stack.
        Importing app.llm constructs a ChatOllama client, and 8B-1 is
        explicitly offline -- a real provider arrives in 8B-2.
        """
        script = dedent(
            """
            import sys
            import app.embeddings
            import app.semantic
            forbidden = [m for m in ("app.llm", "langchain_ollama", "ollama")
                         if m in sys.modules]
            print(forbidden)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]", (
            f"8B-1 modules pulled in network/LLM modules: {completed.stdout}"
        )
