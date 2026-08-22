"""
Real Ollama embedding provider (Task 8B-2b-ii, Phase A).

A THIN implementation of the frozen app.embeddings.EmbeddingProvider
Protocol, backed by the installed `ollama` Python client. It adds no
new abstraction: cosine similarity, caching, the UNKNOWN contract, and
per-employment aggregation all already exist in frozen modules
(app.semantic, app.embeddings.CachingEmbeddingProvider,
app.semantic_match) and are reused untouched.

This module is the ONLY place in the codebase that talks to an
embedding model. Everything above it is model-agnostic by construction:
swapping this class for FakeEmbeddingProvider changes nothing but the
numbers.


WHY THE `ollama` CLIENT AND NOT langchain_ollama.OllamaEmbeddings
--------------------------------------------------------------------------
model_id must carry the model DIGEST (see below), and OllamaEmbeddings
exposes no way to read it -- using it would mean opening a second,
separately-configured HTTP path purely to probe /api/tags. The `ollama`
client does both the embedding call and the digest probe over one
configured connection, with one timeout and one host setting. (app.llm
uses LangChain's ChatOllama because it needs structured-output
machinery; embedding needs none of that.)


NO I/O AT CONSTRUCTION
--------------------------------------------------------------------------
ollama.Client() builds an httpx client without connecting (measured:
~6ms against a dead port). Constructing this provider therefore never
touches the network, so importing or instantiating it is safe in an
offline test suite, at module import time, or on a machine with no
Ollama installed. Every network call happens in is_available() or
embed(), and only when actually invoked.


NEVER SUBSTITUTES A MODEL
--------------------------------------------------------------------------
If the configured embedding model is missing, this provider reports
unavailable and its embed() raises -- it does NOT silently fall back to
whatever model happens to be installed. qwen2.5:3b is a completion
model; its vectors would produce numbers that look like similarities
but mean nothing. A missing model must surface as UNKNOWN (which the
frozen compute_semantic_evidence does automatically), never as a
fabricated measurement.
"""
import os
from typing import Optional, Sequence

import ollama

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_TIMEOUT_SECONDS = 30.0

UNKNOWN_DIGEST = "unknown"

# Read at construction, not at import, so tests and callers can override
# per-instance. Deliberately NOT routed through app.config.Settings:
# that requires DATABASE_URL, and embedding must not become unusable
# just because no database is configured.
_MODEL_ENV_VAR = "EMBEDDING_MODEL"
_HOST_ENV_VAR = "OLLAMA_HOST"
_TIMEOUT_ENV_VAR = "EMBEDDING_TIMEOUT_SECONDS"


def _normalize_model_name(name: str) -> str:
    """
    Ollama reports an untagged model as "name:latest" but accepts
    "name" in requests, so the two spellings must compare equal when
    checking whether the configured model is installed.
    """
    return name[: -len(":latest")] if name.endswith(":latest") else name


class OllamaEmbeddingProvider:
    """
    Satisfies app.embeddings.EmbeddingProvider structurally (the
    Protocol is not subclassed -- it never needs to be).

    Configuration precedence: explicit constructor argument, then the
    corresponding environment variable, then the module default. An
    unparseable EMBEDDING_TIMEOUT_SECONDS falls back to the default
    rather than raising, since a malformed timeout must not prevent the
    application from starting.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[object] = None,
    ):
        self._model = model or os.getenv(_MODEL_ENV_VAR) or DEFAULT_EMBEDDING_MODEL
        self._host = host if host is not None else os.getenv(_HOST_ENV_VAR)

        if timeout is not None:
            self._timeout = timeout
        else:
            raw_timeout = os.getenv(_TIMEOUT_ENV_VAR)
            try:
                self._timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
            except ValueError:
                self._timeout = DEFAULT_TIMEOUT_SECONDS

        # Builds an httpx client; performs no network I/O. `client` is
        # an injection point for tests -- there is no other way to
        # exercise this class without a live daemon.
        self._client = client if client is not None else ollama.Client(
            host=self._host, timeout=self._timeout
        )

        self._resolved_digest: Optional[str] = None
        self._available: bool = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_id(self) -> str:
        """
        "<model>@<first 12 digest chars>", falling back to
        "<model>@unknown" when the digest cannot be read.

        Read by app.embeddings.cache_key for EVERY text, so it must be
        cheap and stable: a successful digest is memoized for the life
        of the instance and never re-probed. A FAILED resolution is not
        memoized, so a provider constructed while the daemon was down
        still reports its true digest once the daemon returns -- at the
        cost of a cache-key change at that moment, which costs a cache
        miss and nothing more (no vectors are cached while embedding is
        failing anyway).
        """
        if self._resolved_digest is None:
            self._resolve_digest()

        digest = self._resolved_digest or UNKNOWN_DIGEST
        return f"{self._model}@{digest}"

    def _installed_models(self) -> dict[str, str]:
        """Map of normalized model name -> digest, from the daemon."""
        response = self._client.list()
        installed: dict[str, str] = {}
        for entry in getattr(response, "models", []) or []:
            name = getattr(entry, "model", None)
            if not name:
                continue
            installed[_normalize_model_name(name)] = getattr(entry, "digest", "") or ""
        return installed

    def _resolve_digest(self) -> None:
        """
        Populate self._resolved_digest, memoizing ONLY on success.
        Never raises: an unresolvable digest degrades model_id to
        "@unknown", it does not break embedding.
        """
        try:
            installed = self._installed_models()
        except Exception:  # noqa: BLE001 -- model_id must never raise
            return

        digest = installed.get(_normalize_model_name(self._model))
        if digest:
            self._resolved_digest = digest[:12]

    def is_available(self) -> bool:
        """
        True only when the daemon responds AND the configured model is
        actually installed. Never raises -- a dead daemon
        (httpx.ConnectTimeout), a 404 (ollama.ResponseError), or
        anything else all resolve to False, which the frozen
        compute_semantic_evidence turns into UNKNOWN.

        A successful probe is memoized: app.semantic_match calls the
        frozen single-pair function once per employment record, and
        each of those calls checks availability, so without memoization
        an N-position résumé would trigger N round trips. A provider
        that dies after a successful probe still surfaces correctly --
        embed() fails and the frozen layer reports UNKNOWN. A FAILED
        probe is never memoized, so availability stays retryable.
        """
        if self._available:
            return True

        try:
            installed = self._installed_models()
        except Exception:  # noqa: BLE001 -- see docstring
            return False

        digest = installed.get(_normalize_model_name(self._model))
        if digest is None:
            return False

        if digest:
            self._resolved_digest = digest[:12]

        self._available = True
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        One vector per input text, in input order.

        Failures PROPAGATE rather than being swallowed: the frozen
        app.semantic.compute_semantic_evidence is the single place that
        converts an embedding failure into UNKNOWN, and duplicating
        that here would mean two places could disagree about what
        counts as a failure.

        A response whose vector count does not match the request is
        treated as a failure, not silently accepted -- returning fewer
        vectors than requested would misalign candidate and job
        embeddings and produce a similarity between the wrong pair of
        texts.
        """
        texts = list(texts)
        if not texts:
            return []

        response = self._client.embed(model=self._model, input=texts)
        vectors = getattr(response, "embeddings", None)

        if vectors is None:
            raise ValueError(
                f"Ollama returned no embeddings for model {self._model!r}."
            )

        vectors = [list(vector) for vector in vectors]

        if len(vectors) != len(texts):
            raise ValueError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} "
                f"texts (model {self._model!r})."
            )

        return vectors
