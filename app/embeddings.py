"""
Embedding provider abstraction (Task 8B-1).

Defines WHAT an embedding provider must do, ships one fully offline
deterministic implementation (FakeEmbeddingProvider), and a caching
decorator. Deliberately contains NO Ollama/network code: wiring a real
model is Task 8B-2, and this module is the seam that lets 8B-1 be
implemented, tested, and reviewed without pulling a model or making a
single network call.

Nothing here imports app.matching or app.scoring, and nothing here can
affect structured matching or scoring -- see app.semantic for how the
result is attached beside (never inside) MatchEvidence.


WHY hashlib AND NOT THE BUILT-IN hash()
--------------------------------------------------------------------------
FakeEmbeddingProvider derives vectors from text via hashlib.sha256, not
Python's built-in hash(). str.__hash__ is randomized per interpreter
process by PYTHONHASHSEED, so a hash()-based vectorizer would produce
different embeddings -- and therefore different similarity scores -- in
every new process. sha256 is stable across processes, machines, and
Python versions, which is what makes the determinism guarantee in
app.semantic ("same texts + same model identity -> same result")
actually true rather than merely usually true. tests/test_embeddings.py
::TestHashSeedIndependence pins this with real subprocesses.
"""
import hashlib
import re
from typing import Protocol, Sequence

DEFAULT_FAKE_DIMENSIONS = 256

DEFAULT_CACHE_MAXSIZE = 512


class EmbeddingProvider(Protocol):
    """
    The contract app.semantic depends on. Structural (typing.Protocol),
    not an ABC, so a real provider added in Task 8B-2 -- or a stub in a
    test -- satisfies it by shape without importing or subclassing
    anything from here.

    model_id must identify the model closely enough that two vectors
    carrying the same model_id are genuinely comparable (for a real
    provider: name plus version/digest, not just "ollama").

    is_available() must never raise; it answers "would embed() have a
    realistic chance of working right now" and is allowed to be
    cheap/optimistic. embed() MAY raise -- callers in app.semantic
    treat both an unavailable provider and a raising one identically,
    as UNKNOWN.
    """

    @property
    def model_id(self) -> str: ...

    def is_available(self) -> bool: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN_SPLIT = re.compile(r"[^a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase word split that keeps '+', '#', and '.' inside tokens so
    technology names survive as single tokens ("c++", "c#", "node.js")
    rather than being shredded into punctuation. Returns a LIST (not a
    set) -- order and repetition are both meaningful to the caller, and
    a set would introduce exactly the kind of iteration-order
    dependence this codebase keeps out of scored output.
    """
    return [token for token in _TOKEN_SPLIT.split(text.lower()) if token]


class FakeEmbeddingProvider:
    """
    A real, deterministic, fully offline embedding provider -- not a
    mock that returns canned vectors.

    Implements signed feature hashing (the "hashing trick"): every
    token is mapped by sha256 to one dimension and one sign, and
    accumulated. This is a genuine vectorizer, which matters for
    testing: texts sharing vocabulary produce genuinely high cosine
    similarity and texts with disjoint vocabulary produce genuinely
    near-zero similarity, so the similarity path is exercised for real
    rather than being asserted against hardcoded numbers.

    It is NOT semantically smart -- it has no notion that "Postgres"
    and "PostgreSQL" are related, which a real embedding model would.
    That is precisely the gap Task 8B-2 fills by swapping in a real
    provider behind the same protocol. What this class does provide is
    every property 8B-1 needs to verify: determinism, offline
    operation, empty/zero-vector handling, and cache behavior.

    `available=False` simulates a provider that cannot serve requests
    (model not pulled, daemon down) without needing a real outage.
    """

    def __init__(
        self,
        dimensions: int = DEFAULT_FAKE_DIMENSIONS,
        model_id: str = "fake-hashing-v1",
        available: bool = True,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._model_id = model_id
        self._available = available
        self.embed_call_count = 0
        self.embedded_texts: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def is_available(self) -> bool:
        return self._available

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Returns one vector per input text, in input order. Text with no
        tokens (empty or punctuation-only) yields an all-zero vector --
        deliberately not special-cased here: app.semantic treats a
        zero-norm vector as UNKNOWN, so the degenerate case is handled
        once, in one place, for every provider.
        """
        if not self._available:
            raise RuntimeError(
                f"FakeEmbeddingProvider({self._model_id}) is configured as unavailable."
            )
        self.embed_call_count += 1
        self.embedded_texts.extend(texts)
        return [self._embed_one(text) for text in texts]


def cache_key(model_id: str, text: str) -> str:
    """
    Content-addressed cache key. Includes model_id because the same
    text embedded by two different models is two different vectors that
    must never collide in one cache. The NUL separator prevents the
    boundary ambiguity a plain concatenation would allow (model "a" +
    text "bc" vs model "ab" + text "c").
    """
    payload = f"{model_id}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CachingEmbeddingProvider:
    """
    Wraps any EmbeddingProvider with a bounded, content-addressed,
    in-process cache, and satisfies EmbeddingProvider itself so callers
    cannot tell the difference.

    Purpose: comparing one job against N candidates re-embeds the same
    job text N times. This makes that one call instead.

    Bounded by maxsize with FIFO eviction (dicts preserve insertion
    order, so the oldest key is simply the first). FIFO rather than LRU
    on purpose: LRU would need to reorder on every read, and the win
    here is repeated re-embedding of the SAME few texts within one
    matching run, which FIFO already captures at a fraction of the
    complexity.

    This is an in-process cache ONLY -- no database, no pgvector, no
    disk. Persisting vectors is deliberately out of scope for 8B-1.

    Caching is transparent to results: a cache hit returns the exact
    vector the wrapped provider returned, so it cannot change any
    similarity score. tests/test_embeddings.py asserts both halves of
    that (identical vectors, and the wrapped provider called once).
    """

    def __init__(self, provider: EmbeddingProvider, maxsize: int = DEFAULT_CACHE_MAXSIZE):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._provider = provider
        self._maxsize = maxsize
        self._cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    def is_available(self) -> bool:
        return self._provider.is_available()

    def __len__(self) -> int:
        return len(self._cache)

    def _store(self, key: str, vector: list[float]) -> None:
        if key in self._cache:
            return
        while len(self._cache) >= self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embeds only the texts not already cached, then reassembles the
        results in the caller's original order. Duplicate texts within
        one call are embedded once.

        If the wrapped provider raises, nothing is cached and the
        exception propagates unchanged -- app.semantic converts it to
        UNKNOWN. A failed call must never poison the cache with a
        partial or fabricated vector.

        Every returned vector is a fresh list copy, so a caller that
        mutates a result cannot corrupt the cached vector (nor any
        other result in the same call, since duplicate texts would
        otherwise share one list object). Without this the cache would
        hand out references to its own storage and a single in-place
        edit by any consumer would silently change every later cache
        hit for that text.
        """
        keys = [cache_key(self.model_id, text) for text in texts]

        resolved: dict[str, list[float]] = {}
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, texts):
            if key in self._cache:
                self.hits += 1
                resolved[key] = self._cache[key]
            elif key not in resolved and key not in missing_keys:
                self.misses += 1
                missing_keys.append(key)
                missing_texts.append(text)

        if missing_texts:
            fetched = self._provider.embed(missing_texts)
            if len(fetched) != len(missing_texts):
                raise ValueError(
                    f"{type(self._provider).__name__}.embed returned "
                    f"{len(fetched)} vectors for {len(missing_texts)} texts."
                )
            for key, vector in zip(missing_keys, fetched):
                # resolved is keyed independently of the cache so this
                # call's own results are always returned in full, even
                # if maxsize is small enough that _store evicts one of
                # them before the loop below reads it back.
                resolved[key] = vector
                self._store(key, vector)

        return [list(resolved[key]) for key in keys]
