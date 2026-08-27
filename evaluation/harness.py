"""
Evaluation harness (Task 8C-1).

Answers ONE question: does semantic evidence add ranking value beyond
the existing Task 8A structured score? It changes no production code,
adds no weights, and deliberately cannot: every scoring arm below is
computed HERE, from an unmodified MatchResult plus an unmodified
SemanticEvidence.

Offline by default (FakeEmbeddingProvider). A real-model run is an
explicit choice by the caller; the normal test suite never depends on
Ollama.


RANKING IS ELIGIBILITY-FIRST
--------------------------------------------------------------------------
app.scoring.score_match returns overall_score but MatchResult carries no
eligibility, and overall_score does not gate on it. Measured on a real
pair during 8C planning: a candidate MISSING a required skill
(eligibility="fail") and a candidate meeting every required skill
(eligibility="pass") both scored exactly 0.6923. Ranking on
overall_score alone would interleave hard-ineligible candidates with
eligible ones.

So every arm here ranks lexicographically:

    (eligibility_tier, arm_score)

with eligible candidates always above ineligible ones, whatever any
score says. This is a property of the HARNESS's ranking function, not a
production change -- whether MatchResult should gain the same guarantee
is a separate decision (D-C7) and is NOT implemented in 8C-1.

It is also what makes "semantic can never rescue a hard failure"
structural rather than a hope about weight magnitudes.


ARMS
--------------------------------------------------------------------------
    A  : 8A structured score only (the baseline)
    B  : semantic similarity only (reference point, never shippable)
    F0 : 8A only -- the null hypothesis, identical to A by construction
    F4 : eligibility -> 8A score -> semantic as pure tie-breaker
    F1 : semantic as an extra weighted component
    F2 : bounded: 8A score nudged by at most +/- delta
    F3 : gated: F1, but semantic applies only when eligibility passes
         and no structured dimension is UNKNOWN

No formula is chosen here. Weights/deltas passed in are EXPLORATORY
parameters for measurement, never a production default.


UNKNOWN NEUTRALITY
--------------------------------------------------------------------------
When semantic evidence is unavailable (no provider, provider failure,
empty narrative, missing narrative) every arm falls back to exactly the
8A score. Semantic UNKNOWN never becomes 0.0 and never becomes FAIL --
a missing signal must not be a penalty. With no embedding model at all,
every arm collapses to the baseline, which is the correct behaviour.
"""
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.embeddings import CachingEmbeddingProvider, EmbeddingProvider, FakeEmbeddingProvider
from app.matching import build_match_evidence
from app.schemas import MatchStatus
from app.scoring import DEFAULT_WEIGHTS, score_match
from app.semantic_match import compute_employment_semantic_evidence
from evaluation.schema import Dataset, FixtureCase, require_labelled

# Eligible candidates always outrank ineligible ones. "unknown" sits
# between: not proven ineligible, not proven eligible.
_ELIGIBILITY_TIER: dict[MatchStatus, int] = {"pass": 3, "partial": 2, "unknown": 1, "fail": 0}

DEFAULT_BOOTSTRAP_SAMPLES = 2000

DEFAULT_BOOTSTRAP_SEED = 8_0301

# Seed for the neutral tie-break permutation in rank(). Fixed and
# documented so a ranking is reproducible across runs and machines.
TIEBREAK_SEED = 80305

# Stand-in used ONLY where an arm's sort key needs a semantic slot for a
# candidate whose semantic evidence is UNKNOWN. It is a neutral
# mid-range placeholder, NOT a measurement and NOT a score: 0.0 would
# rank an unmeasurable candidate below one that genuinely scored 0.01,
# which is precisely the "missing signal treated as a negative signal"
# error the UNKNOWN contract exists to prevent.
NEUTRAL_SEMANTIC = 0.5


@dataclass
class Scored:
    """One candidate evaluated against one job, before any arm is applied."""

    jd_id: str
    candidate_id: str
    stratum: str
    relevance: int
    labelled_eligible: bool
    structured_score: float
    eligibility: MatchStatus
    semantic_score: Optional[float]
    semantic_status: MatchStatus
    has_unknown_dimension: bool

    @property
    def eligibility_tier(self) -> int:
        return _ELIGIBILITY_TIER[self.eligibility]


@dataclass
class CaseRanking:
    jd_id: str
    ranked: list[Scored]


ArmKey = Callable[[Scored], tuple]


def arm_baseline(s: Scored) -> tuple:
    """A / F0: eligibility, then the unmodified 8A score."""
    return (s.eligibility_tier, s.structured_score)


def arm_semantic_only(s: Scored) -> tuple:
    """B: eligibility, then semantic alone. Reference only."""
    return (
        s.eligibility_tier,
        s.semantic_score if s.semantic_score is not None else NEUTRAL_SEMANTIC,
    )


def arm_tiebreak(s: Scored) -> tuple:
    """
    F4: eligibility -> 8A score -> semantic. Semantic can only reorder
    candidates the 8A score leaves exactly tied, so it is structurally
    incapable of damaging a ranking 8A already gets right.

    An UNKNOWN semantic contributes NEUTRAL_SEMANTIC, never 0.0: a
    candidate whose narrative could not be embedded must not be ranked
    below one that was measured and scored badly. The key always has
    three elements so tuples are never compared at differing lengths
    (a shorter tuple would sort below an otherwise-equal longer one,
    silently penalising the UNKNOWN candidate again).
    """
    return (
        s.eligibility_tier,
        s.structured_score,
        s.semantic_score if s.semantic_score is not None else NEUTRAL_SEMANTIC,
    )


def make_arm_weighted(semantic_weight: float) -> ArmKey:
    """
    F1: semantic as a sixth weighted component, blended into the 8A
    score. Reproduces app.scoring's weighted-average shape without
    touching it: the 8A score already equals sum(contribution)/sum(w),
    so re-weighting is (score*W + sem*w)/(W+w).
    """
    total_structured_weight = sum(
        (
            DEFAULT_WEIGHTS.required_skills,
            DEFAULT_WEIGHTS.preferred_skills,
            DEFAULT_WEIGHTS.experience,
            DEFAULT_WEIGHTS.education,
            DEFAULT_WEIGHTS.seniority,
        )
    )

    def key(s: Scored) -> tuple:
        if s.semantic_score is None:
            return (s.eligibility_tier, s.structured_score)
        blended = (
            s.structured_score * total_structured_weight + s.semantic_score * semantic_weight
        ) / (total_structured_weight + semantic_weight)
        return (s.eligibility_tier, blended)

    return key


def make_arm_bounded(delta: float) -> ArmKey:
    """
    F2: the 8A score adjusted by at most +/- delta, where semantic is
    re-centred on 0.5 so an average similarity is neutral. Caps how far
    semantic can move any candidate.
    """

    def key(s: Scored) -> tuple:
        if s.semantic_score is None:
            return (s.eligibility_tier, s.structured_score)
        adjustment = max(-delta, min(delta, (s.semantic_score - 0.5) * 2 * delta))
        return (s.eligibility_tier, s.structured_score + adjustment)

    return key


def make_arm_gated(semantic_weight: float) -> ArmKey:
    """
    F3: F1, but semantic applies ONLY to candidates who pass every hard
    constraint and whose structured evidence has no UNKNOWN dimension.
    Semantic therefore never compensates for missing structured data --
    it only refines an already fully-known, eligible candidate.
    """
    weighted = make_arm_weighted(semantic_weight)

    def key(s: Scored) -> tuple:
        if s.eligibility != "pass" or s.has_unknown_dimension or s.semantic_score is None:
            return (s.eligibility_tier, s.structured_score)
        return weighted(s)

    return key


def default_arms() -> dict[str, ArmKey]:
    """
    Exploratory parameters only. These are measurement settings, NOT
    proposed production weights -- picking a shipping weight is 8C-2 and
    only if the decision gate allows it.
    """
    return {
        "A_baseline": arm_baseline,
        "B_semantic_only": arm_semantic_only,
        "F0_no_change": arm_baseline,
        "F4_tiebreak": arm_tiebreak,
        "F1_weighted_w1.0": make_arm_weighted(1.0),
        "F1_weighted_w2.0": make_arm_weighted(2.0),
        "F2_bounded_d0.05": make_arm_bounded(0.05),
        "F2_bounded_d0.10": make_arm_bounded(0.10),
        "F3_gated_w1.0": make_arm_gated(1.0),
    }


def score_case(
    case: FixtureCase, provider: Optional[EmbeddingProvider]
) -> list[Scored]:
    """
    Run the real, unmodified pipeline for one JD: build_match_evidence
    (Task 7) -> score_match (Task 8A) -> compute_employment_semantic_
    evidence (Task 8B-2b-i). Nothing here re-implements matching or
    scoring.
    """
    job = case.job.to_profile()
    rows: list[Scored] = []

    for fixture_candidate in case.candidates:
        candidate = fixture_candidate.to_profile()
        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)
        semantic = compute_employment_semantic_evidence(candidate, job, provider)

        rows.append(
            Scored(
                jd_id=case.jd_id,
                candidate_id=fixture_candidate.candidate_id,
                stratum=fixture_candidate.stratum,
                relevance=fixture_candidate.labels.relevance,
                labelled_eligible=fixture_candidate.labels.eligible,
                structured_score=result.overall_score,
                eligibility=evidence.eligibility,
                semantic_score=semantic.similarity_score,
                semantic_status=semantic.status,
                has_unknown_dimension=any(
                    c.status == "unknown" for c in result.components
                ),
            )
        )

    return rows


def _neutral_tiebreak(candidate_id: str) -> str:
    """
    Final tie-break for candidates an arm key cannot separate.

    WHY NOT candidate_id DIRECTLY
    ----------------------------------------------------------------------
    Sorting on the raw id was deterministic but NOT neutral. Ids in a
    fixture tend to encode the stratum ("-t1".."-t4", "-key1", "-mis1",
    "-ine1"), and stratum correlates with relevance -- so a lexical sort
    on the id is a sort on a proxy for the answer. Measured on the 12-JD
    corpus: 90 of 108 candidates (83.3%) shared a baseline key, and
    reverse-alphabetical id order placed them within 0.015 nDCG@5 of the
    BEST achievable ordering, inflating the baseline from ~0.68 to 0.94.
    That is a property of how the fixtures were named, not of any arm.

    Feeding the id through SHA-256 with a fixed seed destroys its lexical
    structure while keeping the permutation deterministic, reproducible
    across runs/machines/Python versions, and identical for every arm.

    The permutation depends on NOTHING but the id and the seed: no label,
    no relevance, no eligibility, no stratum, no structured or semantic
    score. It cannot encode the answer because it never sees it.
    """
    return hashlib.sha256(
        f"{TIEBREAK_SEED}:{candidate_id}".encode("utf-8")
    ).hexdigest()


def rank(rows: list[Scored], key: ArmKey) -> list[Scored]:
    """
    Deterministic descending ranking. A seeded neutral permutation of
    candidate_id is the final tie-break, so two candidates with identical
    keys never swap order between runs -- without it the report would
    differ run to run for reasons unrelated to the arm being measured --
    while the tie order itself carries no information about relevance.
    See _neutral_tiebreak.
    """
    return sorted(
        rows, key=lambda s: (key(s), _neutral_tiebreak(s.candidate_id)), reverse=True
    )


# METRICS


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked: list[Scored], k: int) -> Optional[float]:
    if not ranked:
        return None
    actual = dcg([s.relevance for s in ranked[:k]])
    ideal = dcg(sorted([s.relevance for s in ranked], reverse=True)[:k])
    if ideal == 0:
        return None
    return actual / ideal


def precision_at_k(ranked: list[Scored], k: int, threshold: int = 2) -> Optional[float]:
    if not ranked:
        return None
    top = ranked[:k]
    return sum(1 for s in top if s.relevance >= threshold) / len(top)


def eligibility_respect(ranked: list[Scored]) -> bool:
    """
    True when no candidate the human labelled ineligible is ranked above
    any candidate they labelled eligible. This is a hard guardrail: any
    arm that violates it is disqualified regardless of its other numbers.
    """
    seen_ineligible = False
    for row in ranked:
        if not row.labelled_eligible:
            seen_ineligible = True
        elif seen_ineligible:
            return False
    return True


def tied_pairs(rows: list[Scored], tolerance: float = 1e-9) -> list[tuple[Scored, Scored]]:
    """
    Pairs the 8A score cannot separate (identical within tolerance) but
    which the human ranked differently. These are exactly the cases
    semantic exists to resolve, and where the baseline is a coin flip.
    Pairs the human ranked equally are excluded: there is no correct
    answer to score.
    """
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if abs(a.structured_score - b.structured_score) <= tolerance:
                if a.relevance != b.relevance:
                    pairs.append((a, b))
    return pairs


def pairwise_accuracy(pairs: list[tuple[Scored, Scored]], key: ArmKey) -> Optional[float]:
    """
    Fraction of tied pairs the arm orders in agreement with the human
    labels. An arm that cannot separate a pair scores 0.5 for it -- a
    coin flip, neither rewarded nor punished.
    """
    if not pairs:
        return None

    correct = 0.0
    for a, b in pairs:
        ka, kb = key(a), key(b)
        if ka == kb:
            correct += 0.5
        else:
            arm_prefers_a = ka > kb
            human_prefers_a = a.relevance > b.relevance
            correct += 1.0 if arm_prefers_a == human_prefers_a else 0.0

    return correct / len(pairs)


def kendall_tau(ranked: list[Scored]) -> Optional[float]:
    """Rank correlation between the arm's order and the human labels."""
    n = len(ranked)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # ranked is already in arm order, so i precedes j.
            diff = ranked[i].relevance - ranked[j].relevance
            if diff > 0:
                concordant += 1
            elif diff < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


@dataclass
class ArmMetrics:
    arm: str
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    precision_at_3: Optional[float] = None
    kendall_tau: Optional[float] = None
    tied_pair_accuracy: Optional[float] = None
    tied_pair_count: int = 0
    eligibility_respected: bool = True
    eligibility_violations: list[str] = field(default_factory=list)
    per_jd_ndcg_at_5: dict[str, float] = field(default_factory=dict)
    per_jd_tied_accuracy: dict[str, float] = field(default_factory=dict)


def _mean(values: list[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def evaluate_arm(cases_rows: dict[str, list[Scored]], arm: str, key: ArmKey) -> ArmMetrics:
    metrics = ArmMetrics(arm=arm)
    ndcg5, ndcg10, p3, taus = [], [], [], []
    all_pairs_correct, all_pairs_total = 0.0, 0

    for jd_id, rows in sorted(cases_rows.items()):
        ranked = rank(rows, key)

        n5 = ndcg_at_k(ranked, 5)
        if n5 is not None:
            ndcg5.append(n5)
            metrics.per_jd_ndcg_at_5[jd_id] = n5
        n10 = ndcg_at_k(ranked, 10)
        if n10 is not None:
            ndcg10.append(n10)
        p = precision_at_k(ranked, 3)
        if p is not None:
            p3.append(p)
        tau = kendall_tau(ranked)
        if tau is not None:
            taus.append(tau)

        if not eligibility_respect(ranked):
            metrics.eligibility_respected = False
            metrics.eligibility_violations.append(jd_id)

        pairs = tied_pairs(rows)
        if pairs:
            accuracy = pairwise_accuracy(pairs, key)
            metrics.per_jd_tied_accuracy[jd_id] = accuracy
            all_pairs_correct += accuracy * len(pairs)
            all_pairs_total += len(pairs)

    metrics.ndcg_at_5 = _mean(ndcg5)
    metrics.ndcg_at_10 = _mean(ndcg10)
    metrics.precision_at_3 = _mean(p3)
    metrics.kendall_tau = _mean(taus)
    metrics.tied_pair_count = all_pairs_total
    metrics.tied_pair_accuracy = (
        all_pairs_correct / all_pairs_total if all_pairs_total else None
    )
    return metrics


def cluster_bootstrap_ci(
    per_jd_values: dict[str, float],
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> Optional[tuple[float, float]]:
    """
    Percentile bootstrap CI resampling JOB DESCRIPTIONS, not individual
    pairs.

    This is the whole point: candidates and pairs within one JD are
    correlated, so resampling pairs would treat ~30 correlated pairs as
    30 independent observations and report a CI several times narrower
    than reality. Resampling whole JDs keeps the cluster structure and
    yields an honest (much wider) interval.

    Seeded, so the same inputs always produce the same interval.
    """
    values = list(per_jd_values.values())
    if len(values) < 2:
        return None

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()

    tail = (1 - confidence) / 2
    lo = means[int(tail * samples)]
    hi = means[min(int((1 - tail) * samples), samples - 1)]
    return lo, hi


def build_provider(real: bool = False) -> EmbeddingProvider:
    """
    Offline FakeEmbeddingProvider by default. A real-model run must be
    requested explicitly, so importing or running the harness never
    requires Ollama.
    """
    if real:
        from app.ollama_embeddings import OllamaEmbeddingProvider

        return CachingEmbeddingProvider(OllamaEmbeddingProvider())
    return CachingEmbeddingProvider(FakeEmbeddingProvider())


def run(
    dataset: Dataset,
    split: str = "dev",
    provider: Optional[EmbeddingProvider] = None,
    arms: Optional[dict[str, ArmKey]] = None,
) -> dict[str, ArmMetrics]:
    """
    Score one split under every arm. Refuses to run on unlabelled data.

    `split` is "dev" or "holdout". The holdout is intended to be
    measured EXACTLY ONCE, after a formula has been chosen on dev --
    that discipline is the caller's responsibility and cannot be
    enforced in code.
    """
    require_labelled(dataset)

    cases = dataset.dev if split == "dev" else dataset.holdout
    if not cases:
        raise ValueError(f"No cases in split {split!r}.")

    provider = provider if provider is not None else build_provider(real=False)
    arms = arms if arms is not None else default_arms()

    cases_rows = {case.jd_id: score_case(case, provider) for case in cases}
    return {name: evaluate_arm(cases_rows, name, key) for name, key in arms.items()}
