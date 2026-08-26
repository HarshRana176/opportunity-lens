"""
Task 8C-1 tests: the evaluation harness itself.

These test the MEASURING INSTRUMENT, never a conclusion about semantic
similarity. There is deliberately no test asserting that any arm beats
any other -- that question is answered by labelled data against
pre-registered thresholds, not by a test the implementer wrote.

Fully offline: FakeEmbeddingProvider only, no Ollama, no network.
"""
import json

import pytest

from app.embeddings import FakeEmbeddingProvider
from evaluation.harness import (
    Scored,
    arm_baseline,
    arm_tiebreak,
    cluster_bootstrap_ci,
    default_arms,
    dcg,
    eligibility_respect,
    evaluate_arm,
    kendall_tau,
    make_arm_bounded,
    make_arm_gated,
    make_arm_weighted,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    rank,
    run,
    score_case,
    tied_pairs,
)
from evaluation.schema import (
    Dataset,
    FixtureCase,
    UnlabelledDatasetError,
    load_dataset,
    require_labelled,
)

TEMPLATE_PATH = "evaluation/datasets/_TEMPLATE.json"


def _scored(cid, structured, relevance, eligible=True, semantic=None,
            eligibility="pass", semantic_status="pass", unknown_dim=False, jd="jd-1"):
    return Scored(
        jd_id=jd, candidate_id=cid, stratum="obvious_match", relevance=relevance,
        labelled_eligible=eligible, structured_score=structured, eligibility=eligibility,
        semantic_score=semantic, semantic_status=semantic_status,
        has_unknown_dimension=unknown_dim,
    )


class TestFixtureSchema:
    def test_template_parses(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        cases = [FixtureCase.model_validate(c) for c in raw]

        assert len(cases) == 1
        assert cases[0].candidates

    def test_template_is_excluded_from_the_corpus(self):
        # Underscore-prefixed files must not load, so the template can
        # never be mistaken for real evaluation data.
        dataset = load_dataset("evaluation/datasets")

        assert all(c.jd_id != "jd-example-001" for c in dataset.cases)

    def test_template_labels_are_all_empty(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        case = FixtureCase.model_validate(raw[0])

        assert not case.is_labelled
        assert len(case.unlabelled_ids()) == len(case.candidates)

    def test_template_covers_distinct_strata(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        case = FixtureCase.model_validate(raw[0])

        strata = {c.stratum for c in case.candidates}
        assert {"structured_tied", "keyword_stuffed", "ineligible_but_impressive",
                "missing_narrative"} <= strata

    def test_fixture_converts_to_real_app_schemas(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        case = FixtureCase.model_validate(raw[0])

        job = case.job.to_profile()
        candidate = case.candidates[0].to_profile()

        assert job.required_skills and job.responsibilities
        assert candidate.employment_history[0].responsibilities
        assert candidate.education is not None

    def test_relevance_is_bounded(self):
        from evaluation.schema import Labels

        with pytest.raises(Exception):
            Labels(relevance=4, eligible=True)
        with pytest.raises(Exception):
            Labels(relevance=-1, eligible=True)

    def test_duplicate_jd_ids_are_rejected(self, tmp_path):
        payload = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate jd_id"):
            load_dataset(tmp_path)


class TestLabelGate:
    def test_unlabelled_dataset_refuses_to_be_scored(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        dataset = Dataset(cases=[FixtureCase.model_validate(raw[0])])

        with pytest.raises(UnlabelledDatasetError):
            require_labelled(dataset)

    def test_error_names_the_unlabelled_candidates(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        dataset = Dataset(cases=[FixtureCase.model_validate(raw[0])])

        with pytest.raises(UnlabelledDatasetError, match="c-001"):
            require_labelled(dataset)

    def test_run_refuses_unlabelled_data(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        dataset = Dataset(cases=[FixtureCase.model_validate(raw[0])])

        with pytest.raises(UnlabelledDatasetError):
            run(dataset, split="dev", provider=FakeEmbeddingProvider())

    def test_partially_labelled_is_still_refused(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        case = FixtureCase.model_validate(raw[0])
        case.candidates[0].labels.relevance = 3
        case.candidates[0].labels.eligible = True

        with pytest.raises(UnlabelledDatasetError):
            require_labelled(Dataset(cases=[case]))


class TestEligibilityFirstRanking:
    def test_eligible_outranks_ineligible_regardless_of_score(self):
        ineligible_high = _scored("hi", 0.99, 3, eligibility="fail")
        eligible_low = _scored("lo", 0.01, 1, eligibility="pass")

        ranked = rank([ineligible_high, eligible_low], arm_baseline)

        assert ranked[0].candidate_id == "lo"

    @pytest.mark.parametrize("arm_name", list(default_arms()))
    def test_no_arm_can_promote_an_ineligible_candidate(self, arm_name):
        key = default_arms()[arm_name]
        ineligible = _scored("bad", 0.99, 3, eligibility="fail", semantic=0.99)
        eligible = _scored("good", 0.01, 1, eligibility="pass", semantic=0.0)

        ranked = rank([ineligible, eligible], key)

        assert ranked[0].candidate_id == "good", f"{arm_name} rescued a hard failure"

    def test_eligibility_respect_detects_a_violation(self):
        ranked = [_scored("a", 0.9, 3, eligible=False), _scored("b", 0.1, 2, eligible=True)]

        assert eligibility_respect(ranked) is False

    def test_eligibility_respect_accepts_a_clean_order(self):
        ranked = [_scored("a", 0.9, 3, eligible=True), _scored("b", 0.1, 2, eligible=False)]

        assert eligibility_respect(ranked) is True


class TestUnknownNeutrality:
    @pytest.mark.parametrize("arm_name", list(default_arms()))
    def test_missing_semantic_falls_back_to_the_baseline_ordering(self, arm_name):
        if arm_name == "B_semantic_only":
            pytest.skip("semantic-only arm is a reference point, not a fallback path")
        key = default_arms()[arm_name]
        rows = [
            _scored("high", 0.80, 3, semantic=None, semantic_status="unknown"),
            _scored("low", 0.20, 1, semantic=None, semantic_status="unknown"),
        ]

        assert [s.candidate_id for s in rank(rows, key)] == [
            s.candidate_id for s in rank(rows, arm_baseline)
        ]

    def test_unknown_semantic_is_not_treated_as_zero(self):
        # A candidate with UNKNOWN semantic must not be pushed below an
        # otherwise-identical candidate whose semantic merely scored low.
        weighted = make_arm_weighted(2.0)
        unknown = _scored("unknown", 0.60, 2, semantic=None, semantic_status="unknown")
        low = _scored("low", 0.60, 2, semantic=0.0)

        assert weighted(unknown) > weighted(low)

    def test_gated_arm_ignores_semantic_when_a_dimension_is_unknown(self):
        gated = make_arm_gated(2.0)
        row = _scored("x", 0.60, 2, semantic=0.99, unknown_dim=True)

        assert gated(row) == arm_baseline(row)

    def test_gated_arm_ignores_semantic_when_not_eligible(self):
        gated = make_arm_gated(2.0)
        row = _scored("x", 0.60, 2, semantic=0.99, eligibility="unknown")

        assert gated(row) == arm_baseline(row)

    @pytest.mark.parametrize("arm_name", list(default_arms()))
    def test_unknown_semantic_never_ranks_below_a_measured_bad_score(self, arm_name):
        """
        A candidate whose narrative could not be embedded must not be
        pushed below one that WAS measured and scored badly -- that
        would turn a missing signal into a negative one.
        """
        if arm_name == "B_semantic_only":
            pytest.skip("semantic-only arm has no structured fallback by design")
        key = default_arms()[arm_name]
        unknown = _scored("unknown", 0.60, 2, semantic=None, semantic_status="unknown")
        measured_badly = _scored("bad", 0.60, 2, semantic=0.01)

        ranked = rank([measured_badly, unknown], key)

        assert ranked[0].candidate_id == "unknown", (
            f"{arm_name} ranked an UNKNOWN candidate below a measured-bad one"
        )


class TestTieBreakerCannotHarm:
    def test_f4_preserves_baseline_order_when_scores_differ(self):
        rows = [
            _scored("a", 0.80, 3, semantic=0.0),
            _scored("b", 0.20, 1, semantic=1.0),
        ]

        assert [s.candidate_id for s in rank(rows, arm_tiebreak)] == ["a", "b"]

    def test_f4_reorders_only_exact_ties(self):
        rows = [
            _scored("a", 0.50, 1, semantic=0.10),
            _scored("b", 0.50, 3, semantic=0.90),
        ]

        assert [s.candidate_id for s in rank(rows, arm_tiebreak)] == ["b", "a"]


class TestBoundedArm:
    def test_adjustment_is_capped_by_delta(self):
        bounded = make_arm_bounded(0.05)
        best = bounded(_scored("x", 0.50, 2, semantic=1.0))[1]
        worst = bounded(_scored("y", 0.50, 2, semantic=0.0))[1]

        assert best <= 0.50 + 0.05 + 1e-9
        assert worst >= 0.50 - 0.05 - 1e-9

    def test_average_similarity_is_neutral(self):
        bounded = make_arm_bounded(0.10)

        assert bounded(_scored("x", 0.50, 2, semantic=0.5))[1] == pytest.approx(0.50)


class TestMetrics:
    def test_dcg_rewards_earlier_positions(self):
        assert dcg([3, 0]) > dcg([0, 3])

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        ranked = [_scored("a", 0.9, 3), _scored("b", 0.5, 2), _scored("c", 0.1, 0)]

        assert ndcg_at_k(ranked, 5) == pytest.approx(1.0)

    def test_ndcg_is_below_one_for_an_inverted_ranking(self):
        ranked = [_scored("c", 0.1, 0), _scored("b", 0.5, 2), _scored("a", 0.9, 3)]

        assert ndcg_at_k(ranked, 5) < 1.0

    def test_ndcg_is_none_when_no_candidate_is_relevant(self):
        assert ndcg_at_k([_scored("a", 0.5, 0)], 5) is None

    def test_precision_at_k(self):
        ranked = [_scored("a", 0.9, 3), _scored("b", 0.5, 1), _scored("c", 0.1, 2)]

        assert precision_at_k(ranked, 3, threshold=2) == pytest.approx(2 / 3)

    def test_kendall_tau_perfect_and_inverted(self):
        good = [_scored("a", 0.9, 3), _scored("b", 0.1, 1)]
        bad = [_scored("b", 0.1, 1), _scored("a", 0.9, 3)]

        assert kendall_tau(good) == pytest.approx(1.0)
        assert kendall_tau(bad) == pytest.approx(-1.0)


class TestTiedPairs:
    def test_identifies_pairs_the_structured_score_cannot_separate(self):
        rows = [_scored("a", 0.50, 3), _scored("b", 0.50, 1), _scored("c", 0.90, 2)]

        pairs = tied_pairs(rows)

        assert len(pairs) == 1
        assert {pairs[0][0].candidate_id, pairs[0][1].candidate_id} == {"a", "b"}

    def test_excludes_pairs_the_human_ranked_equally(self):
        # No correct answer exists for these, so they must not be scored.
        rows = [_scored("a", 0.50, 2), _scored("b", 0.50, 2)]

        assert tied_pairs(rows) == []

    def test_pairwise_accuracy_rewards_the_correct_order(self):
        rows = [_scored("a", 0.50, 3, semantic=0.9), _scored("b", 0.50, 1, semantic=0.1)]

        assert pairwise_accuracy(tied_pairs(rows), arm_tiebreak) == pytest.approx(1.0)

    def test_pairwise_accuracy_punishes_the_wrong_order(self):
        rows = [_scored("a", 0.50, 3, semantic=0.1), _scored("b", 0.50, 1, semantic=0.9)]

        assert pairwise_accuracy(tied_pairs(rows), arm_tiebreak) == pytest.approx(0.0)

    def test_baseline_scores_a_coin_flip_on_ties(self):
        # The baseline cannot separate a tie, which is exactly why the
        # tie-break question is worth asking.
        rows = [_scored("a", 0.50, 3), _scored("b", 0.50, 1)]

        assert pairwise_accuracy(tied_pairs(rows), arm_baseline) == pytest.approx(0.5)

    def test_no_tied_pairs_returns_none_not_zero(self):
        assert pairwise_accuracy([], arm_tiebreak) is None


class TestClusterBootstrap:
    def test_resamples_clusters_and_is_deterministic(self):
        values = {f"jd-{i}": 0.5 + 0.01 * i for i in range(6)}

        first = cluster_bootstrap_ci(values)
        second = cluster_bootstrap_ci(values)

        assert first == second

    def test_fewer_clusters_give_a_wider_interval(self):
        """The honest-uncertainty property: less data, wider interval."""
        few = {f"jd-{i}": v for i, v in enumerate([0.4, 0.9, 0.5, 0.8])}
        many = {f"jd-{i}": v for i, v in enumerate([0.4, 0.9, 0.5, 0.8] * 8)}

        lo_few, hi_few = cluster_bootstrap_ci(few)
        lo_many, hi_many = cluster_bootstrap_ci(many)

        assert (hi_few - lo_few) > (hi_many - lo_many)

    def test_single_cluster_cannot_produce_an_interval(self):
        assert cluster_bootstrap_ci({"jd-1": 0.7}) is None

    def test_interval_brackets_the_mean(self):
        values = {f"jd-{i}": v for i, v in enumerate([0.4, 0.6, 0.5, 0.7, 0.55])}
        mean = sum(values.values()) / len(values)

        lo, hi = cluster_bootstrap_ci(values)

        assert lo <= mean <= hi


class TestEndToEndWithLabelledFixture:
    def _labelled_case(self):
        raw = json.loads(open(TEMPLATE_PATH, encoding="utf-8").read())
        case = FixtureCase.model_validate(raw[0])
        # Synthetic labels for HARNESS testing only -- never a claim
        # about which candidate is genuinely better.
        for i, candidate in enumerate(case.candidates):
            candidate.labels.relevance = [3, 1, 0, 3, 2][i]
            candidate.labels.eligible = candidate.stratum != "ineligible_but_impressive"
        return case

    def test_scores_a_case_through_the_real_pipeline(self):
        rows = score_case(self._labelled_case(), FakeEmbeddingProvider())

        assert len(rows) == 5
        assert all(0.0 <= r.structured_score <= 1.0 for r in rows)
        assert all(r.eligibility in ("pass", "fail", "unknown", "partial") for r in rows)

    def test_missing_narrative_candidate_gets_unknown_semantic(self):
        rows = score_case(self._labelled_case(), FakeEmbeddingProvider())

        empty = next(r for r in rows if r.stratum == "missing_narrative")
        assert empty.semantic_status == "unknown"
        assert empty.semantic_score is None

    def test_run_produces_metrics_for_every_arm(self):
        dataset = Dataset(cases=[self._labelled_case()])

        results = run(dataset, split="dev", provider=FakeEmbeddingProvider())

        assert set(results) == set(default_arms())
        assert all(m.arm for m in results.values())

    def test_run_is_deterministic(self):
        dataset = Dataset(cases=[self._labelled_case()])

        first = run(dataset, split="dev", provider=FakeEmbeddingProvider())
        second = run(dataset, split="dev", provider=FakeEmbeddingProvider())

        assert {k: v.ndcg_at_5 for k, v in first.items()} == {
            k: v.ndcg_at_5 for k, v in second.items()
        }
        assert {k: v.tied_pair_accuracy for k, v in first.items()} == {
            k: v.tied_pair_accuracy for k, v in second.items()
        }

    def test_run_without_a_provider_still_works(self):
        """No embedding model at all must degrade to the baseline, not crash."""
        dataset = Dataset(cases=[self._labelled_case()])

        results = run(dataset, split="dev", provider=None)

        assert results["A_baseline"].ndcg_at_5 is not None

    def test_empty_split_raises(self):
        dataset = Dataset(cases=[self._labelled_case()])

        with pytest.raises(ValueError, match="No cases in split"):
            run(dataset, split="holdout", provider=FakeEmbeddingProvider())

    def test_evaluate_arm_records_per_jd_values_for_bootstrapping(self):
        case = self._labelled_case()
        rows = {case.jd_id: score_case(case, FakeEmbeddingProvider())}

        metrics = evaluate_arm(rows, "A_baseline", arm_baseline)

        assert case.jd_id in metrics.per_jd_ndcg_at_5


class TestNoProductionCoupling:
    def test_harness_does_not_require_ollama(self):
        """
        Must run in a clean subprocess: conftest.py imports app.llm ->
        langchain_ollama -> ollama, so an in-process sys.modules check
        would be polluted by the rest of the suite and prove nothing.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path
        from textwrap import dedent

        repo_root = Path(__file__).resolve().parents[1]
        script = dedent(
            """
            import sys
            import evaluation.harness
            import evaluation.schema
            import evaluation.report
            print([m for m in ("app.ollama_embeddings", "ollama", "langchain_ollama", "app.llm")
                   if m in sys.modules])
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root), env=env, capture_output=True, text=True, timeout=60,
        )

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "[]", (
            f"evaluation package pulled in an LLM/embedding module: {completed.stdout}"
        )

    def test_scoring_module_is_untouched_by_the_harness(self):
        """The harness computes arms itself; it must not add weights to 8A.

        MatchWeights legitimately grew a sixth field (project_evidence,
        Phase 4) as an approved, explicit production change to
        app.schemas/app.scoring -- NOT something evaluation/harness.py
        did. This test's actual guard is unchanged: the field set below
        must still match app.schemas exactly, so any FUTURE unapproved
        addition made by the harness (rather than production code) is
        still caught.
        """
        from app.scoring import DEFAULT_WEIGHTS
        from app.schemas import MatchWeights

        assert set(MatchWeights.model_fields) == {
            "version", "required_skills", "preferred_skills",
            "experience", "education", "seniority", "project_evidence",
        }
        assert DEFAULT_WEIGHTS.version == "v1"
        assert DEFAULT_WEIGHTS.project_evidence == 0.0
