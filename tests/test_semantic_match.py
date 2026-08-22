"""
Task 8B-2b-i tests: app.semantic_match (per-employment semantic
orchestration and aggregation).

Fully offline -- FakeEmbeddingProvider only, no Ollama, no network, no
model pull.

The evaluation class at the bottom (TestStructuredTiedEvaluation) is
report-oriented: it builds candidates that Task 8A scores IDENTICALLY
and checks the semantic layer can still tell them apart. It asserts
only relative ordering, never a threshold or a tuned weight.
"""
import pytest

from app.embeddings import CachingEmbeddingProvider, FakeEmbeddingProvider
from app.matching import build_match_evidence
from app.schemas import (
    CandidateEmployment,
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
from app.semantic_match import (
    MAX_EMBED_CHARS,
    attach_employment_semantic_evidence,
    build_candidate_employment_text,
    build_job_text,
    compute_employment_semantic_evidence,
)

PAYMENTS_BULLETS = [
    "Built and operated backend payment services handling high volume transaction processing",
    "Owned reliability and latency budgets for distributed payment infrastructure",
]
PAYMENTS_JD = [
    "Maintain and scale backend payment services for high volume transaction processing",
    "Improve reliability and latency across distributed payment infrastructure",
]
VISION_BULLETS = [
    "Researched convolutional architectures for semantic segmentation of medical imagery",
    "Published experiments on visual representation learning and image augmentation",
]
FLORISTRY_BULLETS = [
    "Designed seasonal window displays and styled mannequins for retail storefronts",
    "Coordinated floral arrangements and bouquet inventory for weddings",
]


def _employment(company, role="Engineer", responsibilities=None, duration_months=24,
                start_date="Jan 2020", end_date="Jan 2022"):
    return CandidateEmployment(
        company=company, role=role, start_date=start_date, end_date=end_date,
        duration_months=duration_months, responsibilities=responsibilities or [],
    )


def _candidate(employment_history=None, **overrides):
    defaults = dict(
        candidate_name="Jane Doe", skills=[], total_experience_months=24,
        total_experience_years=2.0, raw_text="resume text",
        employment_history=employment_history or [],
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(responsibilities=None, **overrides):
    # `is None` rather than `or`: an explicitly EMPTY list is a real test
    # case (no JD responsibilities) and must not fall back to the default.
    defaults = dict(
        title="Engineer", required_skills=[], preferred_skills=[],
        experience=ExperienceRequirement(), education=EducationRequirement(),
        raw_text="job text",
        responsibilities=list(PAYMENTS_JD) if responsibilities is None else responsibilities,
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _provider():
    return FakeEmbeddingProvider()


class TestSingleEmployment:
    def test_single_position_produces_a_score(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.status == "pass"
        assert evidence.similarity_score is not None
        assert len(evidence.per_employment) == 1
        assert evidence.per_employment[0].company == "Acme"

    def test_headline_equals_the_only_score(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.similarity_score == evidence.per_employment[0].similarity_score

    def test_aggregation_is_labelled_max(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.aggregation == "max"


class TestMultipleEmployments:
    def test_every_position_appears_in_evidence(self):
        candidate = _candidate([
            _employment("Acme", responsibilities=PAYMENTS_BULLETS),
            _employment("Globex", responsibilities=VISION_BULLETS),
            _employment("Initech", responsibilities=FLORISTRY_BULLETS),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert [e.company for e in evidence.per_employment] == ["Acme", "Globex", "Initech"]

    def test_headline_is_the_maximum(self):
        candidate = _candidate([
            _employment("Acme", responsibilities=FLORISTRY_BULLETS),
            _employment("Globex", responsibilities=PAYMENTS_BULLETS),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        scores = [e.similarity_score for e in evidence.per_employment]
        assert evidence.similarity_score == max(scores)

    def test_unrelated_earlier_job_does_not_drag_headline_down(self):
        relevant_only = _candidate([_employment("Globex", responsibilities=PAYMENTS_BULLETS)])
        with_unrelated = _candidate([
            _employment("Acme", responsibilities=FLORISTRY_BULLETS),
            _employment("Globex", responsibilities=PAYMENTS_BULLETS),
        ])

        a = compute_employment_semantic_evidence(relevant_only, _job(), _provider())
        b = compute_employment_semantic_evidence(with_unrelated, _job(), _provider())

        assert a.similarity_score == b.similarity_score

    def test_order_is_preserved_not_sorted_by_score(self):
        # Weakest match listed first must STAY first.
        candidate = _candidate([
            _employment("Weak", responsibilities=FLORISTRY_BULLETS),
            _employment("Strong", responsibilities=PAYMENTS_BULLETS),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert [e.company for e in evidence.per_employment] == ["Weak", "Strong"]
        assert evidence.per_employment[0].similarity_score < evidence.per_employment[1].similarity_score


class TestOrderIndependence:
    def _scores(self, employments):
        evidence = compute_employment_semantic_evidence(
            _candidate(employments), _job(), _provider()
        )
        return evidence.similarity_score, evidence.weighted_mean_score

    def test_headline_and_weighted_mean_are_order_independent(self):
        forward = [
            _employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=12),
            _employment("B", responsibilities=VISION_BULLETS, duration_months=36),
            _employment("C", responsibilities=FLORISTRY_BULLETS, duration_months=6),
        ]
        reversed_order = list(reversed(forward))

        assert self._scores(forward) == self._scores(reversed_order)

    def test_per_employment_reflects_the_given_order(self):
        forward = [
            _employment("A", responsibilities=PAYMENTS_BULLETS),
            _employment("B", responsibilities=VISION_BULLETS),
        ]
        ev_forward = compute_employment_semantic_evidence(
            _candidate(forward), _job(), _provider()
        )
        ev_reversed = compute_employment_semantic_evidence(
            _candidate(list(reversed(forward))), _job(), _provider()
        )

        assert [e.company for e in ev_forward.per_employment] == ["A", "B"]
        assert [e.company for e in ev_reversed.per_employment] == ["B", "A"]


class TestWeightedMean:
    def test_weighted_mean_is_recorded(self):
        candidate = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=48),
            _employment("B", responsibilities=FLORISTRY_BULLETS, duration_months=2),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.weighted_mean_score is not None

    def test_longer_role_pulls_the_mean_toward_its_score(self):
        long_relevant = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=48),
            _employment("B", responsibilities=FLORISTRY_BULLETS, duration_months=2),
        ])
        long_irrelevant = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=2),
            _employment("B", responsibilities=FLORISTRY_BULLETS, duration_months=48),
        ])

        a = compute_employment_semantic_evidence(long_relevant, _job(), _provider())
        b = compute_employment_semantic_evidence(long_irrelevant, _job(), _provider())

        assert a.weighted_mean_score > b.weighted_mean_score
        # The headline max is unaffected by duration.
        assert a.similarity_score == b.similarity_score

    @pytest.mark.parametrize("duration", [None, 0, -5])
    def test_missing_or_zero_duration_still_participates_with_weight_one(self, duration):
        candidate = _candidate([_employment("A", responsibilities=PAYMENTS_BULLETS,
                                             duration_months=duration)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.weighted_mean_score == pytest.approx(evidence.similarity_score)

    def test_all_durations_missing_degenerates_to_plain_mean(self):
        candidate = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=None),
            _employment("B", responsibilities=FLORISTRY_BULLETS, duration_months=None),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        scores = [e.similarity_score for e in evidence.per_employment]
        assert evidence.weighted_mean_score == pytest.approx(sum(scores) / len(scores))


class TestSkippedAndUnknown:
    def test_position_without_bullets_is_skipped_not_scored_zero(self):
        candidate = _candidate([
            _employment("Acme", responsibilities=[]),
            _employment("Globex", responsibilities=PAYMENTS_BULLETS),
        ])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        skipped = evidence.per_employment[0]
        assert skipped.similarity_score is None
        assert skipped.similarity_score != 0.0
        assert skipped.status == "unknown"
        assert skipped.skipped_reason

    def test_skipped_position_is_excluded_from_weighted_mean(self):
        with_empty = _candidate([
            _employment("Empty", responsibilities=[]),
            _employment("Real", responsibilities=PAYMENTS_BULLETS),
        ])
        only_real = _candidate([_employment("Real", responsibilities=PAYMENTS_BULLETS)])

        a = compute_employment_semantic_evidence(with_empty, _job(), _provider())
        b = compute_employment_semantic_evidence(only_real, _job(), _provider())

        assert a.weighted_mean_score == pytest.approx(b.weighted_mean_score)

    def test_no_candidate_responsibilities_at_all_is_unknown(self):
        candidate = _candidate([_employment("Acme", responsibilities=[])])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert evidence.reason
        assert len(evidence.per_employment) == 1

    def test_no_employment_history_is_unknown(self):
        evidence = compute_employment_semantic_evidence(_candidate([]), _job(), _provider())

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_empty_job_responsibilities_is_unknown(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job([]), _provider())

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert "no responsibilities" in evidence.reason.lower()

    def test_blank_job_responsibilities_is_unknown(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(["  ", "\n"]), _provider())

        assert evidence.status == "unknown"

    def test_unknown_is_never_fail_and_never_zero(self):
        evidence = compute_employment_semantic_evidence(_candidate([]), _job(), _provider())

        assert evidence.status != "fail"
        assert evidence.similarity_score is not 0.0  # noqa: F632 - identity is the point
        assert evidence.similarity_score is None


class TestProviderFailure:
    def test_no_provider_is_unknown(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), None)

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_unavailable_provider_is_unknown(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(
            candidate, _job(), FakeEmbeddingProvider(available=False)
        )

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None
        assert evidence.per_employment[0].skipped_reason

    def test_raising_provider_is_unknown(self):
        class Boom:
            model_id = "boom"

            def is_available(self):
                return True

            def embed(self, texts):
                raise RuntimeError("backend exploded")

        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), Boom())

        assert evidence.status == "unknown"
        assert evidence.similarity_score is None

    def test_model_id_is_propagated(self):
        provider = FakeEmbeddingProvider(model_id="fake-xyz")
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])

        evidence = compute_employment_semantic_evidence(candidate, _job(), provider)

        assert evidence.model_id == "fake-xyz"


class TestTextConstructionAndLeakage:
    def test_candidate_text_uses_only_responsibilities(self):
        employment = _employment(
            "SecretCorp", role="Principal Staff Engineer",
            responsibilities=["Built payment APIs"],
            start_date="Jan 2019", end_date="Dec 2023", duration_months=60,
        )

        text, _ = build_candidate_employment_text(employment)

        assert text == "Built payment APIs"
        for leaked in ["SecretCorp", "Principal", "Staff", "Jan 2019", "Dec 2023", "60"]:
            assert leaked not in text

    def test_job_text_uses_only_responsibilities(self):
        job = _job(
            responsibilities=["Scale payment services"],
            title="Staff Payments Engineer",
            required_skills=[SkillRequirement(raw="Kubernetes", match_key="kubernetes",
                                               canonical="kubernetes", category=None,
                                               resolution="taxonomy", requirement_level="required")],
            experience=ExperienceRequirement(min_months=60, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.MASTERS, is_required=True),
            seniority=Seniority.PRINCIPAL,
            raw_text="THE ENTIRE JOB DESCRIPTION BODY",
        )

        text, _ = build_job_text(job)

        assert text == "Scale payment services"
        for leaked in ["Staff", "Kubernetes", "60", "MASTERS", "ENTIRE JOB DESCRIPTION"]:
            assert leaked not in text

    def test_no_structured_field_reaches_the_provider(self):
        """The strongest leakage guard: inspect exactly what was embedded."""
        provider = FakeEmbeddingProvider()
        candidate = _candidate(
            [_employment("SecretCorp", role="Principal Engineer",
                          responsibilities=["Built payment APIs"])],
            candidate_name="Confidential Person",
            skills=[CandidateSkill(raw="Kubernetes", match_key="kubernetes",
                                    canonical="kubernetes", category=None, resolution="taxonomy")],
            seniority=Seniority.SENIOR,
            raw_text="ENTIRE RESUME BODY WITH ADDRESS AND PHONE",
            education=EducationBackground(
                records=[EducationRecord(degree_raw="PhD", degree_key="phd",
                                          level=EducationLevel.DOCTORATE, resolution="taxonomy")],
                highest_level=EducationLevel.DOCTORATE,
            ),
        )
        job = _job(responsibilities=["Scale payment services"], title="Staff Engineer",
                    raw_text="ENTIRE JD BODY")

        compute_employment_semantic_evidence(candidate, job, provider)

        embedded = " ".join(provider.embedded_texts)
        for leaked in ["SecretCorp", "Confidential Person", "Kubernetes", "PhD",
                       "ENTIRE RESUME BODY", "ENTIRE JD BODY", "Staff Engineer",
                       "Principal Engineer", "ADDRESS", "PHONE"]:
            assert leaked not in embedded
        assert "Built payment APIs" in embedded
        assert "Scale payment services" in embedded

    def test_whitespace_is_normalized(self):
        employment = _employment("A", responsibilities=["  Built    payment\n\n APIs  "])

        text, _ = build_candidate_employment_text(employment)

        assert text == "Built payment APIs"

    def test_bullets_are_joined_with_newlines(self):
        employment = _employment("A", responsibilities=["First bullet", "Second bullet"])

        text, _ = build_candidate_employment_text(employment)

        assert text == "First bullet\nSecond bullet"

    def test_blank_bullets_are_dropped(self):
        employment = _employment("A", responsibilities=["", "   ", "Real"])

        text, truncated = build_candidate_employment_text(employment)

        assert text == "Real"
        assert truncated is False


class TestLengthCap:
    def test_under_cap_is_not_truncated(self):
        employment = _employment("A", responsibilities=["x" * 100, "y" * 100])

        text, truncated = build_candidate_employment_text(employment)

        assert truncated is False
        assert len(text) <= MAX_EMBED_CHARS

    def test_over_cap_truncates_at_a_bullet_boundary(self):
        bullet = "z" * 3000
        employment = _employment("A", responsibilities=[bullet, bullet, bullet, bullet])

        text, truncated = build_candidate_employment_text(employment)

        assert truncated is True
        assert len(text) <= MAX_EMBED_CHARS
        # Whole bullets only -- never a partial bullet.
        assert all(part == bullet for part in text.split("\n"))

    def test_truncation_is_reported_not_silent(self):
        bullet = "z" * 5000
        candidate = _candidate([_employment("A", responsibilities=[bullet, bullet])])

        evidence = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert evidence.per_employment[0].truncated is True

    def test_single_oversized_bullet_is_kept_whole(self):
        bullet = "z" * (MAX_EMBED_CHARS + 500)
        employment = _employment("A", responsibilities=[bullet])

        text, truncated = build_candidate_employment_text(employment)

        assert text == bullet
        assert truncated is False


class TestCacheReuse:
    def test_job_text_is_embedded_once_across_positions(self):
        inner = FakeEmbeddingProvider()
        cached = CachingEmbeddingProvider(inner)
        candidate = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS),
            _employment("B", responsibilities=VISION_BULLETS),
            _employment("C", responsibilities=FLORISTRY_BULLETS),
        ])

        compute_employment_semantic_evidence(candidate, _job(), cached)

        job_text, _ = build_job_text(_job())
        assert inner.embedded_texts.count(job_text) == 1

    def test_caching_does_not_change_the_result(self):
        candidate = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS),
            _employment("B", responsibilities=VISION_BULLETS),
        ])

        uncached = compute_employment_semantic_evidence(candidate, _job(), FakeEmbeddingProvider())
        cached = compute_employment_semantic_evidence(
            candidate, _job(), CachingEmbeddingProvider(FakeEmbeddingProvider())
        )

        assert uncached.similarity_score == cached.similarity_score
        assert uncached.weighted_mean_score == cached.weighted_mean_score


class TestDeterminism:
    def test_repeated_calls_are_identical(self):
        candidate = _candidate([
            _employment("A", responsibilities=PAYMENTS_BULLETS),
            _employment("B", responsibilities=VISION_BULLETS),
        ])

        first = compute_employment_semantic_evidence(candidate, _job(), _provider())
        second = compute_employment_semantic_evidence(candidate, _job(), _provider())

        assert first.model_dump() == second.model_dump()
        assert first.model_dump_json() == second.model_dump_json()


class TestAttachment:
    def _pair(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])
        return candidate, _job()

    def test_returns_a_copy_and_does_not_mutate_source(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)
        before = evidence.model_dump()

        attached = attach_employment_semantic_evidence(evidence, candidate, job, _provider())

        assert evidence.model_dump() == before
        assert evidence.semantic is None
        assert attached is not evidence
        assert attached.semantic is not None

    def test_structured_evidence_is_unchanged(self):
        candidate, job = self._pair()
        evidence = build_match_evidence(candidate, job)

        attached = attach_employment_semantic_evidence(evidence, candidate, job, _provider())

        before, after = evidence.model_dump(), attached.model_dump()
        assert before.pop("semantic") is None
        assert after.pop("semantic") is not None
        assert before == after

    def test_build_match_evidence_still_returns_none_semantic(self):
        candidate, job = self._pair()

        assert build_match_evidence(candidate, job).semantic is None


def _score_outputs(result):
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestScoringUnaffected:
    def test_score_is_identical_with_and_without_semantic(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])
        job = _job()
        evidence = build_match_evidence(candidate, job)
        attached = attach_employment_semantic_evidence(evidence, candidate, job, _provider())

        assert _score_outputs(score_match(evidence)) == _score_outputs(score_match(attached))

    def test_high_and_low_similarity_score_the_same(self):
        job = _job()
        strong = _candidate([_employment("A", responsibilities=PAYMENTS_BULLETS)])
        weak = _candidate([_employment("A", responsibilities=FLORISTRY_BULLETS)])

        strong_attached = attach_employment_semantic_evidence(
            build_match_evidence(strong, job), strong, job, _provider()
        )
        weak_attached = attach_employment_semantic_evidence(
            build_match_evidence(weak, job), weak, job, _provider()
        )

        assert strong_attached.semantic.similarity_score != weak_attached.semantic.similarity_score
        assert _score_outputs(score_match(strong_attached)) == _score_outputs(score_match(weak_attached))

    def test_no_score_component_is_named_semantic(self):
        candidate = _candidate([_employment("Acme", responsibilities=PAYMENTS_BULLETS)])
        job = _job()
        attached = attach_employment_semantic_evidence(
            build_match_evidence(candidate, job), candidate, job, _provider()
        )

        names = [c.name for c in score_match(attached).components]
        assert "semantic" not in names
        assert len(names) == 5


class TestStructuredTiedEvaluation:
    """
    The evaluation fixtures: two candidates Task 8A scores IDENTICALLY
    (same skills, experience, education, seniority) who differ ONLY in
    work context. 8A must tie them; the semantic layer should separate
    them, and in the right direction.

    Report-only -- nothing here modifies MatchResult or scoring, and no
    threshold or weight is asserted. Note that FakeEmbeddingProvider is
    a bag-of-words vectorizer, so this proves the MECHANISM (the tie is
    breakable and the plumbing is wired correctly), not that a real
    embedding model understands the domains. That claim needs 8B-2b-ii.
    """

    def _tied_pair(self):
        shared = dict(
            skills=[CandidateSkill(raw="Python", match_key="python", canonical="python",
                                    category=None, resolution="taxonomy")],
            total_experience_months=48,
            total_experience_years=4.0,
            seniority=Seniority.SENIOR,
            education=EducationBackground(
                records=[EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                          level=EducationLevel.BACHELORS, resolution="taxonomy")],
                highest_level=EducationLevel.BACHELORS,
            ),
        )
        payments = _candidate(
            [_employment("A", responsibilities=PAYMENTS_BULLETS, duration_months=48)], **shared
        )
        vision = _candidate(
            [_employment("A", responsibilities=VISION_BULLETS, duration_months=48)], **shared
        )
        job = _job(
            responsibilities=list(PAYMENTS_JD),
            title="Senior Backend Engineer",
            required_skills=[SkillRequirement(raw="Python", match_key="python",
                                               canonical="python", category=None,
                                               resolution="taxonomy", requirement_level="required")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
            seniority=Seniority.SENIOR,
        )
        return payments, vision, job

    def test_task_8a_genuinely_ties_the_two_candidates(self):
        payments, vision, job = self._tied_pair()

        a = score_match(build_match_evidence(payments, job))
        b = score_match(build_match_evidence(vision, job))

        assert a.overall_score == b.overall_score
        assert _score_outputs(a) == _score_outputs(b)

    def test_semantic_layer_separates_the_tied_candidates(self):
        payments, vision, job = self._tied_pair()

        a = compute_employment_semantic_evidence(payments, job, _provider())
        b = compute_employment_semantic_evidence(vision, job, _provider())

        assert a.status == "pass" and b.status == "pass"
        assert a.similarity_score != b.similarity_score

    def test_semantic_prefers_the_matching_work_context(self):
        payments, vision, job = self._tied_pair()

        a = compute_employment_semantic_evidence(payments, job, _provider())
        b = compute_employment_semantic_evidence(vision, job, _provider())

        assert a.similarity_score > b.similarity_score

    def test_breaking_the_tie_still_does_not_change_the_8a_score(self):
        payments, vision, job = self._tied_pair()

        a = attach_employment_semantic_evidence(
            build_match_evidence(payments, job), payments, job, _provider()
        )
        b = attach_employment_semantic_evidence(
            build_match_evidence(vision, job), vision, job, _provider()
        )

        assert a.semantic.similarity_score > b.semantic.similarity_score
        assert _score_outputs(score_match(a)) == _score_outputs(score_match(b))
