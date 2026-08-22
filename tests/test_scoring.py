"""
Task 8A tests: app.scoring (MatchWeights -> MatchResult scoring layer).

Fixture builders mirror tests/test_match_evidence.py's, extended with a
`_rich_pair()` helper that gives every dimension N > 1 items -- the
condition under which a set/dict iteration-order bug would actually be
visible in a diff (see class TestSetAndDictOrderCannotAffectScoring and
TestReproducibilityAcrossHashSeeds below).
"""
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

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
    MatchWeights,
    Seniority,
    SkillRequirement,
)
from app.scoring import DEFAULT_WEIGHTS, DEFAULT_WEIGHTS_VERSION, score_match

REPO_ROOT = Path(__file__).resolve().parents[1]


def _skill(raw, canonical=None, resolution="taxonomy", requirement_level="required"):
    return SkillRequirement(
        raw=raw, match_key=raw.lower(), canonical=canonical, category=None,
        resolution=resolution, requirement_level=requirement_level,
    )


def _cskill(raw, canonical=None, resolution="taxonomy"):
    return CandidateSkill(raw=raw, match_key=raw.lower(), canonical=canonical, category=None, resolution=resolution)


def _candidate(**overrides):
    defaults = dict(
        candidate_name="Jane Doe",
        skills=[],
        total_experience_months=0,
        total_experience_years=0.0,
        raw_text="resume text",
    )
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _job(**overrides):
    defaults = dict(
        title="Engineer",
        required_skills=[],
        preferred_skills=[],
        experience=ExperienceRequirement(),
        education=EducationRequirement(),
        raw_text="job text",
    )
    defaults.update(overrides)
    return JobProfile(**defaults)


def _rich_pair():
    candidate = _candidate(
        skills=[
            _cskill("Python", canonical="python"),
            _cskill("PostgreSQL", canonical="postgresql"),
            _cskill("Kafka", canonical=None),
            _cskill("Docker", canonical="docker"),
            _cskill("Terraform", canonical=None),
        ],
        total_experience_months=54,
        seniority=Seniority.SENIOR,
        education=EducationBackground(
            records=[
                EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                 level=EducationLevel.BACHELORS, resolution="taxonomy"),
                EducationRecord(degree_raw="M.S. Computer Science", degree_key="ms_cs",
                                 level=EducationLevel.MASTERS, resolution="taxonomy",
                                 field_of_study_raw="Computer Science"),
            ],
            highest_level=EducationLevel.MASTERS,
        ),
    )
    job = _job(
        title="Senior Backend Engineer",
        required_skills=[
            _skill("Python", canonical="python"),
            _skill("PostgreSQL", canonical="postgresql"),
            _skill("Kafka", canonical=None),
        ],
        preferred_skills=[
            _skill("Docker", canonical="docker", requirement_level="preferred"),
            _skill("Kubernetes", canonical="kubernetes", requirement_level="preferred"),
            _skill("Go", canonical="go", requirement_level="preferred"),
        ],
        experience=ExperienceRequirement(min_months=36, is_specified=True),
        education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True,
                                        fields_of_study=["Computer Science"]),
        seniority=Seniority.SENIOR,
    )
    return candidate, job


class TestMatchWeightsHardening:
    def test_frozen_weights_rejects_mutation(self):
        weights = MatchWeights(
            version="v-test", required_skills=1, preferred_skills=1,
            experience=1, education=1, seniority=1,
        )

        with pytest.raises(ValidationError):
            weights.required_skills = 5

    def test_empty_version_is_rejected(self):
        with pytest.raises(ValidationError):
            MatchWeights(
                version="", required_skills=1, preferred_skills=1,
                experience=1, education=1, seniority=1,
            )

    def test_normal_versioned_weights_still_work(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        weights = MatchWeights(
            version="v-test", required_skills=2, preferred_skills=1,
            experience=1.5, education=1, seniority=1,
        )

        result = score_match(evidence, weights)

        assert result.weights_version == "v-test"
        assert 0.0 <= result.overall_score <= 1.0
        assert [c.weight for c in result.components] == [2, 1, 1.5, 1, 1]


class TestWeightsVersion:
    def test_default_weights_have_a_version(self):
        assert DEFAULT_WEIGHTS.version == DEFAULT_WEIGHTS_VERSION
        assert isinstance(DEFAULT_WEIGHTS.version, str) and DEFAULT_WEIGHTS.version

    def test_match_result_carries_default_weights_version(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        result = score_match(evidence)

        assert result.weights_version == DEFAULT_WEIGHTS.version

    def test_match_result_carries_custom_weights_version(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        weights = MatchWeights(
            version="experimental-2026-08", required_skills=1, preferred_skills=1,
            experience=1, education=1, seniority=1,
        )

        result = score_match(evidence, weights)

        assert result.weights_version == "experimental-2026-08"

    def test_component_weights_match_the_weights_object_used(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        weights = MatchWeights(
            version="v-test", required_skills=3, preferred_skills=2,
            experience=1, education=1, seniority=1,
        )

        result = score_match(evidence, weights)

        assert [c.weight for c in result.components] == [3, 2, 1, 1, 1]

    def test_every_match_result_has_a_non_empty_weights_version(self):
        # Sweeps a handful of representative evidence shapes to check
        # the invariant holds broadly, not just for one fixture.
        pairs = [
            (_candidate(), _job()),
            _rich_pair(),
            (_candidate(total_experience_months=200), _job(experience=ExperienceRequirement(min_months=12, max_months=60, is_specified=True))),
        ]
        for candidate, job in pairs:
            result = score_match(build_match_evidence(candidate, job))
            assert isinstance(result.weights_version, str) and result.weights_version


class TestDeterministicScoring:
    def test_same_evidence_and_weights_produce_identical_result(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        first = score_match(evidence, DEFAULT_WEIGHTS)
        second = score_match(evidence, DEFAULT_WEIGHTS)

        assert first.model_dump() == second.model_dump()
        assert first.model_dump_json() == second.model_dump_json()

    def test_rebuilding_evidence_from_scratch_still_scores_identically(self):
        candidate, job = _rich_pair()

        result_a = score_match(build_match_evidence(candidate, job))
        result_b = score_match(build_match_evidence(candidate, job))

        assert result_a.model_dump() == result_b.model_dump()

    def test_components_are_always_in_fixed_order(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        result = score_match(evidence)

        assert [c.name for c in result.components] == [
            "required_skills", "preferred_skills", "experience", "education", "seniority",
        ]

    def test_overall_score_is_a_weighted_average_in_unit_range(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        result = score_match(evidence)

        assert 0.0 <= result.overall_score <= 1.0

    def test_zero_weights_score_to_zero_without_dividing_by_zero(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)
        zero_weights = MatchWeights(
            version="zero", required_skills=0, preferred_skills=0,
            experience=0, education=0, seniority=0,
        )

        result = score_match(evidence, zero_weights)

        assert result.overall_score == 0.0

    def test_full_pass_scores_higher_than_full_fail(self):
        strong_candidate = _candidate(
            skills=[_cskill("Python", canonical="python")],
            total_experience_months=48,
            seniority=Seniority.SENIOR,
            education=EducationBackground(
                records=[EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                          level=EducationLevel.BACHELORS, resolution="taxonomy")],
                highest_level=EducationLevel.BACHELORS,
            ),
        )
        weak_candidate = _candidate(total_experience_months=0)
        job = _job(
            required_skills=[_skill("Python", canonical="python")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS, is_required=True),
            seniority=Seniority.SENIOR,
        )

        strong_result = score_match(build_match_evidence(strong_candidate, job))
        weak_result = score_match(build_match_evidence(weak_candidate, job))

        assert strong_result.overall_score > weak_result.overall_score


class TestRequiredSkillsReusesHardConstraintVerdict:
    def test_required_skills_component_matches_hard_constraint_entry(self):
        candidate, job = _rich_pair()
        evidence = build_match_evidence(candidate, job)

        result = score_match(evidence)

        hard_constraint_status = next(
            c.status for c in evidence.hard_constraints if c.kind == "required_skills"
        )
        score_component_status = next(
            c.status for c in result.components if c.name == "required_skills"
        )
        assert score_component_status == hard_constraint_status

    def test_vacuous_pass_when_no_required_skills_stated(self):
        candidate = _candidate()
        job = _job()  # no required_skills at all

        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)

        component = next(c for c in result.components if c.name == "required_skills")
        assert component.status == "pass"
        assert component.raw_value == 1.0


class TestPreferredSkillsStatus:
    def test_no_preferred_skills_is_pass(self):
        candidate, job = _rich_pair()
        job = job.model_copy(update={"preferred_skills": []})

        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)

        component = next(c for c in result.components if c.name == "preferred_skills")
        assert component.status == "pass"

    def test_some_but_not_all_preferred_matched_is_partial(self):
        candidate = _candidate(skills=[_cskill("Docker", canonical="docker")])
        job = _job(preferred_skills=[
            _skill("Docker", canonical="docker", requirement_level="preferred"),
            _skill("Kubernetes", canonical="kubernetes", requirement_level="preferred"),
        ])

        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)

        component = next(c for c in result.components if c.name == "preferred_skills")
        assert component.status == "partial"

    def test_none_matched_is_fail(self):
        candidate = _candidate(skills=[])
        job = _job(preferred_skills=[
            _skill("Kubernetes", canonical="kubernetes", requirement_level="preferred"),
        ])

        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)

        component = next(c for c in result.components if c.name == "preferred_skills")
        assert component.status == "fail"


class TestExperiencePartialIsNotSilentlyRemappedForScoring:
    def test_over_qualified_candidate_scores_partial_not_full_pass(self):
        candidate = _candidate(total_experience_months=200)
        job = _job(experience=ExperienceRequirement(min_months=12, max_months=60, is_specified=True))

        evidence = build_match_evidence(candidate, job)
        assert evidence.experience.status == "partial"  # sanity check on the fixture
        hard_constraint_status = next(
            c.status for c in evidence.hard_constraints if c.kind == "experience"
        )
        assert hard_constraint_status == "pass"  # eligibility gate remaps it

        result = score_match(evidence)
        component = next(c for c in result.components if c.name == "experience")

        assert component.status == "partial"  # scoring keeps the distinction
        assert component.raw_value == 0.75


def _score_projection(result):
    """
    The part of a MatchResult that should be invariant to input LIST
    order (candidate.skills / job.required_skills / job.preferred_skills
    order is caller-controlled and legitimately echoed back inside
    evidence.skills.*, so a full model_dump() comparison is the wrong
    tool here -- see the two tests below). weights_version, overall_score,
    and each component's (name, status, weight, raw_value, contribution)
    are what must be order-invariant; that's what this projects out.
    """
    return (
        result.weights_version,
        result.overall_score,
        [(c.name, c.status, c.weight, c.raw_value, c.contribution) for c in result.components],
    )


class TestSetAndDictOrderCannotAffectScoring:
    def test_candidate_skill_list_order_does_not_change_the_score(self):
        job = _job(required_skills=[_skill("Python", canonical="python")])
        skills_forward = [
            _cskill("Java", canonical="java"),
            _cskill("Python", canonical="python"),
            _cskill("Go", canonical="go"),
        ]
        skills_reversed = list(reversed(skills_forward))

        result_forward = score_match(build_match_evidence(_candidate(skills=skills_forward), job))
        result_reversed = score_match(build_match_evidence(_candidate(skills=skills_reversed), job))

        assert _score_projection(result_forward) == _score_projection(result_reversed)

    def test_requirement_list_order_does_not_change_the_score(self):
        candidate, _ = _rich_pair()
        required_forward = [
            _skill("Python", canonical="python"),
            _skill("PostgreSQL", canonical="postgresql"),
            _skill("Kafka", canonical=None),
        ]
        required_reversed = list(reversed(required_forward))

        result_forward = score_match(build_match_evidence(candidate, _job(required_skills=required_forward)))
        result_reversed = score_match(build_match_evidence(candidate, _job(required_skills=required_reversed)))

        assert _score_projection(result_forward) == _score_projection(result_reversed)


class TestReproducibilityAcrossHashSeeds:
    """
    Promotes the manual scratchpad probe run during Task 8A planning
    (build_match_evidence + score_match, serialized, diffed across
    PYTHONHASHSEED values) into a committed regression test. Spawns
    real subprocesses -- the whole point is that PYTHONHASHSEED only
    takes effect at interpreter start, so this cannot be simulated by
    mutating os.environ within the already-running test process.
    """

    _SCRIPT = dedent(
        """
        from app.matching import build_match_evidence
        from app.scoring import score_match
        from app.schemas import (
            CandidateProfile, CandidateSkill, EducationBackground, EducationLevel,
            EducationRecord, EducationRequirement, ExperienceRequirement, JobProfile,
            Seniority, SkillRequirement,
        )

        def _skill(raw, canonical=None, requirement_level="required"):
            return SkillRequirement(raw=raw, match_key=raw.lower(), canonical=canonical,
                                     category=None, resolution="taxonomy",
                                     requirement_level=requirement_level)

        def _cskill(raw, canonical=None):
            return CandidateSkill(raw=raw, match_key=raw.lower(), canonical=canonical,
                                   category=None, resolution="taxonomy")

        candidate = CandidateProfile(
            candidate_name="Jane Doe",
            skills=[_cskill("Python", "python"), _cskill("PostgreSQL", "postgresql"),
                    _cskill("Kafka", None), _cskill("Docker", "docker"),
                    _cskill("Terraform", None)],
            total_experience_months=54, total_experience_years=4.5, raw_text="resume text",
            seniority=Seniority.SENIOR,
            education=EducationBackground(
                records=[
                    EducationRecord(degree_raw="B.Tech", degree_key="btech",
                                     level=EducationLevel.BACHELORS, resolution="taxonomy"),
                    EducationRecord(degree_raw="M.S. CS", degree_key="ms_cs",
                                     level=EducationLevel.MASTERS, resolution="taxonomy",
                                     field_of_study_raw="Computer Science"),
                ],
                highest_level=EducationLevel.MASTERS,
            ),
        )
        job = JobProfile(
            title="Senior Backend Engineer",
            required_skills=[_skill("Python", "python"), _skill("PostgreSQL", "postgresql"),
                              _skill("Kafka", None)],
            preferred_skills=[_skill("Docker", "docker", "preferred"),
                               _skill("Kubernetes", "kubernetes", "preferred"),
                               _skill("Go", "go", "preferred")],
            experience=ExperienceRequirement(min_months=36, is_specified=True),
            education=EducationRequirement(minimum_level=EducationLevel.BACHELORS,
                                            is_required=True,
                                            fields_of_study=["Computer Science"]),
            seniority=Seniority.SENIOR,
            raw_text="job text",
        )

        evidence = build_match_evidence(candidate, job)
        result = score_match(evidence)
        print(result.model_dump_json())
        """
    )

    def _run_with_hashseed(self, hashseed):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        if hashseed is None:
            env.pop("PYTHONHASHSEED", None)
        else:
            env["PYTHONHASHSEED"] = str(hashseed)

        completed = subprocess.run(
            [sys.executable, "-c", self._SCRIPT],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    def test_identical_output_across_fixed_and_random_hash_seeds(self):
        outputs = {
            "seed_0": self._run_with_hashseed(0),
            "seed_1": self._run_with_hashseed(1),
            "seed_98765": self._run_with_hashseed(98765),
            "random_a": self._run_with_hashseed(None),
            "random_b": self._run_with_hashseed(None),
        }

        distinct_outputs = set(outputs.values())
        assert len(distinct_outputs) == 1, (
            "MatchResult serialization differs across PYTHONHASHSEED runs "
            f"(hash-seed nondeterminism leaked into output): {outputs}"
        )
