"""
Characterizes app.job_extractor.extract_job: pipeline wiring, skill
dedupe, warning propagation, and malformed/ambiguous JD handling.

No real Ollama calls: app.llm.job_core_extraction_chain and
app.llm.job_requirements_extraction_chain are both replaced with stubs.
app.skills.batch_skill_classifier_chain is also stubbed where a test's
skills include anything outside the taxonomy, so no test in this file
depends on network access or a running Ollama instance.
"""
from unittest.mock import Mock

import pytest

import app.job_extractor as job_extractor_module
import app.skills as skills_module
from app.schemas import BatchSkillClassification, RawJobCoreExtraction, RawSkillMention


def _core(title="Software Engineer", responsibilities=None, skill_mentions=None):
    return RawJobCoreExtraction(
        title=title,
        responsibilities=responsibilities or [],
        skill_mentions=skill_mentions or [],
    )


def _reqs(experience_text=None, education_text=None):
    from app.schemas import RawJobRequirementsExtraction

    return RawJobRequirementsExtraction(
        experience_text=experience_text, education_text=education_text
    )


def _mention(name, level="required"):
    return RawSkillMention(name=name, level=level)


def _stub_chains(monkeypatch, core, reqs, batch_items=None):
    core_chain = Mock(invoke=Mock(return_value=core))
    reqs_chain = Mock(invoke=Mock(return_value=reqs))
    monkeypatch.setattr(job_extractor_module, "job_core_extraction_chain", core_chain)
    monkeypatch.setattr(job_extractor_module, "job_requirements_extraction_chain", reqs_chain)

    batch_chain = Mock()
    batch_chain.invoke.return_value = BatchSkillClassification(items=batch_items or [])
    monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", batch_chain)

    return core_chain, reqs_chain, batch_chain


class TestExtractJobValidation:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_or_whitespace_text_raises_value_error(self, text, monkeypatch):
        core_chain = Mock()
        monkeypatch.setattr(job_extractor_module, "job_core_extraction_chain", core_chain)

        with pytest.raises(ValueError, match="empty"):
            job_extractor_module.extract_job(text)

        # Must fail before ever calling the LLM.
        core_chain.invoke.assert_not_called()


class TestExtractJobCoreWiring:
    def test_title_and_responsibilities_pass_through(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(title="Data Analyst", responsibilities=["Build dashboards", "Write SQL"]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.title == "Data Analyst"
        assert profile.responsibilities == ["Build dashboards", "Write SQL"]
        assert profile.raw_text == "some JD text"

    def test_known_skill_resolves_via_taxonomy(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(skill_mentions=[_mention("Python", "required")]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        skill = profile.required_skills[0]
        assert skill.canonical == "python"
        assert skill.category == "programming_languages"
        assert skill.resolution == "taxonomy"
        assert skill.requirement_level == "required"

    def test_preferred_skill_lands_in_preferred_list(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(skill_mentions=[_mention("Docker", "preferred")]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.required_skills == []
        assert len(profile.preferred_skills) == 1
        assert profile.preferred_skills[0].requirement_level == "preferred"

    def test_resume_side_excluded_technology_is_dropped(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(skill_mentions=[_mention("RAG", "required"), _mention("Python", "required")]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        names = {s.raw for s in profile.required_skills}
        assert names == {"Python"}

    def test_jd_only_excluded_term_is_dropped(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(
                skill_mentions=[_mention("Agile", "required"), _mention("Python", "required")]
            ),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        names = {s.raw for s in profile.required_skills}
        assert names == {"Python"}

    def test_unresolved_skill_is_enriched_via_batch_call(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(skill_mentions=[_mention("Kafka", "required")]),
            _reqs(),
            batch_items=[{"name": "Kafka", "category": "tool"}],
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        skill = profile.required_skills[0]
        assert skill.resolution == "llm"
        assert skill.category == "tools"
        assert skill.canonical is None  # LLM never supplies canonical identity

    def test_unresolvable_skill_is_retained_not_dropped(self, monkeypatch):
        # Batch call returns nothing for it -- must stay in the profile,
        # unresolved, never silently disappear.
        _stub_chains(
            monkeypatch,
            _core(skill_mentions=[_mention("SomeBrandNewThing", "required")]),
            _reqs(),
            batch_items=[],
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        assert profile.required_skills[0].raw == "SomeBrandNewThing"
        assert profile.required_skills[0].resolution == "unresolved"


class TestSkillDedupe:
    def test_same_skill_required_and_preferred_keeps_required_only(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(
                skill_mentions=[
                    _mention("Python", "preferred"),
                    _mention("Python", "required"),
                ]
            ),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        assert profile.preferred_skills == []

    def test_dedupe_is_order_independent(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(
                skill_mentions=[
                    _mention("Python", "required"),
                    _mention("python", "preferred"),  # different casing, same match_key
                ]
            ),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        assert profile.preferred_skills == []

    def test_two_different_skills_are_not_merged(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(
                skill_mentions=[_mention("Python", "required"), _mention("Docker", "preferred")]
            ),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        assert len(profile.preferred_skills) == 1


class TestExtractJobExperienceAndEducation:
    def test_experience_and_education_are_interpreted_deterministically(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(),
            _reqs(
                experience_text="3+ years",
                education_text="Bachelor's degree in Computer Science",
            ),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.experience.min_months == 36
        assert profile.experience.is_specified is True
        assert profile.education.fields_of_study == ["Computer Science"]

    def test_seniority_derived_from_title(self, monkeypatch):
        _stub_chains(monkeypatch, _core(title="Senior Backend Engineer"), _reqs())

        profile = job_extractor_module.extract_job("some JD text")

        from app.schemas import Seniority

        assert profile.seniority == Seniority.SENIOR

    def test_no_experience_or_education_mentioned_yields_no_warnings(self, monkeypatch):
        _stub_chains(monkeypatch, _core(), _reqs(experience_text=None, education_text=None))

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.experience.is_specified is False
        assert profile.education.minimum_level is None
        assert profile.parse_warnings == []

    def test_unparseable_experience_text_adds_a_warning(self, monkeypatch):
        _stub_chains(monkeypatch, _core(), _reqs(experience_text="a few years"))

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.experience.is_specified is False
        assert any("experience" in w.lower() for w in profile.parse_warnings)

    def test_unparseable_education_text_adds_a_warning(self, monkeypatch):
        _stub_chains(monkeypatch, _core(), _reqs(education_text="some vague requirement"))

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.education.minimum_level is None
        assert any("education" in w.lower() for w in profile.parse_warnings)


class TestMalformedAndAmbiguousJobDescriptions:
    def test_skills_only_jd_with_no_responsibilities(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(responsibilities=[], skill_mentions=[_mention("Python", "required")]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.responsibilities == []
        assert len(profile.required_skills) == 1

    def test_responsibilities_only_jd_with_no_skills(self, monkeypatch):
        _stub_chains(
            monkeypatch,
            _core(responsibilities=["Do the thing"], skill_mentions=[]),
            _reqs(),
        )

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.responsibilities == ["Do the thing"]
        assert profile.required_skills == []
        assert profile.preferred_skills == []

    def test_no_recognizable_seniority_in_title(self, monkeypatch):
        _stub_chains(monkeypatch, _core(title="Data Wizard"), _reqs())

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.seniority is None

    def test_batch_enrichment_failure_does_not_fail_the_whole_extraction(self, monkeypatch):
        core_chain = Mock(
            invoke=Mock(return_value=_core(skill_mentions=[_mention("Kafka", "required")]))
        )
        reqs_chain = Mock(invoke=Mock(return_value=_reqs()))
        monkeypatch.setattr(job_extractor_module, "job_core_extraction_chain", core_chain)
        monkeypatch.setattr(job_extractor_module, "job_requirements_extraction_chain", reqs_chain)

        batch_chain = Mock()
        batch_chain.invoke.side_effect = RuntimeError("ollama unreachable")
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", batch_chain)

        profile = job_extractor_module.extract_job("some JD text")

        assert len(profile.required_skills) == 1
        assert profile.required_skills[0].resolution == "unresolved"
        assert len(profile.parse_warnings) == 1

    def test_completely_empty_extraction_still_produces_a_valid_profile(self, monkeypatch):
        _stub_chains(monkeypatch, _core(title="Mystery Role"), _reqs())

        profile = job_extractor_module.extract_job("some JD text")

        assert profile.title == "Mystery Role"
        assert profile.required_skills == []
        assert profile.preferred_skills == []
        assert profile.responsibilities == []
        assert profile.experience.is_specified is False
        assert profile.education.minimum_level is None
        assert profile.seniority is None
        assert profile.parse_warnings == []
