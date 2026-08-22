"""
Task 8B-2a tests: candidate work-narrative extraction and match-back.

Fully offline -- extract_text_from_pdf, extraction_chain, the batch
skill classifier, and work_narrative_extraction_chain are all
monkeypatched, following tests/test_candidate_extractor.py's
conventions.

The regression classes at the bottom are the important ones: 8B-2a adds
a THIRD LLM chain to build_candidate_profile, and the whole design bet
is that keeping it separate leaves the frozen résumé chain -- including
its documented 31->0 skill-extraction collapse -- completely untouched.
"""
from unittest.mock import Mock

import pytest

import app.candidate_extractor as candidate_extractor
import app.skills as skills_module
from app.schemas import (
    BatchSkillClassification,
    BatchSkillItem,
    EmploymentPeriod,
    RawEmploymentNarrative,
    RawResumeExtraction,
    RawWorkNarrativeExtraction,
    Seniority,
)


def _raw(candidate_name="Jane Doe", employment_history=None, skills=None):
    return RawResumeExtraction(
        candidate_name=candidate_name,
        employment_history=employment_history or [],
        skills=skills or [],
    )


def _period(company, role="Engineer", start_date="Jan 2020", end_date="Jan 2022"):
    return EmploymentPeriod(
        company=company, role=role, start_date=start_date, end_date=end_date
    )


def _narrative(*entries):
    return RawWorkNarrativeExtraction(
        positions=[
            RawEmploymentNarrative(company=company, role=role, responsibilities=bullets)
            for company, role, bullets in entries
        ]
    )


def _stub(monkeypatch, raw, narrative=None, batch_items=None, text="Jane Doe resume text"):
    monkeypatch.setattr(candidate_extractor, "extract_text_from_pdf", lambda path: text)
    monkeypatch.setattr(
        candidate_extractor, "extraction_chain", Mock(invoke=Mock(return_value=raw))
    )
    narrative_chain = Mock(
        invoke=Mock(return_value=narrative if narrative is not None else _narrative())
    )
    monkeypatch.setattr(
        candidate_extractor, "work_narrative_extraction_chain", narrative_chain
    )
    batch_chain = Mock()
    batch_chain.invoke.return_value = BatchSkillClassification(items=batch_items or [])
    monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", batch_chain)
    return narrative_chain


class TestSchemaDefault:
    def test_responsibilities_defaults_to_empty_list(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[_period("Acme")]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == []

    def test_profile_without_any_narrative_has_no_warnings(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[_period("Acme")]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.parse_warnings == []


class TestMultipleEmploymentNarratives:
    def test_each_position_receives_its_own_bullets(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme"), _period("Globex"), _period("Initech")]),
            narrative=_narrative(
                ("Acme", "Engineer", ["Built payment APIs", "Owned latency budgets"]),
                ("Globex", "Engineer", ["Maintained billing pipeline"]),
                ("Initech", "Engineer", ["Wrote reporting jobs"]),
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        by_company = {e.company: e.responsibilities for e in profile.employment_history}
        assert by_company["Acme"] == ["Built payment APIs", "Owned latency budgets"]
        assert by_company["Globex"] == ["Maintained billing pipeline"]
        assert by_company["Initech"] == ["Wrote reporting jobs"]

    def test_positions_without_narrative_stay_empty(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme"), _period("Globex")]),
            narrative=_narrative(("Acme", "Engineer", ["Built payment APIs"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        by_company = {e.company: e.responsibilities for e in profile.employment_history}
        assert by_company["Acme"] == ["Built payment APIs"]
        assert by_company["Globex"] == []

    def test_narrative_order_does_not_matter(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme"), _period("Globex")]),
            narrative=_narrative(
                ("Globex", "Engineer", ["Second listed"]),
                ("Acme", "Engineer", ["First listed"]),
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        by_company = {e.company: e.responsibilities for e in profile.employment_history}
        assert by_company["Acme"] == ["First listed"]
        assert by_company["Globex"] == ["Second listed"]

    def test_repeated_company_attaches_to_first_record_only(self, monkeypatch):
        # A promotion within one employer: attaching the same bullets to
        # both records would double-count that text in every later
        # per-employment semantic comparison.
        _stub(
            monkeypatch,
            _raw(employment_history=[
                _period("Acme", role="Senior Engineer", start_date="Jan 2022"),
                _period("Acme", role="Engineer", start_date="Jan 2020"),
            ]),
            narrative=_narrative(("Acme", "Senior Engineer", ["Led platform work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == ["Led platform work"]
        assert profile.employment_history[1].responsibilities == []


class TestExactCompanyMatching:
    def test_exact_match_attaches(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme Corp")]),
            narrative=_narrative(("Acme Corp", "Engineer", ["Did the work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == ["Did the work"]

    @pytest.mark.parametrize("narrative_company", ["acme corp", "ACME CORP", "Acme  Corp", "Acme"])
    def test_non_exact_company_does_not_attach(self, monkeypatch, narrative_company):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme Corp")]),
            narrative=_narrative((narrative_company, "Engineer", ["Did the work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == []
        assert any("does not match" in w for w in profile.parse_warnings)


class TestUnmatchedNarrativeEntry:
    def test_unmatched_entry_produces_a_warning(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Nonexistent Ltd", "Engineer", ["Invented work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert any("Nonexistent Ltd" in w for w in profile.parse_warnings)

    def test_unmatched_entry_never_invents_an_employment_record(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Nonexistent Ltd", "Engineer", ["Invented work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.employment_history) == 1
        assert profile.employment_history[0].company == "Acme"
        assert [e.company for e in profile.employment_history] == ["Acme"]

    def test_unmatched_bullets_appear_nowhere_in_the_profile(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Nonexistent Ltd", "Engineer", ["Invented work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        all_bullets = [b for e in profile.employment_history for b in e.responsibilities]
        assert "Invented work" not in all_bullets

    def test_narrative_with_no_employment_history_at_all(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[]),
            narrative=_narrative(("Ghost Co", "Engineer", ["Phantom work"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history == []
        assert any("Ghost Co" in w for w in profile.parse_warnings)


class TestEmptyAndBlankBullets:
    def test_entry_with_no_bullets_is_ignored_without_warning(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Acme", "Engineer", [])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == []
        assert profile.parse_warnings == []

    def test_blank_bullets_are_dropped(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Acme", "Engineer", ["  ", "", "Real bullet", "\n\t"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == ["Real bullet"]

    def test_bullets_are_stripped(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Acme", "Engineer", ["  Built things  "])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == ["Built things"]

    def test_unmatched_entry_with_only_blank_bullets_warns_nothing(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")]),
            narrative=_narrative(("Nonexistent Ltd", "Engineer", ["   "])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.parse_warnings == []


class TestChainFailure:
    def test_failure_yields_empty_responsibilities_and_a_warning(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[_period("Acme")]))
        monkeypatch.setattr(
            candidate_extractor,
            "work_narrative_extraction_chain",
            Mock(invoke=Mock(side_effect=RuntimeError("ollama down"))),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == []
        assert any("Work narrative extraction failed" in w for w in profile.parse_warnings)

    def test_failure_does_not_break_profile_construction(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                candidate_name="Jane Doe",
                employment_history=[_period("Acme", role="Senior Engineer")],
                skills=["Python"],
            ),
        )
        monkeypatch.setattr(
            candidate_extractor,
            "work_narrative_extraction_chain",
            Mock(invoke=Mock(side_effect=RuntimeError("ollama down"))),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.candidate_name == "Jane Doe"
        assert [s.raw for s in profile.skills] == ["Python"]
        assert profile.employment_history[0].company == "Acme"
        assert profile.total_experience_months > 0
        assert profile.seniority == Seniority.SENIOR

    def test_malformed_chain_output_is_caught(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[_period("Acme")]))
        monkeypatch.setattr(
            candidate_extractor,
            "work_narrative_extraction_chain",
            Mock(invoke=Mock(return_value=object())),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == []
        assert any("Work narrative extraction failed" in w for w in profile.parse_warnings)


class TestExistingBehaviorUnchanged:
    """
    8B-2a adds a third LLM chain to build_candidate_profile. Nothing
    that existed before it may move.
    """

    def test_employment_fields_are_untouched_by_narrative_attachment(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[
                _period("Acme", role="Senior Engineer", start_date="Jan 2020", end_date="Jan 2022")
            ]),
            narrative=_narrative(("Acme", "Senior Engineer", ["Built payment APIs"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")
        employment = profile.employment_history[0]

        assert employment.company == "Acme"
        assert employment.role == "Senior Engineer"
        assert employment.start_date == "Jan 2020"
        assert employment.end_date == "Jan 2022"
        assert employment.duration_months == 25
        assert employment.seniority == Seniority.SENIOR
        assert employment.is_current is False

    def test_narrative_does_not_change_experience_totals(self, monkeypatch):
        history = [_period("Acme", start_date="Jan 2020", end_date="Jan 2022")]

        _stub(monkeypatch, _raw(employment_history=history))
        without = candidate_extractor.build_candidate_profile("fake.pdf")

        _stub(
            monkeypatch,
            _raw(employment_history=history),
            narrative=_narrative(("Acme", "Engineer", ["Lots", "Of", "Bullets"])),
        )
        with_narrative = candidate_extractor.build_candidate_profile("fake.pdf")

        assert with_narrative.total_experience_months == without.total_experience_months
        assert with_narrative.total_experience_years == without.total_experience_years

    def test_narrative_does_not_change_seniority_or_current_role(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", role="Senior Engineer")]),
            narrative=_narrative(("Acme", "Senior Engineer", ["Interned briefly"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.seniority == Seniority.SENIOR
        assert profile.current_role == "Senior Engineer"

    def test_narrative_does_not_change_skills(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")], skills=["Python", "Docker"]),
            narrative=_narrative(("Acme", "Engineer", ["Used Kubernetes and Rust daily"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        raws = sorted(s.raw for s in profile.skills)
        assert raws == ["Docker", "Python"]


class TestSkillExtractionRegression:
    """
    The 31->0 collapse guard. Adding narrative extraction as a field on
    RawResumeExtraction is what empirically destroyed skill extraction
    on this repo's model; keeping it a separate chain is the entire
    reason 8B-2a is shaped this way. These pin that the résumé chain's
    contract and behavior are untouched.
    """

    def test_raw_resume_extraction_has_no_narrative_field(self):
        assert set(RawResumeExtraction.model_fields) == {
            "candidate_name", "employment_history", "skills",
        }

    def test_resume_chain_is_called_once_with_only_resume_text(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["Python"]))
        resume_chain = candidate_extractor.extraction_chain

        candidate_extractor.build_candidate_profile("fake.pdf")

        resume_chain.invoke.assert_called_once_with({"resume_text": "Jane Doe resume text"})

    def test_many_skills_all_survive_alongside_narrative(self, monkeypatch):
        # The literal 31->0 scenario: a skill-rich résumé must keep every
        # skill while narrative extraction runs alongside it.
        many = [f"Skill{i}" for i in range(31)]
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")], skills=many),
            narrative=_narrative(("Acme", "Engineer", ["Did work"])),
            batch_items=[BatchSkillItem(name=s, category="tool") for s in many],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 31
        assert sorted(s.raw for s in profile.skills) == sorted(many)

    def test_kafka_still_survives_with_narrative_present(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme")], skills=["Python", "Kafka", "Docker"]),
            narrative=_narrative(("Acme", "Engineer", ["Ran Kafka clusters"])),
            batch_items=[BatchSkillItem(name="Kafka", category="exclude")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        raws = [s.raw for s in profile.skills]
        assert "Kafka" in raws

    def test_narrative_chain_receives_the_resume_text(self, monkeypatch):
        narrative_chain = _stub(
            monkeypatch, _raw(employment_history=[_period("Acme")]), text="full resume body"
        )

        candidate_extractor.build_candidate_profile("fake.pdf")

        narrative_chain.invoke.assert_called_once_with({"resume_text": "full resume body"})


class TestSemanticTextIsolation:
    """
    8B-2a exists to give the semantic dimension a candidate-side text
    that structured matching does not already own. responsibilities
    must therefore carry ONLY verbatim bullets -- no skills, dates,
    education, titles, or names leak into it from this code path.
    """

    def test_responsibilities_contain_only_the_verbatim_bullets(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                candidate_name="Jane Doe",
                employment_history=[_period("Acme", role="Senior Engineer",
                                             start_date="Jan 2020", end_date="Jan 2022")],
                skills=["Python", "Docker"],
            ),
            narrative=_narrative(("Acme", "Senior Engineer", ["Built payment APIs"])),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].responsibilities == ["Built payment APIs"]
