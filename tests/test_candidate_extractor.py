"""
Characterizes app.candidate_extractor.build_candidate_profile.

No real PDF, Ollama, or Postgres access: app.candidate_extractor's
`extract_text_from_pdf` and `extraction_chain` are monkeypatched (both
are looked up from that module's globals at call time), and
app.skills.batch_skill_classifier_chain is stubbed wherever a test's
skills include anything outside the taxonomy.

The most important test in this file is
TestNoSkillIsSilentlyLost::test_kafka_survives_end_to_end -- the
regression guard for the entire reason this module exists. See the
class docstring there.
"""
from unittest.mock import Mock

import pytest

import app.candidate_extractor as candidate_extractor
import app.skills as skills_module
from app.schemas import (
    BatchSkillClassification,
    BatchSkillItem,
    EmploymentPeriod,
    RawResumeExtraction,
    Seniority,
)


def _raw(candidate_name="Jane Doe", employment_history=None, skills=None):
    return RawResumeExtraction(
        candidate_name=candidate_name,
        employment_history=employment_history or [],
        skills=skills or [],
    )


def _period(company, role, start_date, end_date):
    return EmploymentPeriod(
        company=company, role=role, start_date=start_date, end_date=end_date
    )


def _stub(monkeypatch, raw, batch_items=None, text="Jane Doe resume text"):
    monkeypatch.setattr(
        candidate_extractor, "extract_text_from_pdf", lambda path: text
    )
    monkeypatch.setattr(
        candidate_extractor, "extraction_chain", Mock(invoke=Mock(return_value=raw))
    )
    batch_chain = Mock()
    batch_chain.invoke.return_value = BatchSkillClassification(items=batch_items or [])
    monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", batch_chain)
    return batch_chain


class TestValidation:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_pdf_text_raises_before_calling_the_llm(self, monkeypatch, text):
        monkeypatch.setattr(
            candidate_extractor, "extract_text_from_pdf", lambda path: text
        )
        chain = Mock()
        monkeypatch.setattr(candidate_extractor, "extraction_chain", chain)

        with pytest.raises(ValueError, match="Could not extract text"):
            candidate_extractor.build_candidate_profile("fake.pdf")

        chain.invoke.assert_not_called()


class TestNoSkillIsSilentlyLost:
    """
    The reason app.candidate_extractor exists as a separate path.

    app.skills.build_technical_stack (the résumé pipeline's path) routes
    unknown skills through classify_unknown_skill, which DELETES a skill
    on an "exclude" verdict -- and that verdict is empirically wrong for
    real technologies ("Kafka" classified alone returns "exclude"). Once
    dropped there, the skill is unrecoverable from ResumeExtraction.

    A candidate profile must never lose a skill a job might require, so
    this path normalizes raw skills and uses the never-delete batched
    enrichment instead. These tests pin that guarantee.
    """

    def test_kafka_survives_end_to_end(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(skills=["Python", "Kafka", "Docker"]),
            batch_items=[BatchSkillItem(name="Kafka", category="tool")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        raws = [s.raw for s in profile.skills]
        assert "Kafka" in raws

    def test_kafka_survives_even_when_the_llm_says_exclude(self, monkeypatch):
        # The exact failure mode of the résumé path: an "exclude"
        # verdict must downgrade to unresolved-but-retained, never delete.
        _stub(
            monkeypatch,
            _raw(skills=["Kafka"]),
            batch_items=[BatchSkillItem(name="Kafka", category="exclude")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 1
        assert profile.skills[0].raw == "Kafka"
        assert profile.skills[0].resolution == "unresolved"

    def test_skill_survives_when_enrichment_returns_nothing(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["SomeBrandNewThing"]), batch_items=[])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert [s.raw for s in profile.skills] == ["SomeBrandNewThing"]
        assert profile.skills[0].resolution == "unresolved"

    def test_skill_survives_when_enrichment_raises(self, monkeypatch):
        monkeypatch.setattr(
            candidate_extractor, "extract_text_from_pdf", lambda path: "text"
        )
        monkeypatch.setattr(
            candidate_extractor,
            "extraction_chain",
            Mock(invoke=Mock(return_value=_raw(skills=["SomeBrandNewThing"]))),
        )
        chain = Mock()
        chain.invoke.side_effect = RuntimeError("ollama unreachable")
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert [s.raw for s in profile.skills] == ["SomeBrandNewThing"]
        assert len(profile.parse_warnings) >= 1

    def test_classify_unknown_skill_is_never_used_in_the_candidate_path(
        self, monkeypatch
    ):
        # Guards the mechanism, not just the outcome: if a future change
        # routed candidate skills through the per-skill classifier, this
        # fails even if that classifier happened to answer correctly.
        per_skill_chain = Mock()
        monkeypatch.setattr(skills_module, "skill_classifier_chain", per_skill_chain)
        _stub(
            monkeypatch,
            _raw(skills=["Kafka"]),
            batch_items=[BatchSkillItem(name="Kafka", category="tool")],
        )

        candidate_extractor.build_candidate_profile("fake.pdf")

        per_skill_chain.invoke.assert_not_called()

    def test_raw_skill_strings_are_preserved_verbatim(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["  PyTorch  "]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.skills[0].raw == "PyTorch"


class TestSkillNormalization:
    def test_known_skill_resolves_via_taxonomy(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["Python"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        skill = profile.skills[0]
        assert skill.canonical == "python"
        assert skill.category == "programming_languages"
        assert skill.resolution == "taxonomy"

    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("Postgres", "postgresql"),
            ("PostgreSQL", "postgresql"),
            ("sklearn", "scikit-learn"),
            ("AWS Fundamentals", "aws"),
            ("Jupyter Notebook", "jupyter"),
        ],
    )
    def test_aliases_converge_on_the_canonical_name(self, monkeypatch, raw, canonical):
        _stub(monkeypatch, _raw(skills=[raw]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.skills[0].canonical == canonical

    def test_c_cpp_csharp_remain_three_distinct_skills(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["C", "C++", "C#"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        canonicals = {s.canonical for s in profile.skills}
        assert canonicals == {"c", "c++", "c#"}
        assert len(profile.skills) == 3

    def test_excluded_technologies_are_still_dropped(self, monkeypatch):
        # The curated résumé-side exclusion set is the ONLY thing that
        # may remove a candidate skill.
        _stub(monkeypatch, _raw(skills=["Python", "rag", "deep learning"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert [s.raw for s in profile.skills] == ["Python"]

    def test_jd_excluded_terms_are_NOT_applied_to_candidates(self, monkeypatch):
        # JD_EXCLUDED_TERMS is JD-only by design. On a résumé, "Agile"
        # must be retained as an unresolved skill rather than dropped --
        # candidate information is never discarded by a JD-side rule.
        _stub(monkeypatch, _raw(skills=["Agile"]), batch_items=[])

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert [s.raw for s in profile.skills] == ["Agile"]
        assert profile.skills[0].resolution == "unresolved"

    def test_blank_skill_entries_are_skipped(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["Python", "", "   "]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 1

    def test_llm_never_supplies_a_canonical_name(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(skills=["Kafka"]),
            batch_items=[BatchSkillItem(name="Kafka", category="tool")],
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.skills[0].resolution == "llm"
        assert profile.skills[0].category == "tools"
        assert profile.skills[0].canonical is None


class TestSkillDedupe:
    def test_case_variants_collapse_to_one_skill(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["Python", "python", "PYTHON"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 1

    def test_first_occurrence_casing_wins(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["PyThOn", "python"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.skills[0].raw == "PyThOn"

    def test_aliases_of_the_same_technology_collapse(self, monkeypatch):
        # Different spellings, same canonical -> one skill.
        _stub(monkeypatch, _raw(skills=["Postgres", "PostgreSQL"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 1
        assert profile.skills[0].canonical == "postgresql"

    def test_distinct_skills_are_not_merged(self, monkeypatch):
        _stub(monkeypatch, _raw(skills=["Python", "Docker"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert len(profile.skills) == 2


class TestExperiencePreservation:
    def test_total_experience_matches_the_existing_aggregate_calculation(
        self, monkeypatch
    ):
        from app.experience import calculate_total_experience

        history = [
            _period("Acme", "Engineer", "May 2025", "Aug 2025"),
            _period("Beta", "Engineer", "Jan 2024", "Mar 2024"),
        ]
        _stub(monkeypatch, _raw(employment_history=history))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")
        expected = calculate_total_experience(history)

        assert profile.total_experience_months == expected["months"]
        assert profile.total_experience_years == expected["years"]

    def test_inclusive_month_semantics_are_preserved(self, monkeypatch):
        # May 2025 -> Aug 2025 inclusive == 4 months, matching
        # tests/test_experience.py's characterization.
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", "Engineer", "May 2025", "Aug 2025")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.total_experience_months == 4

    def test_overlapping_periods_are_not_double_counted(self, monkeypatch):
        history = [
            _period("Acme", "Engineer", "Jan 2025", "Jun 2025"),
            _period("Beta", "Consultant", "Apr 2025", "Sep 2025"),
        ]
        _stub(monkeypatch, _raw(employment_history=history))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        # Union Jan..Sep inclusive == 9
        assert profile.total_experience_months == 9

    def test_empty_employment_history_yields_zero(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.total_experience_months == 0
        assert profile.total_experience_years == 0.0


class TestEmploymentHistory:
    def test_verbatim_fields_are_preserved(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                employment_history=[
                    _period("CloudO Solutions Private Limited", "Software Developer Intern",
                            "Jan 2026", "Present")
                ]
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")
        entry = profile.employment_history[0]

        assert entry.company == "CloudO Solutions Private Limited"
        assert entry.role == "Software Developer Intern"
        assert entry.start_date == "Jan 2026"
        assert entry.end_date == "Present"

    def test_derived_interval_fields_are_populated(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", "Engineer", "May 2025", "Aug 2025")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")
        entry = profile.employment_history[0]

        assert entry.duration_months == 4
        assert entry.start_month_index is not None
        assert entry.end_month_index is not None

    def test_unparseable_dates_keep_verbatim_strings_and_warn(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                employment_history=[
                    _period("Ghost Co", "Engineer", "sometime", "later")
                ]
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")
        entry = profile.employment_history[0]

        # Never dropped -- verbatim data survives normalization failure.
        assert entry.company == "Ghost Co"
        assert entry.start_date == "sometime"
        assert entry.end_date == "later"
        assert entry.duration_months is None
        assert len(profile.parse_warnings) >= 1

    def test_role_may_be_absent(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", None, "May 2025", "Aug 2025")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].role is None
        assert profile.employment_history[0].seniority is None

    def test_per_role_seniority_is_derived(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                employment_history=[
                    _period("Acme", "Senior Backend Engineer", "Jan 2020", "Dec 2022")
                ]
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].seniority == Seniority.SENIOR

    def test_is_current_flag_is_set_for_present(self, monkeypatch, frozen_today):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", "Engineer", "Jan 2029", "Present")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.employment_history[0].is_current is True


class TestSeniorityAndCurrentRole:
    def test_seniority_comes_from_the_current_position(self, monkeypatch, frozen_today):
        _stub(
            monkeypatch,
            _raw(
                employment_history=[
                    _period("Old Co", "Senior Engineer", "Jan 2020", "Dec 2022"),
                    _period("New Co", "Engineering Intern", "Jan 2029", "Present"),
                ]
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        # Once-senior-now-intern: the CURRENT role governs, not the max
        # seniority ever held.
        assert profile.seniority == Seniority.INTERN
        assert profile.current_role == "Engineering Intern"

    def test_latest_started_position_wins_when_none_is_current(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                employment_history=[
                    _period("Old Co", "Junior Engineer", "Jan 2020", "Dec 2020"),
                    _period("Newer Co", "Lead Engineer", "Jan 2023", "Dec 2023"),
                ]
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.seniority == Seniority.LEAD
        assert profile.current_role == "Lead Engineer"

    def test_input_order_does_not_affect_selection(self, monkeypatch):
        history = [
            _period("Newer Co", "Lead Engineer", "Jan 2023", "Dec 2023"),
            _period("Old Co", "Junior Engineer", "Jan 2020", "Dec 2020"),
        ]
        _stub(monkeypatch, _raw(employment_history=history))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.current_role == "Lead Engineer"

    def test_unrecognized_title_yields_no_seniority(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", "Data Wizard", "Jan 2023", "Dec 2023")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.seniority is None
        assert profile.current_role == "Data Wizard"

    def test_no_employment_history_yields_no_seniority_or_role(self, monkeypatch):
        _stub(monkeypatch, _raw(employment_history=[]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.seniority is None
        assert profile.current_role is None

    def test_seniority_is_never_inferred_from_years_of_experience(self, monkeypatch):
        # A long tenure with an unrecognized title must NOT be promoted
        # to a seniority level -- derivation is title-based only.
        _stub(
            monkeypatch,
            _raw(employment_history=[_period("Acme", "Data Wizard", "Jan 2010", "Dec 2023")]),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.total_experience_months > 100
        assert profile.seniority is None


class TestProfileAssembly:
    def test_candidate_name_and_raw_text_are_preserved(self, monkeypatch):
        _stub(monkeypatch, _raw(candidate_name="Jane Doe"), text="FULL RESUME TEXT HERE")

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.candidate_name == "Jane Doe"
        assert profile.raw_text == "FULL RESUME TEXT HERE"

    def test_education_is_always_none_in_task_5(self, monkeypatch):
        # Approved decision D5: résumé education extraction is deferred
        # to Task 6. The field exists so adding it later needs no
        # re-shaping, but it is never populated here.
        _stub(monkeypatch, _raw(skills=["Python"]))

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.education is None

    def test_clean_profile_has_no_warnings(self, monkeypatch):
        _stub(
            monkeypatch,
            _raw(
                skills=["Python"],
                employment_history=[_period("Acme", "Engineer", "May 2025", "Aug 2025")],
            ),
        )

        profile = candidate_extractor.build_candidate_profile("fake.pdf")

        assert profile.parse_warnings == []
