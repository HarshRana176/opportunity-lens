"""
Characterizes the Task 4 (JD-side) additions to app.skills:
compute_match_key, normalize_skill, enrich_unresolved_skills.

Does NOT touch build_technical_stack or classify_unknown_skill --
those remain exactly as characterized by tests/test_skills.py and
tests/test_skills_fallback.py. This file proves the new JD-side path
is correct in isolation AND that it leaves the résumé-side path alone.

No real Ollama calls: app.skills.batch_skill_classifier_chain is
replaced with a stub exposing `.invoke(...)`, the same pattern already
established for skill_classifier_chain in test_skills_fallback.py.
"""
from unittest.mock import Mock

import pytest

import app.skills as skills_module
from app.schemas import BatchSkillClassification, BatchSkillItem, NormalizedSkill
from app.skills import (
    build_technical_stack,
    compute_match_key,
    enrich_unresolved_skills,
    normalize_skill,
)
from app.taxonomy import EXCLUDED_TECHNOLOGIES, JD_EXCLUDED_TERMS, SKILL_CATEGORIES


def _stub_batch_chain(items=None, raises=None):
    chain = Mock()
    if raises is not None:
        chain.invoke.side_effect = raises
    else:
        chain.invoke.return_value = BatchSkillClassification(items=items or [])
    return chain


class TestComputeMatchKey:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Python", "python"),
            ("  Python  ", "python"),
            ("C", "c"),
            ("C++", "c++"),
            ("C#", "c#"),
            ("Node.js", "node.js"),
            ("Multiple   Spaces", "multiple spaces"),
            ("REST API", "rest api"),
        ],
    )
    def test_lowercase_and_collapse_whitespace_only(self, raw, expected):
        assert compute_match_key(raw) == expected

    def test_c_cpp_csharp_produce_distinct_match_keys(self):
        # The core regression guard: no punctuation stripping.
        keys = {compute_match_key("C"), compute_match_key("C++"), compute_match_key("C#")}
        assert keys == {"c", "c++", "c#"}

    def test_does_not_strip_punctuation(self):
        assert compute_match_key("CI/CD") == "ci/cd"
        assert compute_match_key(".NET") == ".net"


class TestNormalizeSkill:
    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_blank_input_returns_none(self, blank):
        assert normalize_skill(blank) is None

    @pytest.mark.parametrize("excluded_term", sorted(EXCLUDED_TECHNOLOGIES))
    def test_resume_side_excluded_technologies_are_still_dropped(self, excluded_term):
        assert normalize_skill(excluded_term) is None

    @pytest.mark.parametrize("jd_term", sorted(JD_EXCLUDED_TERMS))
    def test_jd_excluded_terms_are_dropped_when_passed_as_extra_excluded(self, jd_term):
        assert normalize_skill(jd_term, extra_excluded=JD_EXCLUDED_TERMS) is None

    def test_jd_excluded_terms_are_not_dropped_without_extra_excluded(self):
        # JD_EXCLUDED_TERMS is JD-caller-supplied, never baked in --
        # normalize_skill alone does not know about it.
        result = normalize_skill("agile", extra_excluded=None)
        assert result is not None
        assert result.resolution == "unresolved"

    def test_known_skill_resolves_via_taxonomy(self):
        result = normalize_skill("Python")
        assert result.resolution == "taxonomy"
        assert result.canonical == "python"
        assert result.category == "programming_languages"
        assert result.raw == "Python"
        assert result.match_key == "python"

    @pytest.mark.parametrize(
        "raw, canonical",
        [
            ("Postgres", "postgresql"),
            ("PostgreSQL", "postgresql"),
            ("sklearn", "scikit-learn"),
            ("nextjs", "next.js"),
        ],
    )
    def test_alias_resolves_to_the_shared_canonical_name(self, raw, canonical):
        result = normalize_skill(raw)
        assert result.canonical == canonical

    def test_c_cpp_csharp_resolve_to_three_distinct_canonicals(self):
        c = normalize_skill("C")
        cpp = normalize_skill("C++")
        csharp = normalize_skill("C#")
        assert {c.canonical, cpp.canonical, csharp.canonical} == {"c", "c++", "c#"}

    def test_unknown_skill_is_retained_as_unresolved_not_dropped(self):
        result = normalize_skill("SomeBrandNewTechnology")
        assert result is not None
        assert result.resolution == "unresolved"
        assert result.canonical is None
        assert result.category is None
        assert result.raw == "SomeBrandNewTechnology"
        assert result.match_key == "somebrandnewtechnology"

    def test_taxonomy_lookup_is_whitespace_tolerant(self):
        # match_key collapses whitespace, so this now resolves even
        # though it wouldn't via the résumé side's exact .lower() key.
        result = normalize_skill("  Power   BI  ")
        assert result.resolution == "taxonomy"
        assert result.canonical == "power bi"


class TestResumeSideUnaffectedByTask4:
    """Guards that adding normalize_skill/enrich_unresolved_skills did
    not alter build_technical_stack's behavior in any way."""

    def test_build_technical_stack_still_works_for_known_skills(self):
        stack = build_technical_stack(["Python", "Docker", "rag"])
        assert stack.programming_languages == ["Python"]
        assert stack.tools == ["Docker"]

    def test_excluded_technologies_table_size_unchanged(self):
        assert len(EXCLUDED_TECHNOLOGIES) == 39

    def test_skill_categories_table_size_unchanged(self):
        assert len(SKILL_CATEGORIES) == 64


class TestEnrichUnresolvedSkills:
    def _unresolved(self, raw):
        return NormalizedSkill(
            raw=raw, match_key=compute_match_key(raw), resolution="unresolved"
        )

    def _taxonomy(self, raw, canonical, category):
        return NormalizedSkill(
            raw=raw,
            match_key=compute_match_key(raw),
            canonical=canonical,
            category=category,
            resolution="taxonomy",
        )

    def test_no_unresolved_skills_short_circuits_without_calling_the_chain(self, monkeypatch):
        chain = _stub_batch_chain(items=[])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._taxonomy("Python", "python", "programming_languages")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert enriched == skills
        assert warnings == []
        chain.invoke.assert_not_called()

    def test_successful_classification_sets_category_and_resolution_llm(self, monkeypatch):
        chain = _stub_batch_chain(
            items=[BatchSkillItem(name="Kafka", category="tool")]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert len(enriched) == 1
        assert enriched[0].resolution == "llm"
        assert enriched[0].category == "tools"
        assert warnings == []

    def test_llm_cannot_invent_a_canonical_name(self, monkeypatch):
        chain = _stub_batch_chain(
            items=[BatchSkillItem(name="Kafka", category="tool")]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        enriched, _ = enrich_unresolved_skills([self._unresolved("Kafka")])

        assert enriched[0].canonical is None

    def test_exclude_verdict_does_not_delete_the_skill(self, monkeypatch):
        chain = _stub_batch_chain(
            items=[BatchSkillItem(name="Kafka", category="exclude")]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert len(enriched) == 1
        assert enriched[0].raw == "Kafka"
        assert enriched[0].resolution == "unresolved"  # never deleted, never invented a category

    def test_taxonomy_resolved_skills_are_never_sent_to_the_batch_call(self, monkeypatch):
        chain = _stub_batch_chain(items=[BatchSkillItem(name="Kafka", category="tool")])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [
            self._taxonomy("Python", "python", "programming_languages"),
            self._unresolved("Kafka"),
        ]
        enrich_unresolved_skills(skills)

        sent = chain.invoke.call_args[0][0]["skills"]
        assert "Kafka" in sent
        assert "Python" not in sent

    def test_taxonomy_resolved_skills_pass_through_unchanged(self, monkeypatch):
        chain = _stub_batch_chain(items=[BatchSkillItem(name="Kafka", category="tool")])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        python_skill = self._taxonomy("Python", "python", "programming_languages")
        enriched, _ = enrich_unresolved_skills([python_skill, self._unresolved("Kafka")])

        assert enriched[0] == python_skill

    def test_match_back_is_by_exact_string_not_position(self, monkeypatch):
        # Response returns items in a DIFFERENT order than the input.
        chain = _stub_batch_chain(
            items=[
                BatchSkillItem(name="Redis", category="tool"),
                BatchSkillItem(name="Kafka", category="tool"),
            ]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka"), self._unresolved("Redis")]
        enriched, _ = enrich_unresolved_skills(skills)

        by_raw = {s.raw: s for s in enriched}
        assert by_raw["Kafka"].resolution == "llm"
        assert by_raw["Redis"].resolution == "llm"

    def test_skill_missing_from_the_response_stays_unresolved_with_a_warning(self, monkeypatch):
        chain = _stub_batch_chain(items=[])  # model returned nothing
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert enriched[0].resolution == "unresolved"
        assert len(warnings) == 1
        assert "1" in warnings[0]

    def test_hallucinated_extra_response_entries_are_ignored(self, monkeypatch):
        chain = _stub_batch_chain(
            items=[
                BatchSkillItem(name="Kafka", category="tool"),
                BatchSkillItem(name="SomethingNotInInput", category="tool"),
            ]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert len(enriched) == 1
        assert enriched[0].raw == "Kafka"
        assert warnings == []

    def test_chain_failure_is_caught_and_leaves_all_skills_unresolved(self, monkeypatch):
        chain = _stub_batch_chain(raises=RuntimeError("ollama unreachable"))
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Kafka"), self._unresolved("Redis")]
        enriched, warnings = enrich_unresolved_skills(skills)

        assert enriched == skills
        assert len(warnings) == 1

    def test_ceiling_exceeded_skips_the_call_and_retains_everything(self, monkeypatch):
        chain = _stub_batch_chain(items=[])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved(f"Tech{i}") for i in range(51)]
        enriched, warnings = enrich_unresolved_skills(skills, ceiling=50)

        chain.invoke.assert_not_called()
        assert enriched == skills
        assert all(s.resolution == "unresolved" for s in enriched)
        assert len(warnings) == 1
        assert "50" in warnings[0]

    def test_exactly_at_ceiling_still_calls_the_chain(self, monkeypatch):
        chain = _stub_batch_chain(items=[])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved(f"Tech{i}") for i in range(50)]
        enrich_unresolved_skills(skills, ceiling=50)

        chain.invoke.assert_called_once()

    def test_unrecognized_category_value_stays_unresolved(self, monkeypatch):
        # Defensive: BatchSkillItem's Literal validation should prevent
        # this in practice, but enrich_unresolved_skills must not crash
        # or invent a category if it somehow occurs.
        item = Mock(name="Weird", category="something_unexpected")
        item.name = "Weird"
        chain = Mock()
        chain.invoke.return_value = Mock(items=[item])
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [self._unresolved("Weird")]
        enriched, _ = enrich_unresolved_skills(skills)

        assert enriched[0].resolution == "unresolved"

    def test_output_preserves_original_list_order(self, monkeypatch):
        chain = _stub_batch_chain(
            items=[
                BatchSkillItem(name="Kafka", category="tool"),
                BatchSkillItem(name="Terraform", category="tool"),
            ]
        )
        monkeypatch.setattr(skills_module, "batch_skill_classifier_chain", chain)

        skills = [
            self._taxonomy("Python", "python", "programming_languages"),
            self._unresolved("Kafka"),
            self._taxonomy("Docker", "docker", "tools"),
            self._unresolved("Terraform"),
        ]
        enriched, _ = enrich_unresolved_skills(skills)

        assert [s.raw for s in enriched] == ["Python", "Kafka", "Docker", "Terraform"]
