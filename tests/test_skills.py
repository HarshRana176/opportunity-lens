"""
Characterizes app.extractor.build_technical_stack for the deterministic
path: known technologies (SKILL_CATEGORIES) and explicitly excluded
concepts (EXCLUDED_TECHNOLOGIES). No LLM calls are involved here --
every skill in these two tables is resolved without the fallback
classifier.
"""
import pytest

from app.extractor import (
    EXCLUDED_TECHNOLOGIES,
    SKILL_CATEGORIES,
    build_technical_stack,
)


CATEGORY_TO_FIELD = {
    "programming_languages": "programming_languages",
    "frameworks": "frameworks",
    "tools": "tools",
}


@pytest.mark.parametrize("skill, category", sorted(SKILL_CATEGORIES.items()))
def test_every_known_skill_lands_in_its_documented_bucket(skill, category):
    stack = build_technical_stack([skill])

    field = CATEGORY_TO_FIELD[category]
    other_fields = [f for f in CATEGORY_TO_FIELD.values() if f != field]

    assert skill in getattr(stack, field)
    for other in other_fields:
        assert getattr(stack, other) == []


@pytest.mark.parametrize("excluded_skill", sorted(EXCLUDED_TECHNOLOGIES))
def test_every_excluded_technology_is_dropped(excluded_skill):
    stack = build_technical_stack([excluded_skill])

    assert stack.programming_languages == []
    assert stack.frameworks == []
    assert stack.tools == []


def test_taxonomy_has_no_key_in_both_known_and_excluded_tables():
    # If a term were in both, its outcome would depend on dict iteration
    # order rather than an explicit rule. Guards against that drifting in.
    known = set(SKILL_CATEGORIES)
    excluded = set(EXCLUDED_TECHNOLOGIES)
    assert known & excluded == set()


@pytest.mark.parametrize(
    "variants",
    [
        ["Python", "PYTHON", "python"],
        ["Docker", "docker", "DOCKER"],
    ],
)
def test_lookup_is_case_insensitive(variants):
    stack = build_technical_stack(variants)

    all_skills = stack.programming_languages + stack.frameworks + stack.tools
    assert len(all_skills) == 1


def test_first_encountered_casing_is_preserved_on_dedupe():
    stack = build_technical_stack(["Python", "PYTHON", "python"])

    assert stack.programming_languages == ["Python"]


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_and_none_entries_are_skipped_without_error(blank):
    stack = build_technical_stack(["Python", blank])

    assert stack.programming_languages == ["Python"]


def test_empty_skill_list_yields_empty_stack():
    stack = build_technical_stack([])

    assert stack.programming_languages == []
    assert stack.frameworks == []
    assert stack.tools == []


def test_mlops_without_a_space_is_not_currently_recognized_as_excluded():
    # EXCLUDED_TECHNOLOGIES contains "mlo ps" (with a space), not "mlops".
    # This is very likely a typo, but Task 1 only characterizes existing
    # behavior -- it does not fix taxonomy content. Flagged here so the
    # gap is visible: today, "MLOps" is neither a known skill nor
    # excluded, so it silently falls through to the LLM fallback
    # classifier (see test_skills_fallback.py) instead of being dropped.
    assert "mlops" not in SKILL_CATEGORIES
    assert "mlops" not in EXCLUDED_TECHNOLOGIES
    assert "mlo ps" in EXCLUDED_TECHNOLOGIES
