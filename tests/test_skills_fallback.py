"""
Characterizes the unknown-skill fallback path: app.skills'
classify_unknown_skill() and its use inside build_technical_stack().

No real Ollama/LLM calls are made. `app.skills.skill_classifier_chain`
is replaced with a stub exposing `.invoke(...)`. The patch target is
app.skills (where these functions are defined and where they resolve
`skill_classifier_chain` as a global at call time), not app.extractor --
even though app.extractor re-exports classify_unknown_skill and
build_technical_stack for import convenience, patching an attribute on
app.extractor would not affect the module the functions actually run in.
"""
from unittest.mock import Mock

import pytest

import app.skills as skills_module
from app.extractor import SkillCategory, build_technical_stack, classify_unknown_skill


def _stub_chain(category=None, raises=None):
    chain = Mock()
    if raises is not None:
        chain.invoke.side_effect = raises
    else:
        chain.invoke.return_value = SkillCategory(category=category)
    return chain


@pytest.mark.parametrize(
    "returned_category, expected_field",
    [
        ("programming_language", "programming_languages"),
        ("framework", "frameworks"),
        ("tool", "tools"),
    ],
)
def test_classify_unknown_skill_routes_by_llm_category(
    monkeypatch, returned_category, expected_field
):
    monkeypatch.setattr(
        skills_module, "skill_classifier_chain", _stub_chain(category=returned_category)
    )

    assert classify_unknown_skill("SomeBrandNewTool") == returned_category


def test_classify_unknown_skill_exclude_category(monkeypatch):
    monkeypatch.setattr(
        skills_module, "skill_classifier_chain", _stub_chain(category="exclude")
    )

    assert classify_unknown_skill("Some Methodology") == "exclude"


def test_classify_unknown_skill_excludes_on_classifier_exception(monkeypatch):
    monkeypatch.setattr(
        skills_module,
        "skill_classifier_chain",
        _stub_chain(raises=RuntimeError("ollama unreachable")),
    )

    # Never propagates -- an unknown skill that can't be classified is
    # excluded rather than guessed. See the comment in
    # app/extractor.py::classify_unknown_skill.
    assert classify_unknown_skill("Whatever") == "exclude"


@pytest.mark.parametrize(
    "returned_category, expected_field",
    [
        ("programming_language", "programming_languages"),
        ("framework", "frameworks"),
        ("tool", "tools"),
    ],
)
def test_build_technical_stack_places_unknown_skill_via_fallback(
    monkeypatch, returned_category, expected_field
):
    monkeypatch.setattr(
        skills_module, "skill_classifier_chain", _stub_chain(category=returned_category)
    )

    stack = build_technical_stack(["SomeBrandNewTool"])

    assert getattr(stack, expected_field) == ["SomeBrandNewTool"]


def test_build_technical_stack_drops_unknown_skill_excluded_by_llm(monkeypatch):
    monkeypatch.setattr(
        skills_module, "skill_classifier_chain", _stub_chain(category="exclude")
    )

    stack = build_technical_stack(["Some Methodology"])

    assert stack.programming_languages == []
    assert stack.frameworks == []
    assert stack.tools == []


def test_classifier_is_never_called_for_known_or_excluded_skills(monkeypatch):
    chain = _stub_chain(category="tool")
    monkeypatch.setattr(skills_module, "skill_classifier_chain", chain)

    # "python" is in SKILL_CATEGORIES, "rag" is in EXCLUDED_TECHNOLOGIES --
    # both must resolve via the static tables without ever reaching the LLM.
    build_technical_stack(["python", "rag"])

    chain.invoke.assert_not_called()


def test_classifier_is_called_exactly_once_per_unknown_skill(monkeypatch):
    chain = _stub_chain(category="tool")
    monkeypatch.setattr(skills_module, "skill_classifier_chain", chain)

    build_technical_stack(["python", "SomeBrandNewTool", "rag"])

    chain.invoke.assert_called_once_with({"skill": "SomeBrandNewTool"})


def test_mixed_known_excluded_and_unknown_skills_in_one_call(monkeypatch):
    monkeypatch.setattr(
        skills_module, "skill_classifier_chain", _stub_chain(category="framework")
    )

    stack = build_technical_stack(["python", "rag", "SomeBrandNewFramework"])

    assert stack.programming_languages == ["python"]
    assert stack.frameworks == ["SomeBrandNewFramework"]
    assert stack.tools == []
