"""
Characterizes the Task 4 taxonomy additions: SKILL_CANONICAL,
JD_EXCLUDED_TERMS, EDUCATION_LEVEL_TERMS, SENIORITY_TERMS.

Does NOT touch SKILL_CATEGORIES or EXCLUDED_TECHNOLOGIES -- those remain
exactly as characterized by tests/test_skills.py. This file exists to
prove the new additions are internally consistent and, critically, that
they do not silently alter the existing two tables.
"""
import pytest

from app.taxonomy import (
    EDUCATION_LEVEL_TERMS,
    EXCLUDED_TECHNOLOGIES,
    JD_EXCLUDED_TERMS,
    SENIORITY_TERMS,
    SKILL_CANONICAL,
    SKILL_CATEGORIES,
)


class TestResumeTablesUntouchedByTask4:
    """
    Guards the D6-refinement requirement: Task 4 must not modify
    EXCLUDED_TECHNOLOGIES or SKILL_CATEGORIES. Fixed counts here make
    "unchanged" a checkable fact, not an assumption.
    """

    def test_excluded_technologies_still_has_exactly_39_entries(self):
        assert len(EXCLUDED_TECHNOLOGIES) == 39

    def test_skill_categories_still_has_exactly_64_entries(self):
        assert len(SKILL_CATEGORIES) == 64

    def test_no_jd_excluded_term_appears_in_excluded_technologies(self):
        assert JD_EXCLUDED_TERMS & EXCLUDED_TECHNOLOGIES == set()

    def test_no_jd_excluded_term_is_a_known_canonical_skill_name(self):
        assert JD_EXCLUDED_TERMS & set(SKILL_CANONICAL.values()) == set()


class TestSkillCanonical:
    def test_every_skill_categories_key_has_a_canonical_mapping(self):
        assert set(SKILL_CATEGORIES) == set(SKILL_CANONICAL)

    def test_canonical_mapping_is_idempotent(self):
        # canonical(canonical(x)) == canonical(x) for every entry --
        # every canonical target is itself a self-mapping key.
        for key, canonical in SKILL_CANONICAL.items():
            assert SKILL_CANONICAL.get(canonical) == canonical

    def test_c_cpp_csharp_remain_three_distinct_canonical_skills(self):
        # Regression guard: naive punctuation stripping would collapse
        # these into one key. They must stay distinct identities.
        canonicals = {SKILL_CANONICAL["c"], SKILL_CANONICAL["c++"], SKILL_CANONICAL["c#"]}
        assert canonicals == {"c", "c++", "c#"}

    @pytest.mark.parametrize(
        "alias, canonical",
        [
            ("postgres", "postgresql"),
            ("sklearn", "scikit-learn"),
            ("next.js", "next.js"),
            ("nextjs", "next.js"),
            ("node.js", "node.js"),
            ("nodejs", "node.js"),
            ("power bi", "power bi"),
            ("powerbi", "power bi"),
            ("jupyter", "jupyter"),
            ("jupyter notebook", "jupyter"),
            ("airflow", "airflow"),
            ("apache airflow", "airflow"),
            ("transformers", "transformers"),
            ("huggingface transformers", "transformers"),
            ("aws", "aws"),
            ("aws fundamentals", "aws"),
        ],
    )
    def test_known_alias_groups_converge(self, alias, canonical):
        assert SKILL_CANONICAL[alias] == canonical

    def test_alias_group_members_share_the_same_category(self):
        groups = {}
        for key, canonical in SKILL_CANONICAL.items():
            groups.setdefault(canonical, set()).add(SKILL_CATEGORIES[key])

        inconsistent = {c: cats for c, cats in groups.items() if len(cats) > 1}
        assert inconsistent == {}

    def test_no_canonical_value_is_itself_an_alias_of_something_else(self):
        # Every canonical target must map to itself, not to a further
        # canonical target -- there is exactly one hop, never a chain.
        canonical_targets = set(SKILL_CANONICAL.values())
        for target in canonical_targets:
            assert SKILL_CANONICAL[target] == target


class TestEducationLevelTerms:
    VALID_LEVELS = {"HIGH_SCHOOL", "ASSOCIATE", "BACHELORS", "MASTERS", "DOCTORATE"}

    def test_every_value_is_a_recognized_level_name(self):
        assert set(EDUCATION_LEVEL_TERMS.values()) <= self.VALID_LEVELS

    def test_every_level_has_at_least_one_term(self):
        assert set(EDUCATION_LEVEL_TERMS.values()) == self.VALID_LEVELS

    def test_all_terms_are_lowercase(self):
        assert all(term == term.lower() for term in EDUCATION_LEVEL_TERMS)


class TestSeniorityTerms:
    VALID_LEVELS = {"INTERN", "JUNIOR", "MID", "SENIOR", "LEAD", "PRINCIPAL"}

    def test_every_value_is_a_recognized_seniority_name(self):
        assert set(SENIORITY_TERMS.values()) <= self.VALID_LEVELS

    def test_every_level_has_at_least_one_term(self):
        assert set(SENIORITY_TERMS.values()) == self.VALID_LEVELS

    def test_all_terms_are_lowercase(self):
        assert all(term == term.lower() for term in SENIORITY_TERMS)
