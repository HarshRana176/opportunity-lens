import re

from app.llm import batch_skill_classifier_chain, skill_classifier_chain
from app.schemas import NormalizedSkill, TechnicalStack
from app.taxonomy import EXCLUDED_TECHNOLOGIES, SKILL_CANONICAL, SKILL_CATEGORIES


# UNKNOWN TECHNOLOGY CLASSIFIER


def classify_unknown_skill(skill: str):

    try:

        result = skill_classifier_chain.invoke({
            "skill": skill
        })

        return result.category

    except Exception:

        # Safer to exclude an unknown item
        # than incorrectly categorize it.
        return "exclude"


# BUILD FINAL TECHNICAL STACK


def build_technical_stack(
    skills: list[str]
):

    categorized = {
        "programming_languages": [],
        "frameworks": [],
        "tools": [],
    }

    seen = set()

    for skill in skills:

        if not skill:
            continue

        clean_skill = skill.strip()

        if not clean_skill:
            continue

        key = clean_skill.lower()

        # Explicit exclusion
        if key in EXCLUDED_TECHNOLOGIES:
            continue

        # Avoid duplicates
        if key in seen:
            continue

        seen.add(key)


        # Known technology


        category = SKILL_CATEGORIES.get(key)

        if category:

            categorized[category].append(
                clean_skill
            )

            continue

        # Unknown technology


        category = classify_unknown_skill(
            clean_skill
        )

        if category == "programming_language":

            categorized[
                "programming_languages"
            ].append(clean_skill)

        elif category == "framework":

            categorized[
                "frameworks"
            ].append(clean_skill)

        elif category == "tool":

            categorized[
                "tools"
            ].append(clean_skill)

        # exclude -> ignore

    return TechnicalStack(
        programming_languages=categorized[
            "programming_languages"
        ],
        frameworks=categorized[
            "frameworks"
        ],
        tools=categorized[
            "tools"
        ],
    )


# ---------------------------------------------------------------------------
# Task 4 additions below (Job Description skill normalization). Nothing
# above this line is touched: build_technical_stack and
# classify_unknown_skill remain the résumé pipeline's unchanged path.
# ---------------------------------------------------------------------------


def compute_match_key(raw: str) -> str:
    """
    Lowercase + collapse internal whitespace -- deliberately NOT
    punctuation stripping. A naive strip would collapse "C", "C++", and
    "C#" into the same key, which is wrong (see app.taxonomy.SKILL_CANONICAL's
    docstring). This is always computable and always populated on a
    NormalizedSkill, so even a skill with no taxonomy entry stays
    matchable by exact match_key equality.
    """
    return re.sub(r"\s+", " ", raw.strip().lower())


def normalize_skill(
    raw: str, *, extra_excluded: set[str] | None = None
) -> NormalizedSkill | None:
    """
    Resolve one JD skill mention against the curated taxonomy.

    Returns None for a blank mention or one matching EXCLUDED_TECHNOLOGIES
    (unchanged, résumé-side) union `extra_excluded` (JD-only noise terms,
    e.g. app.taxonomy.JD_EXCLUDED_TERMS) -- these are not retained even
    as unresolved, since they are not technologies at all. Everything
    else is always returned: a taxonomy match sets canonical/category
    from the curated SKILL_CANONICAL/SKILL_CATEGORIES tables (resolution
    "taxonomy"); anything else comes back with resolution "unresolved",
    canonical=None, but a populated match_key, per the never-delete
    principle -- normalize_skill alone never consults the LLM.
    """
    if not raw or not raw.strip():
        return None

    clean_raw = raw.strip()
    match_key = compute_match_key(clean_raw)

    excluded = EXCLUDED_TECHNOLOGIES | (extra_excluded or set())
    if match_key in excluded:
        return None

    canonical = SKILL_CANONICAL.get(match_key)
    if canonical:
        return NormalizedSkill(
            raw=clean_raw,
            match_key=match_key,
            canonical=canonical,
            category=SKILL_CATEGORIES[canonical],
            resolution="taxonomy",
        )

    return NormalizedSkill(
        raw=clean_raw,
        match_key=match_key,
        canonical=None,
        category=None,
        resolution="unresolved",
    )


def skill_identity(skill: NormalizedSkill) -> str:
    """
    The single shared rule for deciding whether two normalized skills
    refer to the same technology: the curated canonical name when one
    is known, otherwise the match_key.

    Used by BOTH pipelines -- app.job_extractor for required/preferred
    dedupe and app.candidate_extractor for candidate-skill dedupe -- so
    the identity rule that future matching depends on exists in exactly
    one place and cannot drift between the two sides.

    Note this is the same rule matching itself will use: two skills
    match when their canonicals are equal, or (when either is
    unresolved) when their match_keys are equal.
    """
    return skill.canonical or skill.match_key


# Above this ceiling, enrichment is skipped entirely rather than
# truncated -- every unresolved skill is always retained either way;
# this only controls whether a batch LLM call is attempted for them.
ENRICHMENT_CEILING = 50

_BATCH_CATEGORY_MAP = {
    "programming_language": "programming_languages",
    "framework": "frameworks",
    "tool": "tools",
    # "exclude" is intentionally absent -- see enrich_unresolved_skills.
}


def enrich_unresolved_skills(
    skills: list[NormalizedSkill], *, ceiling: int = ENRICHMENT_CEILING
) -> tuple[list[NormalizedSkill], list[str]]:
    """
    Attempt to categorize every resolution=="unresolved" skill in
    `skills` with ONE batched LLM call, and return (possibly-updated
    skills, parse_warnings).

    Enrichment is advisory only:
      - An "exclude" verdict, an unrecognized category, or a skill the
        model simply did not return in its response all leave that
        skill unchanged (still resolution=="unresolved", still
        retained) -- the LLM has no authority to remove a skill from
        the list, only to add a category to one.
      - A successful match sets category and resolution="llm", but
        NEVER canonical: canonical names come only from the curated
        SKILL_CANONICAL table, never from an LLM judgment.
      - Match-back to the input is by exact string equality on `raw`
        (what was actually sent to the model), never by list position
        -- safe against the model reordering, omitting, or adding
        entries not in the input.
      - Any failure (Ollama unreachable, malformed output, anything)
        is caught; the input list is returned unchanged plus a warning.
        Enrichment failing must never fail JD extraction as a whole.

    Above `ceiling` unresolved skills, no call is made at all (skipped,
    not truncated) and a warning explains why -- see ENRICHMENT_CEILING.
    """
    unresolved = [s for s in skills if s.resolution == "unresolved"]

    if not unresolved:
        return list(skills), []

    if len(unresolved) > ceiling:
        return list(skills), [
            f"Skipped batch skill classification: {len(unresolved)} unresolved "
            f"skill(s) exceeds the ceiling of {ceiling}. All are retained unresolved."
        ]

    try:
        response = batch_skill_classifier_chain.invoke({
            "skills": "\n".join(f"- {skill.raw}" for skill in unresolved)
        })
    except Exception:
        return list(skills), [
            f"Batch skill classification failed; {len(unresolved)} skill(s) "
            f"remain unresolved."
        ]

    response_by_name = {}
    for item in response.items:
        if item.name not in response_by_name:
            response_by_name[item.name] = item.category

    replacements = {}
    unmatched_count = 0

    for skill in unresolved:
        verdict = response_by_name.get(skill.raw)

        if verdict is None:
            unmatched_count += 1
            continue

        mapped_category = _BATCH_CATEGORY_MAP.get(verdict)
        if mapped_category is None:
            # verdict == "exclude", or something unrecognized -- stays
            # unresolved either way. Never delete.
            continue

        replacements[id(skill)] = NormalizedSkill(
            raw=skill.raw,
            match_key=skill.match_key,
            canonical=None,
            category=mapped_category,
            resolution="llm",
        )

    enriched_skills = [replacements.get(id(skill), skill) for skill in skills]

    warnings = []
    if unmatched_count:
        warnings.append(
            f"{unmatched_count} skill(s) were not returned by batch "
            f"classification and remain unresolved."
        )

    return enriched_skills, warnings
