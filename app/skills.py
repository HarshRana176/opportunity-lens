from app.llm import skill_classifier_chain
from app.schemas import TechnicalStack
from app.taxonomy import EXCLUDED_TECHNOLOGIES, SKILL_CATEGORIES


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
