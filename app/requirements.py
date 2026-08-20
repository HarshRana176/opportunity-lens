"""
Deterministic interpretation of job-description requirement phrases.

Everything here is pure Python: no LLM calls, no I/O, no imports from
app.extractor or app.job_extractor. Inputs are verbatim phrases the LLM
extraction step already pulled out of the JD text (e.g. "3+ years",
"Bachelor's in Computer Science or related field") -- this module's job
is only to interpret those phrases into structured, comparable values,
the same separation app.experience already establishes on the résumé
side (LLM extracts dates verbatim; Python computes durations).
"""
import re

from app.schemas import EducationLevel, EducationRequirement, ExperienceRequirement, Seniority
from app.taxonomy import EDUCATION_LEVEL_TERMS, EDUCATION_OPTIONAL_MARKERS, SENIORITY_TERMS


# EXPERIENCE REQUIREMENT PARSING


_UNIT_PATTERN = r"(years?|yrs?|months?|mos?)"

_RANGE_PATTERN = re.compile(rf"(\d+)\s*(?:-|to)\s*(\d+)\s*{_UNIT_PATTERN}")
_PLUS_PATTERN = re.compile(rf"(\d+)\s*\+\s*{_UNIT_PATTERN}")
_AT_LEAST_PATTERN = re.compile(rf"(?:minimum|at least|min\.?)\s*(?:of\s*)?(\d+)\s*{_UNIT_PATTERN}")
_UP_TO_PATTERN = re.compile(rf"(?:up to|maximum|max\.?)\s*(\d+)\s*{_UNIT_PATTERN}")
_OVER_PATTERN = re.compile(rf"(?:over|more than)\s*(\d+)\s*{_UNIT_PATTERN}")
_BARE_PATTERN = re.compile(rf"(\d+)\s*{_UNIT_PATTERN}")


def _to_months(number_text: str, unit_text: str) -> int:
    number = int(number_text)
    if unit_text.startswith("year") or unit_text.startswith("yr"):
        return number * 12
    return number


def parse_experience_requirement(text: str | None) -> ExperienceRequirement:
    """
    Interpret a verbatim experience phrase into a month range.

    Patterns are tried most-specific first so e.g. a range ("3-5 years")
    is never partially matched by the bare-number pattern. A phrase with
    no recognizable number+unit (missing, empty, or genuinely unparseable
    text like "a few years" or a bare "3+" with no unit) yields
    is_specified=False with min/max left None -- this must be treated as
    "no requirement stated", never as "0 months required", by any future
    scoring logic.

    "over N years" / "more than N years" is interpreted as min=N months
    with no upper bound -- an approximation (strictly it should exceed
    N), documented here since it is a judgment call, not a fact pulled
    from the text.
    """
    if not text or not text.strip():
        return ExperienceRequirement(
            min_months=None, max_months=None, raw_text=None, is_specified=False
        )

    normalized = text.strip().lower()

    match = _RANGE_PATTERN.search(normalized)
    if match:
        lo, hi, unit = match.groups()
        return ExperienceRequirement(
            min_months=_to_months(lo, unit),
            max_months=_to_months(hi, unit),
            raw_text=text,
            is_specified=True,
        )

    match = _PLUS_PATTERN.search(normalized)
    if match:
        number, unit = match.groups()
        return ExperienceRequirement(
            min_months=_to_months(number, unit),
            max_months=None,
            raw_text=text,
            is_specified=True,
        )

    match = _AT_LEAST_PATTERN.search(normalized)
    if match:
        number, unit = match.groups()
        return ExperienceRequirement(
            min_months=_to_months(number, unit),
            max_months=None,
            raw_text=text,
            is_specified=True,
        )

    match = _UP_TO_PATTERN.search(normalized)
    if match:
        number, unit = match.groups()
        return ExperienceRequirement(
            min_months=None,
            max_months=_to_months(number, unit),
            raw_text=text,
            is_specified=True,
        )

    match = _OVER_PATTERN.search(normalized)
    if match:
        number, unit = match.groups()
        return ExperienceRequirement(
            min_months=_to_months(number, unit),
            max_months=None,
            raw_text=text,
            is_specified=True,
        )

    match = _BARE_PATTERN.search(normalized)
    if match:
        number, unit = match.groups()
        months = _to_months(number, unit)
        return ExperienceRequirement(
            min_months=months, max_months=months, raw_text=text, is_specified=True
        )

    # Text was present but no recognizable number+unit was found.
    return ExperienceRequirement(
        min_months=None, max_months=None, raw_text=text, is_specified=False
    )


# EDUCATION REQUIREMENT PARSING


def _find_education_level(normalized_text: str) -> EducationLevel | None:
    # Multiple terms may match (e.g. "Bachelor's or Master's"); the
    # lowest level found is treated as the practical minimum, since an
    # "X or Y" phrasing states the floor, not the ceiling.
    found = [
        EducationLevel[level_name]
        for term, level_name in EDUCATION_LEVEL_TERMS.items()
        if term in normalized_text
    ]
    return min(found) if found else None


_FIELD_OF_STUDY_PATTERN = re.compile(r"\bin\s+([a-z0-9 ,/&\-]+)", re.IGNORECASE)
_FIELD_OF_STUDY_SPLIT_PATTERN = re.compile(r"\s*,\s*or\s+|\s+or\s+|,\s*")


def _derive_fields_of_study(text: str) -> list[str]:
    """
    Best-effort deterministic extraction of field-of-study names from an
    education requirement phrase, e.g. "Bachelor's in Computer Science
    or related field" -> ["Computer Science", "related field"].

    This exists because the LLM does NOT reliably supply this
    separately (see app.schemas.RawJobRequirementsExtraction's
    docstring): adding a third structured-output field for it
    destabilized the (reliable) education_text/experience_text
    extraction on this repo's 3B local model. Deriving it from the
    already-reliable education_text via regex avoids that failure mode
    entirely. Gracefully returns [] when the phrase doesn't match the
    pattern -- never guesses.
    """
    match = _FIELD_OF_STUDY_PATTERN.search(text)
    if not match:
        return []

    tail = re.split(r"\bor equivalent\b", match.group(1), flags=re.IGNORECASE)[0]
    parts = _FIELD_OF_STUDY_SPLIT_PATTERN.split(tail)

    return [part.strip().rstrip(".") for part in parts if part.strip()]


def parse_education_requirement(
    text: str | None, fields_of_study: list[str] | None = None
) -> EducationRequirement:
    """
    Interpret a verbatim education phrase into a structured requirement.

    minimum_level comes only from the curated EDUCATION_LEVEL_TERMS
    lookup -- never guessed. fields_of_study defaults to a deterministic
    regex derivation from `text` (see _derive_fields_of_study); pass an
    explicit list to override that derivation (e.g. from a test, or if
    a future extraction path supplies field names directly). is_required
    defaults to True whenever a level is found, unless an
    EDUCATION_OPTIONAL_MARKERS phrase (e.g. "preferred", "or equivalent
    experience") is present.
    """
    if not text or not text.strip():
        return EducationRequirement(
            minimum_level=None, fields_of_study=[], raw_text=None, is_required=False
        )

    normalized = text.strip().lower()
    minimum_level = _find_education_level(normalized)

    resolved_fields = (
        fields_of_study if fields_of_study is not None else _derive_fields_of_study(text)
    )

    if minimum_level is None:
        return EducationRequirement(
            minimum_level=None,
            fields_of_study=resolved_fields,
            raw_text=text,
            is_required=False,
        )

    is_optional = any(marker in normalized for marker in EDUCATION_OPTIONAL_MARKERS)

    return EducationRequirement(
        minimum_level=minimum_level,
        fields_of_study=resolved_fields,
        raw_text=text,
        is_required=not is_optional,
    )


# SENIORITY DERIVATION


_SENIORITY_ORDER = [
    Seniority.PRINCIPAL,
    Seniority.LEAD,
    Seniority.SENIOR,
    Seniority.MID,
    Seniority.JUNIOR,
    Seniority.INTERN,
]


def _scrub(text: str) -> str:
    """Lowercase and replace non-alphanumerics with single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower())


def derive_seniority(title: str | None) -> Seniority | None:
    """
    Derive an ordinal Seniority from a job title via curated term
    matching (app.taxonomy.SENIORITY_TERMS), word-boundary safe (a
    scrubbed comparison, so "Sr." and "Sr" match the same term, and
    "lead" does not spuriously match inside an unrelated word).

    When a title matches more than one term (e.g. "Senior Staff
    Engineer" matches both "senior" and "staff"), the MOST senior level
    found wins -- a deliberate choice, since a combined title is at
    least as senior as its most senior modifier.
    """
    if not title:
        return None

    scrubbed_title = f" {_scrub(title)} "

    matched_levels = set()
    for term, level_name in SENIORITY_TERMS.items():
        scrubbed_term = f" {_scrub(term)} "
        if scrubbed_term in scrubbed_title:
            matched_levels.add(Seniority[level_name])

    if not matched_levels:
        return None

    for level in _SENIORITY_ORDER:
        if level in matched_levels:
            return level

    return None
