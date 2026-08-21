"""
Deterministic interpretation of a candidate's education records.

Pure Python, no LLM calls, no I/O -- mirrors app.requirements'
separation for the JD side: the LLM (app.llm.education_extraction_chain,
via app.candidate_extractor) extracts education facts verbatim; this
module normalizes and canonicalizes them.

Does not import from app.extractor, app.job_extractor, or
app.candidate_extractor.
"""
import re

from app.schemas import EducationBackground, EducationLevel, EducationRecord, RawEducationRecord
from app.taxonomy import DEGREE_CANONICAL

# Parenthetical qualifiers ("(Hons)", "(Distinction)") describe how a
# degree was awarded, not a different degree -- stripped (replaced with
# a space, so words on either side don't glue together) before anything
# else.
_PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")

# Used for the final exact-match key: every character that is not a
# letter or digit is removed -- not just '.' and whitespace, but also
# apostrophes ("Bachelor's" -> "bachelors"), commas, slashes, hyphens,
# etc. One general squash rule instead of hand-enumerating a
# punctuation variant for every degree in DEGREE_CANONICAL.
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")

# A comma or slash sometimes separates two equivalent degree
# abbreviations ("B.A./B.S.") or a degree from a field that leaked into
# the raw degree text ("M.S., Computer Science") -- used only as a
# fallback when the whole string doesn't resolve; see
# _resolve_level_name.
_SEGMENT_SPLIT_PATTERN = re.compile(r"[,/]")


def _prepare(degree_text: str) -> str:
    """Lowercase and drop parenthetical qualifiers; punctuation/whitespace stay."""
    lowered = degree_text.strip().lower()
    return _PARENTHETICAL_PATTERN.sub(" ", lowered)


def compute_degree_key(degree_text: str) -> str:
    """
    Reduce a degree string to an exact-match identity key: parenthetical
    qualifiers dropped, then every non-alphanumeric character removed.
    "B. Tech", "B.Tech", "BTech", "Bachelor's", and "B.Tech (Hons)" all
    collapse to the same key family DEGREE_CANONICAL expects.

    Deliberately produces an EXACT key for DEGREE_CANONICAL lookup, not
    a substring-search input -- see DEGREE_CANONICAL's docstring in
    app.taxonomy for why substring matching on short degree
    abbreviations is unsafe (e.g. "BA" inside "MBA").

    This is always the WHOLE-STRING identity key, used as the record's
    stable degree_key regardless of whether it resolves to a level --
    see _resolve_level_name for the (separate) segment-based fallback
    used only for level resolution, which does not change what this
    function returns.
    """
    return _NON_ALPHANUMERIC_PATTERN.sub("", _prepare(degree_text))


def _resolve_level_name(degree_text: str) -> str | None:
    """
    Look up the EducationLevel NAME for a degree, trying progressively
    less specific interpretations and never guessing:

      1. The whole string as one exact key (compute_degree_key).
      2. If that fails and the text contains a comma or slash, each
         comma/slash-separated segment in turn, using the first segment
         that resolves ("B.A./B.S." -> "B.A." resolves to BACHELORS;
         "M.S., Computer Science" -> "M.S." resolves to MASTERS). This
         is safe specifically because BOTH sides of such a separator
         are, in every case this handles, different phrasings of the
         SAME level -- never a level judgment call.

    Returns None if nothing resolves at either step.
    """
    whole_key = compute_degree_key(degree_text)
    if whole_key in DEGREE_CANONICAL:
        return DEGREE_CANONICAL[whole_key]

    prepared = _prepare(degree_text)
    for segment in _SEGMENT_SPLIT_PATTERN.split(prepared):
        segment_key = _NON_ALPHANUMERIC_PATTERN.sub("", segment)
        if segment_key in DEGREE_CANONICAL:
            return DEGREE_CANONICAL[segment_key]

    return None


def normalize_education_record(raw: RawEducationRecord) -> EducationRecord:
    """
    Resolve one raw education record against the curated
    DEGREE_CANONICAL table.

    Every raw field is preserved exactly as extracted regardless of
    whether resolution succeeds -- normalization failing must never
    rewrite or discard the original wording (e.g. "Class X" stays
    degree_raw="Class X" even though it resolves to HIGH_SCHOOL; an
    unrecognized degree stays exactly as written with level=None,
    resolution="unresolved", and is still returned, never dropped). An
    empty string in an optional raw field (some LLM responses use ""
    rather than null for "not mentioned") is preserved as-is, exactly
    like any other raw value -- never coerced to None or vice versa.

    completion_raw is passed through untouched: it is never parsed into
    a date, since résumé completion text is sometimes a year, sometimes
    a percentage/CGPA line, sometimes "In Progress" -- none of which
    should be coerced into a fabricated date value.
    """
    degree_key = compute_degree_key(raw.degree)
    level_name = _resolve_level_name(raw.degree)

    return EducationRecord(
        degree_raw=raw.degree,
        field_of_study_raw=raw.field_of_study,
        institution_raw=raw.institution,
        completion_raw=raw.completion_text,
        degree_key=degree_key,
        level=EducationLevel[level_name] if level_name else None,
        resolution="taxonomy" if level_name else "unresolved",
    )


def build_education_background(
    raw_records: list[RawEducationRecord], raw_text: str | None
) -> tuple[EducationBackground | None, list[str]]:
    """
    Normalize every extracted education record into an
    EducationBackground, or return (None, []) when the résumé had no
    education section at all.

    Returning None here (as opposed to an EducationBackground with an
    empty `records` list) is what lets
    CandidateProfile.education distinguish "no education section found"
    from "an education section was found but nothing in it could be
    canonicalized" -- collapsing these would force a future matcher to
    guess which case it is looking at.

    highest_level is the maximum EducationLevel across records whose
    degree resolved; None when no record resolved (which is NOT the
    same as this function returning None outright -- see above).

    A warning is produced for each record whose degree could not be
    resolved, so the caller can surface it without treating it as an
    error: the record itself is still retained in full, unresolved.
    """
    if not raw_records:
        return None, []

    records = [normalize_education_record(r) for r in raw_records]

    warnings = [
        f"Could not determine education level for degree: {record.degree_raw!r}"
        for record in records
        if record.resolution == "unresolved"
    ]

    resolved_levels = [record.level for record in records if record.level is not None]
    highest_level = max(resolved_levels) if resolved_levels else None

    return (
        EducationBackground(
            records=records,
            highest_level=highest_level,
            raw_text=raw_text,
        ),
        warnings,
    )
