"""
High-level resume extraction pipeline orchestration.

extract_resume() wires together the focused modules that used to live
here as one file:

    app.pdf         -- PDF -> raw text (external boundary: PyMuPDF)
    app.llm         -- LLM client, prompts, chains (external boundary: Ollama)
    app.experience  -- deterministic date parsing + duration calculation
    app.skills      -- deterministic + LLM-fallback skill categorization
    app.schemas     -- Pydantic contracts shared across all of the above
    app.taxonomy    -- static known/excluded technology tables

The names below are re-exported from their new homes so existing imports
of `app.extractor.<name>` (including test monkeypatches of
`extractor.pymupdf` and `extractor.extraction_chain`, both of which are
looked up from THIS module's globals at call time inside
extract_resume()) keep working unchanged.
"""
import pymupdf  # noqa: F401 -- re-exported; see module docstring

from app.experience import (  # noqa: F401
    _PRESENT_VALUES,
    calculate_total_experience,
    date_to_month_index,
    parse_resume_date,
)
from app.llm import extraction_chain
from app.pdf import extract_text_from_pdf
from app.schemas import (  # noqa: F401
    EmploymentPeriod,
    RawResumeExtraction,
    ResumeExtraction,
    SkillCategory,
    TechnicalStack,
)
from app.skills import build_technical_stack, classify_unknown_skill  # noqa: F401
from app.taxonomy import EXCLUDED_TECHNOLOGIES, SKILL_CATEGORIES  # noqa: F401


def extract_resume(pdf_path: str) -> ResumeExtraction:

    # PDF -> text

    full_text = extract_text_from_pdf(
        pdf_path
    )

    if not full_text.strip():

        raise ValueError(
            "Could not extract text from the PDF."
        )

    # LLM extracts facts

    raw_result = extraction_chain.invoke({
        "resume_text": full_text
    })

    # Python calculates experience

    experience = calculate_total_experience(
        raw_result.employment_history
    )

    # Python categorizes technical stack

    technical_stack = build_technical_stack(
        raw_result.skills
    )

    # Final structured result

    result = ResumeExtraction(

        candidate_name=raw_result.candidate_name,

        technical_stack=technical_stack,

        employment_history=(
            raw_result.employment_history
        ),

        total_experience_months=(
            experience["months"]
        ),

        total_experience_years=(
            experience["years"]
        ),
    )

    return result
