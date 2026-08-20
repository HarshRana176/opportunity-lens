"""
Characterizes:
  1. Pydantic validation rules on the extraction schemas
     (RawResumeExtraction, EmploymentPeriod, TechnicalStack,
     ResumeExtraction, SkillCategory).
  2. The extract_resume() pipeline wiring, with PyMuPDF and the LLM
     extraction chain both stubbed -- no real PDF file or Ollama call
     is involved.
"""
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import app.extractor as extractor
from app.extractor import (
    EmploymentPeriod,
    RawResumeExtraction,
    ResumeExtraction,
    SkillCategory,
    TechnicalStack,
    extract_resume,
)


class TestRawResumeExtractionSchema:
    def test_candidate_name_is_required(self):
        with pytest.raises(ValidationError):
            RawResumeExtraction()

    def test_employment_history_and_skills_default_to_empty_lists(self):
        result = RawResumeExtraction(candidate_name="Jane Doe")

        assert result.employment_history == []
        assert result.skills == []


class TestEmploymentPeriodSchema:
    def test_company_start_date_and_end_date_are_required(self):
        with pytest.raises(ValidationError):
            EmploymentPeriod(company="Acme")

    def test_role_defaults_to_none(self):
        period = EmploymentPeriod(
            company="Acme", start_date="May 2025", end_date="Present"
        )
        assert period.role is None

    def test_end_date_of_none_is_rejected(self):
        # Ties to the "incomplete employment dates" requirement: a
        # genuinely missing end_date must be a validation error, not a
        # value that later gets treated as "Present" by accident.
        with pytest.raises(ValidationError):
            EmploymentPeriod(company="Acme", start_date="May 2025", end_date=None)

    def test_empty_string_end_date_is_accepted_by_the_schema(self):
        # The schema itself only requires `str`, not a non-empty one.
        # Emptiness is handled downstream by parse_resume_date raising
        # ValueError, which calculate_total_experience catches and skips
        # (see test_experience.py::test_incomplete_end_date_is_not_assumed_to_be_present).
        period = EmploymentPeriod(company="Acme", start_date="May 2025", end_date="")
        assert period.end_date == ""


class TestSkillCategorySchema:
    @pytest.mark.parametrize(
        "category", ["programming_language", "framework", "tool", "exclude"]
    )
    def test_accepts_each_documented_category(self, category):
        assert SkillCategory(category=category).category == category

    def test_rejects_a_category_outside_the_literal(self):
        with pytest.raises(ValidationError):
            SkillCategory(category="database")


class TestResumeExtractionSchema:
    def test_requires_experience_and_technical_stack(self):
        with pytest.raises(ValidationError):
            ResumeExtraction(candidate_name="Jane Doe")

    def test_accepts_a_fully_populated_result(self):
        result = ResumeExtraction(
            candidate_name="Jane Doe",
            technical_stack=TechnicalStack(programming_languages=["Python"]),
            employment_history=[],
            total_experience_months=12,
            total_experience_years=1.0,
        )
        assert result.candidate_name == "Jane Doe"
        assert result.technical_stack.programming_languages == ["Python"]


class _FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self, mode):
        assert mode == "text"
        return self._text


class _FakeDoc:
    def __init__(self, pages_text):
        self._pages = [_FakePage(t) for t in pages_text]

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


class TestExtractResumePipeline:
    def test_raises_on_pdf_with_no_extractable_text(self, monkeypatch):
        monkeypatch.setattr(
            extractor.pymupdf, "open", lambda path: _FakeDoc(["", "   "])
        )
        invoke = Mock()
        monkeypatch.setattr(extractor, "extraction_chain", Mock(invoke=invoke))

        with pytest.raises(ValueError, match="Could not extract text"):
            extract_resume("fake.pdf")

        # Must fail fast, before ever calling the LLM.
        invoke.assert_not_called()

    def test_wires_raw_extraction_into_a_complete_result(self, monkeypatch):
        monkeypatch.setattr(
            extractor.pymupdf, "open", lambda path: _FakeDoc(["Jane Doe resume text"])
        )

        raw = RawResumeExtraction(
            candidate_name="Jane Doe",
            employment_history=[
                EmploymentPeriod(
                    company="Acme", start_date="May 2025", end_date="Aug 2025"
                )
            ],
            skills=["python", "rag"],
        )
        monkeypatch.setattr(
            extractor, "extraction_chain", Mock(invoke=Mock(return_value=raw))
        )

        result = extract_resume("fake.pdf")

        assert isinstance(result, ResumeExtraction)
        assert result.candidate_name == "Jane Doe"
        # "rag" is excluded, "python" is deterministic -- no LLM fallback
        # call needed, so no classifier stub is required for this test.
        assert result.technical_stack.programming_languages == ["python"]
        # May 2025 -> Aug 2025 inclusive == 4 months; matches
        # test_experience.py::TestDurationSemantics::test_four_month_period_is_four_months.
        assert result.total_experience_months == 4
