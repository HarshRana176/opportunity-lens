from typing import Literal, Optional

from pydantic import BaseModel, Field


class EmploymentPeriod(BaseModel):
    company: str = Field(
        description="Employer/company name exactly as written."
    )

    role: Optional[str] = Field(
        default=None,
        description="Job title exactly as written."
    )

    start_date: str = Field(
        description="Employment start date exactly as written."
    )

    end_date: str = Field(
        description=(
            "Employment end date exactly as written. "
            "If ongoing, return exactly 'Present'."
        )
    )


class RawResumeExtraction(BaseModel):
    candidate_name: str = Field(
        description="Candidate's full name exactly as written."
    )

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete technical technologies explicitly "
            "mentioned in the resume. Return as a flat list."
        )
    )


class TechnicalStack(BaseModel):
    programming_languages: list[str] = Field(
        default_factory=list
    )

    frameworks: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )


class ResumeExtraction(BaseModel):
    candidate_name: str

    technical_stack: TechnicalStack

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    total_experience_months: int

    total_experience_years: float


class SkillCategory(BaseModel):

    category: Literal[
        "programming_language",
        "framework",
        "tool",
        "exclude"
    ]
