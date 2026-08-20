from sqlalchemy import Column, Integer, String, Float, JSON
from app.database import Base


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)

    candidate_name = Column(String, nullable=False)

    technical_stack = Column(JSON, nullable=True)

    employment_history = Column(JSON, nullable=True)

    total_experience_months = Column(Integer, nullable=True)

    total_experience_years = Column(Float, nullable=True)


class JobDescription(Base):
    """
    Task 4 addition. New table -- does not alter the `resumes` table or
    its schema in any way. JSON columns mirror Resume's storage
    approach (D3: no Alembic, no JSONB migration yet). seniority is
    stored as a plain Integer (the underlying value of the
    app.schemas.Seniority IntEnum) for the same reason
    total_experience_months is a plain Integer on Resume.
    """

    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String, nullable=False)

    seniority = Column(Integer, nullable=True)

    required_skills = Column(JSON, nullable=True)

    preferred_skills = Column(JSON, nullable=True)

    experience = Column(JSON, nullable=True)

    education = Column(JSON, nullable=True)

    responsibilities = Column(JSON, nullable=True)

    raw_text = Column(String, nullable=False)

    parse_warnings = Column(JSON, nullable=True)