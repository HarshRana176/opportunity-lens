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