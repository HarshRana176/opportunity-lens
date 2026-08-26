from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
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

    # ---- Online job-discovery additions (all nullable/additive) ----
    #
    # A job created by POST /jobs (caller-supplied text) leaves every
    # one of these NULL and behaves exactly as it always has. They are
    # populated only for jobs discovered through an external JobSource
    # (see app.services.ingest_external_listing).
    #
    # CAVEAT: Base.metadata.create_all() (app.main's lifespan) only
    # creates MISSING tables -- it does not ALTER an existing
    # job_descriptions table. A fresh database gets these columns for
    # free; an already-deployed one needs the manual migration
    # documented in README.md (no Alembic, per this repo's existing D3
    # decision on JSON columns).

    source = Column(String, nullable=True, index=True)

    external_job_id = Column(String, nullable=True, index=True)

    job_url = Column(String, nullable=True)

    company = Column(String, nullable=True)

    location = Column(String, nullable=True)

    posted_at = Column(String, nullable=True)


class CandidateProfile(Base):
    """
    Task 5 addition. New table -- does not alter `resumes` or
    `job_descriptions` in any way. JSON columns mirror both existing
    tables' storage approach (no Alembic, no JSONB migration).

    Separate from `resumes` rather than extending it: a Resume is the
    protected record of what was uploaded and parsed (its API contract
    is pinned by existing tests), while a CandidateProfile is a
    derived, re-derivable matching representation with a different
    shape and lifecycle. Keeping them apart lets the profile evolve --
    including gaining a vector column later -- without touching the
    résumé contract.

    resume_id is a nullable lineage link: a profile is normally built
    from an uploaded résumé, but the column is nullable so a profile
    can also be built directly from a PDF path without first creating
    a Resume row.
    """

    __tablename__ = "candidate_profiles"

    id = Column(Integer, primary_key=True, index=True)

    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=True, index=True)

    candidate_name = Column(String, nullable=False)

    seniority = Column(Integer, nullable=True)

    current_role = Column(String, nullable=True)

    skills = Column(JSON, nullable=True)

    total_experience_months = Column(Integer, nullable=True)

    total_experience_years = Column(Float, nullable=True)

    employment_history = Column(JSON, nullable=True)

    education = Column(JSON, nullable=True)

    projects = Column(JSON, nullable=True)
    """
    Match-orchestration addition. Additive, nullable -- an existing row
    written before this column existed simply reads back as NULL/[],
    never an error. JSON list of app.schemas.CandidateProject dumps,
    mirroring employment_history's storage approach exactly.

    CAVEAT: Base.metadata.create_all() (app.main's lifespan) only
    creates MISSING tables -- it does not ALTER an already-existing
    candidate_profiles table in a live database. A fresh database gets
    this column for free; an already-deployed one needs a manual
    `ALTER TABLE candidate_profiles ADD COLUMN projects JSON` (no
    Alembic yet, per this repo's existing D3 decision on JSON columns).
    """

    raw_text = Column(String, nullable=False)

    parse_warnings = Column(JSON, nullable=True)