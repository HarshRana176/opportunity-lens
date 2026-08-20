"""
Application-layer orchestration for resume upload, listing, and lookup,
and (Task 4) job-description creation, listing, and lookup.

Business logic (PDF extraction, experience calculation, skill
categorization; JD parsing, requirement interpretation) stays in
app.extractor / app.experience / app.skills / app.job_extractor /
app.requirements; this module only coordinates storage/extraction and
persistence, and owns failure cleanup so routes in app.main can stay
thin HTTP adapters.
"""
from sqlalchemy.orm import Session

from app import models
from app.extractor import extract_resume
from app.job_extractor import extract_job
from app.storage import cleanup, save_upload


def create_resume_from_upload(
    db: Session,
    file_obj,
    original_filename: str | None,
) -> models.Resume:
    """
    Store the uploaded file, run it through the extraction pipeline,
    and persist the result.

    On any failure after the file has been stored -- extraction error
    or a database error -- the stored file is removed, the session is
    rolled back, and the exception is re-raised for the caller to map
    to an HTTP response. A failure during storage itself (bad size,
    bad content) has already cleaned up after itself in app.storage
    and is simply re-raised here.
    """
    final_path, _sanitized_original = save_upload(file_obj, original_filename)

    try:
        result = extract_resume(str(final_path))

        resume = models.Resume(
            candidate_name=result.candidate_name,
            technical_stack=result.technical_stack.model_dump(),
            employment_history=[
                item.model_dump() for item in result.employment_history
            ],
            total_experience_months=result.total_experience_months,
            total_experience_years=result.total_experience_years,
        )

        db.add(resume)
        db.commit()
        db.refresh(resume)

        return resume

    except Exception:
        db.rollback()
        cleanup(final_path)
        raise


def list_resumes(db: Session, limit: int, offset: int) -> list[models.Resume]:
    return (
        db.query(models.Resume)
        .order_by(models.Resume.id)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_resume(db: Session, resume_id: int) -> models.Resume | None:
    return (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id)
        .first()
    )


# ---------------------------------------------------------------------------
# Task 4 additions below (Job Description parsing).
# ---------------------------------------------------------------------------


def create_job_from_text(db: Session, job_text: str) -> models.JobDescription:
    """
    Run job_text through the JD extraction pipeline and persist the
    result. Unlike create_resume_from_upload, there is no file to store
    or clean up (D1: text-only JSON input) -- extract_job() either
    raises (propagated to the caller to map to an HTTP response) or
    returns a complete JobProfile; only the database write itself needs
    a rollback-on-failure guard.
    """
    profile = extract_job(job_text)

    job = models.JobDescription(
        title=profile.title,
        seniority=int(profile.seniority) if profile.seniority is not None else None,
        required_skills=[skill.model_dump() for skill in profile.required_skills],
        preferred_skills=[skill.model_dump() for skill in profile.preferred_skills],
        experience=profile.experience.model_dump(),
        education=profile.education.model_dump(),
        responsibilities=profile.responsibilities,
        raw_text=profile.raw_text,
        parse_warnings=profile.parse_warnings,
    )

    try:
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise


def list_jobs(db: Session, limit: int, offset: int) -> list[models.JobDescription]:
    return (
        db.query(models.JobDescription)
        .order_by(models.JobDescription.id)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_job(db: Session, job_id: int) -> models.JobDescription | None:
    return (
        db.query(models.JobDescription)
        .filter(models.JobDescription.id == job_id)
        .first()
    )
