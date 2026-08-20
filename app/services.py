"""
Application-layer orchestration for resume upload, listing, and lookup.

Business logic (PDF extraction, experience calculation, skill
categorization) stays in app.extractor / app.experience / app.skills;
this module only coordinates storage, extraction, and persistence, and
owns failure cleanup so routes in app.main can stay thin HTTP adapters.
"""
from sqlalchemy.orm import Session

from app import models
from app.extractor import extract_resume
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
