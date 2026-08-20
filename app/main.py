import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends

from app.extractor import extract_resume
from app.database import engine, Base, get_db
from app import models


app = FastAPI(
    title="Resume Parser API",
    version="1.0.0"
)


# Create database tables
Base.metadata.create_all(bind=engine)


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Root endpoint


@app.get("/")
def root():
    return {
        "message": "Resume Parser API is running"
    }


# Upload and parse resume


@app.post("/resumes")
async def upload_resume(
    file: UploadFile = File(...),
    db=Depends(get_db)
):

    # Only accept PDF files
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Create file path
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    # Save uploaded PDF
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run resume extraction pipeline
    result = extract_resume(file_path)

    # Create database record
    resume = models.Resume(
        candidate_name=result.candidate_name,
        technical_stack=result.technical_stack.model_dump(),
        employment_history=[
            item.model_dump()
            for item in result.employment_history
        ],
        total_experience_months=result.total_experience_months,
        total_experience_years=result.total_experience_years
    )

    # Save record to PostgreSQL
    db.add(resume)
    db.commit()
    db.refresh(resume)

    # Return extracted result
    return result.model_dump()


# Get all resumes


@app.get("/resumes")
def get_resumes(
    db=Depends(get_db)
):

    resumes = db.query(models.Resume).all()

    return [
        {
            "id": resume.id,
            "candidate_name": resume.candidate_name,
            "technical_stack": resume.technical_stack,
            "employment_history": resume.employment_history,
            "total_experience_months": resume.total_experience_months,
            "total_experience_years": resume.total_experience_years
        }
        for resume in resumes
    ]


# Get a single resume by ID


@app.get("/resumes/{resume_id}")
def get_resume(
    resume_id: int,
    db=Depends(get_db)
):

    resume = (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id)
        .first()
    )

    if resume is None:
        raise HTTPException(
            status_code=404,
            detail="Resume not found"
        )

    return {
        "id": resume.id,
        "candidate_name": resume.candidate_name,
        "technical_stack": resume.technical_stack,
        "employment_history": resume.employment_history,
        "total_experience_months": resume.total_experience_months,
        "total_experience_years": resume.total_experience_years
    }