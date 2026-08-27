from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.exceptions import OutputParserException

from app import services
from app.database import Base, get_db, get_engine
from app.schemas import (
    CandidateProfileResponse,
    JobCreateRequest,
    JobResponse,
    JobSearchResponse,
    MatchRequest,
    MatchResult,
    RankedJobMatch,
    ResumeResponse,
)
from app.storage import InvalidUploadError, UploadTooLargeError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    Base.metadata.create_all(bind=get_engine())
    yield


app = FastAPI(
    title="Resume Parser API",
    version="1.0.0",
    lifespan=lifespan,
)

# Local-dev frontend (Vite) runs on a different origin than this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint


@app.get("/")
def root():
    return {
        "message": "Resume Parser API is running"
    }


# Liveness endpoint -- intentionally does not query PostgreSQL or call
# Ollama; it only confirms the process is up and serving requests.


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# Upload and parse resume


@app.post("/resumes", response_model=ResumeResponse)
def upload_resume(
    file: UploadFile = File(...),
    db=Depends(get_db)
):

    # Only accept PDF files
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        resume = services.create_resume_from_upload(
            db, file.file, file.filename
        )

    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc)
        ) from exc

    except InvalidUploadError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except OutputParserException as exc:
        # langchain_core.exceptions.OutputParserException is itself a
        # ValueError subclass (confirmed empirically: raised by
        # PydanticOutputParser when the LLM's structured output fails
        # schema validation -- a realistic failure mode for a small
        # local model, not a malformed-PDF case). Caught here, before
        # the generic ValueError branch below, so it is not
        # misreported as "the PDF has no extractable text" with a 422
        # whose detail would otherwise leak the raw LLM completion and
        # internal pydantic validation errors to the client.
        raise HTTPException(
            status_code=503,
            detail="Extraction service returned an unusable response."
        ) from exc

    except ValueError as exc:
        # extract_resume() raises a plain ValueError when the PDF has
        # no extractable text -- the only ValueError this branch is
        # meant to handle now that OutputParserException (also a
        # ValueError subclass) is caught separately above.
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except httpx.ConnectError as exc:
        # Empirically confirmed exception raised by langchain_ollama's
        # underlying httpx client when Ollama is unreachable (verified
        # by pointing ChatOllama at a closed port). Deliberately narrow
        # -- a genuine bug elsewhere in extraction must still surface
        # as an uncaught error, not be masked as a 503.
        raise HTTPException(
            status_code=503,
            detail="Extraction service is currently unavailable."
        ) from exc

    return resume


# Get all resumes


@app.get("/resumes", response_model=list[ResumeResponse])
def get_resumes(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db=Depends(get_db)
):

    return services.list_resumes(db, limit=limit, offset=offset)


# Get a single resume by ID


@app.get("/resumes/{resume_id}", response_model=ResumeResponse)
def get_resume(
    resume_id: int,
    db=Depends(get_db)
):

    resume = services.get_resume(db, resume_id)

    if resume is None:
        raise HTTPException(
            status_code=404,
            detail="Resume not found"
        )

    return resume


# ---------------------------------------------------------------------------
# Task 4 additions below (Job Description parsing). Text-only JSON input
# (D1) -- no file upload, so no storage/magic-byte/size-limit handling
# is needed here, unlike the resume upload route above.
# ---------------------------------------------------------------------------


# Create and parse a job description


@app.post("/jobs", response_model=JobResponse)
def create_job(
    payload: JobCreateRequest,
    db=Depends(get_db)
):

    try:
        job = services.create_job_from_text(db, payload.job_text)

    except OutputParserException as exc:
        # See the equivalent branch on the résumé upload route above --
        # same reasoning applies to JD structured-output parsing.
        raise HTTPException(
            status_code=503,
            detail="Extraction service returned an unusable response."
        ) from exc

    except ValueError as exc:
        # extract_job() raises ValueError when the JD text is empty.
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Extraction service is currently unavailable."
        ) from exc

    return job


# Get all job descriptions


@app.get("/jobs", response_model=list[JobResponse])
def get_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db=Depends(get_db)
):

    return services.list_jobs(db, limit=limit, offset=offset)


# Get a single job description by ID


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: int,
    db=Depends(get_db)
):

    job = services.get_job(db, job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job description not found"
        )

    return job


# ---------------------------------------------------------------------------
# Match-orchestration additions below: upload+persist a CandidateProfile
# (Task 5's build_candidate_profile, including projects, gains its first
# HTTP route here), and match a persisted CandidateProfile against a
# persisted JobDescription. Every route above this line is unchanged.
# ---------------------------------------------------------------------------


# Upload and parse a résumé into a CandidateProfile


@app.post("/candidate-profiles", response_model=CandidateProfileResponse)
def upload_candidate_profile(
    file: UploadFile = File(...),
    db=Depends(get_db)
):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        candidate = services.create_candidate_profile_from_upload(
            db, file.file, file.filename
        )

    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc)
        ) from exc

    except InvalidUploadError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except OutputParserException as exc:
        # Same reasoning as the equivalent branch on POST /resumes.
        raise HTTPException(
            status_code=503,
            detail="Extraction service returned an unusable response."
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Extraction service is currently unavailable."
        ) from exc

    return candidate


# Match a persisted CandidateProfile against a persisted JobDescription


@app.post("/match", response_model=MatchResult)
def match_candidate_and_job(
    payload: MatchRequest,
    db=Depends(get_db)
):

    result = services.match_candidate_to_job(
        db,
        payload.candidate_profile_id,
        payload.job_id,
        project_evidence_weight=payload.project_evidence_weight,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Candidate profile or job description not found."
        )

    return result


# ---------------------------------------------------------------------------
# Product-facing workflow: resume -> ranked jobs, in one call. POST /match
# above remains available as a low-level/internal primitive (one known
# candidate against one known job); this is the actual product surface --
# the caller never needs to know a candidate_profile_id or a job_id.
# ---------------------------------------------------------------------------


@app.post("/job-matches", response_model=JobSearchResponse)
def search_job_matches(
    file: UploadFile = File(...),
    project_evidence_weight: float | None = Form(default=None, ge=0),
    search_online: bool = Form(default=True),
    what: str | None = Form(default=None),
    where: str | None = Form(default=None),
    limit: int | None = Form(default=None, ge=1),
    db=Depends(get_db)
):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:
        candidate, results, discovery = services.create_candidate_profile_and_search_jobs(
            db, file.file, file.filename,
            project_evidence_weight=project_evidence_weight,
            search_online=search_online,
            what=what,
            where=where,
            limit=limit,
        )

    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc)
        ) from exc

    except InvalidUploadError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except OutputParserException as exc:
        # Same reasoning as the equivalent branch on POST /resumes.
        raise HTTPException(
            status_code=503,
            detail="Extraction service returned an unusable response."
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Extraction service is currently unavailable."
        ) from exc

    matches = [
        RankedJobMatch(
            job_id=job_row.id,
            job_title=job_row.title,
            result=result,
            source=job_row.source,
            job_url=job_row.job_url,
            company=job_row.company,
            location=job_row.location,
        )
        for job_row, result in results
    ]

    return JobSearchResponse(
        candidate_profile_id=candidate.id,
        matches=matches,
        discovery=discovery,
    )
