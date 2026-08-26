# OpportunityLens

An end-to-end AI-assisted job-matching API. It accepts a PDF résumé, extracts structured
candidate information and projects, discovers real online jobs through Adzuna, extracts
structured job requirements, evaluates candidate/job compatibility (including optional project
evidence), computes a deterministic weighted score, and returns ranked jobs with explanations
and application URLs.

## Key features

- PDF résumé parsing (PyMuPDF)
- LLM-based structured résumé/project extraction (Ollama, Qwen 2.5 3B)
- Structured job-description extraction from free text
- Online job discovery via Adzuna, with country-scoped search (e.g. India via `ADZUNA_COUNTRY=in`)
- Deduplication/reuse of previously ingested jobs across searches
- Required/preferred skill matching, experience matching, education matching, seniority matching
- Project evidence: technology overlap, evidence-depth classification, embedding-based similarity
- Deterministic weighted scoring and ranking (no LLM ever assigns a score directly)
- PostgreSQL persistence via SQLAlchemy
- FastAPI + Swagger/OpenAPI documentation
- Idempotent database migration support for existing deployments

## Architecture

```text
Resume PDF
   ↓
PyMuPDF
   ↓
Ollama (Qwen 2.5 3B)
   ↓
CandidateProfile + Projects
   ↓
Adzuna
   ↓
Online Job Listings
   ↓
Ollama JD Extraction
   ↓
MatchEvidence
   ├── Required Skills
   ├── Preferred Skills
   ├── Experience
   ├── Education
   ├── Seniority
   └── Project Evidence (opt-in)
          ├── Technology overlap
          ├── Evidence depth
          └── Embedding similarity
   ↓
Deterministic Scoring (app/scoring.py)
   ↓
Ranked MatchResult
```

LLMs and the embedding model only ever produce **evidence** — extracted facts, classifications,
similarity scores. `app/scoring.py` is the only place a number becomes a score: it is pure
Python, with no LLM call, no I/O, and no randomness. No model ever decides a match score
directly.

## Project evidence

`project_evidence_weight` (a request parameter on `/job-matches`/`/match`) controls an optional
**sixth** scoring component built from the candidate's projects:

- **Evidence depth** — `title_only` / `tutorial_or_basic` / `substantive`, the primary signal,
  classified by the LLM only when a project has real description/outcome text.
- **Technology overlap** — deterministic: does the project actually name any of the job's
  required/preferred skills.
- **Embedding similarity** — the project's text embedded and compared against the job's
  responsibilities.

At `0` or omitted (the default), none of the above runs — no extra LLM/embedding calls are
made, and scoring produces exactly the original five-component result. At `> 0`, it is added as
one additional weighted component. In both cases it **never modifies** eligibility, hard
constraints, experience, education, seniority, or required-skills evidence — those are decided
before project evidence is ever computed.

## Online job discovery

`/job-matches` does not just match against whatever is already in the database — it actively
searches for new listings. **Adzuna** is the current (and only) online job source: the app
calls Adzuna's public search API, runs each returned listing through the same job-description
extraction used for manually submitted JDs, persists it, and only then matches the candidate
against the full pool of persisted jobs. Listings already seen (identified by
`source` + Adzuna's own listing id) are reused rather than re-extracted. This is a documented
API integration, not scraping.

Country is configurable via `ADZUNA_COUNTRY` (default `gb`); pass `where=India` and set
`ADZUNA_COUNTRY=in` to search the Indian market specifically.

## Main API

### `POST /job-matches`

The primary, product-facing endpoint. One call: upload a résumé, get back ranked real jobs.

**Request** (`multipart/form-data`):

| field | meaning |
|---|---|
| `file` | the résumé PDF (required) |
| `project_evidence_weight` | optional weight for the project-evidence component (default: off) |
| `search_online` | query Adzuna for new listings (default `true`) |
| `what` | search keywords (default: derived from the résumé's current role/skills) |
| `where` | location filter |
| `limit` | max listings to fetch this call (default `10`, hard cap `50`) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/job-matches \
  -F "file=@resume.pdf" \
  -F "project_evidence_weight=2" \
  -F "search_online=true" \
  -F "what=AI Engineer" \
  -F "where=India" \
  -F "limit=5"
```

**Response**, at a high level: which `CandidateProfile` was created, a `discovery` block
describing how the online search went (status, counts fetched/newly ingested/reused/failed —
never credentials), and a `matches` list ranked by score descending. Each match includes the
job's title, company, location, source, and application URL, plus a `result` with the overall
score, the per-dimension evidence behind it, and eligibility — so every score is explainable,
never a bare number. Full schema: `/docs`.

### Other endpoints

- `POST /candidate-profiles` — upload a résumé and persist a `CandidateProfile` without
  searching for jobs.
- `POST /match` — score one already-persisted candidate against one already-persisted job
  (the low-level primitive `/job-matches` is built on).

## Quick start

```powershell
# 1. Python environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# edit .env: DATABASE_URL, and optionally ADZUNA_APP_ID / ADZUNA_APP_KEY

# 3. Ollama models
ollama pull qwen2.5:3b          # required — all extraction steps
ollama pull nomic-embed-text    # optional — only used when project_evidence_weight > 0

# 4. Apply the database migration (existing databases only — see below)
python scripts/apply_migrations.py

# 5. Run
uvicorn app.main:app --reload
```

`.env` (copy from `.env.example`, placeholders only):

```
DATABASE_URL=postgresql://postgres:password@localhost:5432/resume_parser
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
ADZUNA_COUNTRY=in
```

Without Adzuna credentials, `/job-matches` still works — it skips online discovery and matches
against whatever jobs are already persisted (`discovery.status: "not_configured"`).

The API serves at `http://127.0.0.1:8000`; interactive Swagger docs are at
`http://127.0.0.1:8000/docs`.

## Database migration

`app.main`'s startup runs `Base.metadata.create_all()`, which creates **missing tables only** —
it never `ALTER`s an existing one. A database created before the job-matching columns existed
(`candidate_profiles.projects`, and `job_descriptions.source/external_job_id/job_url/company/
location/posted_at`) will be missing them, and inserts will fail with `UndefinedColumn`.

`migrations/001_add_job_matching_columns.sql` closes that gap — every statement is
`ADD COLUMN IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS`, so it is safe to run repeatedly.
Apply it with:

```powershell
python scripts/apply_migrations.py            # apply
python scripts/apply_migrations.py --check    # report drift only, changes nothing
```

This repository does not use Alembic; a future schema change gets its own numbered
`migrations/*.sql` file instead.

## Testing

```powershell
pytest
```

Latest verified run: **1312 passed, 2 skipped**. Coverage includes deterministic scoring and
reproducibility, hard-eligibility invariants, project-evidence evaluation (both the weight-off
and weight-on paths), online job discovery and deduplication, the full `/job-matches`
orchestration, and a structural guard ensuring production code under `app/` never imports
evaluation artifacts.

## Security

- Real credentials (`DATABASE_URL`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`) live only in `.env`,
  which is listed in `.gitignore` and is never committed.
- `.env.example` contains placeholder values only.
- Adzuna's app key is read from the environment and never appears in a log line, an API
  response, persisted job data, or an exception message.

## Limitations

- Adzuna's search API returns a truncated description snippet, not the full posting —
  requirements stated only in the untruncated text parse as unspecified, which the matcher
  reports as `unknown` rather than pass or fail.
- Online job discovery currently has one source: Adzuna.
- Listings fetched/ranked per search are bounded by `limit` (default 10, hard cap 50).
- This is a backend/API project — there is no consumer-facing frontend; interaction today is
  via `/docs` (Swagger) or direct HTTP calls.

## Project structure

```text
app/
  main.py                 FastAPI app and routes, incl. POST /job-matches
  services.py              orchestration: upload → extract → discover → match → rank
  candidate_extractor.py   résumé → CandidateProfile (incl. projects)
  job_extractor.py         job text → JobProfile (structured requirements)
  job_sources.py           JobSource protocol + offline StaticJobSource
  adzuna.py                Adzuna JobSource implementation
  matching.py              MatchEvidence construction, eligibility
  scoring.py               deterministic weighted scoring
  project_relevance.py     project-evidence computation
  semantic_match.py        per-employment semantic similarity (see Future work)
  models.py                SQLAlchemy models
  schemas.py               Pydantic schemas (all request/response/internal types)

migrations/                idempotent SQL migrations
scripts/                   apply_migrations.py
tests/                     pytest suite (1312 passed, 2 skipped)
evaluation/                offline evaluation harness/datasets (not imported by app/)
```

## Design / engineering highlights

- Every stage has a typed, validated Pydantic schema — no dict-shaped data crosses a module
  boundary.
- Scoring is deterministic Python, separate from every LLM/embedding call that produces the
  evidence it scores — a score is always reproducible from its evidence.
- Hard eligibility (required skills, experience, required education) is decided independently
  of, and before, the weighted score — a candidate is never scored favorably into eligibility.
- Project evidence is strictly additive: omitted or zero-weight requests are byte-identical to
  the original five-component result.
- Online job discovery is a separate, swappable layer (`JobSource` protocol) from job-text
  extraction and matching — adding a second job source touches no matching or scoring code.
- Discovered jobs are persisted and deduplicated by `(source, external_job_id)`, so repeat
  searches reuse prior extraction work instead of re-running the LLM.
- 1312 passing tests guard scoring reproducibility, eligibility invariants, and the discovery/
  ingestion/matching pipeline against regressions.

## Demo

```bash
curl -X POST http://127.0.0.1:8000/job-matches \
  -F "file=@resume.pdf" -F "where=India" -F "limit=5"
```

Or use the interactive Swagger UI at `http://127.0.0.1:8000/docs` to upload a résumé and try
`/job-matches` directly in the browser.

## Future work

- Wire per-employment semantic similarity (`app/semantic_match.py`, implemented and tested) into
  the `/job-matches` scoring path — it currently exists as evidence infrastructure but is not yet
  attached in `app/services.py`, so `MatchEvidence.semantic` is always `None` in production today.
- Additional online job sources beyond Adzuna.
- A consumer-facing frontend (none exists yet — this project is currently API-only).
