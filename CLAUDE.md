# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

An AI-powered Resume Parser API. It accepts a PDF resume upload, extracts raw text with PyMuPDF, uses a locally hosted LLM (Qwen 2.5 3B via Ollama, orchestrated with LangChain) to extract structured facts (candidate name, employment history, raw skills list), then performs experience-duration calculation and skill categorization deterministically in Python before persisting the result to PostgreSQL via SQLAlchemy.

The core design principle: the LLM only extracts facts verbatim from the resume text — it never calculates experience duration. All arithmetic (experience duration) is deterministic Python. Skill categorization is *mostly* deterministic: known technologies and explicitly excluded concepts are matched against static Python lookup tables, but a technology the LLM extracts that isn't in either table falls back to an LLM classification call (see "Skill categorization" below) — it is not fully deterministic end-to-end today.

## Commands

```powershell
# Activate the virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the API (auto-reload for development)
uvicorn app.main:app --reload
```

The API serves at `http://127.0.0.1:8000`; interactive Swagger docs (used for manual testing/validation — there is no automated test suite) are at `http://127.0.0.1:8000/docs`.

### External dependencies required to run

- **PostgreSQL** — connection string read from `DATABASE_URL` in `.env` (e.g. `postgresql://postgres:password@localhost:5432/resume_parser`). Tables are created automatically on startup via `Base.metadata.create_all`.
- **Ollama**, running locally with the model pulled: `ollama pull qwen2.5:3b`.

## Architecture

Pipeline (`app/extractor.py` is the heart of the system):

```
PDF upload (app/main.py)
  → PyMuPDF text extraction (extract_text_from_pdf)
  → LangChain + ChatOllama structured extraction (extraction_chain) → RawResumeExtraction
       (candidate_name, employment_history, flat `skills` list — facts only, no math/categorization)
  → Python: calculate_total_experience() — merges overlapping employment intervals, sums months
  → Python: build_technical_stack() — categorizes each raw skill into
       programming_languages / frameworks / tools
  → ResumeExtraction (final Pydantic model) → persisted as app.models.Resume → PostgreSQL
```

Key files:
- `app/main.py` — FastAPI app and routes: `POST /resumes` (upload+parse+persist), `GET /resumes`, `GET /resumes/{id}`. Uploaded PDFs are saved to `uploads/` (gitignored).
- `app/extractor.py` — everything LLM- and calculation-related: Pydantic schemas (`RawResumeExtraction`, `TechnicalStack`, `ResumeExtraction`, `EmploymentPeriod`), the extraction prompt, date parsing, experience-duration math, and skill categorization. This is the file to touch for any change to extraction behavior.
- `app/models.py` — SQLAlchemy `Resume` model (stores `technical_stack` and `employment_history` as JSON columns).
- `app/database.py` — engine/session setup, reads `DATABASE_URL` from `.env` via `python-dotenv`.
- `app/schemas.py`, `app/services.py` — currently empty; not wired into the app.

### Skill categorization (two-tier)

1. **`SKILL_CATEGORIES`** (extractor.py) — a static lowercase lookup table mapping known technology names to `programming_languages` / `frameworks` / `tools`. Anything in `EXCLUDED_TECHNOLOGIES` (concepts, protocols, ML architectures, methodologies — e.g. "REST API", "CNN", "RAG", "OOP") is dropped before categorization.
2. **Unknown skills** (not in either table) fall through to `classify_unknown_skill()`, which asks the LLM (`skill_classifier_chain`) to classify a single term as `programming_language` / `framework` / `tool` / `exclude`. On any error, unknown skills are excluded rather than guessed — see the comment in `classify_unknown_skill`.

When adding support for a new technology name, add it to `SKILL_CATEGORIES` (or `EXCLUDED_TECHNOLOGIES` if it's a concept/methodology, not a concrete technology) rather than relying on the LLM fallback classifier.

### Date handling

`parse_resume_date()` supports a fixed set of formats (`%d %b %Y`, `%d %B %Y`, `%b %Y`, `%B %Y`, `%Y-%m`, `%Y`) plus a set of "Present"-equivalent strings (`_PRESENT_VALUES`). Unparseable employment entries are silently skipped rather than raising, both in total-experience calculation and (implicitly) per-entry duration calculation. `calculate_total_experience()` merges overlapping employment periods before summing so concurrent jobs/internships aren't double-counted.

### Extraction prompt conventions

The system prompt in `extraction_prompt` explicitly enumerates non-technology concepts (DSA, OOP, NLP, RAG, etc.) that must not be extracted as skills — this list should stay in sync with `EXCLUDED_TECHNOLOGIES` when either is edited.

## Project direction

This repository is a resume parser today and is being evolved into a **Job Intelligence / Resume Matching Engine**. Treat the current resume parser (extraction, experience calculation, skill categorization) as the working baseline — planned capabilities build on top of it, not replace it. Planned capabilities, roughly in dependency order:

- Robust resume parsing (hardening of the existing pipeline)
- Job description parsing
- Structured candidate profiles
- Structured job profiles
- Embeddings (for candidate/job semantic representation)
- PostgreSQL with **pgvector** — for storing and querying those embeddings
- Explainable candidate/job matching
- Match scoring
- Ranking
- **LangGraph**-based workflow orchestration — for coordinating the multi-step parse → embed → match → score pipeline

Notes on scope:
- `pgvector` and `LangGraph` are on the roadmap because they serve specific needs (vector similarity search; multi-step workflow coordination) — do not introduce either speculatively or "because the roadmap mentions it." Each addition should have a clear architectural reason tied to the feature being built, stated before implementation.
- Keep the extraction/deterministic-calculation split established in `app/extractor.py`: LLMs extract or interpret unstructured text (resume text, job descriptions); deterministic Python owns calculations, scoring components, validation, and any logic that doesn't require language understanding. Match scoring and ranking should follow this same split — LLM involvement should be justified (e.g. explaining *why* a match scored the way it did), not used for the scoring arithmetic itself.
- When LangGraph is introduced, it should orchestrate control flow between steps/services — not contain business logic inline in graph nodes. Business logic (parsing, scoring, matching) should live in testable service functions/modules that graph nodes call into.

## Working agreements

- **Incremental over rewrite.** Prefer small, additive changes. Do not remove or rewrite working extraction, experience-calculation, or skill-categorization behavior without a clear reason and validation that the replacement behaves at least as well.
- **Propose before major changes.** Before any major architectural change (new orchestration layer, new storage engine, schema changes, pipeline restructuring), inspect the relevant existing code first, then explain the proposed design and which files it touches, and wait for approval before implementing.
- **Implementation workflow:** inspect → plan → implement → validate/test → diagnose failures → fix → validate again. If a test or command fails, diagnose and fix it rather than stopping or working around it.
- **Secrets and sensitive data.** Never read, print, commit, or modify `.env` contents unless a task explicitly requires a configuration change there — and confirm with the user before editing it. Never commit resume files (`uploads/`), credentials, API keys, or other sensitive data; these are already gitignored — keep it that way.
- **Repository scope.** A separate "interview version" of this project exists in another repository. This repository is the development version; only operate within it unless explicitly instructed to work elsewhere.
