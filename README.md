# AI-Powered Resume Parser API

An AI-powered Resume Parser API that extracts structured information from PDF resumes using a locally hosted LLM and stores the results in PostgreSQL.

## Overview

The system accepts a resume in PDF format, extracts its text, uses Qwen through Ollama and LangChain for structured information extraction, calculates total professional experience programmatically, and stores the final result in PostgreSQL.

The parser is designed to work with different resumes rather than relying on hard-coded resume-specific information.

## Architecture

```text
PDF Resume
    │
    ▼
FastAPI Upload Endpoint
    │
    ▼
PDF Text Extraction
(PyMuPDF)
    │
    ▼
LangChain Processing
    │
    ├── Candidate Name
    ├── Technical Stack
    └── Employment History
            │
            ▼
    Pydantic Structured Output
            │
            ▼
    Python Experience Calculation
            │
            ▼
       PostgreSQL



Tech Stack
Python — Core implementation
FastAPI — REST API
LangChain — LLM orchestration and structured extraction
Qwen 2.5 3B — Local LLM
Ollama — Local LLM runtime
Pydantic — Structured data validation
PyMuPDF — PDF text extraction
SQLAlchemy — Database ORM
PostgreSQL — Persistent storage
Swagger/OpenAPI — API documentation and testing

Key Features
1. PDF Resume Upload

The API accepts PDF resumes through a multipart file upload endpoint.

POST /resumes

Only PDF files are accepted.

2. Resume Text Extraction

PyMuPDF extracts the raw text from the uploaded PDF before sending it to the LLM.

3. Structured Information Extraction

The LLM extracts:

Candidate name
Programming languages
Frameworks/libraries
Tools
Employment history
Company names
Job roles
Employment start dates
Employment end dates

The extracted information is validated using Pydantic models.

4. Technical Stack Categorization

Technologies are categorized into:

Programming Languages
Frameworks/Libraries
Tools

The extraction prompt instructs the model to use only technologies explicitly present in the resume and avoid inventing technologies.

5. Employment Experience Calculation

Employment dates are extracted by the LLM, but duration calculation is performed programmatically in Python.

For example:

January 2026 → Present
May 2025 → June 2025

The system converts these periods into months and calculates total professional experience.

6. PostgreSQL Storage

The structured extraction result is persisted in PostgreSQL using SQLAlchemy.

The database stores:

Candidate name
Technical stack
Employment history
Total experience in months
Total experience in years


API Documentation

After starting the application, interactive API documentation is available at:

http://127.0.0.1:8000/docs

The Swagger interface can be used to upload a PDF and inspect the extracted JSON response.

Local Setup
1. Clone the repository
git clone https://github.com/HarshRana176/resume-parser.git
cd resume-parser
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Configure PostgreSQL

Create a PostgreSQL database and configure the connection string in .env:

DATABASE_URL=postgresql://postgres:password@localhost:5432/resume_parser
5. Install and run Ollama

Make sure Ollama is installed and the Qwen model is available:

ollama pull qwen2.5:3b
6. Start the API
uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs
Project Structure
resume-parser/
│
├── app/
│   ├── database.py
│   ├── extractor.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── services.py
│
├── uploads/
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
Challenges Faced
1. Local API Requests Hanging

During development, requests to the FastAPI server initially appeared to hang.

The issue was investigated using direct HTTP requests with curl and by checking the Uvicorn server logs.

The application was eventually verified successfully through the Swagger interface.

2. Reliable Structured Extraction

The initial extraction could incorrectly categorize concepts as technologies.

This was addressed by refining the extraction prompts and explicitly defining what belongs in each category.

3. Employment Duration Calculation

Rather than asking the LLM to calculate experience, employment dates are extracted first and duration is calculated deterministically using Python.

This makes the calculation more predictable.

4. Sensitive Local Files

Resume PDFs and environment variables should not be committed to the repository.

A .gitignore file was therefore added to exclude:

.venv/
__pycache__/
*.pyc
.env
uploads/
Validation

The application was tested using the FastAPI Swagger interface by uploading a PDF resume.

The resulting structured JSON was successfully returned by the API and the extracted data was also persisted in PostgreSQL.

The database records were verified using pgAdmin.







The FastAPI application provides endpoints for:

GET  /
GET  /resumes
POST /resumes
GET  /resumes/{resume_id}

FastAPI automatically generates interactive Swagger/OpenAPI documentation.

Why Qwen?

Qwen 2.5 3B was selected because it is:

Small enough to run locally
Suitable for structured information extraction
Compatible with Ollama
Does not require an external API key
Has no external inference cost
Keeps resume data local during development
Fast enough for a prototype/API workflow

The model is run locally through Ollama:

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

Using temperature=0 helps make the extraction more deterministic.

Why LangChain?

LangChain is used to structure the interaction between the application and the LLM.

It provides:

Prompt templates
LLM chains
Structured output handling
Separation between extraction tasks

Different extraction tasks are handled independently, such as employment history and technical stack extraction.

Avoiding Hard-Coded Resume Data

A key requirement was that the parser should not be designed specifically for one resume.

The system therefore:

Accepts an arbitrary PDF through the API.
Extracts its text dynamically.
Sends the extracted text to the LLM.
Uses Pydantic schemas to structure the response.
Calculates experience from the extracted dates.

The prompts explicitly instruct the model to:

Extract only information present in the resume.
Never invent information.
Preserve company names and job titles.
Preserve employment dates as written.
Extract internships as employment positions.
Preserve "Present" for ongoing employment.
Avoid inferring technologies that are not explicitly listed.

No candidate-specific values such as names, companies, or technologies are hard-coded into the extraction logic.

Handling Employment Dates

The LLM extracts employment dates as text rather than calculating durations.

Python then parses supported date formats and calculates the duration.

This separation was intentional:

LLM
 ↓
Extract factual information


Python
 ↓
Perform deterministic calculations

This avoids relying on the LLM for arithmetic.

Project Status

Completed prototype

The current implementation demonstrates an end-to-end AI resume processing pipeline:

Resume PDF
   ↓
FastAPI
   ↓
PyMuPDF
   ↓
LangChain + Qwen
   ↓
Pydantic
   ↓
Python calculations
   ↓
PostgreSQL
   ↓
Structured API response