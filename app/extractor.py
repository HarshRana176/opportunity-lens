from datetime import date, datetime
from typing import Optional, Literal

import pymupdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# 1. LLM


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)



# 2. SCHEMAS


class EmploymentPeriod(BaseModel):
    company: str = Field(
        description="Employer/company name exactly as written."
    )

    role: Optional[str] = Field(
        default=None,
        description="Job title exactly as written."
    )

    start_date: str = Field(
        description="Employment start date exactly as written."
    )

    end_date: str = Field(
        description=(
            "Employment end date exactly as written. "
            "If ongoing, return exactly 'Present'."
        )
    )


class RawResumeExtraction(BaseModel):
    candidate_name: str = Field(
        description="Candidate's full name exactly as written."
    )

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete technical technologies explicitly "
            "mentioned in the resume. Return as a flat list."
        )
    )


class TechnicalStack(BaseModel):
    programming_languages: list[str] = Field(
        default_factory=list
    )

    frameworks: list[str] = Field(
        default_factory=list
    )

    tools: list[str] = Field(
        default_factory=list
    )


class ResumeExtraction(BaseModel):
    candidate_name: str

    technical_stack: TechnicalStack

    employment_history: list[EmploymentPeriod] = Field(
        default_factory=list
    )

    total_experience_months: int

    total_experience_years: float


# 3. PDF TEXT EXTRACTION


def extract_text_from_pdf(pdf_path: str) -> str:

    doc = pymupdf.open(pdf_path)

    try:
        pages_text = []

        for page in doc:
            pages_text.append(
                page.get_text("text")
            )

        return "\n".join(pages_text)

    finally:
        doc.close()



# 4. LLM EXTRACTION


extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a precise resume information extractor.

Your job is ONLY to extract information explicitly present
in the resume.

Rules:

- Never invent information.
- Never calculate experience.
- Never categorize skills.
- Copy the candidate name exactly.
- Extract every employment position.
- Include internships.
- Copy company names exactly.
- Copy roles exactly.
- Copy employment dates exactly.
- If a position says Present, return exactly "Present".
- Extract concrete technologies explicitly mentioned.
- Return technical technologies as one flat skills list.

Do NOT treat these as technical technologies:

DSA
OOP
SDLC
Debugging
NLP
Deep Learning
Computer Vision
Control Systems
PID
System Modeling
Stability Analysis
Semantic Segmentation
Prompt Engineering
Embeddings
Tokenization
Context
RAG
AI Agents
Text Generation
Vector Search

Do not calculate total experience.

Do not categorize the skills.

Only extract information actually present in the resume.
"""
    ),
    (
        "human",
        """
Extract the structured information from this resume:

{resume_text}
"""
    ),
])


structured_extraction_llm = llm.with_structured_output(
    RawResumeExtraction
)

extraction_chain = (
    extraction_prompt
    | structured_extraction_llm
)


# 5. DATE PARSING


_PRESENT_VALUES = {
    "present",
    "current",
    "now",
    "ongoing",
    "till date",
    "to date",
}


def parse_resume_date(date_text: str):

    date_text = date_text.strip()

    if date_text.lower() in _PRESENT_VALUES:

        today = date.today()

        return {
            "date": today,
            "precision": "month"
        }

    formats = [

        ("%d %b %Y", "day"),
        ("%d %B %Y", "day"),

        ("%b %Y", "month"),
        ("%B %Y", "month"),

        ("%Y-%m", "month"),

        ("%Y", "year"),
    ]

    for fmt, precision in formats:

        try:

            parsed = datetime.strptime(
                date_text,
                fmt
            )

            return {
                "date": parsed.date(),
                "precision": precision
            }

        except ValueError:
            continue

    raise ValueError(
        f"Unsupported resume date format: '{date_text}'"
    )


# 6. EXPERIENCE CALCULATION


def date_to_month_index(parsed_date):

    return (
        parsed_date.year * 12
        + (parsed_date.month - 1)
    )


def calculate_duration(
    start_text: str,
    end_text: str
):

    start = parse_resume_date(start_text)
    end = parse_resume_date(end_text)

    start_month = date_to_month_index(
        start["date"]
    )

    end_month = date_to_month_index(
        end["date"]
    )

    if end_month < start_month:

        return {
            "months": 0,
            "years": 0.0
        }

    months = (
        end_month
        - start_month
        + 1
    )

    return {
        "months": months,
        "years": round(
            months / 12,
            2
        )
    }


def calculate_total_experience(
    employment_history: list[EmploymentPeriod]
):

    intervals = []

    for employment in employment_history:

        try:

            start = parse_resume_date(
                employment.start_date
            )

            end = parse_resume_date(
                employment.end_date
            )

            start_month = date_to_month_index(
                start["date"]
            )

            end_month = date_to_month_index(
                end["date"]
            )

            if end_month < start_month:
                continue

            # End month is inclusive.
            intervals.append([
                start_month,
                end_month + 1
            ])

        except ValueError:

            # Ignore an employment entry whose
            # dates cannot be parsed.
            continue

    if not intervals:

        return {
            "months": 0,
            "years": 0.0
        }

    # Sort intervals by start date.
    intervals.sort()

    # Merge overlapping employment periods.
    merged = [intervals[0]]

    for start, end in intervals[1:]:

        previous_end = merged[-1][1]

        if start <= previous_end:

            merged[-1][1] = max(
                previous_end,
                end
            )

        else:

            merged.append([
                start,
                end
            ])

    total_months = sum(
        end - start
        for start, end in merged
    )

    return {
        "months": total_months,
        "years": round(
            total_months / 12,
            2
        )
    }


# 7. TECHNOLOGY CLASSIFICATION


# Known technologies are categorized deterministically.
#
# This is NOT hard-coding a particular resume.
# It is a reusable technology taxonomy.

SKILL_CATEGORIES = {

    # Programming languages
    

    "python": "programming_languages",
    "c": "programming_languages",
    "c++": "programming_languages",
    "c#": "programming_languages",
    "java": "programming_languages",
    "javascript": "programming_languages",
    "typescript": "programming_languages",
    "sql": "programming_languages",
    "r": "programming_languages",
    "go": "programming_languages",
    "rust": "programming_languages",
    "php": "programming_languages",

    # Frameworks / libraries


    "django": "frameworks",
    "flask": "frameworks",
    "fastapi": "frameworks",
    "react": "frameworks",
    "next.js": "frameworks",
    "nextjs": "frameworks",
    "node.js": "frameworks",
    "nodejs": "frameworks",
    "express": "frameworks",

    ".net": "frameworks",
    "entity framework": "frameworks",

    "pytorch": "frameworks",
    "tensorflow": "frameworks",
    "scikit-learn": "frameworks",
    "sklearn": "frameworks",

    "transformers": "frameworks",
    "huggingface transformers": "frameworks",
    "langchain": "frameworks",
    "langgraph": "frameworks",

    "pandas": "frameworks",
    "numpy": "frameworks",
    "matplotlib": "frameworks",
    "opencv": "frameworks",

    # Tools / platforms / databases
   

    "matlab": "tools",
    "simulink": "tools",

    "git": "tools",
    "github": "tools",
    "github copilot": "tools",

    "docker": "tools",
    "kubernetes": "tools",

    "postgresql": "tools",
    "postgres": "tools",
    "mysql": "tools",
    "mongodb": "tools",
    "sqlite": "tools",

    "aws": "tools",
    "aws fundamentals": "tools",
    "azure": "tools",
    "gcp": "tools",

    "jupyter": "tools",
    "jupyter notebook": "tools",
    "google colab": "tools",

    "ollama": "tools",

    "airflow": "tools",
    "apache airflow": "tools",

    "power bi": "tools",
    "powerbi": "tools",

    "linux": "tools",
    "windows": "tools",

    "vs code": "tools",
    "visual studio": "tools",

    "postman": "tools",
}


# 8. THINGS THAT SHOULD NOT ENTER THE 3 CATEGORIES


EXCLUDED_TECHNOLOGIES = {

    # APIs / protocols / concepts
    "rest api",
    "rest apis",
    "graphql",

    # ML architectures / models
    "cnn",
    "convolutional neural network",
    "gan",
    "generative adversarial network",
    "u-net",
    "unet",
    "deeplabv3",
    "deeplabv3-resnet50",
    "segformer",
    "segformer-b2",

    # Concepts
    "nlp",
    "deep learning",
    "computer vision",
    "semantic segmentation",
    "control systems",
    "pid",
    "pid control",
    "system modeling",
    "stability analysis",

    # General engineering skills
    "dsa",
    "oop",
    "sdlc",
    "debugging",

    # GenAI concepts
    "prompt engineering",
    "embeddings",
    "tokenization",
    "context",
    "rag",
    "retrieval-augmented generation",
    "ai agents",
    "text generation",
    "vector search",

    # General techniques
    "preprocessing",
    "data augmentation",
    "model evaluation",
    "mlo ps",
}


# 9. KNOWN TECHNOLOGY CATEGORIZATION


def categorize_known_skills(
    skills: list[str]
):

    result = {
        "programming_languages": [],
        "frameworks": [],
        "tools": [],
    }

    seen = set()

    for skill in skills:

        if not skill:
            continue

        clean_skill = skill.strip()

        if not clean_skill:
            continue

        key = clean_skill.lower()

        if key in EXCLUDED_TECHNOLOGIES:
            continue

        if key in seen:
            continue

        seen.add(key)

        category = SKILL_CATEGORIES.get(key)

        if category:

            result[category].append(
                clean_skill
            )

    return result


# 10. UNKNOWN TECHNOLOGY CLASSIFIER


class SkillCategory(BaseModel):

    category: Literal[
        "programming_language",
        "framework",
        "tool",
        "exclude"
    ]


skill_classifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Classify one item from a resume.

Return exactly one of:

programming_language
framework
tool
exclude

Programming language:
Python, C++, Java, SQL, JavaScript, etc.

Framework/library:
FastAPI, Django, PyTorch, TensorFlow,
Scikit-learn, Transformers, LangChain, etc.

Tool/platform/database:
Git, Docker, PostgreSQL, MySQL,
AWS, Jupyter, MATLAB, etc.

Exclude:
- APIs
- protocols
- algorithms
- ML architectures
- ML models
- concepts
- methodologies
- domains
- general skills
- techniques

Examples to exclude:

REST APIs
GraphQL
CNN
GAN
U-Net
DeepLabv3
SegFormer
NLP
Deep Learning
Computer Vision
RAG
Embeddings
Prompt Engineering
DSA
OOP
SDLC
Debugging

Return exactly one category.
"""
    ),
    (
        "human",
        "Technology: {skill}"
    )
])


structured_skill_classifier = llm.with_structured_output(
    SkillCategory
)

skill_classifier_chain = (
    skill_classifier_prompt
    | structured_skill_classifier
)


def classify_unknown_skill(skill: str):

    try:

        result = skill_classifier_chain.invoke({
            "skill": skill
        })

        return result.category

    except Exception:

        # Safer to exclude an unknown item
        # than incorrectly categorize it.
        return "exclude"



# 11. BUILD FINAL TECHNICAL STACK


def build_technical_stack(
    skills: list[str]
):

    categorized = {
        "programming_languages": [],
        "frameworks": [],
        "tools": [],
    }

    seen = set()

    for skill in skills:

        if not skill:
            continue

        clean_skill = skill.strip()

        if not clean_skill:
            continue

        key = clean_skill.lower()

        # Explicit exclusion
        if key in EXCLUDED_TECHNOLOGIES:
            continue

        # Avoid duplicates
        if key in seen:
            continue

        seen.add(key)

  
        # Known technology
     

        category = SKILL_CATEGORIES.get(key)

        if category:

            categorized[category].append(
                clean_skill
            )

            continue

        # Unknown technology
    

        category = classify_unknown_skill(
            clean_skill
        )

        if category == "programming_language":

            categorized[
                "programming_languages"
            ].append(clean_skill)

        elif category == "framework":

            categorized[
                "frameworks"
            ].append(clean_skill)

        elif category == "tool":

            categorized[
                "tools"
            ].append(clean_skill)

        # exclude -> ignore

    return TechnicalStack(
        programming_languages=categorized[
            "programming_languages"
        ],
        frameworks=categorized[
            "frameworks"
        ],
        tools=categorized[
            "tools"
        ],
    )


# 12. MAIN EXTRACTION FUNCTION


def extract_resume(pdf_path: str):

    # PDF -> text


    full_text = extract_text_from_pdf(
        pdf_path
    )

    if not full_text.strip():

        raise ValueError(
            "Could not extract text from the PDF."
        )

    # LLM extracts facts


    raw_result = extraction_chain.invoke({
        "resume_text": full_text
    })

    # Python calculates experience


    experience = calculate_total_experience(
        raw_result.employment_history
    )

    # Python categorizes technical stack


    technical_stack = build_technical_stack(
        raw_result.skills
    )

    # Final structured result

    result = ResumeExtraction(

        candidate_name=raw_result.candidate_name,

        technical_stack=technical_stack,

        employment_history=(
            raw_result.employment_history
        ),

        total_experience_months=(
            experience["months"]
        ),

        total_experience_years=(
            experience["years"]
        ),
    )

    return result