from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas import (
    BatchSkillClassification,
    RawEducationExtraction,
    RawJobCoreExtraction,
    RawJobRequirementsExtraction,
    RawResumeExtraction,
    SkillCategory,
)


# LLM


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


# RESUME EXTRACTION CHAIN


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


# UNKNOWN-SKILL CLASSIFICATION CHAIN


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


# BATCH UNKNOWN-SKILL CLASSIFICATION CHAIN
#
# Used by app.skills.enrich_unresolved_skills (Job Description parsing,
# Task 4) to classify all taxonomy-unresolved skills from one JD in a
# single call, instead of one call per skill. Empirically measured on
# this repo's qwen2.5:3b: a single batched call for 20 unknown skills
# completed in ~4.5s versus ~6-22s sequentially (and was MORE accurate
# -- "Kafka" alone misclassified as "exclude" per-skill but correctly
# as "tool" when classified alongside other infrastructure terms). This
# chain is advisory only: app.skills never lets its output delete a
# skill, and match-back to the input is by exact string equality.
# Résumé-side classification (skill_classifier_chain above) is
# unchanged and intentionally still per-skill.


batch_skill_classifier_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Classify EACH technology in the list below.

Return exactly one entry per input item. Copy each name EXACTLY as
given in the "name" field -- do not rename, reformat, correct
spelling, or change casing.

Categories:

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
AWS, Jupyter, MATLAB, Kubernetes, Kafka, Redis, etc.

Exclude:
- APIs
- protocols
- algorithms
- ML architectures
- ML models
- concepts
- methodologies
- domains
- general/soft skills
- techniques

Examples to exclude:

REST APIs
GraphQL
CNN
GAN
NLP
Deep Learning
Computer Vision
RAG
Embeddings
Prompt Engineering
Agile
Scrum
Communication
Leadership

Return exactly one entry for every item in the list, in any order.
"""
    ),
    (
        "human",
        "Classify each of these:\n{skills}"
    ),
])


structured_batch_skill_classifier = llm.with_structured_output(
    BatchSkillClassification
)

batch_skill_classifier_chain = (
    batch_skill_classifier_prompt
    | structured_batch_skill_classifier
)


# JOB DESCRIPTION EXTRACTION CHAINS (Task 4)
#
# Two chains, not one -- see RawJobCoreExtraction/RawJobRequirementsExtraction
# in app.schemas for why. Both extract facts verbatim, never compute,
# never normalize/categorize. app.job_extractor calls both and merges
# their results into a RawJobExtraction; everything else (skill
# normalization via app.skills, experience/education/seniority
# interpretation via app.requirements) happens after, deterministically.


job_core_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a precise job description information extractor.

Extract ONLY the job title, responsibilities, and every mentioned
technology/skill from this job description. Do not invent information.

Rules:

- Copy the job title exactly.
- Extract every responsibility/duty as a separate list item, copied
  verbatim (do not merge or summarize bullets).
- Extract every concrete technology/skill explicitly mentioned.
- For each skill's "name", extract ONLY the technology name itself.
  Strip surrounding phrasing such as "experience with", "familiarity
  with", "knowledge of", "strong", "solid understanding of", etc. --
  the name field must contain just the technology.
  Example: from "Familiarity with Kafka is a plus", extract name="Kafka".
  Example: from "Strong experience with Python and FastAPI", extract
  two separate entries: name="Python" and name="FastAPI".
- Label each skill "required" if the JD presents it as a must-have
  (e.g. under a "Requirements" / "Must Have" heading, or phrased as
  required/mandatory), or "preferred" if presented as nice-to-have /
  bonus / a plus. If genuinely ambiguous, use "required".
- Never categorize or normalize skill names -- copy them exactly.

Do NOT treat these as technical skills:

Agile
Scrum
Communication
Leadership
Teamwork
Problem-solving
Time management
CI/CD (as a general practice, not a specific tool)

Only extract information actually present in the job description.
"""
    ),
    (
        "human",
        """
Extract the title, responsibilities, and skills from this job description:

{job_text}
"""
    ),
])


structured_job_core_extraction_llm = llm.with_structured_output(
    RawJobCoreExtraction
)

job_core_extraction_chain = (
    job_core_extraction_prompt
    | structured_job_core_extraction_llm
)


job_requirements_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract ONLY the experience and education requirements from this job
description, exactly as written. Do not calculate or interpret
anything -- copy phrases verbatim.

Return null for anything not mentioned. Never invent a requirement
that is not present in the text.
"""
    ),
    (
        "human",
        """
Extract the experience and education requirements from this job description:

{job_text}
"""
    ),
])


structured_job_requirements_extraction_llm = llm.with_structured_output(
    RawJobRequirementsExtraction
)

job_requirements_extraction_chain = (
    job_requirements_extraction_prompt
    | structured_job_requirements_extraction_llm
)


# CANDIDATE EDUCATION EXTRACTION CHAIN (Task 6)
#
# A separate, focused chain -- consumed ONLY by app.candidate_extractor,
# never by app.extractor.extract_resume(). Adding an education field
# directly to RawResumeExtraction was tried during planning and
# empirically collapsed skill extraction (31 skills -> 0, reproducibly)
# on this repo's model, the same destabilization Task 4 documented on
# the JD side (see RawJobRequirementsExtraction's docstring). A
# separate single-purpose call avoids that entirely and was verified
# deterministic (3/3 identical runs) on a real résumé during planning.


education_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract ONLY the education entries from this résumé, exactly as
written.

Copy each degree, field of study, institution, and completion
year/date/status verbatim. Do not interpret, normalize, categorize, or
infer anything -- do not calculate a person's education level, and do
not invent an entry that is not present in the text.

Return an empty list if the résumé has no education section.
"""
    ),
    (
        "human",
        """
Extract the education entries from this résumé:

{resume_text}
"""
    ),
])


structured_education_extraction_llm = llm.with_structured_output(
    RawEducationExtraction
)

education_extraction_chain = (
    education_extraction_prompt
    | structured_education_extraction_llm
)
