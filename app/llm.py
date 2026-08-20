from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas import RawResumeExtraction, SkillCategory


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
