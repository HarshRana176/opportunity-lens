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


# Things that should not enter the 3 categories above.


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


# ---------------------------------------------------------------------------
# Task 4 additions below. Everything above this line (SKILL_CATEGORIES,
# EXCLUDED_TECHNOLOGIES) is untouched -- the résumé pipeline's behavior
# and app/skills.py::build_technical_stack must not change.
# ---------------------------------------------------------------------------


# CANONICAL SKILL NAMES
#
# Every key in SKILL_CATEGORIES maps to exactly one canonical name below.
# Alias groups (e.g. "postgres"/"postgresql") converge on one canonical
# spelling; a key with no alias maps to itself. This is a curated map,
# not an algorithm -- naive punctuation stripping would collapse "c",
# "c++", and "c#" into the same key, which is wrong, so each stays its
# own canonical identity. Canonical names are used by app.skills's JD-side
# normalization (Task 4) and are NOT consulted by the résumé pipeline.

SKILL_CANONICAL = {

    # Programming languages -- no aliases among these; each is canonical.

    "python": "python",
    "c": "c",
    "c++": "c++",
    "c#": "c#",
    "java": "java",
    "javascript": "javascript",
    "typescript": "typescript",
    "sql": "sql",
    "r": "r",
    "go": "go",
    "rust": "rust",
    "php": "php",

    # Frameworks / libraries

    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "react": "react",

    "next.js": "next.js",
    "nextjs": "next.js",

    "node.js": "node.js",
    "nodejs": "node.js",

    "express": "express",

    ".net": ".net",
    "entity framework": "entity framework",

    "pytorch": "pytorch",
    "tensorflow": "tensorflow",

    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",

    "transformers": "transformers",
    "huggingface transformers": "transformers",

    "langchain": "langchain",
    "langgraph": "langgraph",

    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "opencv": "opencv",

    # Tools / platforms / databases

    "matlab": "matlab",
    "simulink": "simulink",

    "git": "git",
    "github": "github",
    "github copilot": "github copilot",

    "docker": "docker",
    "kubernetes": "kubernetes",

    "postgresql": "postgresql",
    "postgres": "postgresql",

    "mysql": "mysql",
    "mongodb": "mongodb",
    "sqlite": "sqlite",

    "aws": "aws",
    "aws fundamentals": "aws",

    "azure": "azure",
    "gcp": "gcp",

    "jupyter": "jupyter",
    "jupyter notebook": "jupyter",

    "google colab": "google colab",

    "ollama": "ollama",

    "airflow": "airflow",
    "apache airflow": "airflow",

    "power bi": "power bi",
    "powerbi": "power bi",

    "linux": "linux",
    "windows": "windows",

    "vs code": "vs code",
    "visual studio": "visual studio",

    "postman": "postman",
}


# JOB-DESCRIPTION-ONLY EXCLUSION TERMS
#
# Separate from EXCLUDED_TECHNOLOGIES on purpose: EXCLUDED_TECHNOLOGIES is
# read directly by app.skills.build_technical_stack, which the résumé
# pipeline calls, and is parametrized over by an existing test
# (tests/test_skills.py). Adding JD-only noise terms (soft skills,
# methodologies) to that table would silently change résumé extraction
# output and silently grow that test's case count. JD_EXCLUDED_TERMS is
# consulted only by the JD-side skill normalization path (app.skills's
# normalize_skill, used by app.job_extractor) -- the résumé pipeline
# never reads this set. A JD's skill filtering is the UNION of this set
# and EXCLUDED_TECHNOLOGIES, so JD parsing still screens out the same
# non-technology concepts the résumé side does, plus these JD-specific
# ones. See CLAUDE.md / Task 4 checklist for the plan to reconsider
# unifying the two sets once the résumé side adopts shared normalization.

JD_EXCLUDED_TERMS = {
    "agile",
    "scrum",
    "kanban",
    "waterfall",

    "communication",
    "communication skills",
    "verbal communication",
    "written communication",

    "teamwork",
    "team work",
    "collaboration",
    "cross-functional collaboration",

    "leadership",
    "mentoring",
    "stakeholder management",

    "problem-solving",
    "problem solving",
    "critical thinking",
    "analytical skills",
    "interpersonal skills",
    "attention to detail",
    "adaptability",
    "multitasking",
    "time management",

    "ci/cd",
    "continuous integration",
    "continuous deployment",
    "devops",

    "presentation skills",
    "self-starter",
    "fast-paced environment",
}


# EDUCATION LEVEL TERMS
#
# Maps a lowercase phrase fragment found in a JD's education requirement
# text to the name of an app.schemas.EducationLevel member. Kept as plain
# strings (not the enum itself) so this module has no dependency on
# app.schemas, consistent with the rest of this file being pure data.
# app.requirements does the phrase -> EducationLevel lookup.

EDUCATION_LEVEL_TERMS = {
    "phd": "DOCTORATE",
    "ph.d": "DOCTORATE",
    "doctorate": "DOCTORATE",
    "doctoral": "DOCTORATE",

    "master's": "MASTERS",
    "masters": "MASTERS",
    "master of": "MASTERS",
    "msc": "MASTERS",
    "m.sc": "MASTERS",
    "mba": "MASTERS",
    "graduate degree": "MASTERS",

    "bachelor's": "BACHELORS",
    "bachelors": "BACHELORS",
    "bachelor of": "BACHELORS",
    "bsc": "BACHELORS",
    "b.sc": "BACHELORS",
    "undergraduate degree": "BACHELORS",

    "associate's degree": "ASSOCIATE",
    "associate degree": "ASSOCIATE",

    "high school": "HIGH_SCHOOL",
    "high school diploma": "HIGH_SCHOOL",
    "ged": "HIGH_SCHOOL",
}


# SENIORITY TERMS
#
# Maps a lowercase phrase fragment found in a job title to the name of an
# app.schemas.Seniority member. Same plain-string convention as
# EDUCATION_LEVEL_TERMS, for the same reason. app.requirements does the
# title -> Seniority lookup, matching on word boundaries so a substring
# like "lead" does not spuriously match inside an unrelated word.

SENIORITY_TERMS = {
    "intern": "INTERN",
    "internship": "INTERN",
    "trainee": "INTERN",

    "junior": "JUNIOR",
    "jr": "JUNIOR",
    "entry level": "JUNIOR",
    "entry-level": "JUNIOR",

    "mid level": "MID",
    "mid-level": "MID",
    "intermediate": "MID",

    "senior": "SENIOR",
    "sr": "SENIOR",

    "lead": "LEAD",
    "staff": "LEAD",

    "principal": "PRINCIPAL",
    "head of": "PRINCIPAL",
}


# EDUCATION OPTIONALITY MARKERS
#
# Lowercase phrase fragments that, when present in a JD's education
# requirement text, indicate the requirement is a preference rather than
# a hard requirement (e.g. "Bachelor's preferred", "or equivalent
# experience"). Used by app.requirements.parse_education_requirement to
# set EducationRequirement.is_required. Absence of any of these markers
# in a non-empty education phrase means the requirement is treated as
# required by default.
#
# Deliberately "or equivalent experience", NOT the bare phrase "or
# equivalent": a bare "or equivalent" is ambiguous between "the degree
# is optional if you have equivalent experience" (Bachelor's degree or
# equivalent experience -- should be optional) and "an equivalent
# credential also satisfies this" (High school diploma or equivalent,
# i.e. a GED -- should stay required). The longer phrase disambiguates
# to the first sense, which is the one that should affect is_required.

EDUCATION_OPTIONAL_MARKERS = {
    "preferred",
    "nice to have",
    "plus",
    "bonus",
    "desired",
    "or equivalent experience",
    "a plus",
}
