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
