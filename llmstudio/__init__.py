name = "version"
__version__ = "0.2.0"

__requirements__ = [
    # core
    "pydantic",
    "requests",
    "pydantic",
    "numpy",
    "torch",
    "sentence-transformers",
    "fastapi",
    "uvicorn",
    "PyYaml",
    # llm_engine
    "openai",
    "tiktoken",
    "google-auth",
    "google-cloud-aiplatform",
    "fastapi",
    "boto3"
]

__requirements__ = [
    # core
    "requests<3",
    "pydantic==1.10.9",
    "click",
    "numpy",
    "requests",
    "torch",
    "sentence-transformers",
    # llm_engine
    "gunicorn==19.9.0",
    "openai",
    "google-auth",
    "google-cloud-aiplatform",
    "tiktoken",
    "fastapi",
    "uvicorn",
    "uuid",
    "boto3",
    "PyYaml"
]

from .client import LLMStudio
