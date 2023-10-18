name = "version"
__version__ = "0.1.6"

__requirements__ = [
    # core
    "requests<3",
    "pydantic==1.10.9",
    "click",
    # llm_engine
    "gunicorn==19.9.0",
    "openai",
    "google.auth",
    "google-cloud-aiplatform",
    "tiktoken",
    "fastapi",
    "uvicorn",
    "uuid",
    "boto3",
]

from .client import LLMStudio
