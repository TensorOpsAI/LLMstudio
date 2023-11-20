import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from llmstudio.engine.providers import *

ENGINE_BASE_ENDPOINT = "/api/engine"
ENGINE_HEALTH_ENDPOINT = "/health"
ENGINE_TITLE = "LLMstudio Engine API"
ENGINE_DESCRIPTION = "The core API for LLM interactions"
ENGINE_VERSION = "0.1.0"
ENGINE_HOST = os.getenv("ENGINE_HOST", "localhost")
ENGINE_PORT = int(os.getenv("ENGINE_PORT", 8000))
ENGINE_URL = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
UI_HOST = os.getenv("ENGINE_HOST", "localhost")
UI_PORT = int(os.getenv("UI_PORT", 8000))
UI_URL = f"http://{UI_HOST}:{UI_PORT}"
LOG_LEVEL = os.getenv("LOG_LEVEL", "critical")


# Models for Configuration
class ModelConfig(BaseModel):
    mode: str
    max_tokens: int
    input_token_cost: float
    output_token_cost: float


class ProviderConfig(BaseModel):
    provider: str
    name: str
    chat: bool
    embed: bool
    keys: Optional[List[str]] = None
    models: Optional[Dict[str, ModelConfig]] = None
    parameters: Optional[Dict[str, Any]] = None


class EngineConfig(BaseModel):
    providers: Dict[str, ProviderConfig]


# Configuration Loading
def _load_engine_config() -> EngineConfig:
    config_path = Path(os.path.join(os.path.dirname(__file__), "config.yaml"))
    try:
        config_data = yaml.safe_load(config_path.read_text())
        return EngineConfig(**config_data)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML configuration: {e}")
    except ValidationError as e:
        raise RuntimeError(f"Error in configuration data: {e}")


# Functions for API Operations
def create_engine_app(config: EngineConfig = _load_engine_config()) -> FastAPI:
    app = FastAPI(
        title=ENGINE_TITLE,
        description=ENGINE_DESCRIPTION,
        version=ENGINE_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(ENGINE_HEALTH_ENDPOINT)
    def health_check():
        """Health check endpoint to ensure the API is running."""
        return {"status": "healthy", "message": "Engine is up and running"}

    @app.get(f"{ENGINE_BASE_ENDPOINT}/providers")
    def get_providers():
        """Return all providers supported."""
        return list(config.providers.keys())

    @app.get(f"{ENGINE_BASE_ENDPOINT}/models")
    def get_models(provider: Optional[str] = None):
        """Return all models supported, filter by provider if given."""
        if provider:
            provider_config = config.providers.get(provider)
            if provider_config and provider_config.models:
                return list(provider_config.models.keys())
            else:
                return (
                    []
                )  # or raise an HTTPException if the provider does not exist or has no models
        else:
            all_models = []
            for provider_config in config.providers.values():
                if provider_config.models:
                    all_models.extend(provider_config.models.keys())
            return list(set(all_models))  # Use set to avoid duplicates if any

    # Function to create a chat handler for a provider
    def create_chat_handler(provider_config):
        async def chat_handler(request: Request):
            """Endpoint for chat functionality."""
            provider_class = globals()[f"{provider_config.name}Provider"]
            provider_instance = provider_class(provider_config)
            return await provider_instance.chat(await request.json())

        return chat_handler

    # Dynamic route creation based on the 'chat' boolean
    for provider_name, provider_config in config.providers.items():
        if provider_config.chat:
            app.post(f"{ENGINE_BASE_ENDPOINT}/chat/{provider_name}")(
                create_chat_handler(provider_config)
            )

    return app


def is_api_running(url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def run_engine_app():
    print(f"Running Engine on {ENGINE_HOST}:{ENGINE_PORT}")
    try:
        engine = create_engine_app()
        uvicorn.run(
            engine,
            host=ENGINE_HOST,
            port=ENGINE_PORT,
        )
    except Exception as e:
        print(f"Error running the Engine app: {e}")
