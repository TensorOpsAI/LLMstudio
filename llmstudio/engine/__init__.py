import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from llmstudio.engine.providers import *

ENGINE_BASE_ENDPOINT = "/api/engine"
ENGINE_HEALTH_ENDPOINT = "/health"
ENGINE_TITLE = "LLMstudio Engine API"
ENGINE_DESCRIPTION = "The core API for LLM interactions"
ENGINE_VERSION = "0.1.0"


# Models for Configuration
class ModelConfig(BaseModel):
    mode: str
    max_tokens: int
    input_token_cost: float
    output_token_cost: float


class ProviderConfig(BaseModel):
    id: str
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
    default_config_path = Path(os.path.join(os.path.dirname(__file__), "config.yaml"))
    local_config_path = Path(os.getcwd(), "config.yaml")

    def _merge_configs(config1, config2):
        for key in config2:
            if key in config1:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    _merge_configs(config1[key], config2[key])
                elif isinstance(config1[key], list) and isinstance(config2[key], list):
                    config1[key].extend(config2[key])
                else:
                    config1[key] = config2[key]
            else:
                config1[key] = config2[key]
        return config1

    try:
        default_config_data = yaml.safe_load(default_config_path.read_text())
        local_config_data = (
            yaml.safe_load(local_config_path.read_text())
            if local_config_path.exists()
            else {}
        )
        config_data = _merge_configs(default_config_data, local_config_data)
        return EngineConfig(**config_data)
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
        allow_origins=["*"],
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
        """Return all models supported with the provider as a key."""
        all_models = {}
        for provider_name, provider_config in config.providers.items():
            if provider and provider_name != provider:
                continue
            if provider_config.models:
                all_models[provider_name] = {}
                all_models[provider_name]["name"] = provider_config.name
                all_models[provider_name]["models"] = list(
                    provider_config.models.keys()
                )
        return all_models[provider] if provider else all_models

    # Function to create a chat handler for a provider
    def create_chat_handler(provider_config):
        async def chat_handler(request: Request):
            """Endpoint for chat functionality."""
            provider_class = provider_registry.get(f"{provider_config.name}Provider")
            provider_instance = provider_class(provider_config)
            return await provider_instance.chat(await request.json())

        return chat_handler

    # Dynamic route creation based on the 'chat' boolean
    for provider_name, provider_config in config.providers.items():
        if provider_config.chat:
            app.post(f"{ENGINE_BASE_ENDPOINT}/chat/{provider_name}")(
                create_chat_handler(provider_config)
            )

    @app.get(f"{ENGINE_BASE_ENDPOINT}/parameters")
    def get_parameters(provider: str, model: Optional[str] = None):
        """Return parameters for a given provider and model in an array JSON format."""
        provider_config = config.providers.get(provider)
        if not provider_config:
            return {"error": f"Provider {provider} not found"}, 404
        parameters = provider_config.parameters
        parameters_array = [{"id": key, **value} for key, value in parameters.items()]
        return parameters_array

    @app.post("/api/export")
    async def export(request: Request):
        data = await request.json()
        csv_content = ""

        if len(data) > 0:
            csv_content += ";".join(data[0].keys()) + "\n"
            for execution in data:
                csv_content += (
                    ";".join([json.dumps(value) for value in execution.values()]) + "\n"
                )

        headers = {"Content-Disposition": "attachment; filename=parameters.csv"}
        return StreamingResponse(
            iter([csv_content]), media_type="text/csv", headers=headers
        )

    return app


def run_engine_app():
    print(
        f"Running Engine on http://{os.getenv('LLMSTUDIO_ENGINE_HOST')}:{os.getenv('LLMSTUDIO_ENGINE_PORT')}"
    )
    try:
        engine = create_engine_app()
        uvicorn.run(
            engine,
            host=os.getenv("LLMSTUDIO_ENGINE_HOST"),
            port=int(os.getenv("LLMSTUDIO_ENGINE_PORT")),
        )
    except Exception as e:
        print(f"Error running the Engine app: {e}")
