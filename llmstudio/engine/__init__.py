import json
import os
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llmstudio.config import ENGINE_HOST, ENGINE_PORT
from llmstudio_core.providers import _load_engine_config
from llmstudio_core.providers.provider import provider_registry

ENGINE_BASE_ENDPOINT = "/api/engine"
ENGINE_HEALTH_ENDPOINT = "/health"
ENGINE_TITLE = "LLMstudio Engine API"
ENGINE_DESCRIPTION = "The core API for LLM interactions"
ENGINE_VERSION = "0.1.0"


class CostRange(BaseModel):
    range: List[Optional[int]]
    cost: float


class ModelConfig(BaseModel):
    mode: str
    max_tokens: int
    input_token_cost: Union[float, List[CostRange]]
    output_token_cost: Union[float, List[CostRange]]


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


def create_engine_app(
    started_event: Event, config: EngineConfig = _load_engine_config()
) -> FastAPI:
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

    def create_chat_handler(provider_config):
        async def chat_handler(request: Request):
            """Endpoint for chat functionality."""
            provider_class = provider_registry.get(f"{provider_config.name}".lower())
            provider_instance = provider_class(provider_config)
            request_dict = await request.json()
            result = await provider_instance.achat(**request_dict)
            return result

        return chat_handler

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

    @app.on_event("startup")
    async def startup_event():
        started_event.set()
        print(f"Running LLMstudio Engine on http://{ENGINE_HOST}:{ENGINE_PORT} ")

    return app


def run_engine_app(started_event: Event):
    try:
        engine = create_engine_app(started_event)
        uvicorn.run(
            engine,
            host=ENGINE_HOST,
            port=ENGINE_PORT,
            log_level="warning",
        )
    except Exception as e:
        print(f"Error running LLMstudio Engine: {e}")
