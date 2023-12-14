import asyncio
import json
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import requests
from fastapi import HTTPException
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class OllamaParameters(BaseModel):
    temperature: Optional[float] = Field(0.8, ge=0, le=1)
    num_predict: Optional[int] = Field(128, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    top_k: Optional[int] = Field(40, ge=0, le=500)


class OllamaRequest(ChatRequest):
    parameters: Optional[OllamaParameters] = OllamaParameters()


@provider
class OllamaProvider(Provider):
    def __init__(self, config):
        super().__init__(config)

    def validate_request(self, request: OllamaRequest):
        return OllamaRequest(**request)

    async def generate_client(
        self, request: OllamaRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an Ollama client"""
        try:
            return await asyncio.to_thread(
                requests.post,
                url="http://localhost:11434/api/generate",
                json={
                    "model": request.model,
                    "prompt": request.chat_input,
                    **request.parameters.dict(),
                },
                stream=True,
            )
        except requests.RequestException:
            raise HTTPException(
                status_code=500,
                detail="Ollama is not running",
            )

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "error" in chunk:
                raise HTTPException(status_code=500, detail=chunk["error"])
            if chunk.get("done"):
                break
            yield chunk["response"]
