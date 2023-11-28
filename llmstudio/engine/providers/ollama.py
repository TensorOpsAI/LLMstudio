import asyncio
import requests
import json
import time
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider


class OllamaParameters(BaseModel):
    temperature: Optional[float] = Field(0.8, ge=0, le=1)
    num_predict: Optional[int] = Field(128, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    top_k: Optional[int] = Field(40, ge=0, le=500)


class OllamaRequest(ChatRequest):
    parameters: Optional[OllamaParameters] = OllamaParameters()


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
                    **request.parameters.model_dump(),
                },
                stream=True,
            )
        except requests.RequestException:
            raise HTTPException(
                status_code=500,
                detail="Ollama is not running",
            )

    async def handle_response(
        self, request: OllamaRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from the Ollama API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "error" in chunk:
                raise HTTPException(status_code=500, detail=chunk["error"])
            if chunk.get("done"):
                break

            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chat_output += chunk["response"]
            if request.is_stream:
                yield chunk["response"]

        usage = self.calculate_usage(request.chat_input, chat_output, request.model)

        metrics = self.calculate_metrics(
            start_time, time.time(), first_token_time, token_times, token_count
        )

        if request.is_stream and request.has_end_token:
            yield self.get_end_token_string(usage, metrics)

        response = self.generate_response(request, chat_output, usage, metrics)

        self.save_log(response)

        if not request.is_stream:
            yield response
