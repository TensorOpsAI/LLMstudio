import asyncio
import os
import time
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import anthropic
from anthropic import Anthropic
from fastapi import HTTPException
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class ClaudeParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens_to_sample: Optional[int] = Field(256, ge=1)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[int] = Field(5, ge=0, le=500)


class AnthropicRequest(ChatRequest):
    parameters: Optional[ClaudeParameters] = ClaudeParameters()


@provider
class AnthropicProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("ANTHROPIC_API_KEY")

    def validate_request(self, request: AnthropicRequest):
        return AnthropicRequest(**request)

    async def generate_client(
        self, request: AnthropicRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an Anthropic client"""
        try:
            client = Anthropic(api_key=request.api_key or self.API_KEY)
            return await asyncio.to_thread(
                client.completions.create,
                model=request.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {request.chat_input} {anthropic.AI_PROMPT}",
                stream=True,
                **request.parameters.model_dump(),
            )
        except anthropic._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def handle_response(
        self, request: AnthropicRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from the Anthropic API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        for chunk in response:
            if chunk.stop_reason != "stop_sequence":
                token_count += 1
                current_time = time.time()
                first_token_time = first_token_time or current_time
                if previous_token_time is not None:
                    token_times.append(current_time - previous_token_time)
                previous_token_time = current_time

                chat_output += chunk.completion
                if request.is_stream:
                    yield chunk.completion

        metrics = self.calculate_metrics(
            request.chat_input,
            chat_output,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
        )

        if request.is_stream and request.has_end_token:
            yield self.get_end_token_string(metrics)

        response = self.generate_response(request, chat_output, metrics)

        self.save_log(response)

        if not request.is_stream:
            yield response
