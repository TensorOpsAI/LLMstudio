import asyncio
import os
import time
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import cohere
from fastapi import HTTPException
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class CommandParameters(BaseModel):
    temperature: Optional[float] = Field(0.75, ge=0, le=5)
    max_tokens: Optional[int] = Field(256, ge=1)
    p: Optional[float] = Field(0, ge=0, le=0.99)
    k: Optional[int] = Field(0, ge=0, le=500)
    frequency_penalty: Optional[float] = Field(0, ge=0)
    presence_penalty: Optional[float] = Field(0, ge=0, le=1)


class CohereRequest(ChatRequest):
    parameters: Optional[CommandParameters] = CommandParameters()


@provider
class CohereProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("COHERE_API_KEY")

    def validate_request(self, request: CohereRequest):
        return CohereRequest(**request)

    async def generate_client(
        self, request: CohereRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate a Cohere client"""
        try:
            co = cohere.Client(api_key=request.api_key or self.API_KEY)
            return await asyncio.to_thread(
                co.generate,
                model=request.model,
                prompt=request.chat_input,
                stream=True,
                **request.parameters.model_dump(),
            )
        except cohere.CohereAPIError or cohere.CohereConnectionError as e:
            raise HTTPException(status_code=e.http_status, detail=str(e))

    async def handle_response(
        self, request: CohereRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from the Cohere API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        for chunk in response:
            if not chunk.is_finished:
                token_count += 1
                current_time = time.time()
                first_token_time = first_token_time or current_time
                if previous_token_time is not None:
                    token_times.append(current_time - previous_token_time)
                previous_token_time = current_time

                chat_output += chunk.text
                if request.is_stream:
                    yield chunk.text

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
