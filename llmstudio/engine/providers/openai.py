import asyncio
import os
import time
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider


class OpenAIParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(ChatRequest):
    parameters: Optional[OpenAIParameters] = OpenAIParameters()


class OpenAIProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("OPENAI_API_KEY")

    def validate_request(self, request: OpenAIRequest):
        return OpenAIRequest(**request)

    async def generate_client(
        self, request: OpenAIRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""
        try:
            client = OpenAI(api_key=request.api_key or self.API_KEY)
            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=[{"role": "user", "content": request.chat_input}],
                stream=True,
                **request.parameters.model_dump(),
            )
        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def handle_response(
        self, request: OpenAIRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from the OpenAI API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        for chunk in response:
            if chunk.choices[0].finish_reason not in ["stop", "length"]:
                token_count += 1
                current_time = time.time()
                first_token_time = first_token_time or current_time
                if previous_token_time is not None:
                    token_times.append(current_time - previous_token_time)
                previous_token_time = current_time

                chat_output += chunk.choices[0].delta.content
                if request.is_stream:
                    yield chunk.choices[0].delta.content

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
