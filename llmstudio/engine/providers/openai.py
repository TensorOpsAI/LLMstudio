import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class OpenAIParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(ChatRequest):
    parameters: Optional[OpenAIParameters] = OpenAIParameters()


@provider
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
                **request.parameters.dict(),
            )
        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            if chunk.choices[0].finish_reason not in ["stop", "length"]:
                yield chunk.choices[0].delta.content
