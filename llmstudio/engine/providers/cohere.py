import asyncio
import os
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
                **request.parameters.dict(),
            )
        except cohere.CohereAPIError or cohere.CohereConnectionError as e:
            raise HTTPException(status_code=e.http_status, detail=str(e))

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            if not chunk.is_finished:
                yield chunk.text
