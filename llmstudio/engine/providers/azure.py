import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import openai
from fastapi import HTTPException
from openai import AzureOpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class AzureParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class AzureRequest(ChatRequest):
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = "2023-05-15"
    parameters: Optional[AzureParameters] = AzureParameters()


@provider
class AzureProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("AZURE_API_KEY")
        self.API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
        self.API_VERSION = os.getenv("AZURE_API_VERSION")

    def validate_request(self, request: AzureRequest):
        return AzureRequest(**request)

    async def generate_client(
        self, request: AzureRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""
        try:
            client = AzureOpenAI(
                api_key=request.api_key or self.API_KEY,
                azure_endpoint=request.api_endpoint or self.API_ENDPOINT,
                api_version=request.api_version or self.API_VERSION,
            )
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
            if (
                chunk.choices[0].finish_reason not in ["stop", "length"]
                and chunk.choices[0].delta.content is not None
            ):
                yield chunk.choices[0].delta.content
