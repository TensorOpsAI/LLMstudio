import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from llmstudio_core.exceptions import ProviderError
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio_core.providers.provider import ChatRequest, BaseProvider, provider



@provider
class OpenAIProvider(BaseProvider):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.API_KEY = self.API_KEY if self.API_KEY else os.getenv("OPENAI_API_KEY")

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""

        try:
            client = OpenAI(api_key=self.API_KEY)
            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=request.is_stream,
                **request.parameters,
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))
        
    def generate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""

        try:
            client = OpenAI(api_key=self.API_KEY)
            return client.chat.completions.create(
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=request.is_stream,
                **request.parameters,
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            yield chunk.model_dump()

    def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> Generator:
        for chunk in response:
            yield chunk.model_dump()