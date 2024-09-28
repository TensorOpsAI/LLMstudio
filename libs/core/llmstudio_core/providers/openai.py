import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from llmstudio_core.exceptions import ProviderError
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio_core.providers.provider import ChatRequest, Provider, provider


class OpenAIParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(ChatRequest):
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class OpenAIProvider(Provider):
    def __init__(self, config, api_key=None):
        super().__init__(config)
        self.API_KEY = api_key or os.getenv("OPENAI_API_KEY")

    def validate_request(self, request: OpenAIRequest):
        return OpenAIRequest(**request)

    async def generate_client(
        self, request: OpenAIRequest
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
                # tools=request.tools,
                functions=request.functions,
                function_call="auto" if request.functions else None,
                stream=True,
                response_format=request.response_format,
                **request.parameters.model_dump(),
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            yield chunk.model_dump()
