import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import requests
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class ClaudeParameters(BaseModel):
    temperature: Optional[float] = Field(default=0, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=4096, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[int] = Field(default=5, ge=0, le=500)


class AnthropicRequest(ChatRequest):
    parameters: Optional[ClaudeParameters] = ClaudeParameters()
    tools: Optional[list[str]] = Field(default=None)


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
            api_key = request.api_key or self.API_KEY
            url = f"https://api.anthropic.com/v1/messages"
            headers = {
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            data = {
                "model": request.model,
                "stream": True,
                # "tools": request.tools,
                "messages": (
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                **request.parameters.model_dump(),
            }

            return await asyncio.to_thread(
                requests.post,
                url,
                headers=headers,
                json=data,
                stream=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response.iter_content(chunk_size=None):
            chunk = chunk.decode("utf-8")
            if chunk.startswith(("event: content_block_stop", "event: message_stop")):
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()
            else:
                try:
                    chunk = json.loads(
                        chunk.split("event: content_block_delta\ndata: ")[1]
                    )
                    yield ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    content=chunk.get("delta").get("text"),
                                    role="assistant",
                                ),
                                finish_reason=None,
                                index=0,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump()
                except:
                    pass
