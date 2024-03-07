import asyncio
import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import anthropic
from anthropic import Anthropic
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class ClaudeParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(4096, ge=1)
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
                client.messages.stream,
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                **request.parameters.dict(),
            )
        except anthropic._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        with response as stream:
            for chunk in stream:
                if isinstance(
                    chunk,
                    anthropic.types.content_block_delta_event.ContentBlockDeltaEvent,
                ):
                    yield ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    content=chunk.delta.text, role="assistant"
                                ),
                                finish_reason=None,
                                index=0,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump(),
                elif isinstance(
                    chunk, anthropic.types.message_stop_event.MessageStopEvent
                ):
                    yield ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump(),
