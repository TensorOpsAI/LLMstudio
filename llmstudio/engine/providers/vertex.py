import asyncio
import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import google.generativeai as genai
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class VertexAIParameters(BaseModel):
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[float] = Field(default=1, ge=0, le=1)
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_output_tokens: Optional[float] = Field(default=8192, ge=0, le=8192)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class VertexAIRequest(ChatRequest):
    parameters: Optional[VertexAIParameters] = VertexAIParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    chat_input: Any


@provider
class VertexAIProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    def validate_request(self, request: VertexAIRequest):
        return VertexAIRequest(**request)

    async def generate_client(
        self, request: VertexAIRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""
        try:
            # Init genai
            genai.configure(api_key=request.api_key or self.GOOGLE_API_KEY)

            # Define model
            model = genai.GenerativeModel(request.model)

            # Generate content
            return await asyncio.to_thread(
                model.generate_content, request.chat_input, stream=True
            )

        except Exception as e:
            # Handle any other exceptions that might occur
            raise HTTPException(status_code=500, detail=str(e))

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:

            # Parse google chunk response into ChatCompletionChunk
            yield ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=chunk.text, role="assistant"),
                        finish_reason=None,
                        index=0,
                    )
                ],
                created=int(time.time()),
                model=kwargs.get("request").model,
                object="chat.completion.chunk",
            ).model_dump()

        # Create the closing chunk
        yield ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()
