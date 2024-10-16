import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Generator, Optional

import requests
from llmstudio_core.exceptions import ProviderError
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel, Field

from llmstudio_core.providers.provider import ChatRequest, BaseProvider, provider


class OllamaParameters(BaseModel):
    temperature: Optional[float] = Field(0.8, ge=0, le=1)
    num_predict: Optional[int] = Field(128, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    top_k: Optional[int] = Field(40, ge=0, le=500)


class OllamaRequest(ChatRequest):
    parameters: Optional[OllamaParameters] = OllamaParameters()


@provider
class OllamaProvider(BaseProvider):
    def __init__(self, config, api_key=None):
        super().__init__(config)

    @staticmethod
    def _provider_config_name():
        return "ollama"

    def validate_request(self, request: OllamaRequest):
        return OllamaRequest(**request)

    async def generate_client(
        self, request: OllamaRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an Ollama client"""
        try:
            return await asyncio.to_thread(
                requests.post,
                url="http://localhost:11434/api/generate", #TODO make this configurable
                json={
                    "model": request.model,
                    "prompt": request.chat_input,
                    "stream": True,
                    **request.parameters.dict(),
                },
                stream=True,
            )
        except requests.RequestException:
            raise ProviderError("Ollama is not running")

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "error" in chunk:
                raise ProviderError(chunk["error"])
            if chunk.get("done"):
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()
                break

            if chunk["response"] is not None:
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                content=chunk["response"], role="assistant"
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()
