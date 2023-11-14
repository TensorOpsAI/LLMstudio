import asyncio
import random
import time
from typing import Optional

import anthropic
from anthropic import Anthropic
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmstudio.engine.config import AnthropicConfig
from llmstudio.engine.constants import END_TOKEN
from llmstudio.engine.providers.base_provider import BaseProvider
from llmstudio.engine.utils import validate_provider_config


class ClaudeParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(300, ge=1, le=2048)
    top_p: Optional[float] = Field(0.999, ge=0, le=1)
    top_k: Optional[int] = Field(250, ge=1, le=500)


class AnthropicRequest(BaseModel):
    api_key: Optional[str]
    model: str
    chat_input: str
    parameters: Optional[ClaudeParameters] = ClaudeParameters()
    is_stream: Optional[bool] = False
    end_token: Optional[bool] = True


class AnthropicTest(BaseModel):
    api_key: Optional[str]
    api_secret: Optional[str]
    api_region: Optional[str]
    model: str


class AnthropicProvider(BaseProvider):
    def __init__(self, config: AnthropicConfig, api_key: dict):
        super().__init__()
        self.anthropic_config = validate_provider_config(config, api_key)

    async def chat(self, data: AnthropicRequest) -> dict:
        data = AnthropicRequest(**data)
        self.validate_model_field(data, ["claude-instant-1", "claude-instant-1.2", "claude-2"])
        loop = asyncio.get_event_loop()

        client = Anthropic(api_key=data.api_key)

        response = await loop.run_in_executor(
            None,
            lambda: client.completions.create(
                model=data.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {data.chat_input} {anthropic.AI_PROMPT}",
                max_tokens_to_sample=data.parameters.max_tokens,
                stream=data.is_stream,
                temperature=data.parameters.temperature,
                top_p=data.parameters.top_p,
                top_k=data.parameters.top_k,
            ),
        )

        if data.is_stream:
            return StreamingResponse(generate_stream_response(response, data))
        else:
            return {
                "id": random.randint(0, 1000),
                "chatInput": data.chat_input,
                "chatOutput": response.completion,
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
                "cost": 0,
                "timestamp": time.time(),
                "model": data.model,
                "parameters": data.parameters.dict(),
                "latency": 0,
            }

    async def test(self, data: AnthropicTest) -> bool:
        return 1


def get_cost(input_tokens: int, output_tokens: int) -> float:
    return None


def generate_stream_response(response, data):
    chat_output = ""
    for chunk in response:
        if chunk.stop_reason != "stop_sequence":
            chunk_content = chunk.completion
            chat_output += chunk_content
            yield chunk_content
        else:
            if data.end_token:
                yield f"{END_TOKEN},{0},{0},{0}"  # json
