import asyncio
import os
import time
import uuid
from typing import Optional

import anthropic
from anthropic import Anthropic
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from llmstudio.engine.providers.provider import ChatRequest, Provider


class ClaudeParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens_to_sample: Optional[int] = Field(256, ge=1)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[int] = Field(5, ge=0, le=500)


class AnthropicRequest(ChatRequest):
    parameters: Optional[ClaudeParameters] = ClaudeParameters()


class AnthropicProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("ANTHROPIC_API_KEY")

    async def chat(self, request: AnthropicRequest):
        """Chat with the Anthropic API"""
        try:
            request = AnthropicRequest(**request)
            await super().chat(request)
            client = Anthropic(api_key=request.api_key or self.API_KEY)

            start_time = time.time()
            response = await asyncio.to_thread(
                client.completions.create,
                model=request.model,
                prompt=f"{anthropic.HUMAN_PROMPT} {request.chat_input} {anthropic.AI_PROMPT}",
                stream=request.is_stream,
                **request.parameters.model_dump(),
            )

            if request.is_stream:
                return StreamingResponse(
                    self.generate_stream(response, request, start_time)
                )
            else:
                return self.generate_response(
                    response, request, time.time() - start_time
                )
        except ValidationError as e:
            errors = e.errors()
            raise HTTPException(status_code=422, detail=errors)
        except anthropic._exceptions.APIError as e:
            raise HTTPException(
                status_code=e.status_code, detail=e.response.json()
            )

    def generate_response(
        self, response: dict, request: AnthropicRequest, latency: float
    ):
        """Generates a response from the Anthropic API"""
        input_tokens, input_cost = self.calculate_tokens_and_cost(
            request.chat_input, request.model, "input"
        )
        output_tokens, output_cost = self.calculate_tokens_and_cost(
            response.completion, request.model, "output"
        )

        return {
            "id": uuid.uuid4(),
            "chatInput": request.chat_input,
            "chatOutput": response.completion,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "timestamp": time.time(),
            "model": request.model,
            "parameters": request.parameters.model_dump(),
            "latency": latency,
        }

    def generate_stream(
        self, response: dict, request: AnthropicRequest, start_time: float
    ):
        """Generates a stream of responses from the Anthropic API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []

        for chunk in response:
            current_time = time.time()
            if first_token_time is None:
                first_token_time = current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            if chunk.stop_reason != "stop_sequence":
                chunk_content = chunk.completion
                chat_output += chunk_content
                yield chunk_content
            else:
                if request.has_end_token:
                    input_tokens, input_cost = self.calculate_tokens_and_cost(
                        request.chat_input, request.model, "input"
                    )
                    (
                        output_tokens,
                        output_cost,
                    ) = self.calculate_tokens_and_cost(
                        chat_output, request.model, "output"
                    )
                    total_time = current_time - start_time
                    ttft = first_token_time - start_time
                    inter_token_latency = (
                        sum(token_times) / len(token_times)
                        if token_times
                        else 0
                    )
                    tokens_per_second = (
                        output_tokens / total_time if total_time > 0 else 0
                    )

                    yield f"{self.END_TOKEN},input_tokens={input_tokens},output_tokens={output_tokens},cost={input_cost + output_cost},latency={total_time:.5f},time_to_first_token={ttft:.5f},inter_token_latency={inter_token_latency:.5f},tokens_per_second={tokens_per_second:.2f}"

    def calculate_tokens_and_cost(self, input: str, model: str, type: str):
        """Returns the number of tokens and the cost of the input/output string"""
        model_config = self.config.models[model]
        tokens = Anthropic().count_tokens(input)

        token_cost = (
            model_config.input_token_cost
            if type == "input"
            else model_config.output_token_cost
        )
        return tokens, token_cost * tokens
