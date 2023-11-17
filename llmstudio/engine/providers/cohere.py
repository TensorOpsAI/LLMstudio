import asyncio
import os
import time
import uuid
from typing import Optional

import cohere
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from tokenizers import Tokenizer

from llmstudio.engine.providers.provider import ChatRequest, Provider


class CommandParameters(BaseModel):
    temperature: Optional[float] = Field(0.75, ge=0, le=5)
    max_tokens: Optional[int] = Field(256, ge=1)
    p: Optional[float] = Field(0, ge=0, le=0.99)
    k: Optional[int] = Field(0, ge=0, le=500)
    frequency_penalty: Optional[float] = Field(0, ge=0)
    presence_penalty: Optional[float] = Field(0, ge=0, le=1)


class CohereRequest(ChatRequest):
    parameters: Optional[CommandParameters] = CommandParameters()


class CohereProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("COHERE_API_KEY")

    async def chat(self, request: CohereRequest):
        """Chat with the Cohere API"""
        try:
            request = CohereRequest(**request)
            await super().chat(request)
            co = cohere.Client(api_key=request.api_key or self.API_KEY)

            start_time = time.time()
            response = await asyncio.to_thread(
                co.generate,
                model=request.model,
                prompt=request.chat_input,
                stream=request.is_stream,
                **dict(request.parameters),
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

    def generate_response(
        self, response: dict, request: CohereRequest, latency: float
    ):
        """Generates a response from the Cohere API"""
        input_tokens, input_cost = self.calculate_tokens_and_cost(
            request.chat_input, request.model, "input"
        )
        output_tokens, output_cost = self.calculate_tokens_and_cost(
            response.generations[0].text, request.model, "output"
        )

        return {
            "id": uuid.uuid4(),
            "chatInput": request.chat_input,
            "chatOutput": response.generations[0].text,
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
        self, response: dict, request: CohereRequest, start_time: float
    ):
        """Generates a stream of responses from the Cohere API"""
        for chunk in response:
            if not chunk.is_finished:
                yield chunk.text

        if request.has_end_token:
            input_tokens, input_cost = self.calculate_tokens_and_cost(
                request.chat_input, request.model, "input"
            )
            output_tokens, output_cost = self.calculate_tokens_and_cost(
                response.generations[0].text, request.model, "output"
            )
            yield f"<END_TOKEN>,{input_tokens},{output_tokens},{input_cost+output_cost},{time.time()-start_time}"

    def calculate_tokens_and_cost(self, input: str, model: str, type: str):
        """Returns the number of tokens and the cost of the input/output string"""
        model_config = self.config.models[model]
        tokenizer = Tokenizer.from_pretrained("Cohere/command-nightly")
        tokens = len(tokenizer.encode(input).ids)

        token_cost = (
            model_config.input_token_cost
            if type == "input"
            else model_config.output_token_cost
        )
        return tokens, token_cost * tokens
