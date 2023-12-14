import json
import os
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    Optional,
    Tuple,
    Union,
)

import tiktoken
from anthropic import Anthropic
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from tokenizers import Tokenizer

from llmstudio.tracking.tracker import tracker

provider_registry = {}


def provider(cls):
    """Decorator to register a new provider."""
    provider_registry[cls.__name__] = cls
    return cls


class ChatRequest(BaseModel):
    api_key: Optional[str] = None
    model: str
    chat_input: str
    parameters: Optional[BaseModel] = None
    is_stream: Optional[bool] = False
    has_end_token: Optional[bool] = False


class Provider:
    END_TOKEN = "<END_TOKEN>"

    def __init__(self, config):
        self.config = config
        self.tokenizer: Tokenizer = self._get_tokenizer()

    async def chat(
        self, request: ChatRequest
    ) -> Union[StreamingResponse, JSONResponse]:
        """Makes a chat connection with the provider's API"""
        try:
            request = self.validate_request(request)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        self.validate_model(request)

        start_time = time.time()
        response = await self.generate_client(request)

        response_handler = self.handle_response(request, response, start_time)
        if request.is_stream:
            return StreamingResponse(response_handler)
        else:
            return JSONResponse(content=await response_handler.__anext__())

    def validate_request(self, request: ChatRequest):
        pass

    def validate_model(self, request: ChatRequest):
        if request.model not in self.config.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} is not supported by {self.config.name}",
            )

    async def generate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate the provider's client"""

    async def handle_response(
        self, request: ChatRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from an API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        async for chunk in self.parse_response(response):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chat_output += chunk
            if request.is_stream:
                yield chunk

        metrics = self.calculate_metrics(
            request.chat_input,
            chat_output,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
        )

        if request.is_stream and request.has_end_token:
            yield self.get_end_token_string(metrics)

        response = self.generate_response(request, chat_output, metrics)

        self.save_log(response)

        if not request.is_stream:
            yield response

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        pass

    def generate_response(
        self,
        request: ChatRequest,
        chat_output: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generates a complete response with metrics"""
        return {
            "id": str(uuid.uuid4()),
            "chat_input": request.chat_input,
            "chat_output": chat_output,
            "timestamp": time.time(),
            "provider": self.config.id,
            "model": request.model,
            "metrics": metrics,
            "parameters": request.parameters.dict(),
        }

    def calculate_metrics(
        self,
        input: str,
        output: str,
        model: str,
        start_time: float,
        end_time: float,
        first_token_time: float,
        token_times: Tuple[float, ...],
        token_count: int,
    ) -> Dict[str, Any]:
        """Calculates metrics based on token times and output"""
        model_config = self.config.models[model]
        input_tokens = len(self.tokenizer.encode(input))
        output_tokens = len(self.tokenizer.encode(output))

        input_cost = model_config.input_token_cost * input_tokens
        output_cost = model_config.output_token_cost * output_tokens

        total_time = end_time - start_time
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": input_cost + output_cost,
            "latency": total_time,
            "time_to_first_token": first_token_time - start_time,
            "inter_token_latency": sum(token_times) / len(token_times),
            "tokens_per_second": token_count / total_time,
        }

    def get_end_token_string(self, metrics: Dict[str, Any]) -> str:
        return f"{self.END_TOKEN},input_tokens={metrics['input_tokens']},output_tokens={metrics['output_tokens']},cost={metrics['cost']},latency={metrics['latency']:.5f},time_to_first_token={metrics['time_to_first_token']:.5f},inter_token_latency={metrics['inter_token_latency']:.5f},tokens_per_second={metrics['tokens_per_second']:.2f}"

    def _get_tokenizer(self) -> Tokenizer:
        return {
            "anthropic": Anthropic().get_tokenizer(),
            "cohere": Tokenizer.from_pretrained("Cohere/command-nightly"),
        }.get(self.config.id, tiktoken.get_encoding("cl100k_base"))

    def save_log(self, response: Dict[str, Any]):
        local = False  # NB: Make this dynamic
        if local:
            file_name = Path(
                os.path.join(os.path.dirname(__file__), "..", "logs.jsonl")
            )
            if not os.path.exists(file_name):
                with open(file_name, "w") as f:
                    pass

            with open(file_name, "a") as f:
                f.write(json.dumps(response) + "\n")
        else:
            tracker.log(response)
