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
    List,
    Optional,
    Tuple,
    Union,
)

import tiktoken
from anthropic import Anthropic
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import Function
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
    chat_input: Any
    parameters: Optional[BaseModel] = None
    is_stream: Optional[bool] = False
    has_end_token: Optional[bool] = False
    functions: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None


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
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []

        async for chunk in self.parse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chunks.append(chunk)
            if request.is_stream:
                chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                if chunk.get("choices")[0].get("finish_reason") != "stop":
                    yield chunk.get("choices")[0].get("delta").get("content")

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]

        response, output_string = self.join_chunks(chunks, request)

        metrics = self.calculate_metrics(
            request.chat_input,
            response,
            request.model,
            start_time,
            time.time(),
            first_token_time,
            token_times,
            token_count,
        )

        response = {
            **response.model_dump(),
            "id": str(uuid.uuid4()),
            "session_id": request.session_id,
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": output_string,
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": request.model,
            "timestamp": time.time(),
            "parameters": request.parameters.model_dump(),
            "metrics": metrics,
        }

        self.save_log(response)

        if not request.is_stream:
            yield response

    def join_chunks(self, chunks, request):
        from llmstudio.engine.providers.azure import AzureRequest
        from llmstudio.engine.providers.openai import OpenAIRequest

        if chunks[-1].get("choices")[0].get("finish_reason") == "tool_calls":
            tool_calls = [
                chunk.get("choices")[0].get("delta").get("tool_calls")[0]
                for chunk in chunks[1:-1]
            ]
            tool_call_id = tool_calls[0].get("id")
            tool_call_name = tool_calls[0].get("function").get("name")
            tool_call_type = tool_calls[0].get("type")
            tool_call_arguments = ""
            for chunk in tool_calls[1:]:
                tool_call_arguments += chunk.get("function").get("arguments")

            return (
                ChatCompletion(
                    id=chunks[-1].get("id"),
                    created=chunks[-1].get("created"),
                    model=chunks[-1].get("model"),
                    object="chat.completion",
                    choices=[
                        Choice(
                            finish_reason="tool_calls",
                            index=0,
                            logprobs=None,
                            message=ChatCompletionMessage(
                                content=None,
                                role="assistant",
                                function_call=None,
                                tool_calls=[
                                    ChatCompletionMessageToolCall(
                                        id=tool_call_id,
                                        function=Function(
                                            arguments=tool_call_arguments,
                                            name=tool_call_name,
                                        ),
                                        type=tool_call_type,
                                    )
                                ],
                            ),
                        )
                    ],
                ),
                tool_call_arguments,
            )
        elif chunks[-1].get("choices")[0].get("finish_reason") == "function_call":
            function_calls = [
                chunk.get("choices")[0].get("delta").get("function_call")
                for chunk in chunks[1:-1]
            ]

            if isinstance(request, AzureRequest):
                function_call_name = function_calls[0].get("name")
            elif isinstance(request, OpenAIRequest):
                function_call_name = (
                    chunks[0]
                    .get("choices")[0]
                    .get("delta")
                    .get("function_call")
                    .get("name")
                )
            function_call_arguments = ""
            for chunk in function_calls:
                if isinstance(request, AzureRequest):
                    part = chunk.get("arguments", "")
                    if part:
                        function_call_arguments += part
                elif isinstance(request, OpenAIRequest):
                    function_call_arguments += chunk.get("arguments")

            return (
                ChatCompletion(
                    id=chunks[-1].get("id"),
                    created=chunks[-1].get("created"),
                    model=chunks[-1].get("model"),
                    object="chat.completion",
                    choices=[
                        Choice(
                            finish_reason="function_call",
                            index=0,
                            logprobs=None,
                            message=ChatCompletionMessage(
                                content=None,
                                role="assistant",
                                tool_calls=None,
                                function_call=FunctionCall(
                                    arguments=function_call_arguments,
                                    name=function_call_name,
                                ),
                            ),
                        )
                    ],
                ),
                function_call_arguments,
            )
        elif chunks[-1].get("choices")[0].get("finish_reason") == "stop":
            if isinstance(request, AzureRequest) or isinstance(request, OpenAIRequest):
                start_index = 1
            else:
                start_index = 0

            stop_content = "".join(
                filter(
                    None,
                    [
                        chunk.get("choices")[0].get("delta").get("content")
                        for chunk in chunks[start_index:]
                    ],
                )
            )

            return (
                ChatCompletion(
                    id=chunks[-1].get("id"),
                    created=chunks[-1].get("created"),
                    model=chunks[-1].get("model"),
                    object="chat.completion",
                    choices=[
                        Choice(
                            finish_reason="stop",
                            index=0,
                            logprobs=None,
                            message=ChatCompletionMessage(
                                content=stop_content,
                                role="assistant",
                                function_call=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                ),
                stop_content,
            )

    async def parse_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[str, None]:
        pass

    def calculate_metrics(
        self,
        input: Any,
        output: Any,
        model: str,
        start_time: float,
        end_time: float,
        first_token_time: float,
        token_times: Tuple[float, ...],
        token_count: int,
    ) -> Dict[str, Any]:
        """Calculates metrics based on token times and output"""
        model_config = self.config.models[model]
        input_tokens = len(self.tokenizer.encode(self.input_to_string(input)))
        output_tokens = len(self.tokenizer.encode(self.output_to_string(output)))

        input_cost = model_config.input_token_cost * input_tokens
        output_cost = model_config.output_token_cost * output_tokens

        total_time = end_time - start_time
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": input_cost + output_cost,
            "latency_s": total_time,
            "time_to_first_token_s": first_token_time - start_time,
            "inter_token_latency_s": sum(token_times) / len(token_times),
            "tokens_per_second": token_count / total_time,
        }

    def input_to_string(self, input):
        if isinstance(input, str):
            return input
        else:
            return "".join(
                [
                    message.get("content", "")
                    for message in input
                    if message.get("content") is not None
                ]
            )

    def output_to_string(self, output):
        if output.choices[0].finish_reason == "stop":
            return output.choices[0].message.content
        elif output.choices[0].finish_reason == "tool_calls":
            return output.choices[0].message.tool_calls[0].function.arguments
        elif output.choices[0].finish_reason == "function_call":
            return output.choices[0].message.function_call.arguments

    def get_end_token_string(self, metrics: Dict[str, Any]) -> str:
        return f"{self.END_TOKEN},input_tokens={metrics['input_tokens']},output_tokens={metrics['output_tokens']},cost_usd={metrics['cost_usd']},latency_s={metrics['latency_s']:.5f},time_to_first_token_s={metrics['time_to_first_token_s']:.5f},inter_token_latency_s={metrics['inter_token_latency_s']:.5f},tokens_per_second={metrics['tokens_per_second']:.2f}"

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
