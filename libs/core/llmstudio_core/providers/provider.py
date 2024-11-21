import time
import uuid
from abc import ABC, abstractmethod
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
from llmstudio_core.exceptions import ProviderError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import Function
from pydantic import BaseModel, ValidationError

provider_registry = {}


def provider(cls):
    """Decorator to register a new provider."""
    provider_registry[cls._provider_config_name()] = cls

    return cls


class ChatRequest(BaseModel):
    chat_input: Any
    model: str
    is_stream: Optional[bool] = False
    retries: Optional[int] = 0
    parameters: Optional[dict] = {}

    def __init__(self, **data):
        super().__init__(**data)
        base_model_fields = self.model_fields.keys()
        additional_params = {
            k: v for k, v in data.items() if k not in base_model_fields
        }
        self.parameters.update(additional_params)


class Provider(ABC):
    END_TOKEN = "<END_TOKEN>"

    def __init__(
        self,
        config: Any,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        base_url: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        self.config = config
        self.API_KEY = api_key
        self.api_endpoint = api_endpoint
        self.api_version = api_version
        self.base_url = base_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer()
        self.count = 0

    @abstractmethod
    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> Coroutine[Any, Any, Union[ChatCompletionChunk, ChatCompletion]]:
        raise NotImplementedError("Providers needs to have achat method implemented.")

    @abstractmethod
    def chat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ) -> Union[ChatCompletionChunk, ChatCompletion]:
        raise NotImplementedError("Providers needs to have chat method implemented.")

    @staticmethod
    @abstractmethod
    def _provider_config_name():
        raise NotImplementedError(
            "Providers need to implement the '_provider_config_name' property."
        )


class ProviderCore(Provider):
    END_TOKEN = "<END_TOKEN>"

    @abstractmethod
    def validate_request(self, request: ChatRequest):
        raise NotImplementedError("Providers need to implement the 'validate_request'.")

    @abstractmethod
    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate the provider's client"""
        raise NotImplementedError("Providers need to implement the 'agenerate_client'.")

    @abstractmethod
    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate the provider's client"""
        raise NotImplementedError("Providers need to implement the 'generate_client'.")

    @abstractmethod
    async def aparse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a aparse_response method.")

    @abstractmethod
    def parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        raise NotImplementedError("ProviderCore needs a parse_response method.")

    def validate_model(self, request: ChatRequest):
        if request.model not in self.config.models:
            raise ProviderError(
                f"Model {request.model} is not supported by {self.config.name}"
            )

    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ):

        """Makes a chat connection with the provider's API"""
        try:
            request = self.validate_request(
                dict(
                    chat_input=chat_input,
                    model=model,
                    is_stream=is_stream,
                    retries=retries,
                    parameters=parameters,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise ProviderError(str(e))

        self.validate_model(request)

        for _ in range(request.retries + 1):
            try:
                start_time = time.time()
                response = await self.agenerate_client(request)
                response_handler = self.ahandle_response(request, response, start_time)

                if request.is_stream:
                    return response_handler
                else:
                    return await response_handler.__anext__()
            # except HTTPException as e:
            #     if e.status_code == 429:
            #         continue  # Retry on rate limit error
            #     else:
            #         raise e  # Raise other HTTP exceptions
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")

    def chat(
        self,
        chat_input: Any,
        model: str,
        is_stream: Optional[bool] = False,
        retries: Optional[int] = 0,
        parameters: Optional[dict] = {},
        **kwargs,
    ):

        """Makes a chat connection with the provider's API"""
        try:
            request = self.validate_request(
                dict(
                    chat_input=chat_input,
                    model=model,
                    is_stream=is_stream,
                    retries=retries,
                    parameters=parameters,
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise ProviderError(str(e))

        self.validate_model(request)

        for _ in range(request.retries + 1):
            try:
                start_time = time.time()
                response = self.generate_client(request)
                response_handler = self.handle_response(request, response, start_time)

                if request.is_stream:
                    return response_handler
                else:
                    return response_handler.__next__()
            # except HTTPExceptio as e:
            #     if e.status_code == 429:
            #         continue  # Retry on rate limit error
            #     else:
            #         raise e  # Raise other HTTP exceptions
            except Exception as e:
                raise ProviderError(str(e))
        raise ProviderError("Too many requests")

    async def ahandle_response(
        self, request: ChatRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from an API"""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []

        async for chunk in self.aparse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chunks.append(chunk)
            if request.is_stream:
                chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                model = chunk.get("model")
                if chunk.get("choices")[0].get("finish_reason") != "stop":
                    chat_output = chunk.get("choices")[0].get("delta").get("content")
                    chunk = {
                        **chunk,
                        "id": str(uuid.uuid4()),
                        "chat_input": (
                            request.chat_input
                            if isinstance(request.chat_input, str)
                            else request.chat_input[-1]["content"]
                        ),
                        "chat_output": None,
                        "chat_output_stream": chat_output if chat_output else "",
                        "context": (
                            [{"role": "user", "content": request.chat_input}]
                            if isinstance(request.chat_input, str)
                            else request.chat_input
                        ),
                        "provider": self.config.id,
                        "model": (
                            request.model
                            if model and model.startswith(request.model)
                            else (model or request.model)
                        ),
                        "deployment": (
                            model
                            if model and model.startswith(request.model)
                            else (request.model if model != request.model else None)
                        ),
                        "timestamp": time.time(),
                        "parameters": request.parameters,
                        "metrics": None,
                    }
                    yield ChatCompletionChunk(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

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
            **(chunk if request.is_stream else response.model_dump()),
            "id": str(uuid.uuid4()),
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": output_string,
            "chat_output_stream": "",
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": (
                request.model
                if model and model.startswith(request.model)
                else (model or request.model)
            ),
            "deployment": (
                model
                if model and model.startswith(request.model)
                else (request.model if model != request.model else None)
            ),
            "timestamp": time.time(),
            "parameters": request.parameters,
            "metrics": metrics,
        }

        if request.is_stream:
            yield ChatCompletionChunk(**response)
        else:
            yield ChatCompletion(**response)

    def handle_response(
        self, request: ChatRequest, response: Generator, start_time: float
    ) -> Generator:
        """Handles the response from an API"""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0
        chunks = []

        for chunk in self.parse_response(response, request=request):
            token_count += 1
            current_time = time.time()
            first_token_time = first_token_time or current_time
            if previous_token_time is not None:
                token_times.append(current_time - previous_token_time)
            previous_token_time = current_time

            chunks.append(chunk)
            if request.is_stream:
                chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                model = chunk.get("model")
                if chunk.get("choices")[0].get("finish_reason") != "stop":
                    chat_output = chunk.get("choices")[0].get("delta").get("content")
                    chunk = {
                        **chunk,
                        "id": str(uuid.uuid4()),
                        "chat_input": (
                            request.chat_input
                            if isinstance(request.chat_input, str)
                            else request.chat_input[-1]["content"]
                        ),
                        "chat_output": None,
                        "chat_output_stream": chat_output if chat_output else "",
                        "context": (
                            [{"role": "user", "content": request.chat_input}]
                            if isinstance(request.chat_input, str)
                            else request.chat_input
                        ),
                        "provider": self.config.id,
                        "model": (
                            request.model
                            if model and model.startswith(request.model)
                            else (model or request.model)
                        ),
                        "deployment": (
                            model
                            if model and model.startswith(request.model)
                            else (request.model if model != request.model else None)
                        ),
                        "timestamp": time.time(),
                        "parameters": request.parameters,
                        "metrics": None,
                    }
                    yield ChatCompletionChunk(**chunk)

        chunks = [chunk[0] if isinstance(chunk, tuple) else chunk for chunk in chunks]
        model = next(chunk["model"] for chunk in chunks if chunk.get("model"))

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
            **(chunk if request.is_stream else response.model_dump()),
            "id": str(uuid.uuid4()),
            "chat_input": (
                request.chat_input
                if isinstance(request.chat_input, str)
                else request.chat_input[-1]["content"]
            ),
            "chat_output": output_string,
            "chat_output_stream": "",
            "context": (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            ),
            "provider": self.config.id,
            "model": (
                request.model
                if model and model.startswith(request.model)
                else (model or request.model)
            ),
            "deployment": (
                model
                if model and model.startswith(request.model)
                else (request.model if model != request.model else None)
            ),
            "timestamp": time.time(),
            "parameters": request.parameters,
            "metrics": metrics,
        }

        if request.is_stream:
            yield ChatCompletionChunk(**response)
        else:
            yield ChatCompletion(**response)

    def join_chunks(self, chunks, request):

        finish_reason = chunks[-1].get("choices")[0].get("finish_reason")
        if finish_reason == "tool_calls":
            tool_calls = {}
            for chunk in chunks:
                try:
                    data = chunk.get("choices")[0].get("delta").get("tool_calls")[0]
                    tool_calls.setdefault(data["index"], []).append(data)
                except TypeError:
                    continue

            tool_call_ids = [t[0].get("id") for t in tool_calls.values()]
            tool_call_names = [
                t[0].get("function").get("name") for t in tool_calls.values()
            ]
            tool_call_types = [
                t[0].get("function").get("type", "function")
                for t in tool_calls.values()
            ]

            tool_call_arguments_all = []
            for t in tool_calls.values():
                tool_call_arguments_all.append(
                    "".join(
                        chunk.get("function", {}).get("arguments", "") for chunk in t
                    )
                )

            tool_calls_parsed = [
                ChatCompletionMessageToolCall(
                    id=tool_call_id,
                    function=Function(
                        arguments=tool_call_arguments, name=tool_call_name
                    ),
                    type=tool_call_type,
                )
                for tool_call_arguments, tool_call_name, tool_call_type, tool_call_id in zip(
                    tool_call_arguments_all,
                    tool_call_names,
                    tool_call_types,
                    tool_call_ids,
                )
            ]

            try:
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
                                    tool_calls=tool_calls_parsed,
                                ),
                            )
                        ],
                    ),
                    str(tool_call_names + tool_call_arguments_all),
                )
            except Exception as e:
                raise e
        elif finish_reason == "function_call":
            function_calls = [
                chunk.get("choices")[0].get("delta").get("function_call")
                for chunk in chunks[:-1]
                if chunk.get("choices")
                and chunk.get("choices")[0].get("delta")
                and chunk.get("choices")[0].get("delta").get("function_call")
            ]

            function_call_name = function_calls[0].get("name")

            function_call_arguments = ""
            for chunk in function_calls:
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

        elif finish_reason == "stop" or finish_reason == "length":
            if self.__class__.__name__ in ("OpenAIProvider", "AzureProvider"):
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

        input_cost = self.calculate_cost(input_tokens, model_config.input_token_cost)
        output_cost = self.calculate_cost(output_tokens, model_config.output_token_cost)

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

    def calculate_cost(
        self, token_count: int, token_cost: Union[float, List[Dict[str, Any]]]
    ) -> float:
        if isinstance(token_cost, list):
            for cost_range in token_cost:
                if token_count >= cost_range.range[0] and (
                    token_count <= cost_range.range[1] or cost_range.range[1] is None
                ):
                    return cost_range.cost * token_count
        else:
            return token_cost * token_count
        return 0

    def input_to_string(self, input):
        if isinstance(input, str):
            return input
        else:
            result = []
            for message in input:
                if message.get("content") is not None:
                    if isinstance(message["content"], str):
                        result.append(message["content"])
                    elif (
                        isinstance(message["content"], list)
                        and message.get("role") == "user"
                    ):
                        for item in message["content"]:
                            if item.get("type") == "text":
                                result.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                result.append(url)
            return "".join(result)

    def output_to_string(self, output):
        if output.choices[0].finish_reason == "stop":
            return output.choices[0].message.content
        elif output.choices[0].finish_reason == "tool_calls":
            return output.choices[0].message.tool_calls[0].function.arguments
        elif output.choices[0].finish_reason == "function_call":
            return output.choices[0].message.function_call.arguments

    def get_end_token_string(self, metrics: Dict[str, Any]) -> str:
        return f"{self.END_TOKEN},input_tokens={metrics['input_tokens']},output_tokens={metrics['output_tokens']},cost_usd={metrics['cost_usd']},latency_s={metrics['latency_s']:.5f},time_to_first_token_s={metrics['time_to_first_token_s']:.5f},inter_token_latency_s={metrics['inter_token_latency_s']:.5f},tokens_per_second={metrics['tokens_per_second']:.2f}"

    def _get_tokenizer(self):
        return {}.get(self.config.id, tiktoken.get_encoding("cl100k_base"))
