import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Generator, Union

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import OpenAI


@provider
class SelfHostedProvider(ProviderCore):
    def __init__(
        self,
        config,
        api_key=None,
        base_url=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.API_KEY = api_key or os.getenv("API_KEY")
        self.BASE_URL = base_url or os.getenv("BASE_URL")
        self.has_tools_functions = False

        self._client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.BASE_URL,
        )

    @staticmethod
    def _provider_config_name():
        return "self-hosted"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate an OpenAI client"""

        try:
            return self._client.chat.completions.create(
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=True,
                **request.parameters,
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))

    def prepare_messages(self, request: ChatRequest):
        return (
            [{"role": "user", "content": request.chat_input}]
            if isinstance(request.chat_input, str)
            else request.chat_input
        )

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = self.parse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        """
        Processes a generator response and yields processed chunks.

        If `is_llama` is True and tools or functions are enabled, it processes the response
        using `handle_tool_response`. Otherwise, it processes each chunk and yields only those
        containing "choices".

        Args:
            response (Generator): The response generator to process.
            **kwargs: Additional arguments for tool handling.

        Yields:
            Any: Processed response chunks.
        """

        for chunk in response:
            c = chunk.model_dump()
            if c.get("choices"):
                yield c

    def handle_tool_response(self, response: AsyncGenerator, **kwargs) -> Generator:
        """
        Asynchronously handles tool responses by parsing the content for function calls or tool activations.
        It processes the response chunks to extract and execute embedded function or tool calls, then yields
        the processed chunks or the results of these calls.

        Args:
            response: An asynchronous generator yielding response chunks from the tool or function.
            **kwargs: Additional keyword arguments that may be required for processing.

        Yields:
            An asynchronous generator of strings or processed chunks, depending on whether the response
            contains function or tool calls that need to be executed or just plain content to be returned.
        """

        function_call_buffer = ""
        saving = False
        normal_call_chunks = []
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                if (
                    "§" in chunk.choices[0].delta.content
                    or "<|python_tag|>" in chunk.choices[0].delta.content
                ):
                    saving = True

            if saving:
                function_call_buffer += chunk.choices[0].delta.content
                if chunk.choices[0].finish_reason == "stop":
                    cleaned_buffer = function_call_buffer.replace("§", "").replace(
                        "<|python_tag|>", ""
                    )
                    result_dict = eval(cleaned_buffer)

                    if self.has_functions:

                        # Create first chunk
                        first_chunk = self.create_tool_first_chunk(kwargs)
                        yield first_chunk

                        name_chunk = self.create_function_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self.create_function_argument_chunk(
                            parameters, kwargs
                        )
                        yield args_chunk

                        finish_chunk = self.create_function_finish_chunk(kwargs)
                        yield finish_chunk

                    if self.has_tools:
                        name_chunk = self.create_tool_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self.create_tool_argument_chunk(parameters, kwargs)
                        yield args_chunk

                        finish_chunk = self.create_tool_finish_chunk(kwargs)
                        yield finish_chunk

            else:
                normal_call_chunks.append(chunk)
                if chunk.choices[0].finish_reason == "stop":
                    for chunk in normal_call_chunks:
                        normal_call_chunks.append(chunk)
                if chunk.choices[0].finish_reason == "stop":
                    for chunk in normal_call_chunks:
                        yield chunk.model_dump()

    def create_tool_name_chunk(self, function_name: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=str(uuid.uuid4()),
                                function=ChoiceDeltaToolCallFunction(
                                    name=function_name,
                                    arguments="",
                                    type="function",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_function_name_chunk(self, function_name: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=ChoiceDeltaFunctionCall(
                            arguments="", name=function_name
                        ),
                        role="assistant",
                        tool_calls=None,
                        refusal=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def create_tool_finish_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(),
                    finish_reason="tool_calls",
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_tool_argument_chunk(self, content: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments=content,
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
        ).model_dump()

    def create_function_argument_chunk(self, content: str, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=ChoiceDeltaFunctionCall(
                            arguments=content, name=None
                        ),
                        role=None,
                        tool_calls=None,
                    ),
                    finish_reason=None,
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def create_tool_first_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None,
                        function_call=None,
                        role="assistant",
                        tool_calls=None,
                    ),
                    index=0,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            usage=None,
        ).model_dump()

    def create_function_finish_chunk(self, kwargs: dict) -> dict:
        return ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None, function_call=None, role=None, tool_calls=None
                    ),
                    finish_reason="function_call",
                    index=0,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=kwargs.get("request").model,
            object="chat.completion.chunk",
            system_fingerprint=None,
            usage=None,
        ).model_dump()

    def convert_to_openai_format(self, message: Union[str, list]) -> list:
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        return message

    def format_message(self, message: dict) -> str:
        """
        Formats a single message dictionary into a structured string for a conversation.

        The formatting depends on the content of the message, such as tool calls,
        function calls, or simple user/assistant messages. Each type of message
        is formatted with specific headers and tags.

        Args:
            message (dict): A dictionary containing message details. Expected keys
                            include "role", "content", and optionally "tool_calls",
                            "tool_call_id", or "function_call".

        Returns:
            str: A formatted string representing the message. Returns an empty
            string if the message cannot be formatted.
        """
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                return f"""
        <|start_header_id|>assistant<|end_header_id|>
        <function={function_name}>{arguments}</function>
        <|eom_id|>
        """
        elif "tool_call_id" in message:
            tool_response = message["content"]
            return f"""
    <|start_header_id|>ipython<|end_header_id|>
    {tool_response}
    <|eot_id|>
    """
        elif "function_call" in message:
            function_name = message["function_call"]["name"]
            arguments = message["function_call"]["arguments"]
            return f"""
        <|start_header_id|>assistant<|end_header_id|>
        <function={function_name}>{arguments}</function>
        <|eom_id|>
        """
        elif (
            message["role"] in ["assistant", "user"] and message["content"] is not None
        ):
            return f"""
    <|start_header_id|>{message['role']}<|end_header_id|>
    {message['content']}
    <|eot_id|>
    """
        elif message["role"] == "function":
            function_response = message["content"]
            return f"""
    <|start_header_id|>ipython<|end_header_id|>
    {function_response}
    <|eot_id|>
    """
        return ""
