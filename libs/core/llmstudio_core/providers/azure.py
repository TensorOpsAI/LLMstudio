import ast
import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Generator, Union

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


@provider
class AzureProvider(ProviderCore):
    def __init__(
        self, config, api_key=None, api_endpoint=None, api_version=None, base_url=None
    ):
        super().__init__(config)
        self.API_KEY = api_key or os.getenv("AZURE_API_KEY")
        self.API_ENDPOINT = api_endpoint
        self.API_VERSION = api_version or os.getenv("AZURE_API_VERSION")
        self.BASE_URL = base_url
        self.is_llama = False
        self.has_tools_functions = False

        if self.BASE_URL and (self.API_ENDPOINT is None):
            self._client = OpenAI(
                api_key=self.API_KEY,
                base_url=self.BASE_URL,
            )
        else:
            self._client = AzureOpenAI(
                api_key=self.API_KEY,
                azure_endpoint=self.API_ENDPOINT,
                api_version=self.API_VERSION,
            )

    @staticmethod
    def _provider_config_name():
        return "azure"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(self, request: ChatRequest) -> Any:
        """Generate an AzureOpenAI client"""

        self.is_llama = "llama" in request.model.lower()
        self.is_openai = "gpt" in request.model.lower()
        self.has_tools = request.parameters.get("tools") is not None
        self.has_functions = request.parameters.get("functions") is not None

        try:
            messages = self.prepare_messages(request)

            # Prepare the optional tool-related arguments
            tool_args = {}
            if not self.is_llama and self.has_tools and self.is_openai:
                tool_args = {
                    "tools": request.parameters.get("tools"),
                    "tool_choice": "auto" if request.parameters.get("tools") else None,
                }

            # Prepare the optional function-related arguments
            function_args = {}
            if not self.is_llama and self.has_functions and self.is_openai:
                function_args = {
                    "functions": request.parameters.get("functions"),
                    "function_call": "auto"
                    if request.parameters.get("functions")
                    else None,
                }

            # Prepare the base arguments
            base_args = {
                "model": request.model,
                "messages": messages,
                "stream": True,
            }

            # Combine all arguments
            combined_args = {
                **base_args,
                **tool_args,
                **function_args,
                **request.parameters,
            }
            # Perform the asynchronous call
            return await asyncio.to_thread(
                self._client.chat.completions.create, **combined_args
            )

        except openai._exceptions.APIConnectionError as e:
            raise ProviderError(f"There was an error reaching the endpoint: {e}")

        except openai._exceptions.APIStatusError as e:
            raise ProviderError(e.response.json())

    def generate_client(self, request: ChatRequest) -> Any:
        """Generate an AzureOpenAI client"""

        self.is_llama = "llama" in request.model.lower()
        self.is_openai = "gpt" in request.model.lower()
        self.has_tools = request.parameters.get("tools") is not None
        self.has_functions = request.parameters.get("functions") is not None

        try:
            messages = self.prepare_messages(request)

            # Prepare the optional tool-related arguments
            tool_args = {}
            if not self.is_llama and self.has_tools and self.is_openai:
                tool_args = {
                    "tools": request.parameters.get("tools"),
                    "tool_choice": "auto" if request.parameters.get("tools") else None,
                }

            # Prepare the optional function-related arguments
            function_args = {}
            if not self.is_llama and self.has_functions and self.is_openai:
                function_args = {
                    "functions": request.parameters.get("functions"),
                    "function_call": "auto"
                    if request.parameters.get("functions")
                    else None,
                }

            # Prepare the base arguments
            base_args = {
                "model": request.model,
                "messages": messages,
                "stream": True,
            }

            # Combine all arguments
            combined_args = {
                **base_args,
                **tool_args,
                **function_args,
                **request.parameters,
            }
            return self._client.chat.completions.create(**combined_args)

        except openai._exceptions.APIConnectionError as e:
            raise ProviderError(f"There was an error reaching the endpoint: {e}")

        except openai._exceptions.APIStatusError as e:
            raise ProviderError(e.response.json())

    def prepare_messages(self, request: ChatRequest):
        if self.is_llama and (self.has_tools or self.has_functions):
            user_message = self.convert_to_openai_format(request.chat_input)
            content = "<|begin_of_text|>"
            content = self.add_system_message(
                user_message,
                content,
                request.parameters.get("tools"),
                request.parameters.get("functions"),
            )
            content = self.add_conversation(user_message, content)
            return [{"role": "user", "content": content}]
        else:
            return (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            )

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        if self.is_llama and (self.has_tools or self.has_functions):
            for chunk in self.handle_tool_response(response, **kwargs):
                if chunk:
                    yield chunk
        else:
            for chunk in response:
                c = chunk.model_dump()
                if c.get("choices"):
                    yield c

    def parse_response(self, response: Generator, **kwargs) -> Any:
        if self.is_llama and (self.has_tools or self.has_functions):
            for chunk in self.handle_tool_response(response, **kwargs):
                if chunk:
                    yield chunk
        else:
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

    def add_system_message(
        self, openai_message: list, llama_message: str, tools: list, functions: list
    ) -> str:
        system_message = ""
        system_message_found = False
        for message in openai_message:
            if message["role"] == "system" and message["content"] is not None:
                system_message_found = True
                system_message = f"""
        <|start_header_id|>system<|end_header_id|>
        {message['content']}
        """
        if not system_message_found:
            system_message = """
      <|start_header_id|>system<|end_header_id|>
      You are a helpful AI assistant.
      """

        if tools:
            system_message = system_message + self.add_tool_instructions(tools)

        if functions:
            system_message = system_message + self.add_function_instructions(functions)

        end_tag = "\n<|eot_id|>"
        return llama_message + system_message + end_tag

    def add_tool_instructions(self, tools: list) -> str:
        tool_prompt = """
    You have access to the following tools:
    """

        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                tool_prompt += (
                    f"Use the function '{func['name']}' to '{func['description']}':\n"
                )
                params_info = json.dumps(func["parameters"], indent=4)
                tool_prompt += f"Parameters format:\n{params_info}\n\n"

        tool_prompt += """
If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}
IMPORTANT: IT IS VITAL THAT YOU NEVER ADD A PREFIX OR A SUFFIX TO THE FUNCTION CALL.

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}
NOTE: There is no prefix before the symbol '§' and nothing comes after the call is done.

    Reminder:
    - Function calls MUST follow the specified format.
    - Only call one function at a time.
    - Required parameters MUST be specified.
    - Put the entire function call reply on one line.
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
    - If you have already called a tool and got the response for the users question please reply with the response.
    """

        return tool_prompt

    def add_function_instructions(self, functions: list) -> str:
        function_prompt = """
You have access to the following functions:
"""

        for func in functions:
            function_prompt += (
                f"Use the function '{func['name']}' to: '{func['description']}'\n"
            )
            params_info = json.dumps(func["parameters"], indent=4)
            function_prompt += f"{params_info}\n\n"

        function_prompt += """
If you choose to use a function to produce this response, ONLY reply in the following format with no prefix or suffix:
§{"type": "function", "name": "FUNCTION_NAME", "parameters": {"PARAMETER_NAME": PARAMETER_VALUE}}

Here is an example of the output I desiere when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}

Reminder:
- Function calls MUST follow the specified format.
- Only call one function at a time.
- NEVER call more than one function at a time.
- Required parameters MUST be specified.
- Put the entire function call reply on one line.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
- If you have already called a function and got the response for the user's question, please reply with the response.
"""
        return function_prompt

    def add_conversation(self, openai_message: list, llama_message: str) -> str:
        conversation_parts = []
        for message in openai_message:
            if message["role"] == "system":
                continue
            elif message["role"] == "user" and isinstance(message["content"], str):
                try:
                    # Attempt to safely evaluate the string to a Python object
                    content_as_list = ast.literal_eval(message["content"])
                    if isinstance(content_as_list, list):
                        # If the content is a list, process each nested message
                        for nested_message in content_as_list:
                            conversation_parts.append(
                                self.format_message(nested_message)
                            )
                    else:
                        # If the content is not a list, append it directly
                        conversation_parts.append(self.format_message(message))
                except (ValueError, SyntaxError):
                    # If evaluation fails or content is not a list/dict string, append the message directly
                    conversation_parts.append(self.format_message(message))
            else:
                # For all other messages, use the existing formatting logic
                conversation_parts.append(self.format_message(message))

        return llama_message + "".join(conversation_parts)

    def format_message(self, message: dict) -> str:
        """Format a single message for the conversation."""
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
