import ast
import asyncio
import json
import os
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import openai
import requests
from fastapi import HTTPException
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class AzureParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class AzureRequest(ChatRequest):
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Optional[AzureParameters] = AzureParameters()
    tools: Optional[List[Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class AzureProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("AZURE_API_KEY")
        self.API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
        self.API_VERSION = os.getenv("AZURE_API_VERSION")
        self.BASE_URL = os.getenv("AZURE_BASE_URL")
        self.is_llama = False
        self.has_tools = False
        self.has_functions = False

    def validate_request(self, request: AzureRequest):
        return AzureRequest(**request)

    async def generate_client(
        self, request: AzureRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""

        self.is_llama = "llama" in request.model.lower()
        self.is_openai = "gpt" in request.model.lower()
        self.has_tools = request.tools is not None
        self.has_functions = request.functions is not None

        # 1. Build headers
        headers = self._build_headers()
        messages = self._prepare_messages(request)
        payload = self._build_payload(request, messages)
        endpoint = (
            self.API_ENDPOINT
            if self.is_openai
            else self.BASE_URL
            if self.is_llama
            else ""
        )

        return await asyncio.to_thread(
            requests.post,
            endpoint,
            headers=headers,
            json=payload,
            stream=True,
        )

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:

        if self.is_openai:
            async for chunk in self._handle_openai_response(response):
                yield chunk

        if self.is_llama:
            async for chunk in self._handle_llama_response(response, **kwargs):
                yield chunk

    async def _handle_openai_response(
        self, response: AsyncGenerator
    ) -> AsyncGenerator[dict, None]:
        """
        Handles the response from the OpenAI API.

        Args:
            response (AsyncGenerator): The response generator from the OpenAI API.

        Yields:
            dict: Parsed JSON chunks from the response.
        """
        for binary_chunk in response.iter_lines():
            chunk = self._binary_chunk_to_json(binary_chunk)

            if not chunk:
                continue

            if chunk == "[DONE]":
                break

            yield chunk

    async def _handle_llama_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[dict, None]:
        """
        Handles the response from the Llama API, including tool and function calls.

        Args:
            response (AsyncGenerator): The response generator from the Llama API.
            **kwargs: Additional keyword arguments.

        Yields:
            dict: Parsed JSON chunks from the response.
        """
        function_call_buffer = ""
        saving = False
        normal_call_chunks = []

        for binary_chunk in response:
            chunk = self._binary_chunk_to_json(binary_chunk)

            # Ignore empty chunks
            if not chunk:
                continue

            # End loop
            if chunk == "[DONE]":
                break

            # Yield chunks when we have normal calls
            if not (self.has_tools or self.has_functions):
                yield chunk

            # Handle chunks when we have a tool call.
            if "content" in chunk["choices"][0]["delta"]:
                if (
                    "§" in chunk["choices"][0]["delta"]["content"]
                    or "<|python_tag|>" in chunk["choices"][0]["delta"]["content"]
                ):
                    saving = True

            if saving:
                function_call_buffer += chunk["choices"][0]["delta"]["content"]
                if chunk["choices"][0]["finish_reason"] == "stop":
                    cleaned_buffer = function_call_buffer.replace("§", "").replace(
                        "<|python_tag|>", ""
                    )
                    result_dict = eval(cleaned_buffer)

                    if self.has_functions:
                        # Create first chunk
                        first_chunk = self._create_first_chunk(kwargs)
                        yield first_chunk

                        name_chunk = self._create_function_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self._create_function_argument_chunk(
                            parameters, kwargs
                        )
                        yield args_chunk

                        finish_chunk = self._create_function_finish_chunk(kwargs)
                        yield finish_chunk
                        break

                    if self.has_tools:
                        # Create first chunk
                        first_chunk = self._create_first_chunk(kwargs)
                        yield first_chunk

                        name_chunk = self._create_tool_name_chunk(
                            result_dict["name"], kwargs
                        )
                        yield name_chunk

                        parameters = json.dumps(result_dict["parameters"])
                        args_chunk = self._create_tool_argument_chunk(
                            parameters, kwargs
                        )
                        yield args_chunk

                        finish_chunk = self._create_tool_finish_chunk(kwargs)
                        yield finish_chunk
                        break

            else:
                normal_call_chunks.append(chunk)
                if chunk["choices"][0]["finish_reason"] == "stop":
                    for chunk in normal_call_chunks:
                        yield chunk
                    break

    @staticmethod
    def _binary_chunk_to_json(binary_chunk: bytes) -> Union[dict, str]:
        """
        Convert a binary chunk to a JSON object or string.

        Args:
            binary_chunk (bytes): The binary data to convert.

        Returns:
            Union[dict, str]: The converted JSON object if successful, otherwise the original string.
        """
        json_str = binary_chunk.decode("utf-8").replace("data: ", "")
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            return json_str

    def _build_headers(self) -> dict:
        """
        Build the headers for the request based on the model type.

        Returns:
            dict: A dictionary containing the headers for the request.
        """

        if self.is_llama:
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.API_KEY}",
            }

        elif self.is_openai:
            return {
                "Content-Type": "application/json",
                "api-key": f"{self.API_KEY}",
            }

    def _build_payload(self, request: AzureRequest, messages: list) -> dict:
        """
        Build the payload for the request.

        Args:
            request (AzureRequest): The request object containing parameters and other details.
            messages (list): The list of messages to include in the payload.

        Returns:
            dict: The payload dictionary for the request.
        """

        # Payload for the request
        base_args = {
            "messages": messages,
            "temperature": request.parameters.temperature,
            "top_p": request.parameters.top_p,
            "max_tokens": request.parameters.max_tokens,
            "stream": True,
        }

        tool_args = {}
        function_args = {}
        if self.is_openai:

            # Prepare the optional tool-related arguments
            if self.has_tools:
                tool_args = {
                    "tools": request.tools,
                    "tool_choice": "auto" if request.tools else None,
                }

            # Prepare the optional function-related arguments
            if self.has_functions:
                function_args = {
                    "functions": request.functions,
                    "function_call": "auto" if request.functions else None,
                }

        payload = {
            **base_args,
            **tool_args,
            **function_args,
            **request.parameters.model_dump(),
        }

        return payload

    def _prepare_messages(self, request: AzureRequest) -> list:
        """
        Prepare the messages for the request.

        Args:
            request (AzureRequest): The request object containing parameters and other details.

        Returns:
            list: The list of messages to include in the payload.
        """
        if self.is_llama:
            user_message = self._convert_to_openai_format(request.chat_input)
            content = "<|begin_of_text|>"
            content = self._add_system_message(
                user_message, content, request.tools, request.functions
            )
            content = self._add_conversation(user_message, content)
            return [{"role": "user", "content": content}]
        else:
            return (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            )

    @staticmethod
    def _convert_to_openai_format(message: Union[str, list]) -> list:
        """
        Convert a message to OpenAI format.

        Args:
            message (Union[str, list]): The message to convert. It can be a string or a list of messages.

        Returns:
            list: A list of messages in OpenAI format.
        """
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        return message

    def _add_system_message(
        self, openai_message: list, llama_message: str, tools: list, functions: list
    ) -> str:
        """
        Add a system message to the Llama message.

        Args:
            openai_message (list): List of messages in OpenAI format.
            llama_message (str): The initial Llama message.
            tools (list): List of tools available for use.
            functions (list): List of functions available for use.

        Returns:
            str: The Llama message with the added system message.
        """
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
            system_message = system_message + self._add_tool_instructions(tools)

        if functions:
            system_message = system_message + self._add_function_instructions(functions)

        end_tag = "\n<|eot_id|>"
        return llama_message + system_message + end_tag

    def _add_tool_instructions(self, tools: list) -> str:
        """
        Generate a tool instruction prompt based on the provided tools.

        Args:
            tools (list): A list of tools, where each tool is a dictionary containing
                          information about the tool, including its type and function details.

        Returns:
            str: A formatted string containing the tool instructions.
        """

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

Here is an example of the output I desire when performing function call:
§{"type": "function", "name": "python_repl_ast", "parameters": {"query": "print(df.shape)"}}
NOTE: There is no prefix before the symbol '§' and nothing comes after the call is done.

    Reminder:
    - Function calls MUST follow the specified format.
    - Only call one function at a time.
    - Required parameters MUST be specified.
    - Put the entire function call reply on one line.
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
    - If you have already called a tool and got the response for the user's question please reply with the response.
    """

        return tool_prompt

    def _add_function_instructions(self, functions: list) -> str:
        """
        Generate a function instruction prompt based on the provided functions.

        Args:
            functions (list): A list of functions, where each function is a dictionary containing
                              information about the function, including its name, description, and parameters.

        Returns:
            str: A formatted string containing the function instructions.
        """
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

Here is an example of the output I desire when performing function call:
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

    def _add_conversation(self, openai_message: list[dict], llama_message: str) -> str:
        """
        Add conversation parts from OpenAI messages to the Llama message.

        Args:
            openai_message (list[dict]): A list of messages from OpenAI, where each message is a dictionary.
            llama_message (str): The initial Llama message to which conversation parts will be added.

        Returns:
            str: The combined conversation string.
        """
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
                                self._format_message(nested_message)
                            )
                    else:
                        # If the content is not a list, append it directly
                        conversation_parts.append(self._format_message(message))
                except (ValueError, SyntaxError):
                    # If evaluation fails or content is not a list/dict string, append the message directly
                    conversation_parts.append(self._format_message(message))
            else:
                # For all other messages, use the existing formatting logic
                conversation_parts.append(self._format_message(message))

        return llama_message + "".join(conversation_parts)

    def _format_message(self, message: dict) -> str:
        """
        Format a single message for the conversation.

        Args:
            message (dict): A dictionary containing the message details.

        Returns:
            str: The formatted message string.
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

    @staticmethod
    def _create_tool_name_chunk(function_name: str, kwargs: dict) -> dict:
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
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=function_name,
                                    arguments="",
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

    @staticmethod
    def _create_function_name_chunk(function_name: str, kwargs: dict) -> dict:
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

    @staticmethod
    def _create_tool_finish_chunk(kwargs: dict) -> dict:
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

    @staticmethod
    def _create_tool_argument_chunk(content: str, kwargs: dict) -> dict:
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

    @staticmethod
    def _create_function_argument_chunk(content: str, kwargs: dict) -> dict:
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

    @staticmethod
    def _create_first_chunk(kwargs: dict) -> dict:
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

    @staticmethod
    def _create_function_finish_chunk(kwargs: dict) -> dict:
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
