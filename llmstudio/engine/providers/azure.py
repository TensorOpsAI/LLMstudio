import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from fastapi import HTTPException
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
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

    def validate_request(self, request: AzureRequest):
        return AzureRequest(**request)

    async def generate_client(
        self, request: AzureRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""

        self.is_llama = "llama" in request.model.lower()
        self.has_tools = request.tools is not None

        try:
            if request.base_url or self.BASE_URL:
                client = OpenAI(
                    api_key=request.api_key or self.API_KEY,
                    base_url=request.base_url or self.BASE_URL,
                )
            else:
                client = AzureOpenAI(
                    api_key=request.api_key or self.API_KEY,
                    azure_endpoint=request.api_endpoint or self.API_ENDPOINT,
                    api_version=request.api_version or self.API_VERSION,
                )

            messages = self.prepare_messages(request)

            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=messages,
                stream=True,
                **(
                    {}
                    if self.is_llama
                    else {
                        "tools": request.tools,
                        "tool_choice": "auto" if request.tools else None,
                        "response_format": request.response_format,
                    }
                ),
                **request.parameters.model_dump(),
            )

        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    def prepare_messages(self, request: AzureRequest):
        if self.is_llama:
            user_message = self.convert_to_openai_format(request.chat_input)
            content = "<|begin_of_text|>"
            content = self.add_system_message(user_message, content, request.tools)
            content = self.add_conversation(user_message, content)
            return [{"role": "user", "content": content}]
        else:
            return (
                [{"role": "user", "content": request.chat_input}]
                if isinstance(request.chat_input, str)
                else request.chat_input
            )

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        if self.is_llama and self.has_tools:
            async for chunk in self.handle_tool_response(response, **kwargs):
                yield chunk
        else:
            for chunk in response:
                yield chunk.model_dump()

    async def handle_tool_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        saving = False
        function_name = ""
        function_response = False
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                if "§" in content:
                    if saving:
                        # End of function call, stop saving
                        saving = False
                        function_response = True
                        function_name += content.split("§")[0]
                        first_chunk = self.create_tool_first_chunk(kwargs)
                        yield first_chunk
                        name_chunk = self.create_tool_call_chunk(function_name, kwargs)
                        yield name_chunk
                        function_name = ""
                    else:
                        # Start of function call, start saving
                        saving = True
                        function_name += content.split("§")[1]
                elif saving:
                    function_name += content
                elif function_response:
                    if chunk.choices[0].finish_reason == "stop":
                        chunk = self.create_tool_finish_chunk(kwargs)
                        yield chunk
                    else:
                        chunk = self.create_tool_argument_chunk(content, kwargs)
                        yield chunk
                else:
                    yield chunk.model_dump()

    def create_tool_call_chunk(self, function_name, kwargs):
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

    def create_tool_finish_chunk(self, kwargs):
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

    def create_tool_argument_chunk(self, content, kwargs):
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

    def create_tool_first_chunk(self, kwargs):
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

    def convert_to_openai_format(self, message):
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        return message

    def add_system_message(self, openai_message, llama_message, tools):
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

        end_tag = "\n<|eot_id|>"
        return llama_message + system_message + end_tag

    def add_tool_instructions(self, tools):
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
    If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
    §function_name§{{"param_name": "param_value"}}

    Reminder:
    - Function calls MUST follow the specified format.
    - Only call one function at a time.
    - NEVER call more than one function at a time.
    - Required parameters MUST be specified.
    - Put the entire function call reply on one line.
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
    - If you have already called a tool and got the response for the users question please reply with the response.
    """

        return tool_prompt

    def add_conversation(self, openai_message, llama_message):
        conversation_parts = []
        for message in openai_message:
            if message["role"] == "system":
                continue
            elif "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    conversation_parts.append(
                        f"""
            <|start_header_id|>assistant<|end_header_id|>
            <function={function_name}>{arguments}</function>
            <|eom_id|>
            """
                    )
            elif "tool_call_id" in message:
                tool_response = message["content"]
                conversation_parts.append(
                    f"""
          <|start_header_id|>ipython<|end_header_id|>
          {tool_response}
          <|eot_id|>
          """
                )
            elif (
                message["role"] in ["assistant", "user"]
                and message["content"] is not None
            ):
                conversation_parts.append(
                    f"""
          <|start_header_id|>{message['role']}<|end_header_id|>
          {message['content']}
          <|eot_id|>
          """
                )

        return llama_message + "".join(conversation_parts)
