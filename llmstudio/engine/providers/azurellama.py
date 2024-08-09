import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider
import json
import uuid
import time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


class AzureLlamaParameters(BaseModel):  # Renamed from AzureParameters
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class AzureLlamaRequest(ChatRequest):  # Renamed from AzureRequest
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Optional[AzureLlamaParameters] = AzureLlamaParameters()  # Updated reference
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class AzureLlamaProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("LLAMA_API_KEY")
        self.BASE_URL = os.getenv("LLAMA_BASE_URL")
        self.has_functions = False
        self.has_tools = False
        self.functions_called = False
        self.tools_called = False

    def validate_request(self, request: AzureLlamaRequest):  # Updated reference
        return AzureLlamaRequest(**request)  # Updated reference

    async def generate_client(
        self, request: AzureLlamaRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""

        print(f'request.chat_input: {request.chat_input}')
        print(f'request.tools: {request.tools}')
        print(f'request.functions: {request.functions}')

        # Set functions and tools indicators
        if request.functions:
            self.has_functions = True
        elif request.tools:
            self.has_tools = True

        try:
            # 1. Initialize Client
            client = OpenAI(
                api_key=request.api_key or self.API_KEY,
                base_url=request.base_url or self.BASE_URL,
            )
            
            # 2. Transoform prompt into openai format if not already.
            message = self.ensure_openai_format(request.chat_input)
            
            # 3. Add functions to the conversation, if functions are provided.
            if self.has_functions:
                message = self.create_conversation_with_function_prompt(message, request.functions)
            elif self.has_tools:
                message = self.create_conversation_with_tool_prompt(message, request.tools)

            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=message,
                stream=True,
            )

        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    def ensure_openai_format(self, chat_input):
        # Check if the input is already in the expected list of dictionaries format
        if isinstance(chat_input, list):
            # Check if each item in the list is a dictionary with 'role' and 'content' keys
            if all(isinstance(item, dict) and 'role' in item and 'content' in item for item in chat_input):
                return chat_input
            else:
                # If it's a list but doesn't have the proper structure, consider it invalid
                raise ValueError("Each item in the list must be a dictionary with 'role' and 'content' keys.")
        elif isinstance(chat_input, str):
            # Convert string to the OpenAI chat format
            return [{"role": "user", "content": chat_input}]
        else:
            # If it's neither a list nor a string, it's not a supported format
            raise TypeError("Input must be either a string or a list of dictionaries in the expected format.")
    
    def create_conversation_with_tool_prompt(self, message, tools):
        # Constructing the tool prompt
        tool_prompt = "You have access to the following tools:\n\n"

        for tool in tools:
            if tool['type'] == 'function':
                func = tool['function']
                tool_prompt += f"Use the function '{func['name']}' to '{func['description']}':\n"
                # Formatting the parameters to show the user how to use them
                params_info = json.dumps(func['parameters'], indent=4)
                tool_prompt += f"Parameters format:\n{params_info}\n\n"

        tool_prompt += """
        If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
        §function_name§{{"param_name": "param_value"}}

        Reminder:
        - Function calls MUST follow the specified format.
        - Only call one function at a time
        - Required parameters MUST be specified.
        - Put the entire function call reply on one line.
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
        """

        new_message = self.inject_tool_prompt(tool_prompt, message)
        return new_message

    def create_conversation_with_function_prompt(self, message, functions):
        # Constructing the tool prompt
        tool_prompt = "You have access to the following functions:\n\n"

        for function in functions:
            tool_prompt += f"Use the function '{function['name']}' to '{function['description']}':\n"
            tool_prompt += f"{json.dumps(function)}\n\n"

        tool_prompt += """
        If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
        §example_function_name§{{\"example_name\": \"example_value\"}}

        Reminder:
        - Function calls MUST follow the specified format.
        - Only call one function at a time
        - Required parameters MUST be specified.
        - Put the entire function call reply on one line.
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
        """

        new_message = self.inject_tool_prompt(tool_prompt, message)
        return new_message

    def inject_tool_prompt(self,tool_prompt, message):

        # Creating the new message list starting with the system prompt if it exists
        new_message = []
        if message and message[0]['role'] == 'system':
            new_message.append(message[0])  # Add the system prompt if present

        # Adding the tool prompt
        new_message.append({"role": "user", "content": tool_prompt})
        # Adding the assistant's initial reply
        new_message.append({"role": "assistant", "content": "Ok."})

        # Adding the rest of the conversation after the system prompt and the assistant's initial reply
        if message and message[0]['role'] == 'system':
            new_message.extend(message[1:])
        else:
            new_message.extend(message)

        return new_message

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        # ... existing code before the if self.has_functions ...

        if self.has_functions:
            print('has_functions')
            async for chunk in self.handle_function_response(response, **kwargs):
                yield chunk
        elif self.has_tools:
            print('has_tools')
            async for chunk in self.handle_tool_response(response, **kwargs):
                yield chunk
        else:
            for chunk in response:
                yield chunk.model_dump()

    async def handle_function_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        saving = False
        function_name = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                if '§' in content:
                    if saving:
                        # End of function call, stop saving
                        saving = False
                        function_name += content.split('§')[0]
                        print(f'function_name: {function_name}')

                        # First chunk
                        chunk = ChatCompletionChunk(
                            id=str(uuid.uuid4()),
                            choices=[
                                Choice(
                                    delta=ChoiceDelta(
                                        role="assistant",
                                        function_call=ChoiceDeltaFunctionCall(
                                            name=function_name, arguments=""
                                        ),
                                    ),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=int(time.time()),
                            model=kwargs.get("request").model,
                            object="chat.completion.chunk",
                        ).model_dump()
                        print(chunk)
                        yield chunk

                        function_name = ""
                    else:
                        # Start of function call, start saving
                        saving = True
                        function_name += content.split('§')[1]
                elif saving:
                    function_name += content
                # Check if finish_reason is 'stop'
                elif chunk.choices[0].finish_reason == 'stop':
                    chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(), finish_reason="function_call", index=0
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump()
                    print(chunk)
                    yield chunk
                else:
                    chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    role=None,
                                    function_call=ChoiceDeltaFunctionCall(
                                        arguments=content, name=None
                                    ),
                                ),
                                finish_reason=None,
                                index=0,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump()
                    print(chunk)
                    yield chunk

    async def handle_tool_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        print('handle tool response')
        saving = False
        function_name = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content is not None:
                if '§' in content:
                    if saving:
                        
                        # End of function call, stop saving
                        saving = False
                        function_name += content.split('§')[0]
                        print(f'function_name: {function_name}')
                        # First chunk
                        chunk = ChatCompletionChunk(
                            id=str(uuid.uuid4()),
                            choices=[
                                Choice(
                                    delta=ChoiceDelta(
                                        role="assistant",
                                        tool_calls=[ChoiceDeltaToolCall(
                                            index=0, 
                                            id='call_' + 'testid1234',
                                            function=ChoiceDeltaToolCallFunction(
                                                name=function_name,
                                                arguments="",
                                                type="function"
                                            )
                                        )],        
                                    ),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=int(time.time()),
                            model=kwargs.get("request").model,
                            object="chat.completion.chunk",
                        ).model_dump()
                        print(chunk)
                        yield chunk

                        function_name = ""
                    else:
                        # Start of function call, start saving
                        print(f'Is saving')
                        saving = True
                        function_name += content.split('§')[1]
                elif saving:
                    function_name += content
                # Check if finish_reason is 'stop'
                elif chunk.choices[0].finish_reason == 'stop':
                    chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(), finish_reason="tool_calls", index=0
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    ).model_dump()
                    print(chunk)
                    yield chunk
                else:
                    
                    chunk = ChatCompletionChunk(
                            id=str(uuid.uuid4()),
                            choices=[
                                Choice(
                                    delta=ChoiceDelta(
                                        tool_calls=[ChoiceDeltaToolCall(
                                            index=0, 
                                            function=ChoiceDeltaToolCallFunction(
                                                arguments=content,
                                            )
                                        )],        
                                    ),
                                    finish_reason=None,
                                    index=0,
                                )
                            ],
                            created=int(time.time()),
                            model=kwargs.get("request").model,
                            object="chat.completion.chunk",
                        ).model_dump()
                    print(chunk)
                    yield chunk