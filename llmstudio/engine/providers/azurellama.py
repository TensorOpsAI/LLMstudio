import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider
import json


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
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class AzureLlamaProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("LLAMA_API_KEY")
        self.BASE_URL = os.getenv("LLAMA_BASE_URL")

    def validate_request(self, request: AzureLlamaRequest):  # Updated reference
        return AzureLlamaRequest(**request)  # Updated reference

    async def generate_client(
        self, request: AzureLlamaRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""
        try:
            # 1. Initialize Client
            client = OpenAI(
                api_key=request.api_key or self.API_KEY,
                base_url=request.base_url or self.BASE_URL,
            )
            
            # 2. Transoform prompt into openai format if not already.
            message = self.ensure_openai_format(request.chat_input)
            
            # 3. Add functions to the conversation, if functions are provided.
            if request.functions:
                message = self.create_conversation_with_tool_prompt(message, request.functions)

            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=message,
                stream=True,
            )

        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            yield chunk.model_dump()

    def create_functioncall_prompt(functions):
        toolPrompt = "You have access to the following functions:\n\n"

        for function in functions:
            toolPrompt += f"Use the function '{function['name']}' to '{function['description']}':\n"
            toolPrompt += f"{json.dumps(function)}\n\n"

        toolPrompt += """
        If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
        §example_function_name§{{\"example_name\": \"example_value\"}}

        Reminder:
        - Function calls MUST follow the specified format.
        - Required parameters MUST be specified.
        - Only call one function at a time.
        - Put the entire function call reply on one line.
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
        """

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

    def create_conversation_with_tool_prompt(self, message, functions):
        # Constructing the tool prompt
        tool_prompt = "You have access to the following functions:\n\n"
        for function in functions:
            tool_prompt += f"Use the function '{function['name']}' to '{function['description']}':\n"
            tool_prompt += f"{json.dumps(function)}\n\n"

        tool_prompt += """
        If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
        <function=example_function_name>{{"example_name": "example_value"}}</function>

        Reminder:
        - Function calls MUST follow the specified format.
        - Required parameters MUST be specified.
        - Only call one function at a time.
        - Put the entire function call reply on one line.
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
        """

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


