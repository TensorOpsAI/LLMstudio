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

import requests
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
)
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider


class newVertexParameters(BaseModel):
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[float] = Field(default=1, ge=0, le=1)
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_output_tokens: Optional[float] = Field(default=8192, ge=0, le=8192)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class newVertexAIRequest(ChatRequest):
    parameters: Optional[newVertexParameters] = newVertexParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    chat_input: Union[str, List[Dict[str, Any]]]


@provider
class newVertexProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    def validate_request(self, request: newVertexAIRequest):
        return newVertexAIRequest(**request)

    async def generate_client(
        self, request: newVertexAIRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""
        try:
            # Init genai
            api_key = request.api_key or self.GOOGLE_API_KEY
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            # Convert the chat input into VertexAI format
            message = self.convert_openai_to_vertexai(request.chat_input)
            print(f'message: {message}')

            # Generate content
            return await asyncio.to_thread(
                requests.post, url, headers=headers, json=message, stream=True
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response.iter_content(chunk_size=None):
            chunk = json.loads(chunk.decode("utf-8").lstrip("data: "))
            chunk = chunk.get("candidates")[0].get("content")

            # Check if it is a function call
            if chunk.get("parts")[0].get("function_call"):
                name = chunk.get("parts")[0].__dict__["_pb"].function_call.name
                args = chunk.get("parts")[0].__dict__["_pb"].function_call.args.fields
                openai_args = self.parse_string_to_dict(args)
                str(uuid.uuid4())

                # First chunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                role="assistant",
                                function_call=ChoiceDeltaFunctionCall(
                                    name=name, arguments=""
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

                # Midle chunk with arguments
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                role=None,
                                function_call=ChoiceDeltaFunctionCall(
                                    arguments=openai_args, name=None
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

                yield ChatCompletionChunk(
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

            # Check if it is a normal call
            elif chunk.get("parts")[0].get("text"):
                # Parse google chunk response into ChatCompletionChunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                content=chunk.get("parts")[0].get("text"),
                                role="assistant",
                            ),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

                # Create the closing chunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()

    def convert_openai_to_vertexai(self, input_data):
        # Check if the input is a simple string
        if isinstance(input_data, str):
            # Return a Vertex AI formatted message with a user message
            return {
                "system_instruction": {
                    "parts": {
                        "text": ""  # Empty system instruction
                    }
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": input_data}]
                    }
                ]
            }
        
        # Validate if input_data is a list and each element is a dictionary with the correct structure
        if not isinstance(input_data, list) or not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in input_data):
            raise ValueError("Input must be a list of dictionaries, each containing 'role' and 'content' keys.")

        # Initialize the Vertex AI format if the input is not a simple string
        vertexai_format = {
            "system_instruction": {
                "parts": {
                    "text": ""
                }
            },
            "contents": []
        }
        
        # Loop through the OpenAI formatted messages
        for message in input_data:
            if message["role"] == "system":
                # Set the system instruction
                vertexai_format["system_instruction"]["parts"]["text"] = message["content"]
            elif message["role"] in ["user", "assistant"]:
                # Convert roles: 'assistant' -> 'model'
                role = "model" if message["role"] == "assistant" else "user"
                # Append the message to the contents list in Vertex AI format
                vertexai_format["contents"].append({
                    "role": role,
                    "parts": [{"text": message["content"]}]
                })
            else:
                raise ValueError(f"Invalid role: {message['role']}. Expected 'system', 'user', or 'assistant'.")
        
        return vertexai_format
