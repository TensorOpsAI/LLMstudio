import asyncio
import os
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import google.generativeai as genai
from fastapi import HTTPException
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta, ChoiceDeltaFunctionCall
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider

import re
import json

class VertexAIParameters(BaseModel):
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    top_k: Optional[float] = Field(default=1, ge=0, le=1)
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_output_tokens: Optional[float] = Field(default=8192, ge=0, le=8192)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class VertexAIRequest(ChatRequest):
    parameters: Optional[VertexAIParameters] = VertexAIParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    functions: Optional[Any] = None
    chat_input: Any


@provider
class VertexAIProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    def validate_request(self, request: VertexAIRequest):
        return VertexAIRequest(**request)

    async def generate_client(
        self, request: VertexAIRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""
        try:
            # Init genai
            genai.configure(api_key=request.api_key or self.GOOGLE_API_KEY)
            print(f'vertex.py - request: {request.chat_input}')

            # Define model
            if request.functions:
                model = genai.GenerativeModel(request.model, tools=request.functions)
            else:
                model = genai.GenerativeModel(request.model)
                model.generate_content

            # Generate content
            return await asyncio.to_thread(
                model.generate_content, request.chat_input, stream=True
            )

        except Exception as e:
            # Handle any other exceptions that might occur
            raise HTTPException(status_code=500, detail=str(e))

    def parse_string_to_dict(self, args):
        response_dict = {}
        for i in args:
                key = i
                value = args[i].ListFields()[0][1]
                
                # Check if the value is a number
                if isinstance(value, (int, float)):
                    # Convert to int if it has no decimal part, otherwise keep as float
                    value = int(value) if value == int(value) else float(value)
                response_dict[key] = value

        response_json = json.dumps(response_dict).replace(' ','')
        return response_json
    
    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response:
            
            # Check if it is a function call
            if chunk.parts[0].function_call:
                name = chunk.parts[0].__dict__['_pb'].function_call.name
                args = chunk.parts[0].__dict__['_pb'].function_call.args.fields
                openai_args = self.parse_string_to_dict(args)
                id = str(uuid.uuid4())

                # First chunk
                first_chunk = ChatCompletionChunk(
                    id=id,
                    choices=[
                        Choice(
                            delta=ChoiceDelta(role='assistant', 
                                              function_call=ChoiceDeltaFunctionCall(name=name,
                                                                                    arguments='')),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                )
                yield first_chunk.model_dump()

                # Midle chunk with arguments
                middle_chunk = ChatCompletionChunk(
                    id=id,
                    choices=[
                        Choice(
                            delta=ChoiceDelta(role=None,
                                            function_call=ChoiceDeltaFunctionCall(arguments=openai_args,
                                                                                    name=None)),
                            finish_reason=None,
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                )
                yield middle_chunk.model_dump()

                final_chunk = ChatCompletionChunk(
                    id=id,
                    choices=[Choice(delta=ChoiceDelta(), finish_reason="function_call", index=0)],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                )
                yield final_chunk.model_dump()

            # Check if it is a normal call
            elif chunk.parts[0].text:
                # Parse google chunk response into ChatCompletionChunk
                yield ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(content=chunk.text, role="assistant"),
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
                    choices=[Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
                    created=int(time.time()),
                    model=kwargs.get("request").model,
                    object="chat.completion.chunk",
                ).model_dump()
