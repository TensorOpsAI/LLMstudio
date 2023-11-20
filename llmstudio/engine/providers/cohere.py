import asyncio
import os
import time
from typing import AsyncGenerator, Optional, Union

import cohere
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from llmstudio.engine.providers.provider import ChatRequest, Provider


class CommandParameters(BaseModel):
    temperature: Optional[float] = Field(0.75, ge=0, le=5)
    max_tokens: Optional[int] = Field(256, ge=1)
    p: Optional[float] = Field(0, ge=0, le=0.99)
    k: Optional[int] = Field(0, ge=0, le=500)
    frequency_penalty: Optional[float] = Field(0, ge=0)
    presence_penalty: Optional[float] = Field(0, ge=0, le=1)


class CohereRequest(ChatRequest):
    parameters: Optional[CommandParameters] = CommandParameters()


class CohereProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("COHERE_API_KEY")

    async def chat(
        self, request: CohereRequest
    ) -> Union[StreamingResponse, JSONResponse]:
        """Chat with the Cohere API"""
        try:
            request = CohereRequest(**request)
            await super().chat(request)
            co = cohere.Client(api_key=request.api_key or self.API_KEY)

            start_time = time.time()
            response = await asyncio.to_thread(
                co.generate,
                model=request.model,
                prompt=request.chat_input,
                stream=True,
                **request.parameters.model_dump(),
            )

            response_handler = self.handle_response(request, response, start_time)
            if request.is_stream:
                return StreamingResponse(response_handler)
            else:
                return JSONResponse(content=await response_handler.__anext__())

        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        except (cohere.CohereAPIError, cohere.CohereAPIError) as e:
            raise HTTPException(status_code=e.http_status, detail=str(e))

    async def handle_response(
        self, request: CohereRequest, response: AsyncGenerator, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Handles the response from the Cohere API"""
        chat_output = ""
        first_token_time = None
        previous_token_time = None
        token_times = []
        token_count = 0

        for chunk in response:
            if not chunk.is_finished:
                token_count += 1
                current_time = time.time()
                first_token_time = first_token_time or current_time
                if previous_token_time is not None:
                    token_times.append(current_time - previous_token_time)
                previous_token_time = current_time

                chat_output += chunk.text
                if request.is_stream:
                    yield chunk.text

        usage = self.calculate_usage(request.chat_input, chat_output, request.model)

        metrics = self.calculate_metrics(
            start_time, time.time(), first_token_time, token_times, token_count
        )

        if request.is_stream and request.has_end_token:
            yield self.get_end_token_string(usage, metrics)

        if not request.is_stream:
            yield self.generate_response(request, chat_output, usage, metrics)
