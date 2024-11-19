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
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from llmstudio_core.utils import OpenAIToolFunction
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import ValidationError


@provider
class VertexAIProvider(ProviderCore):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.API_KEY = self.API_KEY if self.API_KEY else os.getenv("GOOGLE_API_KEY")

    @staticmethod
    def _provider_config_name():
        return "vertexai"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""

        try:
            # Init genai
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.API_KEY,
            }

            tool_payload = self._process_tools(request.parameters)
            payload = self._create_request_payload(request.chat_input, tool_payload)

            return requests.post(url, headers=headers, json=payload, stream=True)

        except Exception as e:
            raise ProviderError(str(e))

    def parse_response(self, response: AsyncGenerator[Any, None], **kwargs) -> Any:

        for chunk in response.iter_content(chunk_size=None):

            chunk = json.loads(chunk.decode("utf-8").lstrip("data: "))
            chunk = chunk.get("candidates")[0].get("content")

            if not chunk:
                continue

            if (
                "functionCall" in chunk["parts"][0]
                and chunk["parts"][0]["functionCall"] is not None
            ):
                first_chunk = ChatCompletionChunk(
                    id="chatcmpl-9woLM1b1qGErhTbXA3UBQf2FhUAho",
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
                )
                yield first_chunk.model_dump()

                for index, functioncall in enumerate(chunk["parts"]):

                    name_chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    role="assistant",
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=index,
                                            id="call_" + str(uuid.uuid4())[:29],
                                            function=ChoiceDeltaToolCallFunction(
                                                name=functioncall["functionCall"].get(
                                                    "name"
                                                ),
                                                arguments="",
                                                type="function",
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                                index=index,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    )
                    yield name_chunk.model_dump()

                    args_chunk = ChatCompletionChunk(
                        id=str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=index,
                                            function=ChoiceDeltaToolCallFunction(
                                                arguments=json.dumps(
                                                    functioncall["functionCall"]["args"]
                                                ),
                                            ),
                                        )
                                    ],
                                ),
                                finish_reason=None,
                                index=index,
                            )
                        ],
                        created=int(time.time()),
                        model=kwargs.get("request").model,
                        object="chat.completion.chunk",
                    )
                    yield args_chunk.model_dump()

                final_chunk = ChatCompletionChunk(
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
                )
                yield final_chunk.model_dump()

            elif chunk.get("parts")[0].get("text"):

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

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = self.parse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def _create_request_payload(
        self, input_data: Union[Dict, str, List[Dict]], tool_payload: Optional[Any]
    ) -> Dict:

        if isinstance(input_data, str):
            return self._create_vertexai_payload(
                user_payload=input_data, tool_payload=tool_payload
            )

        elif isinstance(input_data, list):
            payload = self._create_vertexai_payload(tool_payload=tool_payload)
            for message in input_data:
                if message.get("role") == "system":
                    payload["system_instruction"]["parts"]["text"] = message["content"]

                if message.get("role") in ["user", "assistant"]:
                    if message.get("tool_calls"):
                        tool_call = message["tool_calls"][0]
                        payload["contents"].append(
                            {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": tool_call["function"]["name"],
                                            "args": json.loads(
                                                tool_call["function"]["arguments"]
                                            ),
                                        }
                                    }
                                ],
                            }
                        )
                    else:
                        payload["contents"].append(
                            {
                                "role": message.get("role"),
                                "parts": [{"text": message["content"]}],
                            }
                        )
                elif message.get("role") == "tool":
                    function_name = message["name"]
                    response = message["content"]
                    payload["system_instruction"]["parts"][
                        "text"
                    ] += f"\nYou have called {function_name} and got the following response: {response}."

            return payload

    @staticmethod
    def _create_vertexai_payload(
        user_payload: Optional[str] = None, tool_payload: Optional[Any] = None
    ) -> Dict:
        """
        Initializes the basic structure of a VertexAI payload.

        Args:
            user_payload (Optional[str]): The user's payload to include in the
            tool_payload (Optional[Any]): The tool payload for the

        Returns:
            Dict: The initialized VertexAI payload structure.
        """
        return {
            "system_instruction": {"parts": {"text": "You are a helpful assistant"}},
            "contents": [{"role": "user", "parts": [{"text": user_payload}]}]
            if user_payload
            else [],
            "tools": tool_payload,
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }

    @staticmethod
    def _process_tools(parameters: dict) -> dict:

        if parameters.get("tools") is None and parameters.get("functions") is None:
            return None
        try:
            if parameters.get("tools"):
                parsed_tools = [
                    OpenAIToolFunction(**tool["function"])
                    for tool in parameters["tools"]
                ]

            if parameters.get("functions"):
                parsed_tools = [
                    OpenAIToolFunction(**tool) for tool in parameters["functions"]
                ]

            function_declarations = []
            for tool in parsed_tools:
                function_declarations.append(tool.model_dump())
            return {"function_declarations": function_declarations}
        except ValidationError:
            return parameters.get("tools", parameters.get("functions"))
