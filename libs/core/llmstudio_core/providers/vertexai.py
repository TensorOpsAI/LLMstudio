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
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, ValidationError


class OpenAIToolParameter(BaseModel):
    type: str
    description: Optional[str] = None


class OpenAIToolParameters(BaseModel):
    type: str
    properties: Dict[str, OpenAIToolParameter]
    required: List[str]


class OpenAIToolFunction(BaseModel):
    name: str
    description: str
    parameters: OpenAIToolParameters


class OpenAITool(BaseModel):
    type: str
    function: OpenAIToolFunction


class VertexAIToolParameter(BaseModel):
    type: str
    description: str


class VertexAIToolParameters(BaseModel):
    type: str
    properties: Dict[str, VertexAIToolParameter]
    required: List[str]


class VertexAIFunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: VertexAIToolParameters


class VertexAI(BaseModel):
    function_declarations: List[VertexAIFunctionDeclaration]


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

        try:
            # Init genai
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.API_KEY,
            }

            # Convert the chat input into VertexAI format
            tool_payload = self.process_tools(request.parameters.get("tools"))
            message = self.convert_input_to_vertexai(request.chat_input, tool_payload)

            # Generate content
            return await asyncio.to_thread(
                requests.post, url, headers=headers, json=message, stream=True
            )

        except Exception as e:
            raise ProviderError(str(e))

    def generate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Generator]:
        """Initialize Vertex AI"""

        try:
            # Init genai
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:streamGenerateContent?alt=sse"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.API_KEY,
            }

            # Convert the chat input into VertexAI format
            tool_payload = self.process_tools(request.parameters.get("tools"))
            message = self.convert_input_to_vertexai(request.chat_input, tool_payload)

            # Generate content
            return requests.post(url, headers=headers, json=message, stream=True)

        except Exception as e:
            raise ProviderError(str(e))

    def parse_response(self, response: AsyncGenerator[Any, None], **kwargs) -> Any:

        for chunk in response.iter_content(chunk_size=None):

            chunk = json.loads(chunk.decode("utf-8").lstrip("data: "))
            chunk = chunk.get("candidates")[0].get("content")

            if not chunk:
                continue

            # Check if it is a function call
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
                                                name=functioncall["functionCall"][
                                                    "name"
                                                ],
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

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        for chunk in response.iter_content(chunk_size=None):

            chunk = json.loads(chunk.decode("utf-8").lstrip("data: "))
            chunk = chunk.get("candidates")[0].get("content")

            # Check if it is a function call
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
                                                name=functioncall["functionCall"][
                                                    "name"
                                                ],
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

    def convert_input_to_vertexai(
        self, input_data: Union[Dict, str, List[Dict]], tool_payload: Optional[Any]
    ) -> Dict:
        """
        Converts OpenAI formatted input to VertexAI format.

        Args:
            input_data (Union[Dict, str, List[Dict]]): The input data in OpenAI format.
            tool_payload (Optional[Any]): The tool payload for the request.

        Returns:
            Dict: The converted input in VertexAI format.
        """
        if isinstance(input_data, dict) and "input" in input_data:
            return self._handle_simple_string_input(input_data["input"], tool_payload)

        if isinstance(input_data, str):
            return self._handle_simple_string_input(input_data, tool_payload)

        if isinstance(input_data, list):
            return self._convert_list_input_to_vertexai(input_data, tool_payload)

        raise ValueError("Invalid input type. Expected dict, str, or list.")

    def _handle_simple_string_input(
        self, input_data: str, tool_payload: Optional[Any]
    ) -> Dict:
        """
        Handles simple string input and converts it to VertexAI format.

        Args:
            input_data (str): The input data as a simple string.
            tool_payload (Optional[Any]): The tool payload for the request.

        Returns:
            Dict: The converted input in VertexAI format.
        """
        return self._initialize_vertexai_message(
            user_message=input_data, tool_payload=tool_payload
        )

    def _convert_list_input_to_vertexai(
        self, input_data: List[Dict], tool_payload: Optional[Any]
    ) -> Dict:
        """
        Converts a list of messages in OpenAI format to VertexAI format.

        Args:
            input_data (List[Dict]): The input data as a list of messages.
            tool_payload (Optional[Any]): The tool payload for the request.

        Returns:
            Dict: The converted input in VertexAI format.
        """
        vertexai_format = self._initialize_vertexai_message(tool_payload=tool_payload)
        for message in input_data:
            role = message.get("role")
            if role == "system":
                vertexai_format["system_instruction"]["parts"]["text"] = (
                    message["content"] or "You are a helpful assistant"
                )
            elif role == "user":
                vertexai_format["contents"].append(
                    {"role": "user", "parts": [{"text": message["content"]}]}
                )
            elif role == "assistant":
                if message["content"] is None and "tool_calls" in message:
                    tool_call = message["tool_calls"][0]
                    vertexai_format["contents"].append(
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
                    vertexai_format["contents"].append(
                        {"role": "model", "parts": [{"text": message["content"]}]}
                    )
            elif role == "tool":
                function_name = message["name"]
                response = message["content"]
                vertexai_format["system_instruction"]["parts"][
                    "text"
                ] += f"\nYou have called {function_name} and got the following response: {response}."
            else:
                raise ValueError(
                    f"Invalid role: {role}. Expected 'system', 'user', 'assistant', or 'tool'."
                )
        return vertexai_format

    def _initialize_vertexai_message(
        self, user_message: Optional[str] = None, tool_payload: Optional[Any] = None
    ) -> Dict:
        """
        Initializes the basic structure of a VertexAI message.

        Args:
            user_message (Optional[str]): The user's message to include in the request.
            tool_payload (Optional[Any]): The tool payload for the request.

        Returns:
            Dict: The initialized VertexAI message structure.
        """
        message_format = {
            "system_instruction": {"parts": {"text": "You are a helpful assistant"}},
            "contents": [],
            "tools": tool_payload,
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
        }
        if user_message:
            message_format["contents"].append(
                {"role": "user", "parts": [{"text": user_message}]}
            )
        return message_format

    def process_tools(
        self, tools: Optional[Union[List[Dict], Dict]]
    ) -> Optional[VertexAI]:
        if tools is None:
            return None

        try:
            # Try to parse as OpenAI format
            parsed_tools = (
                [OpenAITool(**tool) for tool in tools]
                if isinstance(tools, list)
                else [OpenAITool(**tools)]
            )
            # Convert to VertexAI format
            function_declarations = []
            for tool in parsed_tools:
                function = tool.function
                properties = {
                    name: VertexAIToolParameter(
                        type=param.type, description=param.description or ""
                    )
                    for name, param in function.parameters.properties.items()
                }
                function_decl = VertexAIFunctionDeclaration(
                    name=function.name,
                    description=function.description,
                    parameters=VertexAIToolParameters(
                        type=function.parameters.type,
                        properties=properties,
                        required=function.parameters.required,
                    ),
                )
                function_declarations.append(function_decl)
            return VertexAI(function_declarations=function_declarations).model_dump()
        except ValidationError:
            # If the format is not OpenAI, attempt to validate as VertexAI format
            try:
                return VertexAI(**tools).model_dump()
            except ValidationError:
                # If it fails to validate as VertexAI, throw an error
                raise ValueError(
                    "Invalid tool format. Tool data must be in OpenAI or VertexAI format."
                )
