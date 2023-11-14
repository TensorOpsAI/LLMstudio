import asyncio
import random
import time
from typing import Optional

import tiktoken
import vertexai
from fastapi.responses import StreamingResponse
from google.oauth2 import service_account
from pydantic import BaseModel, Field
from vertexai.language_models import (
    ChatModel,
    CodeChatModel,
    CodeGenerationModel,
    TextGenerationModel,
)

from llmstudio.engine.config import VertexAIConfig
from llmstudio.engine.constants import END_TOKEN, VERTEXAI_TOKEN_PRICE
from llmstudio.engine.providers.base_provider import BaseProvider
from llmstudio.engine.utils import validate_provider_config

VERTEXAI_MODEL_MAP = {
    "text-bison": TextGenerationModel,
    "chat-bison": ChatModel,
    "code-bison": CodeGenerationModel,
    "codechat-bison": CodeChatModel,
}
"""
VERTEXAI_MODEL_MAP: A dictionary that maps model names to their respective classes.
    Keys:
        - 'text-bison': Refers to TextGenerationModel
        - 'chat-bison': Refers to ChatModel
        - 'code-bison': Refers to CodeGenerationModel
        - 'codechat-bison': Refers to CodeChatModel

    Values:
        - TextGenerationModel: A class for text generation models
        - ChatModel: A class for chat models
        - CodeGenerationModel: A class for code generation models
        - CodeChatModel: A class for models that handle code and chat
"""

VERTEXAI_INPUT_MAP = {
    TextGenerationModel: "prompt",
    CodeGenerationModel: "prefix",
    ChatModel: "message",
    CodeChatModel: "message",
}
"""
VERTEXAI_INPUT_MAP: A dictionary that maps model classes to their input type.
    Keys:
        - TextGenerationModel: The class for text generation models
        - CodeGenerationModel: The class for code generation models
        - ChatModel: The class for chat models
        - CodeChatModel: The class for models that handle code and chat

    Values:
        - 'prompt': The input type for TextGenerationModel
        - 'prefix': The input type for CodeGenerationModel
        - 'message': The input type for ChatModel and CodeChatModel
"""


class VertexAIParameters(BaseModel):
    """
    A Pydantic model that encapsulates parameters used for VertexAI API requests.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        top_k (Optional[float]): Sets the number of the most likely next tokens to filter for.
    """

    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(256, ge=1, le=1024)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[float] = Field(40, ge=1, le=40)


class VertexAIRequest(BaseModel):
    """
    A Pydantic model that represents a request to the Vertex AI API.

    Attributes:
        api_key (Optional[dict]): API key for authentication, if required.
        model (str): The name of the machine learning model to be used.
        chat_input (str): The input string for the chat interaction.
        parameters (Optional[VertexAIParameters]): Additional parameters for Vertex AI, defaults to an empty VertexAIParameters object.
        is_stream (Optional[bool]): Whether the request should be streamed, defaults to False.
    """

    api_key: Optional[dict]
    model: str
    chat_input: str
    parameters: Optional[VertexAIParameters] = VertexAIParameters()
    is_stream: Optional[bool] = False


class VertexAITest(BaseModel):
    """
    A Pydantic model that represents a test request to the Vertex AI API.

    Attributes:
        api_key (Optional[dict]): API key for authentication, if required.
    """

    api_key: Optional[dict]


class VertexAIProvider(BaseProvider):
    """
    A provider class to interact with VertexAI API for chat and text generation.

    Attributes:
        vertexai_config (dict): Configurations for VertexAI API.
    """

    def __init__(self, config: VertexAIConfig, api_key: dict):
        """
        Initializes a new VertexAIProvider instance.

        Args:
            config (VertexAIConfig): The configuration settings for VertexAI.
            api_key (dict): The API key for VertexAI.
        """
        super().__init__()
        self.vertexai_config = validate_provider_config(config, api_key)

    async def chat(self, data: VertexAIRequest) -> dict:
        loop = asyncio.get_event_loop()
        """
        FastAPI endpoint to interact with the VertexAI API for text generation or chat completions.
        Args:
            data (VertexAIRequest): Object containing necessary parameters for the API call.
        Returns:
            dict: A dictionary containing the chat input, chat output, tokens data, cost, and other metadata.
        """
        credentials = service_account.Credentials.from_service_account_info(
            self.vertexai_config["api_key"]
        )
        vertexai.init(
            project=self.vertexai_config["api_key"]["project_id"],
            credentials=credentials,
        )

        data = VertexAIRequest(**data)

        self.validate_model_field(data, VERTEXAI_MODEL_MAP.keys())
        model_class = VERTEXAI_MODEL_MAP.get(data.model)
        input_arg_name = VERTEXAI_INPUT_MAP.get(model_class)

        kwargs = {
            "temperature": data.parameters.temperature,
            "max_output_tokens": data.parameters.max_tokens,
        }
        if data.model not in {"code-bison", "codechat-bison"}:
            kwargs.update(
                {
                    "top_p": data.parameters.top_p,
                    "top_k": data.parameters.top_k,
                }
            )

        if model_class in {TextGenerationModel, CodeGenerationModel}:
            model = model_class.from_pretrained(data.model)
            response = await self.predict(
                model, input_arg_name, data.chat_input, data.is_stream, loop, **kwargs
            )
        else:
            model = model_class.from_pretrained(data.model)
            response = await self.chat_predict(
                model, input_arg_name, data.chat_input, data.is_stream, loop, **kwargs
            )

        if data.is_stream:
            return StreamingResponse(generate_stream_response(response, data))

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data.chat_input,
            "chatOutput": response.text,
            "inputTokens": len(data.chat_input),
            "outputTokens": len(response.text),
            "totalTokens": len(data.chat_input) + len(response.text),
            "cost": get_cost(len(data.chat_input), len(response.text)),
            "timestamp": time.time(),
            "model": data.model,
            "parameters": dict(data.parameters),
        }

        return data

    async def test(self, data: VertexAITest) -> bool:
        """
        Test the validity of the Vertex AI API key.
        Args:
            data (VertexAITest): A model instance which includes the API key for Vertex AI.
        Returns:
            bool: `True` if the API key is valid and initialization succeeds, otherwise `False`.
        """
        data = VertexAITest(**data)
        try:
            credentials = service_account.Credentials.from_service_account_info(
                self.vertexai_config["api_key"]
            )
            vertexai.init(
                project=self.vertexai_config["api_key"]["project_id"],
                credentials=credentials,
            )
            return True
        except Exception:
            return False

    async def predict(
        self, model, input_str: str, chat_input: str, is_stream: bool, loop, **kwargs
    ):
        """
        Makes a prediction using the specified model.
        Args:
            model: The model to use for making predictions.
            arg_name (str): The name of the argument to pass the chat input as.
            chat_input (str): The input string for the chat.
            is_stream (bool): Whether to stream the response.
            **kwargs: Additional parameters for the prediction function.
        Returns:
            The prediction response.
        """
        args = {input_str: chat_input, **kwargs}
        if is_stream:
            return await loop.run_in_executor(
                self.executor, lambda: model.predict_streaming(**args)
            )
        return await loop.run_in_executor(self.executor, lambda: model.predict(**args))

    async def chat_predict(
        self, model, input_str: str, chat_input: str, is_stream: bool, loop, **kwargs
    ):
        """
        Makes a prediction using the specified chat model.
        Args:
            model: The chat model to use for making predictions.
            arg_name (str): The name of the argument to pass the chat input as.
            chat_input (str): The input string for the chat.
            is_stream (bool): Whether to stream the response.
            **kwargs: Additional parameters for the chat prediction function.
        Returns:
            The chat prediction response.
        """
        args = {input_str: chat_input, **kwargs}
        chat = model.start_chat()
        if is_stream:
            return await loop.run_in_executor(
                self.executor, lambda: chat.send_message_streaming(**args)
            )
        return await loop.run_in_executor(self.executor, lambda: chat.send_message(**args))


def get_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of using the OpenAI API based on token usage and model.

    Args:
        input_tokens (int): Number of tokens in the input.
        output_tokens (int): Number of tokens in the output.
        model (str): Identifier of the model used.

    Returns:
        float: The calculated cost for the API usage.
    """
    return VERTEXAI_TOKEN_PRICE * (input_tokens + output_tokens)


def get_tokens(chat_input: str, model: str) -> int:
    """
    Determine the number of tokens in a given input string using the specified model’s tokenizer.

    Args:
        chat_input (str): Text to be tokenized.
        model (str): Identifier of the model, determines tokenizer used.

    Returns:
        int: Number of tokens in the input string.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(chat_input))


def generate_stream_response(response: dict, data: VertexAIProvider):
    """
    Generate stream responses, yielding chat output or tokens and cost information at stream end.

    Args:
        response (dict): Dictionary containing chunks of responses from the Vertex AI API.
        data (VertexAIRequest): Object containing necessary parameters for the API call.

    Yields:
        str: A chunk of chat output or, at stream end, tokens counts and cost information.
    """
    chat_output = ""

    for chunk in response:
        print(chunk)
        print(type(chunk))
        chat_output += ""
        yield ""

    input_tokens = len(data.chat_input)
    output_tokens = len(chat_output)
    cost = get_cost(input_tokens, output_tokens)
    yield f"{END_TOKEN},{input_tokens},{output_tokens},{cost}"  # json
