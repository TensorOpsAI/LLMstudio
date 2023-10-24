import asyncio
import random
import time
from typing import Optional

import openai
import tiktoken
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmstudio.engine.config import OpenAIConfig
from llmstudio.engine.constants import END_TOKEN, OPENAI_PRICING_DICT
from llmstudio.engine.providers.base_provider import BaseProvider
from llmstudio.engine.utils import validate_provider_config


class OpenAIParameters(BaseModel):
    """
    A Pydantic model for encapsulating parameters used in OpenAI API requests.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        frequency_penalty (Optional[float]): Modifies the likelihood of tokens appearing based on their frequency.
        presence_penalty (Optional[float]): Adjusts the likelihood of new tokens appearing.
    """

    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(BaseModel):
    """
    A Pydantic model that represents a request to an OpenAI API.

    Attributes:
        api_key (Optional[str]): The API key to use for authenticating the request.
        model_name (str): The name of the language model to query.
        chat_input (str): The input text to send to the model.
        parameters (Optional[OpenAIParameters]): An optional instance of OpenAIParameters to further configure the request.
        is_stream (Optional[bool]): Indicates if the request should be a streaming request; default is False.

    """

    api_key: Optional[str]
    model_name: str
    chat_input: str
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False


class OpenAITest(BaseModel):
    """
    A Pydantic model for validating OpenAI API requests.

    Attributes:
        api_key (str): The API key provided by the user authentication with OpenAI API.
        model_name (str): The name of the model to be used for generating text

    Methods:
        validate_model_name: Ensures that `model_name` is one of the allowed values.
    ```
    """

    api_key: Optional[str]
    model_name: str


class OpenAIProvider(BaseProvider):
    """
    A provider class to handle interactions with the OpenAI GPT models.

    Attributes:
        openai_config (OpenAIConfig): Configuration settings for OpenAI API.
    """

    def __init__(self, config: OpenAIConfig, api_key: str):
        """
        Initialize the OpenAIProvider with given config and API key.

        Args:
            config (OpenAIConfig): Configuration settings for OpenAI API.
            api_key (str): API key for authentication.
        """
        super().__init__()
        if isinstance(config, OpenAIConfig):
            self.openai_config = config
        else:
            self.openai_config = OpenAIConfig(**validate_provider_config(config, api_key))

    async def chat(self, data: OpenAIRequest) -> dict:
        """
        Generate chat-based model completions using OpenAI API.

        Args:
            data (OpenAIRequest): A model instance containing chat input, model name, and additional parameters.

        Returns:
            dict: A dictionary containing chat input, chat output, tokens information, cost, and other metadata.

        Raises:
            ValueError: If the specified model field is invalid.
        """

        data = OpenAIRequest(**data)

        self.validate_model_field(data, OPENAI_PRICING_DICT.keys())
        openai.api_key = self.openai_config.api_key

        # Asynchronous call, for parallelism
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            self.executor,
            lambda: openai.ChatCompletion.create(
                model=data.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": data.chat_input,
                    }
                ],
                temperature=data.parameters.temperature,
                max_tokens=data.parameters.max_tokens,
                top_p=data.parameters.top_p,
                frequency_penalty=data.parameters.frequency_penalty,
                presence_penalty=data.parameters.presence_penalty,
                stream=data.is_stream,
            ),
        )

        if data.is_stream:
            return StreamingResponse(generate_stream_response(response, data))

        input_tokens = get_tokens(data.chat_input, data.model_name)
        output_tokens = get_tokens(response["choices"][0]["message"]["content"], data.model_name)

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data.chat_input,
            "chatOutput": response["choices"][0]["message"]["content"],
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "cost": get_cost(input_tokens, output_tokens, data.model_name),
            "timestamp": time.time(),
            "modelName": data.model_name,
            "parameters": data.parameters.dict(),
        }
        return data

    async def test(self, data: OpenAITest) -> bool:
        """
        Test the validity of the OpenAI API key.

        Args:
            data (OpenAITest): A model instance which includes the API key for OpenAI.

        Returns:
            bool: `True` if the API key is valid and initialization succeeds, otherwise `False`.
        """
        openai.api_key = self.openai_config.api_key
        data = OpenAITest(**data)
        try:
            self.validate_model_field(data, OPENAI_PRICING_DICT.keys())
            openai.Model.retrieve(data.model_name)
            return True
        except Exception:
            return False


def get_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate the cost of using the OpenAI API based on token usage and model.

    Args:
        input_tokens (int): Number of tokens in the input.
        output_tokens (int): Number of tokens in the output.
        model_name (str): Identifier of the model used.

    Returns:
        float: The calculated cost for the API usage.
    """
    return (
        OPENAI_PRICING_DICT[model_name]["input_tokens"] * input_tokens
        + OPENAI_PRICING_DICT[model_name]["output_tokens"] * output_tokens
    )


def get_tokens(chat_input: str, model_name: str) -> int:
    """
    Determine the number of tokens in a given input string using the specified model’s tokenizer.

    Args:
        chat_input (str): Text to be tokenized.
        model_name (str): Identifier of the model, determines tokenizer used.

    Returns:
        int: Number of tokens in the input string.
    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    return len(tokenizer.encode(chat_input))


def generate_stream_response(response: dict, data: OpenAIProvider):
    """
    Generate stream responses, yielding chat output or tokens and cost information at stream end.

    Args:
        response (dict): Dictionary containing chunks of responses from the OpenAI API.
        data (OpenAIRequest): OpenAIRequest object containing necessary parameters for the API call.

    Yields:
        str: A chunk of chat output or, at stream end, tokens counts and cost information.
    """
    chat_output = ""
    for chunk in response:
        if (
            chunk["choices"][0]["finish_reason"] != "stop"
            and chunk["choices"][0]["finish_reason"] != "length"
        ):
            chunk_content = chunk["choices"][0]["delta"]["content"]
            chat_output += chunk_content
            yield chunk_content
        else:
            input_tokens = get_tokens(data.chat_input, data.model_name)
            output_tokens = get_tokens(chat_output, data.model_name)
            cost = get_cost(input_tokens, output_tokens, data.model_name)
            yield f"{END_TOKEN},{input_tokens},{output_tokens},{cost}"  # json
