import asyncio
import random
import time
from typing import Optional

import openai
import tiktoken
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.config import OpenAIConfig
from llmstudio.engine.constants import (
    DEFAULT_OUTPUT_MARGIN,
    END_TOKEN,
    GPT_4_MAX_TOKENS,
    GPT_35_MAX_TOKENS,
    GPT_35_TURBO_16K,
    OPENAI_MAX_RETRIES,
    OPENAI_PRICING_DICT,
)
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
        model (str): The name of the language model to query.
        chat_input (str): The input text to send to the model.
        parameters (Optional[OpenAIParameters]): An optional instance of OpenAIParameters to further configure the request.
        is_stream (Optional[bool]): Indicates if the request should be a streaming request; default is False.

    """

    api_key: Optional[str]
    model: str
    chat_input: str
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False
    safety_margin: Optional[float] = float(DEFAULT_OUTPUT_MARGIN)
    end_token: Optional[bool] = True
    custom_max_tokens: Optional[int] = None


class OpenAITest(BaseModel):
    """
    A Pydantic model for validating OpenAI API requests.

    Attributes:
        api_key (str): The API key provided by the user authentication with OpenAI API.
        model (str): The name of the model to be used for generating text

    Methods:
        validate_model: Ensures that `model` is one of the allowed values.
    ```
    """

    api_key: Optional[str]
    model: str


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

        loop = asyncio.get_event_loop()

        start_time = time.time()
        response = await self.execute_openai_api_call(loop, data, OPENAI_MAX_RETRIES)
        duration = time.time() - start_time

        if data.is_stream:
            return StreamingResponse(generate_stream_response(response, data))

        return await format_response(response, data, duration)

    async def test(self, data: OpenAITest) -> bool:
        """
        Test the validity of the OpenAI API key.

        Args:
            data (OpenAITest): A model instance which includes the API key for OpenAI.

        Returns:
            bool: `True` if the API key is valid and initialization succeeds, otherwise `False`.
        """
        client = OpenAI(api_key=self.openai_config.api_key)
        data = OpenAITest(**data)
        try:
            self.validate_model_field(data, OPENAI_PRICING_DICT.keys())
            client.models.retrieve(data.model)
            return True
        except Exception:
            return False

    async def execute_openai_api_call(
        self, loop, request: OpenAIRequest, max_retries: int
    ) -> dict:
        """
        Execute an OpenAI API call asynchronously, with retry logic and model selection.

        Parameters:
        - loop: The event loop where the function should be executed.
        - request (OpenAIRequest): The object containing parameters for the OpenAI API call.
        - max_retries (int): The maximum number of retries for the API call.

        Returns:
        - dict: The response from the OpenAI API.

        Raises:
        - ValueError: If the maximum number of retries is reached and the API call is unsuccessful.

        Notes:
        - The function incorporates retry logic and may switch to a higher capacity model if an InvalidRequestError is encountered.
        """
        client = OpenAI(api_key=self.openai_config.api_key)
        retry_count = 0
        use_higher_capacity_model = False

        while retry_count < max_retries:
            try:
                model = _select_appropriate_model(
                    input_text=request.chat_input,
                    selected_model=request.model,
                    safety_margin=request.safety_margin,
                    custom_max_tokens=request.custom_max_tokens,
                    use_higher_capacity_model=use_higher_capacity_model,
                )
                return await loop.run_in_executor(
                    self.executor,
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": request.chat_input}],
                        temperature=request.parameters.temperature,
                        max_tokens=request.parameters.max_tokens,
                        top_p=request.parameters.top_p,
                        frequency_penalty=request.parameters.frequency_penalty,
                        presence_penalty=request.parameters.presence_penalty,
                        stream=request.is_stream,
                    ),
                )
            except openai.error.InvalidRequestError as e:
                retry_count += 1
                use_higher_capacity_model = True
        raise ValueError("Maximum retries reached, cannot generate output within token limits.")


async def format_response(response: dict, request: OpenAIRequest, duration: float) -> dict:
    """
    Format the OpenAI API response and include additional details.

    Parameters:
    - response (dict): The dictionary containing the raw response from the OpenAI API.
    - request (OpenAIRequest): The object containing parameters for the original OpenAI API call.

    Returns:
    - dict: A dictionary containing the formatted response along with additional details like token counts, cost, and timestamp.

    Notes:
    - The function calculates the number of tokens used in both the input and output and includes this information in the returned dictionary.
    """
    input_tokens = get_tokens(request.chat_input, request.model)
    output_tokens = get_tokens(response.choices[0].message.content, request.model)

    return {
        "id": random.randint(0, 1000),
        "chatInput": request.chat_input,
        "chatOutput": response.choices[0].message.content,
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
        "cost": get_cost(input_tokens, output_tokens, request.model),
        "timestamp": time.time(),
        "model": request.model,
        "parameters": request.parameters.dict(),
        "latency": duration,
    }


def get_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Calculate the cost of using the OpenAI API based on token usage and model.

    Args:
        input_tokens (int): Number of tokens in the input.
        output_tokens (int): Number of tokens in the output.
        model (str): Identifier of the model used.

    Returns:
        float: The calculated cost for the API usage.
    """
    return (
        OPENAI_PRICING_DICT[model]["input_tokens"] * input_tokens
        + OPENAI_PRICING_DICT[model]["output_tokens"] * output_tokens
    )


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
        if chunk.choices[0].finish_reason != "stop" and chunk.choices[0].finish_reason != "length":
            chunk_content = chunk.choices[0].delta.content
            chat_output += chunk_content
            yield chunk_content
        else:
            if data.end_token:
                input_tokens = get_tokens(data.chat_input, data.model)
                output_tokens = get_tokens(chat_output, data.model)
                cost = get_cost(input_tokens, output_tokens, data.model)
                yield f"{END_TOKEN},{input_tokens},{output_tokens},{cost}"  # json


def _select_appropriate_model(
    input_text,
    selected_model,
    safety_margin=DEFAULT_OUTPUT_MARGIN,
    custom_max_tokens=None,
    use_higher_capacity_model=False,
):
    """
    Selects the appropriate model based on token count and other parameters.

    Parameters:
    - input_text (str): The input text that needs to be processed by the model.
    - selected_model (str): The default model selected for text processing.
    - safety_margin (float, optional): A margin to reserve tokens for the output. Defaults to 0.2.
    - custom_max_tokens (int, optional): Custom maximum tokens, overrides model's maximum if provided.
    - use_higher_capacity_model (bool, optional): If True, uses a higher capacity model as a fallback. Defaults to False.

    Returns:
    - str: The chosen model based on the input parameters and token count.
    Notes:
    - The safety_margin is applied to reserve space for the output. For example, a safety_margin of 0.2 reserves 20% of the model's maximum tokens
    for the output and allows 80% to be used for the input.
    """
    if safety_margin is None:
        safety_margin = DEFAULT_OUTPUT_MARGIN
    elif safety_margin > 1:
        print(
            f"Error: safety_margin can not be greater than 1. Defaulting to {DEFAULT_OUTPUT_MARGIN}."
        )
        safety_margin = DEFAULT_OUTPUT_MARGIN
    encoder = tiktoken.encoding_for_model(selected_model)
    token_count = len(encoder.encode(input_text))

    high_capacity_model = GPT_35_TURBO_16K

    if use_higher_capacity_model:
        print(f"Warning: Tokens exceeded, using fallback model: {high_capacity_model}")
        return high_capacity_model

    if custom_max_tokens:
        effective_max_tokens = custom_max_tokens
    else:
        model_max_tokens = (
            GPT_35_MAX_TOKENS if "gpt-3.5-turbo" in selected_model else GPT_4_MAX_TOKENS
        )

        effective_max_tokens = int(model_max_tokens * (1 - safety_margin))

    chosen_model = high_capacity_model if token_count > effective_max_tokens else selected_model

    return chosen_model
