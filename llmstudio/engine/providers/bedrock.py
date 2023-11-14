import asyncio
import json
import random
import time
from typing import Optional, Tuple

import boto3
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from llmstudio.engine.config import BedrockConfig
from llmstudio.engine.constants import BEDROCK_MODELS, CLAUDE_MODELS, END_TOKEN, TITAN_MODELS
from llmstudio.engine.providers.base_provider import BaseProvider
from llmstudio.engine.utils import validate_provider_config


class ClaudeParameters(BaseModel):
    """
    Model for validating and storing parameters specific to Claude model.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
        top_k (Optional[float]): Sets the number of the most likely next tokens to filter for.
    """

    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(300, ge=1, le=2048)
    top_p: Optional[float] = Field(0.999, ge=0, le=1)
    top_k: Optional[int] = Field(250, ge=1, le=500)


class TitanParameters(BaseModel):
    """
    Model for validating and storing parameters specific to Titan model.

    Attributes:
        temperature (Optional[float]): Controls randomness in the model's output.
        max_tokens (Optional[int]): The maximum number of tokens in the output.
        top_p (Optional[float]): Influences the diversity of output by controlling token sampling.
    """

    temperature: Optional[float] = Field(0, ge=0, le=1)
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    top_p: Optional[float] = Field(0.9, ge=0.1, le=1)


class BedrockRequest(BaseModel):
    """
    Represents a request to the Bedrock API.

    Attributes:
        api_key (Optional[str]): The API key for authenticating with the Bedrock API.
        api_secret (Optional[str]): The API secret for authenticating with the Bedrock API.
        api_region (Optional[str]): The region where the Bedrock API is hosted.
        model (str): The name of the model to be used for the request.
        chat_input (str): The input string for the chat.
        parameters (Optional[BaseModel]): Additional parameters for the model, encapsulated in a BaseModel.
        is_stream (Optional[bool]): Flag to indicate if the request is for streaming. Defaults to False.
    """

    api_key: Optional[str]
    api_secret: Optional[str]
    api_region: Optional[str]
    model: str
    chat_input: str
    parameters: Optional[BaseModel]
    is_stream: Optional[bool] = False

    @validator("parameters", pre=True, always=True)
    def validate_parameters_based_on_model(cls, parameters, values):
        """
        Validate and convert parameters based on the model.

        Args:
            parameters (Dict[str, Any]): Parameters to validate and convert.
            values (Dict[str, Any]): Contains previously validated fields.

        Returns:
            BaseModel: An instance of `TitanParameters` or `ClaudeParameters` based on `model`.

        Raises:
            ValueError if model is invalid.
        """
        model = values.get("model")
        if model in TITAN_MODELS:
            return TitanParameters(**parameters)
        if model in CLAUDE_MODELS:
            return ClaudeParameters(**parameters)

        raise ValueError(f"Invalid model: {model}")


class BedrockTest(BaseModel):
    """
    A Pydantic model for validating Bedrock API requests.

    Attributes:
        api_key (str): The API key provided by the user for authentication with Bedrock's API.
        api_secret (str): The API secret key provided by the user for authentication.
        api_region (str): The API region for Bedrock API requests.
        model (str): The name of the model intended for use with the Bedrock API.

    Methods:
        validate_model: Ensures that `model` is one of the allowed values.
    """

    api_key: Optional[str]
    api_secret: Optional[str]
    api_region: Optional[str]
    model: str


class BedrockProvider(BaseProvider):
    """
    BedrockProvider class to interact with the Bedrock API.

    Attributes:
        bedrock_config (BedrockConfig): Configuration for the Bedrock API.
    """

    def __init__(self, config: BedrockConfig, api_key: dict):
        """
        Initialize the BedrockProvider class.

        Args:
            config (BedrockConfig): Configuration for the Bedrock API.
            api_key (dict): API key required for the Bedrock API.

        Raises:
            ValidationError: If the provided config and API key are invalid.
        """
        super().__init__()
        self.bedrock_config = validate_provider_config(config, api_key)

    async def chat(self, data: BedrockRequest) -> dict:
        """
        Endpoint to process chat input via Bedrock API and generate a model's response.

        Args:
            data (BedrockRequest): Validated API request data.

        Returns:
            Union[StreamingResponse, dict]: Streaming response if is_stream is True, otherwise a dict with chat and token data.
        """
        data = BedrockRequest(**data)
        self.validate_model_field(data, BEDROCK_MODELS)
        loop = asyncio.get_event_loop()
        session = boto3.Session(
            aws_access_key_id=self.bedrock_config["api_key"],
            aws_secret_access_key=self.bedrock_config["api_secret"],
        )
        bedrock = session.client(
            service_name="bedrock", region_name=self.bedrock_config["api_region"]
        )

        body, response_keys = generate_body_and_response(data)

        if data.is_stream:
            response = await loop.run_in_executor(
                None,
                lambda: bedrock.invoke_model_with_response_stream(
                    body=json.dumps(body),
                    modelId=data.model,
                    accept="application/json",
                    contentType="application/json",
                ).get("body"),
            )
            return StreamingResponse(generate_stream_response(response, response_keys))
        else:
            response = await loop.run_in_executor(
                None,
                lambda: json.loads(
                    bedrock.invoke_model(
                        body=json.dumps(body),
                        modelId=data.model,
                        accept="application/json",
                        contentType="application/json",
                    )
                    .get("body")
                    .read()
                ),
            )

        response = response["results"][0] if response_keys["use_results"] else response

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data.chat_input,
            "chatOutput": response[response_keys["output_key"]],
            "inputTokens": response.get(response_keys["input_tokens_key"], 0),
            "outputTokens": response.get(response_keys["output_tokens_key"], 0),
            "totalTokens": response.get(response_keys["input_tokens_key"], 0)
            + response.get(response_keys["output_tokens_key"], 0),
            "cost": 0,  # TODO
            "timestamp": time.time(),
            "model": data.model,
            "parameters": data.parameters.dict(),
        }
        return data

    async def test(self, data: BedrockTest) -> bool:
        """
        Test the validity of the Bedrock API credentials and model name.

        Args:
            data (BedrockTest): A model instance containing the Bedrock API credentials
                            and model name to test.

        Returns:
            bool: `True` if the API credentials and model name are valid, otherwise `False`.
        """
        data = BedrockTest(**data)
        try:
            session = boto3.Session(
                aws_access_key_id=self.bedrock_config["api_key"],
                aws_secret_access_key=self.bedrock_config["api_secret"],
            )
            bedrock = session.client(
                service_name="bedrock", region_name=self.bedrock_config["api_region"]
            )
            response = bedrock.list_foundation_models()

            if data.model in [i["modelId"] for i in response["modelSummaries"]]:
                return True
            else:
                return False
        except Exception:
            return False


def generate_body_and_response(data: BedrockProvider) -> Tuple[dict, dict]:
    """
    Generate request body and response keys based on model name.

    Args:
        data (BedrockRequest): Validated API request data.

    Returns:
        Tuple[dict, dict]: Tuple of request body and response keys.

    Raises:
        ValueError if model name is invalid.
    """
    if data.model in TITAN_MODELS:
        return {
            "inputText": data.chat_input,
            "textGenerationConfig": {
                "maxTokenCount": data.parameters.max_tokens,
                "temperature": data.parameters.temperature,
                "topP": data.parameters.top_p,
            },
        }, {
            "output_key": "outputText",
            "input_tokens_key": "inputTextTokenCount",
            "output_tokens_key": "tokenCount",
            "use_results": True,
        }
    if data.model in CLAUDE_MODELS:
        return {
            "prompt": data.chat_input,
            "max_tokens_to_sample": data.parameters.max_tokens,
            "temperature": data.parameters.temperature,
            "top_k": data.parameters.top_k,
            "top_p": data.parameters.top_p,
        }, {
            "output_key": "completion",
            "input_tokens_key": None,
            "output_tokens_key": None,
            "use_results": False,
        }
    else:
        raise ValueError(f"Invalid model: {data.model}")


def get_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost based on input and output tokens.

    Args:
        input_tokens (int): Number of tokens in the input.
        output_tokens (int): Number of tokens in the output.

    Returns:
        float: Cost.
    """
    return None


def generate_stream_response(response, response_keys):
    """
    Generate streaming response based on response events and keys.

    Args:
        response (Any): Response from the Bedrock API call.
        response_keys (Dict[str, Any]): Keys to extract relevant data from the response.

    Yields:
        str: Extracted data from response chunks.
    """
    chat_output = ""
    for event in response:
        chunk = event.get("chunk")
        if chunk:
            chunk_content = json.loads(chunk.get("bytes").decode())[response_keys["output_key"]]
            chat_output += chunk_content
            yield chunk_content

    input_tokens = 0
    output_tokens = 0
    cost = 0
    yield f"{END_TOKEN},{input_tokens},{output_tokens},{cost}"
