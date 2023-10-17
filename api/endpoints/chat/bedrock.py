import json
import os
from typing import Any, Dict, Optional, Tuple

import boto3
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from api.utils import append_log
from api.worker.config import celery_app

end_token = "<END_TOKEN>"

router = APIRouter()

TITAN_MODELS = ["amazon.titan-tg1-large"]
CLAUDE_MODELS = [
    "anthropic.claude-instant-v1",
    "anthropic.claude-v1",
    "anthropic.claude-v2",
]


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
    Model for validating and storing API request data.

    Attributes:
        api_key (str): API key for authentication.
        api_secret (str): API secret for authentication.
        api_region (str): AWS region where the API is hosted.
        model_name (str): Name of the model to be used.
        chat_input (str): Input text to be processed by the model.
        parameters (Optional[BaseModel]): Parameters controlling model's generation behavior.
        is_stream (Optional[bool]): Bool indicating whether response should be a stream.


    Methods:
        validate_model_name: Ensures that the chosen model_name is one of the allowed models.
        validate_parameters_based_on_model_name: Ensures that the parameters class belong to the model selected.
    """

    api_key: str
    api_secret: str
    api_region: str
    model_name: str
    chat_input: str
    parameters: Optional[BaseModel]
    is_stream: Optional[bool] = False

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        """
        Validate model_name against allowed values.

        Args:
            value (str): Input model_name to validate.

        Returns:
            str: Validated model_name.

        Raises:
            ValueError if model_name is not in allowed values.
        """
        allowed_values = TITAN_MODELS + CLAUDE_MODELS
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value

    @validator("parameters", pre=True, always=True)
    def validate_parameters_based_on_model_name(cls, parameters, values):
        """
        Validate and convert parameters based on the model_name.

        Args:
            parameters (Dict[str, Any]): Parameters to validate and convert.
            values (Dict[str, Any]): Contains previously validated fields.

        Returns:
            BaseModel: An instance of `TitanParameters` or `ClaudeParameters` based on `model_name`.

        Raises:
            ValueError if model_name is invalid.
        """
        model_name = values.get("model_name")
        if model_name in TITAN_MODELS:
            return TitanParameters(**parameters)
        if model_name in CLAUDE_MODELS:
            return ClaudeParameters(**parameters)

        raise ValueError(f"Invalid model_name: {model_name}")


def generate_body_and_response(data: BedrockRequest) -> Tuple[dict, dict]:
    """
    Generate request body and response keys based on model name.

    Args:
        data (BedrockRequest): Validated API request data.

    Returns:
        Tuple[dict, dict]: Tuple of request body and response keys.

    Raises:
        ValueError if model name is invalid.
    """
    if data.model_name in TITAN_MODELS:
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
    if data.model_name in CLAUDE_MODELS:
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
        raise ValueError(f"Invalid model_name: {data.model_name}")


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
    yield f"{end_token},{input_tokens},{output_tokens},{cost}"


@router.post("/bedrock")
async def get_bedrock_chat(data: BedrockRequest):
    """
    Endpoint to process chat input via Bedrock API and generate a model's response.

    Args:
        data (BedrockRequest): Validated API request data.

    Returns:
        Union[StreamingResponse, dict]: Streaming response if is_stream is True, otherwise a dict with chat and token data.
    """
    session = boto3.Session(aws_access_key_id=data.api_key, aws_secret_access_key=data.api_secret)
    bedrock = session.client(service_name="bedrock", region_name=data.api_region)

    body, response_keys = generate_body_and_response(data)

    if data.is_stream:
        response = bedrock.invoke_model_with_response_stream(
            body=json.dumps(body),
            modelId=data.model_name,
            accept="application/json",
            contentType="application/json",
        ).get("body")
        return StreamingResponse(generate_stream_response(response, response_keys))

    response = json.loads(
        bedrock.invoke_model(
            body=json.dumps(body),
            modelId=data.model_name,
            accept="application/json",
            contentType="application/json",
        )
        .get("body")
        .read()
    )

    response = response["results"][0] if response_keys["use_results"] else response

    import random  # delete
    import time

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
        "modelName": data.model_name,
        "parameters": data.parameters.dict(),
    }

    append_log(data)
    return data
