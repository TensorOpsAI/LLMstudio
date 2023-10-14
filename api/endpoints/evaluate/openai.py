from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
import openai


router = APIRouter()

# This class is repeated code from chat
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


class OpenAIEvaluator(BaseModel):
    """
    A Pydantic model for encapsulating data needed to make a test requests to the OpenAI API.
    
    Attributes:
        api_key (str): Authentication key for the OpenAI API.
        model_name (str): The identifier of the GPT model to be used.
        tests (str): The input tests for the model.
        parameters (Optional[OpenAIParameters]): Additional parameters to control the model’s output.
        is_stream (Optional[bool]): Flag to determine whether to receive streamed responses from the API.
    
    Methods:
        validate_model_name: Ensures that the chosen model_name is one of the allowed models.
    """
    api_key: str
    model_name: str
    tests: dict
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        """
        Validate that the model name is one of the allowed options.
        
        Args:
            value: The model name to validate.
        
        Returns:
            str: The validated model name.
        
        Raises:
            ValueError: If the model name is not an allowed value.
        """
        allowed_values = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value

    @validator("tests", always=True)
    def validate_tests(cls, value):
        """
        Validate that the tests are not empty
        
        Args:
            value: The tests to validate.
        
        Returns:
            str: The validated tests.
        
        Raises:
            ValueError: If the tests are empty.
        """
        if not value:
            raise ValueError(f"tests should not be empty")
        return value


@router.post("/openai")
async def openai_evaluate_endpoint(data: OpenAIEvaluator):
    """
    FastAPI endpoint to interact with the OpenAI API for chat completions of tests.
    
    Args:
        data (OpenAIEvaluator): OpenAIEvaluator object containing necessary parameters for the API call.
    
    Returns:
        Union[StreamingResponse, dict]: Streaming response if is_stream is True, otherwise a dict with chat and token data.
    """
    openai.api_key = data.api_key

    return "Working"

    """response = openai.ChatCompletion.create(
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
    )

    if data.is_stream:
        return StreamingResponse(generate_stream_response(response, data))
    
    input_tokens = get_tokens(data.chat_input, data.model_name)
    output_tokens = get_tokens(
        response["choices"][0]["message"]["content"], data.model_name
    )

    import random, time # delete

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

    append_log(data)
    return data"""