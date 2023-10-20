from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from ..chat.openai import openai_chat_endpoint, OpenAIRequest


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
        tests (dict): The input tests for the model.
        parameters (Optional[OpenAIParameters]): Additional parameters to control the model’s output.
        is_stream (Optional[bool]): Flag to determine whether to receive streamed responses from the API.
    
    Methods:
        validate_model_name: Ensures that the chosen model_name is one of the allowed models.
        validate_tests: Ensure that tests are not empty or None
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
    def validate_tests(cls, tests):
        """
        Validate that the tests are not empty
        
        Args:
            tests: The tests to validate.
        
        Returns:
            str: The validated tests.
        
        Raises:
            ValueError: If the tests are empty.
        """
        if not tests:
            raise ValueError(f"tests should not be empty")
    
        if not isinstance(tests, dict):
            raise ValueError(f"tests is not a dict")
        for key, value in tests.items():
            if not isinstance(key, str):
                raise ValueError(f"key of tests is not a string")
            if not isinstance(value, dict):
                raise ValueError(f"value of tests is not a Dictionary")
            if 'test' not in value or 'answer' not in value:
                raise ValueError("tests value should be in format \{test: \"...\", answer: \"...\"\}")
            if not (isinstance(value['test'], str) and isinstance(value['answer'], str)):
                raise ValueError(f"test and answer should be strings")
        return tests

@router.post("/openai")
async def openai_evaluate_endpoint(data: OpenAIEvaluator):
    """
    FastAPI endpoint to interact with the OpenAI API for chat completions of tests.
    
    Args:
        data (OpenAIEvaluator): OpenAIEvaluator object containing necessary parameters for the API call.
    
    Returns:
        Union[StreamingResponse, dict]: Streaming response if is_stream is True, otherwise a dict with chat and token data.
    """
    test_responses = {}
    for key in data.tests:
        test, answer = data.tests[key].values()
        print(f"Key: {key} / Test: {test} / Answer: {answer}")

        request_data = OpenAIRequest(api_key=data.api_key,model_name=data.model_name,chat_input=test,parameters=data.parameters,is_stream=data.is_stream)
        response = await openai_chat_endpoint(request_data)
        """response = requests.post(
            "http://localhost:8000/api/chat/openai",
            json={
                "model_name": data.model_name,
                "api_key": data.api_key,
                "chat_input": test,
                "parameters": data.parameters,
                "is_stream": data.is_stream
            },
            headers={"Content-Type": "application/json"},
            timeout=10
        )"""
        test_responses[key] = response

    return test_responses