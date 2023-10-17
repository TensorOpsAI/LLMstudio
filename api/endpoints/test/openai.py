import openai
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, validator

router = APIRouter()


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

    api_key: str
    model_name: str

    @validator("model_name", always=True)
    def validate_model_name(cls, value: str):
        """
        Ensures that the provided `model_name` is one of the allowed values
        to prevent potential invalid requests to the OpenAI API.

        Args:
            value (str): The name of the model, intended to be used for text generation.

        Returns:
            str: The validated `model_name`.

        Raises:
            ValueError: If `model_name` is not one of the allowed values.
        """
        allowed_values = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value


@router.post("/openai")
async def test_openai(data: OpenAITest) -> bool:
    """
    Test the validity of the OpenAI API key.

    Args:
        data (OpenAITest): A model instance which includes the API key for OpenAI.

    Returns:
        bool: `True` if the API key is valid and initialization succeeds, otherwise `False`.
    """
    openai.api_key = data.api_key
    try:
        openai.Model.retrieve(data.model_name)
        return True
    except Exception:
        return False
