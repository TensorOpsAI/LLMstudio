import boto3
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, validator

router = APIRouter()


class BedrockTest(BaseModel):
    """
    A Pydantic model for validating Bedrock API requests.

    Attributes:
        api_key (str): The API key provided by the user for authentication with Bedrock's API.
        api_secret (str): The API secret key provided by the user for authentication.
        api_region (str): The API region for Bedrock API requests.
        model_name (str): The name of the model intended for use with the Bedrock API.

    Methods:
        validate_model_name: Ensures that `model_name` is one of the allowed values.
    """
    api_key: str
    api_secret: str
    api_region: str
    model_name: str

    @validator("model_name", always=True)
    def validate_model_name(cls, value: str):
        """
        Ensures that the provided `model_name` is one of the allowed values 
        to prevent potential invalid requests to the Bedrock API.

        Args:
            value (str): The name of the model.

        Returns:
            str: The validated `model_name`.

        Raises:
            ValueError: If `model_name` is not one of the allowed values.
        ```
        """
        allowed_values = [
            "amazon.titan-tg1-large",
            "anthropic.claude-instant-v1",
            "anthropic.claude-v1",
            "anthropic.claude-v2",
        ]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value


@router.post("/bedrock")
async def test_openai(data: BedrockTest) -> bool:
    """
    Test the validity of the Bedrock API credentials and model name.

    Args:
        data (BedrockTest): A model instance containing the Bedrock API credentials
                          and model name to test.

    Returns:
        bool: `True` if the API credentials and model name are valid, otherwise `False`.
    """
    try:
        session = boto3.Session(
            aws_access_key_id=data.api_key, aws_secret_access_key=data.api_secret
        )
        bedrock = session.client(service_name="bedrock", region_name=data.api_region)
        response = bedrock.list_foundation_models()

        if data.model_name in [i["modelId"] for i in response["modelSummaries"]]:
            return True
        else:
            return False
    except Exception:
        return False
