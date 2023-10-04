import json

from fastapi import APIRouter, Request
from google.oauth2 import service_account
from pydantic import BaseModel, validator
import vertexai


router = APIRouter()


class VertexAITest(BaseModel):
    """
    A Pydantic model to validate the input for testing Vertex AI API key.

    Attributes:
    api_key (Union[dict, str]): A dictionary containing API key data for Vertex AI.

    Methods:
    parse_api_key(cls, value): Parses the `api_key` attribute to ensure it's a dict
    """
    api_key: dict

    @validator("api_key", pre=True, always=True)
    def parse_api_key(cls, value: str):
        """
        Validate and/or parse the `api_key` attribute.

        Args:
        value (Union[dict, str]): API key either as a JSON-formatted string or dictionary.

        Returns:
        dict: The parsed `api_key` as a Python dictionary.

        Raises:
        ValueError: If `value` is a string and is not valid JSON
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception as exception:
                raise ValueError(f"Given string is not valid JSON: {value}") from exception
        return value


@router.post("/vertexai")
async def test_vertexai(data: VertexAITest) -> bool:
    """
    Test the validity of the Vertex AI API key.

    Args:
    data (VertexAITest): A model instance which includes the API key for Vertex AI.

    Returns:
    bool: `True` if the API key is valid and initialization succeeds, otherwise `False`.
    """
    try:
        credentials = service_account.Credentials.from_service_account_info(
            data.api_key
        )
        vertexai.init(project=data.api_key["project_id"], credentials=credentials)
        return True
    except Exception:
        return False
