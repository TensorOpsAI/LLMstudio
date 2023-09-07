import json

from fastapi import APIRouter, Request
from google.oauth2 import service_account
from pydantic import BaseModel, validator
import vertexai


router = APIRouter()


class VertexAITest(BaseModel):
    api_key: dict

    @validator("api_key", pre=True, always=True)
    def parse_api_key(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Given string is not valid JSON: {value}")
        return value


@router.post("/vertexai")
async def test_vertexai(data: VertexAITest):
    try:
        credentials = service_account.Credentials.from_service_account_info(
            data.api_key
        )
        vertexai.init(project=data.api_key["project_id"], credentials=credentials)
        return True
    except Exception:
        return False
