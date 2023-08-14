from fastapi import APIRouter, Request
from pydantic import BaseModel
import vertexai
from google.oauth2 import service_account
import json

router = APIRouter()


class VertexAIRequest(BaseModel):
    apiKey: str


@router.post("/vertexai")
async def test_vertexai(data: VertexAIRequest):
    try:
        json_credential = json.loads(data.apiKey)
        vertexai.init(
            project=json_credential["project_id"],
            credentials=service_account.Credentials.from_service_account_info(
                json_credential
            ),
        )
        return True
    except Exception as e:
        print(e)
        return False
