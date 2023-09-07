import boto3
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, validator

router = APIRouter()


class BedrockTest(BaseModel):
    api_key: str
    api_secret: str
    api_region: str
    model_name: str

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
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
async def test_openai(data: BedrockTest):
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
    except:
        return False
