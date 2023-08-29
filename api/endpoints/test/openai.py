import openai
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, validator

router = APIRouter()


class OpenAITest(BaseModel):
    api_key: str
    model_name: str

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        allowed_values = ["gpt-3.5-turbo", "gpt-4"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value


@router.post("/openai")
async def test_openai(data: OpenAITest):
    openai.api_key = data.api_key
    try:
        openai.Model.retrieve(data.model_name)
        return True
    except:
        return False
