from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import openai

router = APIRouter()


class TestOpenAIRequest(BaseModel):
    apiKey: str
    model: str


@router.post("/openai")
async def test_openai(data: TestOpenAIRequest):
    openai.api_key = data.apiKey
    try:
        openai.Model.retrieve(data.model)
        return True
    except:
        return False
