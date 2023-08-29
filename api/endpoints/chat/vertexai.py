from typing import Optional
import json

from fastapi import APIRouter
from google.oauth2 import service_account
from pydantic import BaseModel, Field, validator, ValidationError
import vertexai
from vertexai.language_models import TextGenerationModel, ChatModel

from api.worker.config import celery_app

router = APIRouter()


class VertexAIParameters(BaseModel):
    temperature: Optional[float] = Field(1, ge=0, le=1)
    max_tokens: Optional[int] = Field(256, ge=1, le=1024)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    top_k: Optional[float] = Field(40, ge=1, le=40)


class VertexAIRequest(BaseModel):
    api_key: dict
    model_name: str
    chat_input: str
    parameters: VertexAIParameters

    @validator("api_key", pre=True, always=True)
    def parse_api_key(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Given string is not valid JSON: {value}")
        return value

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        allowed_values = ["text-bison@001", "chat-bison@001"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value


@router.post("/vertexai")
async def get_vertexai_chat(data: VertexAIRequest):
    def get_cost(input_tokens: int, output_tokens: int) -> float:
        return 0.0000005 * (input_tokens + output_tokens)

    credentials = service_account.Credentials.from_service_account_info(data.api_key)
    vertexai.init(project=data.api_key["project_id"], credentials=credentials)

    if data.model_name == "text-bison@001":
        response = TextGenerationModel.from_pretrained(data.model_name).predict(
            prompt=data.chat_input,
            temperature=data.parameters.temperature,
            max_output_tokens=data.parameters.max_tokens,
            top_p=data.parameters.top_p,
            top_k=data.parameters.top_k,
        )
    elif data.model_name == "chat-bison@001":
        response = (
            ChatModel.from_pretrained("data.model_name")
            .start_chat()
            .send_message(
                message=data.chat_input,
                temperature=data.parameters.temperature,
                max_output_tokens=data.parameters.max_tokens,
                top_p=data.parameters.top_p,
                top_k=data.parameters.top_k,
            )
        )

    return {
        "output": response.text,
        "input_tokens": len(data.chat_input),
        "output_tokens": len(response.text),
        "cost": get_cost(len(data.chat_input), len(response.text)),
    }

    # task = vertexai_chat_worker.delay(
    #     data.api_key, data.model_name, data.input, data.parameters
    # )
    # return {"task_id": task.id}
