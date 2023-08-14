from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import ChatModel, TextGenerationModel
import json

router = APIRouter()


class Parameters(BaseModel):
    temperature: float
    maxTokens: int
    topP: float
    topK: float


class VertexAIRequest(BaseModel):
    apiKey: str
    model: str
    prompt: str
    parameters: dict


def get_vertexai_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    if model == "text-bison@001":
        return 0.000001 * (input_tokens + output_tokens)
    elif model == "chat-bison@001":
        return 0.0000005 * (input_tokens + output_tokens)
    return 0.0


@router.post("/vertexai")
async def get_vertexai_chat(data: VertexAIRequest):
    json_credential = json.loads(data.apiKey)

    vertexai.init(
        project=json_credential["project_id"],
        credentials=service_account.Credentials.from_service_account_info(
            json_credential
        ),
    )

    if data.model == "text-bison@001":
        response = TextGenerationModel.from_pretrained("text-bison@001").predict(
            prompt=data.prompt,
            temperature=data.parameters["temperature"],
            max_output_tokens=data.parameters["maxTokens"],
            top_p=data.parameters["topP"],
            top_k=data.parameters["topK"],
        )
    elif data.model == "chat-bison@001":
        response = (
            ChatModel.from_pretrained("chat-bison@001")
            .start_chat()
            .send_message(
                message=data.prompt,
                temperature=data.parameters["temperature"],
                max_output_tokens=data.parameters["maxTokens"],
                top_p=data.parameters["topP"],
                top_k=data.parameters["topK"],
            )
        )

    return {
        "output": response.text,
        "input_tokens": len(data.prompt),
        "output_tokens": len(response.text),
        "cost": get_vertexai_cost(len(data.prompt), len(response.text), data.model),
    }
