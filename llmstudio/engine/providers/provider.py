from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel


class ChatRequest(BaseModel):
    api_key: Optional[str] = None
    model: str
    chat_input: str
    parameters: Optional[BaseModel] = None
    is_stream: Optional[bool] = False
    has_end_token: Optional[bool] = False


class Provider:
    def __init__(self, config):
        self.config = config
        self.END_TOKEN = "<END_TOKEN>"

    async def chat(self, chat_request: ChatRequest):
        if chat_request.model not in self.config.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {chat_request.model} is not supported by {self.config.name}",
            )

    async def embed(self, chat_request: ChatRequest):
        pass

    def generate_response(self, response: dict, request: ChatRequest):
        pass

    def generate_stream(self, response: dict, request: ChatRequest):
        pass

    def calculate_tokens_and_cost(self, input: str, model: str, type: str):
        pass
