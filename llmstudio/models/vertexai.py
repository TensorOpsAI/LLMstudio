import os
import requests

from .models import LLMModel, LLMVendorClient


class VertexAIClient(LLMVendorClient):
    MODEL_MAPPING = {"text-bison": "TextBison", "chat-bison": "ChatBison"}

    class VertexAIModel(LLMModel):
        CHAT_URL = "http://localhost:8000/api/chat/vertexai"
        TEST_URL = "http://localhost:8000/api/test/vertexai"

        def __init__(self, model_name, api_key):
            self.model_name = model_name
            self.api_key = (
                api_key
                or self._raise_api_key_error()
            )
            self._check_api_access()

        @staticmethod
        def _raise_api_key_error():
            raise ValueError(
                "Please provide api_key parameter."
            )

        def _check_api_access(self):
            response = requests.post(
                self.TEST_URL,
                json={
                    "model_name": self.model_name,
                    "api_key": self.api_key,
                },
                headers={"Content-Type": "application/json"},
            )
            if not response.json():
                raise ValueError(
                    f"The API key doesn't have access to {self.model_name}"
                )

        def chat(self, chat_input: str, parameters: dict = {}, is_stream: bool = False):
            response = requests.post(
                self.CHAT_URL,
                json={
                    "model_name": self.model_name,
                    "api_key": self.api_key,
                    "chat_input": chat_input,
                    "parameters": parameters,
                    "is_stream": is_stream,
                },
                headers={"Content-Type": "application/json"},
            )

            return response.json()

    class TextBison(VertexAIModel):
        def __init__(self, model_name):
            super().__init__(model_name=model_name)

    class ChatBison(VertexAIModel):
        def __init__(self, model_name, api_key):
            super().__init__(model_name=model_name, api_key=api_key)