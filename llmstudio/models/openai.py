import os
import requests

from .models import LLMModel, LLMVendorClient


class OpenAIClient(LLMVendorClient):
    MODEL_MAPPING = {"gpt-3.5-turbo": "GPT3_5", "gpt-4": "GPT4"}

    class OpenAIModel(LLMModel):
        CHAT_URL = "http://localhost:8000/api/chat/openai"
        TEST_URL = "http://localhost:8000/api/test/openai"

        def __init__(self, model_name, api_key):
            self.model_name = model_name
            self.api_key = (
                api_key
                or os.environ.get("LS_OPENAI_KEY")
                or self._raise_api_key_error()
            )
            self._check_api_access()

        @staticmethod
        def _raise_api_key_error():
            raise ValueError(
                "Please provide api_key parameter or set the LS_OPENAI_KEY environment variable."
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

            # if is_stream:
            #     for chunk in response.iter_content(chunk_size=8192):
            #         yield chunk
            # else:
            #     return response.json()

    class GPT3_5(OpenAIModel):
        def __init__(self, model_name, api_key):
            super().__init__(model_name=model_name, api_key=api_key)

    class GPT4(OpenAIModel):
        def __init__(self, model_name, api_key):
            super().__init__(model_name=model_name, api_key=api_key)
