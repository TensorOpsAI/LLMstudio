import requests

from .models import LLMModel, LLMVendorClient


class BedrockClient(LLMVendorClient):
    MODEL_MAPPING = {
        "amazon.titan-tg1-large": "Titan",
        "anthropic.claude-v1": "Claude",
        "anthropic.claude-instant-v1": "Claude",
        "anthropic.claude-v2": "Claude",
    }

    def get_model(self, model_name: str):
        model_class_name = self.MODEL_MAPPING.get(model_name)
        if not model_class_name:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = getattr(self, model_class_name)
        return model_class(
            model_name=model_name,
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_region=self.api_region,
        )

    class BedrockModel(LLMModel):
        CHAT_URL = "http://localhost:8000/api/chat/bedrock"
        TEST_URL = "http://localhost:8000/api/test/bedrock"

        def __init__(
            self, model_name: str, api_key: str, api_secret: str, api_region: str
        ):
            self.model_name = model_name
            self.api_key = api_key
            self.api_secret = api_secret
            self.api_region = api_region
            # self._check_api_access()

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

        def chat(self, chat_input: str, parameters: dict = {}):
            response = requests.post(
                self.CHAT_URL,
                json={
                    "model_name": self.model_name,
                    "api_key": self.api_key,
                    "api_secret": self.api_secret,
                    "api_region": self.api_region,
                    "chat_input": chat_input,
                    "parameters": parameters,
                },
                headers={"Content-Type": "application/json"},
            )

            return response.json()

    class Claude(BedrockModel):
        def __init__(
            self, model_name: str, api_key: str, api_secret: str, api_region: str
        ):
            super().__init__(
                model_name=model_name,
                api_key=api_key,
                api_secret=api_secret,
                api_region=api_region,
            )

    class Titan(BedrockModel):
        def __init__(
            self, model_name: str, api_key: str, api_secret: str, api_region: str
        ):
            super().__init__(
                model_name=model_name,
                api_key=api_key,
                api_secret=api_secret,
                api_region=api_region,
            )
