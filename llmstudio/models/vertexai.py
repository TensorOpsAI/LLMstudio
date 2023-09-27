from .models import LLMModel, LLMVendorClient


class VertexAIClient(LLMVendorClient):
    MODEL_MAPPING = {"text_bison@001": "TextBison", "chat_bison@001": "ChatBison"}

    class VertexAIModel(LLMModel):
        def __init__(self, model_name):
            self.model_name = model_name

    class TextBison(VertexAIModel):
        def __init__(self, model_name):
            super().__init__(model_name=model_name)

        def chat(self, chat_input: str, parameters: dict):
            pass  # make request text bison

    class ChatBison(VertexAIModel):
        def __init__(self, model_name):
            super().__init__(model_name=model_name)

        def chat(self, chat_input: str, parameters: dict):
            pass  # make request chat bison
