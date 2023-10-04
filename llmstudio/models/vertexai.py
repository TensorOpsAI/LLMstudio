from .models import LLMModel, LLMClient
from ..validators import VertexAIParameters


class VertexAIClient(LLMClient):
    """
    Client class for interfacing with Vertex AI LLM models.
    
    This class acts as a specific client tailored for Vertex AI LLM models, 
    which can include various kinds or versions of models as mapped in `MODEL_MAPPING`.

    Attributes:
    MODEL_MAPPING (dict): A dictionary mapping model names to corresponding class names.
    """
    MODEL_MAPPING = {"text-bison": "TextBison", "chat-bison": "ChatBison"}

    class VertexAIModel(LLMModel):
        """
        Model class for interfacing with a generic Vertex AI LLM.

        This class aims to facilitate communication with the Vertex AI API, 
        providing chat functionality through predefined API endpoints.

        Attributes:
        CHAT_URL (str): API endpoint URL for chat functionality.
        TEST_URL (str): API endpoint URL for testing API access.
        """
        CHAT_URL = "http://localhost:8000/api/chat/vertexai"
        TEST_URL = "http://localhost:8000/api/test/vertexai"

        def __init__(self, model_name, api_key):
            super().__init__(
                model_name,
                api_key or self._raise_api_key_error()
            )
            self._check_api_access()

        def validate_parameters(self, parameters: VertexAIParameters = None) -> VertexAIParameters:
            """
            Validate and possibly adjust the provided parameters for Vertex AI models.

            Args:
            parameters (VertexAIParameters): Parameters to validate.

            Returns:
            VertexAIParameters: Validated/adjusted parameters.
            """
            parameters = parameters or {}
            return VertexAIParameters(**parameters).model_dump()

    class TextBison(VertexAIModel):
        """
        Model class for interfacing with the 'TextBison' Vertex AI LLM.

        A specific implementation of `VertexAIModel` tailored to work with the 
        'TextBison' Vertex AI LLM.
        """
        def __init__(self, model_name, api_key, **kwargs):
            super().__init__(model_name=model_name, api_key=api_key)

    class ChatBison(VertexAIModel):
        """
        Model class for interfacing with the 'ChatBison' Vertex AI LLM.

        A specific implementation of `VertexAIModel` meant for interfacing 
        with the 'ChatBison' Vertex AI LLM.
        """
        def __init__(self, model_name, api_key, **kwargs):
            super().__init__(model_name=model_name, api_key=api_key)