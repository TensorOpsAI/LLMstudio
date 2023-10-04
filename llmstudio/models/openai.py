import os

from .models import LLMModel, LLMClient
from ..validators import OpenAIParameters


class OpenAIClient(LLMClient):
    """
    Client class for interfacing with OpenAI LLM models.

    This client is tailored for interfacing with OpenAI LLM models and contains 
    a mapping of human-readable model names to class names in `MODEL_MAPPING`.

    Attributes:
    MODEL_MAPPING (dict): A dictionary mapping model names to corresponding class names.
    """
    MODEL_MAPPING = {"gpt-3.5-turbo": "GPT3_5", "gpt-4": "GPT4"}

    class OpenAIModel(LLMModel):
        """
        Model class for interfacing with a generic OpenAI LLM.

        This class is designed to interact with the OpenAI API, offering chat 
        functionality through predefined API endpoints.

        Attributes:
        CHAT_URL (str): Endpoint URL for chat functionality.
        TEST_URL (str): Endpoint URL for API access testing.
        """
        CHAT_URL = "http://localhost:8000/api/chat/openai"
        TEST_URL = "http://localhost:8000/api/test/openai"

        def __init__(self, model_name, api_key):
            super().__init__(
                model_name,
                api_key or os.environ.get("OPENAI_API_KEY") or self._raise_api_key_error()
            )
            self._check_api_access()

        def validate_parameters(self, parameters: OpenAIParameters = None) -> OpenAIParameters:
            """
            Validate and possibly adjust the provided parameters for OpenAI models.

            Args:
            parameters (OpenAIParameters): Parameters to validate.

            Returns:
            OpenAIParameters: Validated/adjusted parameters.
            """
            parameters = parameters or {}
            return OpenAIParameters(**parameters).model_dump()

    class GPT3_5(OpenAIModel):
        """
        Model class for interfacing with the 'GPT-3.5-turbo' OpenAI LLM.

        A specialized implementation of `OpenAIModel` designed to work with the 
        'GPT-3.5-turbo' OpenAI LLM.
        """
        def __init__(self, model_name, api_key, **kwargs):
            super().__init__(model_name=model_name, api_key=api_key)

    class GPT4(OpenAIModel):
        """
        Model class for interfacing with the 'GPT-4' OpenAI LLM.

        A specialized implementation of `OpenAIModel` meant for interfacing 
        with the 'GPT-4' OpenAI LLM.
        """
        def __init__(self, model_name, api_key, **kwargs):
            super().__init__(model_name=model_name, api_key=api_key)
