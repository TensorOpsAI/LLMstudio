import os

from llmstudio.engine.config import EngineConfig

from ..validators import OpenAIParameters
from .models import LLMClient, LLMModel


class OpenAIClient(LLMClient):
    """
    Client class for interfacing with OpenAI LLM models.

    This client is tailored for interfacing with OpenAI LLM models and contains
    a mapping of human-readable model names to class names in `MODEL_MAPPING`.

    Attributes:
        MODEL_MAPPING (dict): A dictionary mapping model names to corresponding class names.
    """

    MODEL_MAPPING = {"gpt-3.5-turbo": "GPT3_5", "gpt-4": "GPT4"}

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
        engine_config: EngineConfig = EngineConfig(),
    ):
        """
        Initialize the OpenAIClient instance.

        Args:
            engine_config (EngineConfig): The configuration object containing routes and other settings.
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            api_region=api_region,
            engine_config=engine_config,
        )

    class OpenAIModel(LLMModel):
        """
        Model class for interfacing with a generic OpenAI LLM.

        This class is designed to interact with the OpenAI API, offering chat
        functionality through predefined API endpoints.

        Attributes:
            CHAT_URL (str): Endpoint URL for chat functionality.
            TEST_URL (str): Endpoint URL for API access testing.
        """

        PROVIDER = "openai"

        def __init__(
            self,
            model: str,
            api_key: str,
            engine_config: EngineConfig,
            parameters: OpenAIParameters = None,
        ):
            super().__init__(
                model,
                api_key or os.environ.get("OPENAI_API_KEY") or self._raise_api_key_error(),
                engine_config=engine_config,
                parameters=parameters,
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

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: OpenAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class GPT4(OpenAIModel):
        """
        Model class for interfacing with the 'GPT-4' OpenAI LLM.

        A specialized implementation of `OpenAIModel` meant for interfacing
        with the 'GPT-4' OpenAI LLM.
        """

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: OpenAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )
