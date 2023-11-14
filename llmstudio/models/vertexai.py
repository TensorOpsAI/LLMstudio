from llmstudio.engine.config import EngineConfig

from ..validators import VertexAIParameters
from .models import LLMClient, LLMModel


class VertexAIClient(LLMClient):
    """
    Client class for interfacing with Vertex AI LLM models.

    This class acts as a specific client tailored for Vertex AI LLM models,
    which can include various kinds or versions of models as mapped in `MODEL_MAPPING`.

    Attributes:
        MODEL_MAPPING (dict): A dictionary mapping model names to corresponding class names.
    """

    MODEL_MAPPING = {
        "text-bison": "TextBison",
        "chat-bison": "ChatBison",
        "code-bison": "CodeBison",
        "codechat-bison": "CodeChatBison",
    }

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
        engine_config: EngineConfig = EngineConfig(),
    ):
        """
        Initialize the VertexAI Client instance.

        Args:
            engine_config (EngineConfig): The configuration object containing routes and other settings.
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            api_region=api_region,
            engine_config=engine_config,
        )

    class VertexAIModel(LLMModel):
        """
        Model class for interfacing with a generic Vertex AI LLM.

        This class aims to facilitate communication with the Vertex AI API,
        providing chat functionality through predefined API endpoints.

        Attributes:
            CHAT_URL (str): API endpoint URL for chat functionality.
            TEST_URL (str): API endpoint URL for testing API access.
        """

        PROVIDER = "vertexai"

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: VertexAIParameters = None,
        ):
            super().__init__(
                model,
                api_key or self._raise_api_key_error(),
                engine_config=engine_config,
                parameters=parameters,
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

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: VertexAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class ChatBison(VertexAIModel):
        """
        Model class for interfacing with the 'ChatBison' Vertex AI LLM.

        A specific implementation of `VertexAIModel` meant for interfacing
        with the 'ChatBison' Vertex AI LLM.
        """

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: VertexAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class CodeBison(VertexAIModel):
        """
        Model class for interfacing with the 'CodeBison' Vertex AI LLM.

        A specific implementation of `VertexAIModel` meant for interfacing
        with the 'ChatBison' Vertex AI LLM.
        """

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: VertexAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class CodeChatBison(VertexAIModel):
        """
        Model class for interfacing with the 'CodeChatBison' Vertex AI LLM.

        A specific implementation of `VertexAIModel` meant for interfacing
        with the 'ChatBison' Vertex AI LLM.
        """

        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: VertexAIParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )
