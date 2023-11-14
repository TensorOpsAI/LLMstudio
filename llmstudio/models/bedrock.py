from llmstudio.engine.config import EngineConfig

from ..validators import ClaudeParameters, TitanParameters
from .models import LLMClient, LLMModel


class BedrockClient(LLMClient):
    """
    Client class for interfacing with Bedrock LLM models.

    This class serves as a specific client for Bedrock LLM models, which may include various versions
    or types of models. The available models are mapped in `MODEL_MAPPING`.

    Attributes:
        MODEL_MAPPING (dict): A dictionary mapping human-readable model names to corresponding class names.
    """

    MODEL_MAPPING = {
        "amazon.titan-tg1-large": "Titan",
        "anthropic.claude-v1": "Claude",
        "anthropic.claude-instant-v1": "Claude",
        "anthropic.claude-v2": "Claude",
    }

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        api_region: str = None,
        engine_config: EngineConfig = EngineConfig(),
    ):
        """
        Initialize the BedrockClient instance.

        Args:
            engine_config (EngineConfig): The configuration object containing routes and other settings.
        """
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            api_region=api_region,
            engine_config=engine_config,
        )

    class BedrockModel(LLMModel):
        """
        Model class for interfacing with a generic Bedrock LLM.

        This class is meant to serve as a way to communicate with the Bedrock API, providing
        chat functionality through an API. It uses predefined URLs for checking API access and chatting.

        Attributes:
            CHAT_URL (str): Endpoint URL for chat functionality.
            TEST_URL (str): Endpoint URL for testing API access.
        """

        PROVIDER = "bedrock"

        def __init__(
            self,
            model: str,
            api_key: str,
            api_secret: str,
            api_region: str,
            engine_config: EngineConfig,
        ):
            super().__init__(model, api_key, api_secret, api_region, engine_config)
            self._check_api_access()

    class Claude(BedrockModel):
        """
        Model class for interfacing with the 'Claude' Bedrock LLM.

        A specific implementation of `BedrockModel` designed to work with different versions
        of the 'Claude' Bedrock LLM.
        """

        def __init__(
            self,
            model: str,
            api_key: str,
            api_secret: str,
            api_region: str,
            engine_config: EngineConfig,
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                api_secret=api_secret,
                api_region=api_region,
                engine_config=engine_config,
            )

        def validate_parameters(self, parameters: ClaudeParameters = None) -> ClaudeParameters:
            """
            Validate and possibly adjust the provided parameters for Claude model.

            Args:
                parameters (ClaudeParameters): Parameters to validate.

            Returns:
                ClaudeParameters: Validated/adjusted parameters.
            """
            parameters = parameters or {}
            return ClaudeParameters(**parameters).model_dump()

    class Titan(BedrockModel):
        """
        Model class for interfacing with the 'Titan' Bedrock LLM.

        A specialized implementation of `BedrockModel` intended for use with the 'Titan'
        Bedrock LLM model, leveraging predefined chat and testing URLs.
        """

        def __init__(
            self,
            model: str,
            api_key: str,
            api_secret: str,
            api_region: str,
            engine_config: EngineConfig,
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                api_secret=api_secret,
                api_region=api_region,
                engine_config=engine_config,
            )

        def validate_parameters(self, parameters: TitanParameters = None) -> TitanParameters:
            """
            Validate and possibly adjust the provided parameters for Titan model.

            Args:
                parameters (TitanParameters): Parameters to validate.

            Returns:
                TitanParameters: Validated/adjusted parameters.
            """
            parameters = parameters or {}
            return TitanParameters(**parameters).model_dump()
