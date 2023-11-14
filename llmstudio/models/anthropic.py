from llmstudio.engine.config import EngineConfig

from ..validators import AnthropicParameters
from .models import LLMClient, LLMModel


class AnthropicClient(LLMClient):

    MODEL_MAPPING = {
        "claude-2": "Claude2",
        "claude-instant-1": "Claude1",
        "claude-instant-1.2": "Claude1_2",
    }

    def __init__(
        self,
        api_key: str = None,
        engine_config: EngineConfig = EngineConfig(),
    ):
        super().__init__(
            api_key=api_key,
            engine_config=engine_config,
        )

    class AnthropicModel(LLMModel):
        PROVIDER = "anthropic"

        def __init__(
            self,
            model: str,
            api_key: str,
            engine_config: EngineConfig,
            parameters: AnthropicParameters = None,
        ):
            super().__init__(
                model,
                api_key or self._raise_api_key_error(),
                engine_config=engine_config,
                parameters=parameters,
            )
            self._check_api_access()

        def validate_parameters(
            self, parameters: AnthropicParameters = None
        ) -> AnthropicParameters:
            parameters = parameters or {}
            return AnthropicParameters(**parameters).model_dump()

    class Claude2(AnthropicModel):
        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: AnthropicParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class Claude1(AnthropicModel):
        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: AnthropicParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )

    class Claude1_2(AnthropicModel):
        def __init__(
            self,
            model,
            api_key,
            engine_config: EngineConfig,
            parameters: AnthropicParameters,
            **kwargs
        ):
            super().__init__(
                model=model,
                api_key=api_key,
                engine_config=engine_config,
                parameters=parameters,
            )
