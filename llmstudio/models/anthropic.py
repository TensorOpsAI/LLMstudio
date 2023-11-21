from ..validators import AnthropicParameters
from .models import LLMClient, LLMModel


class AnthropicClient(LLMClient):
    MODEL_MAPPING = {
        "claude-2.1": "Claude2",
        "claude-2": "Claude2",
        "claude-instant-1": "Claude1",
        "claude-instant-1.2": "Claude1",
    }

    def __init__(
        self,
        api_key: str = None,
    ):
        super().__init__(
            api_key=api_key,
        )

    class AnthropicModel(LLMModel):
        PROVIDER = "anthropic"

        def __init__(
            self,
            model: str,
            api_key: str,
            parameters: AnthropicParameters = None,
        ):
            super().__init__(
                model,
                api_key or self._raise_api_key_error(),
                parameters=parameters,
            )

        def validate_parameters(
            self, parameters: AnthropicParameters = None
        ) -> AnthropicParameters:
            parameters = parameters or {}
            return AnthropicParameters(**parameters).model_dump()

    class Claude2(AnthropicModel):
        def __init__(self, model, api_key, parameters: AnthropicParameters, **kwargs):
            super().__init__(
                model=model,
                api_key=api_key,
                parameters=parameters,
            )

    class Claude1(AnthropicModel):
        def __init__(self, model, api_key, parameters: AnthropicParameters, **kwargs):
            super().__init__(
                model=model,
                api_key=api_key,
                parameters=parameters,
            )
