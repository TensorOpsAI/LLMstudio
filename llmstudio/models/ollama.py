from ..validators import OllamaParameters
from .models import LLMClient, LLMModel


class OllamaClient(LLMClient):
    MODEL_MAPPING = {
        "llama2": "Llama2",
    }

    def __init__(
        self,
        api_key: str = None,
    ):
        super().__init__(
            api_key=api_key,
        )

    class OllamaModel(LLMModel):
        PROVIDER = "ollama"

        def __init__(
            self,
            model: str,
            api_key: str,
            parameters: OllamaParameters = None,
        ):
            super().__init__(
                model,
                api_key or self._raise_api_key_error(),
                parameters=parameters,
            )

        def validate_parameters(
            self, parameters: OllamaParameters = None
        ) -> OllamaParameters:
            parameters = parameters or {}
            return OllamaParameters(**parameters).model_dump()

    class Llama2(OllamaModel):
        def __init__(self, model, api_key, parameters: OllamaParameters, **kwargs):
            super().__init__(
                model=model,
                api_key=api_key,
                parameters=parameters,
            )
