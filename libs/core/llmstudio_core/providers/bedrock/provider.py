from typing import Any, AsyncGenerator, Coroutine, Generator

from llmstudio_core.providers.bedrock.anthropic import BedrockAnthropicProvider
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider


@provider
class BedrockProvider(ProviderCore):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.kwargs = kwargs
        self.selected_model = None

    def _get_provider(self, model):
        if "anthropic." in model:
            return BedrockAnthropicProvider(config=self.config, **self.kwargs)

        raise ValueError(f" provider is not yet supported.")

    @staticmethod
    def _provider_config_name():
        return "bedrock"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Any]:
        self.selected_model = self._get_provider(request.model)
        return await self.selected_model.agenerate_client(request)

    def generate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Generator]:
        self.selected_model = self._get_provider(request.model)
        return self.selected_model.generate_client(request=request)

    async def aparse_response(
        self, response: Any, **kwargs
    ) -> AsyncGenerator[Any, None]:
        result = await self.selected_model.aparse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def parse_response(self, response: AsyncGenerator[Any, None], **kwargs) -> Any:
        return self.selected_model.parse_response(response=response, **kwargs)
