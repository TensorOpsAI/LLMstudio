import os
from typing import Any, AsyncGenerator, Generator

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import AzureOpenAI


@provider
class AzureProvider(ProviderCore):
    def __init__(
        self,
        config,
        api_key=None,
        api_endpoint=None,
        api_version=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.API_KEY = api_key or os.getenv("AZURE_API_KEY")
        self.API_ENDPOINT = api_endpoint
        self.API_VERSION = api_version or os.getenv("AZURE_API_VERSION")

        self._client = AzureOpenAI(
            api_key=self.API_KEY,
            azure_endpoint=self.API_ENDPOINT,
            api_version=self.API_VERSION,
        )

    @staticmethod
    def _provider_config_name():
        return "azure"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(self, request: ChatRequest) -> Any:
        """Generate an AzureOpenAI client"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate an AzureOpenAI client"""

        try:
            return self._client.chat.completions.create(
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=True,
                stream_options={"include_usage": True},
                **request.parameters,
            )
        except openai._exceptions.APIError as e:
            raise ProviderError(str(e))

    async def aparse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
        result = self.parse_response(response=response, **kwargs)
        for chunk in result:
            yield chunk

    def parse_response(self, response: AsyncGenerator, **kwargs) -> Any:
        for chunk in response:
            c = chunk.model_dump()
            if c.get("choices"):
                yield c
