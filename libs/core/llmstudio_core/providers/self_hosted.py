import os
from typing import Any, AsyncGenerator, Coroutine, Generator

import openai
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai import OpenAI


@provider
class SelfHostedProvider(ProviderCore):
    def __init__(
        self,
        config,
        api_key=None,
        base_url=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.API_KEY = api_key or os.getenv("API_KEY")
        self.BASE_URL = base_url or os.getenv("BASE_URL")
        self.has_tools_functions = False

        self._client = OpenAI(
            api_key=self.API_KEY,
            base_url=self.BASE_URL,
        )

    @staticmethod
    def _provider_config_name():
        return "self-hosted"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(
        self, request: ChatRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an OpenAI client"""
        return self.generate_client(request=request)

    def generate_client(self, request: ChatRequest) -> Generator:
        """Generate an OpenAI client"""

        try:
            return self._client.chat.completions.create(
                model=request.model,
                messages=(
                    [{"role": "user", "content": request.chat_input}]
                    if isinstance(request.chat_input, str)
                    else request.chat_input
                ),
                stream=True,
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
