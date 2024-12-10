import asyncio
import json
from typing import Any, Coroutine, Dict, Optional, Union

import requests
from llmstudio_core.providers.provider import Provider
from llmstudio_proxy.server import is_server_running
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel


class ProxyConfig(BaseModel):
    host: Optional[str] = None
    port: Optional[str] = None
    url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.url is None:
            if self.host is not None and self.port is not None:
                self.url = f"http://{self.host}:{self.port}"
            else:
                raise ValueError(
                    "You must provide either both 'host' and 'port', or 'url'."
                )


class LLMProxyProvider(Provider):
    def __init__(self, provider: str, proxy_config: ProxyConfig):
        self.provider = provider
        self.engine_url = proxy_config.url

        if is_server_running(url=self.engine_url):
            print(f"Connected to LLMStudio Proxy @ {self.engine_url}")
        else:
            raise Exception(f"LLMStudio Proxy is not running @ {self.engine_url}")

    @staticmethod
    def _provider_config_name():
        raise "proxy"

    def chat(
        self,
        chat_input: str,
        model: str,
        is_stream: bool = False,
        retries: int = 0,
        parameters: Dict = {},
        **kwargs,
    ) -> Union[ChatCompletion]:
        response = requests.post(
            f"{self.engine_url}/api/engine/chat/{self.provider}",
            json={
                "chat_input": chat_input,
                "model": model,
                "is_stream": is_stream,
                "retries": retries,
                "parameters": parameters,
                **kwargs,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            error_data = response.text
            raise Exception(error_data)

        if is_stream:
            return self.generate_chat(response)
        else:
            return ChatCompletion(**response.json())

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield ChatCompletionChunk(**json.loads(chunk.decode("utf-8")))

    async def achat(
        self,
        chat_input: Any,
        model: str,
        is_stream: bool | None = False,
        retries: int | None = 0,
        parameters: Dict | None = {},
        **kwargs,
    ) -> Coroutine[Any, Any, Coroutine[Any, Any, ChatCompletionChunk | ChatCompletion]]:
        if is_stream:
            return self.async_stream(
                model=model,
                chat_input=chat_input,
                retries=retries,
                parameters=parameters,
            )
        else:
            return await self.async_non_stream(
                model=model,
                chat_input=chat_input,
                retries=retries,
                parameters=parameters,
            )

    async def async_non_stream(
        self, model: str, chat_input: str, retries: int, parameters, **kwargs
    ):
        response = await asyncio.to_thread(
            requests.post,
            f"{self.engine_url}/api/engine/chat/{self.provider}",
            json={
                "chat_input": chat_input,
                "model": model,
                "is_stream": False,
                "retries": retries,
                "parameters": parameters,
                **kwargs,
            },
            headers={"Content-Type": "application/json"},
        )
        if not response.ok:
            error_data = response.text
            raise Exception(error_data)

        return ChatCompletion(**response.json())

    async def async_stream(
        self, model: str, chat_input: str, retries: int, parameters, **kwargs
    ):
        response = await asyncio.to_thread(
            requests.post,
            f"{self.engine_url}/api/engine/chat/{self.provider}",
            json={
                "chat_input": chat_input,
                "model": model,
                "is_stream": True,
                "retries": retries,
                "parameters": parameters,
                **kwargs,
            },
            stream=True,
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            error_data = response.text
            raise Exception(error_data)

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield ChatCompletionChunk(**json.loads(chunk.decode("utf-8")))
