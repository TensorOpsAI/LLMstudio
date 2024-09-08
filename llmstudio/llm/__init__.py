import asyncio
from typing import Dict, List, Union

import requests
from openai.types.chat import ChatCompletion
from tqdm.asyncio import tqdm_asyncio

from llmstudio.config import ENGINE_HOST, ENGINE_PORT
from llmstudio.llm.semaphore import DynamicSemaphore
from llmstudio.server import start_server

start_server()


class LLM:
    def __init__(self, model_id: str, **kwargs):
        self.provider, self.model = model_id.split("/")
        self.session_id = kwargs.get("session_id")
        self.api_key = kwargs.get("api_key")
        self.api_endpoint = kwargs.get("api_endpoint")
        self.api_version = kwargs.get("api_version")
        self.base_url = kwargs.get("base_url")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.top_k = kwargs.get("top_k")
        self.max_tokens = kwargs.get("max_tokens")
        self.frequency_penalty = kwargs.get("frequency_penalty")
        self.presence_penalty = kwargs.get("presence_penalty")

    def chat(self, input: str, is_stream: bool = False, retries: int = 0, **kwargs):
        response = requests.post(
            f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
            json={
                "model": self.model,
                "session_id": self.session_id,
                "api_key": self.api_key,
                "api_endpoint": self.api_endpoint,
                "api_version": self.api_version,
                "base_url": self.base_url,
                "chat_input": input,
                "is_stream": is_stream,
                "retries": retries,
                "parameters": {
                    key: value
                    for key, value in {
                        "temperature": kwargs.get("temperature") or self.temperature,
                        "top_p": kwargs.get("top_p") or self.top_p,
                        "top_k": kwargs.get("top_k") or self.top_k,
                        "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                        "max_output_tokens": kwargs.get("max_tokens")
                        or self.max_tokens,
                        "frequency_penalty": kwargs.get("frequency_penalty")
                        or self.frequency_penalty,
                        "presence_penalty": kwargs.get("presence_penalty")
                        or self.presence_penalty,
                    }.items()
                    if value is not None
                },
                **kwargs,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            try:
                error_data = response.json().get("detail", "LLMstudio Engine error")
            except ValueError:
                error_data = response.text
            raise Exception(error_data)

        if is_stream:
            return self.generate_chat(response)
        else:
            return ChatCompletion(**response.json())

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")

    async def async_chat(self, input: str, is_stream=False, retries: int = 0, **kwargs):
        if is_stream:
            return self.async_stream(input, retries=retries, **kwargs)
        else:
            return await self.async_non_stream(input, retries=retries, **kwargs)

    async def chat_coroutine(
        self,
        input: Union[str, List[Dict[str, str]]],
        semaphore,
        retries,
        error_threshold,
        increment,
        verbose,
    ):

        async with semaphore:
            try:
                response = await self.async_non_stream(
                    input, max_tokens=semaphore.get_max_tokens(), retries=retries
                )
                semaphore.update_computed_max_tokens(response.metrics["total_tokens"])
                return response
            except Exception as e:
                semaphore.error_requests += 1
                semaphore.error_requests_since_last_increase += 1
                return e
            finally:
                semaphore.finished_requests += 1
                semaphore.requests_since_last_increase += 1
                semaphore.try_increase_permits(error_threshold, increment)
                if verbose > 0:
                    print(
                        f"Finished requests: {semaphore.finished_requests}/{semaphore.batch_size}"
                    )
                    print(
                        f"Amount of parallel requests being allowed: {semaphore._permits}"
                    )
                    print(f"Max tokens being used: {semaphore.get_max_tokens()}")
                    print(
                        f"Requests finished with an error: {semaphore.error_requests}"
                    )

    async def batch_chat_coroutine(
        self,
        inputs: List[Union[str, List[Dict[str, str]]]],
        coroutines,
        retries,
        error_threshold,
        increment,
        max_tokens,
        verbose,
    ) -> List[str]:

        semaphore = DynamicSemaphore(
            coroutines, len(inputs), given_max_tokens=max_tokens
        )

        if verbose > 0:
            responses = await asyncio.gather(
                *[
                    self.chat_coroutine(
                        input=input,
                        semaphore=semaphore,
                        retries=retries,
                        error_threshold=error_threshold,
                        increment=increment,
                        verbose=verbose,
                    )
                    for input in inputs
                ],
            )
            return responses
        else:
            responses = await tqdm_asyncio.gather(
                *[
                    self.chat_coroutine(
                        input=input,
                        semaphore=semaphore,
                        retries=retries,
                        error_threshold=error_threshold,
                        increment=increment,
                        verbose=verbose,
                    )
                    for input in inputs
                ],
                desc="Getting chat responses: ",
            )
            return responses

    def batch_chat(
        self,
        inputs: List[Union[str, List[Dict[str, str]]]],
        coroutines: int = 20,
        retries: int = 5,
        error_threshold: int = 5,
        increment: int = 5,
        max_tokens=None,
        verbose=0,
    ) -> List[str]:

        if coroutines > len(inputs):
            raise Exception(
                "num_coroutines can not be higher than the amount of inputs."
            )

        responses = asyncio.run(
            self.batch_chat_coroutine(
                inputs,
                coroutines,
                retries,
                error_threshold,
                increment,
                max_tokens,
                verbose,
            )
        )
        return responses

    async def async_non_stream(self, input: str, retries: int, **kwargs):
        response = await asyncio.to_thread(
            requests.post,
            f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
            json={
                "model": self.model,
                "session_id": self.session_id,
                "api_key": self.api_key,
                "api_endpoint": self.api_endpoint,
                "api_version": self.api_version,
                "base_url": self.base_url,
                "chat_input": input,
                "is_stream": False,
                "retries": retries,
                "parameters": {
                    key: value
                    for key, value in {
                        "temperature": kwargs.get("temperature") or self.temperature,
                        "top_p": kwargs.get("top_p") or self.top_p,
                        "top_k": kwargs.get("top_k") or self.top_k,
                        "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                        "max_output_tokens": kwargs.get("max_tokens")
                        or self.max_tokens,
                        "frequency_penalty": kwargs.get("frequency_penalty")
                        or self.frequency_penalty,
                        "presence_penalty": kwargs.get("presence_penalty")
                        or self.presence_penalty,
                    }.items()
                    if value is not None
                },
                **kwargs,
            },
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            try:
                error_data = response.json().get("detail", "LLMstudio Engine error")
            except ValueError:
                error_data = response.text
            raise Exception(error_data)

        return ChatCompletion(**response.json())

    async def async_stream(self, input: str, retries: int, **kwargs):
        response = await asyncio.to_thread(
            requests.post,
            f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
            json={
                "model": self.model,
                "session_id": self.session_id,
                "api_key": self.api_key,
                "api_endpoint": self.api_endpoint,
                "api_version": self.api_version,
                "base_url": self.base_url,
                "chat_input": input,
                "is_stream": True,
                "retries": retries,
                "parameters": {
                    key: value
                    for key, value in {
                        "temperature": kwargs.get("temperature") or self.temperature,
                        "top_p": kwargs.get("top_p") or self.top_p,
                        "top_k": kwargs.get("top_k") or self.top_k,
                        "max_tokens": kwargs.get("max_tokens") or self.max_tokens,
                        "max_output_tokens": kwargs.get("max_tokens")
                        or self.max_tokens,
                        "frequency_penalty": kwargs.get("frequency_penalty")
                        or self.frequency_penalty,
                        "presence_penalty": kwargs.get("presence_penalty")
                        or self.presence_penalty,
                    }.items()
                    if value is not None
                },
                **kwargs,
            },
            headers={"Content-Type": "application/json"},
            stream=True,
        )

        if not response.ok:
            try:
                error_data = response.json().get("detail", "LLMstudio Engine error")
            except ValueError:
                error_data = response.text
            raise Exception(error_data)

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8")
