import aiohttp
import requests
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from llmstudio.cli import start_server
from llmstudio.config import ENGINE_HOST, ENGINE_PORT

# RESPONSE_PARSER IMPORTS
from pydantic import BaseModel, ValidationError
import json


class LLM:
    def __init__(self, model_id: str, **kwargs):
        start_server()
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

    def chat(self, input: str, is_stream: bool = False, **kwargs):
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
                "parameters": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_tokens": self.max_tokens,
                },
                **kwargs,
            },
            stream=is_stream,
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()

        if is_stream:
            return self.generate_chat(response)
        else:
            return ChatCompletion(**response.json())

    def generate_chat(self, response):
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield ChatCompletionChunk(**chunk.decode("utf-8"))

    async def async_chat(self, input: str, is_stream=False, **kwargs):
        if is_stream:
            return self.async_stream(input)
        else:
            return await self.async_non_stream(input)

    async def async_non_stream(self, input: str, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
                json={
                    "model": self.model,
                    "api_key": self.api_key,
                    "api_secret": self.api_endpoint,
                    "api_region": self.api_version,
                    "chat_input": input,
                    "is_stream": False,
                    **kwargs,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                return ChatCompletion(**await response.json())

    async def async_stream(self, input: str, **kwargs):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{ENGINE_HOST}:{ENGINE_PORT}/api/engine/chat/{self.provider}",
                json={
                    "model": self.model,
                    "api_key": self.api_key,
                    "api_secret": self.api_endpoint,
                    "api_region": self.api_version,
                    "chat_input": input,
                    "is_stream": True,
                    **kwargs,
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                async for chunk in response.content.iter_any():
                    if chunk:
                        yield ChatCompletionChunk(**await chunk.decode("utf-8"))

    ####################################### RESPONSE PARSER #######################################
    def parse_response(self,json_str: str, response_model: BaseModel):
        try:
            # Trim the string to start at the first curly bracket and end at the last curly bracket
            start_index = json_str.find('{')
            end_index = json_str.rfind('}')
            trimmed_str = json_str[start_index:end_index+1]
            if start_index == -1 or end_index == -1:
                raise ValueError("Failed to find JSON in the response.")
            parsed_response = response_model.model_validate(json.loads(trimmed_str))
            return parsed_response
        except json.JSONDecodeError as e:
            custom_error = Exception(f"JSON string: {trimmed_str}\nFailed to decode JSON string: {str(e)}.")
            return custom_error
        except Exception as e:
            return e
