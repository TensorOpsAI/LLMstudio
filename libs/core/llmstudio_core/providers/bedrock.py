import asyncio
import base64
import json
import os
import struct
import time
import uuid
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Union

import requests
from aws_requests_auth.aws_auth import AWSRequestsAuth
from fastapi import HTTPException
from llmstudio_core.exceptions import ProviderError
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, provider
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


@provider
class BedrockProvider(ProviderCore):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.access_key = (
            self.access_key if self.access_key else os.getenv("BEDROCK_ACCESS_KEY")
        )
        self.secret_key = (
            self.secret_key if self.secret_key else os.getenv("BEDROCK_SECRET_KEY")
        )
        self.region = self.region if self.region else os.getenv("BEDROCK_REGION")

    @staticmethod
    def _provider_config_name():
        return "bedrock"

    def validate_request(self, request: ChatRequest):
        return ChatRequest(**request)

    async def agenerate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Any]:
        """Generate an AWS Bedrock client"""
        return await asyncio.to_thread(self.generate_client, request)

    def generate_client(self, request: ChatRequest) -> Coroutine[Any, Any, Generator]:
        """Generate an AWS Bedrock client"""
        try:

            host = f"bedrock-runtime.{self.region}.amazonaws.com"
            endpoint = f"https://{host}"
            uri = f"/model/{request.model}/invoke-with-response-stream"
            url = f"{endpoint}{uri}"
            service = "bedrock"

            if (
                self.access_key is None
                or self.secret_key is None
                or self.region is None
            ):
                raise HTTPException(
                    status_code=400,
                    detail="AWS credentials were not given or not set in environment variables.",
                )

            auth = AWSRequestsAuth(
                aws_access_key=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_host=host,
                aws_region=self.region,
                aws_service=service,
            )

            request_body = {
                "messages": self._generate_input_text(request.chat_input),
            }
            request_body.update(request.parameters)
            headers = {"Content-Type": "application/json"}

            return requests.post(
                url,
                data=json.dumps(request_body),
                headers=headers,
                auth=auth,
                stream=True,
            )
        except Exception as e:
            raise ProviderError(str(e))

    async def aparse_response(
        self, response: Any, **kwargs
    ) -> AsyncGenerator[str, None]:
        iterator = await asyncio.to_thread(
            self.parse_response, response=response, **kwargs
        )
        for item in iterator:
            yield item

    def parse_response(self, response: AsyncGenerator[Any, None], **kwargs) -> Any:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                parsed_chunks = self._parse_event_stream(chunk)
                for parsed_chunk in parsed_chunks:
                    if parsed_chunk["headers"][":event-type"] == "chunk":
                        if (
                            parsed_chunk["payload"]["bytes_decoded"]["type"]
                            == "message_start"
                        ):
                            first_chunk = ChatCompletionChunk(
                                id=parsed_chunk["payload"]["bytes_decoded"]["message"][
                                    "id"
                                ],
                                choices=[
                                    Choice(
                                        delta=ChoiceDelta(
                                            content="",
                                            function_call=None,
                                            role="assistant",
                                            tool_calls=None,
                                            refusal=None,
                                        ),
                                        finish_reason=None,
                                        index=0,
                                        logprobs=None,
                                    )
                                ],
                                created=int(time.time()),
                                model=kwargs.get("request").model,
                                object="chat.completion.chunk",
                                system_fingerprint=None,
                                usage=None,
                            )

                            yield first_chunk.model_dump()

                        elif (
                            parsed_chunk["payload"]["bytes_decoded"]["type"]
                            == "content_block_delta"
                        ):
                            chunk = ChatCompletionChunk(
                                id=str(uuid.uuid4()),
                                choices=[
                                    Choice(
                                        delta=ChoiceDelta(
                                            content=parsed_chunk["payload"][
                                                "bytes_decoded"
                                            ]["delta"]["text"],
                                            function_call=None,
                                            role=None,
                                            tool_calls=None,
                                        ),
                                        finish_reason=None,
                                        index=0,
                                        logprobs=None,
                                    )
                                ],
                                created=int(time.time()),
                                model=kwargs.get("request").model,
                                object="chat.completion.chunk",
                                system_fingerprint=None,
                                usage=None,
                            )
                            yield chunk.model_dump()

                        elif (
                            parsed_chunk["payload"]["bytes_decoded"]["type"]
                            == "message_stop"
                        ):
                            chunk = ChatCompletionChunk(
                                id=str(uuid.uuid4()),
                                choices=[
                                    Choice(
                                        delta=ChoiceDelta(
                                            content=None,
                                            function_call=None,
                                            role=None,
                                            tool_calls=None,
                                        ),
                                        finish_reason="stop",
                                        index=0,
                                        logprobs=None,
                                    )
                                ],
                                created=int(time.time()),
                                model=kwargs.get("request").model,
                                object="chat.completion.chunk",
                                system_fingerprint=None,
                                usage=None,
                            )
                            yield chunk.model_dump()

    @staticmethod
    def _generate_input_text(
        chat_input: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, Union[List[Dict[str, str]], str]]]:
        """
        Generate input text for the Bedrock API based on the provided chat input.

        Args:
            chat_input (Union[str, List[Dict[str, str]]]): The input text or a list of message dictionaries.

        Returns:
            List[Dict[str, Union[List[Dict[str, str]], str]]]: A list of formatted messages for the Bedrock API.

        Raises:
            HTTPException: If the input is invalid.
        """
        if isinstance(chat_input, str):
            return [
                {
                    "role": "user",
                    "content": chat_input,
                }
            ]

        elif isinstance(chat_input, list):
            messages = []
            for message in chat_input:
                role = message.get("role")
                content = message.get("content", "")
                if role in ["user", "assistant"]:
                    messages.append({"role": role, "content": content})
            return messages

    @staticmethod
    def _parse_event_stream(data):
        offset = 0
        messages = []

        while offset < len(data):
            if len(data) - offset < 16:
                print("Insufficient data for message prelude.")
                break

            # Read total length and headers length
            total_length, headers_length = struct.unpack(
                ">II", data[offset : offset + 8]
            )
            prelude_crc = struct.unpack(">I", data[offset + 8 : offset + 12])[0]
            prelude = data[offset : offset + 8]

            # Validate total_length
            if total_length < 16 or (offset + total_length) > len(data):
                print("Invalid total length.")
                break

            # Read headers
            headers_data = data[offset + 12 : offset + 12 + headers_length]
            headers = {}
            i = 0
            while i < len(headers_data):
                # Header name length
                name_len = headers_data[i]
                i += 1
                name = headers_data[i : i + name_len].decode("utf-8")
                i += name_len
                # Header value type
                value_type = headers_data[i]
                i += 1

                # Header value
                if value_type == 7:  # String
                    value_len = struct.unpack(">H", headers_data[i : i + 2])[0]
                    i += 2
                    value = headers_data[i : i + value_len].decode("utf-8")
                    i += value_len
                else:
                    print(f"Unsupported header value type: {value_type}")
                    break

                headers[name] = value

            # Read payload
            payload_start = offset + 12 + headers_length
            payload_end = offset + total_length - 4  # Exclude message CRC
            payload = data[payload_start:payload_end]
            message_crc = struct.unpack(
                ">I", data[payload_end : offset + total_length]
            )[0]

            # Attempt to decode payload
            try:
                payload_str = payload.decode("utf-8")
                payload_json = json.loads(payload_str)

                # Decode base64 'bytes' field if present
                if "bytes" in payload_json:
                    decoded_bytes = base64.b64decode(payload_json["bytes"]).decode(
                        "utf-8"
                    )
                    payload_json["bytes_decoded"] = json.loads(decoded_bytes)
            except Exception as e:
                print(f"Error decoding payload: {e}")
                payload_json = payload.hex()

            messages.append(
                {
                    "headers": headers,
                    "payload": payload_json,
                }
            )

            offset += total_length

        return messages
