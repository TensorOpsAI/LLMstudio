import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

import time
import uuid

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

import datetime
import hashlib
import hmac
import requests
import json
import struct
import binascii

class BedrockParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class BedrockRequest(ChatRequest):
    bedrock_access_key: Optional[str] = None
    bedrock_secret_key: Optional[str] = None
    bedrock_region: Optional[str] = None
    parameters: Optional[BedrockParameters] = BedrockParameters()
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class BedrockProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.BEDROCK_ACCESS_KEY = os.getenv("BEDROCK_ACCESS_KEY")
        self.BEDROCK_SECRET_KEY = os.getenv("BEDROCK_SECRET_KEY")
        self.BEDROCK_REGION = os.getenv("BEDROCK_REGION")

    def validate_request(self, request: BedrockRequest):
        return BedrockRequest(**request)

    async def generate_client(
        self, request: BedrockRequest
    ) -> Coroutine[Any, Any, Any]:
        """Generate an AWS Bedrock client"""

        # AWS credentials
        access_key = request.bedrock_access_key or self.BEDROCK_ACCESS_KEY
        secret_key = request.bedrock_secret_key or self.BEDROCK_SECRET_KEY
        region = request.bedrock_region or self.BEDROCK_REGION
        service = 'bedrock'

        if access_key is None or secret_key is None or region is None:
            raise HTTPException(status_code=400, detail="AWS credentials were not given or not set in environment variables.")

        # Model host and endpoint
        host = f'bedrock-runtime.{region}.amazonaws.com'
        uri = f'/model/{request.model}/converse-stream'
        endpoint = f'https://{host}{uri}'

        # Create payload
        payload = {
            "messages": self._generate_input_text(request.chat_input),
            "inferenceConfig": {
                "maxTokens": request.parameters.max_tokens,
                "stopSequences": [],
                "temperature": request.parameters.temperature,
                "topP": request.parameters.top_p,
            }
        }

        request_parameters = json.dumps(payload)

        request_headers = self._create_headers(
            method='POST',
            uri=uri,
            host=host,
            region=region,
            service=service,
            access_key=access_key,
            secret_key=secret_key,
            request_parameters=request_parameters,
            content_type='application/json',
            accept='application/json',  # Use 'application/json' for EventStream format
        )

        # Use asyncio.to_thread to run the synchronous requests.post in an async context
        return await asyncio.to_thread(
            requests.post, endpoint, data=request_parameters, headers=request_headers, stream=True
        )

    async def parse_response(
        self, response: Any, **kwargs
    ) -> AsyncGenerator[str, None]:
        
        # This buffer stores incomplete byte chunks
        buffer = b''
        for chunk in response:
            if chunk:
                messages, buffer = self._parse_chunk(chunk, buffer)
                
                for headers, payload in messages:

                    payload = json.loads(payload.decode('utf-8'))
                    
                    # Process the message based on event type
                    event_type = headers.get(':event-type')
                    if event_type == 'messageStart':

                        if payload.get("role") == 'assistant':
                            chunk = self._generate_chat_message_start_chunk(kwargs)
                            print(f'start chunk: {chunk}')
                            yield chunk.model_dump()

                    elif event_type == 'contentBlockDelta':
                        delta = payload.get('delta', {})
                        text = delta.get('text', '')
                        chunk = self._generate_chat_message_content_chunk(kwargs, text)
                        print(f'content chunk: {chunk}')
                        yield chunk.model_dump()

                    elif event_type == 'contentBlockStop':
                        delta = payload.get('delta', {})
                        text = delta.get('text', '')
                        chunk = self._generate_chat_message_content_chunk(kwargs, text)
                        print(f'content chunk: {chunk}')
                        yield chunk.model_dump()

                    elif event_type == 'messageStop':
                        chunk = self._generate_chat_message_stop_chunk(kwargs)
                        print(f'stop chunk: {chunk}')
                        yield chunk.model_dump()
                    elif event_type == 'metadata':
                        #TODO We could maybe do something with the call metadata?
                        pass
                    elif event_type == 'error':
                        #TODO Maybe trow a parsing error? Idk yet.
                        pass
                    else:
                        #TODO For sure we will have different events for tool/function calling.
                        # For those i need to add the logic here.
                        pass

    @staticmethod
    def _sign(key, msg):
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

    @classmethod
    def _get_signature_key(cls, key, date_stamp, region_name, service_name):
        k_date = cls._sign(('AWS4' + key).encode('utf-8'), date_stamp)
        k_region = cls._sign(k_date, region_name)
        k_service = cls._sign(k_region, service_name)
        k_signing = cls._sign(k_service, 'aws4_request')
        return k_signing

    @staticmethod
    def _generate_input_text(chat_input):
        if isinstance(chat_input, str):
            return [
                {
                    "content": [
                        {"text": chat_input}
                    ],
                    "role": "user"
                }
            ]

        elif isinstance(chat_input, list):
            messages = []
            for message in chat_input:
                role = message.get("role")
                content = message.get("content", "")
                if role in ["user", "assistant"]:
                    messages.append({
                        "content": [{"text": content}],
                        "role": role
                    })
            return messages
        else:
            raise HTTPException(status_code=400, detail="Invalid input: chat_input must be a string or a list of messages.")

    @staticmethod
    def _generate_system_message(chat_input):
        # TODO
        # Need to get access to Anthropic to try models that support system message.
        return None

    @classmethod
    def _create_headers(cls, method, uri, host, region, service, access_key, secret_key, request_parameters, content_type, accept):
        # Timestamp and date
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')  # Format: YYYYMMDD'T'HHMMSS'Z'
        date_stamp = t.strftime('%Y%m%d')  # Date without time for credential scope

        # Payload hash
        payload_hash = hashlib.sha256(request_parameters.encode('utf-8')).hexdigest()

        # Headers (lowercase)
        headers = {
            'content-type': content_type,
            'host': host,
            'x-amz-date': amz_date,
            'x-amz-content-sha256': payload_hash,
            'accept': accept
        }

        # Canonical headers and signed headers
        sorted_header_keys = sorted(headers.keys())
        canonical_headers = ''
        signed_headers = ''
        for key in sorted_header_keys:
            canonical_headers += f"{key}:{headers[key]}\n"
            signed_headers += f"{key};"
        signed_headers = signed_headers.rstrip(';')

        # Create canonical request
        query_string = ''
        canonical_request = '\n'.join([
            method,
            uri,
            query_string,
            canonical_headers,
            signed_headers,
            payload_hash
        ])

        # Algorithm and credential scope
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"

        # Create string to sign
        string_to_sign = '\n'.join([
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        ])

        # Calculate the signature
        signing_key = cls._get_signature_key(secret_key, date_stamp, region, service)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        # Create authorization header
        authorization_header = (
            f"{algorithm} Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # Add authorization header to headers
        headers['Authorization'] = authorization_header

        # Remove 'host' from the headers for the actual request (requests library adds it automatically)
        request_headers = {k: v for k, v in headers.items() if k.lower() != 'host'}

        return request_headers
    
    @classmethod
    def _parse_event_stream(cls, buffer):
        messages = []
        offset = 0
        while offset + 12 <= len(buffer):
            total_length = struct.unpack('>I', buffer[offset:offset+4])[0]
            if offset + total_length > len(buffer):
                # Not enough data to read the whole message
                break
            message = buffer[offset:offset+total_length]
            headers, payload = cls._parse_event_stream_message(message)
            messages.append((headers, payload))
            offset += total_length
        remaining_buffer = buffer[offset:]
        return messages, remaining_buffer

    @staticmethod
    def _parse_event_stream_message(message):
        total_length = struct.unpack('>I', message[0:4])[0]
        headers_length = struct.unpack('>I', message[4:8])[0]
        prelude_crc = struct.unpack('>I', message[8:12])[0]
        prelude = message[0:8]
        computed_prelude_crc = binascii.crc32(prelude) & 0xffffffff
        if prelude_crc != computed_prelude_crc:
            raise ValueError('Prelude CRC mismatch')

        # Parse headers
        headers = {}
        pos = 12  # Starting position after prelude and prelude CRC
        headers_end = pos + headers_length
        while pos < headers_end:
            name_len = message[pos]
            pos += 1
            name = message[pos:pos+name_len].decode('utf-8')
            pos += name_len
            value_type = message[pos]
            pos += 1
            if value_type == 7:  # String
                value_len = struct.unpack('>H', message[pos:pos+2])[0]
                pos += 2
                value = message[pos:pos+value_len].decode('utf-8')
                pos += value_len
            else:
                # Handle other value types if necessary
                value = None
            headers[name] = value

        # Payload
        payload = message[headers_end:-4]  # Exclude the Message CRC at the end

        # Verify message CRC
        message_crc = struct.unpack('>I', message[-4:])[0]
        computed_message_crc = binascii.crc32(message[:-4]) & 0xffffffff
        if message_crc != computed_message_crc:
            raise ValueError('Message CRC mismatch')

        return headers, payload

    @classmethod
    def _parse_chunk(cls, chunk, buffer):
        buffer += chunk  # Add the new chunk to the buffer
        messages = []
        while True:
            parsed_messages, buffer = cls._parse_event_stream(buffer)
            if not parsed_messages:
                break  # Wait for more data
            messages.extend(parsed_messages)
        return messages, buffer
    
    @staticmethod
    def _generate_chat_message_start_chunk(kwargs):
        return ChatCompletionChunk(
            id=str(uuid.uuid4()), 
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content='', 
                        function_call=None, 
                        role='assistant', 
                        tool_calls=None, 
                        refusal=None
                        ), 
                        finish_reason=None, 
                        index=0, 
                        logprobs=None
                        )
                    ], 
                    created=int(time.time()), 
                    model=kwargs.get("request").model, 
                    object='chat.completion.chunk', 
                    system_fingerprint=None, 
                    usage=None
                    )
    
    @staticmethod
    def _generate_chat_message_content_chunk(kwargs, text):

        return ChatCompletionChunk(
            id=str(uuid.uuid4()), 
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=text, 
                        function_call=None, 
                        role=None, 
                        tool_calls=None
                        ), 
                    finish_reason=None, 
                    index=0, 
                    logprobs=None
                    )
                    ], 
                    created=int(time.time()), 
                    model=kwargs.get("request").model, 
                    object='chat.completion.chunk', 
                    system_fingerprint=None, 
                    usage=None
                    )
    
    @staticmethod
    def _generate_chat_message_stop_chunk(kwargs):
        return ChatCompletionChunk(
            id=str(uuid.uuid4()), 
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=None, 
                        function_call=None, 
                        role=None, 
                        tool_calls=None
                        ), 
                    finish_reason='stop', 
                    index=0, 
                    logprobs=None
                    )
                    ], 
                    created=int(time.time()), 
                    model=kwargs.get("request").model,
                    object='chat.completion.chunk', 
                    system_fingerprint=None, 
                    usage=None
                    )
    
    