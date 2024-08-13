import asyncio
import os
from typing import Any, AsyncGenerator, Coroutine, Dict, Generator, List, Optional

import openai
from fastapi import HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from llmstudio.engine.providers.provider import ChatRequest, Provider, provider
import json
import uuid
import time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


class AzureLlamaParameters(BaseModel):  # Renamed from AzureParameters
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class AzureLlamaRequest(ChatRequest):  # Renamed from AzureRequest
    api_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    parameters: Optional[AzureLlamaParameters] = AzureLlamaParameters()  # Updated reference
    tools: Optional[List[Dict[str, Any]]] = None
    chat_input: Any
    response_format: Optional[Dict[str, str]] = None


@provider
class AzureLlamaProvider(Provider):
    def __init__(self, config):
        super().__init__(config)
        self.API_KEY = os.getenv("AZURE_API_KEY")
        self.BASE_URL = os.getenv("AZURE_BASE_URL")
        self.request = None

    def validate_request(self, request: AzureLlamaRequest):  # Updated reference
        return AzureLlamaRequest(**request)  # Updated reference

    async def generate_client(
        self, request: AzureLlamaRequest
    ) -> Coroutine[Any, Any, Generator]:
        """Generate an AzureOpenAI client"""

        self.request = request

        try:
            
            # 1. Initialize Client
            client = OpenAI(
                api_key=request.api_key or self.API_KEY,
                base_url=request.base_url or self.BASE_URL,
            )
            
            # 2. Create Llama Prompt.
            user_message = self.convert_to_openai_format(request.chat_input)
            content = """<|begin_of_text|>"""
            content = self.add_system_message(user_message, content, request.tools)
            # content = self.add_tool_instructions(content, request.tools)
            content = self.add_conversation(user_message,content)
            llama_message = [{'role': 'user', 'content': content}]
            
            return await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=llama_message,
                stream=True,
            )

        except openai._exceptions.APIError as e:
            raise HTTPException(status_code=e.status_code, detail=e.response.json())

    async def parse_response(
        self, response: AsyncGenerator, **kwargs
    ) -> AsyncGenerator[str, None]:
                    
        if self.request.tools:
            async for chunk in self.handle_tool_response(response, **kwargs):
                yield chunk
        else:
            for chunk in response:
                yield chunk.model_dump()

    async def handle_tool_response(
            self, response: AsyncGenerator, **kwargs
        ) -> AsyncGenerator[str, None]:
            saving = False
            function_name = ""
            function_response = False
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    if '§' in content:
                        if saving:
                            
                            # End of function call, stop saving
                            saving = False
                            function_response = True
                            function_name += content.split('§')[0]
                            # First chunk
                            chunk = ChatCompletionChunk(
                                id=str(uuid.uuid4()),
                                choices=[
                                    Choice(
                                        delta=ChoiceDelta(
                                            role="assistant",
                                            tool_calls=[ChoiceDeltaToolCall(
                                                index=0, 
                                                id='call_' + 'testid1234',
                                                function=ChoiceDeltaToolCallFunction(
                                                    name=function_name,
                                                    arguments="",
                                                    type="function"
                                                )
                                            )],        
                                        ),
                                        finish_reason=None,
                                        index=0,
                                    )
                                ],
                                created=int(time.time()),
                                model=kwargs.get("request").model,
                                object="chat.completion.chunk",
                            ).model_dump()
                            yield chunk

                            function_name = ""
                        else:
                            # Start of function call, start saving
                            saving = True
                            function_name += content.split('§')[1]
                    elif saving:
                        function_name += content
                    # Check if finish_reason is 'stop'
                    elif function_response:
                        if chunk.choices[0].finish_reason == 'stop':
                            chunk = ChatCompletionChunk(
                                id=str(uuid.uuid4()),
                                choices=[
                                    Choice(
                                        delta=ChoiceDelta(), finish_reason="tool_calls", index=0
                                    )
                                ],
                                created=int(time.time()),
                                model=kwargs.get("request").model,
                                object="chat.completion.chunk",
                            ).model_dump()
                            yield chunk
                        else:
                            chunk = ChatCompletionChunk(
                                    id=str(uuid.uuid4()),
                                    choices=[
                                        Choice(
                                            delta=ChoiceDelta(
                                                tool_calls=[ChoiceDeltaToolCall(
                                                    index=0, 
                                                    function=ChoiceDeltaToolCallFunction(
                                                        arguments=content,
                                                    )
                                                )],        
                                            ),
                                            finish_reason=None,
                                            index=0,
                                        )
                                    ],
                                    created=int(time.time()),
                                    model=kwargs.get("request").model,
                                    object="chat.completion.chunk",
                                ).model_dump()
                            yield chunk
                    else:
                        yield chunk.model_dump()

    def convert_to_openai_format(self, message):
        # Check if the input is a simple string
        if isinstance(message, str):
            # Convert the string into the OpenAI message format
            openai_message = [{'role': 'user', 'content': message}]
            return openai_message
        
        # If the message is already in OpenAI format, we can handle it later
        return message

    def add_system_message(self, openai_message, llama_message, tools):
    
        system_message = ""
        # Iterate over each message in the OpenAI format
        system_message_found = False
        for message in openai_message:
            # Check if the role is 'system' and the content is not None
            if message['role'] == 'system' and message['content'] is not None:
                system_message_found = True
                # Create the formatted system message for LLaMA 3.1
                system_message = f"""
<|start_header_id|>system<|end_header_id|>
{message['content']}

"""
        if not system_message_found:
            # Default system message if no system message is found
            system_message = """
<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.

"""
        
        # Constructing the tool prompt
        if tools:
            system_message = system_message + self.add_tool_instructions(tools)
        
        end_tag = "\n<|eot_id|>"
        return llama_message + system_message + end_tag
    
    def add_tool_instructions(self,tools):
        
        # Constructing the tool prompt
        tool_prompt = """
You have access to the following tools:
"""

        for tool in tools:
            if tool['type'] == 'function':
                func = tool['function']
                tool_prompt += f"Use the function '{func['name']}' to '{func['description']}':\n"
                # Formatting the parameters to show the user how to use them
                params_info = json.dumps(func['parameters'], indent=4)
                tool_prompt += f"Parameters format:\n{params_info}\n\n"

        tool_prompt += """
If you choose to call a function, ONLY reply in the following format with no prefix or suffix:
§function_name§{{"param_name": "param_value"}}

Reminder:
- Function calls MUST follow the specified format.
- Only call one function at a time.
- NEVER call more than one function at a time.
- Required parameters MUST be specified.
- Put the entire function call reply on one line.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls.
- If you have already called a tool and got the response for the users question please reply with the response.
"""

        return tool_prompt
    
    def add_conversation(self, openai_message, llama_message):
        # Iterate over each message in the OpenAI message list
        for message in openai_message:
            # Skip system messages
            if message['role'] == 'system':
                continue
            
            # Check if the message has tool calls
            elif 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    function_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']
                    llama_message += f"""
<|start_header_id|>assistant<|end_header_id|>
<function={function_name}>{arguments}</function>
<|eom_id|>
        """
        
            # Check if the message has tool responses
            elif 'tool_call_id' in message:
                tool_response = message['content']
                llama_message += f"""
<|start_header_id|>ipython<|end_header_id|>
{tool_response}
<|eot_id|>
        """
            
            # Check if it is an assistant call
            elif message['role'] == 'assistant' and message['content'] is not None:
                llama_message += f"""
<|start_header_id|>assistant<|end_header_id|>
{message['content']}
<|eot_id|>
            """
            
            # Check if it is an assistant call
            elif message['role'] == 'user' and message['content'] is not None:
                llama_message += f"""
<|start_header_id|>user<|end_header_id|>
{message['content']}
<|eot_id|>
            """
        
        return llama_message