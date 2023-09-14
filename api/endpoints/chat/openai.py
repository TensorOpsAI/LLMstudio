from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import openai
from pydantic import BaseModel, Field, validator
import tiktoken

from api.worker.config import celery_app
from api.utils import append_log

end_token = "<END_TOKEN>"

router = APIRouter()


class OpenAIParameters(BaseModel):
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=2048)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(default=0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=0, le=1)


class OpenAIRequest(BaseModel):
    api_key: str
    model_name: str
    chat_input: str
    parameters: Optional[OpenAIParameters] = OpenAIParameters()
    is_stream: Optional[bool] = False

    @validator("model_name", always=True)
    def validate_model_name(cls, value):
        allowed_values = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]
        if value not in allowed_values:
            raise ValueError(f"model_name should be one of {allowed_values}")
        return value


@router.post("/openai")
async def openai_chat_endpoint(data: OpenAIRequest):
    def get_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
        if model_name == "gpt-3.5-turbo":
            return 0.0000015 * input_tokens + 0.000002 * output_tokens
        elif model_name == "gpt-4":
            return 0.00003 * input_tokens + 0.00006 * output_tokens
        elif model_name == "gpt-3.5-turbo-16k":
            return 0.00003 * input_tokens + 0.00004 * output_tokens

    def get_tokens(chat_input: str, model_name: str) -> int:
        tokenizer = tiktoken.encoding_for_model(model_name)
        return len(tokenizer.encode(chat_input))

    def stream(response: dict):
        chat_output = ""
        for chunk in response:
            if (
                chunk["choices"][0]["finish_reason"] != "stop"
                and chunk["choices"][0]["finish_reason"] != "length"
            ):
                chunk_content = chunk["choices"][0]["delta"]["content"]
                chat_output += chunk_content
                yield chunk_content
            else:
                input_tokens = get_tokens(data.chat_input, data.model_name)
                output_tokens = get_tokens(chat_output, data.model_name)
                cost = get_cost(input_tokens, output_tokens, data.model_name)
                yield f"{end_token},{input_tokens},{output_tokens},{cost}"  # json

    openai.api_key = data.api_key

    response = openai.ChatCompletion.create(
        model=data.model_name,
        messages=[
            {
                "role": "user",
                "content": data.chat_input,
            }
        ],
        temperature=data.parameters.temperature,
        max_tokens=data.parameters.max_tokens,
        top_p=data.parameters.top_p,
        frequency_penalty=data.parameters.frequency_penalty,
        presence_penalty=data.parameters.presence_penalty,
        stream=data.is_stream,
    )

    if data.is_stream:
        return StreamingResponse(stream(response))
    else:
        input_tokens = get_tokens(data.chat_input, data.model_name)
        output_tokens = get_tokens(
            response["choices"][0]["message"]["content"], data.model_name
        )

        import random, time

        data = {
            "id": random.randint(0, 1000),
            "chatInput": data.chat_input,
            "chatOutput": response["choices"][0]["message"]["content"],
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "cost": get_cost(input_tokens, output_tokens, data.model_name),
            "timestamp": time.time(),
            "modelName": data.model_name,
            "parameters": data.parameters.dict(),
        }

        append_log(data)
        return data

    # channel_name = "openaichat"
    # task = celery_app.send_task(
    #     "tasks.openai.openai_chat_worker",  # Task's name as a string
    #     args=[
    #         data.api_key,
    #         data.model_name,
    #         data.input,
    #         data.parameters.dict(),
    #         data.stream,
    #         channel_name,
    #     ],
    # )
    # return {"task_id": task.id, "channel_name": channel_name}
