from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import tiktoken

router = APIRouter()
end_token = "<END_TOKEN>"


class Parameters(BaseModel):
    temperature: float
    maxTokens: int
    topP: float
    frequencyPenalty: float
    presencePenalty: float


class OpenAIRequest(BaseModel):
    apiKey: str
    model: str
    prompt: str
    parameters: Parameters
    stream: bool


def get_openai_tokens(input, output, model):
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(input)), len(tokenizer.encode(output))


def get_openai_cost(input_tokens, output_tokens, model):
    if model == "gpt-3.5-turbo":
        return 0.0000015 * input_tokens + 0.000002 * output_tokens
    elif model == "gpt-4":
        return 0.00003 * input_tokens + 0.00006 * output_tokens


@router.post("/openai")
async def get_openai_chat(data: OpenAIRequest):
    openai.api_key = data.apiKey

    response = openai.ChatCompletion.create(
        model=data.model,
        messages=[
            {
                "role": "user",
                "content": data.prompt,
            }
        ],
        temperature=data.parameters.temperature,
        max_tokens=data.parameters.maxTokens,
        top_p=data.parameters.topP,
        frequency_penalty=data.parameters.frequencyPenalty,
        presence_penalty=data.parameters.presencePenalty,
        stream=data.stream,
    )

    if data.stream:

        def streamer():
            output = ""
            for chunk in response:
                if (
                    chunk["choices"][0]["finish_reason"] != "stop"
                    and chunk["choices"][0]["finish_reason"] != "length"
                ):
                    output += chunk["choices"][0]["delta"]["content"]
                    yield chunk["choices"][0]["delta"]["content"]
                else:
                    input_tokens, output_tokens = get_openai_tokens(
                        data.prompt, output, data.model
                    )
                    cost = get_openai_cost(input_tokens, output_tokens, data.model)
                    yield f"{end_token},{input_tokens},{output_tokens},{cost}"

        return StreamingResponse(streamer())

    else:
        input_tokens, output_tokens = get_openai_tokens(
            data.prompt,
            response["choices"][0]["message"]["content"],
            data.model,
        )
        cost = get_openai_cost(input_tokens, output_tokens, data.model)
        return {
            "output": response["choices"][0]["message"]["content"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
