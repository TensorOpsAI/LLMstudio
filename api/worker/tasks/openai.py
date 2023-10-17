import json

import openai
import tiktoken

from api.worker.config import celery_app, redis_conn


@celery_app.task(name="tasks.openai.openai_chat_worker")
def openai_chat_worker(
    api_key: str,
    model_name: str,
    input: str,
    parameters: dict,
    stream: bool,
    channel_name: str,
):
    def get_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
        if model_name == "gpt-3.5-turbo":
            return 0.0000015 * input_tokens + 0.000002 * output_tokens
        elif model_name == "gpt-4":
            return 0.00003 * input_tokens + 0.00006 * output_tokens

    def get_tokens(input: str, model_name: str) -> int:
        tokenizer = tiktoken.encoding_for_model(model_name)
        return len(tokenizer.encode(input))

    def stream(input: str, response: dict):
        output = ""
        for chunk in response:
            if (
                chunk["choices"][0]["finish_reason"] != "stop"
                and chunk["choices"][0]["finish_reason"] != "length"
            ):
                output += chunk["choices"][0]["delta"]["content"]
                # yield chunk["choices"][0]["delta"]["content"]
                redis_conn.publish(channel_name, chunk["choices"][0]["delta"]["content"])
            else:
                input_tokens = get_tokens(input)
                output_tokens = get_tokens(output)
                cost = get_cost(input_tokens, output_tokens, model_name)
                # yield f"{end_token},{input_tokens},{output_tokens},{cost}"
                redis_conn.publish(
                    channel_name, f"{end_token},{input_tokens},{output_tokens},{cost}"
                )

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        temperature=parameters["temperature"],
        max_tokens=parameters["max_tokens"],
        top_p=parameters["top_p"],
        frequency_penalty=parameters["frequency_penalty"],
        presence_penalty=parameters["presence_penalty"],
        stream=stream,
    )

    if stream:
        # return StreamingResponse(stream(input, response))
        stream(input, response)
    else:
        input_tokens, output_tokens = get_tokens(
            input, response["choices"][0]["message"]["content"]
        )
        cost = get_cost(
            get_tokens(input), get_tokens(response["choices"][0]["message"]["content"])
        )
        # return {
        #     "output": response["choices"][0]["message"]["content"],
        #     "input_tokens": input_tokens,
        #     "output_tokens": output_tokens,
        #     "cost": cost,
        # }
        data = {
            "output": response["choices"][0]["message"]["content"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
        redis_conn.publish(channel_name, json.dumps(data))
