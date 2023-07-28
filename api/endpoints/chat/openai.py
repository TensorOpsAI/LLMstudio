from app import app
from flask import stream_with_context, request, Response, jsonify
import openai
import tiktoken

end_token = "<END_TOKEN>"


def get_openai_tokens(input, output, model):
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(input)), len(tokenizer.encode(output))


def get_openai_cost(input_tokens, output_tokens, model):
    if model == "gpt-3.5-turbo":
        return 0.0000015 * input_tokens + 0.000002 * output_tokens
    elif model == "gpt-4":
        return 0.00003 * input_tokens + 0.00006 * output_tokens


@app.route("/chat/openai", methods=["POST"])
def get_openai_chat():
    openai.api_key = request.json["apiKey"]

    response = openai.ChatCompletion.create(
        model=request.json["model"],
        messages=[
            {
                "role": "user",
                "content": request.json["prompt"],
            }
        ],
        temperature=request.json["parameters"]["temperature"],
        max_tokens=request.json["parameters"]["maxTokens"],
        top_p=request.json["parameters"]["topP"],
        frequency_penalty=request.json["parameters"]["frequencyPenalty"],
        presence_penalty=request.json["parameters"]["presencePenalty"],
        stream=request.json["stream"],
    )

    if request.json["stream"]:

        def stream():
            output = ""
            for chunk in response:
                if chunk["choices"][0]["finish_reason"] != "stop":
                    output += chunk["choices"][0]["delta"]["content"]
                    yield chunk["choices"][0]["delta"]["content"]
                else:
                    input_tokens, output_tokens = get_openai_tokens(
                        request.json["prompt"], output, request.json["model"]
                    )
                    cost = get_openai_cost(
                        input_tokens, output_tokens, request.json["model"]
                    )
                    yield f"{end_token},{input_tokens},{output_tokens},{cost}"

        return Response(stream_with_context(stream()))
    elif not request.json["stream"]:
        input_tokens, output_tokens = get_openai_tokens(
            request.json["prompt"],
            response["choices"][0]["message"]["content"],
            request.json["model"],
        )
        cost = get_openai_cost(input_tokens, output_tokens, request.json["model"])
        return jsonify(
            {
                "output": response["choices"][0]["message"]["content"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }
        )
