from app import app
from flask import stream_with_context, request, jsonify
import json
import vertexai
from google.oauth2 import service_account
from vertexai.language_models import ChatModel, TextGenerationModel


def get_palm_cost(input_tokens, output_tokens, model):
    if model == "text-bison@001":
        return 0.000001 * (input_tokens + output_tokens)
    elif model == "chat-bison@001":
        return 0.0000005 * (input_tokens + output_tokens)


@app.route("/chat/palm", methods=["POST"])
def get_palm_chat():
    vertexai.init(
        credentials=service_account.Credentials.from_service_account_info(
            json.loads(json.dumps(request.json["apiKey"]))
        ),
    )

    if request.json["model"] == "text-bison@001":
        response = TextGenerationModel.from_pretrained("text-bison@001").predict(
            prompt=request.json["prompt"],
            temperature=request.json["temperature"],
            max_output_tokens=request.json["maximumLength"],
            top_p=request.json["topP"],
            top_k=request.json["topK"],
        )
    elif request.json["model"] == "chat-bison@001":
        response = (
            ChatModel.from_pretrained("chat-bison@001")
            .start_chat()
            .send_message(
                message=request.json["prompt"],
                temperature=request.json["temperature"],
                max_output_tokens=request.json["maximumLength"],
                top_p=request.json["topP"],
                top_k=request.json["topK"],
            )
        )

    return jsonify(
        {
            "output": response.text,
            "input_tokens": len(request.json["prompt"]),
            "output_tokens": len(response.text),
            "cost": get_palm_cost(
                len(request.json["prompt"]), len(response.text), request.json["model"]
            ),
        }
    )
