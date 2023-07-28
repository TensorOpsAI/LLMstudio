from app import app
from flask import stream_with_context, request, jsonify
import json
import vertexai
from google.oauth2 import service_account
from vertexai.language_models import ChatModel, TextGenerationModel


def get_vertexai_cost(input_tokens, output_tokens, model):
    if model == "text-bison@001":
        return 0.000001 * (input_tokens + output_tokens)
    elif model == "chat-bison@001":
        return 0.0000005 * (input_tokens + output_tokens)


@app.route("/chat/vertexai", methods=["POST"])
def get_vertexai_chat():
    json_credential = json.loads(request.json["apiKey"])
    vertexai.init(
        project=json_credential["project_id"],
        credentials=service_account.Credentials.from_service_account_info(
            json_credential
        ),
    )

    if request.json["model"] == "text-bison@001":
        response = TextGenerationModel.from_pretrained("text-bison@001").predict(
            prompt=request.json["prompt"],
            temperature=request.json["parameters"]["temperature"],
            max_output_tokens=request.json["parameters"]["maxTokens"],
            top_p=request.json["parameters"]["topP"],
            top_k=request.json["parameters"]["topK"],
        )
    elif request.json["model"] == "chat-bison@001":
        response = (
            ChatModel.from_pretrained("chat-bison@001")
            .start_chat()
            .send_message(
                message=request.json["prompt"],
                temperature=request.json["parameters"]["temperature"],
                max_output_tokens=request.json["parameters"]["maxTokens"],
                top_p=request.json["parameters"]["topP"],
                top_k=request.json["parameters"]["topK"],
            )
        )

    return jsonify(
        {
            "output": response.text,
            "input_tokens": len(request.json["prompt"]),
            "output_tokens": len(response.text),
            "cost": get_vertexai_cost(
                len(request.json["prompt"]), len(response.text), request.json["model"]
            ),
        }
    )
