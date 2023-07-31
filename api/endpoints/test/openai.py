from app import app
from flask import request, json
import openai


@app.route("/test/openai", methods=["POST"])
def test_openai():
    openai.api_key = request.json["apiKey"]
    try:
        openai.Model.retrieve(request.json["model"])
        return json.dumps(True)
    except:
        return json.dumps(False)
