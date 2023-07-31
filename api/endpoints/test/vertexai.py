from app import app
from flask import request, json
import vertexai
from google.oauth2 import service_account


@app.route("/test/vertexai", methods=["POST"])
def test_vertexai():
    try:
        json_credential = json.loads(request.json["apiKey"])
        vertexai.init(
            project=json_credential["project_id"],
            credentials=service_account.Credentials.from_service_account_info(
                json_credential
            ),
        )
        return json.dumps(True)
    except:
        return json.dumps(False)
