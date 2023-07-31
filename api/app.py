from flask import Flask
from flask_cors import CORS

app = Flask(
    __name__,
    static_folder="../build",
    static_url_path="/",
)
CORS(app)


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file("index.html")


@app.route("/")
def index():
    return app.send_static_file("index.html")


import endpoints.export
import endpoints.chat.openai
import endpoints.chat.vertexai
import endpoints.test.openai
import endpoints.test.vertexai
