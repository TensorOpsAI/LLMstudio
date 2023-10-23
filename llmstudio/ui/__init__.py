from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os


def create_ui_app():
    app = FastAPI()

    app.mount(
        "/static",
        StaticFiles(directory=os.path.join("llmstudio", "ui", "build", "static")),
        name="static",
    )
    app.mount(
        "/",
        StaticFiles(directory=os.path.join("llmstudio", "ui", "build"), html=True),
        name="app",
    )

    return app
