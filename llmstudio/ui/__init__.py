import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_ui_app():
    app = FastAPI()

    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "build", "static")),
        name="static",
    )
    app.mount(
        "/",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "build"), html=True),
        name="app",
    )

    return app
