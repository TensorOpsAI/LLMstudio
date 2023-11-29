import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

UI_HOST = "localhost"
UI_PORT = 3000
UI_URL = f"http://{UI_HOST}:{UI_PORT}"
LOG_LEVEL = "critical"

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

def run_ui_app():
    print(f"Running UI on {UI_HOST}:{UI_PORT}")  # Print statement added
    try:
        ui_server_app = create_ui_app()
        uvicorn.run(ui_server_app, host=UI_HOST, port=UI_PORT, log_level=LOG_LEVEL)
    except Exception as e:
        print(f"Error running the UI app: {e}")