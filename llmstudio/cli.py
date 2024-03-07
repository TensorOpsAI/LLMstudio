import os
import signal
import socket
from threading import Thread

import click
import requests
from dotenv import load_dotenv

from llmstudio.engine import run_engine_app
from llmstudio.tracking import run_tracking_app
from llmstudio.ui import run_ui_app

load_dotenv(os.path.join(os.getcwd(), ".env"))


def assign_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


os.environ["LLMSTUDIO_ENGINE_PORT"] = str(assign_port())
os.environ["NEXT_PUBLIC_LLMSTUDIO_ENGINE_PORT"] = os.environ.get(
    "LLMSTUDIO_ENGINE_PORT"
)
os.environ["LLMSTUDIO_TRACKING_PORT"] = str(assign_port())
os.environ["NEXT_PUBLIC_LLMSTUDIO_TRACKING_PORT"] = os.environ.get(
    "LLMSTUDIO_TRACKING_PORT"
)
os.environ["LLMSTUDIO_UI_PORT"] = str(assign_port())


def is_server_running(host, port, path="/health"):
    try:
        response = requests.get(f"http://{host}:{port}{path}")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        pass
    return False


def start_server():
    engine_port = int(os.environ.get("LLMSTUDIO_ENGINE_PORT"))
    tracking_port = int(os.environ.get("LLMSTUDIO_TRACKING_PORT"))
    engine_host = os.environ.get("LLMSTUDIO_ENGINE_HOST", "localhost")
    tracking_host = os.environ.get("LLMSTUDIO_TRACKING_HOST", "localhost")

    if not is_server_running(engine_host, engine_port):
        engine_thread = Thread(target=run_engine_app, daemon=True)
        engine_thread.start()

    if not is_server_running(tracking_host, tracking_port):
        tracking_thread = Thread(target=run_tracking_app, daemon=True)
        tracking_thread.start()

    def handle_shutdown(signum, frame):
        print("Shutting down gracefully...")
        os._exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)


@click.group()
def main():
    pass


@main.command()
@click.option("--ui", is_flag=True, help="Start the UI server.")
def server(ui):
    def handle_shutdown(signum, frame):
        print("Shutting down gracefully...")
        os._exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_shutdown)

    engine_host = os.getenv("LLMSTUDIO_ENGINE_HOST", "localhost")
    tracking_host = os.getenv("LLMSTUDIO_TRACKING_HOST", "localhost")
    engine_port = int(os.getenv("LLMSTUDIO_ENGINE_PORT"))
    tracking_port = int(os.getenv("LLMSTUDIO_TRACKING_PORT"))

    # Start the engine if it's not already running
    if not is_server_running(engine_host, engine_port):
        engine_thread = Thread(target=run_engine_app, daemon=True)
        engine_thread.start()
    else:
        print(f"Engine server already running on {engine_host}:{engine_port}")

    # Start the tracking if it's not already running
    if not is_server_running(tracking_host, tracking_port):
        tracking_thread = Thread(target=run_tracking_app, daemon=True)
        tracking_thread.start()
    else:
        print(f"Tracking server already running on {tracking_host}:{tracking_port}")

    # Start the UI if requested and not already running
    if ui:
        ui_port = int(os.getenv("LLMSTUDIO_UI_PORT"))
        if not is_server_running("localhost", ui_port):
            ui_thread = Thread(target=run_ui_app, daemon=True)
            ui_thread.start()
            ui_thread.join()
        else:
            print(f"UI server already running on localhost:{ui_port}")

    if engine_thread:
        engine_thread.join()
    if tracking_thread:
        tracking_thread.join()
