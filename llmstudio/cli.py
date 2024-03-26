import os
import signal
from threading import Thread

import click
import requests

from llmstudio.config import (
    ENGINE_HOST,
    ENGINE_PORT,
    TRACKING_HOST,
    TRACKING_PORT,
    UI_HOST,
    UI_PORT,
)
from llmstudio.engine import run_engine_app
from llmstudio.tracking import run_tracking_app
from llmstudio.ui import run_ui_app


def is_server_running(host, port, path="/health"):
    try:
        response = requests.get(f"http://{host}:{port}{path}")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        pass
    return False


def start_server():
    if not is_server_running(ENGINE_HOST, ENGINE_PORT):
        engine_thread = Thread(target=run_engine_app, daemon=True)
        engine_thread.start()

    if not is_server_running(TRACKING_HOST, TRACKING_PORT):
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

    # Start the engine if it's not already running
    if not is_server_running(ENGINE_HOST, ENGINE_PORT):
        engine_thread = Thread(target=run_engine_app, daemon=True)
        engine_thread.start()
    else:
        print(f"Engine server already running on {ENGINE_HOST}:{ENGINE_PORT}")

    # Start the tracking if it's not already running
    if not is_server_running(TRACKING_HOST, TRACKING_PORT):
        tracking_thread = Thread(target=run_tracking_app, daemon=True)
        tracking_thread.start()
    else:
        print(f"Tracking server already running on {TRACKING_HOST}:{TRACKING_PORT}")

    # Start the UI if requested and not already running
    if ui:
        if not is_server_running(UI_HOST, UI_PORT):
            ui_thread = Thread(target=run_ui_app, daemon=True)
            ui_thread.start()
            ui_thread.join()
        else:
            print(f"UI server already running on {UI_HOST}:{UI_PORT}")

    if engine_thread:
        engine_thread.join()
    if tracking_thread:
        tracking_thread.join()
