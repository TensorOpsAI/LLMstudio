import os
import signal
import socket
from threading import Thread

import click
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
os.environ["LLMSTUDIO_TRACKING_PORT"] = str(assign_port())
os.environ["LLMSTUDIO_UI_PORT"] = str(assign_port())


def is_server_running(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def start_server():
    engine_port = int(os.environ.get("LLMSTUDIO_ENGINE_PORT"))
    tracking_port = int(os.environ.get("LLMSTUDIO_TRACKING_PORT"))

    if not is_server_running("localhost", engine_port):
        engine_thread = Thread(target=run_engine_app, daemon=True)
        engine_thread.start()

    if not is_server_running("localhost", tracking_port):
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

    # Start the engine and UI in separate threads
    if ui:
        ui_thread = Thread(target=run_ui_app)
    engine_thread = Thread(target=run_engine_app)
    tracking_thread = Thread(target=run_tracking_app)

    if ui:
        ui_thread.daemon = True
    engine_thread.daemon = True
    tracking_thread.daemon = True

    if ui:
        ui_thread.start()
    engine_thread.start()
    tracking_thread.start()

    if ui:
        ui_thread.join()
    engine_thread.join()
    tracking_thread.join()
