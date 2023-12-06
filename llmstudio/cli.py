import os
import signal
from threading import Thread

import click
from dotenv import load_dotenv

from llmstudio.engine import run_engine_app
from llmstudio.tracking import run_tracking_app
from llmstudio.ui import run_ui_app

load_dotenv(os.path.join(os.getcwd(), ".env"))


@click.group()
def main():
    pass


@main.command()
@click.option("--ui", is_flag=True, help="Start the UI server.")
def server(ui):
    import os

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
