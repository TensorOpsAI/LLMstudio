import signal
from threading import Thread

import click
import dotenv

from llmstudio.engine import run_engine_app
from llmstudio.ui import run_ui_app

dotenv.load_dotenv()


@click.group()
def main():
    pass


@main.command()
def server():
    import os

    def handle_shutdown(signum, frame):
        print("Shutting down gracefully...")
        os._exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_shutdown)

    # Start the engine and UI in separate threads
    ui_thread = Thread(target=run_ui_app)
    engine_thread = Thread(target=run_engine_app)

    ui_thread.daemon = True
    engine_thread.daemon = True

    ui_thread.start()
    engine_thread.start()

    ui_thread.join()
    engine_thread.join()
