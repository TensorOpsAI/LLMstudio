import os
import signal
import threading

import click
from llmstudio_proxy.server import setup_engine_server


def handle_shutdown(signum, frame):
    print("Shutting down gracefully...")
    os._exit(0)


@click.group()
def main():
    pass


@main.command()
def server():
    signal.signal(signal.SIGINT, handle_shutdown)

    setup_engine_server()

    print("Press CTRL+C to stop.")

    stop_event = threading.Event()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        print("Shutting down server...")


if __name__ == "__main__":
    main()
