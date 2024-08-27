import os
import signal

import click

from llmstudio.server import start_server


def handle_shutdown(signum, frame):
    print("Shutting down gracefully...")
    os._exit(0)


@click.group()
def main():
    pass


@main.command()
@click.option("--ui", is_flag=True, help="Start the UI server.")
def server(ui):
    signal.signal(signal.SIGINT, handle_shutdown)

    start_server(ui=ui)

    print("Servers are running. Press CTRL+C to stop.")

    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Shutting down servers...")


if __name__ == "__main__":
    main()
