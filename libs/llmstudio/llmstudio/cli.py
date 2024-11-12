import os
import signal
import threading

import click
from llmstudio.server import start_servers


def handle_shutdown(signum, frame):
    print("Shutting down gracefully...")
    os._exit(0)


@click.group()
def main():
    pass


@main.command()
@click.option("--proxy", is_flag=True, help="Start the Proxy server.")
@click.option("--tracker", is_flag=True, help="Start the Tracker server.")
def server(proxy, tracker):
    signal.signal(signal.SIGINT, handle_shutdown)

    start_servers(proxy=proxy, tracker=tracker)

    print("Servers are running. Press CTRL+C to stop.")

    stop_event = threading.Event()
    try:
        stop_event.wait()  # Wait indefinitely until the event is set
    except KeyboardInterrupt:
        print("Shutting down servers...")


if __name__ == "__main__":
    # main()
    server()
