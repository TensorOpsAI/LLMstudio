from threading import Event, Thread

import requests
from llmstudio_proxy.config import ENGINE_HOST, ENGINE_PORT
from llmstudio_proxy.server import run_proxy_app
from llmstudio_tracker.config import TRACKING_HOST, TRACKING_PORT
from llmstudio_tracker.server import run_tracker_app

_servers_started = False


def is_server_running(host, port, path="/health"):
    try:
        response = requests.get(f"http://{host}:{port}{path}")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        pass
    return False


def start_server_component(host, port, run_func, server_name):
    if not is_server_running(host, port):
        started_event = Event()
        thread = Thread(target=run_func, daemon=True, args=(started_event,))
        thread.start()
        started_event.wait()  # wait for startup, this assumes the event is set somewhere
        return thread
    else:
        print(f"{server_name} server already running on {host}:{port}")
        return None


def setup_servers(engine, tracking):
    global _servers_started
    engine_thread, tracking_thread = None, None
    if _servers_started:
        return engine_thread, tracking_thread

    if engine:
        engine_thread = start_server_component(
            ENGINE_HOST, ENGINE_PORT, run_proxy_app, "Proxy"
        )

    if tracking:
        tracking_thread = start_server_component(
            TRACKING_HOST, TRACKING_PORT, run_tracker_app, "Tracker"
        )

    _servers_started = True
    return engine_thread, tracking_thread


def start_servers(proxy=True, tracker=True):
    global _servers_started
    if not _servers_started:
        setup_servers(proxy, tracker)
