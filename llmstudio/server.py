import threading
from threading import Event

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
        thread = threading.Thread(target=run_func, daemon=True, args=(started_event,))
        thread.start()
        started_event.wait()  # wait for startup, this assumes the event is set somewhere
        return thread
    else:
        print(f"{server_name} server already running on {host}:{port}")
        return None


def setup_servers(engine, tracking, ui):
    global _servers_started
    engine_thread, tracking_thread, ui_thread = None, None, None
    if _servers_started:
        return engine_thread, tracking_thread, ui_thread

    if engine:
        engine_thread = start_server_component(
            ENGINE_HOST, ENGINE_PORT, run_engine_app, "Engine"
        )

    if tracking:
        tracking_thread = start_server_component(
            TRACKING_HOST, TRACKING_PORT, run_tracking_app, "Tracking"
        )

    if ui:
        ui_thread = start_server_component(UI_HOST, UI_PORT, run_ui_app, "UI")

    _servers_started = True
    return engine_thread, tracking_thread, ui_thread


def start_server(engine=True, tracking=True, ui=False):
    global _servers_started
    if not _servers_started:
        setup_servers(engine, tracking, ui)
