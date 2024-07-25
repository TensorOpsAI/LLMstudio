import threading
import requests
from llmstudio.engine import run_engine_app
from llmstudio.tracking import run_tracking_app
from llmstudio.ui import run_ui_app
from llmstudio.config import (
    ENGINE_HOST,
    ENGINE_PORT,
    TRACKING_HOST,
    TRACKING_PORT,
    UI_HOST,
    UI_PORT,
)

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
        thread = threading.Thread(target=run_func, daemon=True)
        thread.start()
        return thread
    else:
        print(f"{server_name} server already running on {host}:{port}")
        return None


def setup_servers(start_ui=False):
    global _servers_started
    if _servers_started:
        return None, None, None

    engine_thread = start_server_component(
        ENGINE_HOST, ENGINE_PORT, run_engine_app, "Engine"
    )
    tracking_thread = start_server_component(
        TRACKING_HOST, TRACKING_PORT, run_tracking_app, "Tracking"
    )

    ui_thread = None
    if start_ui:
        ui_thread = start_server_component(UI_HOST, UI_PORT, run_ui_app, "UI")

    _servers_started = True
    return engine_thread, tracking_thread, ui_thread


def ensure_servers_running(start_ui=False):
    global _servers_started
    if not _servers_started:
        setup_servers(start_ui=start_ui)
