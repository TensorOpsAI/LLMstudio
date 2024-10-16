import threading
from threading import Event

import requests

from llmstudio_proxy.config import (
    ENGINE_HOST,
    ENGINE_PORT,
)

from llmstudio_tracker.config import (
    TRACKING_HOST,
    TRACKING_PORT,
)

from llmstudio_proxy import run_proxy_app
from llmstudio_tracker import run_tracker_app

_proxy_started, _tracker_started = False, False


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


def setup_servers(proxy, tracking):
    global _servers_started
    proxy_thread, tracking_thread = None, None
    if _servers_started:
        return proxy_thread, tracking_thread

    if proxy:
        proxy_thread = start_server_component(
            ENGINE_HOST, ENGINE_PORT, run_proxy_app, "Proxy"
        )

    if tracking:
        tracking_thread = start_server_component(
            TRACKING_HOST, TRACKING_PORT, run_tracking_app, "Tracker"
        )

    _servers_started = True
    return proxy_thread, tracking_thread

def setup_proxy(proxy_host, proxy_port):
    global _proxy_started
    proxy_thread = None
    if _proxy_started:
        return proxy_thread

    proxy_thread = start_server_component(
        proxy_host, proxy_port, run_proxy_app, "proxy"
    )

    _proxy_started = True
    return proxy_thread

def setup_tracker(tracker_host, tracker_port):
    global _tracker_started
    tracker_thread = None
    if _tracker_started:
        return tracker_thread

    tracker_thread = start_server_component(
        tracker_host, tracker_port, run_tracker_app, "Tracker"
    )

    _tracker_started = True
    return tracker_thread


def start_servers(proxy=True, tracker=True):
    global _proxy_started, _tracker_started
    if proxy and not _proxy_started:
        setup_proxy(ENGINE_HOST, ENGINE_PORT)
    
    if tracker and not _tracker_started:
        setup_tracker(TRACKING_HOST, TRACKING_PORT)
