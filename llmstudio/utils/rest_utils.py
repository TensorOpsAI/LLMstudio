import webbrowser
from threading import Thread

import requests
import uvicorn

from llmstudio.engine import create_app_from_config
from llmstudio.engine.config import EngineConfig, _load_route_config


def is_api_running(url, name) -> bool:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"API {name} is already running")
            return True
    except requests.RequestException as e:
        return False


def run_engine_app(engine_config=EngineConfig()):
    config = _load_route_config(engine_config.config_path)
    engine = create_app_from_config(config)
    print(f"Running {engine_config.api_name} on {engine_config.host}:{engine_config.port}")
    uvicorn.run(
        engine,
        host=engine_config.host,
        port=engine_config.port,
        log_level="critical",
    )


def run_ui_app(ui_server_app, api_name="UI", host="localhost", port=3000):
    print(f"Running {api_name} on {host}:{port}")
    webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(
        ui_server_app,
        host=host,
        port=port,
        log_level="critical",
    )


def run_apis(engine_config=EngineConfig(), ui_server_app=None, serverless=False):
    if engine_config.localhost and not is_api_running(
        engine_config.health_endpoint, engine_config.api_name
    ):
        thread = Thread(target=run_engine_app, args=(engine_config,))
        thread.daemon = True
        thread.start()

    if ui_server_app:
        thread = Thread(target=run_ui_app, args=(ui_server_app,))
        thread.daemon = True
        thread.start()
        thread.join()
