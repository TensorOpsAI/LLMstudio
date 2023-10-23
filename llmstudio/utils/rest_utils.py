import requests
from llmstudio.engine.config import _load_route_config, LLMEngineConfig
from llmstudio.engine import create_app_from_config
import uvicorn
from threading import Thread


def is_api_running(url, name) -> bool:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"API {name} is already running")
            return True
    except requests.RequestException as e:
        return False


def run_llmengine_app(llm_engine_config=LLMEngineConfig()):
    config = _load_route_config(llm_engine_config.config_path)
    llm_engine = create_app_from_config(config)
    print(
        f"Running {llm_engine_config.api_name} on {llm_engine_config.host}:{llm_engine_config.port}"
    )
    uvicorn.run(
        llm_engine,
        host=llm_engine_config.host,
        port=llm_engine_config.port,
        log_level="critical",
    )


def run_ui_app(ui_server_app):
    uvicorn.run(
        ui_server_app,
        host="localhost",
        port=3000,
        log_level="critical",
    )


def run_apis(llm_engine_config=LLMEngineConfig(), ui_server_app=None, serverless=False):
    if llm_engine_config.localhost and not is_api_running(
        llm_engine_config.health_endpoint, llm_engine_config.api_name
    ):
        thread = Thread(target=run_llmengine_app, args=(llm_engine_config,))
        thread.daemon = True
        thread.start()

    if ui_server_app:
        thread = Thread(target=run_ui_app, args=(ui_server_app,))
        thread.daemon = True
        thread.start()
        thread.join()
