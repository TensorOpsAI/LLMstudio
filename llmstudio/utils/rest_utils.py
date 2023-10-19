import requests
from ..llm_engine.config import _load_route_config, LLMEngineConfig
from ..llm_engine import create_app_from_config
import uvicorn
from threading import Thread

def is_api_running(url) -> bool:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True
    except requests.RequestException as e:
        return False
    
def run_llmengine_app(llm_engine_config = LLMEngineConfig()):
    config = _load_route_config(llm_engine_config.config_path)
    llm_engine = create_app_from_config(config)
    print(f"Running LLMEngineAPI on {llm_engine_config.host}:{llm_engine_config.port}")
    uvicorn.run(llm_engine, host=llm_engine_config.host, port=llm_engine_config.port)


def run_apis(llm_engine_config = LLMEngineConfig()):
    threads = []
    if llm_engine_config.localhost and not is_api_running(llm_engine_config.health_endpoint):
        thread = Thread(target = run_llmengine_app, args = (llm_engine_config,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

