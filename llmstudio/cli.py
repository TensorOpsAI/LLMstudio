import click
import uvicorn

from llmstudio.llm_engine import LlmEngineAPI, create_app_from_config
from llmstudio.llm_engine.config import _load_route_config


@click.group()
def main():
    pass


@main.command()
def server():
    # verficar server is up
    config = _load_route_config(
        "/Users/claudiolemos/Documents/GitHub/LLMStudio/llmstudio/llm_engine/config.yaml"
    )
    llm_engine = create_app_from_config(config)
    uvicorn.run(llm_engine)
