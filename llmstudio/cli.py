import click

from llmstudio.engine.config import LLMEngineConfig
from llmstudio.ui import create_ui_app
from llmstudio.utils.rest_utils import run_apis


@click.group()
def main():
    pass


@main.command()
def server(llm_engine_config=LLMEngineConfig()):
    run_apis(
        llm_engine_config=llm_engine_config,
        ui_server_app=create_ui_app(),
        serverless=True,
    )
