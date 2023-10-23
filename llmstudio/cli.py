import click

from llmstudio.engine.config import EngineConfig
from llmstudio.ui import create_ui_app
from llmstudio.utils.rest_utils import run_apis


@click.group()
def main():
    pass


@main.command()
def server(engine_config=EngineConfig()):
    run_apis(
        engine_config=engine_config,
        ui_server_app=create_ui_app(),
        serverless=True,
    )
