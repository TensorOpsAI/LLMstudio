import click

from .llm_engine.config import LLMEngineConfig
from .utils.rest_utils import run_apis



@click.group()
def main():
    pass


@main.command()
def server(llm_engine_config = LLMEngineConfig()):
    run_apis(llm_engine_config=llm_engine_config)