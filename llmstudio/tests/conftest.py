import pytest
from httpx import AsyncClient

from llmstudio.engine import (
    EngineConfig,
    ModelConfig,
    ProviderConfig,
    create_engine_app,
)


@pytest.fixture
def providers_list():
    return ["Anthropic", "Azure", "Cohere", "Ollama", "OpenAI"]


@pytest.fixture
def engine_config(mocker, providers_list):
    return EngineConfig(
        providers={
            provider.lower(): ProviderConfig(
                id=provider.lower(),
                name=provider,
                chat=True,
                embed=False,
                keys=[f"{provider.lower()}_key1"],
                models={
                    f"{provider.lower()}-model": ModelConfig(
                        mode="chat",
                        max_tokens=2048,
                        input_token_cost=0.0000015,
                        output_token_cost=0.000002,
                    )
                },
            )
            for provider in providers_list
        }
    )


@pytest.fixture
async def test_engine_app(engine_config):
    app = create_engine_app(engine_config)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
