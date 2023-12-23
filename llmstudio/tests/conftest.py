import pytest
from pytest_mock import mocker
from httpx import AsyncClient
from llmstudio.engine import create_engine_app, _load_engine_config
from llmstudio.tracking import create_tracking_app
from llmstudio.engine import EngineConfig, ProviderConfig, ModelConfig


@pytest.fixture
def engine_config():
    return _load_engine_config()


@pytest.fixture
async def test_engine_app():
    app = create_engine_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_engine_config():
    return mocker.patch(
        "llmstudio.engine._load_engine_config",
        return_value=EngineConfig(
            providers={
                "openai": ProviderConfig(
                    id="openai",
                    name="OpenAI",
                    chat=True,
                    embed=False,
                    keys=["key1"],
                    models={
                        "gpt-3.5-turbo": ModelConfig(
                            mode="chat",
                            max_tokens=2048,
                            input_token_cost=0.0000015,
                            output_token_cost=0.000002,
                        )
                    },
                )
            }
        ),
    )


@pytest.fixture
async def test_tracking_app():
    app = create_tracking_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
