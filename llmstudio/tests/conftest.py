import pytest
from httpx import AsyncClient
from llmstudio.engine import create_engine_app, _load_engine_config
from llmstudio.tracking import create_tracking_app


@pytest.fixture
def engine_config():
    return _load_engine_config()


@pytest.fixture
async def test_engine_app():
    app = create_engine_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_tracking_app():
    app = create_tracking_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
