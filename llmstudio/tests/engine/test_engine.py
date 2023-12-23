import pytest
from llmstudio.engine import create_engine_app


def test_create_engine(engine_config):
    app = create_engine_app(engine_config)
    assert app.title == "LLMstudio Engine API"


@pytest.mark.asyncio
async def test_health(test_engine_app):
    response = await test_engine_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "message": "Engine is up and running",
    }


@pytest.mark.asyncio
async def test_get_providers(test_engine_app, engine_config):
    response = await test_engine_app.get("/api/engine/providers")
    assert response.status_code == 200
    assert response.json() == list(engine_config.providers.keys())


@pytest.mark.asyncio
async def test_get_models(test_engine_app, engine_config):
    response = await test_engine_app.get("/api/engine/models")
    assert response.status_code == 200
    assert response.json() == {
        provider: {"name": config.name, "models": list(config.models.keys())}
        for provider, config in engine_config.providers.items()
    }


@pytest.mark.asyncio
async def test_export_functionality(test_engine_app):
    data = [{"input": "test", "output": "result"}]
    response = await test_engine_app.post("/api/export", json=data)
    assert response.status_code == 200
    assert (
        response.headers["Content-Disposition"] == "attachment; filename=parameters.csv"
    )
    assert response.content == b'input;output\n"test";"result"\n'
