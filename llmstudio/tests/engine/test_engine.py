import pytest

from llmstudio.engine import _load_engine_config, create_engine_app


def test_create_engine(engine_config):
    app = create_engine_app(engine_config)
    assert app.title == "LLMstudio Engine API"


def test_load_engine_config(mocker):
    mocker.patch(
        "pathlib.Path.read_text", side_effect=["default_config", "local_config"]
    )
    mocker.patch(
        "yaml.safe_load",
        side_effect=[
            {
                "providers": {
                    "default_provider": {
                        "id": "default",
                        "name": "Default Provider",
                        "chat": True,
                        "embed": True,
                    }
                }
            },
            {
                "providers": {
                    "local_provider": {
                        "id": "local",
                        "name": "Local Provider",
                        "chat": True,
                        "embed": False,
                    }
                }
            },
        ],
    )
    config = _load_engine_config()
    assert "default_provider" in config.providers
    assert "local_provider" in config.providers

    default_provider = config.providers["default_provider"]
    assert default_provider.id == "default"
    assert default_provider.name == "Default Provider"
    assert default_provider.chat is True
    assert default_provider.embed is True

    local_provider = config.providers["local_provider"]
    assert local_provider.id == "local"
    assert local_provider.name == "Local Provider"
    assert local_provider.chat is True
    assert local_provider.embed is False


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
    assert len(engine_config.providers) == 5
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
