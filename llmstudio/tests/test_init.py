from llmstudio.engine import create_engine_app


def test_create_engine_app(engine_config):
    app = create_engine_app(engine_config)
    assert app.title == "LLMstudio Engine API"
