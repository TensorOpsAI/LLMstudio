import pytest


@pytest.mark.asyncio
async def test_chat_functionality(test_engine_app, mocker):
    mock_chat_handler = mocker.patch(
        "llmstudio.engine.providers.Provider.chat",
        return_value={"chat_output": "response"},
    )
    response = await test_engine_app.post(
        "/api/engine/chat/openai",
        json={"model": "gpt-3.5-turbo", "chat_input": "Hello"},
    )
    assert response.status_code == 200
    assert response.json() == {"chat_output": "response"}
    mock_chat_handler.assert_called_once()
