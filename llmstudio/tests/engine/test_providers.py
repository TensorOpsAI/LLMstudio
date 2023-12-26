import pytest
from fastapi import HTTPException

from llmstudio.engine import ModelConfig, ProviderConfig
from llmstudio.engine.providers.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_chat_functionality(test_engine_app, mocker):
    mock_chat_handler = mocker.patch(
        "llmstudio.engine.providers.provider.Provider.chat",
        return_value={"chat_output": "response"},
    )
    response = await test_engine_app.post(
        "/api/engine/chat/openai",
        json={"model": "gpt-3.5-turbo", "chat_input": "Hello"},
    )
    assert response.status_code == 200
    assert response.json() == {"chat_output": "response"}
    mock_chat_handler.assert_called_once()


@pytest.mark.asyncio
async def test_provider_chat(
    test_engine_app,
    mocker,
    providers_list,
):
    for provider_name in providers_list:
        mock_chat_handler = mocker.patch(
            f"llmstudio.engine.providers.{provider_name.lower()}.{provider_name}Provider.chat",
            return_value={"chat_output": "response"},
        )
        response = await test_engine_app.post(
            f"/api/engine/chat/{provider_name.lower()}",
            json={"model": "test-model", "chat_input": "Hello"},
        )
        assert response.status_code == 200
        assert response.json() == {"chat_output": "response"}
        mock_chat_handler.assert_called_once()


async def test_provider_chat_error(test_engine_app, mocker, providers_list):
    for provider_name in providers_list:
        mocker.patch(
            f"llmstudio.engine.providers.{provider_name.lower()}.{provider_name}Provider.chat",
            side_effect=HTTPException(status_code=500, detail="Internal Server Error"),
        )
        response = await test_engine_app.post(
            f"/api/engine/chat/{provider_name.lower()}",
            json={"model": "test-model", "chat_input": "Hello"},
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}


# def test_calculate_metrics():
#     provider = Provider(config=mock_config)
#     metrics = provider.calculate_metrics(
#         input="test input",
#         output="test output",
#         model="test-model",
#         start_time=0,
#         end_time=1,
#         first_token_time=0.5,
#         token_times=(0.1, 0.2, 0.3),
#         token_count=3,
#     )
#     assert metrics["input_tokens"] == len("test input")
#     assert metrics["output_tokens"] == len("test output")
#     assert metrics["total_tokens"] == len("test input") + len("test output")
#     # Add more assertions for other metrics


def test_openai_provider_initialization():
    provider_config = ProviderConfig(
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
    provider = OpenAIProvider(config=provider_config)
    assert provider.config == provider_config


@pytest.mark.asyncio
async def test_openai_parse_response(mocker):
    mock_response = mocker.MagicMock()
    mock_response.__aiter__.return_value = [
        {"choices": [{"finish_reason": "stop", "delta": {"content": "response"}}]}
    ]
    provider_config = ProviderConfig(
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
    provider = OpenAIProvider(config=provider_config)
    async for content in provider.parse_response(mock_response):
        assert content == "response"
