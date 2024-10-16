import pytest
from unittest.mock import AsyncMock, MagicMock
from llmstudio_core.providers.provider import BaseProvider, ChatRequest, ProviderError

request = ChatRequest(chat_input="Hello", model="test_model")

def test_chat(mock_provider):
    mock_provider.generate_client = MagicMock(return_value=MagicMock())
    mock_provider.handle_response = MagicMock(return_value=iter(["response"]))
    
    print(request.model_dump())
    response = mock_provider.chat(request.chat_input, request.model)
    
    assert response is not None

@pytest.mark.asyncio
async def test_achat(mock_provider):
    mock_provider.agenerate_client = AsyncMock(return_value=AsyncMock())
    mock_provider.ahandle_response = AsyncMock(return_value=AsyncMock())
    
    print(request.model_dump())
    response = await mock_provider.achat(request.chat_input, request.model)
    
    assert response is not None


def test_validate_model(mock_provider):
    request = ChatRequest(chat_input="Hello", model="test_model")
    mock_provider.validate_model(request)  # Should not raise

    request_invalid = ChatRequest(chat_input="Hello", model="invalid_model")
    with pytest.raises(ProviderError):
        mock_provider.validate_model(request_invalid)

def test_calculate_metrics(mock_provider):
    metrics = mock_provider.calculate_metrics(
        input="Hello",
        output="World",
        model="test_model",
        start_time=0,
        end_time=1,
        first_token_time=0.5,
        token_times=(0.1, 0.2),
        token_count=2
    )
    
    assert metrics["input_tokens"] == pytest.approx(1)
    assert metrics["output_tokens"] == pytest.approx(1)
    assert metrics["cost_usd"] == pytest.approx(0.03)
    assert metrics["latency_s"] == pytest.approx(1)
    assert metrics["time_to_first_token_s"] == pytest.approx(0.5)
    assert metrics["inter_token_latency_s"] == pytest.approx(0.15)
    assert metrics["tokens_per_second"] == pytest.approx(2)
