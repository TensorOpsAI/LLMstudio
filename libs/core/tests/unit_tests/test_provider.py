from unittest.mock import AsyncMock, MagicMock

import pytest
from llmstudio_core.providers.provider import ChatRequest, ProviderError, ChatCompletion, time

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
    
    mock_provider.tokenizer.encode = lambda x: x.split() # Assuming tokenizer splits "Hello" and "World" into one token each

    metrics = mock_provider.calculate_metrics(
        input="Hello",
        output="Hello World",
        model="test_model",
        start_time=0.0,
        end_time=1.0,
        first_token_time=0.5,
        token_times=(0.1,),
        token_count=2,
    )

    assert metrics["input_tokens"] == 1
    assert metrics["output_tokens"] == 2
    assert metrics["total_tokens"] == 3
    assert metrics["cost_usd"] == 0.01 * 1 + 0.02 * 2  # input_cost + output_cost
    assert metrics["latency_s"] == 1.0  # end_time - start_time
    assert metrics["time_to_first_token_s"] == 0.5 - 0.0  # first_token_time - start_time
    assert metrics["inter_token_latency_s"] == 0.1  # Average of token_times
    assert metrics["tokens_per_second"] == 2 / 1.0  # token_count / total_time
    
def test_calculate_metrics_single_token(mock_provider):
    
    mock_provider.tokenizer.encode = lambda x: x.split()

    metrics = mock_provider.calculate_metrics(
        input="Hello",
        output="World",
        model="test_model",
        start_time=0.0,
        end_time=1.0,
        first_token_time=0.5,
        token_times=(),
        token_count=1,
    )

    assert metrics["input_tokens"] == 1
    assert metrics["output_tokens"] == 1
    assert metrics["total_tokens"] == 2
    assert metrics["cost_usd"] == 0.01 * 1 + 0.02 * 1
    assert metrics["latency_s"] == 1.0
    assert metrics["time_to_first_token_s"] == 0.5 - 0.0
    assert metrics["inter_token_latency_s"] == 0
    assert metrics["tokens_per_second"] == 1 / 1.0
    
def test_handle_response_stop(mock_provider):

    current_time = int(time.time())

    response_generator = iter([
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"content": "Hello, "},
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        },
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"content": "world!"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        },
    ])

    request = ChatRequest(chat_input="Hello", model="test_model")
    result_generator = mock_provider.handle_response(request, response_generator, start_time=time.time())
    result = next(result_generator)

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello, world!"
    
def test_handle_response_stop_single_token(mock_provider):
    """
    testing single token answer. token_times var will be 0
    """

    current_time = int(time.time())

    response_generator = iter([
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"content": "Hello, "},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }
    ])

    request = ChatRequest(chat_input="Hello", model="test_model")
    result_generator = mock_provider.handle_response(request, response_generator, start_time=time.time())
    result = next(result_generator)

    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello, "