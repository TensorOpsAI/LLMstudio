from unittest.mock import MagicMock

import pytest
from llmstudio_core.providers.provider import ChatCompletion, time, ChatCompletionChunk

@pytest.mark.asyncio
async def test_ahandle_response_non_streaming(mock_provider):
    request = MagicMock(is_stream=False, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Non-streamed response"}, "finish_reason": "stop", "index":0}],
        "model": "test_model",
    }
    start_time = time.time()

    async def mock_aparse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.aparse_response = mock_aparse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Non-streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    async for chunk in mock_provider.ahandle_response(request, mock_aparse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletion)
    assert response[0].choices == []
    assert response[0].chat_output == "Non-streamed response"

@pytest.mark.asyncio
async def test_ahandle_response_streaming_length(mock_provider):
    request = MagicMock(is_stream=True, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Streamed response"}, "finish_reason": "length", "index":0}],
        "model": "test_model",
        "object":"chat.completion.chunk",
        "created":0
    }
    start_time = time.time()

    async def mock_aparse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.aparse_response = mock_aparse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    async for chunk in mock_provider.ahandle_response(request, mock_aparse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletionChunk)
    assert response[0].chat_output_stream == "Streamed response"
    
@pytest.mark.asyncio
async def test_ahandle_response_streaming_stop(mock_provider):
    request = MagicMock(is_stream=True, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Streamed response"}, "finish_reason": "stop", "index":0}],
        "model": "test_model",
        "object":"chat.completion.chunk",
        "created":0
    }
    start_time = time.time()

    async def mock_aparse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.aparse_response = mock_aparse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    async for chunk in mock_provider.ahandle_response(request, mock_aparse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletionChunk)
    assert response[0].chat_output == "Streamed response"
    #assert response[0].chat_output_stream == "Streamed response"
    

def test_handle_response_non_streaming(mock_provider):
    request = MagicMock(is_stream=False, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Non-streamed response"}, "finish_reason": "stop", "index":0}],
        "model": "test_model",
    }
    start_time = time.time()

    def mock_parse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.aparse_response = mock_parse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Non-streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    for chunk in mock_provider.handle_response(request, mock_parse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletion)
    assert response[0].choices == []
    assert response[0].chat_output == "Non-streamed response"
    
def test_handle_response_streaming_length(mock_provider):
    request = MagicMock(is_stream=True, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Streamed response"}, "finish_reason": "length", "index":0}],
        "model": "test_model",
        "object":"chat.completion.chunk",
        "created":0
    }
    start_time = time.time()

    def mock_parse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.aparse_response = mock_parse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    for chunk in mock_provider.handle_response(request, mock_parse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletionChunk)
    assert response[0].chat_output_stream == "Streamed response"
    
def test_handle_response_streaming_stop(mock_provider):
    request = MagicMock(is_stream=True, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Streamed response"}, "finish_reason": "stop", "index":0}],
        "model": "test_model",
        "object":"chat.completion.chunk",
        "created":0
    }
    start_time = time.time()

    def mock_parse_response(*args, **kwargs):
        yield response_chunk

    mock_provider.parse_response = mock_parse_response
    mock_provider.join_chunks = MagicMock(return_value=(ChatCompletion(id="id", choices=[], created=0, model="test_model", object="chat.completion"), "Streamed response"))
    mock_provider.calculate_metrics = MagicMock(return_value={"input_tokens": 1})

    response = []
    for chunk in mock_provider.handle_response(request, mock_parse_response(), start_time):
        response.append(chunk)

    assert isinstance(response[0], ChatCompletionChunk)
    assert response[0].chat_output == "Streamed response"
    #assert response[0].chat_output_stream == "Streamed response"