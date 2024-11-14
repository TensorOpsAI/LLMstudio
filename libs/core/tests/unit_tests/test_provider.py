from unittest.mock import AsyncMock, MagicMock

import pytest
from llmstudio_core.providers.provider import ChatRequest, ProviderError, ChatCompletion, time, ChatCompletionChunk

request = ChatRequest(chat_input="Hello World", model="test_model")

    
def test_chat_response_non_stream(mock_provider):
    mock_provider.validate_request = MagicMock()
    mock_provider.validate_model = MagicMock()
    mock_provider.generate_client = MagicMock(return_value="mock_response")
    mock_provider.handle_response = MagicMock(return_value="final_response")

    response = mock_provider.chat(chat_input="Hello", model="test_model")

    assert response == "final_response"
    mock_provider.validate_request.assert_called_once()
    mock_provider.validate_model.assert_called_once()
    
def test_chat_streaming_response(mock_provider):
    mock_provider.validate_request = MagicMock()
    mock_provider.validate_model = MagicMock()
    mock_provider.generate_client = MagicMock(return_value="mock_response_stream")
    mock_provider.handle_response = MagicMock(return_value=iter(["streamed_response_1", "streamed_response_2"]))

    response_stream = mock_provider.chat(chat_input="Hello", model="test_model", is_stream=True)
    assert next(response_stream) == "streamed_response_1"
    assert next(response_stream) == "streamed_response_2"
    mock_provider.validate_request.assert_called_once()
    mock_provider.validate_model.assert_called_once()

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

@pytest.mark.asyncio
async def test_ahandle_response_non_streaming(mock_provider):
    request = MagicMock(is_stream=False, chat_input="Hello", model="test_model", parameters={})
    response_chunk = {
        "choices": [{"delta": {"content": "Non-streamed response"}, "finish_reason": "stop"}],
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
    
def test_join_chunks_finish_reason_stop(mock_provider):
    current_time = int(time.time())
    chunks = [
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
    ]
    response, output_string = mock_provider.join_chunks(chunks)

    assert output_string == "Hello, world!"
    assert response.choices[0].message.content == "Hello, world!"

def test_join_chunks_finish_reason_function_call(mock_provider):
    current_time = int(time.time())
    chunks = [
        {
            "id": "test_id",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {"function_call": {"name": "my_function", "arguments": "arg1"}},
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
                    "delta": {"function_call": {"arguments": "arg2"}},
                    "finish_reason": "function_call",
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
                    "delta": {"function_call": {"arguments": "}"}},
                    "finish_reason": "function_call",
                    "index": 0,
                }
            ],
        },
    ]
    response, output_string = mock_provider.join_chunks(chunks)

    assert output_string == "arg1arg2"
    assert response.choices[0].message.function_call.arguments == "arg1arg2"
    assert response.choices[0].message.function_call.name == "my_function"
    
    
def test_join_chunks_tool_calls(mock_provider):
    current_time = int(time.time())
    
    chunks = [
        {
            "id": "test_id_1",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "function": {"name": "search_tool", "arguments": "{\"query\": \"weather"},
                                "type": "function"
                            }
                        ]
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ]
        },
        {
            "id": "test_id_2",
            "model": "test_model",
            "created": current_time,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "function": {"name": "search_tool", "arguments": " details\"}"}
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                    "index": 0
                }
            ]
        }
    ]

    response, output_string = mock_provider.join_chunks(chunks)
    
    assert output_string == "['search_tool', '{\"query\": \"weather details\"}']"

    

    assert response.object == "chat.completion"
    assert response.choices[0].finish_reason == "tool_calls"
    tool_call = response.choices[0].message.tool_calls[0]

    assert tool_call.id == "tool_1"
    assert tool_call.function.name == "search_tool"
    assert tool_call.function.arguments == "{\"query\": \"weather details\"}"
    assert tool_call.type == "function"
    

def test_calculate_cost_fixed_cost(mock_provider):
    fixed_cost = 0.02
    token_count = 100
    expected_cost = token_count * fixed_cost
    assert mock_provider.calculate_cost(token_count, fixed_cost) == expected_cost

def test_calculate_cost_variable_cost(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    variable_cost = [cost_range_1, cost_range_2]
    token_count = 75
    expected_cost = token_count * 0.02
    assert mock_provider.calculate_cost(token_count, variable_cost) == expected_cost

def test_calculate_cost_variable_cost_higher_range(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, None)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 150
    expected_cost = token_count * 0.03
    assert mock_provider.calculate_cost(token_count, variable_cost) == expected_cost

def test_calculate_cost_variable_cost_no_matching_range(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (0, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, 150)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 200
    expected_cost = 0
    assert mock_provider.calculate_cost(token_count, variable_cost) == expected_cost
    
def test_calculate_cost_variable_cost_no_matching_range_inferior(mock_provider):
    cost_range_1 = MagicMock()
    cost_range_1.range = (10, 50)
    cost_range_1.cost = 0.01

    cost_range_2 = MagicMock()
    cost_range_2.range = (51, 100)
    cost_range_2.cost = 0.02

    cost_range_3 = MagicMock()
    cost_range_3.range = (101, 150)
    cost_range_3.cost = 0.03

    variable_cost = [cost_range_1, cost_range_2, cost_range_3]
    token_count = 5
    expected_cost = 0
    assert mock_provider.calculate_cost(token_count, variable_cost) == expected_cost
    
def test_input_to_string_with_string(mock_provider):
    input_data = "Hello, world!"
    assert mock_provider.input_to_string(input_data) == "Hello, world!"


def test_input_to_string_with_list_of_text_messages(mock_provider):
    input_data = [
        {"content": "Hello"},
        {"content": " world!"},
    ]
    assert mock_provider.input_to_string(input_data) == "Hello world!"


def test_input_to_string_with_list_of_text_and_url(mock_provider):
    input_data = [
        {"role": "user", "content": [{"type": "text", "text": "Hello "}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}]},
        {"role": "user", "content": [{"type": "text", "text": " world!"}]},
    ]
    expected_output = "Hello http://example.com/image.jpg world!"
    assert mock_provider.input_to_string(input_data) == expected_output


def test_input_to_string_with_mixed_roles_and_missing_content(mock_provider):
    input_data = [
        {"role": "assistant", "content": "Admin text;"},
        {"role": "user", "content": [{"type": "text", "text": "User text"}]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/another.jpg"}}]},
    ]
    expected_output = "Admin text;User texthttp://example.com/another.jpg"
    assert mock_provider.input_to_string(input_data) == expected_output


def test_input_to_string_with_missing_content_key(mock_provider):
    input_data = [
        {"role": "user"},
        {"role": "user", "content": [{"type": "text", "text": "Hello again"}]},
    ]
    expected_output = "Hello again"    
    assert mock_provider.input_to_string(input_data) == expected_output


def test_input_to_string_with_empty_list(mock_provider):
    input_data = []
    assert mock_provider.input_to_string(input_data) == ""

