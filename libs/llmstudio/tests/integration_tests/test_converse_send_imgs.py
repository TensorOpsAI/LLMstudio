import os
from pprint import pprint

import pytest
from dotenv import load_dotenv
from llmstudio_core.providers import LLMCore
from llmstudio_core.providers.data_structures import Metrics

load_dotenv()

INPUT_IMAGE_PATH = (
    "./libs/llmstudio/tests/integration_tests/test_data/llmstudio-logo.jpeg"
)


def build_chat_request(
    model: str, chat_input: list, is_stream: bool, max_tokens: int = 1000
):
    """
    Builds a chat request payload for different providers.
    """
    if model.startswith(("o1", "o3")):
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {"max_completion_tokens": max_tokens},
        }
    elif "amazon.nova" in model or "anthropic.claude" in model:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {"maxTokens": max_tokens},
        }
    else:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "functions": None,
            },
        }
    return chat_request


@pytest.fixture(scope="module")
def image_bytes():
    """
    Reads the input image as bytes.
    """
    with open(INPUT_IMAGE_PATH, "rb") as f:
        return f.read()


@pytest.fixture(scope="module")
def messages(image_bytes):
    """
    Creates a message payload with both text and image.
    """
    return [
        {
            "role": "user",
            "content": [
                {"text": "What's in this image?"},
                {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
            ],
        }
    ]


@pytest.fixture(scope="module")
def llm():
    """
    Initializes the LLMCore instance for Bedrock.
    """
    return LLMCore(
        provider="bedrock",
        api_key=None,
        region=os.environ["BEDROCK_REGION"],
        secret_key=os.environ["BEDROCK_SECRET_KEY"],
        access_key=os.environ["BEDROCK_ACCESS_KEY"],
    )


@pytest.fixture(
    scope="module",
    params=[
        ("bedrock", "us.amazon.nova-lite-v1:0"),
        ("bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ],
)
def provider_model(request):
    """
    Provides multiple provider-model pairs for parameterized testing.
    """
    return request.param


def run_provider(llm, provider, model, messages):
    """
    Runs a full test cycle for a given provider and model with an image.
    """
    print(f"\n### RUNNING <{provider}> - <{model}> ###\n")

    metrics = {}

    # Non-Streaming Request
    print("\nSync Non-Stream")
    chat_request = build_chat_request(
        model, chat_input=messages, max_tokens=100, is_stream=False
    )
    response_sync = llm.chat(**chat_request)
    pprint(response_sync)
    metrics["sync non-stream"] = response_sync.metrics

    # Streaming Request
    print("\nSync Stream")
    chat_request = build_chat_request(
        model, chat_input=messages, max_tokens=100, is_stream=True
    )
    response_sync_stream = llm.chat(**chat_request)

    for p in response_sync_stream:
        if p.metrics:
            pprint(p)
            metrics["sync stream"] = p.metrics

    print(f"\n### REPORT for <{provider}> - <{model}> ###\n")
    return metrics


@pytest.fixture(scope="module")
def metrics(llm, provider_model, messages):
    """
    Runs the provider-model test once and returns metrics.
    """
    provider, model = provider_model
    return run_provider(llm, provider, model, messages)


def test_response_output(provider_model, metrics):
    """
    Validates that the model successfully processes image inputs.
    """
    provider, model = provider_model

    for current_metrics in metrics.values():
        assert isinstance(
            current_metrics, Metrics
        ), f"Expected Metrics object, got {type(current_metrics)}"
        assert (
            current_metrics.output_tokens > 0
        ), "Output tokens should be larger than 0"

    print(f"✅ Image Processing Test Passed for {provider} - {model}")
