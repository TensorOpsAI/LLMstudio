import asyncio
import base64
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
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]


@pytest.fixture(scope="module")
def llm(provider_model, **kwargs):
    """
    Initializes the LLMCore instance for Bedrock.
    """

    """
    Initializes LLMCore dynamically based on the provider.
    Uses **kwargs to handle provider-specific parameters.
    """
    provider, _ = provider_model
    provider_args = {"provider": provider}

    if provider == "bedrock":
        provider_args.update(
            {
                "region": os.getenv("BEDROCK_REGION"),
                "secret_key": os.getenv("BEDROCK_SECRET_KEY"),
                "access_key": os.getenv("BEDROCK_ACCESS_KEY"),
            }
        )

    else:  # Default is OpenAI support
        provider_args.update(
            {
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        )

    provider_args.update(kwargs)

    return LLMCore(**provider_args)


@pytest.fixture(
    scope="module",
    params=[
        ("openai", "gpt-4o-mini"),
        ("bedrock", "us.amazon.nova-lite-v1:0"),
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

    # Async Non-Streaming Request
    print("\nAsync Non-Stream")
    chat_request = build_chat_request(
        model, chat_input=messages, max_tokens=100, is_stream=False
    )
    response_sync = asyncio.run(llm.achat(**chat_request))
    pprint(response_sync)
    metrics["async non-stream"] = response_sync.metrics

    # Async Streaming Request
    print("\nAsync Stream")
    chat_request = build_chat_request(
        model, chat_input=messages, max_tokens=100, is_stream=True
    )

    async def async_stream():
        response_async = await llm.achat(**chat_request)
        async for p in response_async:
            if p.metrics:
                pprint(p)
                metrics["async stream"] = p.metrics

    asyncio.run(async_stream())

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
