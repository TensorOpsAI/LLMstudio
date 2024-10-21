from unittest.mock import AsyncMock, MagicMock

import pytest
from llmstudio_core.providers.provider import ChatRequest, ProviderCore, ProviderError


class MockProvider(ProviderCore):
    async def aparse_response(self, response, **kwargs):
        return response

    def parse_response(self, response, **kwargs):
        return response

    def chat(self, chat_input, model, **kwargs):
        # Mock the response to match expected structure
        return MagicMock(choices=[MagicMock(finish_reason="stop")])

    async def achat(self, chat_input, model, **kwargs):
        # Mock the response to match expected structure
        return MagicMock(choices=[MagicMock(finish_reason="stop")])

    def output_to_string(self, output):
        # Handle string inputs
        if isinstance(output, str):
            return output
        if output.choices[0].finish_reason == "stop":
            return output.choices[0].message.content
        return ""

    @staticmethod
    def _provider_config_name():
        return "mock_provider"


@pytest.fixture
def mock_provider():
    config = MagicMock()
    config.models = {
        "test_model": MagicMock(input_token_cost=0.01, output_token_cost=0.02)
    }
    config.id = "mock_provider"
    tokenizer = MagicMock()
    tokenizer.encode = lambda x: x.split()  # Simple tokenizer mock
    return MockProvider(config=config, tokenizer=tokenizer)
