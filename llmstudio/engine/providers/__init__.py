from llmstudio.engine.config import Provider
from llmstudio.engine.providers.base_provider import BaseProvider


def get_provider(provider: Provider) -> BaseProvider:
    from llmstudio.engine.providers.anthropic import AnthropicProvider
    from llmstudio.engine.providers.bedrock import BedrockProvider
    from llmstudio.engine.providers.openai import OpenAIProvider
    from llmstudio.engine.providers.vertexai import VertexAIProvider

    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
        Provider.VERTEXAI: VertexAIProvider,
        Provider.BEDROCK: BedrockProvider,
        Provider.ANTHROPIC: AnthropicProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise ValueError(f"Provider {provider} not found")
