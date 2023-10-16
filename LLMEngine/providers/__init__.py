from LLMEngine.config import Provider
from LLMEngine.providers.base_provider import BaseProvider


def get_provider(provider: Provider) -> BaseProvider:
    from LLMEngine.providers.openai import OpenAIProvider
    from LLMEngine.providers.vertexai import VertexAIProvider
    from LLMEngine.providers.bedrock import BedrockProvider

    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
        Provider.VERTEXAI: VertexAIProvider,
        Provider.BEDROCK: BedrockProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise ValueError(f"Provider {provider} not found")