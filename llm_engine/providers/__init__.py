from llm_engine.config import Provider
from llm_engine.providers.base_provider import BaseProvider


def get_provider(provider: Provider) -> BaseProvider:
    from llm_engine.providers.openai import OpenAIProvider
    from llm_engine.providers.vertexai import VertexAIProvider
    from llm_engine.providers.bedrock import BedrockProvider

    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
        Provider.VERTEXAI: VertexAIProvider,
        Provider.BEDROCK: BedrockProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise ValueError(f"Provider {provider} not found")