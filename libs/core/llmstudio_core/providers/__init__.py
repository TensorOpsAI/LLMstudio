# from llmstudio_core.providers.anthropic import AnthropicProvider #TODO: adpat it
from typing import Optional

from llmstudio_core.providers.azure import AzureProvider
from llmstudio_core.providers.bedrock_converse import BedrockConverseProvider

# from llmstudio_core.providers.ollama import OllamaProvider #TODO: adapt it
from llmstudio_core.providers.openai import OpenAIProvider
from llmstudio_core.providers.provider import ProviderCore, provider_registry
from llmstudio_core.providers.vertexai import VertexAIProvider
from llmstudio_core.utils import _load_providers_config

_engine_config = _load_providers_config()


def LLMCore(provider: str, api_key: Optional[str] = None, **kwargs) -> ProviderCore:
    """
    Factory method to create an instance of a provider.

    Args:
        provider (str): The name of the provider.
        api_key (Optional[str], optional): The API key for the provider. Defaults to None.

    Returns:
        ProviderCore: An instance of the provider.

    Raises:
        NotImplementedError: If the provider is not found in the provider map.
    """
    provider_config = _engine_config.providers.get(provider)
    provider_class = provider_registry.get(provider_config.id)
    if provider_class:
        return provider_class(config=provider_config, api_key=api_key, **kwargs)
    raise NotImplementedError(
        f"Provider not found: {provider_config.id}. Available providers: {str(provider_registry.keys())}"
    )
