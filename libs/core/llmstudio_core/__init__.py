from typing import Optional
from llmstudio_core.providers.provider import BaseProvider, provider_registry

from llmstudio_core.providers import _load_providers_config

_engine_config = _load_providers_config()


def LLMCore(provider: str, api_key: Optional[str] = None, **kwargs) -> BaseProvider:
    """
    Factory method to create an instance of a provider.

    Args:
        provider (str): The name of the provider.
        api_key (Optional[str], optional): The API key for the provider. Defaults to None.

    Returns:
        BaseProvider: An instance of the provider.

    Raises:
        NotImplementedError: If the provider is not found in the provider map.
    """
    provider_config = _engine_config.providers.get(provider)
    provider_class = provider_registry.get(provider_config.id)
    if provider_class:
        return provider_class(config=provider_config, api_key=api_key, **kwargs)
    raise NotImplementedError(f"Provider not found: {provider_config.id}. Available providers: {str(provider_registry.keys())}")
