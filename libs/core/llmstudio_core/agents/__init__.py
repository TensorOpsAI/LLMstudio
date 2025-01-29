from typing import Optional

from lmstudio_core.agents.agent import Agent, agent_registry
from lmstudio_core.utils import _load_config

_engine_config = _load_config()


def AgentManagerCore(provider: str, api_key: Optional[str] = None, **kwargs) -> Agent:
    """
    Factory method to create an instance of an agent.

    Args:
        provider (str): The name of the provider.
        api_key (Optional[str], optional): The API key for the provider. Defaults to None.

    Returns:
        ProviderCore: An instance of the provider.

    Raises:
        NotImplementedError: If the provider is not found in the provider map.
    """
    agent_config = _engine_config.agents.get(provider)
    provider_class = agent_registry.get(agent_config.id)
    if provider_class:
        return provider_class(config=agent_config, api_key=api_key, **kwargs)
    raise NotImplementedError(
        f"Provider not found: {agent_config.id}. Available providers: {str(agent_registry.keys())}"
    )
