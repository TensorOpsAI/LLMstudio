from typing import Optional

from llmstudio_core.agents.openai.manager import OpenAIAgentManager
from llmstudio_core.agents.bedrock.manager import BedrockAgentManager

from llmstudio_core.agents.manager import AgentManager, agent_registry
from llmstudio_core.utils import _load_providers_config

_engine_config = _load_providers_config()


def AgentManagerCore(
    provider: str, api_key: Optional[str] = None, **kwargs
) -> AgentManager:
    """
    Factory method to create an instance of an agent.

    Args:
        provider (str): The name of the provider.
        api_key (Optional[str], optional): The API key for the provider. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the provider class.

    Returns:
        AgentManager: An instance of the agent manager.

    Raises:
        NotImplementedError: If the provider is not found in the agent registry.
    """
    agent_config = _engine_config.agents.get(provider)
    provider_class = agent_registry.get(agent_config.id)
    if provider_class:
        return provider_class(api_key=api_key, **kwargs)
    raise NotImplementedError(
        f"Provider not found: {agent_config.id}. Available providers: {str(agent_registry.keys())}"
    )
