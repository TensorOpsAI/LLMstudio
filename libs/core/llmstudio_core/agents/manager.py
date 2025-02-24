from abc import ABC, abstractmethod
from typing import Optional

from llmstudio_core.agents.data_models import AgentBase, ResultBase, RunBase

agent_registry = {}


def agent_manager(cls):
    """Decorator to register a new agent manager"""
    agent_registry[cls._agent_config_name()] = cls

    return cls


class AgentManager(ABC):
    def __init__(
        self,
        api_key: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):

        self.API_KEY = api_key
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region

    @abstractmethod
    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    @abstractmethod
    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    @abstractmethod
    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    @abstractmethod
    def create_agent(self, **kwargs) -> AgentBase:
        """
        Creates a new instance of the agent.
        """

        raise NotImplementedError("Agents need to implement the 'create' method.")

    @abstractmethod
    def run_agent(self, **kwargs) -> RunBase:
        """
        Runs the agent
        """
        raise NotImplementedError(
            "Agents need to implement the 'create_thread_and_run' method."
        )

    @abstractmethod
    def retrieve_result(self, **kwargs) -> ResultBase:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError("Agents need to implement the 'retrieve' method.")

    @abstractmethod
    def submit_tool_outputs(self, **kwargs) -> ResultBase:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError(
            "Agents need to implement the 'submit_tool_outputs' method."
        )
