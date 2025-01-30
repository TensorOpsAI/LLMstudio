import os

from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.agents.openai.data_models import (
    OpenAIAgent,
    OpenAIResult,
    OpenAIRun,
)
from openai import OpenAI


@agent_manager
class OpenAIAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.API_KEY = self.API_KEY if self.API_KEY else os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self.API_KEY)

    @staticmethod
    def _agent_config_name():
        return "openai"

    def create_agent(self, **kwargs) -> OpenAIAgent:
        """
        Creates a new instance of the agent.
        """
        raise NotImplementedError("Agents need to implement the 'create' method.")

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def run_agent(self, **kwargs) -> OpenAIRun:
        """ """
        raise NotImplementedError(
            "Agents need to implement the 'create_thread_and_run' method."
        )

    def retrieve_result(self, **kwargs) -> OpenAIResult:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError("Agents need to implement the 'retrieve' method.")
