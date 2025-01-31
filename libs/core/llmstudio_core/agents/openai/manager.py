import os
from datetime import datetime

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
        assistant = self._client(
            name=kwargs.get("name"),
            instructions=kwargs.get("instructions"),
            model=kwargs.get("model"),
            tools=kwargs.get("tools"),
        )

        openai_agent = OpenAIAgent(
            id=assistant.id,
            created_at=datetime.now(),
            tools=kwargs.get("tools"),
            model=kwargs.get("model"),
            instructions=kwargs.get("model"),
        )
        return openai_agent

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def run_agent(
        self, openai_agent: OpenAIAgent, messages: list[dict], **kwargs
    ) -> OpenAIRun:
        """ """

        if not openai_agent.thread_id:
            openai_agent.thread_id = self._client.beta.threads.create().id

        for msg in messages:
            self._client.beta.threads.messages.create(
                thread_id=openai_agent.thread_id,
                role=msg["role"],
                content=msg["content"],
            )

        run = self._client.beta.threads.runs.create(
            thread_id=openai_agent.thread_id, assistant_id=openai_agent.assistant.id
        )

        openai_run = OpenAIRun(thread_id=openai_agent.thread_id, run_id=run.id)

        return openai_run

    def retrieve_result(self, openai_run: OpenAIRun, **kwargs) -> OpenAIResult:
        """
        Retrieves an existing agent.
        """
        return self._process_result_without_streaming(openai_run=openai_run)

    def _process_result_without_streaming(self, openai_run: OpenAIRun) -> OpenAIResult:
        while run.status in ["queued", "in_progress"]:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=openai_run.thread_id, run_id=openai_run.run_id
            )
        messages = self._client.beta.threads.messages.list(
            thread_id=openai_run.thread_id
        )
        return OpenAIResult(messages=messages, thread_id=openai_run.thread_id)
