import os

import openai
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.agents.openai.data_models import (
    OpenAIAgent,
    OpenAIFiles,
    OpenAIResult,
    OpenAIRun,
)


@agent_manager
class OpenAIAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.API_KEY = self.API_KEY if self.API_KEY else os.getenv("OPENAI_API_KEY")
        self.api_type = (
            kwargs.get("apy_type")
            if kwargs.get("apy_type")
            else os.getenv("OPENAI_API_TYPE")
        )
        openai.api_key = self.API_KEY
        openai.api_type = self.api_type

    @staticmethod
    def _agent_config_name():
        return "openai"

    def create_agent(self, params: dict = None) -> OpenAIAgent:
        """
        Creates a new instance of the agent.
        """
        assistant = openai.beta.assistants.create(
            name=params.get("name"),
            instructions=params.get("instructions"),
            model=params.get("model"),
            tools=params.get("tools"),
            tool_resources=params.get("tool_resources"),
        )

        openai_agent = OpenAIAgent(id=assistant.id, assistant_id=assistant.id)

        return openai_agent

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_create_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def run_agent(
        self,
        openai_agent: OpenAIAgent,
        messages: list[dict],
        params: dict = None,
        **kwargs
    ) -> OpenAIRun:
        """ """

        if not openai_agent.thread_id:
            openai_agent.thread_id = self.create_new_thread()

        for msg in messages:
            self._add_messages_to_thread(msg, openai_agent.thread_id)

        run_id = self.create_run(openai_agent.thread_id, openai_agent.assistant_id)

        openai_run = OpenAIRun(
            thread_id=openai_agent.thread_id,
            assistant_id=openai_agent.assistant_id,
            run_id=run_id,
        )

        return openai_run

    def retrieve_result(self, openai_run: OpenAIRun, **kwargs) -> OpenAIResult:
        """
        Retrieves an existing agent.
        """
        return self._process_result_without_streaming(openai_run=openai_run)

    def _process_result_without_streaming(self, openai_run: OpenAIRun) -> OpenAIResult:
        run = openai.beta.threads.runs.retrieve(
            thread_id=openai_run.thread_id, run_id=openai_run.run_id
        )
        while run.status in ["queued", "in_progress"]:
            run = openai.beta.threads.runs.retrieve(
                thread_id=openai_run.thread_id, run_id=openai_run.run_id
            )
        messages = openai.beta.threads.messages.list(thread_id=openai_run.thread_id)
        print(messages)
        processed_messages = self._process_messages_without_streaming(messages)
        return OpenAIResult(messages=processed_messages, thread_id=openai_run.thread_id)

    def _process_messages_without_streaming(self, messages: list):
        messages_content = []
        files = []
        messages_file = []

        for message in messages.data:
            if message.role == "user":
                break

            for content_block in message.content:
                if content_block.type == "image_file":
                    files.append(
                        OpenAIFiles(
                            type="image_file", file_id=content_block.image_file.file_id
                        )
                    )
                    continue

                if content_block.type == "text" and not len(
                    content_block.text.annotations
                ):
                    messages_content.append(content_block.text.value)

                if len(content_block.text.annotations):
                    for annotation in content_block.text.annotations:
                        files.append(
                            OpenAIFiles(
                                type="file_path",
                                file_id=annotation.file_path.file_id,
                                file_path=annotation.text,
                            )
                        )

                    messages_file.append(content_block.text.value)

        print(messages_file)
        print(messages_content)
        print(files)

    def upload_file(self, file_path: str) -> str:
        file = openai.files.create(file=open(file_path, "rb"), purpose="assistants")
        return file.id

    def create_new_thread(self) -> str:
        return openai.beta.threads.create().id

    def _add_messages_to_thread(self, message: dict, thread_id):
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role=message.get("role"),
            content=message.get("content"),
            attachments=message.get("attachments"),
        )

    def create_run(self, thread_id: str, assistant_id: str) -> str:
        run = openai.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )
        return run.id
