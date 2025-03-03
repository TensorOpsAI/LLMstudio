import os
from typing import List, Union

import openai
from llmstudio_core.agents.data_models import (
    Annotation,
    Attachment,
    CreateAgentRequest,
    File,
    FileCitation,
    ImageFile,
    ImageFileContent,
    ImageUrlContent,
    ImageUrlObject,
    Message,
    RefusalContent,
    RequiredAction,
    ResultBase,
    RunAgentRequest,
    TextContent,
    TextObject,
    Tool,
    ToolCall,
    ToolCallFunction,
)
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.agents.openai.data_models import (
    OpenAIAgent,
    OpenAIInputMessage,
    OpenAIRun,
)
from llmstudio_core.exceptions import AgentError
from pydantic import ValidationError


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

    def create_agent(self, params: dict) -> OpenAIAgent:
        """
        Creates a new instance of the agent.
        """
        try:
            agent_request = self._validate_create_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        assistant = openai.beta.assistants.create(
            name=agent_request.name,
            instructions=agent_request.instructions,
            model=agent_request.model,
            tools=agent_request.tools,
            tool_resources=agent_request.tool_resources,
        )

        return OpenAIAgent(
            agent_id=assistant.id,
            name=agent_request.name,
            description=agent_request.description,
            instructions=agent_request.instructions,
            tools=agent_request.tools,
            tool_resources=agent_request.tool_resources,
        )

    def _validate_create_request(self, request: dict):
        return CreateAgentRequest(
            model=request.get("model"),
            instructions=request.get("instructions"),
            description=request.get("description"),
            tools=request.get("tools"),
            tool_resources=request.get("tool_resources"),
            name=request.get("name"),
        )

    def _validate_run_request(self, request):
        return RunAgentRequest(**request)

    def _validate_input_messages(self, messages):
        return [OpenAIInputMessage(**msg) for msg in messages]

    def _validate_result_request(self, request):
        if isinstance(request, OpenAIRun):
            return request
        return OpenAIRun(**request)

    def run_agent(self, params: dict) -> OpenAIRun:
        try:
            run_request: RunAgentRequest = self._validate_run_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        if not run_request.thread_id:
            run_request.thread_id = self.create_new_thread()

        for msg in run_request.messages:
            self._add_messages_to_thread(msg, run_request.thread_id)

        run_id = self.create_run(run_request.thread_id, run_request.agent_id)

        openai_run = OpenAIRun(
            thread_id=run_request.thread_id,
            assistant_id=run_request.agent_id,
            run_id=run_id,
        )

        return openai_run

    def retrieve_result(self, params) -> ResultBase:
        """
        Retrieves an existing agent.
        """

        try:
            openai_run = self._validate_result_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        return self._process_result_without_streaming(openai_run=openai_run)

    def _process_required_action_without_streaming(self, run):
        tool_calls = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            function = ToolCallFunction(
                arguments=tool.function.arguments, name=tool.function.name
            )
            tool_calls.append(ToolCall(id=tool.id, function=function, type=tool.type))
        required_action = RequiredAction(submit_tool_outputs=tool_calls)

        return [
            Message(
                run_id=run.id,
                assistant_id=run.assistant_id,
                thread_id=run.thread_id,
                created_at=run.created_at,
                required_action=required_action,
            )
        ]

    def _process_result_without_streaming(self, openai_run: OpenAIRun) -> ResultBase:
        run = openai.beta.threads.runs.retrieve(
            thread_id=openai_run.thread_id, run_id=openai_run.run_id
        )
        while run.status in ["queued", "in_progress"]:
            run = openai.beta.threads.runs.retrieve(
                thread_id=openai_run.thread_id, run_id=openai_run.run_id
            )

        if run.status == "requires_action":
            messages = self._process_required_action_without_streaming(run)
            return ResultBase(
                messages=messages,
                thread_id=openai_run.thread_id,
                run_id=run.id,
                run_status=run.status,
                required_action=messages[0].required_action,
            )

        messages = openai.beta.threads.messages.list(thread_id=openai_run.thread_id)
        processed_messages = self._process_messages_without_streaming(messages)
        usage = {
            "input_tokens": run.usage.prompt_tokens,
            "output_tokens": run.usage.completion_tokens,
        }
        return ResultBase(
            messages=processed_messages,
            thread_id=openai_run.thread_id,
            run_id=run.id,
            usage=usage,
            run_status=run.status,
        )

    def _process_messages_without_streaming(self, messages: list):
        processed_messages = []
        for msg in messages:
            content = []
            attachments = self._process_message_attachments(msg.attachments)
            content = self._process_message_content(msg.content)
            processed_messages.append(
                Message(
                    id=msg.id,
                    assistant_id=msg.assistant_id,
                    object=msg.object,
                    created_at=msg.created_at,
                    thread_id=msg.thread_id,
                    role=msg.role,
                    content=content,
                    attachments=attachments,
                )
            )

        return processed_messages

    def _process_message_attachments(self, attachments: List) -> List[Attachment]:
        processed_attachments = []

        for attachment in attachments:
            tools = []
            for tool in attachment.tools:
                tools.append(Tool(**tool.__dict__))

            processed_attachments.append(
                Attachment(file_id=attachment.file_id, tools=tools)
            )
        return processed_attachments

    def _process_message_content(
        self, content: List
    ) -> list[Union[ImageFileContent, TextContent, RefusalContent, ImageUrlContent]]:
        processed_content = []

        for block in content:
            if getattr(block, "type", "") == "text":
                annotations = []
                for annotation in block.text.annotations:
                    file = File(file_id=annotation.file_path.file_id)
                    file_citation = FileCitation(
                        start_index=annotation.start_index,
                        end_index=annotation.end_index,
                        text=annotation.text,
                    )
                    annotations.append(
                        Annotation(file_citation=file_citation, file=file)
                    )

                processed_content.append(
                    TextContent(
                        text=TextObject(value=block.text.value, annotations=annotations)
                    )
                )

            elif getattr(block, "type", "") == "image_file":
                processed_content.append(
                    ImageFileContent(
                        image_file=ImageFile(
                            file_id=block.image_file.file_id,
                            detail=block.image_file.detail,
                        )
                    )
                )

            elif getattr(block, "type", "") == "refusal":
                processed_content.append(RefusalContent(refusal=block.refusal))

            else:
                processed_content.append(
                    ImageUrlContent(
                        type=block.type,
                        image_url=ImageUrlObject(
                            url=block.image_url.url, detail=block.image_url.detail
                        ),
                    )
                )

        return processed_content

    def upload_file(self, file_path: str) -> str:
        file = openai.files.create(file=open(file_path, "rb"), purpose="assistants")
        return file.id

    def create_new_thread(self) -> str:
        return openai.beta.threads.create().id

    def _add_messages_to_thread(self, message: Message, thread_id):
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role=message.role,
            content=message.content,
            attachments=message.attachments,
        )

    def create_run(self, thread_id: str, assistant_id: str) -> str:
        run = openai.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )
        return run.id

    def submit_tool_outputs(self, params: dict) -> ResultBase:
        """
        Retrieves an existing agent.
        """
        try:
            run_request: RunAgentRequest = self._validate_run_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        tools_outputs = [
            tool.model_dump(exclude="function_name")
            for tool in run_request.tool_outputs
        ]
        run = openai.beta.threads.runs.submit_tool_outputs_and_poll(
            thread_id=run_request.thread_id,
            run_id=run_request.run_id,
            tool_outputs=tools_outputs,
        )

        openai_run = OpenAIRun(
            thread_id=run_request.thread_id,
            assistant_id=run_request.agent_id,
            run_id=run_request.run_id,
        )

        return openai_run
