from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class ExcludeNoneBaseModel(BaseModel):
    def model_dump(self, **kwargs):
        # Ensure exclude_none is set to True by default
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class ToolResources(BaseModel):
    file_ids: Optional[List[str]] = None  # For code_interpreter
    vector_store_ids: Optional[List[str]] = None  # For file_search


class Parameters(BaseModel):
    type: str
    properties: Dict
    required: List[str]


class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters


class Tool(BaseModel):
    type: str
    function: Optional[Function] = None


class FileCitation(BaseModel):
    start_index: int
    end_index: int
    text: str


class File(BaseModel):
    type: str = "file_path"
    file_id: Optional[str] = None
    file_content: Optional[bytes] = None
    file_name: Optional[str] = None


class Annotation(BaseModel):
    file_citation: Optional[FileCitation] = None
    file_path: Optional[File] = None


class TextObject(BaseModel):
    value: str
    annotations: List[Annotation] = []


class ImageUrlObject(BaseModel):
    url: Literal["jpeg", "jpg", "png", "gif", "webp"]
    detail: Optional[Literal["low", "high", "auto"]] = "auto"


class ImageUrlContent(BaseModel):
    type: str
    image_url: ImageUrlObject


class RefusalContent(BaseModel):
    type: Literal["refusal"]
    refusal: str


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: TextObject


class ImageFile(BaseModel):
    file_id: Optional[str] = None
    detail: Optional[Literal["low", "high", "auto"]] = "auto"
    file_name: Optional[str] = None
    file_content: Optional[bytes] = None


class ImageFileContent(BaseModel):
    type: Literal["image_file"] = "image_file"
    image_file: ImageFile


class Attachment(BaseModel):
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    file_content: Optional[bytes] = None
    file_type: Optional[str] = None
    tools: list[Tool] = None


class InputMessageBase(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ToolCallFunction(BaseModel):
    arguments: str
    name: str


class ToolCall(ExcludeNoneBaseModel):
    id: str
    function: ToolCallFunction
    type: str
    action_group: Optional[str] = None


class RequiredAction(BaseModel):
    submit_tool_outputs: List[ToolCall]
    type: Literal["submit_tool_outputs"] = "submit_tool_outputs"


class ToolOutput(ExcludeNoneBaseModel):
    tool_call_id: Optional[str] = None
    output: Optional[str] = None
    action_group: Optional[str] = None
    function_name: Optional[str] = None

    @classmethod
    def from_tool_call(cls, tool_call: ToolCall, tool_output: str) -> "ToolOutput":
        return cls(
            output=tool_output,
            tool_call_id=tool_call.id,
            function_name=tool_call.function.name,
            action_group=tool_call.action_group,
        )


class Message(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = "thread.message"
    created_at: Optional[int] = None
    thread_id: Optional[str] = None
    role: Optional[Literal["user", "assistant"]] = None
    content: Optional[
        Union[
            str,
            list[Union[ImageFileContent, TextContent, RefusalContent, ImageUrlContent]],
        ]
    ] = []
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: list[Attachment] = []
    metadata: Optional[dict] = {}
    required_action: Optional[RequiredAction] = None


class AgentBase(BaseModel):
    agent_id: str
    created_at: Optional[int] = int(datetime.now().timestamp())
    name: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[Tool]] = []
    agent_alias_id: Optional[str] = None


class RunBase(BaseModel):
    thread_id: str
    agent_id: Optional[str] = None


class ResultBase(BaseModel):
    thread_id: str
    messages: List[Message]
    run_id: Optional[str] = None
    usage: Optional[dict] = None
    run_status: Optional[str] = None
    required_action: Optional[RequiredAction] = None


class CreateAgentRequest(ExcludeNoneBaseModel):
    model: str
    instructions: Optional[str]
    description: Optional[str] = None
    tools: Optional[list[Tool]]
    name: Optional[str] = None
    tool_resources: Optional[ToolResources] = None
    agent_resource_role_arn: Optional[str] = None
    agent_alias: Optional[str] = None


class RunAgentRequest(ExcludeNoneBaseModel):
    agent_id: str
    alias_id: Optional[str] = None
    thread_id: Optional[str] = None
    messages: Optional[List[Message]] = None
    tool_outputs: Optional[List[ToolOutput]] = None
    run_id: Optional[str] = None

    @classmethod
    def from_agent(
        cls,
        agent: AgentBase,
        thread_id: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        tool_outputs: Optional[List[ToolOutput]] = None,
        run_id: Optional[str] = None,
    ):
        return cls(
            agent_id=agent.agent_id,
            alias_id=agent.agent_alias_id,
            thread_id=thread_id,
            messages=messages,
            tool_outputs=tool_outputs,
            run_id=run_id,
        )
