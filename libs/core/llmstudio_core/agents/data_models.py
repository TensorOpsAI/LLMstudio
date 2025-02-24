from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


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


class ToolCall(BaseModel):
    id: str
    function: ToolCallFunction
    type: str


class RequiredAction(BaseModel):
    submit_tools_outputs: List[ToolCall]
    type: Literal["submit_tool_outputs"] = "submit_tool_outputs"


class ToolOuput(BaseModel):
    tool_call_id: str
    output: str


class Message(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = "thread.message"
    created_at: Optional[int] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
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
    required_action: RequiredAction = {}


class AgentBase(BaseModel):
    agent_id: str
    created_at: Optional[int] = int(datetime.now().timestamp())
    name: Optional[str] = None
    description: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list[Tool]] = []


class RunBase(BaseModel):
    agent_id: Optional[str] = None


class ResultBase(BaseModel):
    messages: List[Message]


class CreateAgentRequest(BaseModel):
    model: str
    instructions: Optional[str]
    description: Optional[str] = None
    tools: Optional[list[Tool]]


class RunAgentRequest(BaseModel):
    agent: AgentBase
    messages: Union[Message, List[Message]]
