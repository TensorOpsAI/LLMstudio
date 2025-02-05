from typing import Literal, Optional, Union

from pydantic import BaseModel


class Tool(BaseModel):
    type: str


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageFile(BaseModel):
    file_id: Optional[str] = None
    detail: Optional[Literal["low", "high", "auto"]] = "auto"
    file_name: Optional[str] = None  # need this for bedrock
    file_content: Optional[bytes] = None  # need this for bedrock


class ImageFileContent(BaseModel):
    type: Literal["image_file"] = "image_file"
    image_file: ImageFile


class Attachment(BaseModel):
    file_id: Optional[str] = None
    file_name: Optional[str] = None  # need this for bedrock
    file_content: Optional[bytes] = None  # need this for bedrock
    file_type: Optional[str] = None
    tools: list[Tool] = []


class Message(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = "thread.message"
    created_at: Optional[int] = None
    thread_id: Optional[str] = None  # in bedrock represents session_id
    role: Literal["user", "assistant"]
    content: Union[str, list[Union[ImageFileContent, TextContent]]]
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: list[Attachment] = []
    metadata: Optional[dict] = {}


class AgentBase(BaseModel):
    id: str
    created_at: int
    name: str
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: Optional[list[Tool]] = []


class RunBase(BaseModel):
    agent_id: str
    status: str


class ResultBase(BaseModel):
    message: Message


class CreateAgentRequest(BaseModel):
    model: str
    instructions: Optional[str]
    description: Optional[str] = None
    tools: Optional[list[Tool]] = []


class RunAgentRequest(BaseModel):
    agent_id: str
    message: Message


class RetrieveResultRequest(BaseModel):
    run: RunBase
