from datetime import datetime
from typing import List, Literal, Optional, Union

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
    file_type: Optional[str] = None  # need this for bedrock
    tools: list[Tool] = []


class Message(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = "thread.message"
    created_at: Optional[int] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    role: Literal["user", "assistant"]
    content: Union[str, list[Union[ImageFileContent, TextContent]]]
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: list[Attachment] = []
    metadata: Optional[dict] = {}


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
    tools: Optional[list[Tool]] = []


class RunAgentRequest(BaseModel):
    agent_id: str
    message: Union[Message, List[Message]]
