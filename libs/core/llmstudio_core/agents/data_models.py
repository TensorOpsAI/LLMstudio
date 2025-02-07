from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class Tool(BaseModel):
    type: str


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


class Message(BaseModel):
    id: Optional[str] = None
    object: Optional[str] = "thread.message"
    created_at: Optional[int] = None
    thread_id: Optional[str] = None
    session_id: Optional[str] = None
    role: Literal["user", "assistant"]
    content: Union[
        str, list[Union[ImageFileContent, TextContent, RefusalContent, ImageUrlContent]]
    ]
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
    tools: Optional[list[Tool]]


class RunAgentRequest(BaseModel):
    agent_id: str
    message: Union[Message, List[Message]]
