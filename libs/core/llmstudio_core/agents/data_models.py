from typing import Literal, Optional, Union

from pydantic import BaseModel


class Tool(BaseModel):
    type: str


class Message(BaseModel):
    created_at: Optional[str]
    role: Literal["user", "assistant"]
    content: Union[str, list]


class AgentBase(BaseModel):
    id: str
    created_at: int
    name: Optional[str]
    description: Optional[str]
    model: str
    instructions: str
    tools: list[dict]


class RunBase(BaseModel):
    agent_id: Optional[str]
    status: Optional[str]


class ResultBase(BaseModel):
    messages: list[Message]


class CreateAgentRequest(BaseModel):
    model: str
    instructions: Optional[str]
    description: Optional[str]
    tools: Optional[list[Tool]]


class RunAgentRequest(BaseModel):
    assistant_id: str
    message: Message


class RetrieveResultRequest(BaseModel):
    run: RunBase
