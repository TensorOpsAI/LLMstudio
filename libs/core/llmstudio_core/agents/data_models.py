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
    name: str
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: Optional[list[Tool]] = None


class RunBase(BaseModel):
    agent_id: str
    status: str


class ResultBase(BaseModel):
    messages: list[Message]


class CreateAgentRequest(BaseModel):
    model: str
    instructions: Optional[str]
    description: Optional[str] = None
    tools: Optional[list[Tool]] = None


class RunAgentRequest(BaseModel):
    assistant_id: str
    message: Message


class RetrieveResultRequest(BaseModel):
    run: RunBase
