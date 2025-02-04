from datetime import datetime
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
    created_at: int = int(datetime.now().timestamp())
    name: Optional[str] = None
    description: Optional[str] = None
    # model: str
    # instructions: str
    # tools: list[dict]


class RunBase(BaseModel):
    agent_id: Optional[str] = None
    status: Optional[str] = None


class ResultBase(BaseModel):
    messages: list


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
