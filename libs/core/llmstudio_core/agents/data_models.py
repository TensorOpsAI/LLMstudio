from pydantic import BaseModel
from typing import Optional, Dict, Union, Literal

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
    description: str
    model: str
    instructions: str
    tools: list[Tool]


class RunBase(BaseModel):
    agent_id: str
    status: str


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
    thread_id: Optional[str]
    response: Optional[str]
   
    



    

