import uuid
from typing import List, Literal, Optional

from llmstudio_core.agents.data_models import (
    AgentBase,
    CreateAgentRequest,
    RunAgentRequest,
    RunBase,
    Tool,
)
from pydantic import BaseModel, Field


class BedrockAgent(AgentBase):
    agent_resource_role_arn: str
    agent_status: str
    agent_arn: str
    agent_alias_id: str


class BedrockRun(RunBase):
    session_id: str
    response: dict


class BedrockCreateAgentRequest(CreateAgentRequest):
    agent_resource_role_arn: str
    agent_alias: str
    name: str
    tools: List[Tool]


class BedrockRunAgentRequest(RunAgentRequest):
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    alias_id: str


class BedrockToolProperty(BaseModel):
    description: str
    required: bool = False
    type: Literal["string", "number", "integer", "boolean", "array"]


class BedrockTool(BaseModel):
    description: str
    name: str
    parameters: dict[str, BedrockToolProperty]
    requireConfirmation: Literal["ENABLED", "DISABLED"] = "DISABLED"

    @classmethod
    def from_tool(cls, tool: Tool) -> "BedrockTool":

        name = tool.function.name
        description = tool.function.name
        parameters = {
            property_name: BedrockToolProperty(
                description=property_dict["description"], type=property_dict["type"]
            )
            for property_name, property_dict in tool.function.parameters.properties.items()
        }

        for required_property in tool.function.parameters.required:
            parameters[required_property].required = True

        return cls(
            description=description,
            name=name,
            parameters=parameters,
        )
