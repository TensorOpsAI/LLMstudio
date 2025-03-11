from typing import Awaitable, Literal

from llmstudio_core.agents.data_models import AgentBase, RunBase, Tool
from pydantic import BaseModel


class BedrockAgent(AgentBase):
    agent_resource_role_arn: str
    agent_status: str
    agent_arn: str


class BedrockRun(RunBase):
    response: Awaitable

    class Config:
        arbitrary_types_allowed = True


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
