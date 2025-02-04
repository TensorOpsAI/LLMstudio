from llmstudio_core.agents.data_models import (
    AgentBase,
    CreateAgentRequest,
    ResultBase,
    RunAgentRequest,
    RunBase,
)


class BedrockAgent(AgentBase):
    agent_resource_role_arn: str
    agent_status: str
    agent_arn: str
    agent_alias_id: str


class BedrockRun(RunBase):
    session_id: str
    response: dict


class BedrockResult(ResultBase):
    session_id: str


class BedrockCreateAgentRequest(CreateAgentRequest):
    agent_resource_role_arn: str
    agent_alias: str
    name: str


class BedrockRunAgentRequest(RunAgentRequest):
    session_id: str
    agent_alias_id: str
