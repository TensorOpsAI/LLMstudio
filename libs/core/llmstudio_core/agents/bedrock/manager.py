import os

import boto3
from llmstudio_core.agents.bedrock.data_models import (
    BedrockAgent,
    BedrockCreateAgentRequest,
    BedrockResult,
    BedrockRun,
)
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.exceptions import AgentError
from pydantic import ValidationError

SERVICE = "bedrock-agent"


@agent_manager
class BedrockAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client(
            SERVICE,
            region_name=self.region if self.region else os.getenv("BEDROCK_REGION"),
            aws_access_key_id=self.access_key
            if self.access_key
            else os.getenv("BEDROCK_ACCESS_KEY"),
            aws_secret_access_key=self.secret_key
            if self.secret_key
            else os.getenv("BEDROCK_SECRET_KEY"),
        )

    @staticmethod
    def _agent_config_name():
        return "bedrock"

    def _validate_create_request(self, request):
        return BedrockCreateAgentRequest(**request)

    def _validate_run_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def _validate_result_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def create_agent(self, **kwargs) -> BedrockAgent:
        """
        Creates a new instance of the agent.
        """
        try:
            agent_request = self._validate_create_request(
                dict(
                    **kwargs,
                )
            )

        except ValidationError as e:
            raise AgentError(str(e))

        bedrock_create = self._client.create_agent(
            agentName=agent_request.name,
            foundationModel=agent_request.model,
            instruction=agent_request.instructions,
            agentResourceRoleArn=agent_request.agent_resource_role_arn,
        )

        agentId = bedrock_create["agent"]["agentId"]

        # Wait for agent to reach 'NOT_PREPARED' status
        agentStatus = ""
        while agentStatus != "NOT_PREPARED":
            response = self._client.get_agent(agentId=agentId)
            agentStatus = response["agent"]["agentStatus"]

        # Configure code interpreter for the agent
        response = self._client.create_agent_action_group(
            actionGroupName="CodeInterpreterAction",
            actionGroupState="ENABLED",
            agentId=agentId,
            agentVersion="DRAFT",
            parentActionGroupSignature="AMAZON.CodeInterpreter",
        )

        actionGroupId = response["agentActionGroup"]["actionGroupId"]

        # Wait for action group to reach 'ENABLED' status
        actionGroupStatus = ""
        while actionGroupStatus != "ENABLED":
            response = self._client.get_agent_action_group(
                agentId=agentId, actionGroupId=actionGroupId, agentVersion="DRAFT"
            )
            actionGroupStatus = response["agentActionGroup"]["actionGroupState"]

        # Prepare the agent for use
        response = self._client.prepare_agent(agentId=agentId)

        # Wait for agent to reach 'PREPARED' status
        agentStatus = ""
        while agentStatus != "PREPARED":
            response = self._client.get_agent(agentId=agentId)
            agentStatus = response["agent"]["agentStatus"]

        # Create an alias for the agent
        response = self._client.create_agent_alias(
            agentAliasName="test", agentId=agentId
        )

        agentAliasId = response["agentAlias"]["agentAliasId"]

        # Wait for agent alias to be prepared
        agentAliasStatus = ""
        while agentAliasStatus != "PREPARED":
            response = self._client.get_agent_alias(
                agentId=agentId, agentAliasId=agentAliasId
            )
            agentAliasStatus = response["agentAlias"]["agentAliasStatus"]

        return BedrockAgent(
            id=agentId,
            created_at=int(bedrock_create["agent"]["createdAt"].timestamp()),
            name=bedrock_create["agent"]["agentName"],
            description=bedrock_create.get("agent", {}).get("description", None),
            model=agent_request.model,
            instructions=bedrock_create["agent"]["instruction"],
            tools=[],
            agent_arn=bedrock_create["agent"]["agentArn"],
            agent_resource_role_arn=bedrock_create["agent"]["agentResourceRoleArn"],
            agent_status=bedrock_create["agent"]["agentStatus"],
            agent_alias=agent_request.agent_alias,
        )

    def run_agent(self, **kwargs) -> BedrockRun:
        """
        Runs the agent
        """
        raise NotImplementedError(
            "Agents need to implement the 'create_thread_and_run' method."
        )

    def retrieve_result(self, **kwargs) -> BedrockResult:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError("Agents need to implement the 'retrieve' method.")
