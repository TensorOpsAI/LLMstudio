import os

import boto3
from llmstudio_core.agents.bedrock.data_models import (
    BedrockAgent,
    BedrockCreateAgentRequest,
    BedrockResult,
    BedrockRun,
    BedrockRunAgentRequest,
)
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.exceptions import AgentError
from pydantic import ValidationError

AGENT_SERVICE = "bedrock-agent"
RUNTIME_SERVICE = "bedrock-agent-runtime"


@agent_manager
class BedrockAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client(
            service_name=AGENT_SERVICE,
            region_name=self.region if self.region else os.getenv("BEDROCK_REGION"),
            aws_access_key_id=self.access_key
            if self.access_key
            else os.getenv("BEDROCK_ACCESS_KEY"),
            aws_secret_access_key=self.secret_key
            if self.secret_key
            else os.getenv("BEDROCK_SECRET_KEY"),
        )
        self._runtime_client = boto3.client(
            service_name=RUNTIME_SERVICE,
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
        return BedrockRunAgentRequest(**request)

    def _validate_result_request(self, request):
        raise NotImplementedError("Agents need to implement the method")

    def create_agent(self, **kwargs) -> BedrockAgent:
        """
        This method validates the input parameters, creates a new agent using the client,
        waits for the agent to reach the 'NOT_PREPARED' status, adds tools to the agent,
        prepares the agent for use, creates an alias for the agent, and waits for the alias
        to be prepared.

        Args:
            **kwargs: Agent creation parameters.

        Returns:
            BedrockAgent: An instance of the created BedrockAgent.

        Raises:
            AgentError: If there is a validation error or if an unsupported tool type is provided.

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

        # Add tools to the agent
        for tool in agent_request.tools:
            if tool.type == "code_interpreter":
                response = self._client.create_agent_action_group(
                    actionGroupName="CodeInterpreterAction",
                    actionGroupState="ENABLED",
                    agentId=agentId,
                    agentVersion="DRAFT",
                    parentActionGroupSignature="AMAZON.CodeInterpreter",
                )

                actionGroupId = response["agentActionGroup"]["actionGroupId"]

                actionGroupStatus = ""
                while actionGroupStatus != "ENABLED":
                    response = self._client.get_agent_action_group(
                        agentId=agentId,
                        actionGroupId=actionGroupId,
                        agentVersion="DRAFT",
                    )
                    actionGroupStatus = response["agentActionGroup"]["actionGroupState"]
            else:
                raise AgentError(f"Tool {tool.get('type')} not supported")

        # Prepare the agent for use
        response = self._client.prepare_agent(agentId=agentId)

        # Wait for agent to reach 'PREPARED' status
        agentStatus = ""
        while agentStatus != "PREPARED":
            response = self._client.get_agent(agentId=agentId)
            agentStatus = response["agent"]["agentStatus"]

        # Create an alias for the agent
        response = self._client.create_agent_alias(
            agentAliasName=agent_request.agent_alias, agentId=agentId
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
            tools=agent_request.tools,
            agent_arn=bedrock_create["agent"]["agentArn"],
            agent_resource_role_arn=bedrock_create["agent"]["agentResourceRoleArn"],
            agent_status=bedrock_create["agent"]["agentStatus"],
            agent_alias_id=agentAliasId,
        )

    def run_agent(self, **kwargs) -> BedrockRun:
        """
        Runs the agent
        """
        try:
            run_request = self._validate_run_request(
                dict(
                    **kwargs,
                )
            )
        except ValidationError as e:
            raise AgentError(str(e))

        invoke_request = self._runtime_client.invoke_agent(
            agentId=run_request.agent_id,
            agentAliasId=run_request.agent_alias_id,
            sessionId=run_request.session_id,
            inputText=run_request.message.content,
        )

        return BedrockRun(
            agent_id=run_request.agent_id,
            status="completed",
            session_id=run_request.session_id,
            response=invoke_request,
        )

    def retrieve_result(self, **kwargs) -> BedrockResult:
        """
        Retrieves an existing agent.
        """
        raise NotImplementedError("Agents need to implement the 'retrieve' method.")
