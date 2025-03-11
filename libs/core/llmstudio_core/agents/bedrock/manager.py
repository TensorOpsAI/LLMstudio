import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import boto3
from llmstudio_core.agents.bedrock.data_models import (
    BedrockAgent,
    BedrockRun,
    BedrockTool,
)
from llmstudio_core.agents.data_models import (
    Attachment,
    CreateAgentRequest,
    ImageFile,
    ImageFileContent,
    Message,
    RequiredAction,
    ResultBase,
    RunAgentRequest,
    TextContent,
    TextObject,
    ToolCall,
    ToolCallFunction,
    ToolOutput,
)
from llmstudio_core.agents.manager import AgentManager, agent_manager
from llmstudio_core.exceptions import AgentError
from pydantic import ValidationError

AGENT_SERVICE = "bedrock-agent"
RUNTIME_SERVICE = "bedrock-agent-runtime"
DEFAULT_LLMSTUDIO_ALIAS = "llmstudio-agent-alias"


@agent_manager
class BedrockAgentManager(AgentManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client(service_name=AGENT_SERVICE)
        self._runtime_client = boto3.client(service_name=RUNTIME_SERVICE)
        self._executor = ThreadPoolExecutor(max_workers=10)

    @staticmethod
    def _agent_config_name():
        return "bedrock"

    def _validate_create_request(self, request):
        return CreateAgentRequest(**request)

    def _validate_run_request(self, request):
        return RunAgentRequest(**request)

    def _validate_result_request(self, request):
        if isinstance(request, BedrockRun):
            return request
        return BedrockRun(**request)

    def _create_action_group(self, params):
        response = self._client.create_agent_action_group(**params)
        actionGroupId = response["agentActionGroup"]["actionGroupId"]

        actionGroupStatus = ""
        while actionGroupStatus != "ENABLED":
            response = self._client.get_agent_action_group(
                agentId=params["agentId"],
                actionGroupId=actionGroupId,
                agentVersion="DRAFT",
            )
            actionGroupStatus = response["agentActionGroup"]["actionGroupState"]

    def create_agent(self, params: dict = None) -> BedrockAgent:
        """
        This method validates the input parameters, creates a new agent using the client,
        waits for the agent to reach the 'NOT_PREPARED' status, adds tools to the agent,
        prepares the agent for use, creates an alias for the agent, and waits for the alias
        to be prepared.

        Args:
           params: Agent creation parameters.

        Returns:
            BedrockAgent: An instance of the created BedrockAgent.

        Raises:
            AgentError: If there is a validation error or if an unsupported tool type is provided.

        """

        try:
            agent_request = self._validate_create_request(params)

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

        bedrock_tools = [
            BedrockTool.from_tool(tool)
            for tool in agent_request.tools
            if tool.type == "function"
        ]

        if any(tool.type == "code_interpreter" for tool in agent_request.tools):
            action_group_code_interpreter_params = {
                "actionGroupName": "CodeInterpreterAction",
                "actionGroupState": "ENABLED",
                "agentId": agentId,
                "agentVersion": "DRAFT",
                "parentActionGroupSignature": "AMAZON.CodeInterpreter",
            }
            self._create_action_group(action_group_code_interpreter_params)

        action_group_tools_params = {
            "actionGroupExecutor": {
                "customControl": "RETURN_CONTROL",
            },
            "actionGroupName": "AgentActionGroup",
            "actionGroupState": "ENABLED",
            "agentId": agentId,
            "agentVersion": "DRAFT",
            "functionSchema": {
                "functions": [tool.model_dump() for tool in bedrock_tools]
            },
        }

        response = self._client.create_agent_action_group(**action_group_tools_params)

        actionGroupId = response["agentActionGroup"]["actionGroupId"]

        actionGroupStatus = ""
        while actionGroupStatus != "ENABLED":
            response = self._client.get_agent_action_group(
                agentId=agentId,
                actionGroupId=actionGroupId,
                agentVersion="DRAFT",
            )
            actionGroupStatus = response["agentActionGroup"]["actionGroupState"]

        # Prepare the agent for use
        response = self._client.prepare_agent(agentId=agentId)

        # Wait for agent to reach 'PREPARED' status
        agentStatus = ""
        while agentStatus != "PREPARED":
            response = self._client.get_agent(agentId=agentId)
            agentStatus = response["agent"]["agentStatus"]

        if not agent_request.agent_alias:
            agent_request.agent_alias = DEFAULT_LLMSTUDIO_ALIAS

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
            agent_id=agentId,
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

    def run_agent(self, params: dict = None) -> BedrockRun:
        """
        Runs the agent with the provided keyword arguments.

        This method validates the run request and invokes the agent using the runtime client.
        If the validation fails, an AgentError is raised.

        Returns:
            BedrockRun: An object containing the agent ID, status, session ID, and response of the run.

        Raises:
            AgentError: If the run request validation fails.
        """

        try:
            run_request = self._validate_run_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        if not run_request.thread_id:
            run_request.thread_id = str(uuid.uuid4())

        sessionState = {"files": [], "conversationHistory": {"messages": []}}

        if isinstance(run_request.messages, Message):
            last_message = run_request.messages
        elif isinstance(run_request.messages, list) and run_request.messages:
            last_message = run_request.messages.pop()

            for message in run_request.messages:
                bedrock_message = {"role": message.role, "content": []}

                # Extract text content
                if isinstance(message.content, str):
                    bedrock_message["content"].append({"text": message.content})

                elif isinstance(message.content, list):
                    for item in message.content:
                        if isinstance(item, TextContent):
                            bedrock_message["content"].append({"text": item.text.value})

                sessionState["conversationHistory"]["messages"].append(bedrock_message)
        else:
            raise AgentError("No valid messages found in the run request")

        for attachment in last_message.attachments:
            if any(tool.type == "code_interpreter" for tool in attachment.tools):
                sessionState["files"].append(
                    {
                        "name": attachment.file_name,
                        "source": {
                            "byteContent": {
                                "data": attachment.file_content,
                                "mediaType": attachment.file_type,
                            },
                            "sourceType": "BYTE_CONTENT",
                        },
                        "useCase": "CODE_INTERPRETER",
                    }
                )

        if isinstance(last_message.content, str):
            input_text = last_message.content  # Use it directly if it's a string
        elif isinstance(last_message.content, list):
            input_text = " ".join(
                item.text
                for item in last_message.content
                if isinstance(item, TextContent)
            )
        else:
            input_text = ""  # Default to an empty string if content is not valid

        loop = asyncio.get_event_loop()
        invoke_request = loop.run_in_executor(
            self._executor,
            partial(
                self._runtime_client.invoke_agent,
                agentId=run_request.agent_id,
                agentAliasId=run_request.alias_id,
                sessionId=run_request.thread_id,
                inputText=input_text,
                sessionState=sessionState,
                enableTrace=True,
            ),
        )

        return BedrockRun(
            agent_id=run_request.agent_id,
            status="completed",
            thread_id=run_request.thread_id,
            response=invoke_request,
        )

    def retrieve_result(self, run):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aretrieve_result(run))

    async def aretrieve_result(self, run) -> ResultBase:
        """
        Retrieve the result based on the provided keyword arguments.
        This method validates the result request and processes the event stream to
        extract content and attachments. It constructs a message with the extracted
        content and attachments and returns it wrapped in a ResultBase object.

        Returns:
            ResultBase: An object containing the constructed message with content and attachments.
        Raises:
            AgentError: If the result request validation fails.
        """

        try:
            run.response = await run.response
            run = self._validate_result_request(run)

        except ValidationError as e:
            raise AgentError(str(e))

        content = []
        attachments = []
        messages = []
        usage = None
        event_stream = run.response.get("completion")
        for event in event_stream:
            if "trace" in event:
                trace = event["trace"]["trace"]["orchestrationTrace"]

                if "modelInvocationInput" in trace:
                    invocation_in = trace["modelInvocationInput"]
                    text = json.loads(invocation_in["text"])
                    new_messages = [
                        Message(content=message["content"], role=message["role"])
                        for message in text["messages"]
                    ]
                    messages += new_messages

                if "modelInvocationOutput" in trace:
                    invocation_out = trace["modelInvocationOutput"]["rawResponse"][
                        "content"
                    ]
                    invocation_out = json.loads(invocation_out)
                    if "metadata" in invocation_out:
                        usage = invocation_out["metadata"]["usage"]
                    elif "usage" in invocation_out:
                        usage = invocation_out["usage"]

                    messages = invocation_out["content"]
                    new_messages = []
                    for message in messages:
                        if message["type"] == "text":
                            new_messages.append(
                                Message(content=message["text"], role="assistant")
                            )

                        elif message["type"] == "tool_use":
                            tool_name = message["name"]
                            tool_arguments = str(message["input"])
                            tool_call_id = message["id"]

                            tool_call_func = ToolCallFunction(
                                arguments=tool_arguments, name=tool_name
                            )
                            tool_call = ToolCall(
                                id=tool_call_id,
                                function=tool_call_func,
                                type=message["type"],
                            )
                            required_action = RequiredAction(
                                submit_tool_outputs=[tool_call]
                            )
                            new_message = Message(required_action=required_action)
                            new_messages.append(new_message)

            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    content.append(
                        TextContent(
                            text=TextObject(value=chunk["bytes"].decode("utf-8"))
                        )
                    )

            if "files" in event:
                files = event["files"]["files"]
                for file in files:
                    if file["type"] == "image/png":
                        content.append(
                            ImageFileContent(
                                image_file=ImageFile(
                                    file_name=file["name"],
                                    file_content=file["bytes"],
                                    file_type=file["type"],
                                )
                            )
                        )
                    else:
                        attachments.append(
                            Attachment(
                                file_name=file["name"],
                                file_content=file["bytes"],
                                file_type=file["type"],
                            )
                        )
            if "returnControl" in event:
                invocation = event["returnControl"]
                invocation_id = invocation["invocationId"]
                tool_invocations = invocation["invocationInputs"]

                required_action = RequiredAction(submit_tool_outputs=[])

                for tool_invocation in tool_invocations:
                    invocation_input = tool_invocation["functionInvocationInput"]
                    name = invocation_input["function"]
                    parameters = invocation_input["parameters"]
                    invocation_type = invocation_input["actionInvocationType"]
                    action_group = invocation_input["actionGroup"]
                    arguments = json.dumps(
                        {
                            parameter["name"]: parameter["value"]
                            for parameter in parameters
                        }
                    )

                    tool_call_function = ToolCallFunction(
                        arguments=arguments, name=name
                    )

                    tool_call = ToolCall(
                        id=invocation_id,
                        function=tool_call_function,
                        type=invocation_type,
                        action_group=action_group,
                        usage=usage,
                    )

                    required_action.submit_tool_outputs.append(tool_call)

                return ResultBase(
                    messages=[
                        Message(
                            thread_id=run.thread_id,
                            required_action=required_action,
                        )
                    ],
                    thread_id=run.thread_id,
                    usage=usage,
                    run_status="requires_action",
                    required_action=required_action,
                )

        messages = new_messages + [
            Message(
                thread_id=run.thread_id,
                role="assistant",
                content=content,
                attachments=attachments,
            )
        ]

        messages.reverse()

        return ResultBase(
            messages=messages,
            thread_id=run.thread_id,
            usage=usage,
            run_status="completed",
        )

    def submit_tool_outputs(self, params: dict = None) -> ResultBase:
        try:
            run_request = self._validate_run_request(params)
        except ValidationError as e:
            raise AgentError(str(e))

        if not run_request.tool_outputs:
            raise AgentError("No tool outputs found")

        tool_outputs: list[ToolOutput] = run_request.tool_outputs

        invocation_results = [
            {
                "functionResult": {
                    "actionGroup": tool_output.action_group,
                    "function": tool_output.function_name,
                    "responseBody": {"TEXT": {"body": tool_output.output}},
                }
            }
            for tool_output in tool_outputs
        ]

        sessionState = {
            "returnControlInvocationResults": invocation_results,
            "invocationId": tool_outputs[0].tool_call_id,
        }

        loop = asyncio.get_event_loop()
        invoke_request = loop.run_in_executor(
            self._executor,
            partial(
                self._runtime_client.invoke_agent,
                agentId=run_request.agent_id,
                agentAliasId=run_request.alias_id,
                sessionId=run_request.thread_id,
                sessionState=sessionState,
                enableTrace=True,
            ),
        )

        return BedrockRun(
            agent_id=run_request.agent_id,
            status="completed",
            thread_id=run_request.thread_id,
            response=invoke_request,
        )
