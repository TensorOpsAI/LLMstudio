import os
from llmstudio_core.agents import AgentManagerCore
from llmstudio_core.agents.bedrock.data_models import BedrockCreateAgentRequest, BedrockRunAgentRequest, BedrockToolOutput, BedrockToolCall
from llmstudio_core.agents.data_models import ResultBase, ToolCall, ToolOutput
import boto3
import uuid

bedrock_agent_manager = AgentManagerCore("bedrock")

agent_prompt = """You are an advanced AI agent with capabilities in code execution, chart generation, and complex data analysis. Your primary function is to assist users by solving problems and fulfilling requests through these capabilities. Here are your key attributes and instructions:

Code Execution:

You have access to a Python environment where you can write and execute code in real-time.
When asked to perform calculations or data manipulations, always use this code execution capability to ensure accuracy.
After executing code, report the exact output and explain the results.


Data Analysis:

You excel at complex data analysis tasks. This includes statistical analysis, data visualization, and machine learning applications.
Approach data analysis tasks systematically: understand the problem, prepare the data, perform the analysis, and interpret the results.


Problem-Solving Approach:

When presented with a problem or request, break it down into steps.
Clearly communicate your thought process and the steps you're taking.
If a task requires multiple steps or tools, outline your approach before beginning.
"""

tools = [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["Celsius", "Fahrenheit"],
              "description": "The temperature unit to use. Infer this from the user's location."
            }
          },
          "required": ["location", "unit"]
        }
      }
    }
    ]

agent_request = BedrockCreateAgentRequest(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions=agent_prompt,
    tools=tools,
    name=f"test-agent-{uuid.uuid4()}", #BEDROCK EXCLUSIVE
    agent_alias="test-alias", # BEDROCK EXCLUSIVE
    agent_resource_role_arn="arn:aws:iam::563576320055:role/test-agent-ICNQP" # BEDROCK EXCLUSIVE
)

agent = bedrock_agent_manager.create_agent(agent_request.model_dump())


run_agent_request = BedrockRunAgentRequest(
  agent_id = agent.agent_id,
  thread_id="9999",
  alias_id=agent.agent_alias_id,
  messages=[
      {"role": "user", "content": "What is the weather like in Lisbon, PT?"},
  ]
)

run = bedrock_agent_manager.run_agent(run_agent_request.model_dump())

result : ResultBase = bedrock_agent_manager.retrieve_result(run)

if not result.messages[-1].required_action:
    print(result.messages[-1].content)
else:        
    tool_calls : list[BedrockToolCall] = result.messages[-1].required_action.submit_tools_outputs
    
    submit_outputs_request = BedrockRunAgentRequest(
        agent_id=agent.agent_id,
        thread_id=result.thread_id,
        alias_id=agent.agent_alias_id,
        tool_outputs=[]
    )

    for tool_call in tool_calls:
        submit_outputs_request.tool_outputs.append(BedrockToolOutput(tool_call_id=tool_call.id, output="10", action_group=tool_call.action_group, function_name=tool_call.function.name))

    outputs_request = submit_outputs_request.model_dump()
    run = bedrock_agent_manager.submit_tool_outputs(submit_outputs_request.model_dump())
    result : ResultBase = bedrock_agent_manager.retrieve_result(run)
    print(result.messages[-1].content)