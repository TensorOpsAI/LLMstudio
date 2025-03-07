import json
import os
from llmstudio_core.agents import AgentManagerCore
from llmstudio_core.agents.data_models import ToolCall, ToolOutput, ResultBase, CreateAgentRequest, RunAgentRequest

agent_manager = AgentManagerCore("bedrock")

# Define a function to get the temperature
def get_temperature(location):
    """Mock function to get temperature - in a real app, you'd call a weather API"""
    mock_data = {
        "new york": {"temperature": 72, "unit": "F"},
        "london": {"temperature": 18, "unit": "C"},
        "tokyo": {"temperature": 25, "unit": "C"},
        "sydney": {"temperature": 22, "unit": "C"},
    }

    location = location.lower()
    if location in mock_data:
        return mock_data[location]
    else:
        return {
            "temperature": 70,
            "unit": "F",
            "note": "Default temperature (location not found)"
        }


agent_request = CreateAgentRequest(
    name="weather-expert-25",
    agent_resource_role_arn="arn:aws:iam::563576320055:role/test-agent-ICNQP",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions=(
        "You are a helpful weather assistant. Use the get_temperature function "
        "to provide weather information when asked about temperatures in specific locations."
    ),
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get the current temperature for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g., 'New York', 'London', 'Tokyo'"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
)

assistant = agent_manager.create_agent(agent_request.model_dump())

print(f"Assistant created with ID: {assistant.agent_id}")


def run_conversation():
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Weather Assistant: Goodbye!")
            break

        run_agent_request = RunAgentRequest.from_agent(
            agent=assistant,
            messages=[
                {"role": "user", "content": user_input},
            ],
        )

        run = agent_manager.run_agent(run_agent_request.model_dump())
        result : ResultBase = agent_manager.retrieve_result(run)

        # Wait for the run to complete
        while True:
            if result.run_status == "completed":
                break

            elif result.run_status == "requires_action":
                # Handle the function call
                tool_calls = result.required_action.submit_tool_outputs
                tool_outputs = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "get_temperature":
                        location = function_args.get("location")
                        temp_result = get_temperature(location)
                        tool_outputs.append(
                            ToolOutput.from_tool_call(
                                tool_call=tool_call,
                                tool_output = json.dumps(temp_result)
                            )
                        )
                # Submit the outputs back to the assistant
                submit_outputs_request = RunAgentRequest.from_agent(
                    agent=assistant,
                    thread_id=result.thread_id,
                    tool_outputs=tool_outputs,
                    run_id=result.run_id
                )
                run = agent_manager.submit_tool_outputs(submit_outputs_request.model_dump())
                result : ResultBase = agent_manager.retrieve_result(run)

            elif result.run_status in ["failed", "cancelled"]:
                print(f"Run ended with status: {result.run_status}")
                break

        # Get the assistant's response
        messages = result.messages

        # Display the assistant's response (newest message first in the list)
        assistant_messages = [m for m in messages if m.role == "assistant"]
        if assistant_messages:
            latest_message = assistant_messages[0]
            print(f"Weather Assistant: {latest_message.content[0].text.value}")

if __name__ == "__main__":
    print("Weather Assistant is ready! Type 'exit', 'quit', or 'bye' to end the conversation.")
    run_conversation()
