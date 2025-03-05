from llmstudio_tracker.tracker import TrackingConfig
from llmstudio.server import start_servers
start_servers()

from llmstudio.providers import LLM
from llmstudio_proxy.provider import ProxyConfig
from llmstudio_tracker.prompt_manager.manager import PromptManager
from llmstudio_tracker.prompt_manager.schemas import PromptDefault
# from llmstudio_core.agents import AgentManagerCore
import os 
# from llmstudio_core.agents.data_models import ToolOuput

# instructions = "You are a weather bot. Use the provided functions to answer questions."
# agent_manager = AgentManagerCore("openai",api_key=os.environ["OPENAI_API_KEY"])

tools=[
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
    },
    {
      "type": "function",
      "function": {
        "name": "get_rain_probability",
        "description": "Get the probability of rain for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]

# file_id = agent_manager.upload_file("/Users/brunoalho/Downloads/Tensor87/dfs/original_df.csv")

# params = {
#     "name":"Math tutor",
#     "model":"gpt-4o",
#     "tools":tools,
#     "instructions":instructions,
# }

# agent = agent_manager.create_agent(params)

# messages =[{"role":"user","content":"What's the weather in San Francisco today and the likelihood it'll rain?"}]
# # messages =[{"role":"user","content":"How much is 5+5?"}]

# run = agent_manager.run_agent(agent,messages)

# result = agent_manager.retrieve_result(run)
# print(result)
# tool_outputs = []
# for tool in result.required_action.submit_tools_outputs:
#     if tool.function.name == "get_current_temperature":
#         tool_outputs.append(
#         ToolOuput(tool_call_id=tool.id,
#                   output="57")
#        )
#     elif tool.function.name == "get_rain_probability":
#         tool_outputs.append(
#             ToolOuput(tool_call_id=tool.id,
#                   output="0.06"))
        
# run.tool_outputs = tool_outputs
# result = agent_manager.submit_tool_outputs(run)
# print(result)



from llmstudio_core.providers import LLMCore as LLM
from llmstudio.providers import LLM

llm = LLM(provider="azure", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"),
          session_id="sync")

messages =[{"role":"user","content":"What's the weather in San Francisco today and the likelihood it'll rain?"}]


# import asyncio

# prompt_object = PromptDefault(prompt="textststts",
#                             name="bruni",
#                             label="production",
#                             model="bruno",
#                             provider="bruno",
#                             is_active=True)

# prompt_manager = PromptManager(tracking_config=TrackingConfig(host="0.0.0.0", port="8002"))

# # result = prompt_manager.add_prompt(prompt_object)
# # print(result.text)

import json
# prompt_object = PromptDefault(**json.loads(result.text))
# result = prompt_manager.get_prompt(prompt_id="test")
# print(result.text)

# # result = prompt_manager.get_prompt(name="bruno",
# #                                    model=prompt_object.model,
# #                                    provider=prompt_object.provider)
# # print(result.text)

# prompt_object.prompt="ola teste"
# result = prompt_manager.update_prompt(prompt_object)
# print(result.text)

# result = prompt_manager.delete_prompt(prompt_object)
# print(result)

result = llm.chat("What's the weather in San Francisco today and the likelihood it'll rain?",
                   model="gpt-4o-mini",
                   parameters={"tools":tools}
                   )
print(result)

llm = LLM(provider="self-hosted", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"),
          session_id="sync stream")
response = llm.chat("Write a paragfraph about space", model="gpt-4o-mini", is_stream=True)
for i, chunk in enumerate(response):
    if i%20==0:
        print("\n")
    if not chunk.metrics:
        print(chunk.chat_output_stream, end="", flush=True)
    else:
        print("\n\n## Metrics:")
        print(chunk.metrics)


import asyncio

# stream
print("\nasync stream")

async def async_stream():
    llm = LLM(provider="openai", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"),
          session_id="async stream")
    
    response_async_stream = await llm.achat("Write a paragfraph about space", model="gpt-4o", is_stream=True)
    async for p in response_async_stream:
        
        # pprint(p.choices[0].delta.content==p.chat_output)
        # print("metrics: ", p.metrics)
        # print(p)
        if not p.metrics:
            print(p.chat_output_stream, end="", flush=True)
        else:
            print(p.metrics)
asyncio.run(async_stream())

async def async_chat():
    llm = LLM(provider="openai", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"),
          session_id="async")
    response_async = await llm.achat("Write a paragfraph about space", model="gpt-4o", is_stream=False)
    print(response_async)
    
asyncio.run(async_chat())


async def async_chat():
    llm = LLM(provider="openai", 
          proxy_config=ProxyConfig(host="0.0.0.0", port="8001"),
          tracking_config=TrackingConfig(host="0.0.0.0", port="8002"))
    response_async = await llm.achat("Write a paragfraph about space", model="gpt-4o", is_stream=False)
    print(response_async)
    
asyncio.run(async_chat())