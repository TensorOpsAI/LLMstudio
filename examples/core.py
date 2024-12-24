
from llmstudio_core.providers import LLMCore


from pprint import pprint
import os
from dotenv import load_dotenv
import asyncio
load_dotenv()

def run_provider(provider, model, api_key, **kwargs):

    llm = LLMCore(provider=provider, api_key=api_key, **kwargs)
    latencies = {}

    """
    chat_request = build_chat_request(model, chat_input="Hello, my name is Jason Json", is_stream=False)
    
    response_async = asyncio.run(llm.achat(**chat_request))
    pprint(response_async)
    latencies["async (ms)"]= response_async.metrics["latency_s"]*1000
    """

    # stream
    print("\nasync stream")
    async def async_stream():
        chat_request = build_chat_request(model, chat_input="Hello, my name is Tom Json", is_stream=True)
        
        response_async = await llm.achat(**chat_request)
        async for p in response_async:
            if not p.metrics:
                print("that: ",p.chat_output_stream)
            # pprint(p.choices[0].delta.content==p.chat_output)
            # print("metrics: ", p.metrics)
            # print(p)
            if p.metrics:
                pprint(p)
                latencies["async_stream (ms)"]= p.metrics["latency_s"]*1000
    asyncio.run(async_stream())
    
    
    print("# Now sync calls")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Alice Json", is_stream=False)
    
    response_sync = llm.chat(**chat_request)
    pprint(response_sync)
    latencies["sync (ms)"]= response_sync.metrics["latency_s"]*1000

    print("# Now sync calls streaming")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Mary Json", is_stream=True)
    
    response_sync_stream = llm.chat(**chat_request)
    for p in response_sync_stream:
        # pprint(p.chat_output)
        # pprint(p.choices[0].delta.content==p.chat_output)
        # print("metrics: ",p.metrics)
        if p.metrics:
            pprint(p)
            latencies["sync stream (ms)"]= p.metrics["latency_s"]*1000

    print(f"\n\n###REPORT for <{provider}>, <{model}> ###")
    return latencies

def build_chat_request(model: str, chat_input: str, is_stream: bool, max_tokens: int=1000):
    if model == "o1-preview" or model == "o1-mini":
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "max_completion_tokens": max_tokens
            }
        }
    else:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "functions": None,
            }
        }
    return chat_request





provider = "openai"
model = "gpt-4o-mini"
for _ in range(1):
    latencies = run_provider(provider=provider, model=model, api_key=os.environ["OPENAI_API_KEY"])
    pprint(latencies)

    
provider = "openai"
model = "o1-preview"
for _ in range(1):
    latencies = run_provider(provider=provider, model=model, api_key=os.environ["OPENAI_API_KEY"])
    pprint(latencies)
    
provider = "openai"
model = "o1-mini"
for _ in range(1):
    latencies = run_provider(provider=provider, model=model, api_key=os.environ["OPENAI_API_KEY"])
    pprint(latencies)

# provider = "anthropic"
# model = "claude-3-opus-20240229"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
#     pprint(latencies)
# # we need credits

"""     
provider = "azure"
model = "gpt-4o-mini"
for _ in range(1):
    latencies = run_provider(provider=provider, model=model, 
                            api_key=os.environ["AZURE_API_KEY"], 
                            api_version=os.environ["AZURE_API_VERSION"],
                            api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    pprint(latencies)
"""
    
# provider = "azure"
# model = "o1-preview"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, 
#                             api_key=os.environ["AZURE_API_KEY"], 
#                             api_version=os.environ["AZURE_API_VERSION"],
#                             api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#     pprint(latencies)
# # we need a deployment
 
    
# provider = "azure"
# model = "o1-mini"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, 
#                             api_key=os.environ["AZURE_API_KEY"], 
#                             api_version=os.environ["AZURE_API_VERSION"],
#                             api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#     pprint(latencies)
# # we need a deployment

# provider = "azure"
# model = "gpt-4o"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, 
#                             api_key=os.environ["AZURE_API_KEY_llama"], 
#                             base_url=os.environ["AZURE_BASE_URL"]
#                             )
#     pprint(latencies)


# provider = "vertexai"
# model = "gemini-1.5-pro-latest"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, 
#                             api_key=os.environ["GOOGLE_API_KEY"], 
#                             )
#     pprint(latencies)
