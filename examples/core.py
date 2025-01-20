
from llmstudio_core.providers import LLMCore


from pprint import pprint
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

def run_provider(provider, model, api_key, **kwargs):
    print(f"\n\n###RUNNING for <{provider}>, <{model}> ###")
    llm = LLMCore(provider=provider, api_key=api_key, **kwargs)

    latencies = {}
    
    print("\nAsync Non-Stream")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Jason Json", is_stream=False)
    
    response_async = asyncio.run(llm.achat(**chat_request))
    pprint(response_async)
    latencies["async (ms)"]= response_async.metrics["latency_s"]*1000
    
    print("\nAsync Stream")
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
    
    
    print("\nSync Non-Stream")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Alice Json", is_stream=False)
    
    response_sync = llm.chat(**chat_request)
    pprint(response_sync)
    latencies["sync (ms)"]= response_sync.metrics["latency_s"]*1000
    

    print("\nSync Stream")
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


def multiple_provider_runs(provider:str, model:str, num_runs:int, api_key:str, **kwargs):
    for _ in range(num_runs):
        latencies = run_provider(provider=provider, model=model, api_key=api_key, **kwargs)
        pprint(latencies)
    
    

# OpenAI
multiple_provider_runs(provider="openai", model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)
#multiple_provider_runs(provider="openai", model="o1-mini", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)
#multiple_provider_runs(provider="openai", model="o1-preview", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)


# Azure
multiple_provider_runs(provider="azure", model="gpt-4o-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#multiple_provider_runs(provider="azure", model="gpt-4o", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#multiple_provider_runs(provider="azure", model="o1-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#multiple_provider_runs(provider="azure", model="o1-preview", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])


# provider = "anthropic"
# model = "claude-3-opus-20240229"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
#     pprint(latencies)
# # we need credits

#multiple_provider_runs(provider="azure", model="o1-preview", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
#multiple_provider_runs(provider="azure", model="o1-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])


# provider = "vertexai"
# model = "gemini-1.5-pro-latest"
# for _ in range(1):
#     latencies = run_provider(provider=provider, model=model, 
#                             api_key=os.environ["GOOGLE_API_KEY"], 
#                             )
#     pprint(latencies)
