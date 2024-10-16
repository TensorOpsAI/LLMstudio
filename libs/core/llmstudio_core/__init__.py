from abc import ABC
from typing import Optional
from llmstudio_core.providers.provider import BaseProvider, provider_registry

from llmstudio_core.providers import _load_engine_config

_engine_config = _load_engine_config()


def LLMCore(provider: str, api_key: Optional[str] = None, **kwargs) -> BaseProvider:
    """
    Factory method to create an instance of a provider.

    Args:
        provider (str): The name of the provider.
        api_key (Optional[str], optional): The API key for the provider. Defaults to None.

    Returns:
        BaseProvider: An instance of the provider.

    Raises:
        NotImplementedError: If the provider is not found in the provider map.
    """
    provider_config = _engine_config.providers.get(provider)
    provider_class = provider_registry.get(provider_config.id)
    if provider_class:
        return provider_class(config=provider_config, api_key=api_key, **kwargs)
    raise NotImplementedError(f"Provider not found: {provider_config.id}. Available providers: {str(provider_registry.keys())}")


if __name__ == "__main__":
    from pprint import pprint
    import os
    from dotenv import load_dotenv
    load_dotenv()

    def test_stuff(provider, model, api_key, **kwargs):
        llm = LLMCore(provider=provider, api_key=api_key, **kwargs)

        latencies = {}
        chat_request = {
            "chat_input": "Hello, my name is Json",
            "model": model,
            "is_stream": False,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
                "functions": None,
            }
        }


        
        import asyncio
        response_async = asyncio.run(llm.achat(**chat_request))
        pprint(response_async)
        latencies["async (ms)"]= response_async.metrics["latency_s"]*1000

        # stream
        print("\nasync stream")
        async def async_stream():
            chat_request = {
                "chat_input": "Hello, my name is Json",
                "model": model,
                "is_stream": True,
                "retries": 0,
                "parameters": {
                    "temperature": 0,
                    "max_tokens": 100,
                    "response_format": {"type": "json_object"},
                    "functions": None,
                }
            }
            
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
        chat_request = {
            "chat_input": "Hello, my name is Json",
            "model": model,
            "is_stream": False,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
                "functions": None,
            }
        }
        
        response_sync = llm.chat(**chat_request)
        pprint(response_sync)
        latencies["sync (ms)"]= response_sync.metrics["latency_s"]*1000

        print("# Now sync calls streaming")
        chat_request = {
            "chat_input": "Hello, my name is Json",
            "model": model,
            "is_stream": True,
            "retries": 0,
            "parameters": {
                "temperature": 0,
                "max_tokens": 100,
                "response_format": {"type": "json_object"},
                "functions": None,
            }
        }
        
        response_sync_stream = llm.chat(**chat_request)
        for p in response_sync_stream:
            # pprint(p.chat_output)
            # pprint(p.choices[0].delta.content==p.chat_output)
            # print("metrics: ",p.metrics)
            if p.metrics:
                pprint(p)
                latencies["sync stream (ms)"]= p.metrics["latency_s"]*1000

        print(f"\n\n### REPORT for {provider}, {model} ###")
        return latencies


    provider = "openai"
    model = "gpt-4o-mini"
    for _ in range(1):
        latencies = test_stuff(provider=provider, model=model, api_key=os.environ["OPENAI_API_KEY"])
        pprint(latencies)

    # provider = "anthropic"
    # model = "claude-3-opus-20240229"
    # for _ in range(1):
    #     latencies = test_stuff(provider=provider, model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    #     pprint(latencies)
    # # we need credits

    provider = "azure"
    model = "gpt-4o-mini"
    for _ in range(1):
        latencies = test_stuff(provider=provider, model=model, 
                               api_key=os.environ["AZURE_API_KEY"], 
                               api_version=os.environ["AZURE_API_VERSION"],
                               api_endpoint=os.environ["AZURE_API_ENDPOINT"])
        pprint(latencies)

    provider = "azure"
    model = "Meta-Llama-3.1-70B-Instruct"
    for _ in range(1):
        latencies = test_stuff(provider=provider, model=model, 
                               api_key=os.environ["AZURE_API_KEY_llama"], 
                               base_url=os.environ["AZURE_BASE_URL"]
                               )
        pprint(latencies)


    provider = "vertexai"
    model = "gemini-1.5-pro-latest"
    for _ in range(1):
        latencies = test_stuff(provider=provider, model=model, 
                               api_key=os.environ["GOOGLE_API_KEY"], 
                               )
        pprint(latencies)
