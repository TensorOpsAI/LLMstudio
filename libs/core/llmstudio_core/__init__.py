from llmstudio_core.providers import (AnthropicProvider
, AzureProvider
, OllamaProvider
, OpenAIProvider
, provider_registry
, VertexAIProvider)

from llmstudio_core.providers.provider import Provider

from llmstudio_core.providers import _load_engine_config

_engine_config = _load_engine_config()

_providers_map = dict(
    anthropic=AnthropicProvider,
    azure=AzureProvider,
    ollama=OllamaProvider,
    openai=OpenAIProvider,
    vertexai=VertexAIProvider
)


def LLM(provider:str, api_key=None) -> Provider:
    provider_config = _engine_config.providers.get(provider)
    provider_class = _providers_map.get(provider_config.id)
    if provider_class:
        return provider_class(config=provider_config, api_key=api_key)
    raise NotImplementedError(f"Provider not found: {provider_config.id}. Available providers: {str(_providers_map.keys())}")

if __name__ == "__main__":

    import os
    import json
    from dotenv import load_dotenv
    load_dotenv()

    chat_request = dict(
        chat_input="Hello?"

        ,api_key=os.environ["OPENAI_API_KEY"]
        ,model="gpt-3.5-turbo"
        ,parameters={"temperature": 0, "max_tokens":100} 
        ,is_stream = False
        ,has_end_token = False
        ,functions = None
        ,session_id = ""
        ,retries=0)
    
    provider = "openai"

    llm = LLM(provider="openai", api_key=os.environ["OPENAI_API_KEY"])

    import asyncio
    a = asyncio.run(llm.chat(chat_request))
    print(a.json)
