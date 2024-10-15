from llmstudio.server import start_server
start_server()

from llmstudio.engine.provider import LLMProxyProvider


llm = LLMProxyProvider(provider="openai", host="0.0.0.0", port="8001")

result = llm.chat("What's your name", model="gpt-4o")
print(result)

import asyncio

# stream
print("\nasync stream")
async def async_stream():
    
    response_async = await llm.achat("What's your name", model="gpt-4o", is_stream=True)
    async for p in response_async:
        if "}" in p.chat_output:
            p.chat_output
        print("that: ",p.chat_output)
        # pprint(p.choices[0].delta.content==p.chat_output)
        # print("metrics: ", p.metrics)
        # print(p)
        if p.metrics:
            print(p)
asyncio.run(async_stream())
