from llmstudio.server import start_server
start_server()

from llmstudio_proxy.provider import LLMProxyProvider


llm = LLMProxyProvider(provider="openai", host="0.0.0.0", port="8001")

result = llm.chat("Write a paragfraph about space", model="gpt-4o")
print(result)


response = llm.chat("Write a paragfraph about space", model="gpt-4o", is_stream=True)
for i, chunk in enumerate(response):
    if i%20==0:
        print("\n")
    if not chunk.metrics:
        print(chunk.chat_output, end="", flush=True)
    else:
        print("\n\n## Metrics:")
        print(chunk.metrics)


import asyncio

# stream
print("\nasync stream")
async def async_stream():
    
    response_async = await llm.achat("Write a paragfraph about space", model="gpt-4o", is_stream=False)
    print(response_async)
    
    response_async_stream = await llm.achat("Write a paragfraph about space", model="gpt-4o", is_stream=True)
    async for p in response_async_stream:
        
        # pprint(p.choices[0].delta.content==p.chat_output)
        # print("metrics: ", p.metrics)
        # print(p)
        if not p.metrics:
            print(p.chat_output, end="", flush=True)
        else:
            print(p.metrics)
asyncio.run(async_stream())
