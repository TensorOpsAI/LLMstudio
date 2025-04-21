
from llmstudio_core.providers import LLMCore


from pprint import pprint
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

def run_provider(provider, model, api_key=None, **kwargs):
    print(f"\n\n###RUNNING for <{provider}>, <{model}> ###")
    llm = LLMCore(provider=provider, api_key=api_key, **kwargs)

    latencies = {}    
    print("\nAsync Non-Stream")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Jason", is_stream=False)
    string = """
What is Lorem Ipsum? 
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

What is Lorem Ipsum? 
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

What is Lorem Ipsum? 
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
    
    """
    #chat_request = build_chat_request(model, chat_input=string, is_stream=False)
    
    response_async = asyncio.run(llm.achat(**chat_request))
    pprint(response_async)
    latencies["async (ms)"]= response_async.metrics["latency_s"]*1000
    
    
    print("\nAsync Stream")
    async def async_stream():
        chat_request = build_chat_request(model, chat_input="Hello, my name is Tom", is_stream=True)
        
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
    chat_request = build_chat_request(model, chat_input="Hello, my name is Alice", is_stream=False)
    
    response_sync = llm.chat(**chat_request)
    pprint(response_sync)
    latencies["sync (ms)"]= response_sync.metrics["latency_s"]*1000
    

    print("\nSync Stream")
    chat_request = build_chat_request(model, chat_input="Hello, my name is Mary", is_stream=True)

    
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
    if model.startswith(('o1', 'o3')):
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "max_completion_tokens": max_tokens
            }
        }
    elif 'amazon.nova' in model or 'anthropic.claude' in model:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {
                "maxTokens": max_tokens
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
                "functions": None,
            }
        }
    return chat_request


def multiple_provider_runs(provider:str, model:str, num_runs:int, api_key:str, **kwargs):
    for _ in range(num_runs):
        latencies = run_provider(provider=provider, model=model, api_key=api_key, **kwargs)
        pprint(latencies)

        
def run_chat_all_providers():    
    # OpenAI
    multiple_provider_runs(provider="openai", model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)
    multiple_provider_runs(provider="openai", model="o3-mini", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)
    #multiple_provider_runs(provider="openai", model="o1-preview", api_key=os.environ["OPENAI_API_KEY"], num_runs=1)

    # Azure
    multiple_provider_runs(provider="azure", model="gpt-4o-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="gpt-4o", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="o1-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="o1-preview", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])

    # Azure
    multiple_provider_runs(provider="azure", model="gpt-4o-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="gpt-4o", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="o1-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="o1-preview", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])



    #multiple_provider_runs(provider="anthropic", model="claude-3-opus-20240229", num_runs=1, api_key=os.environ["ANTHROPIC_API_KEY"])

    #multiple_provider_runs(provider="azure", model="o1-preview", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])
    #multiple_provider_runs(provider="azure", model="o1-mini", num_runs=1, api_key=os.environ["AZURE_API_KEY"], api_version=os.environ["AZURE_API_VERSION"], api_endpoint=os.environ["AZURE_API_ENDPOINT"])


    multiple_provider_runs(provider="vertexai", model="gemini-1.5-flash", num_runs=1, api_key=os.environ["GOOGLE_API_KEY"])

    # Bedrock
    multiple_provider_runs(provider="bedrock", model="us.amazon.nova-lite-v1:0", num_runs=1, api_key=None, region=os.environ["BEDROCK_REGION"], secret_key=os.environ["BEDROCK_SECRET_KEY"], access_key=os.environ["BEDROCK_ACCESS_KEY"])
    #multiple_provider_runs(provider="bedrock", model="anthropic.claude-3-5-sonnet-20241022-v2:0", num_runs=1, api_key=None, region=os.environ["BEDROCK_REGION"], secret_key=os.environ["BEDROCK_SECRET_KEY"], access_key=os.environ["BEDROCK_ACCESS_KEY"])

run_chat_all_providers()


import base64

def messages(img_path):
    """
    Creates a message payload with both text and image.
    Adapts format based on the provider.
    """
    with open(img_path, "rb") as f:
        image_bytes = f.read()
        
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://awsmp-logos.s3.amazonaws.com/seller-zx4pk43qpmxoa/53d235806f343cec94aac3c577d81c13.png"},
                },
            ],
        }
    ]

def run_send_imgs():
    provider="bedrock"
    model="us.amazon.nova-lite-v1:0"
    chat_input=messages(img_path="./libs/llmstudio/tests/integration_tests/test_data/llmstudio-logo.jpeg")
    chat_request = build_chat_request(model=model, chat_input=chat_input, is_stream=False)
    llm = LLMCore(provider=provider, api_key=os.environ["OPENAI_API_KEY"], region=os.environ["BEDROCK_REGION"], secret_key=os.environ["BEDROCK_SECRET_KEY"], access_key=os.environ["BEDROCK_ACCESS_KEY"])
    response_sync = llm.chat(**chat_request)
    #print(response_sync)
    response_sync.clean_print()
    
    #for p in response_sync:
    #    if p.metrics:
    #        p.clean_print()
    
run_send_imgs()
