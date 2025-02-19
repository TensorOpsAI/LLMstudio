import os

from dotenv import load_dotenv
from llmstudio_core.providers import LLMCore

load_dotenv()


def build_chat_request(
    model: str, chat_input: str, is_stream: bool, max_tokens: int = 1000
):
    if model.startswith(("o1", "o3")):
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {"max_completion_tokens": max_tokens},
        }
    elif "amazon.nova" in model or "anthropic.claude" in model:
        chat_request = {
            "chat_input": chat_input,
            "model": model,
            "is_stream": is_stream,
            "retries": 0,
            "parameters": {"maxTokens": max_tokens},
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
            },
        }
    return chat_request


input_image = "./libs/llmstudio/tests/integration_tests/test_data/llmstudio-logo.jpeg"
messages = []
with open(input_image, "rb") as f:
    image = f.read()

    message = {
        "role": "user",
        "content": [
            {"text": "What's in this image?"},
            {"image": {"format": "jpeg", "source": {"bytes": image}}},
        ],
    }
    messages = [message]

llm = LLMCore(
    provider="bedrock",
    api_key=None,
    region=os.environ["BEDROCK_REGION"],
    secret_key=os.environ["BEDROCK_SECRET_KEY"],
    access_key=os.environ["BEDROCK_ACCESS_KEY"],
)
chat_request = build_chat_request(
    model="us.amazon.nova-lite-v1:0",
    chat_input=messages,
    max_tokens=100,
    is_stream=False,
)
response_sync = llm.chat(**chat_request)

print(response_sync)
print(response_sync.chat_output)


# chat_request = build_chat_request(model="anthropic.claude-3-5-sonnet-20241022-v2:0", chat_input=messages, max_tokens=100, is_stream=False)

# for p in response_sync:
#    if p.metrics:
#        pprint(p)
#        pprint(p.chat_output)
