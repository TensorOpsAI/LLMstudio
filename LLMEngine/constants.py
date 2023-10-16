import pydantic
from packaging import version
from vertexai.language_models import TextGenerationModel, ChatModel, CodeGenerationModel, CodeChatModel

LLM_ENGINE_ROUTE_BASE = "/llmengine/"
LLM_ENGINE_HEALTH_ENDPOINT = "/health"


# Change to llmstudio version
VERSION = '0.1.0'


IS_PYDANTIC_V2 = version.parse(pydantic.version.VERSION) >= version.parse("2.0")


END_TOKEN = "<END_TOKEN>"


VERTEXAI_MODEL_MAP = {
            "text-bison": TextGenerationModel,
            "chat-bison": ChatModel,
            "code-bison": CodeGenerationModel,
            "codechat-bison": CodeChatModel
        }

VERTEXAI_INPUT_MAP = {
            TextGenerationModel: 'prompt',
            CodeGenerationModel: 'prefix',
            ChatModel: 'message',
            CodeChatModel: 'message'
        }

VERTEXAI_TOKEN_PRICE = 0.0000005


OPENAI_PRICING_DICT = {
            "gpt-3.5-turbo": {"input_tokens": 0.0000015, "output_tokens": 0.000002},
            "gpt-4": {"input_tokens": 0.00003, "output_tokens": 0.00006},
            "gpt-3.5-turbo-16k": {"input_tokens": 0.00003, "output_tokens": 0.00004},
        }


TITAN_MODELS = ["amazon.titan-tg1-large"]
CLAUDE_MODELS = [
    "anthropic.claude-instant-v1",
    "anthropic.claude-v1",
    "anthropic.claude-v2",
]

BEDROCK_MODELS = TITAN_MODELS + CLAUDE_MODELS