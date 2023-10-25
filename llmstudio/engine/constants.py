ENGINE_ROUTE_BASE = "/api/engine/"
ENGINE_HEALTH_ENDPOINT = "/health"


# Change to llmstudio version
VERSION = "0.1.0"


END_TOKEN = "<END_TOKEN>"

VERTEXAI_TOKEN_PRICE = 0.0000005

OPENAI_MAX_RETRIES = 2

GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"

GPT_35_MAX_TOKENS = 4096
GPT_4_MAX_TOKENS = 8196

DEFAULT_OUTPUT_MARGIN = 0.2

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
