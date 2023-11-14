export const getChatProvider = (model, uppercase) => {
  const modelToProviderMapping = {
    "gpt-3.5-turbo": "OpenAI",
    "gpt-4": "OpenAI",
    "claude-2": "Anthropic",
    "claude-instant-1": "Anthropic",
    "claude-instant-1.2": "Anthropic",
    "text-bison": "Vertex AI",
    "chat-bison": "Vertex AI",
    "code-bison": "Vertex AI",
    "codechat-bison": "Vertex AI",
    "amazon.titan-tg1-large": "Bedrock",
    "anthropic.claude-v1": "Bedrock",
    "anthropic.claude-instant-v1": "Bedrock",
    "anthropic.claude-v2": "Bedrock",
  };

  const provider = modelToProviderMapping[model];
  if (!provider) return;

  return uppercase ? provider : provider.toLowerCase().replace(" ", "");
};
export const getStatusColor = (status) => {
  if (status === "idle") return "bg-slate-400";
  if (status === "waiting") return "bg-yellow-400";
  if (status === "done") return "bg-green-500";
  if (status === "error") return "bg-red-600";
};
