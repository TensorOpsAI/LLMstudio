export const getChatProvider = (modelName, uppercase) => {
  const modelToProviderMapping = {
    "gpt-3.5-turbo": "OpenAI",
    "gpt-4": "OpenAI",
    "text-bison@001": "Vertex AI",
    "chat-bison@001": "Vertex AI",
    "code-bison@001": "Vertex AI",
    "codechat-bison@001": "Vertex AI",
    "amazon.titan-tg1-large": "Bedrock",
    "anthropic.claude-v1": "Bedrock",
    "anthropic.claude-instant-v1": "Bedrock",
    "anthropic.claude-v2": "Bedrock",
  };

  const provider = modelToProviderMapping[modelName];
  if (!provider) return;

  return uppercase ? provider : provider.toLowerCase().replace(" ", "");
};
export const getStatusColor = (status) => {
  if (status === "idle") return "bg-slate-400";
  if (status === "waiting") return "bg-yellow-400";
  if (status === "done") return "bg-green-500";
  if (status === "error") return "bg-red-600";
};
