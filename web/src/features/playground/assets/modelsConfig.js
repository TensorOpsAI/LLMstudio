export const types = ["OpenAI", "Vertex AI", "Bedrock"];

export const models = [
  {
    id: "gpt-3.5-turbo",
    name: "gpt-3.5-turbo",
    description:
      "Most capable OpenAI model. Can do any task the other models can do, often with higher quality, longer output and better instruction-following. Also supports inserting completions within text.",
    type: "OpenAI",
    strengths:
      "Complex intent, cause and effect, creative generation, search, summarization for audience",
    cost: "Complex intent, cause and effect, creative generation, search, summarization for audience",
  },
  {
    id: "gpt-4",
    name: "gpt-4",
    description: "Very capable, but faster and lower cost than Davinci.",
    type: "OpenAI",
    strengths:
      "Language translation, complex classification, sentiment, summarization",
    cost: "Complex intent, cause and effect, creative generation, search, summarization for audience",
  },
  {
    id: "text-bison@001",
    name: "text-bison@001",
    description:
      "Most capable Vertex AI model. Particularly good at translating natural language to code. In addition to completing code, also supports inserting completions within code.",
    type: "Vertex AI",
    strengths:
      "Language translation, complex classification, sentiment, summarization",
    cost: "Complex intent, cause and effect, creative generation, search, summarization for audience",
  },
  {
    id: "chat-bison@001",
    name: "chat-bison@001",
    description:
      "Slightly faster. This speed advantage may make it preferable for real-time applications.",
    type: "Vertex AI",
    strengths: "Real-time application where low-latency is preferable",
    cost: "Complex intent, cause and effect, creative generation, search, summarization for audience",
  },
  {
    id: "code-bison@001",
    name: "code-bison@001",
    description: "dummy",
    type: "Vertex AI",
    strengths: "dummy",
    cost: "dummy",
  },
  {
    id: "codechat-bison@001",
    name: "codechat-bison@001",
    description: "dummy",
    type: "Vertex AI",
    strengths: "dummy",
    cost: "dummy",
  },
  {
    id: "amazon.titan-tg1-large",
    name: "amazon.titan-tg1-large",
    description: "dummy",
    type: "Bedrock",
    strengths: "dummy",
    cost: "dummy",
  },
  {
    id: "anthropic.claude-instant-v1",
    name: "anthropic.claude-instant-v1",
    description: "dummy",
    type: "Bedrock",
    strengths: "dummy",
    cost: "dummy",
  },
  {
    id: "anthropic.claude-v1",
    name: "anthropic.claude-v1",
    description: "dummy",
    type: "Bedrock",
    strengths: "dummy",
    cost: "dummy",
  },
  {
    id: "anthropic.claude-v2",
    name: "anthropic.claude-v2",
    description: "dummy",
    type: "Bedrock",
    strengths: "dummy",
    cost: "dummy",
  },
];

export const credentials = {
  "gpt-3.5-turbo": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "gpt-4": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "text-bison@001": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "chat-bison@001": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "code-bison@001": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "codechat-bison@001": {
    needsKey: true,
    needsSecret: false,
    needsRegion: false,
  },
  "amazon.titan-tg1-large": {
    needsKey: true,
    needsSecret: true,
    needsRegion: true,
  },
  "anthropic.claude-instant-v1": {
    needsKey: true,
    needsSecret: true,
    needsRegion: true,
  },
  "anthropic.claude-v1": {
    needsKey: true,
    needsSecret: true,
    needsRegion: true,
  },
  "anthropic.claude-v2": {
    needsKey: true,
    needsSecret: true,
    needsRegion: true,
  },
};

export const parameters = {
  openai: [
    {
      id: "temperature",
      name: "Temperature",
      models: ["gpt-3.5-turbo", "gpt-4"],
      defaultValue: 1,
      min: 0,
      max: 2,
      step: 0.1,
      description:
        "Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.",
    },
    {
      id: "maxTokens",
      name: "Maximum length",
      models: ["gpt-3.5-turbo", "gpt-4"],
      defaultValue: 256,
      min: 1,
      max: 2048,
      step: 8,
      description:
        "The maximum number of tokens to generate. Requests can use up to 2,048 or 4,000 tokens shared between prompt and completion. The exact limit varies by model. (One token is roughly 4 characters for normal English text)",
    },
    {
      id: "topP",
      name: "Top P",
      models: ["gpt-3.5-turbo", "gpt-4"],
      defaultValue: 1,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.",
    },
    {
      id: "frequencyPenalty",
      name: "Frequency Penalty",
      models: ["gpt-3.5-turbo", "gpt-4"],
      defaultValue: 0,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "How much to penalize new tokens based on their existing frequency in the text so far. Decreases the model's likelihood to repeat the same line verbatim.",
    },
    {
      id: "presencePenalty",
      name: "Presence Penalty",
      models: ["gpt-3.5-turbo", "gpt-4"],
      defaultValue: 0,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "How much to penalize new tokens based on whether they appear in the text so far. Increases the model's likelihood to talk about new topics.",
    },
  ],
  vertexai: [
    {
      id: "temperature",
      name: "Temperature",
      models: [
        "text-bison@001",
        "chat-bison@001",
        "code-bison@001",
        "codechat-bison@001",
      ],
      defaultValue: 0.2,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "Temperature controls the degree of randomness in token selection. Lower temperatures are good for prompts that expect a true or correct response, while higher temperatures can lead to more diverse or unexpected results. A temperature of 0 is deterministic: the highest probability token is always selected. For most use cases, try starting with a temperature of .2.",
    },
    {
      id: "maxTokens",
      name: "Token limit",
      models: [
        "text-bison@001",
        "chat-bison@001",
        "code-bison@001",
        "codechat-bison@001",
      ],
      defaultValue: 256,
      min: 1,
      max: 1024,
      step: 1,
      description:
        "Token limit determines the maximum amount of text output from one prompt. A token is approximately four characters. The default value is 256.",
    },
    {
      id: "topK",
      name: "Top K",
      models: ["text-bison@001", "chat-bison@001"],
      defaultValue: 40,
      min: 1,
      max: 40,
      step: 1,
      description:
        "Top-k changes how the model selects tokens for output. A top-k of 1 means the selected token is the most probable among all tokens in the model’s vocabulary (also called greedy decoding), while a top-k of 3 means that the next token is selected from among the 3 most probable tokens (using temperature). The default top-k value is 40.",
    },
    {
      id: "topP",
      name: "Top P",
      models: ["text-bison@001", "chat-bison@001"],
      defaultValue: 0.8,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "Top-p changes how the model selects tokens for output. Tokens are selected from most probable to least until the sum of their probabilities equals the top-p value. For example, if tokens A, B, and C have a probability of .3, .2, and .1 and the top-p value is .5, then the model will select either A or B as the next token (using temperature). The default top-p value is .8.",
    },
  ],
  bedrock: [
    {
      id: "temperature",
      name: "Temperature",
      models: [
        "amazon.titan-tg1-large",
        "anthropic.claude-instant-v1",
        "anthropic.claude-v1",
        "anthropic.claude-v2",
      ],
      defaultValue: 1,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "Temperature controls the degree of randomness in token selection. Lower temperatures are good for prompts that expect a true or correct response, while higher temperatures can lead to more diverse or unexpected results. A temperature of 0 is deterministic: the highest probability token is always selected. For most use cases, try starting with a temperature of .2.",
    },
    {
      id: "maxTokens",
      name: "Token limit",
      models: [
        "amazon.titan-tg1-large",
        "anthropic.claude-instant-v1",
        "anthropic.claude-v1",
        "anthropic.claude-v2",
      ],
      defaultValue: 300, // TODO: change ranges and default values depending on model
      min: 1,
      max: 2048,
      step: 1,
      description:
        "Token limit determines the maximum amount of text output from one prompt. A token is approximately four characters. The default value is 256.",
    },
    {
      id: "topK",
      name: "Top K",
      models: [
        "anthropic.claude-instant-v1",
        "anthropic.claude-v1",
        "anthropic.claude-v2",
      ],
      defaultValue: 200,
      min: 1,
      max: 500,
      step: 1,
      description:
        "Top-k changes how the model selects tokens for output. A top-k of 1 means the selected token is the most probable among all tokens in the model’s vocabulary (also called greedy decoding), while a top-k of 3 means that the next token is selected from among the 3 most probable tokens (using temperature). The default top-k value is 40.",
    },
    {
      id: "topP",
      name: "Top P",
      models: [
        "amazon.titan-tg1-large",
        "anthropic.claude-instant-v1",
        "anthropic.claude-v1",
        "anthropic.claude-v2",
      ],
      defaultValue: 0.9,
      min: 0,
      max: 1,
      step: 0.1,
      description:
        "Top-p changes how the model selects tokens for output. Tokens are selected from most probable to least until the sum of their probabilities equals the top-p value. For example, if tokens A, B, and C have a probability of .3, .2, and .1 and the top-p value is .5, then the model will select either A or B as the next token (using temperature). The default top-p value is .8.",
    },
  ],
};
