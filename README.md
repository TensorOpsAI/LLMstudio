# LLMstudio by [TensorOps](http://tensorops.ai "TensorOps")

Prompt Engineering at your fingertips

![LLMstudio logo](https://imgur.com/Xqsj6V2.gif)

> [!IMPORTANT]
> LLMstudio is now supporting OpenAI v1.0 + just added support to Anthropic

## 🌟 Features

![LLMstudio UI](https://imgur.com/wrwiIUs.png)

1.  **Python Client Gateway**:
    - Access models from known providers such as OpenAI, VertexAI and Bedrock. All in one platform.
    - Speed up development with tracking and robustness features from LLMstudio.
    - Continue using popular libraries like LangChain through their LLMstudio-wrapped versions.
2.  **Prompt Editing UI**:
    - An intuitive interface designed for prompt engineering.
    - Quickly iterate between prompts until you reach your desired results.
    - Access the history of your previous prompts and their results.
3.  **History Management**:
    - Track past runs, available for both on the UI and the Client.
    - Log the cost, latency and output of each prompt.
    - Export the history to a CSV.
4.  **Context Limit Adaptability**:
    - Automatic switch to a larger-context fallback model if the current model's context limit is exceeded.
    - Always use the lowest context model and only use the higher context ones when necessary to save costs.
    - For instance, exceeding 4k tokens in gpt-3.5-turbo triggers a switch to gpt-3.5-turbo-16k.

### 👀 Coming soon:

- Side-by-side comparison of multiple LLMs using the same prompt.
- Automated testing and validation for your LLMs. (Create Unit-tests for your LLMs which are evaluated automatically)
- API key administration. (Define quota limits)
- Projects and sessions. (Organize your History and API keys by project)
- Resilience against service provider rate limits.
- Organized tracking of groups of related prompts (Chains, Agents)

## 🚀 Quickstart

Don't forget to check out [https://docs.llmstudio.ai](docs) page.

## Installation

Install the latest version of **LLMstudio** using `pip`. We suggest that you create and activate a new environment using `conda`

```bash
pip install llmstudio
```

Install `bun` if you want to use the UI

```bash
curl -fsSL https://bun.sh/install | bash
```

Create a `.env` file at the same path you'll run **LLMstudio**

```bash
OPENAI_API_KEY="sk-api_key"
ANTHROPIC_API_KEY="sk-api_key"
```

Now you should be able to run **LLMstudio** using the following command.

```bash
llmstudio server --ui
```

When the `--ui` flag is set, you'll be able to access the UI at [http://localhost:3000](http://localhost:3000)

## 🤔 About LLMstudio

Powered by TensorOps, LLMstudio redefines your experience with OpenAI, Vertex Ai and more language model providers. More than just a tool, it’s an evolving environment where teams can experiment, modify, and optimize their interactions with advanced language models.

Benefits include:

- **Streamlined Prompt Engineering**: Simplify and enhance your prompt design process.
- **Execution History**: Keep a detailed log of past executions, track progress, and make iterative improvements effortlessly.
- **Effortless Data Export**: Share your team's endeavors by exporting data to shareable CSV files.

Step into the future of AI with LLMstudio, by watching our [introduction video](https://www.youtube.com/watch?v=I9h701fbD18)

## 📖 Documentation

- [Visit our docs to learn how the SDK works](https://docs.LLMstudio.ai) (coming soon)
- Checkout our [notebook examples](https://github.com/TensorOpsAI/LLMstudio/tree/main/examples) to follow along with interactive tutorials
- Checkout out [LLMstudio Architecture Roadmap](https://github.com/TensorOpsAI/LLMstudio/blob/main/docs/LLMstudio-architecture/LLMstudio-architecture-roadmap.md)

## 👨‍💻 Contributing

- Head on to our [Contribution Guide](https://github.com/TensorOpsAI/LLMstudio/tree/main/CONTRIBUTING.md) to see how you can help LLMstudio.
- Join our [Discord](https://discord.gg/GkAfPZR9wy) to talk with other LLMstudio enthusiasts.

## Training

[![Banner](https://imgur.com/XTRFZ4m.png)](https://www.tensorops.ai/llm-studio-workshop)

---

Thank you for choosing LLMstudio. Your journey to perfecting AI interactions starts here.
