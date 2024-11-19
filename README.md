# LLMstudio by [TensorOps](http://tensorops.ai "TensorOps")

Prompt Engineering at your fingertips

![LLMstudio logo](https://imgur.com/Xqsj6V2.gif)

## 🌟 Features

![LLMstudio UI](https://imgur.com/wrwiIUs.png)

- **LLM Proxy Access**: Seamless access to all the latest LLMs by OpenAI, Anthropic, Google.
- **Custom and Local LLM Support**: Use custom or local open-source LLMs through Ollama.
- **Prompt Playground UI**: A user-friendly interface for engineering and fine-tuning your prompts.
- **Python SDK**: Easily integrate LLMstudio into your existing workflows.
- **Monitoring and Logging**: Keep track of your usage and performance for all requests.
- **LangChain Integration**: LLMstudio integrates with your already existing LangChain projects.
- **Batch Calling**: Send multiple requests at once for improved efficiency.
- **Smart Routing and Fallback**: Ensure 24/7 availability by routing your requests to trusted LLMs.
- **Type Casting (soon)**: Convert data types as needed for your specific use case.

## 🚀 Quickstart

Don't forget to check out [https://docs.llmstudio.ai](docs) page.

## Installation

Install the latest version of **LLMstudio** using `pip`. We suggest that you create and activate a new environment using `conda`

For full version:
```bash
pip install 'llmstudio[proxy,tracker]'
```

For lightweight (core) version:
```bash
pip install llmstudio
```

Create a `.env` file at the same path you'll run **LLMstudio**

```bash
OPENAI_API_KEY="sk-api_key"
ANTHROPIC_API_KEY="sk-api_key"
VERTEXAI_KEY="sk-api-key"
```

Now you should be able to run **LLMstudio** using the following command.

```bash
llmstudio server --proxy --tracker
```

When the `--proxy` flag is set, you'll be able to access the [Swagger at http://0.0.0.0:50001/docs (default port)](http://0.0.0.0:50001/docs)

When the `--tracker` flag is set, you'll be able to access the [Swagger at http://0.0.0.0:50002/docs (default port)](http://0.0.0.0:50002/docs)

## 📖 Documentation

- [Visit our docs to learn how the SDK works](https://docs.LLMstudio.ai) (coming soon)
- Checkout our [notebook examples](https://github.com/TensorOpsAI/LLMstudio/tree/main/examples) to follow along with interactive tutorials

## 👨‍💻 Contributing

- Head on to our [Contribution Guide](https://github.com/TensorOpsAI/LLMstudio/tree/main/CONTRIBUTING.md) to see how you can help LLMstudio.
- Join our [Discord](https://discord.gg/GkAfPZR9wy) to talk with other LLMstudio enthusiasts.

## Training

[![Banner](https://imgur.com/XTRFZ4m.png)](https://www.tensorops.ai/llm-studio-workshop)

---

Thank you for choosing LLMstudio. Your journey to perfecting AI interactions starts here.
