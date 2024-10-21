# LLMstudio-core by [TensorOps](http://tensorops.ai "TensorOps")

Prompt Engineering at your fingertips

![LLMstudio logo](https://imgur.com/Xqsj6V2.gif)

## 🌟 Features
- **Custom and Local LLM Support**: Use custom or local open-source LLMs through Ollama.
- **Python SDK**: Easily integrate LLMstudio into your existing workflows.
- **LangChain Integration**: LLMstudio integrates with your already existing LangChain projects.

## 🚀 Quickstart

Don't forget to check out [https://docs.llmstudio.ai](docs) page.

## Installation

Install the latest version of **LLMstudio** using `pip`. We suggest that you create and activate a new environment using `conda`

```bash
pip install llmstudio-core
```

Create a `.env` file at the same path you'll run **LLMstudio**

```bash
OPENAI_API_KEY="sk-api_key"
GOOGLE_API_KEY="sk-api_key"
```

Now you should be able to run **LLMstudio** Providers using the following code:

```
# You can set OPENAI_API_KEY environment variable, add it to .env, or pass directly as api_key
import os
from llmstudio_core.providers import LLMCore as LLM
llm = LLM("vertexai", api_key=os.environ["GOOGLE_API_KEY"])
response = llm.chat("How are you", model="gemini-1.5-pro-latest")
print(response.chat_output, response.metrics)
```
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
