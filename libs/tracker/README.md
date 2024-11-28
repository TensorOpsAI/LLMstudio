# LLMstudio Tracker

LLMstudio Tracker is the module of LLMstudio that allows monitoring and logging your LLM calls.
It supports seamless integration with the LLMstudio environment through configurable tracking servers, allowing for detailed insights into synchronous and asynchronous chat interactions. By leveraging LLMstudio Tracker, users can gain insights on model performance and streamline development workflows with actionable analytics.

## 🌟 Features

- **Monitoring and Logging**: Keep track of your usage and performance for all requests.
- **Logs Persistence with SQLAlquemy**: You can configure the tracker to use a database of your choice (SQLlite, Postgres, Bigquery, etc...)


## Installation

Install the latest version of **LLMstudio** using `pip`. We suggest that you create and activate a new virtual environment.

```bash
pip install 'llmstudio[tracker]'
```

## How to run

To configure the tracker host, port, and database URI, create a `.env` file at the same path you'll run **LLMstudio** and set values for:
- LLMSTUDIO_TRACKING_HOST (default is 0.0.0.0)
- LLMSTUDIO_TRACKING_PORT (default is 50002)
- LLMSTUDIO_TRACKING_URI (the default is sqlite:///./llmstudio_mgmt.db)

If you skip this step, LLMstudio will just use the default values.


```bash
LLMSTUDIO_TRACKING_HOST=0.0.0.0
LLMSTUDIO_TRACKING_PORT=8002
LLMSTUDIO_TRACKIN_URI="your_db_uri"

```

### Launching from a terminal

Now you should be able to run **LLMstudio Tracker** using the following command:

```bash
llmstudio server --tracker
```

### Launching directly in your code

Alternatively, you can start the server in your code:
```python
from llmstudio.server import start_servers
start_servers(proxy=False, tracker=True)
```

When the `--tracker` flag is set, you'll be able to access the [Swagger at http://0.0.0.0:50002/docs (default port)](http://0.0.0.0:50002/docs)

If you didn't provide the URI to your database, LLMstudio will create an SQLite database at the root of your project and write the logs there.

## Usage

Now, you can initialize an LLM to make calls and link it to your tracking configuration so that the tracker will log the calls.

```python
from llmstudio_tracker.tracker import TrackingConfig

tracker_config = TrackingConfig(host="0.0.0.0", port="50002") # needs to match what was set in your .env file

# You can set OPENAI_API_KEY in your .env file
openai = LLM("openai", tracking_config = tracker_config)

openai.chat("Hey!", model="gpt-4o")
```

### Analysing the logs

```python
from llmstudio_tracker.tracker import Tracker

tracker = Tracker(tracking_config=tracker_config)

logs = tracker.get_logs()
logs.json()
```




## 📖 Documentation

- [Visit our docs to learn how it works](https://docs.LLMstudio.ai)
- Checkout our [notebook examples](https://github.com/TensorOpsAI/LLMstudio/tree/main/examples) to follow along with interactive tutorials, especially:
    - [Intro to LLMstudio Tracking](https://github.com/TensorOpsAI/LLMstudio/tree/main/examples/01_intro_to_llmstudio_with_tracking.ipynb)
    - [BigQuery Integration](https://github.com/TensorOpsAI/LLMstudio/tree/main/examples/04_bigquery_integration.ipynb)

## 👨‍💻 Contributing

- Head on to our [Contribution Guide](https://github.com/TensorOpsAI/LLMstudio/tree/main/CONTRIBUTING.md) to see how you can help LLMstudio.
- Join our [Discord](https://discord.gg/GkAfPZR9wy) to talk with other LLMstudio enthusiasts.


---

Thank you for choosing LLMstudio. Your journey to perfecting AI interactions starts here.
