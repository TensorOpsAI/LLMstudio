from setuptools import setup, find_packages
from llmstudio import __version__ as SDK_VERSION, __requirements__ as REQUIREMENTS

setup(
    name="llmstudio",
    author="TensorOps",
    url="https://llmstudio.ai/",
    project_urls={
        "Source Code": "https://github.com/tensoropsai/llmstudio",
        "Bug Tracker": "https://github.com/tensoropsai/llmstudio/issues",
        "Documentation": "https://docs.llmstudio.ai",
    },
    author_email="contact@tensorops.ai",
    description="Prompt Perfection at Your Fingertips",
    long_description="""
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

    When the `--ui` flag is set, you'll be able to access the UI at [http://localhost:8000](http://localhost:8000)
    """,
    long_description_content_type="text/markdown",
    keywords="ml ai llm llmops openai langchain chatgpt llmstudio tensorops",
    version=SDK_VERSION,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={"console_scripts": ["llmstudio = llmstudio.cli:main"]},
    python_requires="~=3.9",
)
