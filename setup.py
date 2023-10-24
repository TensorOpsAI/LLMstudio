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
    keywords="ml ai llm llmops openai langchain chatgpt llmstudio tensorops",
    version=SDK_VERSION,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={"console_scripts": ["llmstudio = llmstudio.cli:main"]},
    python_requires="~=3.9",
)
