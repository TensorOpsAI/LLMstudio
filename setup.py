import os
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages
from llmstudio import __version__ as SDK_VERSION

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
    keywords="ml ai llm llmstudio tensorops",
    version=SDK_VERSION,
    packages=["llmstudio", "llmstudio.models"],
    package_dir={
        "llmstudio": "llmstudio",
        "llmstudio.models": "llmstudio/models",
    },
    install_requires=["requests<3", "pydantic>2"],
)
