from setuptools import setup, find_packages

setup(
    name="llmstudio",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="TensorOps",
    author_email="claudio.lemos@tensorops.ai",
    description="Prompt Perfection at Your Fingertips",
    package_dir={"llmstudio": "client"},
)
