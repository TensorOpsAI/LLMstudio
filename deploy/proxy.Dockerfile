FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt-get clean && apt-get update

# Install llmstudio
ARG LLMSTUDIO_VERSION
RUN pip install 'llmstudio[proxy]'==${LLMSTUDIO_VERSION}

CMD ["llmstudio", "server", "--proxy"]
