# docker/Dockerfile

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt-get clean && apt-get update

# Install llmstudio
ARG LLMSTUDIO_VERSION
RUN pip install llmstudio==${LLMSTUDIO_VERSION}
RUN pip install psycopg2-binary

# Expose Ports
EXPOSE 8001 8002

CMD ["llmstudio", "server"]
