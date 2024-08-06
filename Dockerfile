FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt clean && apt update && apt install curl zip unzip nodejs -y

# Install llmstudio
RUN pip install llmstudio
RUN pip install psycopg2-binary

# EXPOSE PORTS (as in .env this might vary)
EXPOSE 8001 8002

ENTRYPOINT [ "llmstudio", "server"]
