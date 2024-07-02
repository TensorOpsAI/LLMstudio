FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt clean && apt update && apt install curl zip unzip nodejs -y

# Install bun
RUN curl -fsSL https://bun.sh/install | bash && \
    ln -s $HOME/.bun/bin/bun /usr/local/bin/bun

# Install npm
RUN curl -L https://npmjs.org/install.sh | sh

# Install llmstudio
RUN pip install llmstudio==0.3.4a0
RUN pip install psycopg2-binary

# EXPOSE PORTS (as in .env this might vary)
EXPOSE 3000 8001 8002

ENTRYPOINT [ "llmstudio", "server", "--ui" ]
