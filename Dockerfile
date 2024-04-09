FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install Poetry
RUN apt clean && apt update && apt install curl zip unzip nodejs -y
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

WORKDIR /code
COPY . .

RUN curl -fsSL https://bun.sh/install | bash && \
    ln -s $HOME/.bun/bin/bun /usr/local/bin/bun

RUN curl -L https://npmjs.org/install.sh | sh

RUN poetry install
# RUN pip install llmstudio
# RUN pip install psycopg2-binary

ENV PYTHONPATH=/code
EXPOSE 3000 8001 8002

CMD llmstudio server --ui