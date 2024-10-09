FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt clean && apt update

# Install llmstudio
RUN pip install llmstudio==0.3.11
RUN pip install psycopg2-binary

# Expose Ports
EXPOSE 8001 8002

CMD ["llmstudio", "server"]