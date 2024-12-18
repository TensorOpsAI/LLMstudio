FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Install tools
RUN apt-get clean && apt-get update

# Install llmstudio
ARG LLMSTUDIO_VERSION
RUN pip install 'llmstudio-tracker'==${LLMSTUDIO_VERSION}
RUN pip install sqlalchemy-bigquery=="1.12.*"
RUN pip install google-cloud-bigquery-storage=="2.27.*"

CMD ["llmstudio-tracker", "server"]
