.PHONY: test

test:
	pytest llmstudio/tests

run-build:
	docker compose up --build

publish: run-build
	sudo docker tag llmstudio-llmstudio llmstudio.azurecr.io/llmstudio:latest && \
	az acr login --name llmstudio && \
	sudo docker push llmstudio.azurecr.io/llmstudio:latest

