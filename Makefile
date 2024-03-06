.PHONY: test

test:
	pytest llmstudio/tests

start-db:
	docker compose up --build