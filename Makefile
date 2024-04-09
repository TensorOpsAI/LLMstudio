.PHONY: test

test:
	pytest llmstudio/tests

run-build:
	docker compose up --build