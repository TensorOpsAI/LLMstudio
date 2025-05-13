format:
	pre-commit run --all-files

unit-tests:
	pytest libs/core/tests/unit_tests

integration-tests:
	pytest libs/llmstudio/tests/integration_tests