build-llmstudio-proxy:
	docker build --build-arg LLMSTUDIO_VERSION=$(version) \
	-t tensoropsai/llmstudio-proxy:latest \
	-t tensoropsai/llmstudio-proxy:$(version) \
	-f proxy.Dockerfile \
	.

build-llmstudio-tracker:
	docker build --build-arg LLMSTUDIO_VERSION=$(version) \
	-t tensoropsai/llmstudio-tracker:latest \
	-t tensoropsai/llmstudio-tracker:$(version) \
	-f tracker.Dockerfile \
	.

build: build-llmstudio-tracker build-llmstudio-proxy

run:
	docker compose -f docker-compose.yml up
