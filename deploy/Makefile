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

build-llmstudio-tracker-bigquery:
	docker build --build-arg LLMSTUDIO_VERSION=$(version) \
	-t tensoropsai/llmstudio-tracker-bigquery:latest \
	-t tensoropsai/llmstudio-tracker-bigquery:$(version) \
	-f tracker_bigquery.Dockerfile \
	.

build-llmstudio-tracker-postgresql:
	docker build --build-arg LLMSTUDIO_VERSION=$(version) \
	-t tensoropsai/llmstudio-tracker-postgresql:latest \
	-t tensoropsai/llmstudio-tracker-postgresql:$(version) \
	-f tracker_postgresql.Dockerfile \
	.

build: build-llmstudio-tracker build-llmstudio-proxy build-llmstudio-tracker-bigquery build-llmstudio-tracker-postgresql

run:
	docker compose -f docker-compose.yml up
