.PHONY: docker-build-cpu docker-build-gpu docker-dev-cpu docker-dev-gpu docker-smoke-cpu docker-smoke-gpu docker-mlflow

docker-build-cpu:
	docker compose build dev-cpu

docker-build-gpu:
	docker compose --profile gpu build dev-gpu

docker-dev-cpu:
	docker compose run --rm dev-cpu

docker-dev-gpu:
	docker compose --profile gpu run --rm dev-gpu

docker-smoke-cpu:
	docker compose run --rm smoke-cpu

docker-smoke-gpu:
	docker compose --profile gpu run --rm smoke-gpu

docker-mlflow:
	mkdir -p mlruns
	docker compose up mlflow
