.PHONY: help run test lint format docker-build docker-up docker-down k8s-deploy k8s-delete benchmark tune health

help: ## Show this help message
	@echo "🚀 Frontier Inference API - Makefile Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

run: ## Run the API server locally
	python -m uvicorn inference.api:app --host 0.0.0.0 --port 8080 --reload

test: ## Run tests
	pytest tests/ -v --cov=inference --cov-report=html

lint: ## Run linters
	ruff check inference/
	black --check inference/
	mypy inference/ --ignore-missing-imports || true

format: ## Format code
	black inference/
	ruff check --fix inference/

docker-build: ## Build Docker image
	docker build -t frontier-inference-api -f Dockerfile ..

docker-up: ## Start Docker Compose stack
	docker-compose up -d

docker-down: ## Stop Docker Compose stack
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f inference-api

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/deployment.yaml

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f k8s/deployment.yaml

benchmark: ## Run benchmark
	python -m inference.utils.benchmark --url http://localhost:8080 --requests 100 --concurrency 10

tune: ## Run performance tuning
	python -m inference.utils.performance_tuner --url http://localhost:8080

health: ## Check API health
	curl -s http://localhost:8080/health | jq .

metrics: ## View metrics
	curl -s http://localhost:8080/metrics | grep -v "^#"

load-test: ## Run load test with k6
	k6 run tests/load-test.js

install-dev: ## Install development dependencies
	pip install -r requirements_advanced.txt
	pip install pytest pytest-cov pytest-asyncio httpx rich ruff black mypy k6

clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .pytest_cache/ .ruff_cache/


