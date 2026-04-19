# ─── NeuroRAG Makefile ────────────────────────────────────────────────────────
# Usage: make <target>
# Requires: docker, docker-compose, kubectl, python3.11+, pip

.PHONY: all help install install-gpu lint typecheck format \
        test test-unit test-cov test-load-smoke test-load test-stress \
        test-load-peak eval eval-quick \
        build build-push up down down-volumes logs logs-all \
        shell restart-api migrate migrate-down migrate-status \
        seed rebuild-index \
        k8s-deploy k8s-rollback k8s-status k8s-logs k8s-port-forward \
        health env-check query stats \
        generate-api-key generate-airflow-keys \
        airflow-unpause airflow-trigger-eval airflow-trigger-retrain \
        validate-slos clean clean-data docs

# ─── Config ───────────────────────────────────────────────────────────────────
PYTHON       := python3
PIP          := pip3
IMAGE_NAME   := neurorag
IMAGE_TAG    := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")
REGISTRY     := registry.neurorag.io
K8S_NS       := neurorag
COMPOSE      := docker compose
TEST_FLAGS   := -v --tb=short

# ─── Default ──────────────────────────────────────────────────────────────────
all: lint test build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-26s\033[0m %s\n", $$1, $$2}'

# ─── Development Setup ────────────────────────────────────────────────────────
install: ## Install all Python dependencies (CPU mode for dev)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install faiss-cpu
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

install-gpu: ## Install GPU dependencies (CUDA 12.1)
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.2.2
	$(PIP) install faiss-gpu==1.7.2
	$(PIP) install -r requirements.txt
	@echo "✅ GPU dependencies installed"

# ─── Quality Gates ────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check . --select E,F,W,I,N,UP --ignore E501
	@echo "✅ Lint passed"

typecheck: ## Run mypy type checker
	mypy agents/ rag/ api/ evaluation/ --ignore-missing-imports --no-strict-optional
	@echo "✅ Type check passed"

format: ## Auto-format with ruff
	ruff format .
	ruff check . --fix

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run all unit + integration tests
	pytest tests/test_neurorag.py $(TEST_FLAGS) \
		--cov=agents --cov=rag --cov=evaluation \
		--cov-report=term-missing --cov-report=html:htmlcov
	@echo "✅ Tests passed"

test-unit: ## Run unit tests only (fast, no server needed)
	pytest tests/test_neurorag.py $(TEST_FLAGS) -m "not integration"

test-cov: ## Run tests with coverage report (fails under 70%)
	pytest tests/test_neurorag.py $(TEST_FLAGS) \
		--cov=agents --cov=rag --cov=evaluation --cov=api \
		--cov-report=html:htmlcov --cov-fail-under=70
	@echo "Coverage report: htmlcov/index.html"

test-load-smoke: ## Run load test smoke suite (no server required)
	pytest tests/test_load.py $(TEST_FLAGS) -k "smoke or Smoke"

test-load: ## Run sustained load test — 50 users, 120s (requires running server)
	$(PYTHON) -m tests.test_load_production --mode sustained --users 50 --duration 120
	@echo "Load report: /tmp/async_load_report.json"

test-load-peak: ## Run peak load test — 100 users, 180s (requires running server)
	$(PYTHON) -m tests.test_load_production --mode peak --users 100 --duration 180
	@echo "Load report: /tmp/async_load_report.json"

test-stress: ## Run stress test — 150 users, 120s (find breaking point)
	$(PYTHON) -m tests.test_load_production --mode stress

# ─── Evaluation ───────────────────────────────────────────────────────────────
eval: ## Run full 300-query benchmark evaluation (requires running server)
	$(PYTHON) -m evaluation.evaluation_runner --mode full --output eval_report.json
	@echo "Evaluation report: eval_report.json"

eval-quick: ## Run quick 50-query evaluation
	$(PYTHON) -m evaluation.evaluation_runner --mode quick --output eval_report_quick.json
	@echo "Quick eval report: eval_report_quick.json"

validate-slos: ## Parse eval_report.json and check all SLO thresholds
	@$(PYTHON) - <<'PYEOF'
import json, sys
try:
    r = json.load(open("eval_report.json"))
except FileNotFoundError:
    print("❌  eval_report.json not found — run 'make eval' first.")
    sys.exit(1)
gm = r.get("generation_metrics", {})
checks = [
    ("Token F1",             gm.get("avg_token_f1", 0),      0.70),
    ("Faithfulness (conf)",  gm.get("avg_confidence", 0),    0.85),
    ("Context relevance",    r.get("context_relevance", 0),  0.70),
]
ok = True
for name, val, thr in checks:
    sym = "✅" if val >= thr else "❌"
    print(f"  {sym} {name}: {val:.3f}  (threshold: {thr})")
    if val < thr:
        ok = False
insuff = r.get("insufficient_context_rate", 0)
print(f"  {'✅' if insuff <= 0.05 else '⚠️ '} Insufficient ctx rate: {insuff:.3f}  (target ≤ 0.05)")
print()
print("EVALUATION PASS" if ok else "EVALUATION FAIL — see above")
sys.exit(0 if ok else 1)
PYEOF

# ─── Docker ───────────────────────────────────────────────────────────────────
build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -t $(IMAGE_NAME):latest .
	@echo "✅ Built $(IMAGE_NAME):$(IMAGE_TAG)"

build-push: build ## Build and push to registry
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker tag $(IMAGE_NAME):latest $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo "✅ Pushed $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)"

up: ## Start full stack (docker-compose)
	@if [ ! -f .env ]; then cp .env.example .env; echo "⚠️  Created .env from .env.example — fill in secrets before proceeding."; fi
	$(COMPOSE) up -d
	@echo "✅ Stack started"
	@echo "  API:        http://localhost:8000"
	@echo "  Grafana:    http://localhost:3000"
	@echo "  Airflow:    http://localhost:8080"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Pushgateway:http://localhost:9092"

down: ## Stop all services
	$(COMPOSE) down
	@echo "✅ Stack stopped"

down-volumes: ## Stop and remove all data volumes (DESTRUCTIVE)
	$(COMPOSE) down -v
	@echo "⚠️  All volumes removed"

logs: ## Tail API logs
	$(COMPOSE) logs -f neurorag-api

logs-all: ## Tail all service logs
	$(COMPOSE) logs -f

shell: ## Open bash shell in running API container
	$(COMPOSE) exec neurorag-api bash

restart-api: ## Restart only the API service
	$(COMPOSE) restart neurorag-api

# ─── Database ─────────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	alembic upgrade head
	@echo "✅ Migrations applied"

migrate-down: ## Rollback last migration
	alembic downgrade -1

migrate-status: ## Show migration status
	alembic current
	alembic history

# ─── Data ─────────────────────────────────────────────────────────────────────
seed: ## Ingest sample documents into the index
	$(PYTHON) scripts/seed_data.py
	@echo "✅ Sample data seeded"

rebuild-index: ## Rebuild FAISS + BM25 index from /data/raw
	$(PYTHON) scripts/rebuild_index.py
	@echo "✅ Index rebuilt"

# ─── Kubernetes ───────────────────────────────────────────────────────────────
k8s-deploy: ## Deploy to Kubernetes (production)
	kubectl apply -f infra/k8s/ -n $(K8S_NS)
	kubectl rollout status deployment/neurorag-api -n $(K8S_NS) --timeout=300s
	@echo "✅ Deployed to K8s namespace: $(K8S_NS)"

k8s-rollback: ## Rollback last K8s deployment
	kubectl rollout undo deployment/neurorag-api -n $(K8S_NS)
	@echo "⚠️  Rollback executed"

k8s-status: ## Show K8s pod and deployment status
	kubectl get pods,deployments,svc -n $(K8S_NS)

k8s-logs: ## Stream K8s API pod logs
	kubectl logs -f -l app=neurorag,component=api -n $(K8S_NS) --tail=100

k8s-port-forward: ## Forward API to localhost:8000
	kubectl port-forward svc/neurorag-api 8000:80 -n $(K8S_NS)

# ─── Airflow ──────────────────────────────────────────────────────────────────
airflow-unpause: ## Unpause both MLOps DAGs
	$(COMPOSE) exec airflow airflow dags unpause neurorag_eval_v2
	$(COMPOSE) exec airflow airflow dags unpause neurorag_faithfulness_trigger
	@echo "✅ DAGs unpaused"

airflow-trigger-eval: ## Manually trigger neurorag_eval_v2
	$(COMPOSE) exec airflow airflow dags trigger neurorag_eval_v2
	@echo "✅ neurorag_eval_v2 triggered"

airflow-trigger-retrain: ## Manually trigger neurorag_faithfulness_trigger
	$(COMPOSE) exec airflow airflow dags trigger neurorag_faithfulness_trigger
	@echo "✅ neurorag_faithfulness_trigger triggered"

airflow-set-threshold: ## Set faithfulness threshold (usage: make airflow-set-threshold T=0.99)
	$(COMPOSE) exec airflow airflow variables set FAITHFULNESS_THRESHOLD $(T)
	@echo "✅ FAITHFULNESS_THRESHOLD set to $(T)"

airflow-reset-vars: ## Reset all MLOps Airflow variables to production defaults
	$(COMPOSE) exec airflow airflow variables set FAITHFULNESS_THRESHOLD 0.80
	$(COMPOSE) exec airflow airflow variables set FAITHFULNESS_MIN_DELTA 0.05
	@echo "✅ MLOps variables reset to production defaults"

# ─── Utilities ────────────────────────────────────────────────────────────────
health: ## Check API health
	curl -sf http://localhost:8000/health | python3 -m json.tool

env-check: ## Check which critical env vars are set (no values exposed)
	curl -sf http://localhost:8000/env-check | python3 -m json.tool

query: ## Run a test query (set Q="your question")
	curl -sf -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-H "X-API-Key: $${NEURORAG_API_KEY}" \
		-d '{"query": "$(or $(Q), What is retrieval-augmented generation?)"}' \
		| python3 -m json.tool

stats: ## Show evaluation stats (last 24h)
	curl -sf "http://localhost:8000/stats?hours=24" \
		-H "X-API-Key: $${NEURORAG_API_KEY}" | python3 -m json.tool

circuit-status: ## Check circuit breaker states
	curl -sf http://localhost:8000/circuit-breaker/status \
		-H "X-API-Key: $${NEURORAG_API_KEY}" | python3 -m json.tool

generate-api-key: ## Generate a secure random API key
	@$(PYTHON) -c "import secrets; print(secrets.token_urlsafe(32))"

generate-airflow-keys: ## Generate AIRFLOW_FERNET_KEY and AIRFLOW_SECRET_KEY
	@echo "AIRFLOW_FERNET_KEY=$$($(PYTHON) -c \
		'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
	@echo "AIRFLOW_SECRET_KEY=$$($(PYTHON) -c \
		'import secrets; print(secrets.token_hex(24))')"

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove Python cache, coverage, and report files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml reports/ /tmp/async_load_report.json
	@echo "✅ Clean complete"

clean-data: ## Remove FAISS index and data volumes (DESTRUCTIVE)
	rm -rf data/faiss data/whoosh_index data/drift
	@echo "⚠️  Index data removed"

docs: ## Save OpenAPI spec (requires running server)
	mkdir -p docs
	curl -sf http://localhost:8000/openapi.json | python3 -m json.tool > docs/openapi.json
	@echo "✅ OpenAPI spec saved to docs/openapi.json"
