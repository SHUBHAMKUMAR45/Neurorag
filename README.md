# NeuroRAG — Autonomous Self-Healing Multi-Agent RAG System

> **Version 3.0.0** | Python 3.11 | Windows + Linux | CPU + GPU

> 🚀 **Deploying to GitHub / Production?** Check out the [GitHub Deployment Guide](DEPLOYMENT.md).

---

## Quick Start (Windows, No Docker)

### 1 — Prerequisites

Install these first:
- [Python 3.11](https://python.org/downloads/) — check "Add to PATH"
- [PostgreSQL 16](https://postgresql.org/download/windows/) — remember the superuser password
- [Redis for Windows](https://github.com/tporadowski/redis/releases) — optional but recommended

### 2 — Create virtual environment

```powershell
cd neurorag_final
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```powershell
# Allow script execution if needed:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Run the Windows setup script:
.\install_windows.ps1
```

Or install manually:
```powershell
pip install "numpy==1.26.4" --force-reinstall
pip install faiss-cpu packaging
pip install fastapi==0.111.0 "uvicorn[standard]==0.29.0" pydantic==2.7.1
pip install openai==1.30.1 "sentence-transformers==3.0.1" transformers==4.41.1
pip install whoosh==2.7.4 asyncpg==0.29.0 alembic==1.13.1 psycopg2-binary==2.9.9
pip install "redis[asyncio]==5.0.4" prometheus-client==0.20.0 structlog==24.1.0
pip install pyyaml==6.0.1 python-dotenv==1.0.1 httpx==0.27.0 aiofiles==23.2.1
pip install "opentelemetry-api==1.24.0" "opentelemetry-sdk==1.24.0"
```

### 4 — Configure environment

```powershell
copy .env.example .env
notepad .env
```

Set these values in `.env`:
```
OPENAI_API_KEY=sk-...         # Your OpenAI API key
NEURORAG_API_KEY=             # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"
POSTGRES_URL=postgresql://neurorag_user:StrongPass123@localhost:5432/neurorag
REDIS_URL=redis://localhost:6379
```

Load them into PowerShell:
```powershell
. .\load_env.ps1
```

### 5 — Create PostgreSQL database

Open psql or pgAdmin and run:
```sql
CREATE USER neurorag_user WITH PASSWORD 'StrongPass123';
CREATE DATABASE neurorag OWNER neurorag_user;
GRANT ALL PRIVILEGES ON DATABASE neurorag TO neurorag_user;
```

Then run migrations:
```powershell
alembic upgrade head
```

### 6 — Create data directories

```powershell
New-Item -ItemType Directory -Force -Path data\faiss, data\whoosh_index, data\raw, logs, static
```

### 7 — Start the API

```powershell
# IMPORTANT: use --workers 1 on Windows (no uvloop)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

### 8 — Open the UI

Navigate to **http://localhost:8000** in your browser.

### 9 — Seed the knowledge base (new PowerShell tab)

```powershell
.\.venv\Scripts\Activate.ps1
. .\load_env.ps1
python scripts\seed_data.py
```

Expected: `✅ Seeded: 50 docs → ~180 chunks`

### 10 — Test a query

```powershell
Invoke-RestMethod http://localhost:8000/query `
  -Method Post `
  -Headers @{"Content-Type"="application/json"; "X-API-Key"=$env:NEURORAG_API_KEY} `
  -Body '{"query":"What is retrieval augmented generation?"}' | ConvertTo-Json -Depth 5
```

---

## Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `uvloop does not support Windows` | Do NOT install uvloop. Use `--workers 1` with uvicorn. |
| `faiss-gpu==1.7.2 not found` | Already fixed — requirements.txt uses `faiss-cpu` |
| `numpy.core.multiarray failed to import` | Run: `pip install "numpy==1.26.4" --force-reinstall` |
| `No module named 'packaging'` | Run: `pip install packaging` |
| `No module named 'llama_cpp'` | Already fixed — config.yaml uses `provider: "openai"` |
| `ModuleNotFoundError: No module named 'app'` | Use `uvicorn api.main:app` not `uvicorn app.main:app` |
| `INSUFFICIENT_CONTEXT` on queries | Run `python scripts/seed_data.py` to populate the index |
| `NEURORAG_API_KEY not set` | Run `. .\load_env.ps1` and set key in `.env` |
| `alembic: No module named 'psycopg2'` | Run: `pip install psycopg2-binary` |

---

## Architecture

```
Query → IntentAnalyzer → Planner → HybridRetriever (BM25 + FAISS) 
      → Reranker → Generator → Critic
      → [if confidence < 0.80] → ReflectionAgent → FixerAgent → retry
      → PipelineResult
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | System health + vector count |
| POST | `/query` | Submit a RAG query |
| POST | `/ingest` | Add documents to knowledge base |
| GET | `/stats` | Query statistics (requires Postgres) |
| GET | `/env-check` | Which env vars are set |
| GET | `/circuit-breaker/status` | LLM circuit breaker states |
| GET | `/memory/stats` | Redis cache stats |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Interactive API docs (Swagger) |

## Project Structure

```
neurorag_final/
├── api/              FastAPI server + middleware
├── agents/           IntentAnalyzer, Planner, Generator, Critic, Reflection, Fixer, Memory
├── rag/              HybridRetriever, IngestionEngine, Reranker, LLMClient, CircuitBreaker
├── evaluation/       EvaluationRunner, Evaluator, BenchmarkDataset, RAGMetrics
├── configs/          config.yaml, settings.py
├── dashboard/        Prometheus metrics
├── mlops/dags/       Airflow DAGs
├── infra/            Grafana, Prometheus, K8s, migrations
├── tests/            Unit + load tests
├── scripts/          seed_data.py, rebuild_index.py
├── static/           Web UI (index.html)
├── data/             FAISS index, Whoosh index (created at runtime)
├── install_windows.ps1
├── load_env.ps1
└── requirements.txt
```
