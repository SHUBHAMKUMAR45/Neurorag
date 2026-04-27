# NeuroRAG GitHub Deployment Guide

This document outlines the deployment strategy and CI/CD pipeline for NeuroRAG using GitHub Actions, Docker, and Kubernetes.

## 🚀 CI/CD Pipeline Overview

The project includes an automated GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) that triggers on pushes and pull requests to the `main` and `develop` branches. The pipeline handles the complete lifecycle from code verification to production rollout:

1. **Lint & Type Check**: Runs `ruff` for linting and `mypy` for static type checking.
2. **Unit & Integration Tests**: Spins up Redis and PostgreSQL service containers to run comprehensive `pytest` suites (including coverage and smoke tests).
3. **Docker Build & Push**: Builds the `neurorag-api` Docker image and pushes it to the GitHub Container Registry (`ghcr.io`).
4. **Security Scan**: Utilizes Aqua Security's Trivy to scan the built Docker image for critical and high vulnerabilities.
5. **Staging Deployment**: Automatically deploys the updated image to the staging Kubernetes namespace (`neurorag-staging`) and runs automated health checks.
6. **Production Deployment (Canary)**: 
   - Deploys to a canary deployment receiving 10% of traffic.
   - Monitors error rates via Prometheus for 5 minutes.
   - If the error rate stays below 5%, it promotes the release to full production.
   - Automatically rolls back both the canary and the production deployments if metrics indicate failure.

---

## 🔐 Required GitHub Secrets

To successfully deploy using the provided GitHub Actions workflow, you must configure the following **Repository Secrets** in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

| Secret Name | Description |
|-------------|-------------|
| `KUBECONFIG_STAGING` | Base64-encoded `kubeconfig` file for connecting to your staging Kubernetes cluster. |
| `STAGING_API_KEY` | The `NEURORAG_API_KEY` used to authenticate smoke tests against the staging API (`https://api-staging.neurorag.io`). |
| `KUBECONFIG_PRODUCTION` | Base64-encoded `kubeconfig` file for connecting to your production Kubernetes cluster. |
| `PROMETHEUS_URL` | The public or accessible URL of your Prometheus instance (used during the canary rollout to evaluate error rates). |

*Note: The standard `GITHUB_TOKEN` is automatically provided by GitHub Actions for authenticating pushes to `ghcr.io`.*

---

## 🐳 Local Full-Stack Deployment (Docker Compose)

Before deploying to GitHub, you can test the entire infrastructure stack locally using Docker Compose. The stack includes the FastAPI backend, PostgreSQL, Redis, Prometheus, Alertmanager, Grafana, and Apache Airflow.

### 1. Configure Environment
```bash
cp .env.example .env
# Edit .env and ensure OPENAI_API_KEY, NEURORAG_API_KEY, and other variables are set.
```

### 2. Start the Stack
```bash
docker-compose up -d --build
```

### 3. Access Services
- **NeuroRAG API**: `http://localhost:8000` (Docs at `/docs`)
- **Grafana Dashboards**: `http://localhost:3000`
- **Airflow UI**: `http://localhost:8080`
- **Prometheus**: `http://localhost:9091`

---

## 🔄 Manual Workflow Triggers

You can manually trigger deployments across environments directly from GitHub:
1. Navigate to the **Actions** tab in your repository.
2. Select **NeuroRAG CI/CD Pipeline** from the left sidebar.
3. Click the **Run workflow** dropdown on the right.
4. Choose the target environment (`staging` or `production`) and execute the run.
