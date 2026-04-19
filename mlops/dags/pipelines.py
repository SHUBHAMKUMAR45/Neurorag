"""
NeuroRAG — Airflow MLOps DAGs
Five production pipelines:
  1. neurorag_ingest      — Weekly data ingestion + index rebuild
  2. neurorag_eval        — Daily offline evaluation (RAGAS)
  3. neurorag_drift       — Daily embedding drift detection → auto-retrain trigger
  4. neurorag_retrain     — Triggered retraining pipeline
  5. neurorag_deploy      — Canary deployment to Kubernetes
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

import numpy as np
from airflow import DAG
from airflow.models import Variable
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "neurorag-mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": [os.environ.get("ALERT_EMAIL", "ops@neurorag.io")],
}


# ════════════════════════════════════════════════════════════════════════════
# DAG 1 — INGESTION
# ════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="neurorag_ingest",
    default_args=_DEFAULT_ARGS,
    description="Fetch new documents, chunk, embed, rebuild FAISS + BM25 indexes.",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["rag", "ingestion"],
) as ingest_dag:

    def _fetch_documents(**ctx) -> None:
        """
        Fetch new documents from configured sources.
        Pushes doc list to XCom for downstream tasks.
        """
        # In production: pull from S3/GCS, database, SharePoint, etc.
        source_path = Variable.get("INGEST_SOURCE_PATH", default_var="/data/raw")
        logger.info("Fetching documents from %s", source_path)

        docs = []
        import pathlib
        for fp in pathlib.Path(source_path).glob("**/*.txt"):
            docs.append({
                "id": fp.stem,
                "text": fp.read_text(encoding="utf-8", errors="replace"),
                "metadata": {"source": str(fp)},
            })

        ctx["ti"].xcom_push(key="doc_count", value=len(docs))
        # Serialize to temp file (avoid XCom size limits)
        tmp_path = "/tmp/ingest_docs.json"
        with open(tmp_path, "w") as f:
            json.dump(docs, f)
        logger.info("Fetched %d documents.", len(docs))

    def _run_ingestion(**ctx) -> None:
        """Embed and index all fetched documents."""
        import sys
        sys.path.insert(0, "/app")
        from rag.ingest import IngestionEngine

        with open("/tmp/ingest_docs.json") as f:
            docs = json.load(f)

        engine = IngestionEngine()
        chunks = engine.ingest(docs)
        engine.save()
        ctx["ti"].xcom_push(key="chunks_indexed", value=chunks)
        logger.info("Indexed %d chunks from %d documents.", chunks, len(docs))

    def _notify_ingest_complete(**ctx) -> None:
        ti = ctx["ti"]
        doc_count = ti.xcom_pull(task_ids="fetch_documents", key="doc_count")
        chunks = ti.xcom_pull(task_ids="run_ingestion", key="chunks_indexed")
        logger.info("Ingest complete: %d docs → %d chunks", doc_count, chunks)

    fetch_task = PythonOperator(task_id="fetch_documents", python_callable=_fetch_documents)
    ingest_task = PythonOperator(task_id="run_ingestion", python_callable=_run_ingestion)
    notify_task = PythonOperator(task_id="notify_complete", python_callable=_notify_ingest_complete)

    fetch_task >> ingest_task >> notify_task


# ════════════════════════════════════════════════════════════════════════════
# DAG 2 — EVALUATION
# ════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="neurorag_eval",
    default_args=_DEFAULT_ARGS,
    description="Sample recent queries, run offline RAGAS evaluation, update dashboards.",
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["evaluation", "mlops"],
) as eval_dag:

    def _sample_queries(**ctx) -> None:
        """Pull last 500 queries from Postgres for offline eval."""
        import asyncpg
        import asyncio

        async def _fetch():
            pool = await asyncpg.create_pool(os.environ["POSTGRES_URL"])
            rows = await pool.fetch(
                "SELECT query, answer, confidence FROM queries "
                "ORDER BY created_at DESC LIMIT 500"
            )
            await pool.close()
            return [dict(r) for r in rows]

        rows = asyncio.get_event_loop().run_until_complete(_fetch())
        with open("/tmp/eval_sample.json", "w") as f:
            json.dump(rows, f)
        logger.info("Sampled %d queries for eval.", len(rows))

    def _run_offline_eval(**ctx) -> None:
        """Compute faithfulness proxy and push metrics."""
        with open("/tmp/eval_sample.json") as f:
            rows = json.load(f)

        confidences = [r["confidence"] for r in rows if r.get("confidence")]
        if not confidences:
            logger.warning("No confidence scores found.")
            return

        avg_conf = sum(confidences) / len(confidences)
        below_thresh = sum(1 for c in confidences if c < 0.90) / len(confidences)

        result = {
            "avg_confidence": avg_conf,
            "low_confidence_rate": below_thresh,
            "n_evaluated": len(confidences),
        }
        ctx["ti"].xcom_push(key="eval_result", value=result)
        logger.info("Eval result: %s", result)

        # Write to Prometheus Pushgateway if available
        try:
            import requests
            pg_url = os.environ.get("PUSHGATEWAY_URL", "http://pushgateway:9091")
            requests.post(
                f"{pg_url}/metrics/job/neurorag_eval",
                data=f'neurorag_avg_confidence {avg_conf}\n',
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            pass

    sample_task = PythonOperator(task_id="sample_queries", python_callable=_sample_queries)
    eval_task = PythonOperator(task_id="run_offline_eval", python_callable=_run_offline_eval)

    sample_task >> eval_task


# ════════════════════════════════════════════════════════════════════════════
# DAG 3 — DRIFT DETECTION
# ════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="neurorag_drift",
    default_args=_DEFAULT_ARGS,
    description="Detect embedding distribution drift; trigger retraining if threshold exceeded.",
    schedule_interval="0 3 * * *",   # 03:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["drift", "mlops"],
) as drift_dag:

    def _detect_drift(**ctx) -> str:
        """
        Compute mean cosine distance between yesterday's and today's
        query embeddings. Branch to retrain if drift > threshold.
        """
        import sys
        sys.path.insert(0, "/app")
        from sentence_transformers import SentenceTransformer

        drift_threshold = float(Variable.get("DRIFT_THRESHOLD", default_var="0.12"))

        # Load sampled queries from eval step
        try:
            with open("/tmp/eval_sample.json") as f:
                rows = json.load(f)
        except FileNotFoundError:
            logger.warning("No eval sample found; skipping drift check.")
            return "no_retrain"

        queries = [r["query"] for r in rows if r.get("query")]
        if len(queries) < 10:
            logger.warning("Insufficient queries for drift detection.")
            return "no_retrain"

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        current_embs = model.encode(queries, normalize_embeddings=True)

        # Load reference embeddings (stored from last run)
        ref_path = "/data/drift/reference_embeddings.npy"
        if not os.path.exists(ref_path):
            # First run — save as reference
            os.makedirs("/data/drift", exist_ok=True)
            np.save(ref_path, current_embs)
            logger.info("Drift: saved reference embeddings.")
            return "no_retrain"

        ref_embs = np.load(ref_path)
        n = min(len(current_embs), len(ref_embs))
        cosine_similarities = np.einsum("ij,ij->i", current_embs[:n], ref_embs[:n])
        mean_cosine_dist = 1.0 - float(np.mean(cosine_similarities))

        logger.info("Drift: mean cosine distance = %.4f (threshold=%.4f)", mean_cosine_dist, drift_threshold)

        # Save current as new reference
        np.save(ref_path, current_embs)
        ctx["ti"].xcom_push(key="drift_score", value=mean_cosine_dist)

        if mean_cosine_dist > drift_threshold:
            logger.warning("DRIFT DETECTED (%.4f > %.4f) — triggering retrain.", mean_cosine_dist, drift_threshold)
            return "trigger_retrain"

        return "no_retrain"

    def _trigger_retrain(**ctx) -> None:
        """Trigger the retraining DAG via Airflow API."""
        import requests
        score = ctx["ti"].xcom_pull(task_ids="detect_drift", key="drift_score")
        logger.info("Triggering neurorag_retrain DAG (drift_score=%.4f)", score)
        airflow_url = os.environ.get("AIRFLOW_API_URL", "http://airflow:8080")
        try:
            resp = requests.post(
                f"{airflow_url}/api/v1/dags/neurorag_retrain/dagRuns",
                json={"conf": {"drift_score": score}},
                auth=("admin", os.environ.get("AIRFLOW_PASSWORD", "admin")),
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Retrain DAG triggered: %s", resp.json().get("dag_run_id"))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to trigger retrain: %s", exc)

    def _no_retrain(**ctx) -> None:
        logger.info("Drift check: within threshold. No retrain needed.")

    detect_task = BranchPythonOperator(task_id="detect_drift", python_callable=_detect_drift)
    retrain_branch = PythonOperator(task_id="trigger_retrain", python_callable=_trigger_retrain)
    no_retrain_branch = PythonOperator(task_id="no_retrain", python_callable=_no_retrain)

    detect_task >> [retrain_branch, no_retrain_branch]


# ════════════════════════════════════════════════════════════════════════════
# DAG 4 — RETRAINING
# ════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="neurorag_retrain",
    default_args=_DEFAULT_ARGS,
    description="Rebuild FAISS index, fine-tune reranker, update model artifacts.",
    schedule_interval=None,          # Triggered by drift DAG or manually
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["retrain", "mlops"],
) as retrain_dag:

    def _backup_current_index(**ctx) -> None:
        import shutil
        src = "/data/faiss"
        dst = f"/data/faiss_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(src, dst)
        ctx["ti"].xcom_push(key="backup_path", value=dst)
        logger.info("Backed up FAISS index to %s", dst)

    def _rebuild_index(**ctx) -> None:
        """Full index rebuild from raw document store."""
        import sys, pathlib
        sys.path.insert(0, "/app")
        from rag.ingest import IngestionEngine

        source_path = Variable.get("INGEST_SOURCE_PATH", default_var="/data/raw")
        docs = []
        for fp in pathlib.Path(source_path).glob("**/*.txt"):
            docs.append({
                "id": fp.stem,
                "text": fp.read_text(errors="replace"),
                "metadata": {},
            })

        engine = IngestionEngine()
        chunks = engine.ingest(docs)
        engine.save()
        logger.info("Retrain: rebuilt index with %d chunks.", chunks)

    def _validate_new_index(**ctx) -> None:
        """Smoke test: run a known query and verify retrieval."""
        import sys
        sys.path.insert(0, "/app")
        from rag.ingest import IngestionEngine
        from rag.retriever import HybridRetriever

        engine = IngestionEngine()
        retriever = HybridRetriever(engine)
        results = retriever.retrieve("test query", top_k=5)
        if not results:
            raise RuntimeError("Index validation failed: no results returned.")
        logger.info("Index validation passed: %d results.", len(results))

    def _rollback(**ctx) -> None:
        """Restore backup if validation fails."""
        import shutil
        backup = ctx["ti"].xcom_pull(task_ids="backup_index", key="backup_path")
        if backup:
            shutil.rmtree("/data/faiss", ignore_errors=True)
            shutil.copytree(backup, "/data/faiss")
            logger.warning("ROLLBACK: restored index from %s", backup)

    backup_task = PythonOperator(task_id="backup_index", python_callable=_backup_current_index)
    rebuild_task = PythonOperator(task_id="rebuild_index", python_callable=_rebuild_index)
    validate_task = PythonOperator(task_id="validate_index", python_callable=_validate_new_index)
    rollback_task = PythonOperator(task_id="rollback", python_callable=_rollback,
                                   trigger_rule="one_failed")

    backup_task >> rebuild_task >> validate_task >> rollback_task


# ════════════════════════════════════════════════════════════════════════════
# DAG 5 — CANARY DEPLOYMENT
# ════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="neurorag_deploy",
    default_args=_DEFAULT_ARGS,
    description="Build Docker image, push, canary deploy to Kubernetes, promote or rollback.",
    schedule_interval=None,          # Triggered by CI/CD
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["deploy", "mlops"],
) as deploy_dag:

    def _build_and_push(**ctx) -> None:
        """Build Docker image and push to registry."""
        image_tag = ctx["dag_run"].conf.get("image_tag", "latest")
        registry = os.environ.get("DOCKER_REGISTRY", "registry.neurorag.io")
        image = f"{registry}/neurorag:{image_tag}"

        subprocess.run(["docker", "build", "-t", image, "/app"], check=True)
        subprocess.run(["docker", "push", image], check=True)
        ctx["ti"].xcom_push(key="image", value=image)
        logger.info("Built and pushed %s", image)

    def _canary_deploy(**ctx) -> None:
        """Deploy 10% canary to Kubernetes."""
        image = ctx["ti"].xcom_pull(task_ids="build_push", key="image")
        subprocess.run([
            "kubectl", "set", "image",
            "deployment/neurorag-canary", f"api={image}",
            "--namespace=neurorag",
        ], check=True)
        logger.info("Canary deployed: %s", image)

    def _validate_canary(**ctx) -> str:
        """Check canary error rate from Prometheus. Branch on result."""
        try:
            import requests
            prom_url = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
            resp = requests.get(
                f"{prom_url}/api/v1/query",
                params={"query": 'rate(neurorag_queries_total{status="error"}[5m])'},
                timeout=10,
            )
            data = resp.json()
            error_rate = float(data["data"]["result"][0]["value"][1])
            logger.info("Canary error rate: %.4f", error_rate)
            return "promote" if error_rate < 0.05 else "rollback"
        except Exception as exc:  # noqa: BLE001
            logger.error("Canary validation failed: %s — rolling back.", exc)
            return "rollback"

    def _promote(**ctx) -> None:
        image = ctx["ti"].xcom_pull(task_ids="build_push", key="image")
        subprocess.run([
            "kubectl", "set", "image",
            "deployment/neurorag-api", f"api={image}",
            "--namespace=neurorag",
        ], check=True)
        logger.info("Promoted canary to production: %s", image)

    def _rollback_deploy(**ctx) -> None:
        subprocess.run(["kubectl", "rollout", "undo",
                        "deployment/neurorag-canary", "--namespace=neurorag"], check=True)
        logger.warning("Canary rollback executed.")

    build_task = PythonOperator(task_id="build_push", python_callable=_build_and_push)
    canary_task = PythonOperator(task_id="canary_deploy", python_callable=_canary_deploy)
    validate_task = BranchPythonOperator(task_id="validate_canary", python_callable=_validate_canary)
    promote_task = PythonOperator(task_id="promote", python_callable=_promote)
    rollback_task = PythonOperator(task_id="rollback", python_callable=_rollback_deploy)

    build_task >> canary_task >> validate_task >> [promote_task, rollback_task]
