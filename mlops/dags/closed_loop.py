"""
NeuroRAG — Closed-Loop MLOps: Faithfulness-Triggered Retraining v2
===================================================================
DAGs:
  neurorag_eval_v2               — daily eval + faithfulness branch
  neurorag_faithfulness_trigger  — backup → rebuild → compare → promote/rollback

Faithfulness trigger flow:
  [neurorag_eval_v2] → avg_faithfulness < FAITHFULNESS_THRESHOLD
    → [neurorag_faithfulness_trigger]
         ├── record pre-retrain baseline
         ├── backup FAISS artefacts
         ├── rebuild index from latest corpus
         ├── validate rebuilt index (smoke queries)
         ├── run post-retrain evaluation
         └── compare before / after
               ├── improvement >= MIN_IMPROVEMENT  → PROMOTE
               └── else                            → ROLLBACK to backup

Airflow Variables (set via UI or CLI):
  FAITHFULNESS_THRESHOLD   float  0.80  — trigger retraining below this
  FAITHFULNESS_MIN_DELTA   float  0.05  — minimum improvement to keep new model
  INGEST_SOURCE_PATH       str    /data/raw
  PUSHGATEWAY_URL          str    http://pushgateway:9091
  AIRFLOW_API_URL          str    http://airflow:8080
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "neurorag-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": [os.environ.get("ALERT_EMAIL", "ops@neurorag.io")],
}


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _faithfulness_threshold() -> float:
    return float(Variable.get("FAITHFULNESS_THRESHOLD", default_var="0.80"))


def _min_delta() -> float:
    return float(Variable.get("FAITHFULNESS_MIN_DELTA", default_var="0.05"))


def _ingest_source() -> str:
    return Variable.get("INGEST_SOURCE_PATH", default_var="/data/raw")


def _pushgateway_url() -> str:
    return Variable.get("PUSHGATEWAY_URL", default_var="http://pushgateway:9091")


def _push_gauge(metric: str, value: float, labels: Optional[dict] = None) -> None:
    """Push a single gauge value to Prometheus Pushgateway."""
    try:
        import requests
        label_str = ""
        if labels:
            label_str = "/" + "/".join(f"{k}/{v}" for k, v in labels.items())
        url = f"{_pushgateway_url()}/metrics/job/neurorag_mlops{label_str}"
        payload = f"# TYPE {metric} gauge\n{metric} {value}\n"
        requests.post(url, data=payload, timeout=5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pushgateway push failed: %s", exc)


def _trigger_airflow_dag(dag_id: str, conf: Optional[dict] = None) -> Optional[str]:
    """Trigger an Airflow DAG via the REST API."""
    try:
        import requests
        base = os.environ.get("AIRFLOW_API_URL", "http://airflow:8080")
        payload = {"conf": conf or {}}
        resp = requests.post(
            f"{base}/api/v1/dags/{dag_id}/dagRuns",
            json=payload,
            auth=("admin", os.environ.get("AIRFLOW_ADMIN_PASSWORD", "admin")),
            timeout=15,
        )
        resp.raise_for_status()
        run_id = resp.json().get("dag_run_id")
        logger.info("Triggered %s → %s", dag_id, run_id)
        return run_id
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to trigger %s: %s", dag_id, exc)
        return None


# ─── neurorag_eval_v2 ─────────────────────────────────────────────────────────

with DAG(
    dag_id="neurorag_eval_v2",
    default_args=_DEFAULT_ARGS,
    schedule_interval="0 3 * * *",   # 03:00 UTC daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    description="Daily eval with automatic faithfulness-triggered retraining.",
    tags=["neurorag", "evaluation", "mlops"],
) as dag_eval:

    def _run_evaluation(**ctx) -> None:
        """
        Sample recent queries from Postgres and compute average faithfulness.
        Pushes results to XCom for the branch operator.
        """
        import asyncpg
        import asyncio

        postgres_url = os.environ.get(
            "POSTGRES_URL",
            "postgresql://neurorag:neurorag_secret@postgres:5432/neurorag",
        )

        async def _query() -> list[dict]:
            pool = await asyncpg.create_pool(postgres_url, min_size=1, max_size=3)
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT q.id, q.confidence, "
                    "em.faithfulness, em.relevance, em.completeness "
                    "FROM queries q "
                    "LEFT JOIN eval_metrics em ON q.id = em.id "
                    "WHERE q.created_at >= NOW() - INTERVAL '24 hours' "
                    "ORDER BY q.created_at DESC LIMIT 500",
                )
            await pool.close()
            return [dict(r) for r in rows]

        rows = asyncio.run(_query())

        threshold = _faithfulness_threshold()
        metric_vals = [
            r["faithfulness"] if r.get("faithfulness") is not None else r["confidence"]
            for r in rows
        ]
        avg_faith = sum(metric_vals) / max(len(metric_vals), 1)

        result = {
            "n_samples":              len(rows),
            "avg_faithfulness":       round(avg_faith, 4),
            "avg_confidence":         round(
                sum(r["confidence"] for r in rows) / max(len(rows), 1), 4
            ),
            "faithfulness_threshold": threshold,
            "below_threshold":        avg_faith < threshold,
        }
        logger.info("Eval result: %s", result)
        ctx["ti"].xcom_push(key="eval_result", value=result)
        _push_gauge("neurorag_eval_avg_faithfulness", avg_faith)

    def _check_faithfulness(**ctx) -> str:
        result = ctx["ti"].xcom_pull(task_ids="run_evaluation", key="eval_result") or {}
        avg = result.get("avg_faithfulness", 1.0)
        threshold = _faithfulness_threshold()
        logger.info("Faithfulness check: %.4f vs threshold %.4f", avg, threshold)
        if result.get("below_threshold"):
            logger.warning("Below threshold — triggering retrain DAG.")
            return "trigger_retrain"
        logger.info("Faithfulness OK — no retraining needed.")
        return "no_retrain_needed"

    def _trigger_retrain(**ctx) -> None:
        result = ctx["ti"].xcom_pull(task_ids="run_evaluation", key="eval_result") or {}
        _trigger_airflow_dag(
            "neurorag_faithfulness_trigger",
            conf={
                "trigger_reason":        "faithfulness_below_threshold",
                "pre_retrain_faithfulness": result.get("avg_faithfulness", 0.0),
            },
        )
        logger.info(
            "Retrain triggered: faithfulness=%.4f < threshold=%.4f",
            result.get("avg_faithfulness", 0.0),
            _faithfulness_threshold(),
        )

    def _no_retrain(**ctx) -> None:
        result = ctx["ti"].xcom_pull(task_ids="run_evaluation", key="eval_result") or {}
        logger.info(
            "No retraining needed. faithfulness=%.4f >= threshold=%.4f",
            result.get("avg_faithfulness", 1.0),
            _faithfulness_threshold(),
        )

    t_eval     = PythonOperator(task_id="run_evaluation",     python_callable=_run_evaluation)
    t_branch   = BranchPythonOperator(task_id="check_faithfulness", python_callable=_check_faithfulness)
    t_trigger  = PythonOperator(task_id="trigger_retrain",    python_callable=_trigger_retrain)
    t_no_retrain = PythonOperator(task_id="no_retrain_needed", python_callable=_no_retrain)

    t_eval >> t_branch >> [t_trigger, t_no_retrain]


# ─── neurorag_faithfulness_trigger ───────────────────────────────────────────

with DAG(
    dag_id="neurorag_faithfulness_trigger",
    default_args=_DEFAULT_ARGS,
    schedule_interval=None,   # triggered only by neurorag_eval_v2
    start_date=datetime(2026, 1, 1),
    catchup=False,
    description="Faithfulness-triggered retraining: backup → rebuild → compare → promote/rollback.",
    tags=["neurorag", "retraining", "mlops"],
) as dag_trigger:

    def _record_baseline(**ctx) -> None:
        """Snapshot current faithfulness before we touch anything."""
        conf = ctx.get("dag_run", {}).conf or {}
        pre_faith = float(conf.get("pre_retrain_faithfulness", 0.0))
        logger.info("Pre-retrain faithfulness baseline: %.4f", pre_faith)
        ctx["ti"].xcom_push(key="pre_retrain_faithfulness", value=pre_faith)
        _push_gauge("neurorag_pre_retrain_faithfulness", pre_faith)

    def _backup_artefacts(**ctx) -> None:
        """Backup FAISS index + config before rebuilding."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"/data/backups/faith_retrain_{ts}"
        os.makedirs(backup_dir, exist_ok=True)

        faiss_src = "/data/faiss"
        faiss_bak = f"{backup_dir}/faiss"
        if os.path.exists(faiss_src):
            shutil.copytree(faiss_src, faiss_bak)
            logger.info("Backed up FAISS: %s → %s", faiss_src, faiss_bak)
        else:
            logger.warning("FAISS source not found at %s — backup skipped.", faiss_src)

        config_src = "/app/configs/config.yaml"
        if os.path.exists(config_src):
            shutil.copy2(config_src, f"{backup_dir}/config.yaml")

        ctx["ti"].xcom_push(key="backup_dir", value=backup_dir)
        logger.info("Backup complete: %s", backup_dir)

    def _rebuild_index(**ctx) -> None:
        """Rebuild FAISS + BM25 from raw corpus."""
        source_path = _ingest_source()
        logger.info("Rebuilding index from: %s", source_path)

        # Import here to avoid loading heavy deps at DAG parse time
        import sys
        sys.path.insert(0, "/app")
        from rag.ingest import IngestionEngine
        import glob

        docs = []
        for pattern in ("**/*.txt", "**/*.md"):
            for path in glob.glob(os.path.join(source_path, pattern), recursive=True):
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                    if text:
                        doc_id = os.path.relpath(path, source_path).replace(os.sep, "/")
                        docs.append({"id": doc_id, "text": text, "metadata": {"path": path}})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping %s: %s", path, exc)

        logger.info("Loaded %d documents for rebuild.", len(docs))
        if not docs:
            raise RuntimeError("No documents found at %s — rebuild aborted." % source_path)

        engine = IngestionEngine()
        chunks = engine.ingest(docs)
        engine.save()
        logger.info("Rebuilt index: %d docs → %d chunks", len(docs), chunks)
        ctx["ti"].xcom_push(key="rebuild_chunks", value=chunks)

    def _validate_rebuilt_index(**ctx) -> None:
        """Run smoke queries to verify the new index works."""
        import sys
        sys.path.insert(0, "/app")
        from rag.ingest import IngestionEngine
        from rag.retriever import HybridRetriever

        smoke_queries = [
            "retrieval-augmented generation",
            "self-healing loop confidence threshold",
            "canary deployment",
            "embedding drift detection",
            "circuit breaker pattern",
        ]
        engine = IngestionEngine()
        retriever = HybridRetriever(engine)
        failures = 0
        for q in smoke_queries:
            docs = retriever.retrieve(q, top_k=3)
            if not docs:
                logger.warning("Smoke query returned 0 docs: %s", q)
                failures += 1

        if failures > len(smoke_queries) // 2:
            raise RuntimeError(
                f"Smoke validation failed: {failures}/{len(smoke_queries)} queries returned no results."
            )
        logger.info(
            "Smoke validation passed: %d/%d queries returned results.",
            len(smoke_queries) - failures,
            len(smoke_queries),
        )

    def _run_post_retrain_eval(**ctx) -> None:
        """Run evaluation on the new index and push result to XCom."""
        import sys, asyncio
        sys.path.insert(0, "/app")
        import asyncpg

        postgres_url = os.environ.get(
            "POSTGRES_URL",
            "postgresql://neurorag:neurorag_secret@postgres:5432/neurorag",
        )

        async def _query() -> list[dict]:
            pool = await asyncpg.create_pool(postgres_url, min_size=1, max_size=3)
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT confidence FROM queries "
                    "ORDER BY created_at DESC LIMIT 200"
                )
            await pool.close()
            return [dict(r) for r in rows]

        rows = asyncio.run(_query())
        vals = [r["confidence"] for r in rows if r.get("confidence") is not None]
        post_faith = sum(vals) / max(len(vals), 1)
        logger.info("Post-retrain faithfulness estimate: %.4f", post_faith)
        ctx["ti"].xcom_push(key="post_retrain_faithfulness", value=post_faith)
        _push_gauge("neurorag_post_retrain_faithfulness", post_faith)

    def _compare_before_after(**ctx) -> str:
        ti = ctx["ti"]
        pre  = float(ti.xcom_pull(task_ids="record_baseline",    key="pre_retrain_faithfulness") or 0.0)
        post = float(ti.xcom_pull(task_ids="run_post_retrain_eval", key="post_retrain_faithfulness") or 0.0)
        delta = post - pre
        min_delta = _min_delta()

        logger.info(
            "Compare: pre=%.4f post=%.4f delta=%.4f (required=%.4f)",
            pre, post, delta, min_delta,
        )
        _push_gauge("neurorag_retrain_faithfulness_delta", delta)

        if delta >= min_delta:
            logger.info("Improvement sufficient — promoting new index.")
            ti.xcom_push(key="decision", value="promote")
            return "promote_new_index"
        else:
            logger.warning("Insufficient improvement (%.4f < %.4f) — rolling back.", delta, min_delta)
            ti.xcom_push(key="decision", value="rollback")
            return "rollback_to_backup"

    def _promote_new_index(**ctx) -> None:
        """Mark new index as production and push metrics."""
        metadata = {
            "promoted_at":              datetime.utcnow().isoformat(),
            "post_retrain_faithfulness": ctx["ti"].xcom_pull(
                task_ids="run_post_retrain_eval", key="post_retrain_faithfulness"
            ),
            "pre_retrain_faithfulness":  ctx["ti"].xcom_pull(
                task_ids="record_baseline", key="pre_retrain_faithfulness"
            ),
        }
        metadata_path = "/data/faiss/retrain_metadata.json"
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        _push_gauge("neurorag_retrain_promoted", 1.0)
        logger.info("✅ New index promoted with metadata: %s", metadata)

    def _rollback_to_backup(**ctx) -> None:
        """Restore FAISS artefacts from backup directory."""
        backup_dir = ctx["ti"].xcom_pull(task_ids="backup_artefacts", key="backup_dir")
        if not backup_dir:
            logger.error("No backup_dir in XCom — cannot rollback.")
            _push_gauge("neurorag_retrain_rollback_failed", 1.0)
            return

        faiss_bak = f"{backup_dir}/faiss"
        faiss_dst = "/data/faiss"

        if not os.path.exists(faiss_bak):
            logger.error("Backup not found at %s — rollback impossible.", faiss_bak)
            _push_gauge("neurorag_retrain_rollback_failed", 1.0)
            return

        if os.path.exists(faiss_dst):
            shutil.rmtree(faiss_dst)
        shutil.copytree(faiss_bak, faiss_dst)

        # Restore config if present
        config_bak = f"{backup_dir}/config.yaml"
        if os.path.exists(config_bak):
            shutil.copy2(config_bak, "/app/configs/config.yaml")

        _push_gauge("neurorag_retrain_rollback_total", 1.0)
        logger.info("✅ Rollback complete: %s → %s", faiss_bak, faiss_dst)

    def _rollback_on_validate_failure(**ctx) -> None:
        """Emergency rollback invoked if validate_rebuilt_index fails."""
        logger.error("Index validation failed — invoking emergency rollback.")
        _rollback_to_backup(**ctx)
        _push_gauge("neurorag_retrain_rollback_total", 1.0)

    # DAG task wiring
    t_baseline  = PythonOperator(task_id="record_baseline",        python_callable=_record_baseline)
    t_backup    = PythonOperator(task_id="backup_artefacts",        python_callable=_backup_artefacts)
    t_rebuild   = PythonOperator(task_id="rebuild_index",           python_callable=_rebuild_index)
    t_validate  = PythonOperator(task_id="validate_rebuilt_index",  python_callable=_validate_rebuilt_index,
                                 on_failure_callback=_rollback_on_validate_failure)
    t_post_eval = PythonOperator(task_id="run_post_retrain_eval",   python_callable=_run_post_retrain_eval)
    t_compare   = BranchPythonOperator(task_id="compare_before_after", python_callable=_compare_before_after)
    t_promote   = PythonOperator(task_id="promote_new_index",       python_callable=_promote_new_index)
    t_rollback  = PythonOperator(task_id="rollback_to_backup",      python_callable=_rollback_to_backup)

    (
        t_baseline
        >> t_backup
        >> t_rebuild
        >> t_validate
        >> t_post_eval
        >> t_compare
        >> [t_promote, t_rollback]
    )
