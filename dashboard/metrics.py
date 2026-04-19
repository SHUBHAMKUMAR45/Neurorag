"""
NeuroRAG — Prometheus Metrics v2
Defines all counters, histograms, and gauges.
Added: neurorag_retrain_promoted, neurorag_retrain_rollback_total,
       neurorag_retrain_rollback_failed (wired to MLOps DAG).
"""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

# ─── System Info ─────────────────────────────────────────────────────────────
SYSTEM_INFO = Info("neurorag", "NeuroRAG system metadata")
SYSTEM_INFO.info({"version": "3.0.0", "environment": "production"})

# ─── Query Metrics ───────────────────────────────────────────────────────────
QUERY_TOTAL = Counter(
    "neurorag_queries_total",
    "Total number of queries processed",
    ["status"],   # status: success | insufficient | error
)

QUERY_LATENCY = Histogram(
    "neurorag_query_latency_ms",
    "Query end-to-end latency in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2000, 5000],
)

QUERY_LOOPS = Histogram(
    "neurorag_query_loops",
    "Number of self-healing iterations per query",
    buckets=[1, 2, 3, 4, 5],
)

CONFIDENCE_SCORE = Histogram(
    "neurorag_confidence_score",
    "Critic confidence score per query",
    buckets=[0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# ─── Hallucination ───────────────────────────────────────────────────────────
HALLUCINATION_TOTAL = Counter(
    "neurorag_hallucinations_total",
    "Total queries where hallucination was detected",
)

HALLUCINATION_RATE = Gauge(
    "neurorag_hallucination_rate",
    "Rolling hallucination rate (last 1000 queries)",
)

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_LATENCY = Histogram(
    "neurorag_retrieval_latency_ms",
    "Retrieval (BM25 + vector) latency",
    buckets=[10, 25, 50, 100, 250, 500],
)

RERANK_LATENCY = Histogram(
    "neurorag_rerank_latency_ms",
    "Reranker latency",
    buckets=[10, 25, 50, 100, 250],
)

DOCS_RETRIEVED = Histogram(
    "neurorag_docs_retrieved",
    "Number of documents retrieved before reranking",
    buckets=[1, 5, 10, 15, 20, 30],
)

# ─── Agent Latencies ─────────────────────────────────────────────────────────
AGENT_LATENCY = Histogram(
    "neurorag_agent_latency_ms",
    "Per-agent execution latency",
    ["agent"],   # intent | planner | generator | critic | reflection | fixer
    buckets=[10, 50, 100, 250, 500, 1000, 2000],
)

# ─── GPU ─────────────────────────────────────────────────────────────────────
GPU_UTILIZATION = Gauge(
    "neurorag_gpu_utilization_percent",
    "GPU utilization percentage (NVIDIA SMI)",
)

GPU_MEMORY_USED_MB = Gauge(
    "neurorag_gpu_memory_used_mb",
    "GPU memory used in MB",
)

# ─── Ingestion ───────────────────────────────────────────────────────────────
INGEST_DOCS_TOTAL = Counter(
    "neurorag_ingest_docs_total",
    "Total documents ingested",
)

INGEST_CHUNKS_TOTAL = Counter(
    "neurorag_ingest_chunks_total",
    "Total chunks indexed",
)

FAISS_INDEX_SIZE = Gauge(
    "neurorag_faiss_index_size",
    "Number of vectors in FAISS index",
)

# ─── MLOps Retraining ────────────────────────────────────────────────────────
RETRAIN_PROMOTED = Gauge(
    "neurorag_retrain_promoted",
    "Set to 1 when a new index is promoted after retraining",
)

RETRAIN_ROLLBACK_TOTAL = Counter(
    "neurorag_retrain_rollback_total",
    "Total number of retrain rollbacks executed",
)

RETRAIN_ROLLBACK_FAILED = Counter(
    "neurorag_retrain_rollback_failed",
    "Total number of retrain rollback failures (backup not found)",
)


# ─── GPU Collector (background thread) ───────────────────────────────────────

def _start_gpu_collector(interval_seconds: int = 15) -> None:
    """Poll nvidia-smi in background and update GPU gauges."""
    import subprocess
    import threading

    def _collect() -> None:
        import time
        while True:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                ).decode().strip().split(",")
                GPU_UTILIZATION.set(float(out[0].strip()))
                GPU_MEMORY_USED_MB.set(float(out[1].strip()))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval_seconds)

    t = threading.Thread(target=_collect, daemon=True)
    t.start()


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus HTTP server and GPU collector."""
    start_http_server(port)
    _start_gpu_collector()
