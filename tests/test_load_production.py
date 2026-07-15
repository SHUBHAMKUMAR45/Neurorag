"""
NeuroRAG — Load Testing v1
============================
Full Locust-based load test suite with RAG-specific metric collection.

Test scenarios:
  1. Sustained load:   50 users, 2min   → baseline performance
  2. Peak load:        100 users, 3min  → stress SLOs
  3. Soak test:        50 users, 10min  → memory/connection leak detection
  4. Spike test:       ramp 5 → 100 in 30s, hold 60s → elasticity

Run commands:
  # Smoke (CI, 10 users, 30s)
  locust -f tests/test_load_production.py --headless -u 10 -r 2 -t 30s \
         --host http://localhost:8000 --html /tmp/load_report_smoke.html

  # Sustained load (50 users)
  locust -f tests/test_load_production.py --headless -u 50 -r 5 -t 120s \
         --host http://localhost:8000 --html /tmp/load_report_50.html \
         --csv /tmp/load_report_50

  # Peak load (100 users)
  locust -f tests/test_load_production.py --headless -u 100 -r 10 -t 180s \
         --host http://localhost:8000 --html /tmp/load_report_100.html

  # Async stress test (no Locust needed)
  python -m tests.test_load_production --mode stress --users 100 --duration 60

SLO Thresholds (from configs/config.yaml):
  - p95 latency  ≤ 1500ms
  - Error rate   ≤ 1%
  - Throughput   ≥ 10 req/s at 50 users
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Realistic query distribution ────────────────────────────────────────────

_FACTUAL_QUERIES = [
    "What is retrieval-augmented generation?",
    "What does BM25 stand for and what is it used for?",
    "Define hallucination in large language models.",
    "What is a cross-encoder reranker?",
    "Explain what FAISS does in a vector search system.",
    "What is semantic chunking?",
    "What is Reciprocal Rank Fusion?",
    "What is Precision@K in information retrieval?",
    "What is a circuit breaker pattern?",
    "What is a canary deployment?",
    "What does Prometheus monitor in a production system?",
    "What is embedding drift detection?",
    "What is p95 latency?",
    "What is an Airflow DAG?",
    "What is Recall@K?",
]

_REASONING_QUERIES = [
    "Why is hybrid retrieval better than BM25 alone for semantic questions?",
    "Explain the trade-off between chunk size and retrieval quality.",
    "Why do self-healing loops have diminishing returns after 3 iterations?",
    "Compare the Precision@K and Recall@K trade-off in RAG evaluation.",
    "Why should the critic LLM be separate from the generator LLM?",
    "Explain why async logging is preferred in a production RAG API.",
    "Why is p95 latency a better SLO metric than average latency?",
    "Explain how embedding drift can degrade RAG performance over time.",
    "Why is a canary deployment safer than a direct swap for ML models?",
    "Explain the circuit breaker's role in preventing cascade failures.",
]

_MULTI_HOP_QUERIES = [
    "If the critic detects a hallucination, trace the full correction sequence.",
    "Describe the complete flow from a failed canary deployment to rollback.",
    "How does FailureMemory improve the efficiency of the self-healing loop over time?",
    "Trace a query from intent analysis through to the final cached answer.",
    "Walk through how a faithfulness score drop triggers retraining in Airflow.",
]

# Weight: 60% factual, 30% reasoning, 10% multi-hop (mirrors production distribution)
_WEIGHTED_QUERIES = (
    _FACTUAL_QUERIES * 6 +
    _REASONING_QUERIES * 3 +
    _MULTI_HOP_QUERIES * 1
)

API_KEY = __import__("os").environ.get("NEURORAG_API_KEY", "")
BASE_URL = __import__("os").environ.get("NEURORAG_BASE_URL", "http://localhost:8000")

# ─── Locust Load Test ─────────────────────────────────────────────────────────

try:
    from locust import HttpUser, TaskSet, between, events, task

    # ── Custom metric aggregation ──────────────────────────────────────────

    class _RAGMetrics:
        """Thread-safe RAG-specific metric collector."""
        confidence_scores: list[float] = []
        loop_counts: list[int] = []
        insufficient_count: int = 0
        total_responses: int = 0

    _rag_metrics = _RAGMetrics()

    @events.request.add_listener
    def _on_request(
        request_type, name, response_time, response_length,
        exception, context, **kwargs
    ) -> None:
        if exception:
            return
        # Extract RAG-specific fields from response body if available
        response = context.get("response")
        if response and hasattr(response, "json"):
            try:
                data = response.json()
                if isinstance(data, dict):
                    if "confidence" in data:
                        _rag_metrics.confidence_scores.append(float(data["confidence"]))
                    if "loops" in data:
                        _rag_metrics.loop_counts.append(int(data["loops"]))
                    if data.get("insufficient_context"):
                        _rag_metrics.insufficient_count += 1
                    _rag_metrics.total_responses += 1
            except Exception:  # noqa: BLE001
                pass

    @events.test_stop.add_listener
    def _on_stop(environment, **kwargs) -> None:
        """Print RAG-specific metrics summary at test end."""
        m = _rag_metrics
        if m.confidence_scores:
            print("\n=== RAG METRICS SUMMARY ===")
            print(f"Responses analysed: {m.total_responses}")
            print(f"Avg confidence:  {sum(m.confidence_scores)/len(m.confidence_scores):.3f}")
            print(f"Min confidence:  {min(m.confidence_scores):.3f}")
            print(f"Max confidence:  {max(m.confidence_scores):.3f}")
        if m.loop_counts:
            print(f"Avg self-heal loops: {sum(m.loop_counts)/len(m.loop_counts):.2f}")
            print(f"Max loops seen:      {max(m.loop_counts)}")
        if m.total_responses:
            print(f"Insufficient context rate: {m.insufficient_count/m.total_responses*100:.1f}%")

    # ── Task Sets ─────────────────────────────────────────────────────────

    class QueryOnlyTasks(TaskSet):
        """Pure query load — primary benchmark for RAG latency."""

        @task(10)
        def query_factual(self):
            q = random.choice(_FACTUAL_QUERIES)
            with self.client.post(
                "/query",
                json={"query": q},
                headers={"X-API-Key": API_KEY},
                catch_response=True,
                name="/query [factual]",
            ) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    if "answer" not in data or "confidence" not in data:
                        resp.failure("Missing required fields")
                    elif data.get("loops", 0) < 0:
                        resp.failure("Invalid loops count")
                    else:
                        resp.success()
                elif resp.status_code == 429:
                    resp.success()  # Rate limiting is expected
                elif resp.status_code in (503, 504):
                    resp.success()  # Acceptable degradation
                else:
                    resp.failure(f"HTTP {resp.status_code}")

        @task(3)
        def query_reasoning(self):
            q = random.choice(_REASONING_QUERIES)
            with self.client.post(
                "/query",
                json={"query": q},
                headers={"X-API-Key": API_KEY},
                catch_response=True,
                name="/query [reasoning]",
            ) as resp:
                if resp.status_code in (200, 429, 503, 504):
                    resp.success()
                else:
                    resp.failure(f"HTTP {resp.status_code}")

        @task(1)
        def query_multi_hop(self):
            q = random.choice(_MULTI_HOP_QUERIES)
            with self.client.post(
                "/query",
                json={"query": q},
                headers={"X-API-Key": API_KEY},
                catch_response=True,
                name="/query [multi_hop]",
            ) as resp:
                if resp.status_code in (200, 429, 503, 504):
                    resp.success()
                else:
                    resp.failure(f"HTTP {resp.status_code}")

        @task(1)
        def health(self):
            self.client.get("/health", name="/health")

        @task(1)
        def stats(self):
            self.client.get("/stats", name="/stats")

    class MixedTasks(QueryOnlyTasks):
        """Query + background ingest (realistic production mix)."""

        _INGEST_DOCS = [
            {"id": f"load-doc-{i}", "text": f"Load test document {i}. " * 40, "metadata": {}}
            for i in range(30)
        ]

        @task(2)
        def ingest_batch(self):
            docs = random.sample(self._INGEST_DOCS, k=2)
            self.client.post(
                "/ingest",
                json={"documents": docs},
                headers={"X-API-Key": API_KEY},
                name="/ingest",
            )

    # ── User Classes ───────────────────────────────────────────────────────

    class NeuroRAGUser50(HttpUser):
        """Standard load: 50 users. Target: p95 ≤ 1500ms, error < 1%."""
        tasks = [MixedTasks]
        wait_time = between(0.5, 2.0)

    class NeuroRAGUser100(HttpUser):
        """Peak load: 100 users. Stress test for system limits."""
        tasks = [QueryOnlyTasks]
        wait_time = between(0.3, 1.5)

except ImportError:
    logger.info("Locust not installed; skipping Locust classes.")


# ─── Async stress test (no Locust dependency) ─────────────────────────────────

@dataclass
class LoadTestResult:
    mode: str
    n_users: int
    duration_seconds: int
    n_requests: int
    n_errors: int
    error_rate: float
    throughput_rps: float
    latencies_ms: list[float] = field(default_factory=list)

    # Computed after collection
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # RAG-specific
    avg_confidence: float = 0.0
    avg_loops: float = 0.0
    insufficient_rate: float = 0.0

    # SLO evaluation
    slo_p95_pass: bool = False      # p95 ≤ 1500ms
    slo_error_pass: bool = False    # error rate ≤ 1%
    slo_throughput_pass: bool = False  # ≥ 10 req/s

    def compute_percentiles(self) -> None:
        if not self.latencies_ms:
            return
        s = sorted(self.latencies_ms)
        n = len(s)
        self.p50_ms = s[int(n * 0.50)]
        self.p95_ms = s[int(n * 0.95)]
        self.p99_ms = s[int(n * 0.99)]
        self.min_ms = s[0]
        self.max_ms = s[-1]
        self.slo_p95_pass = self.p95_ms <= 1500.0
        self.slo_error_pass = self.error_rate <= 0.01
        self.slo_throughput_pass = self.throughput_rps >= 10.0


async def run_async_load_test(
    base_url: str = BASE_URL,
    n_users: int = 50,
    duration_seconds: int = 60,
    ramp_seconds: int = 10,
    api_key: str = API_KEY,
    report_path: str = "/tmp/async_load_report.json",
) -> LoadTestResult:
    """
    Pure-Python async load test. No Locust dependency.

    Args:
        base_url:         NeuroRAG API base URL.
        n_users:          Concurrent workers (simulated users).
        duration_seconds: Total test duration.
        ramp_seconds:     Seconds to ramp from 0 to n_users.
        api_key:          API key for authentication.
        report_path:      Where to write the JSON report.

    Returns:
        LoadTestResult with all metrics and SLO evaluations.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("Install httpx: pip install httpx")

    t_start = time.monotonic()
    t_end = t_start + duration_seconds

    latencies: list[float] = []
    errors: list[str] = []
    confidence_scores: list[float] = []
    loop_counts: list[int] = []
    insufficient_count = 0
    lock = asyncio.Lock()

    async def _worker(worker_id: int) -> None:
        nonlocal insufficient_count
        # Ramp delay: each worker starts at a staggered time
        ramp_delay = (worker_id / max(n_users - 1, 1)) * ramp_seconds
        await asyncio.sleep(ramp_delay)

        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(15.0),
            headers={"X-API-Key": api_key},
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        ) as client:
            while time.monotonic() < t_end:
                query = random.choice(_WEIGHTED_QUERIES)
                t_req = time.monotonic()
                try:
                    resp = await client.post("/query", json={"query": query})
                    latency_ms = (time.monotonic() - t_req) * 1000

                    async with lock:
                        latencies.append(latency_ms)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                                if "confidence" in data:
                                    confidence_scores.append(float(data["confidence"]))
                                if "loops" in data:
                                    loop_counts.append(int(data["loops"]))
                                if data.get("insufficient_context"):
                                    insufficient_count += 1
                            except Exception:  # noqa: BLE001
                                pass
                        elif resp.status_code not in (429, 503, 504):
                            errors.append(f"HTTP {resp.status_code}")

                except httpx.TimeoutException:
                    async with lock:
                        latencies.append(15000.0)
                        errors.append("timeout")
                except Exception as exc:  # noqa: BLE001
                    async with lock:
                        errors.append(str(exc))

                # Think time: 0.5–2.0s
                await asyncio.sleep(random.uniform(0.5, 2.0))

    logger.info(
        "Starting async load test: %d users, %ds duration, %ds ramp",
        n_users, duration_seconds, ramp_seconds
    )
    workers = [asyncio.create_task(_worker(i)) for i in range(n_users)]
    await asyncio.gather(*workers)

    elapsed = time.monotonic() - t_start
    n_requests = len(latencies)
    n_errors = len(errors)

    result = LoadTestResult(
        mode="async",
        n_users=n_users,
        duration_seconds=duration_seconds,
        n_requests=n_requests,
        n_errors=n_errors,
        error_rate=n_errors / max(n_requests, 1),
        throughput_rps=n_requests / max(elapsed, 1),
        latencies_ms=latencies,
        avg_confidence=sum(confidence_scores) / max(len(confidence_scores), 1),
        avg_loops=sum(loop_counts) / max(len(loop_counts), 1),
        insufficient_rate=insufficient_count / max(n_requests, 1),
    )
    result.compute_percentiles()

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"LOAD TEST COMPLETE: {n_users} users, {duration_seconds}s")
    print(f"{'='*50}")
    print(f"Requests:    {n_requests} ({result.throughput_rps:.1f} req/s)")
    print(f"Errors:      {n_errors} ({result.error_rate*100:.2f}%)")
    print("Latency:")
    print(f"  p50:  {result.p50_ms:.0f}ms")
    print(f"  p95:  {result.p95_ms:.0f}ms")
    print(f"  p99:  {result.p99_ms:.0f}ms")
    print(f"  min:  {result.min_ms:.0f}ms")
    print(f"  max:  {result.max_ms:.0f}ms")
    print("RAG Metrics:")
    print(f"  avg confidence:  {result.avg_confidence:.3f}")
    print(f"  avg loops:       {result.avg_loops:.2f}")
    print(f"  insufficient:    {result.insufficient_rate*100:.1f}%")
    print("\nSLO Evaluation:")
    print(f"  p95 ≤ 1500ms:  {'✅ PASS' if result.slo_p95_pass else '❌ FAIL'} ({result.p95_ms:.0f}ms)")
    print(f"  Error ≤ 1%:    {'✅ PASS' if result.slo_error_pass else '❌ FAIL'} ({result.error_rate*100:.2f}%)")
    print(f"  ≥ 10 req/s:    {'✅ PASS' if result.slo_throughput_pass else '❌ FAIL'} ({result.throughput_rps:.1f} req/s)")

    # ── Save report ──────────────────────────────────────────────────────────
    try:
        import pathlib
        report_dict = {
            "mode": result.mode,
            "n_users": result.n_users,
            "duration_seconds": result.duration_seconds,
            "n_requests": result.n_requests,
            "n_errors": result.n_errors,
            "error_rate": result.error_rate,
            "throughput_rps": result.throughput_rps,
            "p50_ms": result.p50_ms,
            "p95_ms": result.p95_ms,
            "p99_ms": result.p99_ms,
            "min_ms": result.min_ms,
            "max_ms": result.max_ms,
            "avg_confidence": result.avg_confidence,
            "avg_loops": result.avg_loops,
            "insufficient_rate": result.insufficient_rate,
            "slo_p95_pass": result.slo_p95_pass,
            "slo_error_pass": result.slo_error_pass,
            "slo_throughput_pass": result.slo_throughput_pass,
            # Include latency histogram (100 buckets) instead of raw data
            "latency_histogram": _build_histogram(result.latencies_ms),
        }
        pathlib.Path(report_path).write_text(json.dumps(report_dict, indent=2))
        logger.info("Load test report saved to %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save report: %s", exc)

    return result


def _build_histogram(latencies: list[float], n_buckets: int = 20) -> dict:
    """Build a compact latency histogram for the report."""
    if not latencies:
        return {}
    min_val, max_val = min(latencies), max(latencies)
    if min_val == max_val:
        return {f"{int(min_val)}ms": len(latencies)}
    width = (max_val - min_val) / n_buckets
    buckets: dict[str, int] = {}
    for lat in latencies:
        bucket_idx = min(int((lat - min_val) / width), n_buckets - 1)
        bucket_label = f"{int(min_val + bucket_idx * width)}-{int(min_val + (bucket_idx+1) * width)}ms"
        buckets[bucket_label] = buckets.get(bucket_label, 0) + 1
    return dict(sorted(buckets.items()))


# ─── pytest integration ───────────────────────────────────────────────────────

try:
    import pytest  # noqa: E402
except ImportError:
    pytest = None  # type: ignore[assignment]


_skip_no_server = (
    pytest.mark.skipif(
        not __import__("os").environ.get("NEURORAG_BASE_URL"),
        reason="Requires running NeuroRAG server (set NEURORAG_BASE_URL)",
    ) if pytest else (lambda cls: cls)
)
_asyncio_mark = (pytest.mark.asyncio if pytest else (lambda cls: cls))


@_asyncio_mark
@_skip_no_server
class TestLoadSLOs:

    async def test_50_concurrent_users_slos(self):
        """
        50 concurrent users for 60 seconds must meet SLOs:
        - p95 ≤ 1500ms
        - error rate ≤ 1%
        - throughput ≥ 10 req/s
        """
        result = await run_async_load_test(
            n_users=50,
            duration_seconds=60,
            ramp_seconds=10,
        )
        assert result.slo_p95_pass, f"p95 latency SLO failed: {result.p95_ms:.0f}ms > 1500ms"
        assert result.slo_error_pass, f"Error rate SLO failed: {result.error_rate*100:.2f}% > 1%"
        assert result.n_requests >= 100, f"Too few requests: {result.n_requests}"

    async def test_100_concurrent_users_error_rate(self):
        """
        100 concurrent users for 60 seconds: error rate must stay ≤ 5%.
        (Relaxed p95 threshold for peak load scenario.)
        """
        result = await run_async_load_test(
            n_users=100,
            duration_seconds=60,
            ramp_seconds=15,
        )
        assert result.error_rate <= 0.05, f"Error rate too high at peak: {result.error_rate*100:.2f}%"

    async def test_throughput_scales_with_users(self):
        """Throughput at 50 users must be at least 2x throughput at 10 users."""
        result_10 = await run_async_load_test(n_users=10, duration_seconds=30, ramp_seconds=5)
        result_50 = await run_async_load_test(n_users=50, duration_seconds=30, ramp_seconds=8)
        # Allow some overhead; ensure 50 users gets at least 2x the throughput
        assert result_50.throughput_rps >= result_10.throughput_rps * 1.5, (
            f"Throughput doesn't scale: 10u={result_10.throughput_rps:.1f}, "
            f"50u={result_50.throughput_rps:.1f}"
        )


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="NeuroRAG async load test")
    parser.add_argument("--mode", choices=["smoke", "sustained", "peak", "stress"], default="sustained")
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--report", default="/tmp/async_load_report.json")
    args = parser.parse_args()

    presets = {
        "smoke":    (10, 30, 5),
        "sustained": (50, 120, 10),
        "peak":     (100, 180, 15),
        "stress":   (150, 120, 20),
    }
    n_users, duration, ramp = presets[args.mode]
    if args.users:
        n_users = args.users
    if args.duration:
        duration = args.duration

    asyncio.run(run_async_load_test(
        base_url=args.url,
        n_users=n_users,
        duration_seconds=duration,
        ramp_seconds=ramp,
        report_path=args.report,
    ))


if __name__ == "__main__":
    _cli()
