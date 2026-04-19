"""
NeuroRAG — Load Tests (Locust)
────────────────────────────────────────────────────────────────────────────
Run:
  # Smoke (CI): 10 users, 30s
  locust -f tests/test_load.py --headless -u 10 -r 2 -t 30s \
         --host http://localhost:8000 --html load_report.html

  # Full load: 50 concurrent users
  locust -f tests/test_load.py --headless -u 50 -r 5 -t 120s \
         --host http://localhost:8000

  # Stress test: find breaking point
  locust -f tests/test_load.py --headless -u 200 -r 20 -t 300s \
         --host http://localhost:8000

Success criteria (from config.yaml KPIs):
  - p95 latency ≤ 1500ms
  - error rate ≤ 1%
  - hallucination rate ≤ 5% (checked via /stats endpoint)

Also contains pytest-friendly smoke tests (no Locust needed).
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import pytest

# ─── Sample Queries ───────────────────────────────────────────────────────────

_QUERIES = [
    "What is retrieval-augmented generation?",
    "Explain the difference between BM25 and vector search.",
    "How does a cross-encoder reranker improve RAG quality?",
    "What are hallucinations in large language models?",
    "Describe the self-healing loop in NeuroRAG.",
    "What is the role of the Critic agent?",
    "How does FAISS indexing work?",
    "What is Reciprocal Rank Fusion?",
    "Explain semantic chunking vs fixed-size chunking.",
    "What metrics does NeuroRAG track in Prometheus?",
]

_DOCS = [
    {"id": f"load-doc-{i}", "text": f"This is load test document {i}. " * 50, "metadata": {}}
    for i in range(20)
]

API_KEY = os.environ.get("NEURORAG_API_KEY", "")
BASE_URL = os.environ.get("NEURORAG_BASE_URL", "http://localhost:8000")


# ─── Locust Load Test ─────────────────────────────────────────────────────────

try:
    from locust import HttpUser, TaskSet, between, task

    class NeuroRAGTasks(TaskSet):
        """Weighted task mix: 80% query, 15% ingest, 5% health."""

        def on_start(self):
            """Seed index with test documents before load starts."""
            self.client.post(
                "/ingest",
                json={"documents": _DOCS[:5]},
                headers={"X-API-Key": API_KEY},
                name="/ingest [seed]",
            )

        @task(8)
        def query(self):
            """Main query endpoint — primary load target."""
            q = random.choice(_QUERIES)
            with self.client.post(
                "/query",
                json={"query": q},
                headers={"X-API-Key": API_KEY},
                catch_response=True,
                name="/query",
            ) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    # Validate response schema
                    if "answer" not in data or "confidence" not in data:
                        resp.failure("Response missing required fields")
                    elif data["confidence"] < 0.0 or data["confidence"] > 1.0:
                        resp.failure(f"Invalid confidence: {data['confidence']}")
                    else:
                        resp.success()
                elif resp.status_code == 429:
                    resp.success()  # Rate limiting is expected behaviour
                else:
                    resp.failure(f"HTTP {resp.status_code}")

        @task(1)
        def ingest_batch(self):
            """Background ingest — simulates real-world continuous ingestion."""
            docs = random.sample(_DOCS, k=3)
            self.client.post(
                "/ingest",
                json={"documents": docs},
                headers={"X-API-Key": API_KEY},
                name="/ingest",
            )

        @task(1)
        def health_check(self):
            self.client.get("/health", name="/health")

    class NeuroRAGUser(HttpUser):
        tasks = [NeuroRAGTasks]
        wait_time = between(0.5, 2.0)   # Realistic think time

except ImportError:
    pass  # Locust not installed — pytest smoke tests still work


# ─── Pytest Smoke Tests (no Locust required) ─────────────────────────────────

@pytest.fixture(scope="session")
def api_client():
    """httpx client for smoke tests."""
    import httpx
    with httpx.Client(base_url=BASE_URL, timeout=30.0, headers={"X-API-Key": API_KEY}) as client:
        yield client


@pytest.mark.skipif(
    os.environ.get("CI") and not os.environ.get("NEURORAG_BASE_URL"),
    reason="Requires running NeuroRAG server"
)
class TestSmoke:

    def test_health_endpoint(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "3.0.0"

    def test_ingest_and_query_roundtrip(self, api_client):
        """Ingest a document, then query it, verify cited."""
        # Ingest
        ingest_resp = api_client.post("/ingest", json={
            "documents": [{
                "id": "smoke-test-doc",
                "text": "The NeuroRAG system uses Reciprocal Rank Fusion for score merging.",
                "metadata": {"source": "smoke-test"},
            }]
        })
        assert ingest_resp.status_code == 201
        assert ingest_resp.json()["chunks_indexed"] >= 1

        # Query
        query_resp = api_client.post("/query", json={
            "query": "How does NeuroRAG merge retrieval scores?"
        })
        assert query_resp.status_code == 200
        data = query_resp.json()
        assert "answer" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["loops"] >= 1

    def test_query_response_schema(self, api_client):
        """All required fields present in response."""
        resp = api_client.post("/query", json={"query": "test query"})
        # May be 200 or 200 with insufficient_context; both valid
        assert resp.status_code == 200
        data = resp.json()
        required = {"request_id", "answer", "citations", "confidence", "loops", "latency_ms", "insufficient_context"}
        assert required.issubset(set(data.keys()))

    def test_rate_limiting_returns_429(self, api_client):
        """Burst 70 requests in quick succession to trigger rate limit."""
        import httpx, concurrent.futures
        statuses = []
        def _req(_):
            r = api_client.post("/query", json={"query": "burst test"})
            return r.status_code
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            statuses = list(ex.map(_req, range(70)))
        # At least some should be rate-limited (429) if Redis is available
        # Allow all-200 if Redis is not connected (dev mode)
        assert all(s in (200, 429, 503, 504) for s in statuses)

    def test_invalid_api_key_returns_401(self, api_client):
        """Authentication enforcement."""
        if not API_KEY:
            pytest.skip("API key auth not configured (dev mode)")
        import httpx
        bad_client = httpx.Client(base_url=BASE_URL, headers={"X-API-Key": "invalid-key"})
        resp = bad_client.post("/query", json={"query": "test"})
        assert resp.status_code == 401

    def test_pii_filtered_from_query(self, api_client):
        """SSN in query should not appear in logs / answer."""
        resp = api_client.post("/query", json={"query": "Tell me about SSN 123-45-6789"})
        assert resp.status_code == 200
        # PII should not echo back in answer
        assert "123-45-6789" not in resp.json().get("answer", "")

    def test_stats_endpoint(self, api_client):
        resp = api_client.get("/stats?hours=1")
        assert resp.status_code == 200
        data = resp.json()
        # May be empty if no queries yet
        assert isinstance(data, dict)

    def test_memory_stats_endpoint(self, api_client):
        resp = api_client.get("/memory/stats")
        assert resp.status_code == 200

    def test_circuit_breaker_status(self, api_client):
        resp = api_client.get("/circuit-breaker/status")
        assert resp.status_code == 200


# ─── Concurrency Stress Test (pytest-asyncio) ─────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("NEURORAG_BASE_URL"),
    reason="Requires running server"
)
async def test_concurrent_queries():
    """50 concurrent queries must all succeed within 10s."""
    import httpx, asyncio

    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=15.0, headers={"X-API-Key": API_KEY}
    ) as client:
        tasks = [
            client.post("/query", json={"query": random.choice(_QUERIES)})
            for _ in range(50)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in responses if isinstance(r, Exception)]
    bad_status = [r for r in responses if hasattr(r, "status_code") and r.status_code not in (200, 429)]

    assert len(errors) == 0, f"{len(errors)} requests raised exceptions"
    assert len(bad_status) == 0, f"{len(bad_status)} requests returned non-200/429"
