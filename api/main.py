"""
NeuroRAG — FastAPI Production Server v3
Adds: SecurityConfig wired into settings, /env-check endpoint,
      cleaner startup logging, consistent critic_result passing to Evaluator.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional
import re as _re

import redis.asyncio as aioredis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.orchestrator import NeuroRAGOrchestrator
from api.middleware import (
    AuthMiddleware,
    RequestLoggingMiddleware,
    TimeoutMiddleware,
    TracingMiddleware,
    setup_tracing,
)
from configs.settings import get_config
from dashboard.metrics import (
    CONFIDENCE_SCORE,
    DOCS_RETRIEVED,
    FAISS_INDEX_SIZE,
    HALLUCINATION_TOTAL,
    INGEST_CHUNKS_TOTAL,
    INGEST_DOCS_TOTAL,
    QUERY_LATENCY,
    QUERY_LOOPS,
    QUERY_TOTAL,
    start_metrics_server,
)
from evaluation.evaluator import Evaluator
from rag.ingest import IngestionEngine

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log = structlog.get_logger()
logging.basicConfig(level=logging.INFO)

_cfg = get_config()

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    request_id: Optional[str] = None


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[str]
    confidence: float
    loops: int
    latency_ms: int
    insufficient_context: bool
    from_memory_cache: bool = False
    context_hint_applied: bool = False


class IngestDocumentItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocumentItem] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    chunks_indexed: int
    doc_count: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    faiss_vectors: int


# ─── App State ────────────────────────────────────────────────────────────────

_engine: Optional[IngestionEngine] = None
_orchestrator: Optional[NeuroRAGOrchestrator] = None
_evaluator: Optional[Evaluator] = None
_redis: Optional[aioredis.Redis] = None

# ─── PII Filter ───────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    _re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    _re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    _re.compile(r"\b(?:\+1\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
]


def _filter_pii(text: str) -> str:
    for pat in _PII_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# ─── Cache Helpers ────────────────────────────────────────────────────────────

async def _cache_get(key: str) -> Optional[str]:
    if _redis:
        try:
            return await _redis.get(key)
        except Exception:
            pass
    return None


async def _cache_set(key: str, value: str, ttl: int = 3600) -> None:
    if _redis:
        try:
            await _redis.setex(key, ttl, value)
        except Exception:
            pass


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _orchestrator, _evaluator, _redis
    log.info("NeuroRAG API starting…", version="3.0.0")

    _engine = IngestionEngine()
    _orchestrator = NeuroRAGOrchestrator(_engine)
    _evaluator = Evaluator()

    try:
        _redis = aioredis.from_url(_cfg.cache.redis_url, decode_responses=True)
        await _redis.ping()
        log.info("Redis connected.")
    except Exception as exc:
        log.warning("Redis unavailable — cache disabled.", error=str(exc))
        _redis = None

    start_metrics_server(port=_cfg.monitoring.prometheus_port)
    FAISS_INDEX_SIZE.set(len(_engine.doc_ids))

    log.info(
        "NeuroRAG API ready.",
        faiss_vectors=len(_engine.doc_ids),
        redis=_redis is not None,
        prometheus_port=_cfg.monitoring.prometheus_port,
    )

    yield

    log.info("Shutting down…")
    if _engine:
        _engine.save()
    if _redis:
        await _redis.aclose()


# ─── App + Middleware ─────────────────────────────────────────────────────────

tracer = setup_tracing("neurorag-api")

app = FastAPI(
    title="NeuroRAG API",
    description="Autonomous Self-Healing Multi-Agent RAG System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.api.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(TimeoutMiddleware, timeout_seconds=_cfg.api.timeout)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TracingMiddleware, tracer=tracer)
app.add_middleware(AuthMiddleware)


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _redis and request.url.path == "/query":
        ip = request.client.host or "unknown"
        key = f"rl:{ip}"
        try:
            count = await _redis.incr(key)
            if count == 1:
                await _redis.expire(key, 60)
            if count > _cfg.api.rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                )
        except Exception:
            pass
    return await call_next(request)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, status_code=200)
async def handle_query(req: QueryRequest):
    if not _orchestrator:
        raise HTTPException(503, "System not initialised.")

    query = _filter_pii(req.query.strip())
    request_id = req.request_id or str(uuid.uuid4())

    # Fast-path: Redis response cache
    cache_key = f"query:{hash(query)}"
    cached = await _cache_get(cache_key)
    if cached:
        payload = _json.loads(cached)
        payload["request_id"] = request_id
        return QueryResponse(**payload)

    log.info("Processing query", request_id=request_id, query=query[:80])
    t0 = time.monotonic()

    try:
        result = await asyncio.wait_for(
            _orchestrator.run(query), timeout=_cfg.api.timeout
        )
    except asyncio.TimeoutError:
        QUERY_TOTAL.labels(status="error").inc()
        raise HTTPException(504, "Query timed out.")
    except Exception as exc:
        log.error("Pipeline error", error=str(exc))
        QUERY_TOTAL.labels(status="error").inc()
        raise HTTPException(500, "Internal pipeline error.")

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # Prometheus metrics
    QUERY_LATENCY.observe(elapsed_ms)
    QUERY_LOOPS.observe(result.loops)
    CONFIDENCE_SCORE.observe(result.confidence)
    QUERY_TOTAL.labels(
        status="insufficient" if result.insufficient_context else "success"
    ).inc()
    if any(ft.value == "hallucination" for ft in result.failure_history):
        HALLUCINATION_TOTAL.inc()

    response = QueryResponse(
        request_id=result.request_id,
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        loops=result.loops,
        latency_ms=result.latency_ms,
        insufficient_context=result.insufficient_context,
        from_memory_cache=result.from_memory_cache,
        context_hint_applied=result.context_hint_applied,
    )

    # Cache high-confidence responses
    if result.confidence >= _cfg.self_heal.confidence_threshold:
        await _cache_set(
            cache_key,
            response.model_dump_json(),
            ttl=_cfg.cache.ttl_seconds,
        )

    log.info(
        "Query done",
        request_id=request_id,
        conf=result.confidence,
        loops=result.loops,
        ms=elapsed_ms,
    )
    return response


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def handle_ingest(req: IngestRequest):
    if not _engine:
        raise HTTPException(503, "System not initialised.")
    docs = [
        {"id": d.id, "text": _filter_pii(d.text), "metadata": d.metadata}
        for d in req.documents
    ]
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, _engine.ingest, docs)
    await loop.run_in_executor(None, _engine.save)
    INGEST_DOCS_TOTAL.inc(len(docs))
    INGEST_CHUNKS_TOTAL.inc(chunks)
    FAISS_INDEX_SIZE.set(len(_engine.doc_ids))
    log.info("Ingest done", docs=len(docs), chunks=chunks)
    return IngestResponse(chunks_indexed=chunks, doc_count=len(docs), status="ok")


@app.get("/health", response_model=HealthResponse)
async def health():
    if not _engine or not _orchestrator:
        raise HTTPException(503, "Not ready.")
    return HealthResponse(
        status="ok",
        version="3.0.0",
        faiss_vectors=len(_engine.doc_ids),
    )


@app.get("/stats")
async def stats(hours: int = 24):
    if not _evaluator:
        raise HTTPException(503, "Evaluator not ready.")
    return await _evaluator.get_stats(window_hours=hours)


@app.get("/memory/stats")
async def memory_stats():
    """Return Redis memory store stats."""
    if not _redis:
        return {"status": "redis_unavailable"}
    try:
        info = await _redis.info("memory")
        keys = await _redis.dbsize()
        return {
            "redis_used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            "total_keys": keys,
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/circuit-breaker/status")
async def circuit_breaker_status():
    """Expose circuit breaker states for all LLM clients."""
    if not _orchestrator:
        raise HTTPException(503, "Not ready.")
    try:
        gen_cb    = _orchestrator._generator._llm
        critic_cb = _orchestrator._critic._llm
        return {
            "generator": str(getattr(gen_cb,    "state", "unknown")),
            "critic":    str(getattr(critic_cb, "state", "unknown")),
        }
    except AttributeError:
        return {"status": "circuit_breaker_not_wrapped"}


@app.get("/env-check")
async def env_check():
    """
    Lightweight config sanity endpoint (unauthenticated).
    Returns which critical env vars are set (not their values).
    """
    import os
    return {
        "OPENAI_API_KEY":    bool(os.environ.get("OPENAI_API_KEY")),
        "NEURORAG_API_KEY":  bool(os.environ.get("NEURORAG_API_KEY")),
        "POSTGRES_URL":      bool(os.environ.get("POSTGRES_URL")),
        "REDIS_URL":         bool(os.environ.get("REDIS_URL")),
        "LLAMA_MODEL_PATH":  bool(os.environ.get("LLAMA_MODEL_PATH")),
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=_cfg.api.host,
        port=_cfg.api.port,
        workers=_cfg.api.workers,
        log_level="info",
        loop="uvloop",
    )
