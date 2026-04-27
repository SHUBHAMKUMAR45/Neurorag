"""
NeuroRAG — FastAPI Production Server v3.2
Windows-compatible:
- no uvloop required
- Redis graceful degradation
- PostgreSQL asyncpg + sync fallback fixed
- dotenv loading added
- /env-check endpoint
- clean lifespan logging
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
import uuid
import os
import re as _re

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()   # <-- IMPORTANT loads .env automatically

import structlog
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# ─────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

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
    documents: list[IngestDocumentItem]


class IngestResponse(BaseModel):
    chunks_indexed: int
    doc_count: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    faiss_vectors: int
    redis_connected: bool
    postgres_connected: bool


# ─────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────

_engine = None
_orchestrator = None
_evaluator = None

_redis = None
_redis_ok = False
_postgres_ok = False

# ─────────────────────────────────────────────────────────────
# PII Filter
# ─────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    _re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    _re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]

def _filter_pii(text: str) -> str:
    for pat in _PII_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


# ─────────────────────────────────────────────────────────────
# Cache Helpers
# ─────────────────────────────────────────────────────────────

async def _cache_get(key: str):
    if _redis_ok and _redis:
        try:
            return await _redis.get(key)
        except:
            pass
    return None


async def _cache_set(key: str, value: str, ttl: int = 3600):
    if _redis_ok and _redis:
        try:
            await _redis.setex(key, ttl, value)
        except:
            pass


# ─────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _orchestrator, _evaluator
    global _redis, _redis_ok, _postgres_ok

    log.info("NeuroRAG API starting...", version="3.2")

    _engine = IngestionEngine()
    _orchestrator = NeuroRAGOrchestrator(_engine)
    _evaluator = Evaluator()

    # Redis
    try:
        import redis.asyncio as aioredis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis = aioredis.from_url(redis_url, decode_responses=True)

        await _redis.ping()
        _redis_ok = True
        log.info("Redis connected", url=redis_url)

    except Exception as exc:
        _redis_ok = False
        log.warning("Redis unavailable", error=str(exc))

    # PostgreSQL
    try:
        pool = await _evaluator._get_pool()
        _postgres_ok = pool is not None

        if _postgres_ok:
            log.info("Postgres connected")
        else:
            log.warning("Postgres unavailable")

    except Exception as exc:
        _postgres_ok = False
        log.warning("Postgres unavailable", error=str(exc))

    # Metrics
    try:
        start_metrics_server(port=_cfg.monitoring.prometheus_port)
    except:
        pass

    FAISS_INDEX_SIZE.set(len(_engine.doc_ids))

    log.info(
        "API Ready",
        redis=_redis_ok,
        postgres=_postgres_ok,
        faiss_vectors=len(_engine.doc_ids),
    )

    yield

    if _engine:
        _engine.save()

    if _redis_ok and _redis:
        await _redis.aclose()

    log.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

tracer = setup_tracing("neurorag-api")

app = FastAPI(
    title="NeuroRAG API",
    version="3.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TimeoutMiddleware, timeout_seconds=120)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(TracingMiddleware, tracer=tracer)
app.add_middleware(AuthMiddleware)


# ─────────────────────────────────────────────────────────────
# Query
# ─────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):

    if not _orchestrator:
        raise HTTPException(503, "System not ready")

    query_text = _filter_pii(req.query.strip())
    request_id = req.request_id or str(uuid.uuid4())

    cache_key = f"query:{hash(query_text)}"

    cached = await _cache_get(cache_key)
    if cached:
        payload = _json.loads(cached)
        payload["request_id"] = request_id
        return QueryResponse(**payload)

    try:
        result = await asyncio.wait_for(
            _orchestrator.run(query_text),
            timeout=120
        )

    except asyncio.TimeoutError:
        raise HTTPException(504, "Timeout")

    except Exception as exc:
        raise HTTPException(500, str(exc))

    response = QueryResponse(
        request_id=request_id,
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        loops=result.loops,
        latency_ms=result.latency_ms,
        insufficient_context=result.insufficient_context,
        from_memory_cache=result.from_memory_cache,
        context_hint_applied=result.context_hint_applied,
    )

    await _cache_set(cache_key, response.model_dump_json())

    return response


# ─────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):

    docs = [
        {
            "id": d.id,
            "text": _filter_pii(d.text),
            "metadata": d.metadata
        }
        for d in req.documents
    ]

    loop = asyncio.get_running_loop()

    chunks = await loop.run_in_executor(None, _engine.ingest, docs)
    await loop.run_in_executor(None, _engine.save)

    return IngestResponse(
        chunks_indexed=chunks,
        doc_count=len(docs),
        status="ok"
    )


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="3.2",
        faiss_vectors=len(_engine.doc_ids),
        redis_connected=_redis_ok,
        postgres_connected=_postgres_ok,
    )


# ─────────────────────────────────────────────────────────────
# Env Check
# ─────────────────────────────────────────────────────────────

@app.get("/env-check")
async def env_check():
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "REDIS_URL": bool(os.getenv("REDIS_URL")),
        "POSTGRES_URL": bool(os.getenv("POSTGRES_URL")),
        "redis_connected": _redis_ok,
        "postgres_connected": _postgres_ok,
    }


# ─────────────────────────────────────────────────────────────
# Static UI
# ─────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent / "static"

if STATIC_DIR.exists():

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )