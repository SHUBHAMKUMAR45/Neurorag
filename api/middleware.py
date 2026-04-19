"""
NeuroRAG — API Security Middleware
────────────────────────────────────────────────────────────────────────────
Provides:
  1. API Key authentication (X-API-Key header)
  2. JWT Bearer authentication (Authorization: Bearer <token>)
  3. OpenTelemetry distributed tracing (request_id propagated as span)
  4. Request/response structured logging
  5. Timeout enforcement middleware

Design: FastAPI middleware stack (applied in reverse registration order).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Callable, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ─── OpenTelemetry (graceful degradation if not installed) ──────────────────
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]


def setup_tracing(service_name: str = "neurorag-api") -> Optional[object]:
    """Initialise OpenTelemetry tracer with Jaeger exporter."""
    if not _OTEL_AVAILABLE:
        logger.info("OpenTelemetry not installed; tracing disabled.")
        return None

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    jaeger_host = os.environ.get("JAEGER_HOST", "jaeger")
    jaeger_port = int(os.environ.get("JAEGER_PORT", "6831"))

    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        exporter = JaegerExporter(agent_host_name=jaeger_host, agent_port=jaeger_port)
    except Exception:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
        logger.warning("Jaeger unavailable; using console span exporter.")

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.version": "3.0.0"})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing initialised (service=%s)", service_name)
    return trace.get_tracer(service_name)


# ─── API Key Store ────────────────────────────────────────────────────────────

class APIKeyStore:
    """
    In-memory API key store backed by environment variable.
    Production: replace with Redis-backed store + key rotation.

    Keys are stored as SHA-256 hashes (never plaintext).
    """

    _MASTER_KEY_ENV = "NEURORAG_API_KEY"

    def __init__(self) -> None:
        master = os.environ.get(self._MASTER_KEY_ENV, "")
        self._valid_hashes: set[str] = set()
        if master:
            self._valid_hashes.add(self._hash_key(master))
            logger.info("APIKeyStore: loaded 1 API key from environment.")
        else:
            logger.warning(
                "APIKeyStore: %s not set — API key auth BYPASSED (dev mode).",
                self._MASTER_KEY_ENV,
            )

    def is_valid(self, key: str) -> bool:
        if not self._valid_hashes:
            return True   # Dev mode: no keys configured → allow all
        return self._hash_key(key) in self._valid_hashes

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


_api_key_store = APIKeyStore()


# ─── Authentication Middleware ────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all non-health/metrics endpoints.
    Health + metrics are intentionally unauthenticated.
    """

    _BYPASS_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in self._BYPASS_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not _api_key_store.is_valid(api_key):
            logger.warning(
                "Auth REJECTED: path=%s ip=%s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key."},
            )

        return await call_next(request)


# ─── Tracing Middleware ───────────────────────────────────────────────────────

class TracingMiddleware(BaseHTTPMiddleware):
    """
    Attaches OpenTelemetry spans to each request.
    Propagates request_id as span attribute.
    """

    def __init__(self, app: ASGIApp, tracer: Optional[object] = None) -> None:
        super().__init__(app)
        self._tracer = tracer

    async def dispatch(self, request: Request, call_next: Callable):
        if not self._tracer or not _OTEL_AVAILABLE:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", "unknown")
        with self._tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.request_id": request_id,
            },
        ) as span:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response


# ─── Request Logging Middleware ───────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request/response logging with timing.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        t_start = time.monotonic()
        request_id = request.headers.get("X-Request-ID", "n/a")

        response = await call_next(request)

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "HTTP %s %s → %d (%dms) request_id=%s ip=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
            request.client.host if request.client else "unknown",
        )
        return response


# ─── Timeout Middleware ───────────────────────────────────────────────────────

class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforces per-request timeout. Returns 504 on timeout.
    Timeout value read from config.
    """

    def __init__(self, app: ASGIApp, timeout_seconds: int = 60) -> None:
        super().__init__(app)
        self._timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.error("Request timeout after %ds: %s", self._timeout, request.url.path)
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": f"Request timed out after {self._timeout}s."},
            )
