"""
NeuroRAG — API Middleware (Windows-compatible)
Removed: uvloop dependency, Jaeger hard-import
Added: graceful OpenTelemetry degradation
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Callable, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ─── OpenTelemetry (optional) ─────────────────────────────────────────────────
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
    if not _OTEL_AVAILABLE:
        logger.info("OpenTelemetry not installed; tracing disabled.")
        return None
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name, "service.version": "3.0.0"})
        )
        # Use console exporter — no Jaeger needed locally
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return trace.get_tracer(service_name)
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed: %s", exc)
        return None


# ─── API Key Store ────────────────────────────────────────────────────────────

class APIKeyStore:
    _MASTER_KEY_ENV = "NEURORAG_API_KEY"

    def __init__(self) -> None:
        master = os.environ.get(self._MASTER_KEY_ENV, "")
        self._valid_hashes: set[str] = set()
        if master:
            self._valid_hashes.add(self._hash_key(master))
            logger.info("APIKeyStore: loaded API key from environment.")
        else:
            logger.warning(
                "APIKeyStore: %s not set — API key auth BYPASSED (dev mode).",
                self._MASTER_KEY_ENV,
            )

    def is_valid(self, key: str) -> bool:
        if not self._valid_hashes:
            return True  # Dev mode
        return self._hash_key(key) in self._valid_hashes

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


_api_key_store = APIKeyStore()


# ─── Auth Middleware ──────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    _BYPASS_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc",
                     "/env-check", "/", "/ui", "/static"}

    async def dispatch(self, request: Request, call_next: Callable):
        # Bypass auth for static UI files
        if (request.url.path in self._BYPASS_PATHS
                or request.url.path.startswith("/static")):
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
    def __init__(self, app: ASGIApp, tracer: Optional[object] = None) -> None:
        super().__init__(app)
        self._tracer = tracer

    async def dispatch(self, request: Request, call_next: Callable):
        if not self._tracer or not _OTEL_AVAILABLE:
            return await call_next(request)
        request_id = request.headers.get("X-Request-ID", "unknown")
        with self._tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            attributes={"http.method": request.method, "http.request_id": request_id},
        ) as span:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response


# ─── Request Logging Middleware ───────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        t_start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "HTTP %s %s → %d (%dms) ip=%s",
            request.method, request.url.path, response.status_code, elapsed_ms,
            request.client.host if request.client else "unknown",
        )
        return response


# ─── Timeout Middleware ───────────────────────────────────────────────────────

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout_seconds: int = 120) -> None:
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
