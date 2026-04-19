"""
NeuroRAG — Circuit Breaker for LLM Clients
State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
Wraps any BaseLLMClient with failure tracking, exponential backoff,
and Prometheus state metrics.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from agents.schemas import CircuitState
from rag.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge
    CB_STATE = Gauge("neurorag_circuit_breaker_state",
                     "Circuit state (0=closed,1=open,2=half_open)", ["client"])
    CB_TRANSITIONS = Counter("neurorag_circuit_breaker_transitions_total",
                              "State transitions", ["client", "to_state"])
    CB_BLOCKED = Counter("neurorag_circuit_breaker_blocked_total",
                         "Calls blocked by open circuit", ["client"])
    _PROM = True
except ImportError:
    _PROM = False


class CircuitBreakerOpen(RuntimeError):
    """Raised when a call is attempted against an OPEN circuit."""


class CircuitBreakerLLMClient(BaseLLMClient):
    """
    LLM client wrapper with full circuit breaker + retry logic.
    Both complete() and acomplete() are protected.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        cooldown_seconds: float = 30.0,
        max_retries: int = 3,
        base_backoff_s: float = 0.5,
        client_name: str = "llm",
    ) -> None:
        self._client = client
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._cooldown = cooldown_seconds
        self._max_retries = max_retries
        self._base_backoff = base_backoff_s
        self._name = client_name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # ── Sync interface ───────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Synchronous complete() with circuit breaker.
        Delegates to acomplete() via a fresh event loop when called from
        a non-async context; called from async context via acomplete().
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — use thread pool to avoid deadlock
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, self.acomplete(prompt, system, temperature, max_tokens))
                return future.result(timeout=120)
        except RuntimeError:
            # No running event loop — safe to create one
            return asyncio.run(self.acomplete(prompt, system, temperature, max_tokens))

    # ── Async interface ──────────────────────────────────────────────────────

    async def acomplete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Async complete() with circuit breaker + exponential backoff retry."""
        async with self._lock:
            await self._maybe_transition_half_open()
            if self._state == CircuitState.OPEN:
                if _PROM:
                    CB_BLOCKED.labels(client=self._name).inc()
                remaining = self._cooldown - (time.monotonic() - self._last_failure_time)
                raise CircuitBreakerOpen(
                    f"Circuit OPEN for '{self._name}' — cooldown {remaining:.1f}s remaining"
                )

        last_exc: Exception = RuntimeError("No attempts")
        for attempt in range(self._max_retries):
            try:
                result = await self._client.acomplete(prompt, system, temperature, max_tokens)
                await self._on_success()
                return result
            except CircuitBreakerOpen:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = self._base_backoff * (2 ** attempt) + random.uniform(0, 0.3)
                logger.warning(
                    "CB '%s' attempt %d/%d failed: %s — retry in %.2fs",
                    self._name, attempt + 1, self._max_retries, exc, wait,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(wait)

        await self._on_failure()
        raise last_exc

    # ── State transitions ────────────────────────────────────────────────────

    async def _maybe_transition_half_open(self) -> None:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._cooldown:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("CB '%s': OPEN → HALF_OPEN", self._name)
                self._emit(CircuitState.HALF_OPEN)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("CB '%s': HALF_OPEN → CLOSED", self._name)
                    self._emit(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold and self._state != CircuitState.OPEN:
                logger.error(
                    "CB '%s': %s → OPEN (failures=%d)",
                    self._name, self._state.value, self._failure_count,
                )
                self._state = CircuitState.OPEN
                self._emit(CircuitState.OPEN)

    def _emit(self, state: CircuitState) -> None:
        if not _PROM:
            return
        m = {CircuitState.CLOSED: 0, CircuitState.OPEN: 1, CircuitState.HALF_OPEN: 2}
        CB_STATE.labels(client=self._name).set(m[state])
        CB_TRANSITIONS.labels(client=self._name, to_state=state.value).inc()

    @property
    def state(self) -> CircuitState:
        return self._state


def wrap_with_circuit_breaker(
    client: BaseLLMClient,
    name: str = "llm",
    failure_threshold: int = 5,
    cooldown_seconds: float = 30.0,
) -> CircuitBreakerLLMClient:
    return CircuitBreakerLLMClient(
        client=client,
        failure_threshold=failure_threshold,
        cooldown_seconds=cooldown_seconds,
        client_name=name,
    )
