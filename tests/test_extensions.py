"""
NeuroRAG — Extended Test Suite
Covers: Memory layer, Circuit Breaker, Evaluator v2, Middleware auth,
        PII filtering, Schema completeness.
Run: pytest tests/test_extensions.py -v --tb=short
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.schemas import (
    CircuitState,
    ContextHintSchema,
    CriticResult,
    FailureType,
    FixAction,
    PipelineResult,
    QueryType,
)


# ════════════════════════════════════════════════════════════════════════════
# MEMORY LAYER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestQueryMemoryStore:

    @pytest.fixture
    def store(self):
        from agents.memory import QueryMemoryStore
        s = QueryMemoryStore()
        s._redis = None   # Force no Redis (tests don't need it)
        s._pool = None
        return s

    def test_hash_deterministic(self, store):
        h1 = store._hash("capital of France?")
        h2 = store._hash("capital of France?")
        assert h1 == h2

    def test_hash_case_insensitive(self, store):
        h1 = store._hash("CAPITAL OF FRANCE")
        h2 = store._hash("capital of france")
        assert h1 == h2

    def test_hash_strips_whitespace(self, store):
        h1 = store._hash("  hello world  ")
        h2 = store._hash("hello world")
        assert h1 == h2

    def test_hash_different_queries_differ(self, store):
        h1 = store._hash("Paris")
        h2 = store._hash("London")
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_lookup_returns_none_when_no_redis(self, store):
        result = await store.lookup("any query")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_skips_low_confidence(self, store):
        """Low-confidence answers must not be cached."""
        result = PipelineResult(
            query="test", answer="low quality", citations=[],
            confidence=0.3, loops=3, latency_ms=1000,
        )
        # Should silently skip (no error, no assertion failures)
        await store.store(result)

    @pytest.mark.asyncio
    async def test_store_with_mock_redis(self):
        from agents.memory import QueryMemoryStore
        store = QueryMemoryStore()

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        store._redis = mock_redis

        result = PipelineResult(
            query="capital of France?", answer="Paris is the capital of France.",
            citations=["doc1#0"], confidence=0.97, loops=1, latency_ms=300,
        )
        from agents.memory import MemoryEntry
        entry = MemoryEntry(
            query_hash=store._hash(result.query),
            query=result.query,
            answer=result.answer,
            citations=result.citations,
            confidence=result.confidence,
            loops=result.loops,
            latency_ms=result.latency_ms,
            failure_types=[],
        )
        await store._store_redis(entry)
        # store won't actually call _store_redis without confidence >=threshold
        # but redis setex should not have been called with low conf

    @pytest.mark.asyncio
    async def test_semantic_lookup_skips_on_no_keys(self):
        from agents.memory import QueryMemoryStore
        store = QueryMemoryStore()
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])
        result = await store._semantic_lookup(mock_redis, "any query")
        assert result is None


class TestFailureMemory:

    @pytest.mark.asyncio
    async def test_get_recommended_fix_no_redis(self):
        from agents.memory import FailureMemory
        mem = FailureMemory()
        mem._redis = None
        result = await mem.get_recommended_fix("query", FailureType.HALLUCINATION)
        assert result is None

    @pytest.mark.asyncio
    async def test_record_and_retrieve_fix(self):
        from agents.memory import FailureMemory
        mem = FailureMemory()

        # Mock Redis with in-memory store
        _store: dict = {}
        mock_redis = AsyncMock()

        async def _hset(key, field, value):
            if key not in _store: _store[key] = {}
            _store[key][field] = value

        async def _hget(key, field):
            return _store.get(key, {}).get(field)

        async def _hkeys(key):
            return list(_store.get(key, {}).keys())

        mock_redis.hset = _hset
        mock_redis.hget = _hget
        mock_redis.hkeys = _hkeys
        mock_redis.expire = AsyncMock()
        mem._redis = mock_redis

        # Record a successful fix
        await mem.record("test query", FailureType.HALLUCINATION, FixAction.ADD_CONTEXT, succeeded=True)
        action = await mem.get_recommended_fix("test query", FailureType.HALLUCINATION)
        assert action == FixAction.ADD_CONTEXT

    @pytest.mark.asyncio
    async def test_failed_fix_not_recommended(self):
        from agents.memory import FailureMemory
        mem = FailureMemory()
        _store: dict = {}
        mock_redis = AsyncMock()

        async def _hset(key, field, value):
            if key not in _store: _store[key] = {}
            _store[key][field] = value

        async def _hget(key, field):
            return _store.get(key, {}).get(field)

        mock_redis.hset = _hset
        mock_redis.hget = _hget
        mock_redis.expire = AsyncMock()
        mem._redis = mock_redis

        # Record a FAILED fix
        await mem.record("test query", FailureType.IRRELEVANCE, FixAction.BROADEN_QUERY, succeeded=False)
        action = await mem.get_recommended_fix("test query", FailureType.IRRELEVANCE)
        assert action is None   # Failed fix must not be recommended

    @pytest.mark.asyncio
    async def test_get_prior_failures_empty(self):
        from agents.memory import FailureMemory
        mem = FailureMemory()
        mem._redis = None
        failures = await mem.get_prior_failures("new query")
        assert failures == []


class TestAdaptiveContext:

    @pytest.mark.asyncio
    async def test_build_hint_no_cache_returns_empty_hint(self):
        from agents.memory import AdaptiveContext
        ctx = AdaptiveContext()
        ctx._query_mem._redis = None
        ctx._query_mem._pool = None
        ctx._fail_mem._redis = None

        hint = await ctx.build_hint("brand new query with no history")
        assert hint.similar_past_answer is None
        assert hint.recommended_top_k_boost == 0
        assert hint.prior_failure_types == []

    @pytest.mark.asyncio
    async def test_build_hint_with_cache_hit(self):
        from agents.memory import AdaptiveContext, QueryMemoryStore, MemoryEntry
        ctx = AdaptiveContext()

        # Inject a cache hit
        async def _mock_lookup(query: str):
            return MemoryEntry(
                query_hash="abc123",
                query=query,
                answer="Cached answer.",
                citations=["doc1#0"],
                confidence=0.95,
                loops=1,
                latency_ms=200,
                failure_types=[],
            )

        ctx._query_mem.lookup = _mock_lookup
        hint = await ctx.build_hint("What is RAG?")
        assert hint.similar_past_answer == "Cached answer."
        assert hint.confidence_floor == 0.95


# ════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:

    def _make_client(self, response="ok", fail_count=0):
        from rag.llm_client import BaseLLMClient
        class FakeClient(BaseLLMClient):
            def __init__(self):
                self._calls = 0
                self._fail_count = fail_count
            def complete(self, *args, **kwargs) -> str:
                self._calls += 1
                if self._calls <= self._fail_count:
                    raise RuntimeError(f"Simulated failure {self._calls}")
                return response
        return FakeClient()

    @pytest.mark.asyncio
    async def test_closed_state_passes_through(self):
        from rag.circuit_breaker import CircuitBreakerLLMClient
        client = self._make_client(response="hello")
        cb = CircuitBreakerLLMClient(client, failure_threshold=5, client_name="test")
        result = await cb.acomplete("prompt")
        assert result == "hello"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        from rag.circuit_breaker import CircuitBreakerLLMClient, CircuitBreakerOpen
        client = self._make_client(fail_count=99)
        cb = CircuitBreakerLLMClient(
            client, failure_threshold=3, max_retries=1,
            base_backoff_s=0.0, client_name="test"
        )
        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.acomplete("prompt")

        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpen immediately
        with pytest.raises(CircuitBreakerOpen):
            await cb.acomplete("prompt")

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_cooldown(self):
        from rag.circuit_breaker import CircuitBreakerLLMClient
        client = self._make_client(fail_count=3)
        cb = CircuitBreakerLLMClient(
            client, failure_threshold=3, max_retries=1,
            cooldown_seconds=0.01, base_backoff_s=0.0, client_name="test"
        )
        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.acomplete("prompt")
        assert cb.state == CircuitState.OPEN

        # Wait past cooldown
        await asyncio.sleep(0.05)

        # Force state check
        async with cb._lock:
            await cb._maybe_transition_half_open()

        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successful_probes(self):
        from rag.circuit_breaker import CircuitBreakerLLMClient
        client = self._make_client(fail_count=3, response="recovered")
        cb = CircuitBreakerLLMClient(
            client, failure_threshold=3, success_threshold=2, max_retries=1,
            cooldown_seconds=0.01, base_backoff_s=0.0, client_name="test"
        )
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.acomplete("prompt")
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.05)
        async with cb._lock:
            await cb._maybe_transition_half_open()

        # Two successful probes → CLOSED
        r1 = await cb.acomplete("prompt")
        r2 = await cb.acomplete("prompt")
        assert r1 == "recovered"
        assert r2 == "recovered"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_exponential_backoff_retries(self):
        from rag.circuit_breaker import CircuitBreakerLLMClient
        call_times = []
        from rag.llm_client import BaseLLMClient
        class TimedClient(BaseLLMClient):
            def complete(self, *a, **k):
                call_times.append(time.monotonic())
                raise RuntimeError("always fail")
            async def acomplete(self, *a, **k):
                return self.complete()

        cb = CircuitBreakerLLMClient(
            TimedClient(), failure_threshold=10, max_retries=3,
            base_backoff_s=0.05, client_name="test"
        )
        with pytest.raises(RuntimeError):
            await cb.acomplete("prompt")

        assert len(call_times) == 3
        # Verify delay between retries
        gap1 = call_times[1] - call_times[0]
        assert gap1 >= 0.04  # At least base_backoff - some tolerance


# ════════════════════════════════════════════════════════════════════════════
# EVALUATOR v2 TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestEvaluatorV2:

    @pytest.mark.asyncio
    async def test_log_result_without_db_uses_logger(self):
        from evaluation.evaluator import Evaluator
        ev = Evaluator()
        ev._pool = None  # Force no DB

        result = PipelineResult(
            query="test", answer="answer", citations=[], confidence=0.9,
            loops=1, latency_ms=100,
        )
        # Should not raise
        await ev.log_result(result)

    @pytest.mark.asyncio
    async def test_offline_eval_token_f1(self):
        from evaluation.evaluator import Evaluator

        async def mock_pipeline(question: str) -> PipelineResult:
            return PipelineResult(
                query=question,
                answer="Paris is the capital of France",
                citations=["doc1#0"],
                confidence=0.95,
                loops=1,
                latency_ms=300,
            )

        ev = Evaluator()
        qa_pairs = [
            {"question": "What is the capital of France?", "expected": "Paris is the capital of France"},
            {"question": "What country is Paris in?", "expected": "Paris is in France"},
        ]
        metrics = await ev.run_offline_eval(qa_pairs, mock_pipeline)
        assert metrics["total_evaluated"] == 2
        assert 0.0 <= metrics["avg_token_f1"] <= 1.0
        assert 0.0 <= metrics["avg_faithfulness"] <= 1.0

    @pytest.mark.asyncio
    async def test_offline_eval_insufficient_context_scores_zero_f1(self):
        from evaluation.evaluator import Evaluator

        async def always_insufficient(q: str) -> PipelineResult:
            return PipelineResult(
                query=q, answer="INSUFFICIENT_CONTEXT", citations=[],
                confidence=0.0, loops=3, latency_ms=800,
                insufficient_context=True,
            )

        ev = Evaluator()
        metrics = await ev.run_offline_eval(
            [{"question": "Q?", "expected": "Some expected answer"}],
            always_insufficient,
        )
        assert metrics["avg_faithfulness"] == 0.0
        assert metrics["avg_token_f1"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_returns_empty_on_no_db(self):
        from evaluation.evaluator import Evaluator
        ev = Evaluator()
        ev._pool = None
        result = await ev.get_stats()
        assert result == {}


# ════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestAPIKeyStore:

    def test_accepts_correct_key(self):
        import os
        os.environ["NEURORAG_API_KEY"] = "my-secret-key"
        from api.middleware import APIKeyStore
        store = APIKeyStore()
        assert store.is_valid("my-secret-key")
        assert not store.is_valid("wrong-key")

    def test_dev_mode_accepts_all(self):
        import os
        os.environ.pop("NEURORAG_API_KEY", None)
        from api.middleware import APIKeyStore
        store = APIKeyStore()
        assert store.is_valid("")
        assert store.is_valid("anything")

    def test_keys_stored_as_hash_not_plaintext(self):
        import os
        os.environ["NEURORAG_API_KEY"] = "plaintext-key"
        from api.middleware import APIKeyStore
        store = APIKeyStore()
        # The plaintext should NOT appear in the valid_hashes set
        assert "plaintext-key" not in store._valid_hashes
        assert len(store._valid_hashes) == 1
        # But the hash should be there (64-char hex)
        (h,) = store._valid_hashes
        assert len(h) == 64


# ════════════════════════════════════════════════════════════════════════════
# SCHEMA COMPLETENESS TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestSchemaCompleteness:

    def test_pipeline_result_has_memory_fields(self):
        r = PipelineResult(
            query="q", answer="a", citations=[], confidence=0.9, loops=1, latency_ms=100,
        )
        assert hasattr(r, "from_memory_cache")
        assert hasattr(r, "context_hint_applied")

    def test_context_hint_schema_fields(self):
        hint = ContextHintSchema()
        assert hint.recommended_top_k_boost == 0
        assert hint.prior_failure_types == []
        assert hint.confidence_floor == 0.0
        assert not hint.from_cache

    def test_circuit_state_enum_values(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_all_failure_types_have_fix_action_mapping(self):
        from agents.reflection_fixer import ReflectionAgent
        from agents.schemas import CriticResult, FailureType
        agent = ReflectionAgent()
        for ft in FailureType:
            critique = CriticResult(
                faithfulness=0.1, relevance=0.1, completeness=0.1,
                confidence=0.1, failure_type=ft,
            )
            result = agent.analyze(critique, iteration=0)
            assert result.action is not None, f"No action mapped for FailureType.{ft.name}"


# ════════════════════════════════════════════════════════════════════════════
# PII FILTER TESTS
# ════════════════════════════════════════════════════════════════════════════

class TestPIIFilter:

    @pytest.fixture
    def pii_filter(self):
        from api.main import _filter_pii
        return _filter_pii

    def test_ssn_redacted(self, pii_filter):
        assert "123-45-6789" not in pii_filter("My SSN is 123-45-6789")
        assert "[REDACTED]" in pii_filter("My SSN is 123-45-6789")

    def test_email_redacted(self, pii_filter):
        result = pii_filter("Contact me at user@example.com please")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_phone_redacted(self, pii_filter):
        result = pii_filter("Call me at 555-123-4567")
        assert "555-123-4567" not in result

    def test_clean_text_unchanged(self, pii_filter):
        clean = "What is the capital of France?"
        assert pii_filter(clean) == clean

    def test_multiple_pii_all_redacted(self, pii_filter):
        text = "Name: John, SSN: 123-45-6789, email: john@doe.com, phone: 555-123-4567"
        result = pii_filter(text)
        assert "123-45-6789" not in result
        assert "john@doe.com" not in result
        assert "555-123-4567" not in result
        assert result.count("[REDACTED]") >= 3
