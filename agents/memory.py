"""
NeuroRAG — Adaptive Memory Layer
────────────────────────────────────────────────────────────────────────────
Three interconnected components:

1. QueryMemoryStore   — persists past (query, answer, confidence) tuples to
                        Redis + Postgres; used for semantic cache & dedup.
2. FailureMemory      — tracks failure patterns per query signature; feeds
                        the Fixer with historical correction strategies so
                        the system does NOT repeat the same mistake twice.
3. AdaptiveContext    — summarises memory signals into a ContextHint that
                        the Orchestrator injects into each pipeline run.

Design principles:
  - Redis for hot-path (TTL-based semantic cache)
  - Postgres for durable failure pattern storage
  - Embedding-based similarity for semantic dedup (avoids exact-match only)
  - Fully async; never blocks the query path
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from agents.schemas import FailureType, FixAction, PipelineResult
from configs.settings import get_config

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single stored query-answer record."""
    query_hash: str
    query: str
    answer: str
    citations: list[str]
    confidence: float
    loops: int
    latency_ms: int
    failure_types: list[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class FailurePattern:
    """Accumulated failure signals for a query signature."""
    query_hash: str
    failure_type: str
    fix_action: str
    success_after_fix: bool
    iteration: int
    count: int = 1


@dataclass
class ContextHint:
    """
    Injected into each pipeline run to pre-bias retrieval/generation
    based on learned memory signals.
    """
    similar_past_answer: Optional[str] = None
    recommended_fix_action: Optional[FixAction] = None
    recommended_top_k_boost: int = 0
    prior_failure_types: list[str] = field(default_factory=list)
    confidence_floor: float = 0.0


# ─── Query Memory Store ───────────────────────────────────────────────────────

class QueryMemoryStore:
    """
    Semantic memory of past query-answer pairs.

    Hot path  → Redis (TTL-based; checked before every pipeline run)
    Cold path → Postgres (durable; used for offline analysis)

    Similarity check uses cosine distance on cached embeddings so that
    near-duplicate queries (e.g. "capital of France?" vs
    "What's the capital of France") reuse existing high-confidence answers.
    """

    _REDIS_PREFIX = "qmem:"
    _EMBED_PREFIX = "qemb:"
    _SIM_THRESHOLD = 0.92   # cosine similarity to treat as near-duplicate

    def __init__(self) -> None:
        self._cfg = get_config()
        self._redis: Any = None
        self._pool: Any = None
        self._embedder: Any = None   # lazy-loaded SentenceTransformer

    # ── Public API ──────────────────────────────────────────────────────────

    async def lookup(self, query: str) -> Optional[MemoryEntry]:
        """
        Check if a high-confidence cached answer exists for this query.
        Returns None on cache miss or if cached confidence < threshold.
        """
        redis = await self._get_redis()
        if redis is None:
            return None

        # 1. Exact hash match (cheapest)
        exact_key = self._REDIS_PREFIX + self._hash(query)
        raw = await redis.get(exact_key)
        if raw:
            entry = MemoryEntry(**json.loads(raw))
            if entry.confidence >= self._cfg.self_heal.confidence_threshold:
                logger.info("MemoryStore: EXACT HIT for query '%s'", query[:60])
                return entry

        # 2. Semantic similarity scan (costlier; only if exact misses)
        return await self._semantic_lookup(redis, query)

    async def store(self, result: PipelineResult) -> None:
        """
        Persist a successful pipeline result (confidence >= threshold).
        Fire-and-forget; never raises.
        """
        if result.confidence < self._cfg.self_heal.confidence_threshold:
            return   # Don't cache low-quality answers

        try:
            entry = MemoryEntry(
                query_hash=self._hash(result.query),
                query=result.query,
                answer=result.answer,
                citations=result.citations,
                confidence=result.confidence,
                loops=result.loops,
                latency_ms=result.latency_ms,
                failure_types=[ft.value for ft in result.failure_history],
            )
            await self._store_redis(entry)
            await self._store_postgres(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MemoryStore.store failed (non-fatal): %s", exc)

    # ── Internal: Redis ──────────────────────────────────────────────────────

    async def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._cfg.cache.redis_url, decode_responses=True
                )
                await self._redis.ping()
            except Exception as exc:  # noqa: BLE001
                logger.warning("MemoryStore: Redis unavailable (%s)", exc)
                self._redis = None
        return self._redis

    async def _store_redis(self, entry: MemoryEntry) -> None:
        redis = await self._get_redis()
        if redis is None:
            return
        key = self._REDIS_PREFIX + entry.query_hash
        await redis.setex(key, self._cfg.cache.ttl_seconds, json.dumps(entry.__dict__))

        # Store embedding for semantic lookup
        emb = self._embed(entry.query)
        emb_key = self._EMBED_PREFIX + entry.query_hash
        await redis.setex(emb_key, self._cfg.cache.ttl_seconds, json.dumps(emb.tolist()))

    async def _semantic_lookup(self, redis: Any, query: str) -> Optional[MemoryEntry]:
        """Scan recent embedding keys and find cosine-similar queries."""
        try:
            q_emb = self._embed(query)
            keys = await redis.keys(self._EMBED_PREFIX + "*")
            if not keys:
                return None

            best_sim = 0.0
            best_hash: Optional[str] = None

            for emb_key in keys[:200]:   # Cap scan at 200 for latency
                raw_emb = await redis.get(emb_key)
                if not raw_emb:
                    continue
                stored_emb = np.array(json.loads(raw_emb), dtype=np.float32)
                sim = float(np.dot(q_emb, stored_emb))   # Both normalized
                if sim > best_sim:
                    best_sim = sim
                    best_hash = emb_key.replace(self._EMBED_PREFIX, "")

            if best_sim >= self._SIM_THRESHOLD and best_hash:
                raw = await redis.get(self._REDIS_PREFIX + best_hash)
                if raw:
                    entry = MemoryEntry(**json.loads(raw))
                    if entry.confidence >= self._cfg.self_heal.confidence_threshold:
                        logger.info(
                            "MemoryStore: SEMANTIC HIT sim=%.3f for query '%s'",
                            best_sim, query[:60],
                        )
                        return entry
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic lookup error (non-fatal): %s", exc)
        return None

    # ── Internal: Postgres ───────────────────────────────────────────────────

    async def _get_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self._cfg.database.postgres_url,
                    min_size=1, max_size=5,
                )
                async with self._pool.acquire() as conn:
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS query_memory (
                            query_hash   TEXT PRIMARY KEY,
                            query        TEXT NOT NULL,
                            answer       TEXT,
                            citations    TEXT[],
                            confidence   FLOAT,
                            loops        INT,
                            latency_ms   INT,
                            failure_types TEXT[],
                            created_at   TIMESTAMPTZ DEFAULT NOW(),
                            updated_at   TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MemoryStore: Postgres unavailable (%s)", exc)
        return self._pool

    async def _store_postgres(self, entry: MemoryEntry) -> None:
        pool = await self._get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO query_memory
                    (query_hash, query, answer, citations, confidence,
                     loops, latency_ms, failure_types, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
                ON CONFLICT (query_hash) DO UPDATE
                    SET answer=EXCLUDED.answer,
                        confidence=EXCLUDED.confidence,
                        loops=EXCLUDED.loops,
                        updated_at=NOW()
            """,
                entry.query_hash, entry.query, entry.answer, entry.citations,
                entry.confidence, entry.loops, entry.latency_ms, entry.failure_types,
            )

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:32]

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            ec = get_config().embedding
            self._embedder = SentenceTransformer(ec.model, device=ec.device)
        return self._embedder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)


# ─── Failure Memory ───────────────────────────────────────────────────────────

class FailureMemory:
    """
    Learns which fix actions resolved which failure types for similar queries.

    Storage: Redis hash  mem:fail:{query_hash}
      → Maps failure_type → {"fix_action": str, "success": bool, "count": int}

    The Fixer agent consults this before choosing a correction strategy,
    so the system does NOT re-apply failed fixes.
    """

    _PREFIX = "mem:fail:"

    def __init__(self) -> None:
        self._cfg = get_config()
        self._redis: Any = None

    async def record(
        self,
        query: str,
        failure_type: FailureType,
        fix_action: FixAction,
        succeeded: bool,
    ) -> None:
        """Record outcome of a fix attempt."""
        redis = await self._get_redis()
        if redis is None:
            return
        key = self._PREFIX + QueryMemoryStore._hash(query)
        field_name = failure_type.value
        existing_raw = await redis.hget(key, field_name)
        if existing_raw:
            rec = json.loads(existing_raw)
            rec["count"] += 1
            rec["success"] = rec["success"] or succeeded
            rec["fix_action"] = fix_action.value
        else:
            rec = {"fix_action": fix_action.value, "success": succeeded, "count": 1}
        await redis.hset(key, field_name, json.dumps(rec))
        await redis.expire(key, 86400 * 7)  # 7-day TTL

    async def get_recommended_fix(
        self,
        query: str,
        failure_type: FailureType,
    ) -> Optional[FixAction]:
        """
        Return a previously successful fix action for this failure type,
        or None if no history exists.
        """
        redis = await self._get_redis()
        if redis is None:
            return None
        key = self._PREFIX + QueryMemoryStore._hash(query)
        raw = await redis.hget(key, failure_type.value)
        if not raw:
            return None
        rec = json.loads(raw)
        if rec.get("success"):
            return FixAction(rec["fix_action"])
        return None  # Known failed fix — let Reflection choose differently

    async def get_prior_failures(self, query: str) -> list[str]:
        """Return all failure types seen for this query signature."""
        redis = await self._get_redis()
        if redis is None:
            return []
        key = self._PREFIX + QueryMemoryStore._hash(query)
        return list(await redis.hkeys(key))

    async def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._cfg.cache.redis_url, decode_responses=True
                )
                await self._redis.ping()
            except Exception as exc:  # noqa: BLE001
                logger.warning("FailureMemory: Redis unavailable (%s)", exc)
                self._redis = None
        return self._redis


# ─── Adaptive Context ─────────────────────────────────────────────────────────

class AdaptiveContext:
    """
    Synthesises memory signals into a ContextHint for the Orchestrator.
    Called once per query, before the first retrieval pass.
    """

    def __init__(self) -> None:
        self._query_mem = QueryMemoryStore()
        self._fail_mem = FailureMemory()

    async def build_hint(self, query: str) -> ContextHint:
        """
        Returns a ContextHint pre-populated from:
          - semantic cache hit (existing high-confidence answer)
          - historical failure patterns
          - recommended fix action from failure memory
        """
        # Check semantic cache
        cached = await self._query_mem.lookup(query)
        if cached:
            return ContextHint(
                similar_past_answer=cached.answer,
                confidence_floor=cached.confidence,
                prior_failure_types=cached.failure_types,
            )

        # Gather prior failure types for this query signature
        prior_failures = await self._fail_mem.get_prior_failures(query)

        # If there's a dominant prior failure, pre-recommend a fix
        recommended_action: Optional[FixAction] = None
        top_k_boost = 0
        for ft_str in prior_failures:
            try:
                ft = FailureType(ft_str)
                action = await self._fail_mem.get_recommended_fix(query, ft)
                if action:
                    recommended_action = action
                    if action in (FixAction.ADD_CONTEXT, FixAction.INCREASE_TOP_K):
                        top_k_boost = 4   # Pre-boost retrieval
                    break
            except ValueError:
                continue

        return ContextHint(
            recommended_fix_action=recommended_action,
            recommended_top_k_boost=top_k_boost,
            prior_failure_types=prior_failures,
        )

    async def record_result(self, query: str, result: "PipelineResult") -> None:
        """Called after each pipeline completes; updates both stores."""
        import asyncio
        await asyncio.gather(
            self._query_mem.store(result),
            self._record_failures(query, result),
            return_exceptions=True,
        )

    async def _record_failures(self, query: str, result: "PipelineResult") -> None:
        """Record fix outcomes for failure learning."""
        final_success = result.confidence >= get_config().self_heal.confidence_threshold
        for ft in result.failure_history:
            if ft == FailureType.NONE:
                continue
            # We don't know exactly which fix resolved it, so record last fix as
            # successful if pipeline ultimately passed, failed otherwise.
            await self._fail_mem.record(
                query=query,
                failure_type=ft,
                fix_action=FixAction.ADD_CONTEXT,  # default assumption
                succeeded=final_success,
            )
