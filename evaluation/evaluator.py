"""
NeuroRAG — Evaluator v3
Stores query results and critic scores to Postgres.
Exposes offline eval and live stats with full metric breakdown.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.schemas import CriticResult, PipelineResult

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self) -> None:
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg

                from configs.settings import get_config
                self._pool = await asyncpg.create_pool(
                    get_config().database.postgres_url, min_size=2, max_size=10)
                await self._ensure_schema()
            except Exception as exc:
                logger.warning("Evaluator: DB unavailable (%s)", exc)
        return self._pool

    async def _ensure_schema(self) -> None:
        pool = self._pool
        if not pool:
            return
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT,
                    citations TEXT[],
                    confidence FLOAT,
                    loops INT,
                    latency_ms INT,
                    insufficient BOOLEAN DEFAULT FALSE,
                    from_cache BOOLEAN DEFAULT FALSE,
                    hint_applied BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS eval_metrics (
                    id TEXT PRIMARY KEY REFERENCES queries(id) ON DELETE CASCADE,
                    faithfulness FLOAT,
                    relevance FLOAT,
                    completeness FLOAT,
                    failure_types TEXT[],
                    hallucination BOOLEAN DEFAULT FALSE,
                    evaluated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_eval_faith ON eval_metrics(faithfulness);
                CREATE INDEX IF NOT EXISTS idx_eval_hallucination ON eval_metrics(hallucination);
            """)

    async def log_result(
        self,
        result: PipelineResult,
        critic_result: CriticResult | None = None,
    ) -> None:
        pool = await self._get_pool()
        if pool is None:
            logger.info(
                "EVAL id=%s conf=%.3f loops=%d ms=%d cache=%s",
                result.request_id, result.confidence, result.loops,
                result.latency_ms, result.from_memory_cache,
            )
            return
        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("""
                        INSERT INTO queries (
                            id, query, answer, citations, confidence, loops,
                            latency_ms, insufficient, from_cache, hint_applied
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                        ON CONFLICT (id) DO NOTHING
                    """,
                        result.request_id, result.query, result.answer,
                        result.citations, result.confidence, result.loops,
                        result.latency_ms, result.insufficient_context,
                        result.from_memory_cache, result.context_hint_applied,
                    )

                    failure_types = [ft.value for ft in result.failure_history]
                    hallucination = any(
                        ft.value == "hallucination" for ft in result.failure_history
                    )
                    await conn.execute("""
                        INSERT INTO eval_metrics (
                            id, faithfulness, relevance, completeness,
                            failure_types, hallucination
                        )
                        VALUES ($1,$2,$3,$4,$5,$6)
                        ON CONFLICT (id) DO UPDATE SET
                            faithfulness   = EXCLUDED.faithfulness,
                            relevance      = EXCLUDED.relevance,
                            completeness   = EXCLUDED.completeness,
                            failure_types  = EXCLUDED.failure_types,
                            hallucination  = EXCLUDED.hallucination,
                            evaluated_at   = NOW()
                    """,
                        result.request_id,
                        critic_result.faithfulness if critic_result else None,
                        critic_result.relevance    if critic_result else None,
                        critic_result.completeness if critic_result else None,
                        failure_types,
                        hallucination,
                    )
        except Exception as exc:
            logger.error("Evaluator DB write failed: %s", exc)

    async def get_stats(self, window_hours: int = 24) -> dict:
        pool = await self._get_pool()
        if not pool:
            return {}
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*)                                                        AS total_queries,
                    AVG(q.confidence)                                               AS avg_confidence,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY q.latency_ms)     AS p95_latency_ms,
                    AVG(q.latency_ms)                                               AS avg_latency_ms,
                    AVG(q.loops)                                                    AS avg_loops,
                    SUM(CASE WHEN q.insufficient     THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(*), 0)                                       AS insufficient_rate,
                    SUM(CASE WHEN em.hallucination   THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(*), 0)                                       AS hallucination_rate,
                    AVG(em.faithfulness)                                            AS avg_faithfulness,
                    AVG(em.relevance)                                               AS avg_relevance,
                    AVG(em.completeness)                                            AS avg_completeness,
                    SUM(CASE WHEN q.from_cache       THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(*), 0)                                       AS cache_hit_rate
                FROM queries q
                LEFT JOIN eval_metrics em ON q.id = em.id
                WHERE q.created_at >= NOW() - ($1 || ' hours')::INTERVAL
            """, str(window_hours))

            failure_rows = await conn.fetch("""
                SELECT unnest(failure_types) AS ft, COUNT(*) AS cnt
                FROM eval_metrics em
                JOIN queries q ON em.id = q.id
                WHERE q.created_at >= NOW() - ($1 || ' hours')::INTERVAL
                  AND failure_types IS NOT NULL
                GROUP BY ft
                ORDER BY cnt DESC
            """, str(window_hours))

        result = dict(row) if row else {}
        result["failure_breakdown"] = {r["ft"]: r["cnt"] for r in failure_rows}
        return result

    async def run_offline_eval(
        self,
        qa_pairs: list[dict],
        pipeline_fn: Any,
    ) -> dict:
        faithfulness_scores: list[float] = []
        f1_scores: list[float] = []
        loop_counts: list[int] = []
        cache_hits = 0

        for pair in qa_pairs:
            result: PipelineResult = await pipeline_fn(pair["question"])
            exp = set(pair["expected"].lower().split())
            ans = set(result.answer.lower().split())
            if exp and ans:
                prec = len(exp & ans) / len(ans)
                rec  = len(exp & ans) / len(exp)
                f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            else:
                f1 = 0.0

            faithfulness_scores.append(result.confidence)
            f1_scores.append(f1)
            loop_counts.append(result.loops)
            if result.from_memory_cache:
                cache_hits += 1

        n = max(len(qa_pairs), 1)
        return {
            "avg_faithfulness": sum(faithfulness_scores) / n,
            "avg_token_f1":     sum(f1_scores) / n,
            "avg_loops":        sum(loop_counts) / n,
            "cache_hit_rate":   cache_hits / n,
            "total_evaluated":  len(qa_pairs),
        }
