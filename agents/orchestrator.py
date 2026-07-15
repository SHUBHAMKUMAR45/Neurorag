"""
NeuroRAG — Pipeline Orchestrator v2
Upgrades: AdaptiveContext memory, circuit-breaker LLM clients,
FailureMemory-aware reflection, per-agent Prometheus timing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager

from agents.critic import Critic
from agents.generator import Generator
from agents.intent_analyzer import IntentAnalyzer
from agents.memory import AdaptiveContext, FailureMemory
from agents.planner import Planner
from agents.reflection_fixer import FixerAgent, ReflectionAgent
from agents.schemas import Document, FailureType, FixerResult, PipelineResult
from configs.settings import get_config
from evaluation.evaluator import Evaluator
from rag.circuit_breaker import wrap_with_circuit_breaker
from rag.ingest import IngestionEngine
from rag.llm_client import BaseLLMClient, build_critic_llm_client, build_llm_client
from rag.reranker import Reranker
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str):
    t = time.monotonic()
    yield
    ms = (time.monotonic() - t) * 1000
    try:
        from dashboard.metrics import AGENT_LATENCY
        AGENT_LATENCY.labels(agent=label).observe(ms)
    except Exception:
        pass


class NeuroRAGOrchestrator:
    """Production self-healing RAG orchestrator with memory and circuit breaking."""

    def __init__(
        self,
        engine: IngestionEngine,
        llm: BaseLLMClient | None = None,
        critic_llm: BaseLLMClient | None = None,
    ) -> None:
        self._cfg = get_config()
        sh = self._cfg.self_heal

        raw_llm = llm or build_llm_client()
        raw_critic = critic_llm or build_critic_llm_client()

        self._generator_breaker = wrap_with_circuit_breaker(raw_llm, name="generator", failure_threshold=5)
        self._critic_breaker = wrap_with_circuit_breaker(raw_critic, name="critic", failure_threshold=3)

        self._intent_analyzer = IntentAnalyzer(self._generator_breaker)
        self._planner = Planner(self._generator_breaker)
        self._retriever = HybridRetriever(engine)
        self._reranker = Reranker()
        self._generator = Generator(self._generator_breaker)
        self._critic = Critic(self._critic_breaker)
        self._reflection = ReflectionAgent()
        self._fixer = FixerAgent()
        self._evaluator = Evaluator()
        self._adaptive_ctx = AdaptiveContext()
        self._fail_mem = FailureMemory()

        self._max_loops: int = sh.max_loops
        self._threshold: float = sh.confidence_threshold

    @property
    def generator_breaker(self):
        return self._generator_breaker

    @property
    def critic_breaker(self):
        return self._critic_breaker

    async def run(self, query: str) -> PipelineResult:
        t_start = time.monotonic()
        cfg = self._cfg.retrieval

        # Step 0: Memory / Adaptive Context
        with _timed("memory_lookup"):
            hint = await self._adaptive_ctx.build_hint(query)

        # Semantic cache hit — zero LLM calls
        if hint.similar_past_answer and hint.confidence_floor >= self._threshold:
            logger.info("MEMORY HIT: conf=%.3f", hint.confidence_floor)
            elapsed = int((time.monotonic() - t_start) * 1000)
            result = PipelineResult(
                query=query,
                answer=hint.similar_past_answer,
                citations=[],
                confidence=hint.confidence_floor,
                loops=0,
                latency_ms=elapsed,
                failure_history=[],
                insufficient_context=False,
                from_memory_cache=True,
                context_hint_applied=True,
            )
            asyncio.create_task(self._evaluator.log_result(result))  # noqa: RUF006
            return result

        # Step 1: Intent
        with _timed("intent"):
            intent = self._intent_analyzer.analyze(query)
        logger.info("Intent: %s complexity=%.2f", intent.query_type, intent.complexity)

        # Step 2: Plan
        with _timed("planner"):
            plan = self._planner.plan(query, intent)
        logger.info("Plan: %d sub-queries strategy=%s", len(plan.sub_queries), plan.strategy)

        current_query = query
        current_top_k = cfg.top_k + hint.recommended_top_k_boost
        current_strategy = plan.strategy
        prompt_hint = ""
        failure_history: list[FailureType] = []
        final_answer = "INSUFFICIENT_CONTEXT"
        final_citations: list[str] = []
        final_confidence = 0.0
        loop_count = 0

        for iteration in range(self._max_loops):
            loop_count = iteration + 1
            logger.info("=== LOOP %d/%d top_k=%d ===", loop_count, self._max_loops, current_top_k)

            # Retrieve
            t_ret = time.monotonic()
            all_docs = await self._retrieve_all(
                plan.sub_queries if iteration == 0 else [current_query],
                current_top_k, current_strategy,
            )
            try:
                from dashboard.metrics import DOCS_RETRIEVED, RETRIEVAL_LATENCY
                RETRIEVAL_LATENCY.observe((time.monotonic() - t_ret) * 1000)
                DOCS_RETRIEVED.observe(len(all_docs))
            except Exception:
                pass
            logger.info("Retrieved %d docs", len(all_docs))

            # Rerank
            with _timed("reranker"):
                ranked_docs = self._reranker.rerank(current_query, all_docs)
            logger.info("Reranked to %d docs", len(ranked_docs))

            # Generate
            with _timed("generator"):
                gen_result = self._generator.generate(current_query, ranked_docs, prompt_hint)
            logger.info("Generator: len=%d insufficient=%s", len(gen_result.answer), gen_result.answer == "INSUFFICIENT_CONTEXT")

            # Critique
            with _timed("critic"):
                critic_result = self._critic.evaluate(current_query, gen_result.answer, ranked_docs)
            logger.info(
                "Critic: conf=%.3f faith=%.3f failure=%s hallucination=%s",
                critic_result.confidence, critic_result.faithfulness,
                critic_result.failure_type.value, critic_result.hallucination_detected,
            )

            final_answer = gen_result.answer
            final_citations = gen_result.citations
            final_confidence = critic_result.confidence
            failure_history.append(critic_result.failure_type)

            if critic_result.confidence >= self._threshold:
                logger.info("PASS: conf=%.3f >= %.3f", critic_result.confidence, self._threshold)
                break

            if iteration >= self._max_loops - 1:
                logger.warning("MAX LOOPS reached. Best conf=%.3f", critic_result.confidence)
                break

            # Reflect — consult FailureMemory for learned correction
            with _timed("reflection"):
                mem_action = await self._fail_mem.get_recommended_fix(current_query, critic_result.failure_type)
                reflection = self._reflection.analyze(critic_result, iteration)
                if mem_action and mem_action != reflection.action:
                    logger.info("FailureMemory override: %s → %s", reflection.action.value, mem_action.value)
                    reflection.action = mem_action

            # Fix
            with _timed("fixer"):
                fixer_result: FixerResult = self._fixer.apply(current_query, reflection, iteration)

            current_query = fixer_result.modified_query
            if fixer_result.retrieval_top_k_override:
                current_top_k = fixer_result.retrieval_top_k_override
            if fixer_result.prompt_hint:
                prompt_hint = fixer_result.prompt_hint

            asyncio.create_task(  # noqa: RUF006
                self._fail_mem.record(query, critic_result.failure_type, reflection.action, False)
            )

        latency_ms = int((time.monotonic() - t_start) * 1000)

        result = PipelineResult(
            query=query,
            answer=final_answer,
            citations=final_citations,
            confidence=final_confidence,
            loops=loop_count,
            latency_ms=latency_ms,
            failure_history=failure_history,
            insufficient_context=(final_answer == "INSUFFICIENT_CONTEXT"),
            from_memory_cache=False,
            context_hint_applied=bool(hint.recommended_fix_action or hint.recommended_top_k_boost),
        )

        for task_coro in (
            self._evaluator.log_result(result),
            self._adaptive_ctx.record_result(query, result),
        ):
            if asyncio.iscoroutine(task_coro):
                asyncio.create_task(task_coro)
        return result

    async def _retrieve_all(self, sub_queries: list[str], top_k: int, strategy: str) -> list[Document]:
        tasks = [self._retriever.retrieve_async(sq, top_k=top_k, strategy=strategy) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        seen: dict[str, Document] = {}
        for res in results:
            if isinstance(res, BaseException):
                logger.warning("Retrieval failed for sub-query: %s", res)
                continue
            for doc in res:
                if doc.uid not in seen or doc.score > seen[doc.uid].score:
                    seen[doc.uid] = doc
        return sorted(seen.values(), key=lambda d: d.score, reverse=True)
