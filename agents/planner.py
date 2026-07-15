"""
NeuroRAG — Planner Agent
Decomposes complex queries into targeted sub-queries and chooses retrieval strategy.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.schemas import IntentResult, PlanResult
from configs.settings import get_config

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """
You are a Planner agent inside a production RAG system.

Given a user query and its intent classification, decompose it into
focused sub-queries that maximize retrieval precision.

Rules:
- For "factual" queries: return 1 sub-query (the original, optionally refined)
- For "reasoning" queries: 2–3 sub-queries that isolate each fact needed
- For "multi_hop" queries: 3–5 sub-queries following the reasoning chain
- For "ambiguous" queries: 2 sub-queries covering the most likely interpretations

Each sub-query must be a self-contained search query (not a question to the user).

Output STRICT JSON only:
{
  "sub_queries": ["<query1>", "<query2>", ...],
  "strategy": "<hybrid|bm25_only|vector_only>",
  "notes": "<optional brief note>"
}
""".strip()


class Planner:
    """
    Decomposes user queries into retrieval-optimized sub-queries.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client
        self._cfg = get_config()

    def plan(self, query: str, intent: IntentResult) -> PlanResult:
        """
        Produce a retrieval plan for the given query.

        Args:
            query:  Raw user query.
            intent: Output from IntentAnalyzer.

        Returns:
            PlanResult containing sub-queries and retrieval strategy.
        """
        prompt = (
            f"{_PLANNER_SYSTEM_PROMPT}\n\n"
            f"Query: {query}\n"
            f"Intent Type: {intent.query_type.value}\n"
            f"Complexity: {intent.complexity:.2f}"
        )

        try:
            raw = self._llm.complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
            )
            result = self._parse(raw, query)
            logger.info(
                "Planner: %d sub-queries strategy=%s",
                len(result.sub_queries),
                result.strategy,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("Planner failed (%s), using original query.", exc)
            return PlanResult(
                sub_queries=[query],
                strategy="hybrid",
                notes="Fallback plan",
            )

    def _parse(self, raw: str, fallback_query: str) -> PlanResult:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(cleaned)
        sub_queries = data.get("sub_queries", [fallback_query])
        if not sub_queries:
            sub_queries = [fallback_query]
        return PlanResult(
            sub_queries=sub_queries,
            strategy=data.get("strategy", "hybrid"),
            notes=data.get("notes", ""),
        )
