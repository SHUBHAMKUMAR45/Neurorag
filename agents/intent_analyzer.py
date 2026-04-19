"""
NeuroRAG — Intent Analyzer Agent
Classifies query type and complexity to guide downstream agent behavior.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.schemas import IntentResult, QueryType
from configs.settings import get_config

logger = logging.getLogger(__name__)

_INTENT_SYSTEM_PROMPT = """
You are a query intent classifier inside a production RAG system.

Classify the given user query into EXACTLY ONE of these types:
- "factual": Single, verifiable fact lookup (e.g. "What year was X founded?")
- "reasoning": Requires inference or logical deduction from multiple facts
- "multi_hop": Requires chaining multiple retrieval steps
- "ambiguous": Unclear intent; may require clarification

Also estimate complexity on a 0.0–1.0 scale:
- 0.0–0.3: Simple single-hop
- 0.3–0.6: Moderate, some inference
- 0.6–1.0: Complex, multi-step reasoning

Output STRICT JSON only — no markdown, no preamble:
{
  "query_type": "<factual|reasoning|multi_hop|ambiguous>",
  "complexity": <float 0.0-1.0>,
  "reasoning": "<brief explanation>"
}
""".strip()


class IntentAnalyzer:
    """
    Classifies user query into structured intent metadata.
    Uses the configured LLM with temperature=0 for determinism.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client
        self._cfg = get_config()

    def analyze(self, query: str) -> IntentResult:
        """
        Classify query intent.

        Args:
            query: Raw user query string.

        Returns:
            IntentResult with type, complexity, and reasoning.
        """
        if not query or not query.strip():
            return IntentResult(
                query_type=QueryType.AMBIGUOUS,
                complexity=0.0,
                reasoning="Empty query",
            )

        prompt = f"{_INTENT_SYSTEM_PROMPT}\n\nQuery: {query}"

        try:
            raw = self._llm.complete(
                prompt=prompt,
                temperature=0.0,
                max_tokens=256,
            )
            result = self._parse(raw)
            logger.info(
                "IntentAnalyzer: type=%s complexity=%.2f query=%.60s",
                result.query_type,
                result.complexity,
                query,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("IntentAnalyzer failed (%s), defaulting to factual.", exc)
            return IntentResult(
                query_type=QueryType.FACTUAL,
                complexity=0.3,
                reasoning="Fallback due to classifier error",
            )

    def _parse(self, raw: str) -> IntentResult:
        # Strip any accidental markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(cleaned)
        return IntentResult(
            query_type=QueryType(data["query_type"]),
            complexity=float(data["complexity"]),
            reasoning=data.get("reasoning", ""),
        )
