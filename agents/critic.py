"""
NeuroRAG — Critic Agent
Evaluates generated answers for faithfulness, relevance, completeness.
Detects hallucinations by comparing claims against retrieved context.
Uses GPT-4 (critic_llm) to minimize self-evaluation bias.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.schemas import CriticResult, Document, FailureType
from configs.settings import get_config

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM_PROMPT = """
You are a strict hallucination detector and answer quality evaluator
inside a production RAG system. You are the last line of defense before
an answer reaches the user.

Evaluation criteria:
- faithfulness (0.0–1.0): Every claim in the answer MUST be supported
  verbatim or paraphrastically by the provided context. Any unsupported
  claim = hallucination. Score: fraction of claims supported.
- relevance (0.0–1.0): Does the answer address what the question asked?
- completeness (0.0–1.0): Are important aspects of the question covered?

Hallucination detection: If any claim in the answer cannot be traced
to a specific passage in the context, set hallucination_detected=true
and lower faithfulness accordingly.

confidence = 0.5 * faithfulness + 0.3 * relevance + 0.2 * completeness

Failure type taxonomy:
- "none": Answer is high quality
- "hallucination": Answer contains fabricated claims
- "missing_context": Context lacks the required information
- "irrelevance": Answer does not address the question
- "incomplete": Answer is correct but misses important aspects
- "other": Any other quality issue

Output STRICT JSON only:
{
  "faithfulness": <0.0–1.0>,
  "relevance": <0.0–1.0>,
  "completeness": <0.0–1.0>,
  "confidence": <0.0–1.0>,
  "failure_type": "<none|hallucination|missing_context|irrelevance|incomplete|other>",
  "hallucination_detected": <true|false>,
  "notes": "<brief factual explanation of issues found>"
}
""".strip()


class Critic:
    """
    Evaluates generated answers using a separate (potentially stronger) LLM.
    """

    def __init__(self, critic_llm_client: Any) -> None:
        self._llm = critic_llm_client
        self._cfg = get_config()

    def evaluate(
        self,
        query: str,
        answer: str,
        documents: list[Document],
    ) -> CriticResult:
        """
        Score answer quality and detect hallucinations.

        Args:
            query:     Original user query.
            answer:    Generated answer string.
            documents: Context documents used for generation.

        Returns:
            CriticResult with scores, failure_type, and hallucination flag.
        """
        # Short-circuit for INSUFFICIENT_CONTEXT — not a hallucination
        if answer.strip() == "INSUFFICIENT_CONTEXT":
            return CriticResult(
                faithfulness=1.0,
                relevance=0.5,
                completeness=0.0,
                confidence=0.5,
                failure_type=FailureType.MISSING_CONTEXT,
                hallucination_detected=False,
                notes="Answer correctly flagged missing context.",
            )

        context_block = self._format_context(documents)
        prompt = (
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Context Passages:\n{context_block}"
        )

        try:
            raw = self._llm.complete(
                system=_CRITIC_SYSTEM_PROMPT,
                prompt=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            result = self._parse(raw)
            logger.info(
                "Critic: conf=%.2f faith=%.2f rel=%.2f complete=%.2f failure=%s hallucination=%s",
                result.confidence,
                result.faithfulness,
                result.relevance,
                result.completeness,
                result.failure_type.value,
                result.hallucination_detected,
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.error("Critic error: %s — defaulting to low confidence", exc)
            return CriticResult(
                faithfulness=0.0,
                relevance=0.0,
                completeness=0.0,
                confidence=0.0,
                failure_type=FailureType.OTHER,
                hallucination_detected=True,
                notes=f"Critic invocation failed: {exc}",
            )

    def _format_context(self, documents: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[{doc.uid}] {doc.text.strip()}" for doc in documents
        )

    def _parse(self, raw: str) -> CriticResult:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(cleaned)
        confidence = (
            0.5 * float(data.get("faithfulness", 0))
            + 0.3 * float(data.get("relevance", 0))
            + 0.2 * float(data.get("completeness", 0))
        )
        return CriticResult(
            faithfulness=float(data.get("faithfulness", 0)),
            relevance=float(data.get("relevance", 0)),
            completeness=float(data.get("completeness", 0)),
            confidence=round(float(data.get("confidence", confidence)), 4),
            failure_type=FailureType(data.get("failure_type", "other")),
            hallucination_detected=bool(data.get("hallucination_detected", False)),
            notes=data.get("notes", ""),
        )
