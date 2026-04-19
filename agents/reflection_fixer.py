"""
NeuroRAG — Reflection Agent & Fixer Agent
Reflection: diagnoses failure root cause from critic output.
Fixer:      applies corrective strategy to query / retrieval params.
"""
from __future__ import annotations

import logging

from agents.schemas import (
    CriticResult,
    FailureType,
    FixAction,
    FixerResult,
    ReflectionResult,
)
from configs.settings import get_config

logger = logging.getLogger(__name__)


# ─── Reflection Agent ────────────────────────────────────────────────────────

class ReflectionAgent:
    """
    Analyzes critic feedback and identifies the root cause of failure.
    Pure deterministic logic — no LLM needed for speed and reliability.
    """

    def analyze(self, critique: CriticResult, iteration: int) -> ReflectionResult:
        """
        Map failure signal → root cause → corrective action.

        Args:
            critique:  Output from Critic agent.
            iteration: Current loop iteration (0-indexed).

        Returns:
            ReflectionResult with action and rationale.
        """
        ft = critique.failure_type

        if ft == FailureType.HALLUCINATION:
            return ReflectionResult(
                root_cause="Generator invented facts not present in context.",
                action=FixAction.ADD_CONTEXT,
                details="Increase retrieval top_k to provide more evidence.",
                priority=5,
            )

        if ft == FailureType.MISSING_CONTEXT:
            if iteration == 0:
                return ReflectionResult(
                    root_cause="Query too narrow; retrieved docs lack required information.",
                    action=FixAction.BROADEN_QUERY,
                    details="Expand search terms and increase retrieval scope.",
                    priority=4,
                )
            else:
                return ReflectionResult(
                    root_cause="Even broader query returned insufficient context.",
                    action=FixAction.INCREASE_TOP_K,
                    details=f"Double top_k from default (iteration {iteration}).",
                    priority=3,
                )

        if ft == FailureType.IRRELEVANCE:
            return ReflectionResult(
                root_cause="Generator drifted from question intent.",
                action=FixAction.NARROW_QUERY,
                details="Add explicit constraints to focus retrieval on question entities.",
                priority=4,
            )

        if ft == FailureType.INCOMPLETE:
            return ReflectionResult(
                root_cause="Answer partially correct but missing key aspects.",
                action=FixAction.ADD_CONTEXT,
                details="Retrieve additional passages covering missing aspects.",
                priority=3,
            )

        # Low confidence with no specific failure type
        return ReflectionResult(
            root_cause="Low overall answer quality.",
            action=FixAction.REFINE_PROMPT,
            details="Add explicit grounding hint to generator prompt.",
            priority=2,
        )


# ─── Fixer Agent ─────────────────────────────────────────────────────────────

class FixerAgent:
    """
    Applies corrective strategies recommended by Reflection.
    Modifies query and/or retrieval parameters for the next iteration.
    """

    def __init__(self) -> None:
        self._cfg = get_config()

    def apply(
        self,
        query: str,
        reflection: ReflectionResult,
        iteration: int,
    ) -> FixerResult:
        """
        Transform query and/or parameters based on reflection plan.

        Args:
            query:      Current query string.
            reflection: Output from ReflectionAgent.
            iteration:  Current loop iteration (0-indexed).

        Returns:
            FixerResult with modified query and optional overrides.
        """
        action = reflection.action

        if action == FixAction.BROADEN_QUERY:
            modified = self._broaden(query)
            logger.info("Fixer: BROADEN_QUERY — '%s' → '%s'", query, modified)
            return FixerResult(modified_query=modified)

        if action == FixAction.NARROW_QUERY:
            modified = self._narrow(query)
            logger.info("Fixer: NARROW_QUERY — '%s' → '%s'", query, modified)
            return FixerResult(modified_query=modified)

        if action == FixAction.ADD_CONTEXT:
            top_k_override = min(
                self._cfg.retrieval.top_k * (iteration + 2),
                30,  # hard cap
            )
            logger.info("Fixer: ADD_CONTEXT — top_k override=%d", top_k_override)
            return FixerResult(
                modified_query=query,
                retrieval_top_k_override=top_k_override,
            )

        if action == FixAction.INCREASE_TOP_K:
            top_k_override = self._cfg.retrieval.top_k * 2
            logger.info("Fixer: INCREASE_TOP_K — top_k=%d", top_k_override)
            return FixerResult(
                modified_query=query,
                retrieval_top_k_override=top_k_override,
            )

        if action == FixAction.REFINE_PROMPT:
            hint = (
                "Focus strictly on retrieving direct factual statements. "
                "If context is insufficient, output INSUFFICIENT_CONTEXT."
            )
            logger.info("Fixer: REFINE_PROMPT")
            return FixerResult(modified_query=query, prompt_hint=hint)

        # FixAction.NONE — no modification
        return FixerResult(modified_query=query)

    @staticmethod
    def _broaden(query: str) -> str:
        """Append synonymic expansion signal to the query."""
        return f"{query} overview explanation background"

    @staticmethod
    def _narrow(query: str) -> str:
        """Strip trailing expansion tokens from previous broadening."""
        tokens_to_strip = {"overview", "explanation", "background", "OR", "synonyms"}
        words = query.split()
        narrowed = [w for w in words if w.lower() not in tokens_to_strip]
        return " ".join(narrowed).strip() or query
