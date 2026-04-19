"""
NeuroRAG — Generator Agent
Produces grounded answers conditioned strictly on retrieved context.
Outputs INSUFFICIENT_CONTEXT when evidence is missing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.schemas import Document, GeneratorResult
from configs.settings import get_config

logger = logging.getLogger(__name__)

_GENERATOR_SYSTEM_PROMPT = """
You are a deterministic answer generator inside a production RAG system.

STRICT RULES — violations cause system failure:
1. Use ONLY the provided Context passages to answer.
2. Do NOT add any knowledge from outside the context.
3. Cite every factual claim using [doc_id#chunk_id] inline.
4. If the context does not contain sufficient information to answer, output EXACTLY:
   {"answer": "INSUFFICIENT_CONTEXT", "citations": []}
5. Keep the answer concise and factually precise.
6. No speculation, no filler, no caveats beyond what the context supports.

Output STRICT JSON only:
{
  "answer": "<grounded answer or INSUFFICIENT_CONTEXT>",
  "citations": ["<doc_id#chunk_id>", ...]
}
""".strip()


class Generator:
    """
    Generates grounded answers from retrieved context.
    """

    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client
        self._cfg = get_config()

    def generate(self, query: str, documents: list[Document], prompt_hint: str = "") -> GeneratorResult:
        """
        Generate a grounded answer from provided documents.

        Args:
            query:       Original user query.
            documents:   Reranked, top-k documents from retrieval.
            prompt_hint: Optional hint from Fixer agent to improve generation.

        Returns:
            GeneratorResult with answer and citations.
        """
        if not documents:
            return GeneratorResult(
                answer="INSUFFICIENT_CONTEXT",
                citations=[],
                raw_context_used="",
            )

        context_block = self._format_context(documents)

        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )
        if prompt_hint:
            user_prompt += f"\n\nHint: {prompt_hint}"

        try:
            raw = self._llm.complete(
                system=_GENERATOR_SYSTEM_PROMPT,
                prompt=user_prompt,
                temperature=self._cfg.llm.temperature,
                max_tokens=self._cfg.llm.max_tokens,
            )
            result = self._parse(raw, context_block)
            logger.info(
                "Generator: answer_length=%d citations=%d insufficient=%s",
                len(result.answer),
                len(result.citations),
                result.answer == "INSUFFICIENT_CONTEXT",
            )
            return result

        except Exception as exc:  # noqa: BLE001
            logger.error("Generator error: %s", exc)
            return GeneratorResult(
                answer="INSUFFICIENT_CONTEXT",
                citations=[],
                raw_context_used=context_block,
            )

    def _format_context(self, documents: list[Document]) -> str:
        parts = []
        for doc in documents:
            parts.append(
                f"[{doc.uid}]\n{doc.text.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse(self, raw: str, context_block: str) -> GeneratorResult:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(cleaned)
        return GeneratorResult(
            answer=data.get("answer", "INSUFFICIENT_CONTEXT"),
            citations=data.get("citations", []),
            raw_context_used=context_block,
        )
