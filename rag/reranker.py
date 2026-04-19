"""
NeuroRAG — Reranker
Cross-encoder reranking for improved retrieval precision.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from agents.schemas import Document
from configs.settings import get_config

logger = logging.getLogger(__name__)


class Reranker:
    """
    Uses a cross-encoder to score (query, passage) pairs and
    re-order retrieved documents for maximal relevance.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        rc = self._cfg.reranker
        self._model = CrossEncoder(rc.model, device=rc.device)
        logger.info("Reranker: loaded model '%s'", rc.model)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Score and sort documents by relevance to query.

        Args:
            query:     Search query string.
            documents: Candidate documents from hybrid retrieval.

        Returns:
            Top-k documents sorted by cross-encoder score (descending).
        """
        top_k = self._cfg.reranker.top_k

        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        pairs = [(query, doc.text) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score in ranked[:top_k]:
            doc.score = float(score)
            result.append(doc)

        logger.debug(
            "Reranker: %d → %d docs; top score=%.4f",
            len(documents),
            len(result),
            result[0].score if result else 0.0,
        )
        return result
