"""
NeuroRAG — Hybrid Retriever
Parallel BM25 (Whoosh) + vector (FAISS) retrieval with
Reciprocal Rank Fusion (RRF) for score merging.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from whoosh.qparser import MultifieldParser, OrGroup

from agents.schemas import Document
from configs.settings import get_config
from rag.ingest import IngestionEngine

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """
    Performs BM25 + vector retrieval in parallel and fuses results via RRF.
    """

    def __init__(self, engine: IngestionEngine) -> None:
        self._engine = engine
        self._cfg = get_config()

    # ── Public API ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str = "hybrid",
    ) -> list[Document]:
        """
        Retrieve top-k documents for query using configured strategy.

        Args:
            query:    Search query string.
            top_k:    Override for number of results (uses config default if None).
            strategy: "hybrid" | "bm25_only" | "vector_only"

        Returns:
            List of Documents sorted by fused relevance score (descending).
        """
        k = top_k or self._cfg.retrieval.top_k
        cfg = self._cfg.retrieval

        bm25_docs: list[Document] = []
        vector_docs: list[Document] = []

        if strategy in ("hybrid", "bm25_only"):
            bm25_docs = self._bm25_search(query, k)
        if strategy in ("hybrid", "vector_only"):
            vector_docs = self._vector_search(query, k)

        if strategy == "bm25_only":
            return bm25_docs[:k]
        if strategy == "vector_only":
            return vector_docs[:k]

        # Hybrid: RRF fusion
        return self._fuse(bm25_docs, vector_docs, k)

    async def retrieve_async(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str = "hybrid",
    ) -> list[Document]:
        """Async wrapper — runs BM25 and vector search in parallel threads."""
        k = top_k or self._cfg.retrieval.top_k
        loop = asyncio.get_event_loop()

        bm25_task = loop.run_in_executor(_EXECUTOR, self._bm25_search, query, k)
        vec_task = loop.run_in_executor(_EXECUTOR, self._vector_search, query, k)

        bm25_docs, vector_docs = await asyncio.gather(bm25_task, vec_task)

        if strategy == "bm25_only":
            return bm25_docs[:k]
        if strategy == "vector_only":
            return vector_docs[:k]

        return self._fuse(bm25_docs, vector_docs, k)

    # ── BM25 ────────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[Document]:
        try:
            ix = self._engine.whoosh_ix
            parser = MultifieldParser(
                ["content"],
                schema=ix.schema,
                group=OrGroup,
            )
            q = parser.parse(query)
            with ix.searcher() as s:
                results = s.search(q, limit=top_k)
                docs = []
                for hit in results:
                    docs.append(
                        Document(
                            doc_id=hit["uid"].split("#")[0],
                            chunk_id=hit["uid"].split("#")[1],
                            text=hit["content"],
                            score=float(hit.score),
                        )
                    )
            logger.debug("BM25: %d results for query '%s'", len(docs), query[:50])
            return docs
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 search failed: %s", exc)
            return []

    # ── Vector ──────────────────────────────────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> list[Document]:
        try:
            ec = self._cfg.embedding
            embedder = self._engine.embedder
            doc_ids = self._engine.doc_ids
            faiss_idx = self._engine.faiss_index

            if faiss_idx is None or not doc_ids:
                return []

            q_vec = embedder.encode(
                [query],
                normalize_embeddings=ec.normalize,
                convert_to_numpy=True,
            ).astype(np.float32)

            distances, indices = faiss_idx.search(q_vec, top_k)
            docs = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(doc_ids):
                    continue
                uid = doc_ids[idx]
                parts = uid.split("#")
                docs.append(
                    Document(
                        doc_id=parts[0],
                        chunk_id=parts[1] if len(parts) > 1 else "0",
                        text="",  # text loaded from Whoosh below if needed
                        score=float(dist),
                    )
                )
            logger.debug("Vector: %d results for query '%s'", len(docs), query[:50])
            return docs
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search failed: %s", exc)
            return []

    # ── RRF Fusion ──────────────────────────────────────────────────────────

    def _fuse(
        self,
        bm25_docs: list[Document],
        vector_docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        scores: dict[str, float] = {}
        uid_to_doc: dict[str, Document] = {}

        bm25_w = self._cfg.retrieval.bm25_weight
        vec_w = self._cfg.retrieval.vector_weight

        for rank, doc in enumerate(bm25_docs):
            scores[doc.uid] = scores.get(doc.uid, 0.0) + bm25_w * _rrf_score(rank)
            uid_to_doc[doc.uid] = doc

        for rank, doc in enumerate(vector_docs):
            scores[doc.uid] = scores.get(doc.uid, 0.0) + vec_w * _rrf_score(rank)
            if doc.uid not in uid_to_doc:
                uid_to_doc[doc.uid] = doc

        sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
        results = []
        for uid in sorted_uids[:top_k]:
            doc = uid_to_doc[uid]
            doc.score = scores[uid]
            # Hydrate text if missing (vector results may not carry it)
            if not doc.text:
                doc.text = self._fetch_text(uid)
            results.append(doc)

        return results

    def _fetch_text(self, uid: str) -> str:
        """Retrieve stored text from Whoosh by uid."""
        try:
            from whoosh.query import Term
            ix = self._engine.whoosh_ix
            with ix.searcher() as s:
                results = s.search(Term("uid", uid), limit=1)
                if results:
                    return results[0]["content"]
        except Exception:  # noqa: BLE001
            pass
        return ""
