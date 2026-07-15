"""
NeuroRAG — Data Ingestion Pipeline
Semantic chunking → embedding → FAISS index + BM25 index (Whoosh).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from whoosh import index as whoosh_index
from whoosh.fields import ID, TEXT, Schema

from configs.settings import get_config

logger = logging.getLogger(__name__)


# ─── Semantic Chunker ────────────────────────────────────────────────────────

def semantic_chunk(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into semantically coherent chunks.
    Prefers sentence boundaries over character boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # Overlap: carry last `overlap` chars into next chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


# ─── Ingestion Engine ────────────────────────────────────────────────────────

class IngestionEngine:
    """
    Orchestrates document ingestion: chunk → embed → index (FAISS + BM25).
    Thread-safe for concurrent use; FAISS is not fork-safe across processes.
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        ec = self._cfg.embedding
        vc = self._cfg.vector_store
        bc = self._cfg.bm25

        # Sentence-transformer for embedding
        self._embedder = SentenceTransformer(ec.model, device=ec.device)

        # FAISS index
        self._faiss_index: faiss.Index | None = None
        self._doc_ids: list[str] = []
        self._index_path = Path(vc.index_path)
        self._ids_path = Path(vc.ids_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ids_path.parent.mkdir(parents=True, exist_ok=True)

        # BM25 (Whoosh)
        self._whoosh_schema = Schema(uid=ID(stored=True), content=TEXT(stored=True))
        whoosh_dir = Path(bc.index_path)
        whoosh_dir.mkdir(parents=True, exist_ok=True)
        if whoosh_index.exists_in(str(whoosh_dir)):
            self._whoosh_ix = whoosh_index.open_dir(str(whoosh_dir))
        else:
            self._whoosh_ix = whoosh_index.create_in(str(whoosh_dir), self._whoosh_schema)

        self._load_faiss()

    # ── Public API ──────────────────────────────────────────────────────────

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """
        Ingest a batch of documents.

        Args:
            documents: List of {"id": str, "text": str, "metadata": dict}.

        Returns:
            Number of chunks indexed.
        """
        cfg = self._cfg.retrieval
        texts: list[str] = []
        uids: list[str] = []

        for doc in documents:
            doc_id = doc["id"]
            chunks = semantic_chunk(
                doc["text"],
                chunk_size=cfg.chunk_size,
                overlap=cfg.chunk_overlap,
            )
            for i, chunk in enumerate(chunks):
                uid = f"{doc_id}#{i}"
                texts.append(chunk)
                uids.append(uid)

        if not texts:
            logger.warning("IngestionEngine: no text chunks produced.")
            return 0

        self._embed_and_add_faiss(texts, uids)
        self._add_to_whoosh(texts, uids)

        logger.info("IngestionEngine: ingested %d chunks from %d docs.", len(texts), len(documents))
        return len(texts)

    def save(self) -> None:
        """Persist FAISS index and ID map to disk."""
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self._index_path))
            with open(self._ids_path, "w") as f:
                json.dump(self._doc_ids, f)
            logger.info("IngestionEngine: saved FAISS index (%d vectors).", len(self._doc_ids))

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_faiss(self) -> None:
        if self._index_path.exists() and self._ids_path.exists():
            self._faiss_index = faiss.read_index(str(self._index_path))
            with open(self._ids_path) as f:
                self._doc_ids = json.load(f)
            logger.info("IngestionEngine: loaded FAISS index (%d vectors).", len(self._doc_ids))
        else:
            dim = self._cfg.embedding.dimension
            inner = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIDMap(inner)
            self._doc_ids = []
            logger.info("IngestionEngine: created new FAISS index (dim=%d).", dim)

    def _embed_and_add_faiss(self, texts: list[str], uids: list[str]) -> None:
        ec = self._cfg.embedding
        embeddings = self._embedder.encode(
            texts,
            batch_size=ec.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=ec.normalize,
            show_progress_bar=len(texts) > 200,
        ).astype(np.float32)

        # FAISS needs integer IDs; map uid → sequential int
        start_id = len(self._doc_ids)
        int_ids = np.arange(start_id, start_id + len(uids), dtype=np.int64)
        self._faiss_index.add_with_ids(embeddings, int_ids)
        self._doc_ids.extend(uids)

    def _add_to_whoosh(self, texts: list[str], uids: list[str]) -> None:
        writer = self._whoosh_ix.writer()
        for uid, text in zip(uids, texts):
            writer.update_document(uid=uid, content=text)
        writer.commit()

    # ── Expose for Retriever ─────────────────────────────────────────────────

    @property
    def faiss_index(self) -> faiss.Index:
        return self._faiss_index

    @property
    def doc_ids(self) -> list[str]:
        return self._doc_ids

    @property
    def whoosh_ix(self) -> Any:
        return self._whoosh_ix

    @property
    def embedder(self) -> SentenceTransformer:
        return self._embedder
