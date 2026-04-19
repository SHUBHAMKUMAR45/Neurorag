#!/usr/bin/env python3
"""
NeuroRAG — Seed Data Script
Ingests a small corpus of RAG/ML-domain documents for development and testing.
Run: python3 scripts/seed_data.py
Or:  make seed
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED_DOCUMENTS = [
    {
        "id": "rag-overview",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system "
            "with a large language model to produce grounded, factual answers. The retrieval system "
            "fetches relevant documents from a knowledge base, and the LLM generates an answer "
            "conditioned on those documents. This approach reduces hallucinations by grounding "
            "the model's output in retrieved evidence."
        ),
        "metadata": {"source": "seed", "topic": "rag"},
    },
    {
        "id": "bm25-overview",
        "text": (
            "BM25 (Best Match 25) is a probabilistic retrieval function used in information retrieval. "
            "It ranks documents based on the query terms appearing in each document, considering "
            "term frequency, inverse document frequency, and document length normalization. "
            "BM25 excels at exact keyword matching and is a strong baseline for lexical search."
        ),
        "metadata": {"source": "seed", "topic": "retrieval"},
    },
    {
        "id": "vector-search",
        "text": (
            "Vector search uses dense embeddings to find semantically similar documents. "
            "A neural encoder converts both the query and documents into high-dimensional vectors. "
            "Similarity is measured using cosine similarity or inner product. FAISS (Facebook AI "
            "Similarity Search) is an efficient library for approximate nearest-neighbour search "
            "over large vector collections, supporting GPU acceleration."
        ),
        "metadata": {"source": "seed", "topic": "retrieval"},
    },
    {
        "id": "hybrid-retrieval",
        "text": (
            "Hybrid retrieval combines lexical search (BM25) and semantic search (vector embeddings) "
            "to improve recall and precision. Reciprocal Rank Fusion (RRF) is a popular score merging "
            "strategy: it ranks documents by 1/(k + rank) from each system and sums the scores. "
            "Hybrid retrieval consistently outperforms either method alone, especially on diverse query types."
        ),
        "metadata": {"source": "seed", "topic": "retrieval"},
    },
    {
        "id": "cross-encoder-reranking",
        "text": (
            "Cross-encoder reranking is a two-stage retrieval technique. First, a fast bi-encoder "
            "retrieves a large candidate set. Then, a cross-encoder model scores each (query, passage) "
            "pair jointly, providing much more accurate relevance scores. Models like "
            "cross-encoder/ms-marco-MiniLM-L-6-v2 are popular for this task. Reranking typically "
            "improves retrieval precision by 10–30% over bi-encoder retrieval alone."
        ),
        "metadata": {"source": "seed", "topic": "reranking"},
    },
    {
        "id": "hallucination-detection",
        "text": (
            "Hallucination in LLMs refers to the generation of factually incorrect or unsupported "
            "content. In RAG systems, hallucination detection involves comparing the generated answer "
            "against the retrieved context passages. A critic LLM scores faithfulness: whether every "
            "claim in the answer is supported by the retrieved evidence. Low faithfulness scores "
            "trigger the self-healing loop to retry with additional context."
        ),
        "metadata": {"source": "seed", "topic": "evaluation"},
    },
    {
        "id": "self-healing-loop",
        "text": (
            "The NeuroRAG self-healing loop iteratively improves answer quality. After generation, "
            "a Critic agent scores faithfulness, relevance, and completeness. If the confidence score "
            "is below the threshold (0.90), a Reflection agent diagnoses the failure type and a Fixer "
            "agent modifies the query or retrieval parameters. The loop retries up to max_loops times "
            "before returning the best available answer."
        ),
        "metadata": {"source": "seed", "topic": "architecture"},
    },
    {
        "id": "semantic-chunking",
        "text": (
            "Semantic chunking splits documents at sentence boundaries rather than fixed character counts. "
            "This preserves semantic coherence within each chunk, improving retrieval quality. "
            "An overlap of 64 characters between consecutive chunks ensures that context spanning "
            "chunk boundaries is not lost. Chunk size of 512 characters balances context window "
            "usage with retrieval granularity."
        ),
        "metadata": {"source": "seed", "topic": "ingestion"},
    },
    {
        "id": "prometheus-metrics",
        "text": (
            "NeuroRAG exposes Prometheus metrics at port 9090. Key metrics include: "
            "neurorag_queries_total (counter by status), neurorag_query_latency_ms (histogram), "
            "neurorag_confidence_score (histogram), neurorag_hallucinations_total (counter), "
            "neurorag_gpu_utilization_percent (gauge), and neurorag_faiss_index_size (gauge). "
            "These metrics feed Grafana dashboards and alert rules for production monitoring."
        ),
        "metadata": {"source": "seed", "topic": "observability"},
    },
    {
        "id": "circuit-breaker-pattern",
        "text": (
            "The circuit breaker pattern prevents cascading failures in distributed systems. "
            "It has three states: CLOSED (normal operation), OPEN (blocking calls after N failures), "
            "and HALF_OPEN (probing with one request after cooldown). In NeuroRAG, LLM API calls "
            "are wrapped with circuit breakers. If the generator or critic LLM fails 5 consecutive "
            "times, the circuit opens and returns a graceful error instead of hanging the pipeline."
        ),
        "metadata": {"source": "seed", "topic": "resilience"},
    },
    {
        "id": "airflow-mlops",
        "text": (
            "Apache Airflow orchestrates NeuroRAG's MLOps pipelines. Five DAGs run on schedule: "
            "neurorag_ingest (weekly) rebuilds FAISS and BM25 indexes from new documents. "
            "neurorag_eval (daily) samples recent queries and runs offline evaluation. "
            "neurorag_drift (daily) detects embedding distribution shift and triggers retraining. "
            "neurorag_retrain rebuilds indexes with rollback support. "
            "neurorag_deploy performs canary deployment with automatic promote or rollback."
        ),
        "metadata": {"source": "seed", "topic": "mlops"},
    },
    {
        "id": "faiss-indexing",
        "text": (
            "FAISS (Facebook AI Similarity Search) provides efficient similarity search over dense "
            "vector collections. NeuroRAG uses IndexIDMap wrapping IndexFlatIP (inner product) for "
            "exact search with GPU acceleration. Embeddings are L2-normalized so inner product equals "
            "cosine similarity. The index is persisted to disk after each ingestion and loaded on "
            "startup to avoid re-embedding on restart."
        ),
        "metadata": {"source": "seed", "topic": "vector-store"},
    },
]


def main() -> None:
    import httpx

    base_url = os.environ.get("NEURORAG_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("NEURORAG_API_KEY", "")

    print(f"Seeding {len(SEED_DOCUMENTS)} documents into {base_url}...")

    try:
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            headers = {"X-API-Key": api_key} if api_key else {}
            resp = client.post(
                "/ingest",
                json={"documents": SEED_DOCUMENTS},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"✅ Seeded: {data['doc_count']} docs → {data['chunks_indexed']} chunks")

    except Exception as exc:
        print(f"❌ Seed failed: {exc}")
        print("Is the NeuroRAG API running? Try: make up")
        sys.exit(1)


if __name__ == "__main__":
    main()
