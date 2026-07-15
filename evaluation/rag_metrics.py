"""
NeuroRAG — RAG Retrieval + Generation Metrics v1
=================================================
Implements:
  - Precision@K
  - Recall@K
  - Context Relevance Score (cosine similarity between query and retrieved passages)
  - NDCG@K
  - MRR
  - Token-level F1
  - Hallucination proxy (faithfulness score from confidence)

All functions are pure Python / numpy; no LLM calls required.
Integration: imported by evaluation_runner.py and Airflow eval DAG.
"""
from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalSample:
    """One query with retrieved documents and relevance labels."""
    query: str
    retrieved_doc_ids: list[str]          # Ordered list (rank 1 first)
    relevant_doc_ids: set[str]            # Ground truth relevant set
    relevance_grades: dict[str, int] = field(default_factory=dict)
    # relevance_grades: doc_id → grade (0=irrelevant, 1=partial, 2=highly relevant)


@dataclass
class GenerationSample:
    """One query with generated answer and ground truth."""
    query: str
    generated_answer: str
    ground_truth_answer: str
    retrieved_contexts: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RAGMetrics:
    """Aggregated evaluation metrics for a dataset."""
    n_queries: int = 0
    # Retrieval
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    context_relevance: float = 0.0
    # Generation
    avg_token_f1: float = 0.0
    avg_confidence: float = 0.0
    avg_faithfulness: float = 0.0    # = avg_confidence as proxy
    # Self-healing
    avg_loops: float = 0.0
    cache_hit_rate: float = 0.0
    retry_success_rate: float = 0.0
    convergence_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": round(self.mrr, 4),
            "context_relevance": round(self.context_relevance, 4),
            "avg_token_f1": round(self.avg_token_f1, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_loops": round(self.avg_loops, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "retry_success_rate": round(self.retry_success_rate, 4),
            "convergence_efficiency": round(self.convergence_efficiency, 3),
        }


# ────────────────────────────────────────────────────────────────────────────
# TOKENISATION HELPERS
# ────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenisation for F1 scoring."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "but", "not", "it",
    "its", "this", "that", "these", "those",
})


def _content_tokens(text: str) -> list[str]:
    """Tokenise and remove stopwords (used for context relevance)."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


# ────────────────────────────────────────────────────────────────────────────
# TOKEN-LEVEL F1
# ────────────────────────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between prediction and ground truth strings.
    Matches the SQuAD evaluation protocol (bag-of-tokens overlap).

    Returns:
        F1 in [0.0, 1.0].
    """
    pred_tokens = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_bag: dict[str, int] = {}
    for t in pred_tokens:
        pred_bag[t] = pred_bag.get(t, 0) + 1

    truth_bag: dict[str, int] = {}
    for t in truth_tokens:
        truth_bag[t] = truth_bag.get(t, 0) + 1

    overlap = sum(
        min(pred_bag.get(t, 0), truth_bag[t]) for t in truth_bag
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


# ────────────────────────────────────────────────────────────────────────────
# PRECISION@K  and  RECALL@K
# ────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of top-K retrieved documents that are relevant.

    Args:
        retrieved: Ordered list of document IDs (rank 1 first).
        relevant:  Set of ground-truth relevant document IDs.
        k:         Cutoff rank.

    Returns:
        Precision@K in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of relevant documents that appear in the top-K retrieved.

    Returns:
        Recall@K in [0.0, 1.0]. Returns 1.0 if relevant is empty.
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Average Precision (AP): area under precision-recall curve.
    Used to compute MAP (Mean Average Precision) across queries.
    """
    if not relevant:
        return 1.0
    cumulative_precision = 0.0
    hits = 0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            cumulative_precision += hits / rank
    if hits == 0:
        return 0.0
    return cumulative_precision / len(relevant)


# ────────────────────────────────────────────────────────────────────────────
# NDCG@K
# ────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain at K.

    Args:
        retrieved:        Ordered list of doc IDs (rank 1 first).
        relevance_grades: doc_id → integer relevance grade (0, 1, 2).
        k:                Cutoff rank.

    Returns:
        NDCG@K in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    def _dcg(ordering: list[str], max_k: int) -> float:
        dcg = 0.0
        for rank, doc_id in enumerate(ordering[:max_k], start=1):
            grade = relevance_grades.get(doc_id, 0)
            dcg += (2 ** grade - 1) / math.log2(rank + 1)
        return dcg

    # Ideal ranking: sort by grade descending
    ideal_order = sorted(
        relevance_grades.keys(),
        key=lambda d: relevance_grades[d],
        reverse=True,
    )

    dcg = _dcg(retrieved, k)
    idcg = _dcg(ideal_order, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ────────────────────────────────────────────────────────────────────────────
# MRR
# ────────────────────────────────────────────────────────────────────────────

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal rank of the first relevant document in retrieved list.
    Returns 0.0 if no relevant document is retrieved.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(samples: list[RetrievalSample]) -> float:
    """Compute MRR across a list of retrieval samples."""
    if not samples:
        return 0.0
    return sum(
        reciprocal_rank(s.retrieved_doc_ids, s.relevant_doc_ids)
        for s in samples
    ) / len(samples)


# ────────────────────────────────────────────────────────────────────────────
# CONTEXT RELEVANCE SCORE
# ────────────────────────────────────────────────────────────────────────────

def context_relevance_score(
    query: str,
    contexts: list[str],
    use_embeddings: bool = False,
    embedder=None,
) -> float:
    """
    Measure how relevant retrieved contexts are to the query.

    Two modes:
      1. Lexical (default): Jaccard similarity on content tokens.
         Fast, no model required, suitable for CI and offline eval.
      2. Semantic (use_embeddings=True): cosine similarity using embedder.
         Higher accuracy; requires sentence-transformers installed.

    Args:
        query:           The original query string.
        contexts:        List of retrieved passage strings.
        use_embeddings:  Use semantic mode (requires embedder).
        embedder:        SentenceTransformer instance (optional).

    Returns:
        Average relevance score across contexts in [0.0, 1.0].
    """
    if not contexts:
        return 0.0

    if use_embeddings and embedder is not None:
        return _semantic_context_relevance(query, contexts, embedder)
    else:
        return _lexical_context_relevance(query, contexts)


def _lexical_context_relevance(query: str, contexts: list[str]) -> float:
    """Jaccard similarity between query tokens and each context."""
    query_tokens = set(_content_tokens(query))
    if not query_tokens:
        return 0.0

    scores = []
    for ctx in contexts:
        ctx_tokens = set(_content_tokens(ctx))
        if not ctx_tokens:
            scores.append(0.0)
            continue
        intersection = len(query_tokens & ctx_tokens)
        union = len(query_tokens | ctx_tokens)
        scores.append(intersection / union if union > 0 else 0.0)

    return sum(scores) / len(scores)


def _semantic_context_relevance(query: str, contexts: list[str], embedder) -> float:
    """Cosine similarity between query embedding and each context embedding."""
    texts = [query] + contexts
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    query_emb = embeddings[0]
    ctx_embs = embeddings[1:]
    similarities = np.dot(ctx_embs, query_emb)
    return float(np.mean(similarities))


# ────────────────────────────────────────────────────────────────────────────
# AGGREGATE OVER DATASET
# ────────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    samples: list[RetrievalSample],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """
    Compute Precision@K, Recall@K, NDCG@K, and MRR over a list of samples.

    Returns:
        Dict with keys precision_at_k, recall_at_k, ndcg_at_k, mrr, map.
    """
    if not samples:
        return {}

    precision_k: dict[int, list[float]] = {k: [] for k in k_values}
    recall_k: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_k: dict[int, list[float]] = {k: [] for k in k_values}
    rr_scores: list[float] = []
    ap_scores: list[float] = []

    for s in samples:
        for k in k_values:
            precision_k[k].append(
                precision_at_k(s.retrieved_doc_ids, s.relevant_doc_ids, k)
            )
            recall_k[k].append(
                recall_at_k(s.retrieved_doc_ids, s.relevant_doc_ids, k)
            )
            ndcg_k[k].append(
                ndcg_at_k(
                    s.retrieved_doc_ids,
                    s.relevance_grades or {d: 1 for d in s.relevant_doc_ids},
                    k,
                )
            )
        rr_scores.append(reciprocal_rank(s.retrieved_doc_ids, s.relevant_doc_ids))
        ap_scores.append(average_precision(s.retrieved_doc_ids, s.relevant_doc_ids))

    return {
        "precision_at_k": {k: round(sum(v) / len(v), 4) for k, v in precision_k.items()},
        "recall_at_k": {k: round(sum(v) / len(v), 4) for k, v in recall_k.items()},
        "ndcg_at_k": {k: round(sum(v) / len(v), 4) for k, v in ndcg_k.items()},
        "mrr": round(sum(rr_scores) / len(rr_scores), 4),
        "map": round(sum(ap_scores) / len(ap_scores), 4),
        "n_queries": len(samples),
    }


def compute_generation_metrics(samples: list[GenerationSample]) -> dict:
    """
    Compute token F1, average confidence, and context relevance.

    Returns:
        Dict with avg_token_f1, avg_confidence, avg_context_relevance.
    """
    if not samples:
        return {}

    f1_scores = [token_f1(s.generated_answer, s.ground_truth_answer) for s in samples]
    confidences = [s.confidence for s in samples]
    ctx_relevance = [
        context_relevance_score(s.query, s.retrieved_contexts)
        for s in samples
        if s.retrieved_contexts
    ]

    return {
        "avg_token_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "avg_context_relevance": round(
            sum(ctx_relevance) / len(ctx_relevance) if ctx_relevance else 0.0, 4
        ),
        "n_queries": len(samples),
    }


# ────────────────────────────────────────────────────────────────────────────
# PROMETHEUS LOGGING INTEGRATION
# ────────────────────────────────────────────────────────────────────────────

def log_retrieval_metrics_to_prometheus(metrics: dict) -> None:
    """
    Push retrieval benchmark metrics to Prometheus Gauges.
    Call after compute_retrieval_metrics() in the eval pipeline.
    """
    try:
        from prometheus_client import Gauge

        bench_precision = Gauge(
            "neurorag_bench_precision_at_k",
            "Benchmark Precision@K",
            ["k"],
        )
        bench_recall = Gauge(
            "neurorag_bench_recall_at_k",
            "Benchmark Recall@K",
            ["k"],
        )
        bench_ndcg = Gauge(
            "neurorag_bench_ndcg_at_k",
            "Benchmark NDCG@K",
            ["k"],
        )
        bench_mrr = Gauge("neurorag_bench_mrr", "Benchmark MRR")
        bench_map = Gauge("neurorag_bench_map", "Benchmark MAP")

        for k, v in metrics.get("precision_at_k", {}).items():
            bench_precision.labels(k=str(k)).set(v)
        for k, v in metrics.get("recall_at_k", {}).items():
            bench_recall.labels(k=str(k)).set(v)
        for k, v in metrics.get("ndcg_at_k", {}).items():
            bench_ndcg.labels(k=str(k)).set(v)
        if "mrr" in metrics:
            bench_mrr.set(metrics["mrr"])
        if "map" in metrics:
            bench_map.set(metrics["map"])

        logger.info("Retrieval metrics logged to Prometheus: %s", metrics)

    except Exception as exc:
        logger.warning("Could not log retrieval metrics to Prometheus: %s", exc)


def log_generation_metrics_to_prometheus(metrics: dict) -> None:
    """Push generation benchmark metrics to Prometheus Gauges."""
    try:
        from prometheus_client import Gauge

        bench_f1 = Gauge("neurorag_bench_token_f1", "Benchmark avg token F1")
        bench_conf = Gauge("neurorag_bench_avg_confidence", "Benchmark avg confidence")
        bench_ctx_rel = Gauge("neurorag_bench_context_relevance", "Benchmark context relevance")

        bench_f1.set(metrics.get("avg_token_f1", 0))
        bench_conf.set(metrics.get("avg_confidence", 0))
        bench_ctx_rel.set(metrics.get("avg_context_relevance", 0))

        logger.info("Generation metrics logged to Prometheus: %s", metrics)

    except Exception as exc:
        logger.warning("Could not log generation metrics to Prometheus: %s", exc)
