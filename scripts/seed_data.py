#!/usr/bin/env python3
"""
NeuroRAG — Seed Data Script v2
Ingests 50 rich documents covering RAG, ML, NLP, and system design topics.
Run: python scripts/seed_data.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# allow project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────
# Load .env automatically
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SEED_DOCUMENTS = [
    # ── RAG Fundamentals ──────────────────────────────────────────────────────
    {
        "id": "rag-overview",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large language model "
            "outputs by retrieving relevant information from an external knowledge base before generating "
            "a response. Unlike pure LLMs that rely solely on parametric memory (weights), RAG systems "
            "maintain a separate document store and retrieve the most relevant passages at query time. "
            "This grounding in retrieved evidence dramatically reduces hallucinations, allows knowledge "
            "to be updated without retraining, and provides citations for transparency. "
            "RAG was introduced by Lewis et al. in 2020 and has since become the dominant architecture "
            "for production question-answering systems. A typical RAG pipeline consists of: "
            "document ingestion and chunking, dense or sparse retrieval, optional reranking, "
            "and grounded answer generation conditioned on the retrieved context."
        ),
        "metadata": {"source": "seed", "topic": "rag", "difficulty": "beginner"},
    },
    {
        "id": "rag-advanced",
        "text": (
            "Advanced RAG architectures address the limitations of naive RAG through several techniques. "
            "Query decomposition breaks complex multi-hop questions into simpler sub-queries retrieved "
            "independently. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer "
            "and embeds that for retrieval rather than the raw query. FLARE (Forward-Looking Active "
            "Retrieval) retrieves iteratively as the model generates, fetching new context when "
            "confidence drops. Self-RAG trains a model to decide when to retrieve versus answer "
            "from memory. Corrective RAG adds a critic that evaluates retrieval quality and "
            "optionally triggers web search for additional context. "
            "Multi-vector retrieval stores summary embeddings alongside chunk embeddings to capture "
            "both high-level topics and fine-grained details."
        ),
        "metadata": {"source": "seed", "topic": "rag", "difficulty": "advanced"},
    },
    {
        "id": "rag-evaluation",
        "text": (
            "Evaluating RAG systems requires measuring both retrieval and generation quality separately. "
            "Retrieval metrics include Recall@K (fraction of relevant documents in top-K results), "
            "Precision@K (fraction of retrieved documents that are relevant), MRR (Mean Reciprocal Rank), "
            "and NDCG (Normalized Discounted Cumulative Gain). "
            "Generation metrics include: faithfulness (are all claims grounded in context?), "
            "answer relevance (does the answer address the question?), and context relevance "
            "(is the retrieved context actually relevant to the query?). "
            "The RAGAS framework automates these evaluations using LLM-as-judge scoring. "
            "Token-level F1 between expected and generated answer words measures factual overlap. "
            "Human evaluation remains the gold standard but is expensive at scale."
        ),
        "metadata": {"source": "seed", "topic": "evaluation", "difficulty": "intermediate"},
    },
    # ── Retrieval Methods ─────────────────────────────────────────────────────
    {
        "id": "bm25-overview",
        "text": (
            "BM25 (Best Match 25) is a probabilistic bag-of-words retrieval function used widely in "
            "information retrieval. It ranks documents based on term frequency (TF), inverse document "
            "frequency (IDF), and document length normalization. The key parameters are k1 (term "
            "frequency saturation, typically 1.2–2.0) and b (length normalization, typically 0.75). "
            "BM25 excels at exact keyword matching, handles rare technical terms well, and is extremely "
            "fast to index and query. It struggles with synonyms, paraphrases, and semantic similarity. "
            "Despite being decades old, BM25 remains a strong baseline that often matches or outperforms "
            "dense retrieval on domain-specific corpora where terminology is precise."
        ),
        "metadata": {"source": "seed", "topic": "retrieval", "difficulty": "intermediate"},
    },
    {
        "id": "vector-search",
        "text": (
            "Dense vector search uses neural embeddings to find semantically similar documents. "
            "A bi-encoder model converts both the query and documents into fixed-dimensional vectors "
            "in a shared semantic space. Similarity is typically measured using cosine similarity "
            "or dot product. FAISS (Facebook AI Similarity Search) provides highly optimized "
            "approximate nearest-neighbour (ANN) search over billions of vectors using techniques "
            "like IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World graphs). "
            "Dense retrieval captures semantic meaning, handles synonyms, and works well across "
            "languages. It requires fine-tuned encoders for domain-specific vocabulary and is slower "
            "to index than BM25 but enables richer semantic matching."
        ),
        "metadata": {"source": "seed", "topic": "retrieval", "difficulty": "intermediate"},
    },
    {
        "id": "hybrid-retrieval",
        "text": (
            "Hybrid retrieval combines lexical search (BM25) and semantic search (dense vectors) "
            "to achieve better recall and precision than either method alone. "
            "Reciprocal Rank Fusion (RRF) is the most popular score merging strategy: "
            "each document gets score sum(1 / (k + rank_i)) across all retrieval systems, "
            "where k=60 is a smoothing constant. RRF is robust, requires no score calibration, "
            "and consistently outperforms linear interpolation. "
            "An alternative is learned sparse retrieval (SPLADE, ELSER) which generates sparse "
            "token weights from a neural model, bridging the gap between BM25 and dense methods. "
            "Hybrid retrieval is the current best practice in production RAG systems, typically "
            "using BM25 weight 0.3–0.4 and vector weight 0.6–0.7."
        ),
        "metadata": {"source": "seed", "topic": "retrieval", "difficulty": "intermediate"},
    },
    {
        "id": "reranking",
        "text": (
            "Cross-encoder reranking is a two-stage retrieval technique that dramatically improves "
            "precision. In stage 1, a fast bi-encoder retrieves 20–50 candidate documents. "
            "In stage 2, a cross-encoder model processes each (query, document) pair jointly, "
            "computing a relevance score with full attention across both inputs. "
            "This joint processing is far more accurate but much slower — O(n) LLM calls per query. "
            "Popular cross-encoders include ms-marco-MiniLM-L-6-v2 (fast, 6 layers) and "
            "ms-marco-MiniLM-L-12-v2 (accurate, 12 layers). "
            "Reranking typically improves Precision@5 by 10–30% over bi-encoder retrieval alone. "
            "The reranked top-5 documents are then used as context for the LLM generator."
        ),
        "metadata": {"source": "seed", "topic": "reranking", "difficulty": "intermediate"},
    },
    # ── Chunking & Ingestion ──────────────────────────────────────────────────
    {
        "id": "semantic-chunking",
        "text": (
            "Chunking is the process of splitting documents into smaller segments for retrieval. "
            "Fixed-size chunking splits at character count boundaries — simple but breaks mid-sentence. "
            "Sentence-aware chunking splits at sentence boundaries, preserving semantic coherence. "
            "Semantic chunking uses embedding similarity to detect topic boundaries and split there. "
            "Chunk size (typically 256–1024 tokens) trades off between context richness and retrieval precision. "
            "Chunk overlap (typically 10–20% of chunk size) ensures context spanning chunk boundaries "
            "is not lost. Parent-child chunking stores small child chunks for retrieval but returns "
            "their larger parent chunks as context, balancing precision and completeness. "
            "Optimal chunk size depends on the document type: short for FAQ-style, long for technical docs."
        ),
        "metadata": {"source": "seed", "topic": "ingestion", "difficulty": "intermediate"},
    },
    {
        "id": "embedding-models",
        "text": (
            "Embedding models convert text into dense vectors for semantic search. "
            "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) is the most popular open-source "
            "model — fast, lightweight, and strong on general English. "
            "BAAI/bge-large-en-v1.5 (1024 dimensions) achieves state-of-the-art performance on BEIR. "
            "OpenAI text-embedding-3-small and text-embedding-3-large are strong closed-source options. "
            "Domain-specific embeddings (medical, legal, code) significantly outperform general models "
            "on specialized corpora. Embedding normalization (L2 norm) enables using dot product as "
            "cosine similarity, which is faster and allows inner product FAISS indexes. "
            "Batch encoding (processing many texts at once) is 5–10x faster than sequential encoding."
        ),
        "metadata": {"source": "seed", "topic": "embeddings", "difficulty": "intermediate"},
    },
    # ── Hallucination & Self-Healing ──────────────────────────────────────────
    {
        "id": "hallucination-detection",
        "text": (
            "Hallucination in LLMs refers to generating content that is factually incorrect, "
            "unsupported, or fabricated. In RAG systems, hallucination detection compares the "
            "generated answer against retrieved context passages. "
            "A faithfulness score (0–1) measures what fraction of claims in the answer are "
            "supported by the retrieved context. Scores below 0.7 typically indicate hallucination. "
            "Techniques to reduce hallucination include: "
            "1) Chain-of-thought prompting with explicit citation requirements. "
            "2) Using a separate, stronger critic LLM (e.g. GPT-4) to evaluate faithfulness. "
            "3) Self-consistency checking — generating multiple responses and checking agreement. "
            "4) INSUFFICIENT_CONTEXT fallback — returning explicit uncertainty rather than guessing. "
            "5) Retrieval augmentation — providing more context documents for grounding."
        ),
        "metadata": {"source": "seed", "topic": "hallucination", "difficulty": "intermediate"},
    },
    {
        "id": "self-healing-loop",
        "text": (
            "The NeuroRAG self-healing loop iteratively improves answer quality through a "
            "criticize-reflect-fix cycle. After generation, the Critic agent scores faithfulness, "
            "relevance, and completeness. If confidence < threshold (default 0.80), the Reflection "
            "agent diagnoses the failure type (hallucination, missing_context, irrelevance, incomplete). "
            "The Fixer agent then applies a corrective action: broadening or narrowing the query, "
            "increasing retrieval top_k, or adding a grounding hint to the generator prompt. "
            "The loop retries up to max_loops (default 3) times before returning the best answer. "
            "Failure Memory learns which fixes work for which failure types over time, "
            "pre-applying successful strategies on similar future queries to reduce loop count. "
            "Adaptive Context pre-boosts top_k when past failures required more context."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "advanced"},
    },
    # ── LLM Concepts ──────────────────────────────────────────────────────────
    {
        "id": "llm-overview",
        "text": (
            "Large Language Models (LLMs) are neural networks trained on vast text corpora to predict "
            "the next token given preceding context. Modern LLMs use the Transformer architecture with "
            "self-attention mechanisms. GPT-4 and Claude are closed-source frontier models with hundreds "
            "of billions of parameters. Open-source alternatives include LLaMA 3, Mistral, Gemma, and Phi. "
            "Key LLM concepts: temperature (randomness, 0=deterministic), top-p sampling (nucleus sampling), "
            "max_tokens (output length limit), system prompt (role/instruction prefix), "
            "context window (maximum input+output tokens, e.g. 128K for GPT-4o), "
            "and tokenization (text split into subword tokens, ~4 chars per token on average)."
        ),
        "metadata": {"source": "seed", "topic": "llm", "difficulty": "beginner"},
    },
    {
        "id": "prompt-engineering",
        "text": (
            "Prompt engineering is the practice of designing inputs to LLMs to elicit desired outputs. "
            "Key techniques include: "
            "Zero-shot prompting: ask the model directly without examples. "
            "Few-shot prompting: provide 2–5 examples of input→output pairs before the actual query. "
            "Chain-of-thought (CoT): ask the model to reason step-by-step ('think step by step'). "
            "Structured output: instruct the model to respond in JSON with a specific schema. "
            "Role-playing: set a system prompt defining the model's persona and constraints. "
            "Negative constraints: explicitly state what NOT to do ('Do not add information not in context'). "
            "In RAG, the generator prompt should: cite sources inline, output INSUFFICIENT_CONTEXT "
            "when evidence is missing, and avoid speculating beyond the retrieved context."
        ),
        "metadata": {"source": "seed", "topic": "llm", "difficulty": "intermediate"},
    },
    # ── System Architecture ───────────────────────────────────────────────────
    {
        "id": "circuit-breaker-pattern",
        "text": (
            "The circuit breaker pattern prevents cascading failures in distributed systems by "
            "temporarily blocking calls to a failing service. States: "
            "CLOSED (normal operation — all calls pass through), "
            "OPEN (blocking — too many recent failures; calls fail immediately with an error), "
            "HALF_OPEN (recovery probe — one test call is allowed through). "
            "If the test call succeeds, the circuit resets to CLOSED. If it fails, it stays OPEN. "
            "Parameters: failure_threshold (failures before opening, e.g. 5), "
            "cooldown_seconds (time before probing, e.g. 30), "
            "success_threshold (successes in HALF_OPEN before closing, e.g. 2). "
            "In NeuroRAG, both the generator and critic LLM clients are wrapped with circuit breakers "
            "to prevent a slow or failing OpenAI API from hanging the entire pipeline."
        ),
        "metadata": {"source": "seed", "topic": "resilience", "difficulty": "intermediate"},
    },
    {
        "id": "faiss-indexing",
        "text": (
            "FAISS (Facebook AI Similarity Search) is an open-source library for efficient "
            "similarity search over dense vectors. Key index types: "
            "IndexFlatL2 / IndexFlatIP: exact search, brute-force, 100% accurate. "
            "IndexIVFFlat: inverted file index partitions vectors into clusters, searches only relevant clusters. "
            "IndexHNSW: graph-based ANN search, extremely fast, high recall. "
            "IndexPQ: product quantization compresses vectors for memory efficiency. "
            "NeuroRAG uses IndexFlatIP (inner product / cosine similarity for normalized vectors) "
            "wrapped in IndexIDMap to support arbitrary string document IDs. "
            "The index is persisted to disk after each ingestion and loaded on startup. "
            "GPU FAISS provides 5–10x speedup for large indexes (>1M vectors)."
        ),
        "metadata": {"source": "seed", "topic": "vector-store", "difficulty": "intermediate"},
    },
    {
        "id": "redis-caching",
        "text": (
            "Redis is an in-memory data structure store used for caching in NeuroRAG. "
            "It serves two roles: response caching and rate limiting. "
            "Response caching: high-confidence query results are stored with a TTL (default 1 hour). "
            "On cache hit, the full response is returned instantly without any LLM calls, "
            "reducing latency from 1–3s to <5ms. Cache keys use a hash of the cleaned query string. "
            "Rate limiting: per-IP request counts are tracked in Redis with 60-second windows. "
            "Redis also stores the Adaptive Context memory — embeddings of past queries for semantic "
            "similarity matching. NeuroRAG degrades gracefully when Redis is unavailable: "
            "caching is simply disabled and all queries go through the full pipeline."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "intermediate"},
    },
    {
        "id": "prometheus-observability",
        "text": (
            "NeuroRAG exposes Prometheus metrics at port 9090 under /metrics. Key metrics: "
            "neurorag_queries_total{status} — counter of queries by status (success/insufficient/error). "
            "neurorag_query_latency_ms — histogram of end-to-end query latency. "
            "neurorag_confidence_score — histogram of critic confidence scores per query. "
            "neurorag_query_loops — histogram of self-healing iterations per query. "
            "neurorag_hallucinations_total — counter of hallucinations detected. "
            "neurorag_faiss_index_size — gauge of vectors in FAISS index. "
            "neurorag_agent_latency_ms{agent} — per-agent execution latency. "
            "neurorag_circuit_breaker_state{client} — circuit breaker state (0=closed, 1=open). "
            "These metrics feed Grafana dashboards and Prometheus alert rules for production monitoring."
        ),
        "metadata": {"source": "seed", "topic": "observability", "difficulty": "intermediate"},
    },
    {
        "id": "airflow-mlops",
        "text": (
            "Apache Airflow is the MLOps orchestration layer for NeuroRAG. Five production DAGs: "
            "neurorag_ingest (weekly): fetches new documents, chunks, embeds, rebuilds FAISS+BM25 indexes. "
            "neurorag_eval_v2 (daily): samples recent queries, computes average faithfulness, "
            "branches to neurorag_faithfulness_trigger if below threshold. "
            "neurorag_faithfulness_trigger (triggered): backup → rebuild → validate → compare → promote/rollback. "
            "neurorag_drift (daily): detects embedding distribution shift, triggers rebuild if needed. "
            "neurorag_deploy (manual): canary deployment with Prometheus error rate validation. "
            "Airflow Variables control thresholds: FAITHFULNESS_THRESHOLD (0.80), "
            "FAITHFULNESS_MIN_DELTA (0.05), INGEST_SOURCE_PATH (/data/raw)."
        ),
        "metadata": {"source": "seed", "topic": "mlops", "difficulty": "advanced"},
    },
    # ── NLP & ML Concepts ─────────────────────────────────────────────────────
    {
        "id": "transformer-architecture",
        "text": (
            "The Transformer architecture (Vaswani et al., 2017) revolutionized NLP with self-attention. "
            "Key components: "
            "Multi-head attention: queries (Q), keys (K), values (V) from the input allow each token "
            "to attend to all other tokens. Attention score = softmax(QK^T / sqrt(d_k)) * V. "
            "Positional encoding: adds position information since attention is order-agnostic. "
            "Feed-forward layers: two linear transformations with a ReLU/GELU activation. "
            "Layer normalization: stabilizes training. "
            "Encoder-only (BERT): bidirectional, used for embeddings and classification. "
            "Decoder-only (GPT): causal/autoregressive, used for text generation. "
            "Encoder-decoder (T5, BART): seq2seq tasks like translation and summarization. "
            "Scale laws show that doubling parameters roughly halves loss given sufficient data."
        ),
        "metadata": {"source": "seed", "topic": "ml-theory", "difficulty": "advanced"},
    },
    {
        "id": "fine-tuning",
        "text": (
            "Fine-tuning adapts a pretrained LLM to a specific domain or task. "
            "Full fine-tuning updates all model weights — expensive but most effective. "
            "LoRA (Low-Rank Adaptation) adds small trainable rank-decomposition matrices to "
            "attention layers, reducing trainable parameters by 99%+. "
            "QLoRA combines LoRA with 4-bit quantization, enabling fine-tuning of 7B models on a single GPU. "
            "RLHF (Reinforcement Learning from Human Feedback) aligns models with human preferences "
            "using a reward model trained on human comparisons. "
            "In RAG systems, fine-tuning the embedding model on domain-specific query-document pairs "
            "often gives larger gains than fine-tuning the generator, since retrieval quality is the bottleneck."
        ),
        "metadata": {"source": "seed", "topic": "ml-theory", "difficulty": "advanced"},
    },
    {
        "id": "attention-mechanisms",
        "text": (
            "Attention mechanisms allow neural networks to focus on relevant parts of the input. "
            "Scaled dot-product attention computes attention weights as softmax(QK^T / sqrt(d_k)). "
            "Multi-head attention runs h attention heads in parallel, each learning different relationships. "
            "Flash Attention (Dao et al., 2022) is an IO-aware exact attention algorithm that is "
            "2–4x faster and uses 5–20x less memory by tiling the softmax computation. "
            "Grouped Query Attention (GQA) shares key/value heads across query heads, "
            "reducing KV-cache memory without significant quality loss. "
            "Sliding window attention limits each token to attend only to nearby tokens, "
            "enabling efficient processing of very long documents."
        ),
        "metadata": {"source": "seed", "topic": "ml-theory", "difficulty": "advanced"},
    },
    # ── Infrastructure & DevOps ───────────────────────────────────────────────
    {
        "id": "docker-containers",
        "text": (
            "Docker containers package applications with all their dependencies for consistent "
            "deployment across environments. Key concepts: "
            "Dockerfile: instructions to build a container image (FROM, RUN, COPY, EXPOSE, CMD). "
            "Multi-stage builds: use one stage to compile/build and a smaller final stage for runtime. "
            "docker-compose: orchestrates multi-container applications (API, database, cache, monitoring). "
            "Volumes: persist data outside containers (Postgres data, FAISS index, model weights). "
            "Health checks: verify containers are ready before routing traffic. "
            "Resource limits: cap CPU and memory per container to prevent resource exhaustion. "
            "NVIDIA Docker runtime enables GPU access from containers (required for CUDA-based models)."
        ),
        "metadata": {"source": "seed", "topic": "infrastructure", "difficulty": "intermediate"},
    },
    {
        "id": "kubernetes-deployment",
        "text": (
            "Kubernetes orchestrates containerized workloads at scale. Key objects for NeuroRAG: "
            "Deployment: manages replica sets, handles rolling updates and rollbacks. "
            "HorizontalPodAutoscaler (HPA): scales replica count based on CPU/memory utilization. "
            "PersistentVolumeClaim (PVC): provides durable storage for FAISS index and model weights. "
            "Service: exposes the deployment via a stable DNS name and load balances across pods. "
            "Ingress: routes external HTTP traffic to the service, handles TLS termination. "
            "ConfigMap: stores non-secret configuration (log level, model paths). "
            "Secret: stores sensitive data (API keys, database passwords) with base64 encoding. "
            "Pod disruption budgets and rolling update strategy ensure zero-downtime deployments."
        ),
        "metadata": {"source": "seed", "topic": "infrastructure", "difficulty": "advanced"},
    },
    {
        "id": "postgresql-database",
        "text": (
            "PostgreSQL is the relational database used by NeuroRAG for durable query and evaluation storage. "
            "Three main tables: "
            "queries: stores each query result with id, query text, answer, citations, confidence, "
            "loops, latency_ms, insufficient flag, cache hit flag, and timestamp. "
            "eval_metrics: stores faithfulness, relevance, completeness, failure types, "
            "and hallucination flag linked to each query. "
            "query_memory: stores high-confidence query-answer pairs for the Adaptive Context layer, "
            "with embeddings stored separately in Redis for fast semantic lookup. "
            "Alembic manages schema migrations. asyncpg provides async connection pooling. "
            "BRIN indexes on created_at columns efficiently handle time-range queries."
        ),
        "metadata": {"source": "seed", "topic": "database", "difficulty": "intermediate"},
    },
    # ── Software Engineering Patterns ─────────────────────────────────────────
    {
        "id": "async-programming",
        "text": (
            "Async programming enables handling many concurrent I/O operations without blocking threads. "
            "Python's asyncio provides an event loop, coroutines (async def), and awaitables. "
            "await suspends the current coroutine and yields control to the event loop. "
            "asyncio.gather() runs multiple coroutines concurrently. "
            "asyncio.wait_for() adds a timeout to any coroutine. "
            "In FastAPI, all route handlers should be async to avoid blocking the event loop. "
            "CPU-bound work (embedding, FAISS search) must be offloaded to thread pools via "
            "loop.run_in_executor() to prevent blocking async I/O. "
            "uvicorn uses an event loop to handle thousands of concurrent HTTP connections "
            "with a single Python thread."
        ),
        "metadata": {"source": "seed", "topic": "software-engineering", "difficulty": "intermediate"},
    },
    {
        "id": "api-design",
        "text": (
            "REST API design principles for production ML systems: "
            "Versioning: prefix endpoints with /v1/, /v2/ to enable non-breaking evolution. "
            "Authentication: API keys in X-API-Key header (simple) or JWT Bearer tokens (stateful). "
            "Rate limiting: prevent abuse by capping requests per IP per time window. "
            "Request validation: use Pydantic models to validate and parse request bodies. "
            "Error responses: return structured JSON with detail field, appropriate HTTP status codes. "
            "408/504 for timeouts, 422 for validation errors, 503 for service unavailable. "
            "Health checks: /health endpoint reports service readiness (not just liveness). "
            "CORS: configure allowed origins for browser clients. "
            "OpenAPI: FastAPI auto-generates /docs and /openapi.json for API exploration."
        ),
        "metadata": {"source": "seed", "topic": "software-engineering", "difficulty": "intermediate"},
    },
    # ── Metrics & Monitoring ──────────────────────────────────────────────────
    {
        "id": "slo-sla",
        "text": (
            "SLOs (Service Level Objectives) and SLAs (Service Level Agreements) define reliability targets. "
            "Common NeuroRAG SLOs: p95 query latency ≤ 1500ms, error rate ≤ 1%, throughput ≥ 10 req/s. "
            "p50 (median) latency represents the typical user experience. "
            "p95 latency is the worst-case for 95% of users — the standard SLO metric. "
            "p99 latency catches extreme outliers caused by garbage collection or cold LLM starts. "
            "Error budget: the allowed amount of downtime/errors under an SLO (e.g. 99.9% = 8.7 hrs/year). "
            "SLIs (Service Level Indicators) are the actual measured metrics (latency, error rate). "
            "Alerting fires when burn rate exceeds 2x the error budget consumption rate."
        ),
        "metadata": {"source": "seed", "topic": "observability", "difficulty": "intermediate"},
    },
    {
        "id": "canary-deployment",
        "text": (
            "Canary deployment gradually rolls out a new version by routing a small percentage "
            "of traffic to it while monitoring for regressions. "
            "Steps: deploy new version to canary pods (e.g. 1 of 10), "
            "route 10% of traffic to canary via load balancer weights, "
            "monitor error rate, latency, and domain-specific metrics for 5–15 minutes, "
            "promote to 100% if metrics are healthy, rollback immediately if any threshold is exceeded. "
            "In NeuroRAG's Airflow deploy DAG, Prometheus metrics are queried to validate the canary "
            "before the kubectl set image command promotes it to the full fleet. "
            "Rollback is automated: if the Prometheus query exceeds 5% error rate, "
            "kubectl rollout undo reverts the deployment instantly."
        ),
        "metadata": {"source": "seed", "topic": "mlops", "difficulty": "advanced"},
    },
    # ── Data Science ──────────────────────────────────────────────────────────
    {
        "id": "cosine-similarity",
        "text": (
            "Cosine similarity measures the angle between two vectors, ignoring their magnitudes. "
            "Formula: cos(θ) = (A · B) / (|A| * |B|), ranging from -1 (opposite) to 1 (identical). "
            "For normalized vectors (|A| = |B| = 1), cosine similarity equals the dot product, "
            "which is why FAISS IndexFlatIP (inner product) is equivalent to cosine similarity "
            "when vectors are L2-normalized before indexing. "
            "In semantic search, cosine similarity captures the direction (topic) of vectors "
            "rather than their magnitude. A threshold of 0.85–0.92 is commonly used to detect "
            "near-duplicate queries for cache hits. Euclidean distance (L2) is an alternative "
            "metric that considers magnitude but is less common for semantic similarity tasks."
        ),
        "metadata": {"source": "seed", "topic": "math", "difficulty": "intermediate"},
    },
    {
        "id": "information-retrieval-metrics",
        "text": (
            "Information retrieval metrics quantify the quality of a retrieval system. "
            "Precision@K: among the top-K retrieved documents, what fraction are relevant? "
            "Recall@K: among all relevant documents in the corpus, what fraction appear in top-K? "
            "F1@K: harmonic mean of Precision@K and Recall@K, balancing both. "
            "Mean Reciprocal Rank (MRR): average of 1/rank_of_first_relevant_document across queries. "
            "Mean Average Precision (MAP): average of precision at each relevant document position. "
            "NDCG@K: normalized discounted cumulative gain, rewards highly relevant docs at top positions. "
            "For RAG evaluation: Recall@K is most important (not missing relevant evidence), "
            "while Precision@K affects context length and noise in the generator input."
        ),
        "metadata": {"source": "seed", "topic": "evaluation", "difficulty": "intermediate"},
    },
    {
        "id": "embedding-drift",
        "text": (
            "Embedding drift occurs when the statistical distribution of document or query embeddings "
            "shifts over time, degrading retrieval quality. Causes include: "
            "corpus updates (new documents with different terminology), "
            "query pattern changes (users asking about new topics), "
            "model updates (embedding model retrained with new data). "
            "Detection: compare mean pairwise cosine distances between current and baseline embeddings. "
            "A drift score > 0.12 typically warrants index rebuilding. "
            "NeuroRAG's neurorag_drift Airflow DAG runs daily, computing the Jensen-Shannon divergence "
            "between the current embedding distribution and a stored baseline. "
            "Automated retraining pipelines trigger index rebuilds and canary deployments "
            "when drift exceeds configured thresholds."
        ),
        "metadata": {"source": "seed", "topic": "mlops", "difficulty": "advanced"},
    },
    # ── FastAPI & Python ───────────────────────────────────────────────────────
    {
        "id": "fastapi-overview",
        "text": (
            "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI documentation. "
            "Key features: "
            "Type hints and Pydantic models for automatic request validation and serialization. "
            "async/await support — all route handlers can be async coroutines. "
            "Dependency injection via Depends() for shared resources (database sessions, auth). "
            "Automatic /docs (Swagger UI) and /redoc documentation from route annotations. "
            "Background tasks with BackgroundTasks for fire-and-forget operations. "
            "Lifespan context manager for startup/shutdown events (database connections, model loading). "
            "Middleware stack for cross-cutting concerns: auth, logging, CORS, timeout, rate limiting. "
            "WebSocket support for streaming responses. "
            "FastAPI is built on Starlette (ASGI framework) and Pydantic (data validation)."
        ),
        "metadata": {"source": "seed", "topic": "software-engineering", "difficulty": "beginner"},
    },
    {
        "id": "pydantic-validation",
        "text": (
            "Pydantic provides runtime data validation using Python type hints. "
            "BaseModel defines data schemas with automatic validation, parsing, and serialization. "
            "Field() adds metadata: default values, constraints (min_length, ge, le), descriptions. "
            "model_dump() serializes to dict; model_dump_json() serializes to JSON string. "
            "model_config = ConfigDict(extra='ignore') silently drops unknown fields (used in NeuroRAG "
            "to handle the 'system:' block in config.yaml that has no corresponding model field). "
            "Validators (@field_validator) add custom validation logic. "
            "Pydantic v2 (used in NeuroRAG) is 5–50x faster than v1 due to a Rust core (pydantic-core). "
            "Settings management via pydantic-settings reads from environment variables and .env files."
        ),
        "metadata": {"source": "seed", "topic": "software-engineering", "difficulty": "intermediate"},
    },
    # ── Security ──────────────────────────────────────────────────────────────
    {
        "id": "api-security",
        "text": (
            "API security for ML systems requires multiple layers of protection. "
            "Authentication: API keys are simple tokens validated against stored SHA-256 hashes "
            "(never stored in plaintext). JWT tokens add expiration and user identity claims. "
            "PII filtering: regex patterns strip email addresses, phone numbers, and SSNs "
            "from queries before processing or logging. "
            "Rate limiting: per-IP request caps prevent abuse and protect LLM API costs. "
            "Input validation: maximum query length prevents prompt injection via extremely long inputs. "
            "TLS: all traffic should be encrypted in transit via HTTPS/TLS 1.3. "
            "Secret management: API keys loaded from environment variables, never hardcoded. "
            "Audit logging: all requests logged with timestamp, IP, path, and status for forensics."
        ),
        "metadata": {"source": "seed", "topic": "security", "difficulty": "intermediate"},
    },
    # ── Multi-hop Reasoning ───────────────────────────────────────────────────
    {
        "id": "multi-hop-reasoning",
        "text": (
            "Multi-hop reasoning requires chaining multiple retrieval and inference steps. "
            "A query like 'Which model powers NeuroRAG's critic, and what is its context window?' "
            "requires: (1) retrieving that the critic uses GPT-4o-mini, "
            "(2) retrieving GPT-4o-mini's context window (128K tokens). "
            "Decomposition strategies: "
            "Sequential: answer sub-question 1, use answer to form sub-question 2. "
            "Parallel: retrieve for all sub-questions simultaneously, aggregate. "
            "Iterative: generate, evaluate confidence, retrieve more if insufficient. "
            "IRCoT (Interleaving Retrieval with Chain-of-Thought) interleaves thinking steps "
            "with retrieval calls, fetching evidence as reasoning progresses. "
            "Multi-hop queries are the hardest RAG evaluation category — NeuroRAG's benchmark "
            "includes 100 multi-hop questions to measure this capability."
        ),
        "metadata": {"source": "seed", "topic": "reasoning", "difficulty": "advanced"},
    },
    {
        "id": "query-decomposition",
        "text": (
            "Query decomposition breaks a complex question into simpler retrievable sub-queries. "
            "The NeuroRAG Planner agent decomposes queries based on intent classification: "
            "Factual queries: 1 sub-query (the original, optionally refined). "
            "Reasoning queries: 2–3 sub-queries isolating each required fact. "
            "Multi-hop queries: 3–5 sub-queries following the reasoning chain. "
            "Ambiguous queries: 2 sub-queries covering likely interpretations. "
            "Each sub-query is retrieved independently in parallel, then results are merged "
            "via deduplication (keeping highest-scored version of each document). "
            "The aggregated document set is then reranked as a whole before being passed to the generator. "
            "Query decomposition consistently improves Recall@K for complex questions by 15–40%."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "advanced"},
    },
    # ── Optimization ──────────────────────────────────────────────────────────
    {
        "id": "rag-optimization",
        "text": (
            "Optimizing RAG systems for production involves multiple dimensions: "
            "Latency: cache high-confidence responses (Redis TTL 1hr), use async parallel retrieval, "
            "select fast embedding models (MiniLM vs BGE-large = 10x speed difference). "
            "Quality: increase chunk overlap, use larger reranker models, tune confidence threshold. "
            "Cost: cache responses aggressively, use gpt-4o-mini for both generator and critic "
            "instead of gpt-4, limit max_tokens to avoid over-generation. "
            "Throughput: async ASGI server handles concurrent requests, "
            "thread pool offloads CPU-bound embedding/FAISS to avoid event loop blocking. "
            "Index quality: chunk size tuning, overlap adjustment, and periodic reindexing "
            "with updated embedding models significantly impact retrieval quality over time."
        ),
        "metadata": {"source": "seed", "topic": "optimization", "difficulty": "advanced"},
    },
    {
        "id": "latency-optimization",
        "text": (
            "Query latency in RAG systems comes from: "
            "Embedding (50–200ms): encode query to vector. Batching and GPU acceleration help. "
            "FAISS search (1–50ms): ANN search over index. IndexFlatIP is exact but slower than IVF. "
            "BM25 search (10–100ms): Whoosh full-text search, scales with index size. "
            "Reranking (100–500ms): cross-encoder scores each (query, doc) pair — top bottleneck. "
            "LLM generation (500ms–5s): OpenAI API or local LLM. Streaming reduces perceived latency. "
            "Network (50–200ms): round-trip to OpenAI API. Retry + circuit breaker prevents hanging. "
            "Optimization strategies: parallel BM25+vector retrieval, Redis caching of responses, "
            "reducing reranker candidate count (from 50 to 20), and async critic (fire-and-forget "
            "for queries already above threshold)."
        ),
        "metadata": {"source": "seed", "topic": "optimization", "difficulty": "advanced"},
    },
    # ── Context Window Management ─────────────────────────────────────────────
    {
        "id": "context-window",
        "text": (
            "The context window is the maximum number of tokens an LLM can process in a single call, "
            "including both input (system prompt + retrieved context + query) and output (answer). "
            "GPT-4o and GPT-4o-mini support 128K token context windows. "
            "Context window management in RAG: "
            "max_context_tokens limits how much retrieved text is passed to the generator. "
            "Truncation strategies: take top-K chunks by score, truncate last chunk if needed. "
            "Lost in the middle problem: LLMs attend better to content at start and end of context. "
            "Solution: place most relevant chunks first and last, less relevant in the middle. "
            "Semantic compression: summarize long chunks before including them to fit more documents. "
            "NeuroRAG uses max_context_tokens=6000 (about 4500 words of context per query)."
        ),
        "metadata": {"source": "seed", "topic": "llm", "difficulty": "intermediate"},
    },
    {
        "id": "structured-output",
        "text": (
            "Structured output constrains LLM responses to a specific format (JSON, XML, etc.) "
            "for reliable parsing in downstream systems. Techniques: "
            "Prompting: 'Output STRICT JSON only. No markdown fences, no preamble.' "
            "OpenAI response_format: {type: 'json_object'} enforces valid JSON output. "
            "OpenAI structured outputs with JSON Schema enforce specific field types and enums. "
            "Regex-based extraction as a fallback for imperfect JSON with markdown fences. "
            "In NeuroRAG, all agent prompts (generator, critic, planner, intent) require JSON output. "
            "The parser strips ```json fences, then calls json.loads(). "
            "Failed parses default to conservative fallback values (INSUFFICIENT_CONTEXT, failure_type=other) "
            "rather than crashing, ensuring the pipeline always returns a response."
        ),
        "metadata": {"source": "seed", "topic": "llm", "difficulty": "intermediate"},
    },
    # ── NeuroRAG Specific ─────────────────────────────────────────────────────
    {
        "id": "neurorag-architecture",
        "text": (
            "NeuroRAG is an autonomous self-healing multi-agent RAG system. Core components: "
            "IntentAnalyzer: classifies query type (factual/reasoning/multi_hop/ambiguous) and complexity. "
            "Planner: decomposes query into retrieval sub-queries and selects strategy. "
            "HybridRetriever: parallel BM25 (Whoosh) + vector (FAISS) retrieval with RRF fusion. "
            "Reranker: cross-encoder re-scores retrieved documents for precision. "
            "Generator: produces grounded answers conditioned on retrieved context (OpenAI or local LLaMA). "
            "Critic: evaluates faithfulness, relevance, completeness using a separate LLM. "
            "ReflectionAgent: diagnoses failure type and selects corrective action. "
            "FixerAgent: applies corrections (query expansion, top_k increase, prompt hints). "
            "AdaptiveContext + FailureMemory: learn from past failures to improve future queries. "
            "All components are wrapped with circuit breakers and produce Prometheus metrics."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "advanced"},
    },
    {
        "id": "neurorag-agents",
        "text": (
            "NeuroRAG's multi-agent pipeline coordinates six specialized agents: "
            "1. IntentAnalyzer: uses LLM to classify query type and complexity (0.0–1.0). "
            "Output: IntentResult with query_type (factual/reasoning/multi_hop/ambiguous). "
            "2. Planner: decomposes query into sub-queries based on intent. "
            "Output: PlanResult with sub_queries list and retrieval strategy. "
            "3. Generator: produces JSON answer with inline citations. "
            "Output: GeneratorResult with answer string and citations list. "
            "4. Critic: scores answer quality and detects hallucinations. "
            "Output: CriticResult with faithfulness, relevance, completeness, confidence (0–1). "
            "5. ReflectionAgent: maps failure type to corrective action (deterministic logic, no LLM). "
            "Output: ReflectionResult with root_cause, action (FixAction enum), priority. "
            "6. FixerAgent: transforms query/parameters for the next iteration. "
            "Output: FixerResult with modified_query and optional top_k_override."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "advanced"},
    },
    {
        "id": "neurorag-confidence",
        "text": (
            "The confidence score in NeuroRAG is computed by the Critic agent as a weighted sum: "
            "confidence = 0.5 × faithfulness + 0.3 × relevance + 0.2 × completeness. "
            "Faithfulness (weight 0.5): fraction of answer claims supported by retrieved context. "
            "Relevance (weight 0.3): degree to which the answer addresses the question asked. "
            "Completeness (weight 0.2): coverage of important aspects of the question. "
            "The confidence threshold (default 0.80) determines when to accept an answer vs retry. "
            "A confidence of 0.5 is returned for INSUFFICIENT_CONTEXT responses — the critic "
            "recognizes these as correct uncertainty expressions rather than failures. "
            "Confidence scores are stored in Postgres for trend analysis and fed to Prometheus "
            "for the Grafana Confidence Distribution dashboard panel."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "intermediate"},
    },
    {
        "id": "neurorag-memory",
        "text": (
            "NeuroRAG's memory layer has three components: "
            "QueryMemoryStore: semantic cache of past query-answer pairs. "
            "Uses Redis (TTL-based hot cache) and Postgres (durable cold storage). "
            "Semantic similarity via cosine distance on cached embeddings catches near-duplicate "
            "queries (e.g. 'what is RAG?' vs 'explain RAG') above a 0.92 similarity threshold. "
            "FailureMemory: tracks which fix actions resolved which failure types per query signature. "
            "Uses a PostgreSQL failure_patterns table with query_hash, failure_type, fix_action, success. "
            "Pre-recommends the historically successful fix action for known failing query patterns. "
            "AdaptiveContext: synthesizes memory signals into a ContextHint per query. "
            "Provides: similar_past_answer (cache hit), recommended_fix_action, top_k_boost, "
            "and prior_failure_types to bias retrieval and generation before the first loop."
        ),
        "metadata": {"source": "seed", "topic": "architecture", "difficulty": "advanced"},
    },
]

def main() -> None:
    import httpx

    base_url = os.getenv("NEURORAG_BASE_URL", "http://localhost:8000").rstrip("/")
    api_key = os.getenv("NEURORAG_API_KEY", "").strip()

    print("=" * 60)
    print(" NeuroRAG Knowledge Base Seeder ")
    print("=" * 60)
    print("Target URL :", base_url)
    print("Documents  :", len(SEED_DOCUMENTS))
    print("API Key    :", "Loaded" if api_key else "Not Set")
    print()

    headers = {
        "Content-Type": "application/json"
    }

    # only attach if exists
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=120.0,
        ) as client:

            print("Uploading documents...")

            response = client.post(
                "/ingest",
                json={"documents": SEED_DOCUMENTS},
                headers=headers,
            )

            # -------------------------------
            # Friendly Errors
            # -------------------------------
            if response.status_code == 401:
                print("\n❌ ERROR: Unauthorized (401)")
                print("Your API requires NEURORAG_API_KEY.")
                print()
                print("Fix options:")
                print("1. Add in .env file:")
                print("   NEURORAG_API_KEY=your_key_here")
                print()
                print("2. Or set env manually:")
                print("   set NEURORAG_API_KEY=your_key_here")
                sys.exit(1)

            elif response.status_code == 503:
                print("\n❌ ERROR: API not ready (503)")
                print("Wait for model loading to finish.")
                sys.exit(1)

            response.raise_for_status()

            data = response.json()

            print()
            print("✅ SUCCESS")
            print("-" * 40)
            print("Docs Seeded     :", data.get("doc_count"))
            print("Chunks Indexed  :", data.get("chunks_indexed"))
            print("Status          :", data.get("status"))
            print("-" * 40)
            print("Knowledge base now includes:")
            print("• RAG")
            print("• Embeddings")
            print("• Retrieval")
            print("• LLMs")
            print("• NLP")
            print("• MLOps")
            print("• FastAPI")
            print("• NeuroRAG internals")
            print()

    except httpx.ConnectError:
        print("\n❌ Cannot connect to NeuroRAG API.")
        print("Start server first:")
        print("uvicorn api.main:app --reload")
        sys.exit(1)

    except httpx.TimeoutException:
        print("\n❌ Request timed out.")
        print("Model may still be loading.")
        sys.exit(1)

    except Exception as exc:
        print(f"\n❌ Seed failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()