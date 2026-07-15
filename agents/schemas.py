"""
NeuroRAG — Shared domain models.
All inter-agent data flows through these typed schemas.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ─── Enumerations ────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"


class FailureType(str, Enum):
    NONE = "none"
    HALLUCINATION = "hallucination"
    MISSING_CONTEXT = "missing_context"
    IRRELEVANCE = "irrelevance"
    INCOMPLETE = "incomplete"
    OTHER = "other"


class FixAction(str, Enum):
    NONE = "none"
    ADD_CONTEXT = "add_context"
    BROADEN_QUERY = "broaden_query"
    NARROW_QUERY = "narrow_query"
    REFINE_PROMPT = "refine_prompt"
    INCREASE_TOP_K = "increase_top_k"


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking calls — too many consecutive failures
    HALF_OPEN = "half_open" # Probing — allow one call through


# ─── Document / Retrieval ────────────────────────────────────────────────────

class Document(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"


# ─── Agent I/O Models ────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    query_type: QueryType
    complexity: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class PlanResult(BaseModel):
    sub_queries: list[str]
    strategy: str = "hybrid"
    notes: str = ""


class RetrievalResult(BaseModel):
    documents: list[Document]
    query_used: str
    bm25_count: int = 0
    vector_count: int = 0


class GeneratorResult(BaseModel):
    answer: str
    citations: list[str]
    raw_context_used: str = ""


class CriticResult(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    failure_type: FailureType = FailureType.NONE
    hallucination_detected: bool = False
    notes: str = ""


class ReflectionResult(BaseModel):
    root_cause: str
    action: FixAction
    details: str
    priority: int = Field(ge=1, le=5, default=3)


class FixerResult(BaseModel):
    modified_query: str
    retrieval_top_k_override: int | None = None
    prompt_hint: str = ""


# ─── Memory Schemas ──────────────────────────────────────────────────────────

class ContextHintSchema(BaseModel):
    """Serialisable form of AdaptiveContext.ContextHint."""
    similar_past_answer: str | None = None
    recommended_fix_action: FixAction | None = None
    recommended_top_k_boost: int = 0
    prior_failure_types: list[str] = Field(default_factory=list)
    confidence_floor: float = 0.0
    from_cache: bool = False


class FailurePatternSchema(BaseModel):
    query_hash: str
    failure_type: str
    fix_action: str
    success_after_fix: bool
    count: int = 1


# ─── Pipeline Result ─────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    citations: list[str]
    confidence: float
    loops: int
    latency_ms: int
    failure_history: list[FailureType] = Field(default_factory=list)
    insufficient_context: bool = False
    from_memory_cache: bool = False
    context_hint_applied: bool = False
