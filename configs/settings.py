"""
NeuroRAG — Config Loader
Reads configs/config.yaml and exposes typed Pydantic settings.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    provider: Literal["local", "openai", "gemini"]
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    device: str = "cuda"


class EmbeddingConfig(BaseModel):
    model: str
    batch_size: int
    device: str
    normalize: bool
    dimension: int


class RetrievalConfig(BaseModel):
    top_k: int
    hybrid: bool
    bm25_weight: float
    vector_weight: float
    chunk_size: int
    chunk_overlap: int
    max_context_tokens: int


class RerankerConfig(BaseModel):
    model: str
    top_k: int
    device: str


class SelfHealConfig(BaseModel):
    max_loops: int
    confidence_threshold: float
    failure_cooldown_ms: int


class VectorStoreConfig(BaseModel):
    backend: Literal["faiss", "pinecone"]
    index_path: str
    ids_path: str
    metric: str


class BM25Config(BaseModel):
    index_path: str


class DatabaseConfig(BaseModel):
    postgres_url: str
    pool_size: int = 10
    max_overflow: int = 20


class CacheConfig(BaseModel):
    redis_url: str
    ttl_seconds: int = 3600
    max_size_mb: int = 512


class MonitoringConfig(BaseModel):
    prometheus_port: int
    metrics_path: str
    enable_tracing: bool
    jaeger_host: str
    jaeger_port: int


class APIConfig(BaseModel):
    host: str
    port: int
    workers: int
    timeout: int
    cors_origins: list[str]
    rate_limit_per_minute: int


class SecurityConfig(BaseModel):
    pii_filtering: bool = True
    max_query_length: int = 2000
    api_key_required: bool = True
    api_key_env_var: str = "NEURORAG_API_KEY"


class NeuroRAGConfig(BaseModel):
    # Ignore the top-level `system:` block and any future unknown keys
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig
    critic_llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    self_heal: SelfHealConfig
    vector_store: VectorStoreConfig
    bm25: BM25Config
    database: DatabaseConfig
    cache: CacheConfig
    monitoring: MonitoringConfig
    api: APIConfig
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def _resolve_env(obj: dict) -> dict:
    from typing import Any
    resolved: dict[Any, Any] = {}
    for k, v in obj.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_key = v[2:-1]
            resolved[k] = os.environ.get(env_key, v)
        elif isinstance(v, dict):
            resolved[k] = _resolve_env(v)
        else:
            resolved[k] = v
    return resolved


@lru_cache(maxsize=1)
def get_config() -> NeuroRAGConfig:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw = _resolve_env(raw)
    return NeuroRAGConfig(**raw)
