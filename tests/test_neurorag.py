"""
NeuroRAG — Test Suite
Unit tests for all agents and RAG components.
Integration tests for the full pipeline.
Run with: pytest tests/ -v --tb=short
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.critic import Critic
from agents.generator import Generator
from agents.intent_analyzer import IntentAnalyzer
from agents.orchestrator import NeuroRAGOrchestrator
from agents.planner import Planner
from agents.reflection_fixer import FixerAgent, ReflectionAgent
from agents.schemas import (
    CriticResult,
    Document,
    FailureType,
    FixAction,
    IntentResult,
    QueryType,
    ReflectionResult,
)
from rag.ingest import semantic_chunk
from rag.retriever import _rrf_score

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock LLM client that returns configurable responses."""
    client = MagicMock()
    client.complete = MagicMock()
    async def acomplete(prompt, system="", temperature=0.1, max_tokens=1024):
        return client.complete(prompt, system, temperature, max_tokens)
    client.acomplete = acomplete
    return client


@pytest.fixture
def mock_docs():
    return [
        Document(doc_id="doc1", chunk_id="0", text="Paris is the capital of France.", score=0.9),
        Document(doc_id="doc2", chunk_id="0", text="France is a country in Western Europe.", score=0.85),
        Document(doc_id="doc3", chunk_id="0", text="The Eiffel Tower is located in Paris.", score=0.7),
    ]


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Chunking
# ════════════════════════════════════════════════════════════════════════════

class TestSemanticChunk:

    def test_basic_split(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = semantic_chunk(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) > 0

    def test_empty_text(self):
        chunks = semantic_chunk("", chunk_size=512, overlap=64)
        assert chunks == []

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = semantic_chunk(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_applied(self):
        text = "First sentence. " * 20 + "Second sentence."
        chunks = semantic_chunk(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Intent Analyzer
# ════════════════════════════════════════════════════════════════════════════

class TestIntentAnalyzer:

    def test_factual_intent(self, mock_llm):
        mock_llm.complete.return_value = json.dumps({
            "query_type": "factual",
            "complexity": 0.2,
            "reasoning": "Single fact lookup",
        })
        analyzer = IntentAnalyzer(mock_llm)
        result = analyzer.analyze("What is the capital of France?")
        assert result.query_type == QueryType.FACTUAL
        assert result.complexity == pytest.approx(0.2)

    def test_empty_query_defaults_ambiguous(self, mock_llm):
        analyzer = IntentAnalyzer(mock_llm)
        result = analyzer.analyze("")
        assert result.query_type == QueryType.AMBIGUOUS

    def test_llm_failure_fallback(self, mock_llm):
        mock_llm.complete.side_effect = RuntimeError("LLM unavailable")
        analyzer = IntentAnalyzer(mock_llm)
        result = analyzer.analyze("Some query")
        assert result.query_type == QueryType.FACTUAL  # fallback

    def test_malformed_json_fallback(self, mock_llm):
        mock_llm.complete.return_value = "NOT JSON"
        analyzer = IntentAnalyzer(mock_llm)
        result = analyzer.analyze("Some query")
        assert result.query_type == QueryType.FACTUAL


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Planner
# ════════════════════════════════════════════════════════════════════════════

class TestPlanner:

    def test_single_query_factual(self, mock_llm):
        mock_llm.complete.return_value = json.dumps({
            "sub_queries": ["capital of France"],
            "strategy": "hybrid",
        })
        planner = Planner(mock_llm)
        intent = IntentResult(query_type=QueryType.FACTUAL, complexity=0.1)
        result = planner.plan("What is the capital of France?", intent)
        assert len(result.sub_queries) == 1
        assert result.strategy == "hybrid"

    def test_multi_hop_decomposition(self, mock_llm):
        mock_llm.complete.return_value = json.dumps({
            "sub_queries": ["author of Hamlet", "Hamlet publication date", "Shakespeare biography"],
            "strategy": "hybrid",
        })
        planner = Planner(mock_llm)
        intent = IntentResult(query_type=QueryType.MULTI_HOP, complexity=0.8)
        result = planner.plan("Who wrote Hamlet and when was it published?", intent)
        assert len(result.sub_queries) == 3

    def test_llm_failure_returns_original(self, mock_llm):
        mock_llm.complete.side_effect = Exception("timeout")
        planner = Planner(mock_llm)
        intent = IntentResult(query_type=QueryType.FACTUAL, complexity=0.1)
        result = planner.plan("original query", intent)
        assert result.sub_queries == ["original query"]


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Generator
# ════════════════════════════════════════════════════════════════════════════

class TestGenerator:

    def test_grounded_answer(self, mock_llm, mock_docs):
        mock_llm.complete.return_value = json.dumps({
            "answer": "Paris is the capital of France. [doc1#0]",
            "citations": ["doc1#0"],
        })
        gen = Generator(mock_llm)
        result = gen.generate("What is the capital of France?", mock_docs)
        assert "Paris" in result.answer
        assert "doc1#0" in result.citations

    def test_insufficient_context_no_docs(self, mock_llm):
        gen = Generator(mock_llm)
        result = gen.generate("Some query", [])
        assert result.answer == "INSUFFICIENT_CONTEXT"
        assert result.citations == []

    def test_llm_failure_returns_insufficient(self, mock_llm, mock_docs):
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        gen = Generator(mock_llm)
        result = gen.generate("query", mock_docs)
        assert result.answer == "INSUFFICIENT_CONTEXT"


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Critic
# ════════════════════════════════════════════════════════════════════════════

class TestCritic:

    def test_high_confidence_grounded_answer(self, mock_llm, mock_docs):
        mock_llm.complete.return_value = json.dumps({
            "faithfulness": 0.97,
            "relevance": 0.95,
            "completeness": 0.90,
            "confidence": 0.95,
            "failure_type": "none",
            "hallucination_detected": False,
            "notes": "",
        })
        critic = Critic(mock_llm)
        result = critic.evaluate(
            "What is the capital of France?",
            "Paris is the capital of France. [doc1#0]",
            mock_docs,
        )
        assert result.confidence >= 0.90
        assert not result.hallucination_detected
        assert result.failure_type == FailureType.NONE

    def test_insufficient_context_special_case(self, mock_llm, mock_docs):
        critic = Critic(mock_llm)
        result = critic.evaluate("query", "INSUFFICIENT_CONTEXT", mock_docs)
        assert result.failure_type == FailureType.MISSING_CONTEXT
        assert not result.hallucination_detected
        assert result.faithfulness == 1.0   # Correct to flag missing context

    def test_hallucination_detected(self, mock_llm, mock_docs):
        mock_llm.complete.return_value = json.dumps({
            "faithfulness": 0.1,
            "relevance": 0.6,
            "completeness": 0.5,
            "confidence": 0.25,
            "failure_type": "hallucination",
            "hallucination_detected": True,
            "notes": "Claim not supported by any context passage.",
        })
        critic = Critic(mock_llm)
        result = critic.evaluate("query", "Fabricated answer with invented facts.", mock_docs)
        assert result.hallucination_detected
        assert result.confidence < 0.50
        assert result.failure_type == FailureType.HALLUCINATION


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Reflection & Fixer
# ════════════════════════════════════════════════════════════════════════════

class TestReflection:

    def test_hallucination_maps_to_add_context(self):
        agent = ReflectionAgent()
        critique = CriticResult(
            faithfulness=0.1, relevance=0.6, completeness=0.5, confidence=0.25,
            failure_type=FailureType.HALLUCINATION, hallucination_detected=True,
        )
        result = agent.analyze(critique, iteration=0)
        assert result.action == FixAction.ADD_CONTEXT
        assert result.priority == 5

    def test_missing_context_first_iter_broadens(self):
        agent = ReflectionAgent()
        critique = CriticResult(
            faithfulness=1.0, relevance=0.5, completeness=0.0, confidence=0.5,
            failure_type=FailureType.MISSING_CONTEXT,
        )
        result = agent.analyze(critique, iteration=0)
        assert result.action == FixAction.BROADEN_QUERY

    def test_missing_context_second_iter_increases_topk(self):
        agent = ReflectionAgent()
        critique = CriticResult(
            faithfulness=1.0, relevance=0.5, completeness=0.0, confidence=0.5,
            failure_type=FailureType.MISSING_CONTEXT,
        )
        result = agent.analyze(critique, iteration=1)
        assert result.action == FixAction.INCREASE_TOP_K

    def test_irrelevance_narrows_query(self):
        agent = ReflectionAgent()
        critique = CriticResult(
            faithfulness=0.8, relevance=0.2, completeness=0.5, confidence=0.4,
            failure_type=FailureType.IRRELEVANCE,
        )
        result = agent.analyze(critique, iteration=0)
        assert result.action == FixAction.NARROW_QUERY


class TestFixer:

    def test_broaden_appends_terms(self):
        fixer = FixerAgent()
        reflection = ReflectionResult(
            root_cause="narrow", action=FixAction.BROADEN_QUERY, details=""
        )
        result = fixer.apply("capital of France", reflection, iteration=0)
        assert "capital of France" in result.modified_query
        assert len(result.modified_query) > len("capital of France")

    def test_add_context_increases_topk(self):
        fixer = FixerAgent()
        reflection = ReflectionResult(
            root_cause="hallucination", action=FixAction.ADD_CONTEXT, details=""
        )
        result = fixer.apply("query", reflection, iteration=0)
        assert result.retrieval_top_k_override is not None
        assert result.retrieval_top_k_override > 0

    def test_no_action_passthrough(self):
        fixer = FixerAgent()
        reflection = ReflectionResult(
            root_cause="", action=FixAction.NONE, details=""
        )
        result = fixer.apply("original query", reflection, iteration=0)
        assert result.modified_query == "original query"
        assert result.retrieval_top_k_override is None


# ════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — RRF Score
# ════════════════════════════════════════════════════════════════════════════

class TestRRF:

    def test_rrf_rank_0_highest(self):
        assert _rrf_score(0) > _rrf_score(1) > _rrf_score(10)

    def test_rrf_k_effect(self):
        # Higher k → more uniform scores
        assert abs(_rrf_score(0, k=1) - _rrf_score(10, k=1)) > abs(_rrf_score(0, k=100) - _rrf_score(10, k=100))


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Full Pipeline (mocked LLM + in-memory retrieval)
# ════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:

    @pytest.fixture
    def mock_engine(self, mock_docs):
        """Minimal mock of IngestionEngine for integration tests."""
        engine = MagicMock()
        engine.doc_ids = [d.uid for d in mock_docs]
        engine.faiss_index = MagicMock()
        engine.whoosh_ix = MagicMock()
        engine.embedder = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_successful_pipeline_no_retry(self, mock_llm, mock_engine):
        """Happy path: first-pass answer meets confidence threshold."""
        # Intent
        mock_llm.complete.side_effect = [
            # Intent analyzer
            json.dumps({"query_type": "factual", "complexity": 0.2, "reasoning": ""}),
            # Planner
            json.dumps({"sub_queries": ["capital of France"], "strategy": "hybrid"}),
            # Generator
            json.dumps({"answer": "Paris is the capital of France.", "citations": ["doc1#0"]}),
        ]

        mock_critic_llm = MagicMock()
        mock_critic_llm.complete.return_value = json.dumps({
            "faithfulness": 0.97, "relevance": 0.95, "completeness": 0.90,
            "confidence": 0.95, "failure_type": "none",
            "hallucination_detected": False, "notes": "",
        })
        async def critic_acomplete(prompt, system="", temperature=0.1, max_tokens=1024):
            return mock_critic_llm.complete(prompt, system, temperature, max_tokens)
        mock_critic_llm.acomplete = critic_acomplete

        with patch("agents.orchestrator.HybridRetriever") as mock_retriever_class, \
             patch("agents.orchestrator.Reranker") as mock_reranker_class, \
             patch("agents.orchestrator.Evaluator"):

            mock_retriever = mock_retriever_class.return_value
            mock_retriever.retrieve_async = AsyncMock(return_value=[
                Document(doc_id="doc1", chunk_id="0", text="Paris is the capital of France.", score=0.9)
            ])

            mock_reranker = mock_reranker_class.return_value
            mock_reranker.rerank.return_value = [
                Document(doc_id="doc1", chunk_id="0", text="Paris is the capital of France.", score=0.9)
            ]

            orchestrator = NeuroRAGOrchestrator(mock_engine, mock_llm, mock_critic_llm)
            result = await orchestrator.run("What is the capital of France?")

        assert "Paris" in result.answer
        assert result.loops == 1
        assert result.confidence >= 0.90
        assert not result.insufficient_context

    @pytest.mark.asyncio
    async def test_insufficient_context_flagged(self, mock_llm, mock_engine):
        """System correctly flags queries with no supporting documents."""
        mock_llm.complete.side_effect = [
            json.dumps({"query_type": "factual", "complexity": 0.2, "reasoning": ""}),
            json.dumps({"sub_queries": ["unanswerable query"], "strategy": "hybrid"}),
            json.dumps({"answer": "INSUFFICIENT_CONTEXT", "citations": []}),
        ]

        mock_critic_llm = MagicMock()
        mock_critic_llm.complete.return_value = json.dumps({
            "faithfulness": 1.0, "relevance": 0.5, "completeness": 0.0,
            "confidence": 0.5, "failure_type": "missing_context",
            "hallucination_detected": False, "notes": "No supporting context.",
        })
        async def critic_acomplete(prompt, system="", temperature=0.1, max_tokens=1024):
            return mock_critic_llm.complete(prompt, system, temperature, max_tokens)
        mock_critic_llm.acomplete = critic_acomplete

        with patch("agents.orchestrator.HybridRetriever") as mock_retriever_class, \
             patch("agents.orchestrator.Reranker") as mock_reranker_class, \
             patch("agents.orchestrator.Evaluator"):

            mock_retriever = mock_retriever_class.return_value
            mock_retriever.retrieve_async = AsyncMock(return_value=[])
            mock_reranker = mock_reranker_class.return_value
            mock_reranker.rerank.return_value = []

            orchestrator = NeuroRAGOrchestrator(mock_engine, mock_llm, mock_critic_llm)
            result = await orchestrator.run("Tell me about dragons ruling medieval France.")

        assert result.insufficient_context or result.answer == "INSUFFICIENT_CONTEXT"
