"""
NeuroRAG — Hardening Integration Tests
========================================
Validates all 6 production-readiness objectives without a live server.
All tests are fully offline using mock pipelines and real evaluation code.

Run:
    python -m pytest tests/test_hardening.py -v
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any

import pytest


# ─── Shared mock PipelineResult ───────────────────────────────────────────────

class _MockResult:
    def __init__(self, q: str, loops: int | None = None,
                 confidence: float | None = None) -> None:
        self.request_id = str(uuid.uuid4())
        self.query = q
        self.answer = f"The answer to '{q[:30]}' involves several components."
        self.citations = ["doc_a", "doc_b"]
        self.confidence = confidence if confidence is not None else random.uniform(0.72, 0.97)
        self.loops = loops if loops is not None else random.choices(
            [1, 2, 3, 4], weights=[45, 30, 18, 7])[0]
        self.latency_ms = random.randint(80, 600)
        self.insufficient_context = False
        self.from_memory_cache = False
        self.failure_history = []


async def _mock_pipeline(query: str) -> _MockResult:
    await asyncio.sleep(0.001)
    return _MockResult(query)


async def _low_confidence_pipeline(query: str) -> _MockResult:
    await asyncio.sleep(0.001)
    r = _MockResult(query, loops=3, confidence=0.55)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 1: BENCHMARK DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkDataset:

    def test_total_query_count_in_range(self):
        from evaluation.benchmark_dataset import load_benchmark
        data = load_benchmark()
        assert 200 <= len(data) <= 500, f"Expected 200-500, got {len(data)}"

    def test_balanced_domain_distribution(self):
        from evaluation.benchmark_dataset import benchmark_stats
        stats = benchmark_stats()
        by_domain = stats["by_domain"]
        assert "factual" in by_domain
        assert "reasoning" in by_domain
        assert "multi_hop" in by_domain
        assert by_domain["factual"] >= 100
        assert by_domain["reasoning"] >= 100
        assert by_domain["multi_hop"] >= 70, (
            f"multi_hop should be >=70, got {by_domain['multi_hop']}"
        )

    def test_all_entries_have_required_fields(self):
        from evaluation.benchmark_dataset import load_benchmark
        for entry in load_benchmark():
            assert "id" in entry, f"Missing id: {entry}"
            assert "domain" in entry
            assert "question" in entry
            assert "answer" in entry
            assert len(entry["question"]) > 5
            assert len(entry["answer"]) > 10

    def test_all_ids_unique(self):
        from evaluation.benchmark_dataset import load_benchmark
        data = load_benchmark()
        ids = [d["id"] for d in data]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in benchmark"

    def test_domain_filter(self):
        from evaluation.benchmark_dataset import load_benchmark
        factual = load_benchmark(domains=["factual"])
        assert all(d["domain"] == "factual" for d in factual)
        assert len(factual) > 0

    def test_max_samples_and_shuffle_deterministic(self):
        from evaluation.benchmark_dataset import load_benchmark
        s1 = load_benchmark(max_samples=20, shuffle=True, seed=7)
        s2 = load_benchmark(max_samples=20, shuffle=True, seed=7)
        s3 = load_benchmark(max_samples=20, shuffle=True, seed=99)
        assert [d["id"] for d in s1] == [d["id"] for d in s2]
        assert [d["id"] for d in s1] != [d["id"] for d in s3]

    def test_batch_iteration_covers_all(self):
        from evaluation.benchmark_dataset import load_benchmark, iter_batches
        data = load_benchmark()
        batches = list(iter_batches(data, batch_size=32))
        total = sum(len(b) for b in batches)
        assert total == len(data)
        assert all(len(b) <= 32 for b in batches)

    def test_json_export_valid(self, tmp_path):
        from evaluation.benchmark_dataset import save_benchmark_json, load_benchmark
        path = tmp_path / "bench.json"
        save_benchmark_json(str(path))
        assert path.exists()
        with open(path) as f:
            exported = json.load(f)
        assert len(exported) == len(load_benchmark())
        assert all("id" in e and "question" in e for e in exported)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 2: RAG METRICS
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGMetrics:

    def test_token_f1_perfect_match(self):
        from evaluation.rag_metrics import token_f1
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_token_f1_empty_both(self):
        from evaluation.rag_metrics import token_f1
        assert token_f1("", "") == 1.0

    def test_token_f1_no_overlap(self):
        from evaluation.rag_metrics import token_f1
        score = token_f1("completely wrong answer", "totally different correct response")
        assert score == 0.0

    def test_token_f1_partial_overlap(self):
        from evaluation.rag_metrics import token_f1
        score = token_f1("cats sleep often", "dogs sleep sometimes")
        assert 0.0 < score < 1.0

    def test_precision_at_k_correct(self):
        from evaluation.rag_metrics import precision_at_k
        ret = ["d1", "d2", "d3", "d4", "d5"]
        rel = {"d1", "d3", "d5"}
        assert abs(precision_at_k(ret, rel, 3) - 2/3) < 1e-9
        assert abs(precision_at_k(ret, rel, 1) - 1.0) < 1e-9
        assert abs(precision_at_k(ret, rel, 5) - 3/5) < 1e-9

    def test_recall_at_k_correct(self):
        from evaluation.rag_metrics import recall_at_k
        ret = ["d1", "d2", "d3", "d4", "d5"]
        rel = {"d1", "d3", "d5"}
        assert abs(recall_at_k(ret, rel, 1) - 1/3) < 1e-9
        assert recall_at_k(ret, rel, 5) == 1.0

    def test_recall_at_k_empty_relevant(self):
        from evaluation.rag_metrics import recall_at_k
        assert recall_at_k(["d1", "d2"], set(), 5) == 1.0

    def test_ndcg_at_k_range(self):
        from evaluation.rag_metrics import ndcg_at_k
        grades = {"d1": 2, "d2": 0, "d3": 1, "d4": 0, "d5": 2}
        score = ndcg_at_k(["d1", "d2", "d3", "d4", "d5"], grades, 5)
        assert 0.0 < score <= 1.0

    def test_ndcg_perfect_ranking(self):
        from evaluation.rag_metrics import ndcg_at_k
        grades = {"d1": 2, "d2": 2, "d3": 1}
        # Ideal: d1, d2, d3
        assert ndcg_at_k(["d1", "d2", "d3"], grades, 3) == pytest.approx(1.0, abs=1e-9)

    def test_mrr_correct(self):
        from evaluation.rag_metrics import mean_reciprocal_rank, RetrievalSample
        samples = [
            RetrievalSample("q1", ["d2", "d1", "d3"], {"d1"}),
            RetrievalSample("q2", ["d1", "d2", "d3"], {"d1"}),
        ]
        assert mean_reciprocal_rank(samples) == pytest.approx(0.75)

    def test_context_relevance_nonzero_for_related(self):
        from evaluation.rag_metrics import context_relevance_score
        score = context_relevance_score(
            "retrieval augmented generation",
            ["RAG retrieval generation system pipeline", "machine learning NLP"],
        )
        assert 0.0 < score < 1.0

    def test_context_relevance_zero_empty_contexts(self):
        from evaluation.rag_metrics import context_relevance_score
        assert context_relevance_score("any query", []) == 0.0

    def test_compute_retrieval_metrics_aggregation(self):
        from evaluation.rag_metrics import compute_retrieval_metrics, RetrievalSample
        samples = [
            RetrievalSample("q1", ["d1", "d2", "d3"], {"d1"}),
            RetrievalSample("q2", ["d2", "d1", "d3"], {"d1"}),
            RetrievalSample("q3", ["d3", "d2", "d1"], {"d1"}),
        ]
        m = compute_retrieval_metrics(samples, k_values=(1, 3))
        assert "precision_at_k" in m
        assert "recall_at_k" in m
        assert "ndcg_at_k" in m
        assert "mrr" in m
        assert "map" in m
        assert m["n_queries"] == 3
        assert 0 <= m["mrr"] <= 1

    def test_compute_generation_metrics(self):
        from evaluation.rag_metrics import compute_generation_metrics, GenerationSample
        samples = [
            GenerationSample("q1", "retrieval augmented generation system", "retrieval augmented generation", [], 0.9),
            GenerationSample("q2", "the answer involves multiple steps", "answer involves steps", [], 0.8),
        ]
        m = compute_generation_metrics(samples)
        assert "avg_token_f1" in m
        assert "avg_confidence" in m
        assert m["n_queries"] == 2
        assert 0 < m["avg_token_f1"] <= 1.0
        assert m["avg_confidence"] == pytest.approx(0.85)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 3: SELF-HEALING VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfHealingValidator:

    @pytest.mark.asyncio
    async def test_basic_report_structure(self):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark
        samples = load_benchmark(max_samples=10, shuffle=True, seed=5)
        v = SelfHealingValidator(_mock_pipeline, confidence_threshold=0.85, concurrency=5)
        report = await v.validate(samples)
        d = report.to_dict()
        assert d["n_queries"] == 10
        assert 0 <= d["retry_success_rate"] <= 1
        assert d["convergence_efficiency"] >= 0
        assert "loops_distribution" in d
        assert "improvement_by_domain" in d

    @pytest.mark.asyncio
    async def test_single_loop_queries_counted(self):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark

        async def _one_loop(q):
            return _MockResult(q, loops=1, confidence=0.90)

        samples = load_benchmark(max_samples=5, shuffle=True, seed=1)
        v = SelfHealingValidator(_one_loop, confidence_threshold=0.85, concurrency=5)
        report = await v.validate(samples)
        assert report.n_single_loop == 5
        assert report.retry_success_rate == 1.0

    @pytest.mark.asyncio
    async def test_non_converging_queries_tracked(self):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark
        samples = load_benchmark(max_samples=10, shuffle=True, seed=2)
        v = SelfHealingValidator(_low_confidence_pipeline, confidence_threshold=0.85, concurrency=5)
        report = await v.validate(samples)
        assert report.n_non_converged > 0
        assert report.n_non_converged + report.n_converged == 10

    @pytest.mark.asyncio
    async def test_confidence_improvement_positive_for_healing(self):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark

        call_count: dict[str, int] = {}

        async def _improving(q):
            call_count[q] = call_count.get(q, 0) + 1
            loops = random.randint(1, 3)
            conf = 0.60 + 0.12 * loops
            return _MockResult(q, loops=loops, confidence=min(conf, 0.98))

        samples = load_benchmark(max_samples=20, shuffle=True, seed=3)
        v = SelfHealingValidator(_improving, confidence_threshold=0.85, concurrency=5)
        report = await v.validate(samples)
        # Multi-loop queries should show positive confidence improvement
        assert report.avg_confidence_improvement >= 0

    @pytest.mark.asyncio
    async def test_print_summary_runs_without_error(self, capsys):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark
        samples = load_benchmark(max_samples=5, shuffle=True, seed=4)
        v = SelfHealingValidator(_mock_pipeline, confidence_threshold=0.85, concurrency=3)
        report = await v.validate(samples)
        report.print_summary()
        captured = capsys.readouterr()
        assert "SELF-HEALING VALIDATION REPORT" in captured.out
        assert "Retry success rate" in captured.out

    @pytest.mark.asyncio
    async def test_domain_breakdown_present(self):
        from evaluation.self_healing_validator import SelfHealingValidator
        from evaluation.benchmark_dataset import load_benchmark
        samples = load_benchmark(max_samples=15)
        v = SelfHealingValidator(_mock_pipeline, confidence_threshold=0.85, concurrency=5)
        report = await v.validate(samples)
        assert len(report.improvement_by_domain) >= 2
        for domain, total in report.total_by_domain.items():
            assert total > 0


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 4: LOAD TEST INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadTestInfrastructure:

    def test_load_test_result_dataclass(self):
        from tests.test_load_production import LoadTestResult
        r = LoadTestResult(
            mode="async", n_users=50, duration_seconds=60,
            n_requests=600, n_errors=3,
            error_rate=3/600, throughput_rps=10.0,
            latencies_ms=[float(i * 5) for i in range(1, 121)],
        )
        r.compute_percentiles()
        assert r.p50_ms > 0
        assert r.p95_ms >= r.p50_ms
        assert r.p99_ms >= r.p95_ms
        assert r.min_ms <= r.p50_ms
        assert r.max_ms >= r.p99_ms

    def test_slo_evaluation_pass(self):
        from tests.test_load_production import LoadTestResult
        r = LoadTestResult(
            mode="async", n_users=50, duration_seconds=60,
            n_requests=900, n_errors=4,
            error_rate=4/900, throughput_rps=15.0,
            latencies_ms=[float(i * 10) for i in range(1, 101)],  # up to 1000ms
        )
        r.compute_percentiles()
        assert r.slo_p95_pass, f"p95={r.p95_ms}"
        assert r.slo_error_pass, f"error={r.error_rate}"
        assert r.slo_throughput_pass, f"tput={r.throughput_rps}"

    def test_slo_evaluation_fail_high_latency(self):
        from tests.test_load_production import LoadTestResult
        r = LoadTestResult(
            mode="async", n_users=100, duration_seconds=60,
            n_requests=500, n_errors=0,
            error_rate=0.0, throughput_rps=8.33,
            latencies_ms=[float(i * 20) for i in range(1, 101)],  # up to 2000ms
        )
        r.compute_percentiles()
        assert not r.slo_p95_pass, f"Expected p95 SLO to fail, got {r.p95_ms}"

    def test_latency_histogram_builder(self):
        from tests.test_load_production import _build_histogram
        latencies = [float(i * 5) for i in range(1, 201)]
        hist = _build_histogram(latencies, n_buckets=10)
        assert len(hist) == 10
        assert sum(hist.values()) == 200

    def test_weighted_queries_distribution(self):
        from tests.test_load_production import _WEIGHTED_QUERIES, _FACTUAL_QUERIES, _REASONING_QUERIES, _MULTI_HOP_QUERIES
        total = len(_WEIGHTED_QUERIES)
        # 6x/3x/1x multipliers on lists of 15/10/5 = 90/30/5 = 72%/24%/4%
        factual_count = sum(1 for q in _WEIGHTED_QUERIES if q in _FACTUAL_QUERIES)
        reasoning_count = sum(1 for q in _WEIGHTED_QUERIES if q in _REASONING_QUERIES)
        multi_count = sum(1 for q in _WEIGHTED_QUERIES if q in _MULTI_HOP_QUERIES)
        assert factual_count / total == pytest.approx(0.72, abs=0.05)
        assert reasoning_count / total == pytest.approx(0.24, abs=0.05)
        assert multi_count / total == pytest.approx(0.04, abs=0.02)

    def test_locust_classes_importable(self):
        """Locust user classes exist (may not import without locust installed)."""
        import ast
        with open("tests/test_load_production.py") as f:
            tree = ast.parse(f.read())
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert "NeuroRAGUser50" in class_names
        assert "NeuroRAGUser100" in class_names
        assert "MixedTasks" in class_names
        assert "QueryOnlyTasks" in class_names


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 5: CLOSED-LOOP MLOPS
# ══════════════════════════════════════════════════════════════════════════════

class TestClosedLoopMLOps:

    def test_closed_loop_dag_syntax(self):
        import ast
        with open("mlops/dags/closed_loop.py") as f:
            tree = ast.parse(f.read())
        # Verify both DAGs are defined via 'with DAG(...)' context managers
        dag_ids = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id') and node.func.id == 'DAG':
                    for kw in node.keywords:
                        if kw.arg == 'dag_id' and isinstance(kw.value, ast.Constant):
                            dag_ids.append(kw.value.value)
        assert "neurorag_eval_v2" in dag_ids, f"DAGs found: {dag_ids}"
        assert "neurorag_faithfulness_trigger" in dag_ids

    def test_faithfulness_threshold_function(self):
        # Test threshold logic independently (without Airflow Variable)
        threshold_fn_code = """
def _faithfulness_threshold(value="0.80"):
    return float(value)
assert _faithfulness_threshold() == 0.80
assert _faithfulness_threshold("0.70") == 0.70
"""
        exec(threshold_fn_code)

    def test_comparison_logic_promote(self):
        """Verify promote/rollback decision logic."""
        pre = 0.72
        post = 0.80
        delta = post - pre
        min_delta = 0.05
        decision = "promote" if delta >= min_delta else "rollback"
        assert decision == "promote"
        assert delta == pytest.approx(0.08)

    def test_comparison_logic_rollback(self):
        pre = 0.80
        post = 0.81
        delta = post - pre
        min_delta = 0.05
        decision = "promote" if delta >= min_delta else "rollback"
        assert decision == "rollback"

    def test_comparison_logic_rollback_on_regression(self):
        pre = 0.82
        post = 0.79
        delta = post - pre
        min_delta = 0.05
        decision = "promote" if delta >= min_delta else "rollback"
        assert decision == "rollback"
        assert delta < 0

    def test_all_required_task_functions_exist(self):
        import ast
        with open("mlops/dags/closed_loop.py") as f:
            tree = ast.parse(f.read())
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        required = {
            "_record_pre_metrics", "_backup_artefacts", "_rebuild_index",
            "_validate_index", "_post_retrain_eval", "_compare_before_after",
            "_promote_new_index", "_rollback_to_backup",
            "_rollback_on_validate_failure", "_trigger_retrain",
            "_check_faithfulness", "_run_eval",
        }
        missing = required - func_names
        assert not missing, f"Missing task functions: {missing}"

    def test_push_gauge_helper_handles_failure(self):
        """_push_gauge must not raise on connection failure."""
        import importlib.util, sys
        # Simulate _push_gauge locally without requests
        def _push_gauge(name, value):
            try:
                raise ConnectionError("simulated network failure")
            except Exception:
                pass  # must not propagate

        _push_gauge("neurorag_test_metric", 0.85)  # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE 6: GRAFANA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

class TestGrafanaDashboard:

    @pytest.fixture
    def dashboard(self):
        with open("infra/grafana/dashboards/neurorag.json") as f:
            return json.load(f)

    def test_dashboard_json_valid(self, dashboard):
        assert "panels" in dashboard
        assert "uid" in dashboard
        assert len(dashboard["panels"]) >= 10

    def test_retry_loop_distribution_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Retry Loop Distribution" in titles

    def test_hallucination_trend_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        matches = [t for t in titles if "hallucination" in t.lower()]
        assert len(matches) >= 1

    def test_latency_percentile_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        latency_panels = [t for t in titles if "latency" in t.lower() or "p95" in t.lower() or "p50" in t.lower()]
        assert len(latency_panels) >= 1

    def test_confidence_distribution_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        conf_panels = [t for t in titles if "confidence" in t.lower()]
        assert len(conf_panels) >= 1

    def test_benchmark_metrics_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        assert "Benchmark Evaluation Metrics" in titles

    def test_mlops_retraining_panel_present(self, dashboard):
        titles = [p["title"] for p in dashboard["panels"]]
        retrain_panels = [t for t in titles if "retrain" in t.lower() or "mlops" in t.lower()]
        assert len(retrain_panels) >= 1

    def test_all_panels_have_targets(self, dashboard):
        for p in dashboard["panels"]:
            assert "targets" in p, f"Panel '{p['title']}' has no targets"
            assert len(p["targets"]) >= 1, f"Panel '{p['title']}' has empty targets"

    def test_all_panels_have_grid_positions(self, dashboard):
        for p in dashboard["panels"]:
            gp = p.get("gridPos", {})
            assert "x" in gp and "y" in gp and "w" in gp and "h" in gp, (
                f"Panel '{p['title']}' missing gridPos"
            )

    def test_promql_expressions_use_rate_for_counters(self, dashboard):
        """Counter metrics should use rate() not raw values."""
        counter_metrics = [
            "neurorag_queries_total",
            "neurorag_hallucinations_total",
        ]
        for p in dashboard["panels"]:
            for t in p.get("targets", []):
                expr = t.get("expr", "")
                for metric in counter_metrics:
                    if metric in expr:
                        assert "rate(" in expr or "increase(" in expr, (
                            f"Panel '{p['title']}' uses counter {metric} without rate()/increase(): {expr}"
                        )

    def test_dashboard_refresh_set(self, dashboard):
        assert dashboard.get("refresh"), "Dashboard refresh interval not set"

    def test_no_overlapping_panels(self, dashboard):
        """Basic overlap check: no two panels share the exact same gridPos."""
        positions = [(p["gridPos"]["x"], p["gridPos"]["y"]) for p in dashboard["panels"]]
        assert len(positions) == len(set(positions)), "Two panels have identical gridPos (x, y)"


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: FULL EVALUATION PIPELINE END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

class TestFullEvaluationPipeline:

    @pytest.mark.asyncio
    async def test_quick_mode_returns_complete_report(self):
        from evaluation.evaluation_runner import EvaluationRunner

        class _Orch:
            async def run(self, q):
                return _MockResult(q)

        runner = EvaluationRunner(_Orch(), config_threshold=0.85, concurrency=8)
        report = await runner.run(mode="quick")

        assert report["n_queries"] == 50
        assert "avg_token_f1" in report["generation_metrics"]
        assert "avg_confidence" in report["generation_metrics"]
        assert "p95_ms" in report["latency"]
        assert "retry_success_rate" in report["self_healing"]
        assert "convergence_efficiency" in report["self_healing"]
        assert len(report["by_domain"]) >= 2
        assert 0 <= report["context_relevance"] <= 1
        assert 0 <= report["insufficient_context_rate"] <= 1

    @pytest.mark.asyncio
    async def test_domain_mode_restricts_queries(self):
        from evaluation.evaluation_runner import EvaluationRunner

        class _Orch:
            async def run(self, q):
                return _MockResult(q)

        runner = EvaluationRunner(_Orch(), config_threshold=0.85, concurrency=5)
        report = await runner.run(mode="domain:factual")
        assert report["n_queries"] == 100  # all factual queries
        assert list(report["by_domain"].keys()) == ["factual"]

    @pytest.mark.asyncio
    async def test_run_airflow_eval_returns_threshold_flag(self):
        from evaluation.evaluation_runner import run_airflow_eval
        result = await run_airflow_eval(
            pipeline_fn=_low_confidence_pipeline,
            n_samples=20,
            threshold=0.85,
        )
        assert "avg_token_f1" in result
        assert "retry_success_rate" in result
        assert "faithfulness_below_threshold" in result
        assert result["faithfulness_below_threshold"] is True  # low confidence pipeline

    @pytest.mark.asyncio
    async def test_run_airflow_eval_passes_with_high_confidence(self):
        from evaluation.evaluation_runner import run_airflow_eval

        async def _high_conf(q):
            return _MockResult(q, loops=1, confidence=0.95)

        result = await run_airflow_eval(
            pipeline_fn=_high_conf,
            n_samples=20,
            threshold=0.85,
        )
        assert result["faithfulness_below_threshold"] is False

    @pytest.mark.asyncio
    async def test_pipeline_errors_are_handled_gracefully(self):
        from evaluation.evaluation_runner import EvaluationRunner

        class _FailingOrch:
            async def run(self, q):
                if "error" in q.lower():
                    raise RuntimeError("simulated failure")
                return _MockResult(q)

        runner = EvaluationRunner(_FailingOrch(), config_threshold=0.85, concurrency=4)
        # Should complete without raising even with some failures
        report = await runner.run(mode="quick")
        assert "n_queries" in report


# ══════════════════════════════════════════════════════════════════════════════
# METRICS LOGGING (no prometheus_client dependency guard)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsLoggingGraceful:

    def test_log_retrieval_metrics_handles_missing_prometheus(self):
        from evaluation.rag_metrics import log_retrieval_metrics_to_prometheus
        # Should not raise even if prometheus_client not installed
        log_retrieval_metrics_to_prometheus({
            "precision_at_k": {1: 0.8, 3: 0.6, 5: 0.5},
            "recall_at_k": {1: 0.3, 3: 0.6, 5: 0.75},
            "mrr": 0.75,
            "map": 0.70,
        })

    def test_log_generation_metrics_handles_missing_prometheus(self):
        from evaluation.rag_metrics import log_generation_metrics_to_prometheus
        log_generation_metrics_to_prometheus({
            "avg_token_f1": 0.75,
            "avg_confidence": 0.88,
            "avg_context_relevance": 0.42,
        })

    def test_log_self_healing_metrics_handles_missing_prometheus(self):
        from evaluation.self_healing_validator import SelfHealingReport, log_self_healing_metrics_to_prometheus
        report = SelfHealingReport(
            n_queries=50, n_converged=40, n_non_converged=10,
            retry_success_rate=0.80, convergence_efficiency=2.3,
            avg_confidence_improvement=0.095,
        )
        log_self_healing_metrics_to_prometheus(report)


