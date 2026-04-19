"""
NeuroRAG — Evaluation Runner v1
=================================
Full benchmark evaluation pipeline integrating:
  - benchmark_dataset.py  (300-query benchmark)
  - rag_metrics.py        (Precision@K, Recall@K, context relevance, F1)
  - self_healing_validator.py (retry rate, convergence, improvement)

Entry points:
  1. run_full_benchmark(orchestrator)   → comprehensive report
  2. run_quick_eval(orchestrator)       → 50-query smoke eval for CI
  3. Airflow-compatible async function  → called from neurorag_eval DAG

Run standalone:
  python -m evaluation.evaluation_runner --mode full --output eval_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# CORE RUNNER
# ────────────────────────────────────────────────────────────────────────────

class EvaluationRunner:
    """
    End-to-end evaluation pipeline for NeuroRAG.

    Usage:
        runner = EvaluationRunner(orchestrator=orch, config_threshold=0.85)
        report = await runner.run(mode="full")
        print(json.dumps(report, indent=2))
    """

    def __init__(
        self,
        orchestrator: Any,
        config_threshold: float = 0.85,
        k_values: tuple[int, ...] = (1, 3, 5, 10),
        concurrency: int = 4,
        use_semantic_relevance: bool = False,
    ) -> None:
        self._orchestrator = orchestrator
        self._threshold = config_threshold
        self._k_values = k_values
        self._concurrency = concurrency
        self._use_semantic = use_semantic_relevance

    async def run(self, mode: str = "full") -> dict:
        """
        Run benchmark evaluation.

        Args:
            mode: 'full' (all 300 queries), 'quick' (50 queries), or 'domain:<name>'
        """
        from evaluation.benchmark_dataset import load_benchmark
        from evaluation.rag_metrics import (
            GenerationSample,
            compute_generation_metrics,
            log_generation_metrics_to_prometheus,
            context_relevance_score,
        )
        from evaluation.self_healing_validator import (
            SelfHealingValidator,
            log_self_healing_metrics_to_prometheus,
        )

        # ── 1. Load dataset ────────────────────────────────────────────────
        if mode == "full":
            samples = load_benchmark()
        elif mode == "quick":
            samples = load_benchmark(max_samples=50, shuffle=True, seed=99)
        elif mode.startswith("domain:"):
            domain = mode.split(":", 1)[1]
            samples = load_benchmark(domains=[domain])
        else:
            samples = load_benchmark(max_samples=50, shuffle=True, seed=99)

        logger.info("EvaluationRunner: loaded %d samples (mode=%s)", len(samples), mode)

        # ── 2. Run queries through pipeline ───────────────────────────────
        t_start = time.monotonic()
        results = await self._run_pipeline(samples)
        elapsed = time.monotonic() - t_start
        logger.info(
            "Pipeline completed %d queries in %.1fs (%.1f q/s)",
            len(results), elapsed, len(results) / max(elapsed, 0.001)
        )

        # ── 3. Generation metrics ─────────────────────────────────────────
        gen_samples = [
            GenerationSample(
                query=samples[i]["question"],
                generated_answer=results[i].answer if results[i] else "",
                ground_truth_answer=samples[i]["answer"],
                retrieved_contexts=[],       # context not stored in PipelineResult
                confidence=results[i].confidence if results[i] else 0.0,
            )
            for i in range(len(samples))
        ]
        gen_metrics = compute_generation_metrics(gen_samples)
        log_generation_metrics_to_prometheus(gen_metrics)

        # ── 4. Context relevance (from cached pipeline contexts) ──────────
        ctx_relevance_scores = []
        for i, sample in enumerate(samples):
            if results[i]:
                score = context_relevance_score(
                    query=sample["question"],
                    contexts=results[i].citations or [],
                )
                ctx_relevance_scores.append(score)
        avg_ctx_relevance = (
            sum(ctx_relevance_scores) / len(ctx_relevance_scores)
            if ctx_relevance_scores else 0.0
        )

        # ── 5. Self-healing validation ────────────────────────────────────
        validator = SelfHealingValidator(
            pipeline_fn=self._orchestrator.run,
            confidence_threshold=self._threshold,
            concurrency=self._concurrency,
        )
        # Use subset for validation to avoid double-running full benchmark
        validation_samples = samples[:min(100, len(samples))]
        healing_report = await validator.validate(validation_samples)
        log_self_healing_metrics_to_prometheus(healing_report)
        healing_report.print_summary()

        # ── 6. Latency stats ──────────────────────────────────────────────
        latencies = [r.latency_ms for r in results if r is not None]
        latency_stats = self._latency_percentiles(latencies)

        # ── 7. Loop statistics ────────────────────────────────────────────
        loops_list = [r.loops for r in results if r is not None]
        loop_stats = {
            "avg_loops": round(sum(loops_list) / len(loops_list), 3) if loops_list else 0,
            "max_loops": max(loops_list) if loops_list else 0,
            "loops_distribution": {
                str(k): loops_list.count(k)
                for k in sorted(set(loops_list))
            },
        }

        # ── 8. Insufficient context rate ─────────────────────────────────
        n_insufficient = sum(
            1 for r in results if r is not None and r.insufficient_context
        )

        # ── 9. Assemble final report ──────────────────────────────────────
        report = {
            "mode": mode,
            "n_queries": len(samples),
            "evaluation_time_s": round(elapsed, 2),
            "generation_metrics": gen_metrics,
            "context_relevance": round(avg_ctx_relevance, 4),
            "latency": latency_stats,
            "loop_stats": loop_stats,
            "self_healing": healing_report.to_dict(),
            "insufficient_context_rate": round(
                n_insufficient / max(len(results), 1), 4
            ),
            "by_domain": self._metrics_by_domain(samples, results),
        }

        logger.info("EvaluationRunner complete. Metrics: %s", {
            k: report[k] for k in ["n_queries", "generation_metrics", "context_relevance"]
        })

        return report

    async def _run_pipeline(self, samples: list[dict]) -> list[Any]:
        """Run all samples through orchestrator with bounded concurrency."""
        semaphore = asyncio.Semaphore(self._concurrency)

        async def _run_one(sample: dict) -> Any:
            async with semaphore:
                try:
                    return await self._orchestrator.run(sample["question"])
                except Exception as exc:
                    logger.warning("Pipeline failed for %s: %s", sample["id"], exc)
                    return None

        return await asyncio.gather(*[_run_one(s) for s in samples])

    def _latency_percentiles(self, latencies: list[int]) -> dict:
        if not latencies:
            return {}
        sorted_ms = sorted(latencies)
        n = len(sorted_ms)

        def pct(p: float) -> int:
            idx = max(0, int(n * p / 100) - 1)
            return sorted_ms[idx]

        return {
            "p50_ms": pct(50),
            "p75_ms": pct(75),
            "p90_ms": pct(90),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "avg_ms": round(sum(sorted_ms) / n),
            "max_ms": sorted_ms[-1],
            "n": n,
        }

    def _metrics_by_domain(self, samples: list[dict], results: list[Any]) -> dict:
        from evaluation.rag_metrics import token_f1, GenerationSample, compute_generation_metrics

        domains: dict[str, dict] = {}
        for sample, result in zip(samples, results):
            domain = sample["domain"]
            if domain not in domains:
                domains[domain] = {"samples": [], "results": []}
            domains[domain]["samples"].append(sample)
            domains[domain]["results"].append(result)

        by_domain = {}
        for domain, data in domains.items():
            gen_samps = [
                GenerationSample(
                    query=s["question"],
                    generated_answer=r.answer if r else "",
                    ground_truth_answer=s["answer"],
                    confidence=r.confidence if r else 0.0,
                )
                for s, r in zip(data["samples"], data["results"])
            ]
            by_domain[domain] = compute_generation_metrics(gen_samps)

        return by_domain


# ────────────────────────────────────────────────────────────────────────────
# AIRFLOW-COMPATIBLE ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

async def run_airflow_eval(
    pipeline_fn,
    n_samples: int = 100,
    threshold: float = 0.85,
) -> dict:
    """
    Lightweight eval suitable for the neurorag_eval Airflow DAG.
    Runs 100 benchmark queries and returns metrics for threshold checking.

    Args:
        pipeline_fn:  Async function str → PipelineResult.
        n_samples:    Number of benchmark queries to evaluate.
        threshold:    Confidence threshold for convergence check.

    Returns:
        Dict with avg_confidence, avg_token_f1, retry_success_rate.
    """
    from evaluation.benchmark_dataset import load_benchmark
    from evaluation.rag_metrics import (
        GenerationSample,
        compute_generation_metrics,
        log_generation_metrics_to_prometheus,
    )
    from evaluation.self_healing_validator import (
        SelfHealingValidator,
        log_self_healing_metrics_to_prometheus,
    )

    samples = load_benchmark(max_samples=n_samples, shuffle=True, seed=42)

    semaphore = asyncio.Semaphore(4)

    async def _run(s: dict) -> Any:
        async with semaphore:
            try:
                return await pipeline_fn(s["question"])
            except Exception:
                return None

    results = await asyncio.gather(*[_run(s) for s in samples])

    gen_samples = [
        GenerationSample(
            query=s["question"],
            generated_answer=r.answer if r else "",
            ground_truth_answer=s["answer"],
            confidence=r.confidence if r else 0.0,
        )
        for s, r in zip(samples, results)
    ]

    gen_metrics = compute_generation_metrics(gen_samples)
    log_generation_metrics_to_prometheus(gen_metrics)

    # Self-healing
    validator = SelfHealingValidator(
        pipeline_fn=pipeline_fn,
        confidence_threshold=threshold,
    )
    healing_report = await validator.validate(samples[:50])
    log_self_healing_metrics_to_prometheus(healing_report)

    return {
        **gen_metrics,
        **healing_report.to_dict(),
        "faithfulness_below_threshold": gen_metrics.get("avg_confidence", 1.0) < threshold,
    }


# ────────────────────────────────────────────────────────────────────────────
# STANDALONE CLI
# ────────────────────────────────────────────────────────────────────────────

async def _main_async(args: argparse.Namespace) -> None:
    """CLI entry point: load real orchestrator if available, else mock."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from rag.ingest import IngestionEngine
        from agents.orchestrator import NeuroRAGOrchestrator

        engine = IngestionEngine()
        orchestrator = NeuroRAGOrchestrator(engine)
        logger.info("Using real NeuroRAG orchestrator.")
    except Exception as exc:
        logger.warning("Could not load real orchestrator (%s); using mock.", exc)

        class _MockResult:
            def __init__(self, q: str) -> None:
                import random, uuid
                self.request_id = str(uuid.uuid4())
                self.query = q
                self.answer = f"Mock answer for: {q}"
                self.citations = ["doc1", "doc2"]
                self.confidence = random.uniform(0.7, 1.0)
                self.loops = random.randint(1, 3)
                self.latency_ms = random.randint(100, 800)
                self.insufficient_context = False
                self.from_memory_cache = False

        class _MockOrchestrator:
            async def run(self, query: str) -> _MockResult:
                await asyncio.sleep(0.01)
                return _MockResult(query)

        orchestrator = _MockOrchestrator()

    runner = EvaluationRunner(
        orchestrator=orchestrator,
        config_threshold=args.threshold,
        concurrency=args.concurrency,
    )
    report = await runner.run(mode=args.mode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nEvaluation report saved → {output_path}")
    print(f"Token F1: {report['generation_metrics'].get('avg_token_f1', 'N/A')}")
    print(f"Avg confidence: {report['generation_metrics'].get('avg_confidence', 'N/A')}")
    print(f"Context relevance: {report.get('context_relevance', 'N/A')}")
    print(f"Retry success rate: {report['self_healing'].get('retry_success_rate', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroRAG Evaluation Runner")
    parser.add_argument(
        "--mode",
        default="quick",
        choices=["full", "quick", "domain:factual", "domain:reasoning", "domain:multi_hop"],
        help="Evaluation mode",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="evaluation/eval_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
