"""
Evaluation Harness Runner — orchestrates RAG evaluation over the golden Q&A dataset.

This runner performs an end-to-end evaluation of the Personal Finance Literacy RAG Assistant:
  1. Loads hand-crafted test questions and ground truth from eval/golden_set.json.
  2. Executes vector search retrieval (LanceDB or specified retriever).
  3. Generates grounded answers using the local Ollama LLM.
  4. Computes retrieval quality metrics:
       - Precision@3  (target ≥ 0.70)
       - Recall@5     (target ≥ 0.80)
       - MRR          (target ≥ 0.75)
       - Hit Rate     (target ≥ 0.85)
  5. Computes generation quality metrics via OpenAI LLM-as-a-judge:
       - Answer Relevance  (target ≥ 0.80)
       - Faithfulness      (target ≥ 0.85)
  6. Tracks system latency (retrieval, generation, end-to-end p50 & p95).
  7. Prints a structured terminal summary table.
  8. Saves a comprehensive JSON evaluation report to eval/eval_report.json.

Usage:
  python3.11 eval/runner.py
  python3.11 -m eval.runner --limit 5 --no-judge
  python3.11 eval/runner.py --output ./eval/my_report.json
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import settings
from eval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate,
)
from eval.llm_judge import score_relevance, score_faithfulness
from ingestion.embedder import embed_query
from retrieval import lancedb_retriever
from generation.llm import generate_answer


# KPI targets from PROJECT_BRIEF.md
KPI_TARGETS = {
    "precision_at_3": 0.70,
    "recall_at_5": 0.80,
    "mrr": 0.75,
    "hit_rate": 0.85,
    "answer_relevance": 0.80,
    "faithfulness": 0.85,
}


def compute_percentile(values: list[float], percentile: float) -> float:
    """Calculate a percentile value from a list of numbers.

    Args:
        values: List of numerical measurements.
        percentile: Float between 0 and 100 (e.g., 50.0 or 95.0).

    Returns:
        The calculated percentile float, or 0.0 if values is empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    floor_idx = int(k)
    ceil_idx = floor_idx + 1
    if ceil_idx < len(sorted_vals):
        return sorted_vals[floor_idx] + (k - floor_idx) * (sorted_vals[ceil_idx] - sorted_vals[floor_idx])
    return float(sorted_vals[floor_idx])


def load_golden_set(file_path: str, limit: int | None = None) -> list[dict]:
    """Load the golden evaluation dataset from JSON.

    Args:
        file_path: Absolute or relative path to golden_set.json.
        limit: Optional maximum number of questions to load.

    Returns:
        List of question dictionaries.

    Raises:
        FileNotFoundError: If the golden set file does not exist.
        ValueError: If the file content is not a list.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected golden set to be a JSON array of objects, got {type(data)}")

    if limit is not None and limit > 0:
        return data[:limit]
    return data


def evaluate_single_query(
    item: dict,
    k: int = 5,
    retriever_fn=None,
    generator_fn=None,
    judge_enabled: bool = True,
    judge_model: str | None = None,
) -> dict:
    """Execute end-to-end RAG retrieval, generation, and metrics for one question.

    Args:
        item: A dictionary from the golden set containing 'id', 'question',
              'source', 'category', 'relevant_keywords', 'reference_answer'.
        k: Number of chunks to retrieve (default 5 to evaluate Recall@5).
        retriever_fn: Callable taking (query, k) and returning list[Chunk].
                      Defaults to LanceDB vector search.
        generator_fn: Callable taking (query, chunks) and returning str.
                      Defaults to generation.llm.generate_answer.
        judge_enabled: Whether to run the OpenAI LLM judge.
        judge_model: Optional model override for the judge.

    Returns:
        Dict containing per-query metrics, generated answer, and latency stats.
    """
    question = item["question"]
    relevant_keywords = item.get("relevant_keywords", [])
    reference_answer = item.get("reference_answer", "")

    # Default retrieval function: embed query + LanceDB search
    if retriever_fn is None:
        def default_retriever(q: str, top_k: int):
            q_vec = embed_query(q)
            return lancedb_retriever.search(q_vec, k=top_k)
        retriever_fn = default_retriever

    # Default generator function: Ollama generate_answer
    if generator_fn is None:
        generator_fn = generate_answer

    # 1. Retrieval
    retrieval_start = time.perf_counter()
    retrieved_chunks = retriever_fn(question, k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    # 2. Retrieval Metrics
    p_at_3 = precision_at_k(retrieved_chunks, relevant_keywords, k=3)
    r_at_5 = recall_at_k(retrieved_chunks, relevant_keywords, k=5)
    rr = reciprocal_rank(retrieved_chunks, relevant_keywords)
    hit = hit_rate(retrieved_chunks, relevant_keywords)

    # 3. Generation
    generation_start = time.perf_counter()
    answer = generator_fn(question, retrieved_chunks)
    generation_ms = (time.perf_counter() - generation_start) * 1000.0
    total_ms = retrieval_ms + generation_ms

    # 4. Generation Metrics (LLM Judge)
    relevance_score = None
    relevance_raw = None
    relevance_reasoning = None
    faithfulness_score = None
    faithfulness_raw = None
    faithfulness_reasoning = None

    if judge_enabled:
        # Context chunk texts for faithfulness evaluation
        context_texts = [c.text for c in retrieved_chunks]

        rel_result = score_relevance(
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            model=judge_model,
        )
        relevance_score = rel_result["score"]
        relevance_raw = rel_result["raw_score"]
        relevance_reasoning = rel_result["reasoning"]

        faith_result = score_faithfulness(
            question=question,
            answer=answer,
            context_texts=context_texts,
            model=judge_model,
        )
        faithfulness_score = faith_result["score"]
        faithfulness_raw = faith_result["raw_score"]
        faithfulness_reasoning = faith_result["reasoning"]

    # 5. Compile single query result
    chunk_summaries = [
        {
            "chunk_id": getattr(c, "chunk_id", f"idx_{i}"),
            "source": getattr(c, "source", "unknown"),
            "section": getattr(c, "section", ""),
            "preview": c.text.replace("\n", " ")[:150] + ("..." if len(c.text) > 150 else ""),
        }
        for i, c in enumerate(retrieved_chunks)
    ]

    return {
        "id": item["id"],
        "source": item.get("source", "unknown"),
        "category": item.get("category", "general"),
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": answer,
        "retrieved_chunks": chunk_summaries,
        "retrieval_latency_ms": round(retrieval_ms, 2),
        "generation_latency_ms": round(generation_ms, 2),
        "total_latency_ms": round(total_ms, 2),
        "metrics": {
            "precision_at_3": round(p_at_3, 4),
            "recall_at_5": round(r_at_5, 4),
            "mrr": round(rr, 4),
            "hit_rate": round(hit, 4),
            "answer_relevance": round(relevance_score, 4) if relevance_score is not None else None,
            "answer_relevance_raw": relevance_raw,
            "answer_relevance_reasoning": relevance_reasoning,
            "faithfulness": round(faithfulness_score, 4) if faithfulness_score is not None else None,
            "faithfulness_raw": faithfulness_raw,
            "faithfulness_reasoning": faithfulness_reasoning,
        },
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute mean scores and latency percentiles across all evaluated queries.

    Args:
        results: List of per-query result dictionaries from evaluate_single_query.

    Returns:
        Summary dictionary with aggregate means, KPI statuses, and latency statistics.
    """
    total = len(results)
    if total == 0:
        return {"total_evaluated": 0, "summary_metrics": {}, "latency": {}}

    def safe_mean(key: str) -> float | None:
        vals = [r["metrics"][key] for r in results if r["metrics"].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    mean_p3 = safe_mean("precision_at_3")
    mean_r5 = safe_mean("recall_at_5")
    mean_mrr = safe_mean("mrr")
    mean_hit = safe_mean("hit_rate")
    mean_relevance = safe_mean("answer_relevance")
    mean_faithfulness = safe_mean("faithfulness")

    # Latencies
    retrieval_times = [r["retrieval_latency_ms"] for r in results]
    generation_times = [r["generation_latency_ms"] for r in results]
    total_times = [r["total_latency_ms"] for r in results]

    summary_metrics = {
        "precision_at_3": {
            "score": mean_p3,
            "target": KPI_TARGETS["precision_at_3"],
            "passed": mean_p3 >= KPI_TARGETS["precision_at_3"] if mean_p3 is not None else False,
        },
        "recall_at_5": {
            "score": mean_r5,
            "target": KPI_TARGETS["recall_at_5"],
            "passed": mean_r5 >= KPI_TARGETS["recall_at_5"] if mean_r5 is not None else False,
        },
        "mrr": {
            "score": mean_mrr,
            "target": KPI_TARGETS["mrr"],
            "passed": mean_mrr >= KPI_TARGETS["mrr"] if mean_mrr is not None else False,
        },
        "hit_rate": {
            "score": mean_hit,
            "target": KPI_TARGETS["hit_rate"],
            "passed": mean_hit >= KPI_TARGETS["hit_rate"] if mean_hit is not None else False,
        },
    }

    if mean_relevance is not None:
        summary_metrics["answer_relevance"] = {
            "score": mean_relevance,
            "target": KPI_TARGETS["answer_relevance"],
            "passed": mean_relevance >= KPI_TARGETS["answer_relevance"],
        }

    if mean_faithfulness is not None:
        summary_metrics["faithfulness"] = {
            "score": mean_faithfulness,
            "target": KPI_TARGETS["faithfulness"],
            "passed": mean_faithfulness >= KPI_TARGETS["faithfulness"],
        }

    latency_stats = {
        "retrieval_ms": {
            "p50": round(compute_percentile(retrieval_times, 50.0), 2),
            "p95": round(compute_percentile(retrieval_times, 95.0), 2),
            "mean": round(sum(retrieval_times) / total, 2),
        },
        "generation_ms": {
            "p50": round(compute_percentile(generation_times, 50.0), 2),
            "p95": round(compute_percentile(generation_times, 95.0), 2),
            "mean": round(sum(generation_times) / total, 2),
        },
        "total_ms": {
            "p50": round(compute_percentile(total_times, 50.0), 2),
            "p95": round(compute_percentile(total_times, 95.0), 2),
            "mean": round(sum(total_times) / total, 2),
        },
    }

    return {
        "total_evaluated": total,
        "summary_metrics": summary_metrics,
        "latency": latency_stats,
    }


def format_terminal_table(results: list[dict], aggregate: dict) -> str:
    """Format evaluation results into a clean, human-readable terminal table.

    Args:
        results: List of per-query result dictionaries.
        aggregate: Summary dictionary from aggregate_metrics.

    Returns:
        Formatted multi-line string table.
    """
    lines = []
    lines.append("=" * 105)
    lines.append(" 📊 PERSONAL FINANCE LITERACY RAG ASSISTANT — EVALUATION REPORT")
    lines.append("=" * 105)

    # Per-Question Table
    header = f"{'ID':<8} {'Source':<18} {'Category':<20} {'Hit':<5} {'P@3':<7} {'R@5':<7} {'MRR':<7} {'Rel':<7} {'Faith':<7}"
    lines.append(header)
    lines.append("-" * 105)

    for r in results:
        m = r["metrics"]
        q_id = r["id"][:7]
        source = r["source"][:17]
        cat = r["category"][:19]
        hit_str = "✓" if m["hit_rate"] == 1.0 else "✗"
        p3_str = f"{m['precision_at_3']:.2f}"
        r5_str = f"{m['recall_at_5']:.2f}"
        mrr_str = f"{m['mrr']:.2f}"
        rel_str = f"{m['answer_relevance']:.2f}" if m.get("answer_relevance") is not None else "N/A"
        faith_str = f"{m['faithfulness']:.2f}" if m.get("faithfulness") is not None else "N/A"

        row = f"{q_id:<8} {source:<18} {cat:<20} {hit_str:<5} {p3_str:<7} {r5_str:<7} {mrr_str:<7} {rel_str:<7} {faith_str:<7}"
        lines.append(row)

    lines.append("-" * 105)

    # Summary Table with KPI Targets
    lines.append("\n📈 AGGREGATE METRICS VS KPI TARGETS:")
    lines.append("-" * 75)
    lines.append(f"{'Metric':<25} {'Score':<10} {'Target':<10} {'Status':<10}")
    lines.append("-" * 75)

    metric_names = [
        ("Precision@3", "precision_at_3"),
        ("Recall@5", "recall_at_5"),
        ("Mean Reciprocal Rank", "mrr"),
        ("Hit Rate", "hit_rate"),
        ("Answer Relevance", "answer_relevance"),
        ("Context Faithfulness", "faithfulness"),
    ]

    for label, key in metric_names:
        if key in aggregate["summary_metrics"]:
            info = aggregate["summary_metrics"][key]
            score_str = f"{info['score']:.4f}"
            target_str = f"≥ {info['target']:.2f}"
            status_str = "✅ PASS" if info["passed"] else "❌ FAIL"
            lines.append(f"{label:<25} {score_str:<10} {target_str:<10} {status_str:<10}")

    lines.append("-" * 75)

    # Latency Summary
    lat = aggregate.get("latency", {})
    if lat:
        lines.append("\n⏱️ SYSTEM LATENCY BREAKDOWN:")
        lines.append(f"  Retrieval:   p50 = {lat['retrieval_ms']['p50']:>6.1f} ms  |  p95 = {lat['retrieval_ms']['p95']:>6.1f} ms  |  mean = {lat['retrieval_ms']['mean']:>6.1f} ms")
        lines.append(f"  Generation:  p50 = {lat['generation_ms']['p50']/1000:>6.2f} s   |  p95 = {lat['generation_ms']['p95']/1000:>6.2f} s   |  mean = {lat['generation_ms']['mean']/1000:>6.2f} s")
        lines.append(f"  End-to-End:  p50 = {lat['total_ms']['p50']/1000:>6.2f} s   |  p95 = {lat['total_ms']['p95']/1000:>6.2f} s   |  mean = {lat['total_ms']['mean']/1000:>6.2f} s")
        lines.append("-" * 75)

    lines.append("=" * 105)
    return "\n".join(lines)


def save_report(report_data: dict, output_path: str) -> None:
    """Save full evaluation report dictionary to disk as formatted JSON.

    Args:
        report_data: Dictionary containing metadata, summary metrics, and per-query results.
        output_path: Path to the target JSON file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def run_eval(
    golden_set_path: str = settings.golden_set_path,
    output_path: str = settings.eval_report_path,
    k: int = 5,
    limit: int | None = None,
    judge_enabled: bool = True,
    judge_model: str | None = None,
    retriever_fn=None,
    generator_fn=None,
    retriever_name: str = "lancedb",
) -> dict:
    """High-level entry point to execute the evaluation harness.

    Args:
        golden_set_path: Path to golden_set.json.
        output_path: Path where eval_report.json will be saved.
        k: Top-k chunks to retrieve.
        limit: Optional maximum questions to run.
        judge_enabled: Whether to run the OpenAI judge.
        judge_model: Optional OpenAI judge model override.
        retriever_fn: Optional custom retriever callable.
        generator_fn: Optional custom generator callable.
        retriever_name: Identifier for the retriever being evaluated.

    Returns:
        The complete evaluation report dictionary.
    """
    start_time = datetime.now(timezone.utc)
    print(f"\n🚀 Starting RAG Evaluation...")
    print(f"   Golden Set:  {golden_set_path}")
    print(f"   Retriever:   {retriever_name} (top_k={k})")
    print(f"   LLM Model:   {settings.ollama_model}")
    print(f"   Judge:       {'Enabled (' + (judge_model or settings.eval_judge_model) + ')' if judge_enabled else 'Disabled (--no-judge)'}")
    print(f"   Report Path: {output_path}")

    # Load golden set
    items = load_golden_set(golden_set_path, limit=limit)
    total = len(items)
    print(f"   Loaded {total} evaluation question(s).\n")

    results = []
    for i, item in enumerate(items, 1):
        q_id = item["id"]
        q_text = item["question"]
        print(f"[{i}/{total}] Evaluating {q_id}: \"{q_text[:50]}...\"")

        res = evaluate_single_query(
            item=item,
            k=k,
            retriever_fn=retriever_fn,
            generator_fn=generator_fn,
            judge_enabled=judge_enabled,
            judge_model=judge_model,
        )
        results.append(res)

        # Quick inline feedback
        m = res["metrics"]
        rel_disp = f" | Rel: {m['answer_relevance']:.2f}" if m.get("answer_relevance") is not None else ""
        faith_disp = f" | Faith: {m['faithfulness']:.2f}" if m.get("faithfulness") is not None else ""
        print(f"       → Hit: {m['hit_rate']:.0f} | P@3: {m['precision_at_3']:.2f} | R@5: {m['recall_at_5']:.2f}{rel_disp}{faith_disp} ({res['total_latency_ms']:.0f}ms)")

    # Aggregate
    aggregate = aggregate_metrics(results)

    # Format report
    end_time = datetime.now(timezone.utc)
    report = {
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "configuration": {
            "retriever": retriever_name,
            "top_k": k,
            "ollama_model": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "judge_enabled": judge_enabled,
            "judge_model": judge_model or settings.eval_judge_model if judge_enabled else None,
            "golden_set_path": golden_set_path,
        },
        "aggregate": aggregate,
        "per_query_results": results,
    }

    # Print summary
    table_output = format_terminal_table(results, aggregate)
    print("\n" + table_output + "\n")

    # Save to file
    save_report(report, output_path)
    print(f"💾 Full evaluation report saved to: {output_path}")

    return report


def main():
    """CLI entry point for running the evaluation harness."""
    parser = argparse.ArgumentParser(description="Personal Finance Literacy RAG Assistant — Evaluation Harness")
    parser.add_argument("--golden-set", default=settings.golden_set_path, help="Path to golden Q&A JSON file")
    parser.add_argument("--output", default=settings.eval_report_path, help="Path to write output eval report JSON")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N questions (for testing)")
    parser.add_argument("--retriever", default="lancedb", choices=["lancedb", "pinecone"], help="Retriever to evaluate")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge scoring (retrieval metrics only)")
    parser.add_argument("--judge-model", default=None, help="OpenAI model override for the judge (default: from config)")
    parser.add_argument("--model", default=None, help="Ollama model override for generation")

    args = parser.parse_args()

    if args.model:
        settings.ollama_model = args.model

    run_eval(
        golden_set_path=args.golden_set,
        output_path=args.output,
        k=args.top_k,
        limit=args.limit,
        judge_enabled=not args.no_judge,
        judge_model=args.judge_model,
        retriever_name=args.retriever,
    )


if __name__ == "__main__":
    main()

