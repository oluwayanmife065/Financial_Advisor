"""
Unit tests for eval/runner.py — Evaluation Harness Orchestrator.

These tests use synthetic/mocked inputs (mocked retriever, generator, and judge)
so they run rapidly in isolation without needing a live LanceDB database,
live Ollama server, or OpenAI network calls.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from eval.runner import (
    compute_percentile,
    load_golden_set,
    evaluate_single_query,
    aggregate_metrics,
    format_terminal_table,
    save_report,
    run_eval,
    KPI_TARGETS,
)


@dataclass
class MockChunk:
    """Minimal mock chunk object satisfying retriever return contract."""
    text: str
    source: str = "Test Source"
    section: str = "Section 1"
    chunk_id: str = "chunk_001"


# ── compute_percentile() tests ─────────────────────────────────────

def test_compute_percentile_empty():
    """Empty list returns 0.0."""
    assert compute_percentile([], 50.0) == 0.0


def test_compute_percentile_single():
    """Single item returns its own value for any percentile."""
    assert compute_percentile([42.0], 50.0) == 42.0
    assert compute_percentile([42.0], 95.0) == 42.0


def test_compute_percentile_values():
    """Calculates median (p50) and high percentile (p95) accurately."""
    data = list(range(1, 101))  # 1 to 100
    p50 = compute_percentile(data, 50.0)
    assert p50 == pytest.approx(50.5, rel=1e-2)
    p95 = compute_percentile(data, 95.0)
    assert p95 == pytest.approx(95.05, rel=1e-2)


# ── load_golden_set() tests ────────────────────────────────────────

def test_load_golden_set_success(tmp_path):
    """Loads valid JSON array and respects limit parameter."""
    sample = [
        {"id": "q1", "question": "What is A?"},
        {"id": "q2", "question": "What is B?"},
        {"id": "q3", "question": "What is C?"},
    ]
    file_path = tmp_path / "test_golden.json"
    file_path.write_text(json.dumps(sample), encoding="utf-8")

    loaded = load_golden_set(str(file_path))
    assert len(loaded) == 3
    assert loaded[0]["id"] == "q1"

    limited = load_golden_set(str(file_path), limit=2)
    assert len(limited) == 2
    assert [x["id"] for x in limited] == ["q1", "q2"]


def test_load_golden_set_not_found():
    """Raises FileNotFoundError if file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_golden_set("non_existent_file_path.json")


def test_load_golden_set_invalid_format(tmp_path):
    """Raises ValueError if file is not a JSON list."""
    file_path = tmp_path / "invalid.json"
    file_path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected golden set to be a JSON array"):
        load_golden_set(str(file_path))


# ── evaluate_single_query() tests ──────────────────────────────────

def test_evaluate_single_query_mocked():
    """Full execution of single query evaluation with mocked dependencies."""
    item = {
        "id": "test_01",
        "question": "What is compound interest?",
        "source": "Federal Reserve",
        "category": "Compound Interest",
        "relevant_keywords": ["compound", "interest"],
        "reference_answer": "Interest on interest.",
    }

    mock_retriever = MagicMock(return_value=[
        MockChunk(text="Compound interest accelerates savings over time.", chunk_id="c1"),
        MockChunk(text="Another irrelevant document.", chunk_id="c2"),
    ])
    mock_generator = MagicMock(return_value="Compound interest is interest on accumulated interest.")

    with patch("eval.runner.score_relevance") as mock_rel, \
         patch("eval.runner.score_faithfulness") as mock_faith:
        mock_rel.return_value = {"score": 1.0, "raw_score": 5, "reasoning": "Accurate."}
        mock_faith.return_value = {"score": 1.0, "raw_score": 5, "reasoning": "Grounded."}

        res = evaluate_single_query(
            item=item,
            k=2,
            retriever_fn=mock_retriever,
            generator_fn=mock_generator,
            judge_enabled=True,
        )

        assert res["id"] == "test_01"
        assert res["source"] == "Federal Reserve"
        assert res["generated_answer"] == "Compound interest is interest on accumulated interest."
        assert res["metrics"]["hit_rate"] == 1.0
        assert res["metrics"]["precision_at_3"] == 0.5
        assert res["metrics"]["recall_at_5"] == 1.0
        assert res["metrics"]["mrr"] == 1.0
        assert res["metrics"]["answer_relevance"] == 1.0
        assert res["metrics"]["faithfulness"] == 1.0
        assert "retrieval_latency_ms" in res
        assert "generation_latency_ms" in res


def test_evaluate_single_query_no_judge():
    """When judge_enabled=False, judge calls are skipped and scores are None."""
    item = {
        "id": "test_02",
        "question": "Sample query?",
        "relevant_keywords": ["sample"],
        "reference_answer": "Sample answer.",
    }
    mock_retriever = MagicMock(return_value=[MockChunk(text="Sample content")])
    mock_generator = MagicMock(return_value="Generated answer")

    with patch("eval.runner.score_relevance") as mock_rel, \
         patch("eval.runner.score_faithfulness") as mock_faith:
        res = evaluate_single_query(
            item=item,
            k=1,
            retriever_fn=mock_retriever,
            generator_fn=mock_generator,
            judge_enabled=False,
        )

        mock_rel.assert_not_called()
        mock_faith.assert_not_called()
        assert res["metrics"]["answer_relevance"] is None
        assert res["metrics"]["faithfulness"] is None
        assert res["metrics"]["hit_rate"] == 1.0


# ── aggregate_metrics() tests ─────────────────────────────────────

def test_aggregate_metrics_empty():
    """Empty list returns total_evaluated = 0."""
    res = aggregate_metrics([])
    assert res["total_evaluated"] == 0


def test_aggregate_metrics_calculation():
    """Computes correct averages and KPI pass/fail status."""
    results = [
        {
            "metrics": {
                "precision_at_3": 1.0,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "hit_rate": 1.0,
                "answer_relevance": 1.0,
                "faithfulness": 1.0,
            },
            "retrieval_latency_ms": 10.0,
            "generation_latency_ms": 100.0,
            "total_latency_ms": 110.0,
        },
        {
            "metrics": {
                "precision_at_3": 0.5,
                "recall_at_5": 1.0,
                "mrr": 0.5,
                "hit_rate": 1.0,
                "answer_relevance": 0.5,
                "faithfulness": 0.5,
            },
            "retrieval_latency_ms": 20.0,
            "generation_latency_ms": 200.0,
            "total_latency_ms": 220.0,
        },
    ]

    agg = aggregate_metrics(results)
    assert agg["total_evaluated"] == 2

    # Precision@3 = (1.0 + 0.5) / 2 = 0.75 >= 0.70 -> PASS
    p3 = agg["summary_metrics"]["precision_at_3"]
    assert p3["score"] == 0.75
    assert p3["passed"] is True

    # MRR = (1.0 + 0.5) / 2 = 0.75 >= 0.75 -> PASS
    mrr = agg["summary_metrics"]["mrr"]
    assert mrr["score"] == 0.75
    assert mrr["passed"] is True

    # Answer relevance = (1.0 + 0.5) / 2 = 0.75 < 0.80 -> FAIL
    rel = agg["summary_metrics"]["answer_relevance"]
    assert rel["score"] == 0.75
    assert rel["passed"] is False

    # Latencies
    assert agg["latency"]["retrieval_ms"]["mean"] == 15.0
    assert agg["latency"]["generation_ms"]["mean"] == 150.0


# ── format_terminal_table() & save_report() tests ──────────────────

def test_format_terminal_table():
    """Verifies table string includes expected sections and headers."""
    results = [
        {
            "id": "q1",
            "source": "SEC",
            "category": "Debt",
            "metrics": {
                "hit_rate": 1.0,
                "precision_at_3": 0.67,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "answer_relevance": 0.75,
                "faithfulness": 1.0,
            },
        }
    ]
    agg = {
        "summary_metrics": {
            "precision_at_3": {"score": 0.67, "target": 0.70, "passed": False},
            "recall_at_5": {"score": 1.0, "target": 0.80, "passed": True},
            "mrr": {"score": 1.0, "target": 0.75, "passed": True},
            "hit_rate": {"score": 1.0, "target": 0.85, "passed": True},
            "answer_relevance": {"score": 0.75, "target": 0.80, "passed": False},
            "faithfulness": {"score": 1.0, "target": 0.85, "passed": True},
        },
        "latency": {
            "retrieval_ms": {"p50": 12.0, "p95": 20.0, "mean": 15.0},
            "generation_ms": {"p50": 800.0, "p95": 1200.0, "mean": 900.0},
            "total_ms": {"p50": 812.0, "p95": 1220.0, "mean": 915.0},
        },
    }

    output = format_terminal_table(results, agg)
    assert "PERSONAL FINANCE LITERACY RAG ASSISTANT" in output
    assert "Precision@3" in output
    assert "✅ PASS" in output
    assert "❌ FAIL" in output
    assert "SYSTEM LATENCY BREAKDOWN" in output


def test_save_report(tmp_path):
    """Saves valid JSON report file to target path."""
    report = {"status": "success", "score": 0.95}
    out_file = tmp_path / "reports" / "eval_report.json"
    save_report(report, str(out_file))

    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["score"] == 0.95


# ── run_eval() end-to-end mocked test ──────────────────────────────

def test_run_eval_mocked(tmp_path):
    """End-to-end execution of run_eval using mocked components."""
    sample_golden = [
        {
            "id": "m1",
            "question": "What is 401(k)?",
            "source": "SEC/Investor.gov",
            "category": "Retirement",
            "relevant_keywords": ["401(k)"],
            "reference_answer": "Retirement plan.",
        }
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(sample_golden), encoding="utf-8")
    report_path = tmp_path / "eval_report.json"

    mock_retriever = MagicMock(return_value=[MockChunk(text="401(k) is an employer retirement plan.")])
    mock_generator = MagicMock(return_value="A 401(k) is a retirement plan.")

    report = run_eval(
        golden_set_path=str(golden_path),
        output_path=str(report_path),
        k=3,
        judge_enabled=False,
        retriever_fn=mock_retriever,
        generator_fn=mock_generator,
        retriever_name="mock_retriever",
    )

    assert report["configuration"]["retriever"] == "mock_retriever"
    assert report["aggregate"]["total_evaluated"] == 1
    assert report["aggregate"]["summary_metrics"]["hit_rate"]["passed"] is True
    assert report_path.exists()

