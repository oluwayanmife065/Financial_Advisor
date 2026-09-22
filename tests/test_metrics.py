"""
Tests for eval/metrics.py — retrieval quality metrics.

These tests use synthetic data (MockChunk) so they run instantly
with no Ollama, no LanceDB, no embeddings needed. Pure logic tests.

Test strategy:
    For each metric, we test three cases:
    1. Perfect score  — all chunks relevant → metric = 1.0
    2. Zero score     — no chunks relevant  → metric = 0.0
    3. Partial score  — mixed relevance     → metric = expected float

    Plus edge cases: empty inputs, no keywords, k boundaries.
"""

from dataclasses import dataclass

import pytest

from eval.metrics import (
    is_relevant,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate,
    compute_retrieval_metrics,
)


# ── Test fixture ──────────────────────────────────────────────────
# MockChunk mimics the real Chunk dataclass but only has .text,
# which is all the metrics functions need (they use duck typing).

@dataclass
class MockChunk:
    """Minimal stand-in for ingestion.document.Chunk — just needs .text."""
    text: str


# ── is_relevant() tests ──────────────────────────────────────────

def test_is_relevant_match():
    """Keyword present in text → True."""
    assert is_relevant("This is a relevant document.", ["relevant"]) is True
    # Multiple keywords — only one needs to match
    assert is_relevant("The quick brown fox", ["fox", "dog"]) is True


def test_is_relevant_no_match():
    """Keyword not in text → False."""
    assert is_relevant("This is irrelevant.", ["cat"]) is False


def test_is_relevant_case_insensitive():
    """Matching is case-insensitive — "QUICK" matches "quick" and vice versa."""
    assert is_relevant("The QUICK brown FOX", ["quick"]) is True
    assert is_relevant("the quick brown fox", ["FOX"]) is True


# ── precision_at_k() tests ───────────────────────────────────────

def test_precision_at_k_perfect():
    """All 3 chunks contain a keyword → Precision@3 = 3/3 = 1.0."""
    chunks = [MockChunk("a"), MockChunk("b"), MockChunk("c")]
    assert precision_at_k(chunks, ["a", "b", "c"], k=3) == 1.0


def test_precision_at_k_none():
    """No chunks contain the keyword → Precision = 0/2 = 0.0."""
    chunks = [MockChunk("a"), MockChunk("b")]
    assert precision_at_k(chunks, ["x"], k=2) == 0.0


def test_precision_at_k_partial():
    """2 of 3 chunks contain "hit" → Precision@3 = 2/3 ≈ 0.667."""
    chunks = [MockChunk("hit"), MockChunk("miss"), MockChunk("hit")]
    assert precision_at_k(chunks, ["hit"], k=3) == pytest.approx(0.6666666, rel=1e-5)


def test_precision_at_k_limits_to_k():
    """With 4 chunks but k=2, only evaluates the first 2.
    First 2 are [hit, miss] → Precision@2 = 1/2 = 0.5."""
    chunks = [MockChunk("hit"), MockChunk("miss"), MockChunk("miss"), MockChunk("hit")]
    assert precision_at_k(chunks, ["hit"], k=2) == 0.5


# ── recall_at_k() tests ─────────────────────────────────────────

def test_recall_at_k_found():
    """1 relevant chunk found, 1 expected → Recall = 1/1 = 1.0."""
    chunks = [MockChunk("hit"), MockChunk("miss")]
    assert recall_at_k(chunks, ["hit"], k=2, expected_relevant=1) == 1.0


def test_recall_at_k_not_found():
    """0 relevant chunks found → Recall = 0/1 = 0.0."""
    chunks = [MockChunk("miss"), MockChunk("miss")]
    assert recall_at_k(chunks, ["hit"], k=2, expected_relevant=1) == 0.0


# ── reciprocal_rank() tests ──────────────────────────────────────

def test_reciprocal_rank_first():
    """Relevant chunk at position 1 → RR = 1/1 = 1.0 (best possible)."""
    chunks = [MockChunk("hit"), MockChunk("miss")]
    assert reciprocal_rank(chunks, ["hit"]) == 1.0


def test_reciprocal_rank_third():
    """Relevant chunk at position 3 → RR = 1/3 ≈ 0.333 (user had to scroll)."""
    chunks = [MockChunk("miss"), MockChunk("miss"), MockChunk("hit")]
    assert reciprocal_rank(chunks, ["hit"]) == pytest.approx(0.3333333, rel=1e-5)


def test_reciprocal_rank_none():
    """No relevant chunks → RR = 0.0 (complete miss)."""
    chunks = [MockChunk("miss"), MockChunk("miss")]
    assert reciprocal_rank(chunks, ["hit"]) == 0.0


# ── hit_rate() tests ─────────────────────────────────────────────

def test_hit_rate_hit():
    """At least one chunk is relevant → 1.0."""
    chunks = [MockChunk("miss"), MockChunk("hit")]
    assert hit_rate(chunks, ["hit"]) == 1.0


def test_hit_rate_miss():
    """No chunks are relevant → 0.0."""
    chunks = [MockChunk("miss"), MockChunk("miss")]
    assert hit_rate(chunks, ["hit"]) == 0.0


# ── compute_retrieval_metrics() tests ────────────────────────────

def test_compute_retrieval_metrics_keys():
    """Convenience function returns a dict with all 4 metric keys and correct values."""
    chunks = [MockChunk("hit"), MockChunk("miss")]
    metrics = compute_retrieval_metrics(chunks, ["hit"], k=2)
    # Check structure
    assert isinstance(metrics, dict)
    assert "precision_at_k" in metrics
    assert "recall_at_k" in metrics
    assert "reciprocal_rank" in metrics
    assert "hit_rate" in metrics
    # Check values — 1 hit, 1 miss at k=2
    assert metrics["hit_rate"] == 1.0          # at least one hit exists
    assert metrics["precision_at_k"] == 0.5    # 1 relevant out of 2


# ── Edge case tests ──────────────────────────────────────────────

def test_empty_chunks():
    """Empty chunk list → all metrics return 0.0 (nothing to evaluate)."""
    chunks = []
    assert precision_at_k(chunks, ["hit"]) == 0.0
    assert recall_at_k(chunks, ["hit"]) == 0.0
    assert reciprocal_rank(chunks, ["hit"]) == 0.0
    assert hit_rate(chunks, ["hit"]) == 0.0


def test_empty_keywords():
    """No keywords → we can't determine relevance → everything is 0.0."""
    chunks = [MockChunk("hit")]
    assert is_relevant(chunks[0].text, []) is False
    assert precision_at_k(chunks, []) == 0.0
    assert recall_at_k(chunks, []) == 0.0
    assert reciprocal_rank(chunks, []) == 0.0
    assert hit_rate(chunks, []) == 0.0
