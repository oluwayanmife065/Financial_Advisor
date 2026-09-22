"""
Retrieval Quality Metrics — measures how well our vector search finds relevant chunks.

Why keyword-based relevance?
    In a production RAG eval, you'd manually label each chunk as relevant/irrelevant
    for every question (hundreds of annotations). For a portfolio project, we use a
    pragmatic shortcut: a chunk is "relevant" if it contains at least one expected
    keyword. This is transparent, reproducible, and easy to explain in interviews.

    Example: For the question "What is a mutual fund?", we expect relevant chunks
    to contain keywords like "mutual fund", "pooled investment", "diversification".
    If a retrieved chunk contains any of those, we count it as a hit.

KPI targets (from PROJECT_BRIEF.md):
    Precision@3  ≥ 0.70    Recall@5  ≥ 0.80
    MRR          ≥ 0.75    Hit Rate  ≥ 0.85
"""


def is_relevant(chunk_text: str, relevant_keywords: list[str]) -> bool:
    """Check if a chunk is relevant by keyword presence.

    This is the core relevance function — every other metric calls this
    to decide whether a retrieved chunk counts as a "hit" or "miss".

    Args:
        chunk_text: The text content of the chunk.
        relevant_keywords: Keywords we expect to find in a relevant chunk.

    Returns:
        True if at least one keyword (case-insensitive) appears in the text.
    """
    # No keywords means we can't determine relevance — default to False
    if not relevant_keywords:
        return False

    # Lowercase once for efficiency — avoids re-lowering on every keyword check
    text_lower = chunk_text.lower()
    for kw in relevant_keywords:
        if kw.lower() in text_lower:
            return True
    return False


def precision_at_k(retrieved_chunks: list, relevant_keywords: list[str], k: int = 3) -> float:
    """Fraction of top-k retrieved chunks that are relevant.

    Answers: "Of the chunks we showed the user, how many were actually useful?"

    Example: If we retrieve 3 chunks and 2 are relevant → Precision@3 = 0.667

    Args:
        retrieved_chunks: Chunks from vector search (objects with .text attribute).
        relevant_keywords: Keywords that signal relevance for this question.
        k: How many top results to evaluate (default 3, matching our KPI target).

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 for edge cases (k=0, empty list).
    """
    if k <= 0 or not retrieved_chunks:
        return 0.0

    # Only look at the first k results — the rest don't matter for this metric
    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for chunk in top_k if is_relevant(chunk.text, relevant_keywords))

    # Divide by actual number of chunks (might be less than k if fewer were returned)
    return relevant_count / len(top_k)


def recall_at_k(retrieved_chunks: list, relevant_keywords: list[str], k: int = 5, expected_relevant: int = 1) -> float:
    """Fraction of expected relevant chunks found in top-k.

    Answers: "Did we find all the relevant information, or did we miss some?"

    Unlike precision (which penalizes irrelevant results), recall penalizes
    missing results. With expected_relevant=1, this becomes binary:
    "Did we find at least one relevant chunk?"

    Args:
        retrieved_chunks: Chunks from vector search.
        relevant_keywords: Keywords that signal relevance.
        k: How many top results to search through (default 5).
        expected_relevant: How many relevant chunks we expect exist in the corpus.
            Defaults to 1 because for most questions, finding at least one good
            chunk is sufficient for a correct answer.

    Returns:
        Float between 0.0 and 1.0. Capped at 1.0 even if we find more than expected.
    """
    if k <= 0 or not retrieved_chunks or expected_relevant <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for chunk in top_k if is_relevant(chunk.text, relevant_keywords))

    # Cap at 1.0 — finding 3 relevant chunks when we expected 1 is still 100% recall
    return min(1.0, relevant_count / expected_relevant)


def reciprocal_rank(retrieved_chunks: list, relevant_keywords: list[str]) -> float:
    """1/rank of the first relevant chunk. This is the per-query component of MRR.

    Answers: "How far down did the user have to scroll to find something useful?"

    Examples:
        First chunk relevant  → 1/1 = 1.0   (perfect)
        Third chunk relevant  → 1/3 = 0.333 (user had to scroll)
        No chunk relevant     → 0.0          (complete miss)

    When averaged across all questions in the golden set, this becomes
    Mean Reciprocal Rank (MRR) — our KPI target is ≥ 0.75.

    Args:
        retrieved_chunks: Chunks from vector search (in ranked order).
        relevant_keywords: Keywords that signal relevance.

    Returns:
        Float between 0.0 and 1.0.
    """
    for i, chunk in enumerate(retrieved_chunks):
        if is_relevant(chunk.text, relevant_keywords):
            return 1.0 / (i + 1)  # i is 0-indexed, rank is 1-indexed
    return 0.0


def hit_rate(retrieved_chunks: list, relevant_keywords: list[str]) -> float:
    """1.0 if ANY retrieved chunk is relevant, 0.0 otherwise.

    Answers: "Did we find anything useful at all?"

    This is the simplest metric — a binary pass/fail per question.
    When averaged across all questions, it tells us what percentage
    of queries returned at least one relevant result. Target: ≥ 0.85.

    Args:
        retrieved_chunks: Chunks from vector search.
        relevant_keywords: Keywords that signal relevance.

    Returns:
        1.0 (hit) or 0.0 (miss).
    """
    for chunk in retrieved_chunks:
        if is_relevant(chunk.text, relevant_keywords):
            return 1.0
    return 0.0


def compute_retrieval_metrics(retrieved_chunks: list, relevant_keywords: list[str], k: int = 5) -> dict:
    """Convenience function — runs all metrics at once and returns a dict.

    This is what the eval runner calls for each question in the golden set.
    The runner then averages these dicts across all questions to get
    the aggregate scores that go into the eval report.

    Args:
        retrieved_chunks: Chunks from vector search.
        relevant_keywords: Keywords that signal relevance.
        k: Top-k cutoff for precision and recall.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, reciprocal_rank, hit_rate.
    """
    return {
        "precision_at_k": precision_at_k(retrieved_chunks, relevant_keywords, k=k),
        "recall_at_k": recall_at_k(retrieved_chunks, relevant_keywords, k=k),
        "reciprocal_rank": reciprocal_rank(retrieved_chunks, relevant_keywords),
        "hit_rate": hit_rate(retrieved_chunks, relevant_keywords),
    }
