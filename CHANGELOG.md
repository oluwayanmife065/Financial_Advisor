# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [v0.1.0-ingestion] — 2026-09-17

### Added
- Sentence-aware chunker (512 tokens, 64-token overlap)
- `bge-small-en-v1.5` embedder wrapper via SentenceTransformers
- End-to-end ingestion pipeline (parse → chunk → embed → store)
- LanceDB retriever implementation
- PDF parser with text cleaning and folder batch parsing
- `Document` & `Chunk` dataclasses with hash-based doc IDs
- Full test suite (parser, chunker, embedder, pipeline, retriever)

### Phase 0 (included)
- Initial project scaffold and folder structure
- `config.py` with Pydantic Settings
- `.gitignore`, `CHANGELOG.md`, `PROJECT_BRIEF.md`

---

## [v0.2.0-retrieval] — 2026-09-17

### Added
- Ollama LLM wrapper with grounding system prompt and streaming response (`generation/llm.py`)
- Interactive CLI chat loop with context retrieval and citation display (`main.py`)
- JSON Lines query logger tracking queries, retrieved chunks, response latency, and answers (`query_logging/query_logger.py`)
- Unit tests for LLM generation and query logging (`tests/test_llm.py`, `tests/test_query_logger.py`)

---

## [v0.3.0-eval] — 2026-09-22

### Added
- Golden Q&A evaluation dataset with 25 hand-crafted test questions across SEC, CFPB, and Federal Reserve corpora (`eval/golden_set.json`)
- Retrieval evaluation metrics: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and Hit Rate using keyword-based relevance matching (`eval/metrics.py`)
- Independent LLM-as-a-judge scoring Answer Relevance and Context Faithfulness using OpenAI GPT-4o-mini with retry logic and JSON code-fence stripping (`eval/llm_judge.py`)
- Evaluation runner orchestrator with CLI flags (`--limit`, `--no-judge`, `--retriever`, `--top-k`), terminal table formatting, and detailed JSON report export (`eval/runner.py`)
- Comprehensive test suites for metrics, judge mocking, and runner orchestration (`tests/test_metrics.py`, `tests/test_llm_judge.py`, `tests/test_runner.py`) bringing repository test suite to 111 passing tests
- Live benchmark run results meeting all retrieval KPI targets: Precision@3 = 0.84 (target ≥ 0.70), Recall@5 = 1.00 (target ≥ 0.80), MRR = 0.98 (target ≥ 0.75), Hit Rate = 1.00 (target ≥ 0.85)

---

## [v0.4.0-pinecone] — Phase 4 target

### Planned
- Pinecone retriever implementing `BaseRetriever`
- `--retriever [lancedb|pinecone]` CLI flag
- Side-by-side benchmark table

---

## [v0.5.0-polish] — Phase 5 target

### Planned
- Streamlit UI (`app.py`)
- Full README case study
- Complete pytest suite

---