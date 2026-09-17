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

## [v0.2.0-retrieval] — Phase 2 target
## [v0.2.0-retrieval] — 2026-09-17

### Planned
- `BaseRetriever` abstract interface
- Ollama LLM wrapper + grounding prompt
- CLI chat loop
- JSON Lines query logger
### Added
- Ollama LLM wrapper with grounding system prompt and streaming response (`generation/llm.py`)
- Interactive CLI chat loop with context retrieval and citation display (`main.py`)
- JSON Lines query logger tracking queries, retrieved chunks, response latency, and answers (`query_logging/query_logger.py`)
- Unit tests for LLM generation and query logging (`tests/test_llm.py`, `tests/test_query_logger.py`)

---

## [v0.3.0-eval] — Phase 3 target

### Planned
- Golden Q&A set (20–30 pairs)
- Precision@k, Recall@k, MRR, Hit Rate metrics
- Ollama LLM-as-judge (answer relevance + faithfulness)
- `eval/runner.py` producing `eval_report.json`

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