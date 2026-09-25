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

## [v0.3.0-eval] — Phase 3 target
## [v0.3.0-eval] — 2026-09-22

### Planned
- Golden Q&A set (20–30 pairs)
- Precision@k, Recall@k, MRR, Hit Rate metrics
- Ollama LLM-as-judge (answer relevance + faithfulness)
- `eval/runner.py` producing `eval_report.json`
### Added
- Golden Q&A evaluation dataset with 25 hand-crafted test questions across SEC, CFPB, and Federal Reserve corpora (`eval/golden_set.json`)
- Retrieval evaluation metrics: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and Hit Rate using keyword-based relevance matching (`eval/metrics.py`)
- Independent LLM-as-a-judge scoring Answer Relevance and Context Faithfulness using OpenAI GPT-4o-mini with retry logic and JSON code-fence stripping (`eval/llm_judge.py`)
- Evaluation runner orchestrator with CLI flags (`--limit`, `--no-judge`, `--retriever`, `--top-k`), terminal table formatting, and detailed JSON report export (`eval/runner.py`)
- Comprehensive test suites for metrics, judge mocking, and runner orchestration (`tests/test_metrics.py`, `tests/test_llm_judge.py`, `tests/test_runner.py`) bringing repository test suite to 111 passing tests
- Live benchmark run results meeting all retrieval KPI targets: Precision@3 = 0.84 (target ≥ 0.70), Recall@5 = 1.00 (target ≥ 0.80), MRR = 0.98 (target ≥ 0.75), Hit Rate = 1.00 (target ≥ 0.85)

---

## [v0.4.0-pinecone] — 2026-09-24

### Added
- `retrieval/base.py` — abstract `BaseRetriever` ABC defining the `write / search / count` interface both retrievers must satisfy
- `retrieval/pinecone_retriever.py` — full Pinecone serverless implementation using `pinecone>=3.0` SDK: on-demand index creation (`ServerlessSpec`), batch upserts (100/req), metadata-backed `Chunk` reconstruction, cosine-similarity search, mirroring the `lancedb_retriever` module API exactly
- `ingestion/pipeline.py` — `--target [lancedb|pinecone|both]` CLI flag and `--from-lancedb` fast-path that reads existing embedded chunks from LanceDB and upserts them to Pinecone (no re-parse, no re-embed)
- `eval/runner.py` — `--retriever pinecone` now routes to the real Pinecone retriever; new `--compare` flag runs both retrievers sequentially and prints a side-by-side benchmark table with Δ column, saving two separate JSON reports (`eval_report_lancedb.json` / `eval_report_pinecone.json`)
- `config.py` — added `pinecone_cloud` and `pinecone_region` fields for the serverless SDK; legacy `pinecone_environment` kept for backwards compatibility
- `requirements.txt` — `pinecone>=3.0` activated (was commented out)
- `tests/test_retrievers.py` — 14 mock-based unit tests covering `write` (batch splitting, index creation, metadata shape, delete-before-upsert, missing-embedding error), `search` (chunk reconstruction, empty results, missing index, default top_k, include_metadata flag), `count`, and `_get_client` error — no live API calls required

---

## [v0.5.0-polish] — 2026-09-23

### Added
- Streamlit chat UI (`app.py`) with persistent session-state conversation history, live streaming token output via `st.write_stream()`, collapsible source citation expanders, and retrieval/generation/total latency badges per response
- Streaming LLM variant `stream_answer()` in `generation/llm.py` using Ollama `stream=True` — yields token strings as they arrive; consumed by Streamlit without blocking
- Sidebar controls: Ollama model selector (auto-populated from `ollama list`), top-k retrieval slider (1–10), source citation toggle, corpus chunk count, eval KPI metric badges (Precision@3 / Recall@5 / MRR / Hit Rate) from `eval_report.json`, clear-chat button, and last-5 query log viewer
- Empty-state prompt suggestion buttons for first-time users
- `streamlit>=1.35` added to `requirements.txt`

---