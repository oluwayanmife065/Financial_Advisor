# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [v1.3.0-deploy-readiness] — 2026-10-09

### Added
- **Educational & Legal Compliance Disclaimer Banner** (`app.py`):
  - Prominent UI header disclaimer clarifying that the assistant is an academic & portfolio demonstration synthesizing official US government documents (IRS Pub 519, DHS, USCIS, SEC, CFPB, Federal Reserve).
  - Explicitly states that the tool does not provide certified financial, legal, tax, or immigration advice, and directs users to consult a DSO or licensed professional.
- **Safety & Regulatory Guardrails in System Prompt** (`generation/llm.py`):
  - Enhanced system prompt with explicit boundaries against assisting with unlawful actions (e.g., intentional tax evasion or unauthorized employment under student visa status).


### Added
- **Multi-Turn Conversational Memory & Context Window** (`generation/llm.py`, `app.py`):
  - Configurable conversational context window (`CONTEXT_WINDOW_TURNS = 3`) that maintains the last 3 complete dialogue exchanges (up to 6 user/assistant messages) in the prompt before the current RAG-grounded question.
  - Updated `_build_messages()` to ingest `chat_history`, filtering to valid `user` and `assistant` turns so the model can resolve conversational references and handle follow-up inquiries (e.g., "Can you elaborate on that?", "Does this apply to OPT as well?").
  - Threaded `chat_history` through all generation backends: Groq cloud (`_groq_generate`, `_groq_stream`) and Ollama local (`_ollama_generate`, `_ollama_stream`).
  - Added optional `chat_history` parameter to public APIs `generate_answer()` and `stream_answer()` with backward-compatible defaults (`None`).
  - Updated Streamlit chat application (`app.py`) to snapshot prior conversation state before appending current user input, streaming responses with conversational memory without re-fetching RAG context for past turns.

---

## [v1.1.0-intl-student-rag] — 2026-10-08

### Added
- **International Student Knowledge Base & Regulatory Scraper** (`ingestion/scrapers/intl_student_scraper.py`):
  - Targets 36 live-verified government and regulatory portals across IRS, DHS (Study in the States), USCIS, SEC Investor.gov, US TreasuryDirect, FDIC, and CFPB.
  - Zero-dependency runtime fallback using standard library (`urllib.request`, `re`, `html`, `ssl`) with auto-upgrade if `requests` and `bs4` are installed.
  - Built-in PDF compiler generating a unified 349-page reference guide (`output/intl_student_financial_guide.pdf`, 1.2 MB).
- **Expanded Ingestion Pipeline** (`ingestion/pipeline.py` & `ingestion/parsers/pdf_parser.py`):
  - Added `intl` source mapping pointing to the International Student Regulatory Guide.
  - Ingested **1,157 chunks** (698 total pages across SEC, CFPB, Federal Reserve, and International Student guides) into local LanceDB (`data/lancedb`) and cloud Pinecone (`fin-rag`).
- **Domain-Specific Evaluation Benchmark** (`eval/golden_set_intl.json` & `eval/golden_set_combined.json`):
  - 11 dedicated golden Q&A pairs covering F-1/J-1 tax residency (Form 8843 / 1040-NR), Substantial Presence Test, CPT vs OPT boundaries, SSN eligibility, FICA exemptions, permitted passive investing (stocks, ETFs, T-Bills, HYSAs), and prohibited active trading (pattern day trading, crypto mining, active property management).
  - Benchmark evaluation results on LanceDB: **Precision@3: 0.8182**, **Recall@5: 1.0000**, **MRR: 0.8864**, **Hit Rate: 1.0000** (all KPI targets passed).
- **Streamlit UI Enhancements** (`app.py`):
  - Updated title and subtitle to highlight international student financial topics.
  - Added one-click example prompt buttons for F-1 tax residency, CPT/OPT differences, investing in T-Bills/stocks, and SSN/banking.
  - Added support for loading international student evaluation reports in the sidebar KPI drawer.

### Fixed
- **`eval/runner.py`**: Added graceful error handling during generation when LLM judge is disabled, preventing connection dropouts from aborting retrieval evaluation.
- **`ingestion/scrapers/intl_student_scraper.py`**: Added 30-second request timeout, connection retries, and unverified SSL context handling for strict government servers.

---

## [v1.0.0] — 2026-10-05

### Added
- Full production release with hosted Groq cloud LLM fallback, Pinecone cloud vector database, and Streamlit Community Cloud deployment.

---

## [v0.6.1-hosting] — 2026-10-05


### Fixed
- **`generation/llm.py`** — increased `max_tokens` from `1000` → `4096` in both `_groq_stream` and `_groq_generate`; `qwen/qwen3.8-27b` is a thinking model that burns tokens on internal reasoning before answering, silently exhausting the budget and returning an empty response on Streamlit Cloud
- **`generation/llm.py`** — added `<think>…</think>` block filter in `_groq_stream` to strip qwen3 reasoning tokens from the streamed output so users see only the clean answer
- **`generation/llm.py`** — restructured `try/except` in `_groq_stream` to wrap stream *creation* only (not the yield loop); exceptions now surface correctly in `st.write_stream()` instead of being silently swallowed
- **`app.py`** — replaced `allam-2-7b` (niche/preview model) with `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` (confirmed Groq production models); `qwen/qwen3.8-27b` retained as default

---

## [v0.6.0-hosting] — 2026-10-02

### Added
- **Groq cloud LLM backend** (`generation/llm.py`) — dual-backend routing via `LLM_BACKEND` env var
  - `_groq_generate` / `_groq_stream`: Groq SDK, OpenAI-compatible streaming API
  - `_ollama_generate` / `_ollama_stream`: original Ollama paths preserved unchanged
  - Public API (`generate_answer`, `stream_answer`) is backend-agnostic — zero changes to callers
- **`config.py`** — added `llm_backend`, `groq_api_key`, `groq_model` settings fields
- **`app.py`** — backend-aware model selector in sidebar; Groq Cloud / Ollama Local badge
- **`Dockerfile`** — multi-stage build (builder + slim runtime) targeting port 7860 for HF Spaces
- **`.dockerignore`** — excludes `.env`, logs, cache, and git history from the image
- **`README_HF.md`** — Hugging Face Spaces config with YAML frontmatter (`sdk: docker`, `app_port: 7860`)
- **`requirements.txt`** — `groq>=0.9` added; stale duplicate comments removed

### Changed
- Default `LLM_BACKEND` is now `groq` (cloud); set to `ollama` in local `.env` to restore local inference
- `.env.example` rewritten to document all secrets: `GROQ_API_KEY`, `LLM_BACKEND`, `PINECONE_API_KEY`

### Deployment
- Target: **Streamlit Community Cloud** (share.streamlit.io) — free, connects to GitHub, no Docker required
- Dockerfile retained in repo as production-readiness signal for portfolio / future cloud deployments

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