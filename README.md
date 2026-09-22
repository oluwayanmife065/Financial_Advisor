# 📊 Personal Finance Literacy RAG Assistant

A private, production-grade Retrieval-Augmented Generation (RAG) system for learning personal finance concepts from trusted regulatory sources (SEC, CFPB, Federal Reserve).

> Built as an end-to-end ML engineering portfolio project demonstrating ingestion pipelines, vector retrieval, evaluation harnesses, and full observability.

---

## 🏗️ Architecture

```
Source PDFs (SEC, CFPB, Federal Reserve)
    ↓
Ingestion Pipeline (parse → clean → chunk → embed)
    ↓
Vector Store: LanceDB (local) | Pinecone (cloud)
    ↓
Retrieval Engine (semantic search + metadata filters)
    ↓
LLM Layer (Ollama — local inference)
    ↓
Chat Interface (CLI → Streamlit)
    ↓
Eval Harness (golden Q&A set, precision/recall/MRR)
```

---

## 📂 Project Structure

```
├── ingestion/
│   ├── parsers/
│   │   ├── pdf_parser.py          # PDF → Document extraction (PyMuPDF)
│   │   └── text_parser.py         # Plain text parser
│   ├── scrapers/
│   │   ├── base.py                # Abstract BaseScraper
│   │   └── sec_scraper.py         # SEC/Investor.gov scraper
│   ├── chunker.py                 # Sentence-aware chunking
│   ├── document.py                # Document & Chunk dataclasses
│   ├── embedder.py                # SentenceTransformers wrapper
│   └── pipeline.py                # Ingestion orchestrator
│
├── retrieval/
│   ├── base.py                    # Abstract BaseRetriever
│   ├── lancedb_retriever.py       # LanceDB implementation
│   └── pinecone_retriever.py      # Pinecone implementation
│
├── generation/
│   └── llm.py                     # Ollama wrapper + prompt template
│
├── eval/
│   ├── golden_set.json            # Hand-written Q&A pairs
│   ├── metrics.py                 # Precision@k, Recall@k, MRR, Hit Rate
│   ├── llm_judge.py               # Ollama answer scorer
│   └── runner.py                  # Full eval harness
│
├── query_logging/
│   └── query_logger.py            # JSON Lines query logger
│
├── tests/                         # pytest test suite
├── config.py                      # Pydantic Settings
├── main.py                        # CLI chat loop
├── app.py                         # Streamlit UI
└── PROJECT_BRIEF.md               # Full project brief & interview narrative
```

---

## 🚀 Roadmap & Current Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** — Scaffolding | Project structure, config, data models | ✅ Complete |
| **Phase 1** — Ingestion | Parsers, scrapers, chunker, embedder, LanceDB pipeline | ✅ Complete |
| **Phase 2** — Retrieval + Chat | BaseRetriever interface, LanceDB retriever, CLI chat loop | ✅ Complete |
| **Phase 3** — Eval Harness | Golden Q&A set, retrieval & generation metrics, LLM judge | ✅ Complete |
| **Phase 4** — Pinecone Benchmark | Pinecone retriever, side-by-side benchmark vs LanceDB | ⬚ Next Phase |
| **Phase 5** — Polish + UI | Streamlit app, full README case study | ⬚ Planned |

---

## ✅ What's Built So Far

### Data Models (`ingestion/document.py`)
- `Document` dataclass — represents a parsed source page with full metadata
- `Chunk` dataclass — represents an embeddable text chunk with parent lineage
- Deterministic `doc_id` via `sha256(url + date_ingested)` — prevents duplicate ingestion

### PDF Parser & Ingestion Pipeline (`ingestion/`)
- Extracts clean text from PDFs using PyMuPDF (`fitz`)
- Sentence-aware chunker (512 tokens, 64-token overlap)
- `bge-small-en-v1.5` embeddings via SentenceTransformers
- Automated pipeline indexing 488 chunks into LanceDB across SEC, CFPB, and Federal Reserve

### Retrieval & Grounded Generation (`retrieval/`, `generation/`, `main.py`)
- Embedded vector search via LanceDB
- Grounded generation using local Ollama LLMs with strict anti-hallucination system prompt
- CLI interactive chat loop (`python3.11 main.py`) with real-time latency reporting and source citations
- Structured observability: JSON Lines query logger (`query_logging/query_logger.py`)

### 🧪 Evaluation Harness (`eval/`)
- **Golden Benchmark (`eval/golden_set.json`)**: 25 hand-crafted Q&A pairs with ground-truth reference answers and chunk keywords across SEC, Fed, and CFPB.
- **Retrieval Metrics (`eval/metrics.py`)**: Pure functions computing Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and Hit Rate.
- **LLM Judge (`eval/llm_judge.py`)**: Independent scoring of Answer Relevance and Context Faithfulness using OpenAI GPT-4o-mini with retry logic and JSON code-fence stripping.
- **Eval Runner CLI (`eval/runner.py`)**: Orchestrator executing the benchmark, computing p50/p95 latency percentiles, rendering terminal summary tables, and exporting `eval_report.json`.

#### 📊 Live Benchmark Results (25-Question Test Set)
| Metric | System Score | Project Target | Status |
|---|---|---|---|
| **Precision@3** | **0.8400** | $\ge 0.70$ | **✅ PASSED** (+14% above target) |
| **Recall@5** | **1.0000** | $\ge 0.80$ | **✅ PASSED** (100% recall) |
| **Mean Reciprocal Rank (MRR)** | **0.9800** | $\ge 0.75$ | **✅ PASSED** (Relevant chunk almost always #1) |
| **Hit Rate** | **1.0000** | $\ge 0.85$ | **✅ PASSED** (100% hit rate) |
| **Retrieval Latency (p50 / p95)** | **313 ms / 1,020 ms** | $< 100\text{ ms} / < 300\text{ ms}$ | Fast local LanceDB vector search |
| **End-to-End Latency (p50 / p95)** | **13.0 s / 18.2 s** | $< 10\text{ s}$ | Local 7B LLM on Mac hardware |

### Test Suite (`tests/`)
- **111 unit tests passing** across parsers, chunker, embedder, pipeline, retrievers, query logger, retrieval metrics, LLM judge, and runner harness.

---

## 📚 Source Documents

Source PDFs are **not tracked in Git** (excluded via `.gitignore`). To set up locally, create the following folder structure and download the documents:

```
ingestion/sources/pdfs/
├── cfpb/       # Consumer Financial Protection Bureau publications
├── fed/        # Federal Reserve financial literacy curriculum
└── sec/        # SEC/Investor.gov investor education materials
```

### Sources
| Folder | Source | Content |
|---|---|---|
| `cfpb/` | [CFPB](https://www.consumerfinance.gov/) | Your Money Your Goals — Financial Empowerment Toolkit |
| `fed/` | [Federal Reserve](https://www.federalreserveeducation.org/) | Building Wealth curriculum, investment & risk lessons |
| `sec/` | [Investor.gov](https://www.investor.gov/) | Saving & investing fundamentals |

---

## 🛠️ Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Embeddings** | `bge-small-en-v1.5` (SentenceTransformers) | Strong MTEB retrieval scores, lightweight (384-dim) |
| **Vector DB (local)** | LanceDB | Embedded, zero-infra, fast local vector search |
| **Vector DB (cloud)** | Pinecone | Managed cloud vector DB for side-by-side benchmark (Phase 4) |
| **LLM (local)** | Ollama (`qwen2.5:7b` / `llama3.2:3b`) | Fully local, private, no inference cost |
| **Eval Judge** | OpenAI GPT-4o-mini | Independent judge avoids self-evaluation bias |
| **Config** | Pydantic Settings | Type-safe env var management via `.env` |
| **Testing** | pytest | 111 unit tests with full mocking and isolation |
| **PDF Parsing** | PyMuPDF (`fitz`) | Fast, reliable text extraction and cleaning |

---

## 🏷️ Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

| Tag | Milestone |
|---|---|
| `v0.0.0-scaffold` | Phase 0 complete — project structure, data models, PDF parser + tests |
| `v0.1.0-ingestion` | Phase 1 complete — chunker, embedder, pipeline, LanceDB retriever, tests |
| `v0.2.0-retrieval` | Phase 2 complete — Ollama LLM wrapper, CLI chat loop, query logger, tests |
| `v0.3.0-eval` | Phase 3 complete — Golden Q&A benchmark (25 questions), retrieval metrics, LLM judge, eval runner, 111 tests |

---

## 📝 License

Personal learning project — not intended for financial advice.
