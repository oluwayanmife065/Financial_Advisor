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
| **Phase 3** — Eval Harness | Golden Q&A set, retrieval & generation metrics, LLM judge | 🔧 In Progress |
| **Phase 4** — Pinecone Benchmark | Pinecone retriever, side-by-side benchmark vs LanceDB | ⬚ Planned |
| **Phase 5** — Polish + UI | Streamlit app, full README case study | ⬚ Planned |

---

## ✅ What's Built So Far

### Data Models (`ingestion/document.py`)
- `Document` dataclass — represents a parsed source page with full metadata
- `Chunk` dataclass — represents an embeddable text chunk with parent lineage
- Deterministic `doc_id` via `sha256(url + date_ingested)` — prevents duplicate ingestion

### PDF Parser (`ingestion/parsers/pdf_parser.py`)
- Extracts clean text from PDFs using PyMuPDF (`fitz`)
- One `Document` per non-empty page (skips pages with < 50 characters)
- Text cleaning: hyphenated line-break rejoining, blank line collapsing
- Folder-level batch parsing with automatic source label inference
- Source label mapping: `sec` → "SEC/Investor.gov", `cfpb` → "CFPB", `fed` → "Federal Reserve"

### Test Suite (`tests/test_pdf_parser.py`)
- Full coverage of `parse_pdf`, `parse_pdf_folder`, `infer_source_label`, and `_clean_text`
- Tests for edge cases: missing files, empty folders, pages below content threshold

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
| **Embeddings** | `bge-small-en-v1.5` (SentenceTransformers) | Strong MTEB retrieval scores, lightweight |
| **Vector DB (local)** | LanceDB | Embedded, zero-infra, fast local dev |
| **Vector DB (cloud)** | Pinecone | Managed, production feel, benchmark comparison |
| **LLM** | Ollama (`llama3.2:3b` / `mistral:7b`) | Fully local, private, no API cost |
| **Config** | Pydantic Settings | Type-safe env var management |
| **Testing** | pytest | Standard Python test framework |
| **PDF Parsing** | PyMuPDF (`fitz`) | Fast, reliable text extraction |

---

## 🏷️ Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

| Tag | Milestone |
|---|---|
| `v0.0.0-scaffold` | Phase 0 complete — project structure, data models, PDF parser + tests |
| `v0.1.0-ingestion` | Phase 1 complete — chunker, embedder, pipeline, LanceDB retriever, tests |
| `v0.2.0-retrieval` | Phase 2 complete — Ollama LLM wrapper, CLI chat loop, query logger, tests |

---

## 📝 License

Personal learning project — not intended for financial advice.
