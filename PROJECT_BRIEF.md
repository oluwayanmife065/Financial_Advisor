# 📊 Personal Finance Literacy RAG Assistant — Project Brief

> **Purpose**: A private, personal learning tool AND a production-grade ML engineering portfolio project targeting AI/ML Engineer & Data Engineer roles.

---

## 🏗️ System Architecture Overview

```
Raw Sources (SEC, FINRA, Fed Reserve, CFPB, Notes)
    ↓
Ingestion Pipeline (scrape → clean → chunk → embed)
    ↓
Vector Store Layer: LanceDB (local) | Pinecone (cloud)
    ↓
Retrieval Engine (semantic search + metadata filters)
    ↓
LLM Layer (Ollama — local inference)
    ↓
Chat Interface (CLI → Streamlit)
    ↓
Eval Harness (golden Q&A set, precision/recall/MRR)

Logging & Observability (latency, chunk traces, query logs)
```

---

## ✅ Finalized Architecture Decisions

| Dimension | Choice | Rationale |
|---|---|---|
| **Vector DBs** | LanceDB (local) + Pinecone (cloud) | Benchmark both; Weaviate open for later |
| **Embeddings** | `bge-small-en-v1.5` (SentenceTransformers) | Better MTEB retrieval scores than MiniLM, same API |
| **LLM** | Ollama — `llama3.2:3b` or `mistral:7b` | Local, private, no API cost |
| **Eval Judge** | Ollama (local) | Fully private; tradeoff documented in case study |
| **Chat UI** | CLI first → Streamlit later | Speed now, portfolio polish later |
| **Ingestion Scope (Phase 1)** | SEC/Investor.gov + CFPB | Narrow, trusted, structured — expand after eval harness validated |

---

## 📚 Phase Breakdown

### Phase 0 — Project Scaffolding (~30 min)
- Create full folder structure
- `config.py` with pydantic-settings
- `ingestion/document.py` — `Document` and `Chunk` dataclasses
- `requirements.txt` with pinned dependencies

### Phase 1 — Ingestion Pipeline (2–3 days)
- Sources: SEC/Investor.gov + CFPB
- Steps: Fetch → Parse → Clean → Chunk → Embed → Store (LanceDB)
- Metadata per chunk: `source`, `section`, `url`, `date_ingested`, `chunk_index`, `doc_id`
- Chunking: sentence-aware, 512 tokens, 64-token overlap
- Idempotent: `doc_id = sha256(url + date_ingested)` — no duplicates on re-run

### Phase 2 — Retrieval Layer + Basic Chat Loop (1–2 days)
- Abstract `BaseRetriever` interface → `retrieve(query, k) -> list[Chunk]`
- LanceDB retriever implementation
- Ollama wrapper with grounding system prompt
- CLI chat loop
- Structured per-query logging (JSON Lines)

### Phase 3 — Eval Harness (2–3 days)
- 20–30 hand-written golden Q&A pairs (`eval/golden_set.json`)
- Retrieval metrics: Precision@k, Recall@k, MRR, Hit Rate
- Generation metrics: Answer relevance + Faithfulness (Ollama judge)
- `eval/runner.py` → prints full metrics table, saves `eval_report.json`

### Phase 4 — Pinecone + Benchmark (1–2 days)
- `retrieval/pinecone_retriever.py` implementing `BaseRetriever`
- Run exact same eval harness with `--retriever [lancedb|pinecone]` flag
- Produce side-by-side benchmark table for case study

### Phase 5 — Polish (2–3 days)
- `pytest` tests for chunker, embedder, retrievers, metrics
- Full `README.md` case study writeup
- Streamlit UI (`app.py`)

### Phase 6 — International Student Finetuning & Multi-Store Ingestion
- Regulatory scraper covering 36 official portals (IRS Pub 519, DHS Study in the States, USCIS, SEC)
- Generated unified 349-page PDF guide (`output/intl_student_financial_guide.pdf`)
- Ingested 1,157 chunks across local LanceDB and cloud Pinecone
- Domain evaluation benchmark (`eval/golden_set_intl.json`) achieving 100% Hit Rate and Recall@5

### Phase 7 — Multi-Turn Memory & Contextual Query Reformulation
- Conversational context window tracking the last 3 Q&A turns (6 messages)
- Query reformulation engine (`reformulate_query`): rewrites follow-up inquiries into self-contained search queries before vector embedding

### Phase 8 — Deployment & Demo Readiness
- Academic and portfolio disclaimer banner in Streamlit UI
- Domain-specific AI safety refusal guardrails in generation prompt
- Observability and trace telemetry drawer for latency, retriever, and model tracking

---

## 🎯 KPIs & Success Metrics

### Retrieval Quality
| KPI | Target | How to Measure |
|---|---|---|
| **Precision@3** | ≥ 0.70 | % of top-3 retrieved chunks that are relevant |
| **Recall@5** | ≥ 0.80 | % of known relevant chunks returned in top-5 |
| **MRR** | ≥ 0.75 | Mean Reciprocal Rank across golden set |
| **Hit Rate** | ≥ 0.85 | % of queries where ≥1 correct chunk is in top-k |

### Generation Quality
| KPI | Target | How to Measure |
|---|---|---|
| **Answer Relevance** | ≥ 0.80 | Ollama LLM-as-judge score (0–1) |
| **Faithfulness** | ≥ 0.85 | Answer grounded in retrieved context |
| **ROUGE-L** | > 0.30 | Overlap with reference answers |

### System Performance
| KPI | Target | How to Measure |
|---|---|---|
| **Retrieval latency p50** | < 100ms | Logged per query |
| **Retrieval latency p95** | < 300ms | Logged per query |
| **Generation latency p50** | < 5s | Logged per query |
| **End-to-end latency p95** | < 10s | Logged per query |

### Engineering Quality
| Signal | Evidence |
|---|---|
| Reproducibility | Hash-based doc IDs, versioned embeddings |
| Modularity | Abstract retriever interface — swap DBs without touching app layer |
| Observability | Every query logged with chunk traces and latency |
| Benchmarkability | Same eval harness runs against both DBs |

---

## 🏗️ Project Folder Structure

```
financial_advisor/
│
├── ingestion/
│   ├── scrapers/
│   │   ├── base.py              # Abstract BaseScraper
│   │   ├── sec_scraper.py       # SEC/Investor.gov
│   │   └── cfpb_scraper.py      # CFPB
│   ├── chunker.py               # Sentence-aware chunking
│   ├── embedder.py              # SentenceTransformers wrapper
│   ├── document.py              # Document & Chunk dataclasses
│   └── pipeline.py              # Orchestrator
│
├── retrieval/
│   ├── base.py                  # Abstract BaseRetriever
│   ├── lancedb_retriever.py
│   └── pinecone_retriever.py
│
├── generation/
│   └── llm.py                   # Ollama wrapper + prompt template
│
├── eval/
│   ├── golden_set.json          # 20–30 Q&A pairs
│   ├── metrics.py               # Precision@k, Recall@k, MRR, Hit Rate
│   ├── llm_judge.py             # Ollama answer scorer
│   └── runner.py                # Full eval harness
│
├── logging/
│   └── query_logger.py          # JSON Lines logger
│
├── config.py                    # Pydantic Settings
├── main.py                      # CLI chat loop
├── app.py                       # Streamlit UI (Phase 5)
├── requirements.txt
├── .env.example
├── PROJECT_BRIEF.md             # ← You are here
├── README.md                    # Case study writeup
│
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_retrievers.py
    └── test_metrics.py
```

---

## 🏆 Best Coding Practices

- **Typed Python**: `dataclasses` or `pydantic` for all data models
- **Abstract interfaces**: `BaseRetriever`, `BaseScraper` — swapping implementations is trivial
- **Config management**: `pydantic-settings` for all env vars and hyperparameters
- **Logging**: `loguru` for structured, JSON-friendly logs
- **Tests**: `pytest` — test chunker output, embedding shapes, retrieval contracts, metric calculations
- **Idempotency**: Hash-based doc IDs prevent duplicate ingestion
- **Docstrings**: Google-style on every class and public method

### Git Hygiene
- Conventional commits: `feat:`, `fix:`, `eval:`, `docs:`
- Tag versions: `v0.1-ingestion`, `v0.2-retrieval`, `v0.3-eval`
- Include `CHANGELOG.md`

---

## 💼 Interview Positioning

### What This Project Demonstrates
| Skill | Evidence |
|---|---|
| **ML Engineering** | End-to-end RAG pipeline, embedding selection, chunking strategy |
| **Data Engineering** | Ingestion pipeline, metadata schema design, structured logging |
| **Evaluation Mindset** | Golden test set, quantified retrieval + generation metrics |
| **Systems Thinking** | Abstract retriever interface, apples-to-apples DB benchmark |
| **Production Habits** | Typed models, config management, structured logs, tests |
| **Communication** | Written case study with tradeoffs, results tables, architecture diagram |

### Case Study Narrative Arc
1. **Problem**: I want to learn personal finance but can't trust random internet advice
2. **Solution**: Build a private RAG system over trusted regulatory sources (SEC, CFPB)
3. **Engineering Challenge**: How do you evaluate whether a RAG system actually works?
4. **What I did**: Built a golden Q&A eval harness, benchmarked LanceDB vs Pinecone, measured latency at p50/p95
5. **Results**: [your actual numbers — precision, recall, latency]
6. **Tradeoffs**: LanceDB (embedded, zero infra, fast local dev) vs Pinecone (managed, production feel, higher latency)
7. **What I did next**: Implemented conversational memory and query reformulation to prevent follow-up retrieval blindspots, and added disclaimer & refusal safety guardrails for regulatory compliance
8. **Future directions**: Cross-encoder reranking (ColBERT / BGE-Reranker), hybrid dense/sparse search (BM25 + BGE)

### Key Interview Phrases
- *"I defined a golden test set to measure retrieval precision/recall before relying on qualitative impressions"*
- *"I abstracted the retriever behind an interface so I could benchmark LanceDB vs Pinecone without changing the application layer"*
- *"I noticed standard RAG breaks on conversational follow-ups because the second question lacks entity context, so I implemented a query reformulation step before vector embedding"*
- *"Every query is logged with chunk traces and latency so I can debug retrieval failures and track regression in real time"*
- *"I implemented prompt-level safety refusal guardrails and UI disclaimers to clearly delineate educational literacy from licensed legal/tax advice"*

---

## 🔧 Key Dependencies

```
sentence-transformers>=2.7    # Embeddings (bge-small-en-v1.5)
lancedb>=0.10                 # Local vector store
pinecone-client>=3.0          # Cloud vector store
groq>=0.9                     # Cloud LLM inference
ollama>=0.3                   # Local LLM inference
pydantic-settings>=2.0        # Config management
loguru>=0.7                   # Structured logging
pytest>=8.0                   # Testing
streamlit>=1.35               # UI
```

---

*Last updated: 2026-10-09*


