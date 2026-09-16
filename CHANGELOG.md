# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project scaffold (Phase 0)
- PDF parser (`ingestion/parsers/pdf_parser.py`) with metadata extraction
- Parser unit tests (`tests/test_pdf_parser.py`)

---

## [v0.1.0-ingestion] — Phase 1 target

### Planned
- SEC/Investor.gov scraper
- CFPB scraper
- Sentence-aware chunker (512 tokens, 64-token overlap)
- `bge-small-en-v1.5` embedder
- LanceDB ingestion pipeline
- Hash-based idempotent doc IDs

---

## [v0.2.0-retrieval] — Phase 2 target

### Planned
- `BaseRetriever` abstract interface
- LanceDB retriever implementation
- Ollama LLM wrapper + grounding prompt
- CLI chat loop
- JSON Lines query logger

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
