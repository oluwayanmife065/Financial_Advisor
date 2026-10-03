---
title: Personal Finance Literacy Assistant
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# 📊 Personal Finance Literacy RAG Assistant

A grounded personal finance Q&A assistant powered by a **RAG (Retrieval-Augmented Generation)** pipeline over trusted regulatory documents from the SEC, CFPB, and Federal Reserve.

> **Answers are grounded in source documents — not hallucinated.** Every response cites the chunk and document it came from.

## 🏗️ Architecture

```
SEC / CFPB / Fed Reserve PDFs
        ↓
Ingestion Pipeline (chunk → embed → store)
        ↓
Vector Store: LanceDB (local) | Pinecone (cloud)
        ↓
Retrieval: bge-small-en-v1.5 semantic search
        ↓
LLM: Groq (llama-3.1-8b-instant) — ~300 tok/s
        ↓
Streamlit Chat UI
```

## ⚙️ Running Locally

```bash
git clone https://github.com/your-username/financial-advisor
cd financial-advisor
pip install -r requirements.txt

# Copy and fill in your keys
cp .env.example .env

# Run the app
streamlit run app.py
```

## 🔑 Required Secrets (HF Spaces)

Add these in your Space → **Settings → Repository Secrets**:

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key — [console.groq.com](https://console.groq.com) |
| `PINECONE_API_KEY` | Pinecone API key (optional — LanceDB works without it) |

## 📊 Eval Results

| Metric | Score | Target |
|---|---|---|
| Precision@3 | — | ≥ 0.70 |
| Recall@5 | — | ≥ 0.80 |
| MRR | — | ≥ 0.75 |
| Hit Rate | — | ≥ 0.85 |

*Run `python -m eval.runner` to populate.*
