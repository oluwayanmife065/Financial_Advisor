"""
Streamlit Chat UI — Personal Finance Literacy RAG Assistant.

Run with:
    streamlit run app.py

Features:
  - Persistent chat history (session state)
  - Live streaming token output via st.write_stream() + Ollama stream=True
  - Collapsible source citation expanders per assistant message
  - Retrieval / Generation / Total latency badges per message
  - Sidebar: model selector, top-k slider, corpus stats, eval KPI badges,
    clear-chat button, and optional raw query log viewer
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── ensure project root is importable when run directly from the project dir ──
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import settings
from ingestion.embedder import embed_query
from retrieval.lancedb_retriever import search, count
from generation.llm import stream_answer
from query_logging.query_logger import log_query


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE = "📊 Personal Finance Literacy Assistant"
APP_SUBTITLE = "Grounded answers from SEC, CFPB, and Federal Reserve documents."
DEFAULT_MODELS = ["qwen2.5:7b", "llama3.2:3b", "mistral:7b", "gemma3:4b"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: discover locally available Ollama models
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _get_ollama_models() -> list[str]:
    """Return a list of locally pulled Ollama model names, fallback to defaults."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            # First line is a header (NAME  ID  SIZE  MODIFIED)
            models = [
                line.split()[0]
                for line in lines[1:]
                if line.strip() and not line.startswith("failed")
            ]
            if models:
                # Ensure default model appears first
                if settings.ollama_model in models:
                    models.remove(settings.ollama_model)
                    models.insert(0, settings.ollama_model)
                return models
    except Exception:
        pass
    return DEFAULT_MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load latest eval report for sidebar KPI badges
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _read_report_cached(path_str: str, mtime: float) -> dict | None:
    """Read and parse eval report JSON, cached by path and mtime."""
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _load_eval_report() -> dict | None:
    """Load eval_report.json if it exists, automatically refreshing when modified."""
    report_path = Path(settings.eval_report_path)
    if report_path.exists():
        try:
            mtime = report_path.stat().st_mtime
            return _read_report_cached(str(report_path), mtime)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state():
    """Initialise persistent session state keys on first run."""
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "chunks": list[Chunk]|None, "latency": dict|None}
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[str, int, bool]:
    """
    Render the sidebar and return user-selected controls.

    Returns:
        Tuple of (selected_model, top_k, show_sources).
    """
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model selector
        available_models = _get_ollama_models()
        selected_model = st.selectbox(
            "LLM Model",
            options=available_models,
            index=0,
            help="Ollama models available locally. Pull more with `ollama pull <name>`.",
        )

        # Retrieval top-k
        top_k = st.slider(
            "Retrieved Chunks (top-k)",
            min_value=1, max_value=10,
            value=settings.top_k,
            help="How many context chunks to retrieve from LanceDB per query.",
        )

        # Source citation toggle
        show_sources = st.toggle(
            "Show source citations",
            value=st.session_state.show_sources,
        )
        st.session_state.show_sources = show_sources

        st.divider()

        # ── Corpus stats ──
        st.subheader("🗄️ Vector Store")
        chunk_count = count()
        if chunk_count > 0:
            st.metric("Chunks indexed", f"{chunk_count:,}", help="LanceDB (local)")
        else:
            st.warning(
                "LanceDB is empty.\n\n"
                "Run the ingestion pipeline first:\n"
                "```\npython3.11 -m ingestion.pipeline\n```"
            )

        st.caption(f"Embedder: `{settings.embedding_model}`")
        st.caption(f"DB path: `{settings.lancedb_path}`")

        st.divider()

        # ── Eval KPI badges ──
        report = _load_eval_report()
        if report:
            st.subheader("📊 Eval Benchmark")
            summary = report.get("aggregate", {}).get("summary_metrics", {})
            kpi_map = {
                "Precision@3": "precision_at_3",
                "Recall@5": "recall_at_5",
                "MRR": "mrr",
                "Hit Rate": "hit_rate",
            }
            cols = st.columns(2)
            for i, (label, key) in enumerate(kpi_map.items()):
                info = summary.get(key, {})
                score = info.get("score")
                passed = info.get("passed", False)
                if score is not None:
                    delta_color = "normal" if passed else "inverse"
                    cols[i % 2].metric(
                        label=label,
                        value=f"{score:.2f}",
                        delta="✅ PASS" if passed else "❌ FAIL",
                        delta_color=delta_color,
                    )
            cfg = report.get("configuration", {})
            st.caption(
                f"Run: {report.get('timestamp_end', '—')[:10]}  |  "
                f"n={len(report.get('per_query_results', []))}  |  "
                f"retriever={cfg.get('retriever', '—')}"
            )
        else:
            st.caption("No eval report found. Run `python3.11 -m eval.runner` to generate one.")

        st.divider()

        # ── Clear chat ──
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # ── Query log viewer ──
        with st.expander("📋 Query Log (latest 5)"):
            log_path = Path(settings.log_file)
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                last_five = lines[-5:] if len(lines) >= 5 else lines
                for raw in reversed(last_five):
                    try:
                        entry = json.loads(raw)
                        st.json({
                            "query": entry.get("query", ""),
                            "retrieval_ms": entry.get("retrieval_latency_ms"),
                            "generation_ms": entry.get("generation_latency_ms"),
                            "model": entry.get("model"),
                            "k": entry.get("k"),
                        })
                    except Exception:
                        st.code(raw)
            else:
                st.caption("No queries logged yet.")

    return selected_model, top_k, show_sources


# ─────────────────────────────────────────────────────────────────────────────
# Chat history rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_history(show_sources: bool):
    """Render all previous messages from session state."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Source citations and latency — only on assistant turns
            if role == "assistant":
                chunks = msg.get("chunks") or []
                latency = msg.get("latency") or {}
                _render_latency_badges(latency)
                if show_sources and chunks:
                    _render_source_expander(chunks)


def _render_latency_badges(latency: dict):
    """Show coloured latency info in a single tight row."""
    if not latency:
        return
    r_ms = latency.get("retrieval_ms", 0)
    g_ms = latency.get("generation_ms", 0)
    t_ms = latency.get("total_ms", 0)
    st.caption(
        f"⏱️ Retrieval **{r_ms:.0f} ms** · "
        f"Generation **{g_ms / 1000:.2f} s** · "
        f"Total **{t_ms / 1000:.2f} s**"
    )


def _render_source_expander(chunks):
    """Render a collapsible expander listing retrieved source chunks."""
    with st.expander(f"📚 {len(chunks)} source chunk(s) used"):
        for i, chunk in enumerate(chunks, 1):
            source_label = f"[{i}] **{chunk.source}** — {chunk.section}"
            preview = chunk.text.replace("\n", " ").strip()[:250]
            st.markdown(source_label)
            st.caption(f"> {preview}…")
            if chunk.url:
                st.markdown(f"[🔗 Source document]({chunk.url})")
            if i < len(chunks):
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Query execution
# ─────────────────────────────────────────────────────────────────────────────

def _handle_query(
    query: str,
    model: str,
    top_k: int,
    show_sources: bool,
):
    """
    Run the full RAG pipeline for a user query and append results to session state.

    Pipeline:
      1. embed_query()        → 384-dim vector
      2. lancedb search()     → top-k Chunks
      3. stream_answer()      → token generator → st.write_stream()
      4. log_query()          → append to query_log.jsonl
      5. Append to session state (user + assistant turns)
    """
    # ── 1. Append user message to chat ──
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── 2. Retrieval ──
    with st.spinner("🔍 Searching knowledge base…"):
        retrieval_start = time.perf_counter()
        try:
            query_vector = embed_query(query)
            chunks = search(query_vector, k=top_k)
        except RuntimeError as exc:
            st.error(f"❌ Retrieval failed: {exc}")
            return
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # ── 3. Streaming generation ──
    generation_start = time.perf_counter()
    with st.chat_message("assistant"):
        try:
            token_stream = stream_answer(query, chunks, model=model)
            answer = st.write_stream(token_stream)
        except RuntimeError as exc:
            st.error(
                f"❌ Generation failed: {exc}\n\n"
                "Is Ollama running? `ollama serve` in a terminal."
            )
            return

        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = retrieval_ms + generation_ms

        latency = {
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round(total_ms, 1),
        }

        _render_latency_badges(latency)
        if show_sources and chunks:
            _render_source_expander(chunks)

    # ── 4. Observability logging ──
    try:
        log_query(
            query=query,
            answer=answer,
            chunks=chunks,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            model=model,
            retriever_name="lancedb",
        )
    except Exception:
        pass  # Never let logging crash the UI

    # ── 5. Persist to session state ──
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "latency": latency,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    # ── Header ──
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()

    # ── Sidebar ──
    selected_model, top_k, show_sources = _render_sidebar()

    # ── Render existing conversation ──
    _render_chat_history(show_sources)

    # ── Empty-state prompt suggestions ──
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        example_questions = [
            "What is the difference between a stock and a bond?",
            "How does compound interest work?",
            "What is a 401(k) and how much should I contribute?",
            "What are the risks of investing in individual stocks?",
            "How do I build an emergency fund?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            if cols[i % 2].button(q, use_container_width=True, key=f"suggestion_{i}"):
                _handle_query(q, model=selected_model, top_k=top_k, show_sources=show_sources)
                st.rerun()

    # ── Chat input ──
    if prompt := st.chat_input("Ask a personal finance question…"):
        _handle_query(prompt, model=selected_model, top_k=top_k, show_sources=show_sources)


if __name__ == "__main__":
    main()

