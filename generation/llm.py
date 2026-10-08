"""
LLM Generation Module — supports Groq (cloud) and Ollama (local) backends.

Backend is selected via the LLM_BACKEND environment variable:
  - "groq"   → Groq cloud API (default for Streamlit Cloud hosting)
  - "ollama" → Local Ollama server (local development)

This module handles:
  1. Grounding system prompt definition
  2. Context formatting from retrieved Chunks
  3. Prompt construction with strict anti-hallucination instructions
  4. Backend-routed chat completion (streaming and non-streaming)
"""

from ingestion.document import Chunk
from config import settings


SYSTEM_PROMPT = """You are a knowledgeable, objective personal finance literacy assistant.
Your goal is to help the user understand financial concepts and build financial capability.

CRITICAL INSTRUCTIONS:
1. Base your answer STRICTLY on the provided Context Chunks. Do not introduce outside information or fabricate facts.
2. If the provided context does not contain enough information to answer the question, clearly state: "I do not have enough information in the provided documents to answer this question."
3. Be direct, clear, and educational in your explanation.
4. Where helpful, reference the source and section (e.g. "According to CFPB...") so the user knows where the information originated.
5. Provide educational context only; do not provide personalized financial, legal, or investment advice.
"""

# ── Groq models available on current key ──
GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "allam-2-7b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def format_context(chunks: list[Chunk]) -> str:
    """
    Format a list of retrieved Chunks into a clean string block for the LLM.

    Each chunk includes its source, section, and text body.

    Args:
        chunks: List of retrieved Chunk objects.

    Returns:
        Formatted context string. If chunks is empty, returns an explanatory placeholder.
    """
    if not chunks:
        return "No context chunks provided."

    formatted_sections = []
    for i, chunk in enumerate(chunks, 1):
        formatted_sections.append(
            f"--- Chunk [{i}] | Source: {chunk.source} | Section: {chunk.section} ---\n"
            f"{chunk.text.strip()}"
        )

    return "\n\n".join(formatted_sections)


def build_user_message(query: str, chunks: list[Chunk]) -> str:
    """
    Construct the final user message combining retrieved context and the query.

    Args:
        query: The user's question.
        chunks: List of retrieved context Chunks.

    Returns:
        Formatted string for the user turn in the chat message list.
    """
    context_str = format_context(chunks)
    return (
        f"Context Chunks:\n"
        f"{context_str}\n\n"
        f"User Question:\n"
        f"{query.strip()}\n\n"
        f"Answer based solely on the context above:"
    )


def _build_messages(
    query: str,
    chunks: list[Chunk],
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """
    Build the messages list for either backend.

    Injects up to the last ``CONTEXT_WINDOW_TURNS`` Q&A pairs from
    *chat_history* (alternating user/assistant) before the current user
    turn so the model can reference previous responses in multi-turn
    conversations.

    Args:
        query: The current user question.
        chunks: Retrieved context chunks for this turn.
        chat_history: Optional list of prior message dicts (each with
            ``role`` and ``content`` keys) from session state.  Only
            "user" and "assistant" roles are forwarded; system messages
            are skipped.  Defaults to an empty history when None.

    Returns:
        Ordered list of chat message dicts ready to send to any
        OpenAI-compatible completions endpoint.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Inject conversation history (last N complete Q&A turns) ──
    if chat_history:
        # Filter to user/assistant only, take the last CONTEXT_WINDOW_TURNS * 2 messages
        history_turns = [
            m for m in chat_history if m.get("role") in ("user", "assistant")
        ]
        window = history_turns[-(CONTEXT_WINDOW_TURNS * 2):]
        for turn in window:
            messages.append({"role": turn["role"], "content": turn["content"]})

    # ── Current user turn (with RAG context) ──
    messages.append({"role": "user", "content": build_user_message(query, chunks)})
    return messages


# Number of prior Q&A pairs to keep in the model's context window
CONTEXT_WINDOW_TURNS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Groq backend
# ─────────────────────────────────────────────────────────────────────────────

def _groq_generate(
    query: str,
    chunks: list[Chunk],
    model: str,
    temperature: float,
    chat_history: list[dict] | None = None,
) -> str:
    """Generate a full answer via the Groq cloud API."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq package not installed. Run: pip install groq>=0.9") from exc

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit Cloud Secrets."
        )

    import re
    import time

    client = Groq(api_key=settings.groq_api_key)
    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_build_messages(query, chunks, chat_history),
                temperature=temperature,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "Rate limit" in err_str or "rate_limit_exceeded" in err_str) and attempt < max_retries - 1:
                # Extract suggested wait time from error message if available (e.g. "try again in 23.1s")
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait_sec = float(match.group(1)) + 1.0 if match else (2 ** attempt) * 5.0
                print(f"\n⏳ Groq rate limit hit. Pausing {wait_sec:.1f}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait_sec)
                continue
            raise RuntimeError(f"Groq generation failed with model '{model}': {exc}") from exc


def _groq_stream(
    query: str,
    chunks: list[Chunk],
    model: str,
    temperature: float,
    chat_history: list[dict] | None = None,
):
    """Stream tokens from the Groq cloud API."""
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq package not installed. Run: pip install groq>=0.9") from exc

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or Streamlit Cloud Secrets."
        )

    client = Groq(api_key=settings.groq_api_key)
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=_build_messages(query, chunks, chat_history),
            temperature=temperature,
            max_tokens=4096,   # Increased: qwen3 thinking models consume tokens before answering
            stream=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq streaming failed with model '{model}': {exc}") from exc

    in_think_block = False
    for chunk_resp in stream:
        token = chunk_resp.choices[0].delta.content
        if not token:
            continue
        # Filter out <think>...</think> reasoning tokens emitted by qwen3-series models
        if "<think>" in token:
            in_think_block = True
        if in_think_block:
            if "</think>" in token:
                in_think_block = False
            continue
        yield token


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_generate(
    query: str,
    chunks: list[Chunk],
    model: str,
    temperature: float,
    chat_history: list[dict] | None = None,
) -> str:
    """Generate a full answer via the local Ollama server."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama package not installed. Run: pip install ollama>=0.3") from exc

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=model,
            messages=_build_messages(query, chunks, chat_history),
            options={"temperature": temperature},
        )
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content.strip()
        elif isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip()
        return str(response).strip()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama generation failed with model '{model}' at {settings.ollama_base_url}: {exc}"
        ) from exc


def _ollama_stream(
    query: str,
    chunks: list[Chunk],
    model: str,
    temperature: float,
    chat_history: list[dict] | None = None,
):
    """Stream tokens from the local Ollama server."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("ollama package not installed. Run: pip install ollama>=0.3") from exc

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        stream = client.chat(
            model=model,
            messages=_build_messages(query, chunks, chat_history),
            options={"temperature": temperature},
            stream=True,
        )
        for chunk_resp in stream:
            if hasattr(chunk_resp, "message") and hasattr(chunk_resp.message, "content"):
                token = chunk_resp.message.content
            elif isinstance(chunk_resp, dict):
                token = chunk_resp.get("message", {}).get("content", "")
            else:
                token = ""
            if token:
                yield token
    except Exception as exc:
        raise RuntimeError(
            f"Ollama streaming failed with model '{model}' at {settings.ollama_base_url}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public API — backend-routed
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[Chunk],
    model: str | None = None,
    temperature: float = 0.1,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Generate an answer to the query given retrieved context chunks.

    Routes to Groq or Ollama based on settings.llm_backend.

    Args:
        query: User's question string.
        chunks: List of retrieved Chunk objects to ground the answer.
        model: Optional model name override. Defaults to the active backend's default model.
        temperature: Sampling temperature. Defaults to 0.1 for consistent, grounded output.
        chat_history: Optional list of prior session message dicts (role + content) to
            inject as conversation history.  The last ``CONTEXT_WINDOW_TURNS`` Q&A pairs
            are forwarded to the model so it can answer follow-up questions correctly.

    Returns:
        The generated answer string.

    Raises:
        RuntimeError: If the backend is unreachable or generation fails.
    """
    backend = settings.llm_backend.lower()

    if backend == "groq":
        selected_model = model or settings.groq_model
        return _groq_generate(query, chunks, selected_model, temperature, chat_history)
    elif backend == "ollama":
        selected_model = model or settings.ollama_model
        return _ollama_generate(query, chunks, selected_model, temperature, chat_history)
    else:
        raise RuntimeError(
            f"Unknown LLM_BACKEND '{backend}'. Set to 'groq' or 'ollama' in your .env file."
        )


def stream_answer(
    query: str,
    chunks: list[Chunk],
    model: str | None = None,
    temperature: float = 0.1,
    chat_history: list[dict] | None = None,
):
    """
    Stream an answer token-by-token for the given query and context.

    Routes to Groq or Ollama based on settings.llm_backend. Designed for
    use with Streamlit's st.write_stream().

    Args:
        query: User's question string.
        chunks: List of retrieved Chunk objects to ground the answer.
        model: Optional model name override. Defaults to the active backend's default model.
        temperature: Sampling temperature. Defaults to 0.1.
        chat_history: Optional list of prior session message dicts (role + content) to
            inject as conversation history.  The last ``CONTEXT_WINDOW_TURNS`` Q&A pairs
            are forwarded to the model so it can answer follow-up questions correctly.

    Yields:
        str: Individual token strings from the model response stream.

    Raises:
        RuntimeError: If the backend is unreachable or the stream fails.

    Example:
        # In Streamlit:
        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(query, chunks, chat_history=history))
    """
    backend = settings.llm_backend.lower()

    if backend == "groq":
        selected_model = model or settings.groq_model
        yield from _groq_stream(query, chunks, selected_model, temperature, chat_history)
    elif backend == "ollama":
        selected_model = model or settings.ollama_model
        yield from _ollama_stream(query, chunks, selected_model, temperature, chat_history)
    else:
        raise RuntimeError(
            f"Unknown LLM_BACKEND '{backend}'. Set to 'groq' or 'ollama' in your .env file."
        )

