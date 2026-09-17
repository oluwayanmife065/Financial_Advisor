"""
LLM Generation Module — interfaces with local Ollama instance for RAG answers.

This module handles:
  1. Grounding system prompt definition
  2. Context formatting from retrieved Chunks
  3. Prompt construction with strict anti-hallucination instructions
  4. Ollama chat completion API call
"""

import ollama
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


def generate_answer(
    query: str,
    chunks: list[Chunk],
    model: str | None = None,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer to the query given retrieved context chunks using Ollama.

    Args:
        query: User's question string.
        chunks: List of retrieved Chunk objects to ground the answer.
        model: Optional model name override. Defaults to settings.ollama_model.
        temperature: Sampling temperature. Defaults to 0.1 for consistent, grounded output.

    Returns:
        The generated answer string.

    Raises:
        RuntimeError: If Ollama is unreachable or generation fails.
    """
    selected_model = model or settings.ollama_model
    user_content = build_user_message(query, chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=selected_model,
            messages=messages,
            options={"temperature": temperature},
        )

        # Handle both dict-like and object responses across client versions
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content.strip()
        elif isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip()
        else:
            return str(response).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate answer with Ollama model '{selected_model}' "
            f"at {settings.ollama_base_url}: {exc}"
        ) from exc

