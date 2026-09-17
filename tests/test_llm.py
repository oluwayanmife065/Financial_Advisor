"""
Tests for the LLM generation module (generation/llm.py).

Test structure:
  - TestFormatContext  → unit tests for context chunk formatting
  - TestBuildUserMessage → unit tests for prompt assembly
  - TestGenerateAnswerMocked → unit tests mocking Ollama client (no server needed)
  - TestGenerateAnswerLive → integration tests with live local Ollama server (skipped if offline)
"""

import pytest
from unittest.mock import patch, MagicMock
from ingestion.document import Chunk
from generation.llm import (
    SYSTEM_PROMPT,
    format_context,
    build_user_message,
    generate_answer,
)
from config import settings


def make_chunk(text: str, source: str = "SEC", section: str = "page_1") -> Chunk:
    """Helper to construct a Chunk for testing."""
    return Chunk(
        text=text,
        source=source,
        section=section,
        url="file:///mock.pdf",
        doc_id="mock_doc_1",
        chunk_index=0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestFormatContext
# Pure unit tests checking formatting of retrieved chunks.
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatContext:

    def test_empty_chunk_list_returns_placeholder(self):
        """
        When no chunks are retrieved (e.g., zero retrieval hits),
        format_context should return a clean placeholder message rather than crashing.
        """
        result = format_context([])
        assert "No context chunks" in result

    def test_single_chunk_formatting_includes_metadata_and_text(self):
        """
        A formatted chunk must display its source, section, and text body.
        This provides clear attribution to the LLM during generation.
        """
        chunk = make_chunk("Compound interest grows over time.", source="Federal Reserve", section="page_4")
        result = format_context([chunk])

        assert "Federal Reserve" in result
        assert "page_4" in result
        assert "Compound interest grows over time." in result

    def test_multiple_chunks_have_separators(self):
        """
        When multiple chunks are provided, each should be distinct and numbered.
        """
        c1 = make_chunk("First chunk text.", source="SEC", section="page_1")
        c2 = make_chunk("Second chunk text.", source="CFPB", section="page_2")
        result = format_context([c1, c2])

        assert "Chunk [1]" in result
        assert "Chunk [2]" in result
        assert "First chunk text." in result
        assert "Second chunk text." in result


# ══════════════════════════════════════════════════════════════════════════════
# TestBuildUserMessage
# Tests assembly of the user prompt incorporating question and context.
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildUserMessage:

    def test_user_message_contains_question_and_context(self):
        """
        The constructed user message must contain both the query and the chunk text.
        """
        chunk = make_chunk("Bonds pay fixed interest coupons.")
        query = "What is a bond?"
        message = build_user_message(query, [chunk])

        assert "What is a bond?" in message
        assert "Bonds pay fixed interest coupons." in message
        assert "Context Chunks:" in message


# ══════════════════════════════════════════════════════════════════════════════
# TestGenerateAnswerMocked
# Unit tests mocking the Ollama client. Fast, runs offline.
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateAnswerMocked:

    @patch("generation.llm.ollama.Client")
    def test_generate_answer_invokes_client_with_correct_structure(self, mock_client_cls):
        """
        Verifies that generate_answer calls ollama.Client.chat with the expected
        system prompt, user prompt, and model name.
        """
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        mock_resp = MagicMock()
        mock_resp.message.content = "A mutual fund pools money from multiple investors."
        mock_instance.chat.return_value = mock_resp

        chunks = [make_chunk("A mutual fund is an investment company that pools money.")]
        answer = generate_answer("What is a mutual fund?", chunks, model="qwen2.5:7b")

        assert answer == "A mutual fund pools money from multiple investors."
        mock_instance.chat.assert_called_once()

        # Check call arguments
        call_kwargs = mock_instance.chat.call_args[1]
        assert call_kwargs["model"] == "qwen2.5:7b"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
        assert call_kwargs["messages"][1]["role"] == "user"
        assert "What is a mutual fund?" in call_kwargs["messages"][1]["content"]

    @patch("generation.llm.ollama.Client")
    def test_generate_answer_raises_runtime_error_on_failure(self, mock_client_cls):
        """
        If Ollama fails with an exception (e.g. connection refused),
        generate_answer should wrap it in a clean, descriptive RuntimeError.
        """
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_instance.chat.side_effect = ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Failed to generate answer"):
            generate_answer("Test question", [])


# ══════════════════════════════════════════════════════════════════════════════
# TestGenerateAnswerLive
# Live integration test against the local Ollama daemon.
# Skipped automatically if Ollama is not running.
# ══════════════════════════════════════════════════════════════════════════════

def is_ollama_online() -> bool:
    """Check if the local Ollama server is responsive."""
    import urllib.request
    try:
        req = urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=1.0)
        return req.status == 200
    except Exception:
        return False


class TestGenerateAnswerLive:

    @pytest.mark.skipif(not is_ollama_online(), reason="Local Ollama server not online")
    def test_live_generation_with_context(self):
        """
        Integration test verifying end-to-end prompt generation with local Ollama.
        """
        chunk = make_chunk(
            text="Emergency funds should typically cover 3 to 6 months of living expenses.",
            source="CFPB",
            section="page_5",
        )
        query = "How many months of expenses should an emergency fund cover?"
        answer = generate_answer(query, [chunk], temperature=0.0)

        assert isinstance(answer, str)
        assert len(answer) > 0
        # The grounded answer should reference 3 to 6 months
        assert "3" in answer or "three" in answer.lower()

