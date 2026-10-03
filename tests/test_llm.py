"""
Tests for the LLM generation module (generation/llm.py).
"""

import pytest
from unittest.mock import patch, MagicMock
from ingestion.document import Chunk
from generation.llm import (
    SYSTEM_PROMPT,
    format_context,
    build_user_message,
    generate_answer,
    stream_answer,
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


class TestFormatContext:
    def test_empty_chunk_list_returns_placeholder(self):
        result = format_context([])
        assert "No context chunks" in result

    def test_single_chunk_formatting_includes_metadata_and_text(self):
        chunk = make_chunk("Compound interest grows over time.", source="Federal Reserve", section="page_4")
        result = format_context([chunk])
        assert "Federal Reserve" in result
        assert "page_4" in result
        assert "Compound interest grows over time." in result

    def test_multiple_chunks_have_separators(self):
        c1 = make_chunk("First chunk text.", source="SEC", section="page_1")
        c2 = make_chunk("Second chunk text.", source="CFPB", section="page_2")
        result = format_context([c1, c2])
        assert "Chunk [1]" in result
        assert "Chunk [2]" in result
        assert "First chunk text." in result
        assert "Second chunk text." in result


class TestBuildUserMessage:
    def test_user_message_contains_question_and_context(self):
        chunk = make_chunk("Bonds pay fixed interest coupons.")
        query = "What is a bond?"
        message = build_user_message(query, [chunk])
        assert "What is a bond?" in message
        assert "Bonds pay fixed interest coupons." in message
        assert "Context Chunks:" in message


class TestGroqBackendMocked:
    def test_generate_answer_groq_calls_client_correctly(self):
        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = "Stocks represent ownership in a company."
            mock_client.chat.completions.create.return_value = mock_resp

            from generation.llm import _groq_generate
            with patch.object(settings, "groq_api_key", "gsk_test_key"):
                answer = _groq_generate(
                    query="What is a stock?",
                    chunks=[make_chunk("Stocks are equity instruments.")],
                    model="qwen/qwen3.8-27b",
                    temperature=0.1,
                )

            assert answer == "Stocks represent ownership in a company."
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_answer_groq_raises_runtime_error_on_failure(self):
        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

            from generation.llm import _groq_generate
            with patch.object(settings, "groq_api_key", "gsk_test_key"):
                with pytest.raises(RuntimeError, match="Groq generation failed"):
                    _groq_generate("Test", [], model="qwen/qwen3.8-27b", temperature=0.1)

    def test_groq_raises_runtime_error_when_no_api_key(self):
        from generation.llm import _groq_generate
        with patch.object(settings, "groq_api_key", ""):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
                _groq_generate("Test", [], model="qwen/qwen3.8-27b", temperature=0.1)

    def test_stream_answer_groq_yields_tokens(self):
        def make_stream_chunk(content):
            c = MagicMock()
            c.choices[0].delta.content = content
            return c

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_groq_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = iter([
                make_stream_chunk("A "),
                make_stream_chunk("bond "),
                make_stream_chunk("pays interest."),
            ])

            from generation.llm import _groq_stream
            with patch.object(settings, "groq_api_key", "gsk_test_key"):
                tokens = list(_groq_stream(
                    query="What is a bond?",
                    chunks=[make_chunk("Bonds pay fixed interest.")],
                    model="qwen/qwen3.8-27b",
                    temperature=0.1,
                ))

            assert tokens == ["A ", "bond ", "pays interest."]


class TestOllamaBackendMocked:
    def test_generate_answer_ollama_invokes_client_correctly(self):
        with patch("ollama.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_resp = MagicMock()
            mock_resp.message.content = "A mutual fund pools money from multiple investors."
            mock_client_instance.chat.return_value = mock_resp

            from generation.llm import _ollama_generate
            answer = _ollama_generate(
                query="What is a mutual fund?",
                chunks=[make_chunk("A mutual fund is an investment company.")],
                model="qwen2.5:7b",
                temperature=0.1,
            )

            assert answer == "A mutual fund pools money from multiple investors."
            mock_client_instance.chat.assert_called_once()

    def test_generate_answer_ollama_raises_on_failure(self):
        with patch("ollama.Client") as mock_client_cls:
            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            mock_client_instance.chat.side_effect = ConnectionError("Connection refused")

            from generation.llm import _ollama_generate
            with pytest.raises(RuntimeError, match="Ollama generation failed"):
                _ollama_generate("Test question", [], model="qwen2.5:7b", temperature=0.1)


class TestBackendRouting:
    def test_generate_answer_routes_to_groq_when_backend_is_groq(self):
        with patch("generation.llm._groq_generate", return_value="groq answer") as mock_groq, \
             patch.object(settings, "llm_backend", "groq"):
            result = generate_answer("Test?", [])
            mock_groq.assert_called_once()
            assert result == "groq answer"

    def test_generate_answer_routes_to_ollama_when_backend_is_ollama(self):
        with patch("generation.llm._ollama_generate", return_value="ollama answer") as mock_ollama, \
             patch.object(settings, "llm_backend", "ollama"):
            result = generate_answer("Test?", [])
            mock_ollama.assert_called_once()
            assert result == "ollama answer"

    def test_generate_answer_raises_on_unknown_backend(self):
        with patch.object(settings, "llm_backend", "anthropic"):
            with pytest.raises(RuntimeError, match="Unknown LLM_BACKEND"):
                generate_answer("Test?", [])

    def test_stream_answer_routes_to_groq(self):
        with patch("generation.llm._groq_stream", return_value=iter(["tok1", "tok2"])), \
             patch.object(settings, "llm_backend", "groq"):
            tokens = list(stream_answer("Test?", []))
            assert tokens == ["tok1", "tok2"]

    def test_stream_answer_routes_to_ollama(self):
        with patch("generation.llm._ollama_stream", return_value=iter(["tok1", "tok2"])), \
             patch.object(settings, "llm_backend", "ollama"):
            tokens = list(stream_answer("Test?", []))
            assert tokens == ["tok1", "tok2"]


class TestGroqLive:
    def test_live_groq_generate_returns_grounded_answer(self):
        if not settings.groq_api_key or not settings.groq_api_key.startswith("gsk_"):
            pytest.skip("GROQ_API_KEY not set")

        from generation.llm import _groq_generate

        chunk = make_chunk(
            text="Emergency funds should typically cover 3 to 6 months of living expenses.",
            source="CFPB",
            section="page_5",
        )
        query = "How many months of expenses should an emergency fund cover?"
        answer = _groq_generate(
            query=query,
            chunks=[chunk],
            model="qwen/qwen3.8-27b",
            temperature=0.0,
        )

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "3" in answer or "three" in answer.lower()

    def test_live_groq_stream_yields_tokens(self):
        if not settings.groq_api_key or not settings.groq_api_key.startswith("gsk_"):
            pytest.skip("GROQ_API_KEY not set")

        from generation.llm import _groq_stream

        chunk = make_chunk(
            text="A 401(k) is an employer-sponsored retirement savings plan with tax advantages.",
            source="SEC",
            section="page_2",
        )
        query = "What is a 401(k)?"
        tokens = list(_groq_stream(
            query=query,
            chunks=[chunk],
            model="qwen/qwen3.8-27b",
            temperature=0.0,
        ))

        assert len(tokens) > 0
        full_answer = "".join(tokens)
        assert len(full_answer) > 10
        assert "401" in full_answer or "retirement" in full_answer.lower()
