"""
Tests for eval/llm_judge.py — LLM-as-Judge scoring module.

Why do we mock the OpenAI API in these tests?
    1. Speed & Cost: Making real API calls to OpenAI on every test run would be slow
       and cost money. Mocking runs in milliseconds for $0.00.
    2. Determinism: We want predictable responses so tests never fail randomly due to
       network blips or model variability.
    3. Failure Simulation: We can simulate edge cases (like malformed JSON, markdown fences,
       or unexpected text) that are difficult to trigger reliably with live models.

What these tests verify:
    1. Score Normalization: Converting 1-5 integer ratings into 0.0-1.0 floats.
    2. JSON Parsing & Resilience: Handling clean JSON, markdown-wrapped JSON, and syntax errors.
    3. Public API Contract: Ensuring score_relevance() and score_faithfulness() return
       the exact expected dictionary structure: {'score', 'raw_score', 'reasoning'}.
"""

from unittest.mock import patch, MagicMock

from eval.llm_judge import (
    _call_judge,
    _normalize_score,
    score_relevance,
    score_faithfulness,
)


def _make_mock_response(content: str) -> MagicMock:
    """Helper to construct a mock object mimicking the OpenAI ChatCompletion response structure.

    Real OpenAI response looks like:
        response.choices[0].message.content = "..."

    We build a MagicMock with that exact nested attribute hierarchy so the code
    under test can access it without knowing it's a fake.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ────────────────────────────────────────────────────────────────────
# 1. Score Normalization Tests
# ────────────────────────────────────────────────────────────────────

def test_normalize_score_boundaries():
    """Verify that boundary ratings (1, 3, 5) map accurately to (0.0, 0.5, 1.0).

    Math formula: (score - 1) / 4.0
      - 1 is the worst rating -> (1 - 1) / 4 = 0.00 (0%)
      - 3 is the neutral rating -> (3 - 1) / 4 = 0.50 (50%)
      - 5 is the best rating -> (5 - 1) / 4 = 1.00 (100%)
    """
    assert _normalize_score(1) == 0.0   # Minimum possible score
    assert _normalize_score(3) == 0.5   # Neutral / midpoint score
    assert _normalize_score(5) == 1.0   # Maximum possible score


def test_normalize_score_all_values():
    """Verify all 5 possible discrete integer scores produce exact expected normalized floats.

    This ensures no rounding errors or off-by-one mistakes across the full rating scale.
    """
    expected_mapping = {
        1: 0.0,    # (1-1)/4 = 0/4
        2: 0.25,   # (2-1)/4 = 1/4
        3: 0.5,    # (3-1)/4 = 2/4
        4: 0.75,   # (4-1)/4 = 3/4
        5: 1.0,    # (5-1)/4 = 4/4
    }
    for raw_score, expected_normalized in expected_mapping.items():
        assert _normalize_score(raw_score) == expected_normalized


# ────────────────────────────────────────────────────────────────────
# 2. JSON Parsing & Error Handling Tests
# ────────────────────────────────────────────────────────────────────

@patch("eval.llm_judge._get_client")
def test_parse_valid_json(mock_get_client):
    """Test scenario: OpenAI returns a perfectly formatted JSON string.

    Given:
        OpenAI outputs: {"score": 4, "reasoning": "Good"}
    Expected:
        _call_judge parses this into a Python dictionary with integer score:
        {"score": 4, "reasoning": "Good"}
    """
    # Setup: Create a fake client and configure chat.completions.create to return clean JSON
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response(
        '{"score": 4, "reasoning": "Good"}'
    )

    # Execute
    result = _call_judge("prompt", "gpt-4o-mini")

    # Assert
    assert result == {"score": 4, "reasoning": "Good"}


@patch("eval.llm_judge._get_client")
def test_parse_json_in_code_fence(mock_get_client):
    """Test scenario: OpenAI wraps the JSON in markdown fences (e.g. ```json ... ```).

    LLMs frequently wrap code and JSON blocks in markdown fences even when instructed
    to output raw JSON. The regex in _call_judge should strip the fences and extract the JSON.
    """
    # Setup: Return markdown-wrapped JSON
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response(
        '```json\n{"score": 5, "reasoning": "Perfect"}\n```'
    )

    # Execute
    result = _call_judge("prompt", "gpt-4o-mini")

    # Assert: Fences should be stripped and payload parsed cleanly
    assert result == {"score": 5, "reasoning": "Perfect"}


@patch("eval.llm_judge._get_client")
def test_parse_failure_returns_default(mock_get_client):
    """Test scenario: OpenAI returns completely invalid / non-JSON text.

    If the LLM outputs conversational text instead of JSON:
      1. json.loads() fails with JSONDecodeError
      2. The function retries up to 3 times
      3. Upon exhausting retries, it gracefully falls back to a neutral default:
         {'score': 3, 'reasoning': 'Failed to parse JSON response'}
    This prevents the entire evaluation run from crashing on a single bad response.
    """
    # Setup: Simulate non-JSON conversational text
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "I am an AI, I cannot score this."
    )

    # Execute
    result = _call_judge("prompt", "gpt-4o-mini")

    # Assert: Should return neutral score 3 with explanatory message
    assert result["score"] == 3
    assert "Failed to parse" in result["reasoning"]


# ────────────────────────────────────────────────────────────────────
# 3. Public API Contract Tests
# ────────────────────────────────────────────────────────────────────

@patch("eval.llm_judge._get_client")
def test_score_relevance_returns_expected_keys(mock_get_client):
    """Verify score_relevance() formats inputs, queries the model, and returns the expected contract.

    Return dictionary contract:
      - 'score': float (normalized 0.0 - 1.0)
      - 'raw_score': int (original 1 - 5 from the model)
      - 'reasoning': str (model's justification text)
    """
    # Setup: Mock returns raw score of 4
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response(
        '{"score": 4, "reasoning": "Good"}'
    )

    # Execute
    result = score_relevance(
        question="What is an emergency fund?",
        answer="A savings buffer for unexpected expenses.",
        reference_answer="Funds set aside for unforeseen financial emergencies.",
        model="gpt-4o-mini"
    )

    # Assert: Check dictionary structure and values
    assert "score" in result
    assert "raw_score" in result
    assert "reasoning" in result
    assert result["raw_score"] == 4
    assert result["score"] == 0.75  # (4 - 1) / 4 = 0.75
    assert result["reasoning"] == "Good"


@patch("eval.llm_judge._get_client")
def test_score_faithfulness_returns_expected_keys(mock_get_client):
    """Verify score_faithfulness() formats context chunks, queries the model, and returns the contract.

    Return dictionary contract:
      - 'score': float (normalized 0.0 - 1.0)
      - 'raw_score': int (original 1 - 5 from the model)
      - 'reasoning': str (model's justification text)
    """
    # Setup: Mock returns raw score of 4
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_mock_response(
        '{"score": 4, "reasoning": "Good"}'
    )

    # Execute
    result = score_faithfulness(
        question="What is an emergency fund?",
        answer="A savings buffer for unexpected expenses.",
        context_texts=["Context Chunk 1: Keep 3-6 months of expenses in savings."],
        model="gpt-4o-mini"
    )

    # Assert: Check dictionary structure and values
    assert "score" in result
    assert "raw_score" in result
    assert "reasoning" in result
    assert result["raw_score"] == 4
    assert result["score"] == 0.75  # (4 - 1) / 4 = 0.75
    assert result["reasoning"] == "Good"
