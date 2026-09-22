"""
LLM-as-Judge — uses OpenAI GPT to score generated RAG answers.

Why an LLM judge?
    Retrieval metrics (precision, recall) tell us if we FOUND the right chunks,
    but they don't tell us if the ANSWER is any good. The LLM could retrieve
    perfect chunks and still produce a bad answer (hallucination, irrelevance).

    An LLM judge reads the question, the generated answer, and scores it on
    two dimensions:
      1. Relevance  — Does the answer actually address what was asked?
      2. Faithfulness — Is the answer grounded in the retrieved context,
                        or did the LLM make things up?

Why OpenAI instead of Ollama for judging?
    Using the same local model (qwen2.5:7b) to judge its own answers creates
    self-evaluation bias — it tends to rate itself highly. GPT-4o-mini is a
    stronger, independent judge that gives more reliable scores. The API cost
    is minimal (pennies for 25 questions).

Scoring scale:
    The judge outputs a 1-5 integer score, which we normalize to 0.0-1.0:
        1 → 0.00    (terrible)
        2 → 0.25    (poor)
        3 → 0.50    (mediocre)
        4 → 0.75    (good)
        5 → 1.00    (excellent)

KPI targets (from PROJECT_BRIEF.md):
    Answer Relevance  ≥ 0.80
    Faithfulness       ≥ 0.85
"""

import json
import re

from openai import OpenAI
from config import settings


# ────────────────────────────────────────────────────────────────────
# Prompt templates
# ────────────────────────────────────────────────────────────────────
# These prompts ask the LLM to output a JSON object with a score (1-5)
# and reasoning. The double braces {{ }} are Python f-string escapes
# that produce literal { } in the output (so the JSON example renders
# correctly after .format() fills in the actual variables).

RELEVANCE_PROMPT = """You are an evaluation judge assessing answer quality.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {answer}

Rate the Generated Answer's relevance to the Question on a scale of 1-5:
1 = Completely irrelevant or wrong
2 = Partially relevant but mostly off-topic
3 = Somewhat relevant but missing key information
4 = Mostly relevant and accurate
5 = Highly relevant, accurate, and comprehensive

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

FAITHFULNESS_PROMPT = """You are an evaluation judge assessing answer faithfulness.

Context:
{context}

Question: {question}
Generated Answer: {answer}

Rate how faithfully the Generated Answer is grounded in the provided Context on a scale of 1-5:
1 = Completely fabricated, no basis in context
2 = Mostly fabricated with minor context elements
3 = Partially grounded but contains unsupported claims
4 = Mostly grounded with minor extrapolation
5 = Fully grounded in the provided context

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    """Create an OpenAI client using the API key from settings.

    The key is loaded from .env via pydantic-settings (never hardcoded).

    Returns:
        Configured OpenAI client.

    Raises:
        ValueError: If no API key is configured.
    """
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file. "
            "See .env.example for the format."
        )
    return OpenAI(api_key=settings.openai_api_key)


def _call_judge(prompt: str, model: str) -> dict:
    """Send a scoring prompt to OpenAI and parse the JSON response.

    This is the low-level function that both score_relevance() and
    score_faithfulness() call. It handles:
      1. Sending the prompt to OpenAI with temperature=0 for consistency
      2. Parsing the JSON response (including code-fenced JSON)
      3. Retrying up to 3 times if the LLM returns unparseable output

    Args:
        prompt: The fully formatted scoring prompt.
        model: OpenAI model name (e.g. "gpt-4o-mini").

    Returns:
        Dict with 'score' (int 1-5) and 'reasoning' (str).
        On complete failure, returns score=3 (uncertain) with error explanation.
    """
    client = _get_client()

    # Retry loop — LLMs sometimes return malformed JSON on first attempt
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Zero temp for reproducible scoring
            )

            # Extract the text content from OpenAI's response
            content = response.choices[0].message.content

            # Some models wrap JSON in markdown code fences like ```json ... ```
            # This regex strips those fences to get the raw JSON
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

            # Parse the JSON and validate it has the required fields
            result = json.loads(content)

            if "score" in result and "reasoning" in result:
                result["score"] = int(result["score"])
                return result

        except Exception:
            # JSON parsing failed, API error, etc. — try again
            continue

    # All 3 attempts failed — return a neutral "uncertain" score
    # Score=3 normalizes to 0.5, which won't artificially inflate or deflate metrics
    return {"score": 3, "reasoning": "Failed to parse JSON response"}


def _normalize_score(raw_score: int) -> float:
    """Convert 1-5 integer score to 0.0-1.0 float.

    Mapping:  1→0.00  2→0.25  3→0.50  4→0.75  5→1.00
    Formula:  (score - 1) / 4

    The clamp (max/min) protects against out-of-range values
    if the LLM returns something unexpected like 0 or 6.

    Args:
        raw_score: Integer score from the LLM judge (expected 1-5).

    Returns:
        Normalized float between 0.0 and 1.0.
    """
    return max(0.0, min(1.0, (raw_score - 1) / 4.0))


# ────────────────────────────────────────────────────────────────────
# Public scoring functions
# ────────────────────────────────────────────────────────────────────

def score_relevance(question: str, answer: str, reference_answer: str, model: str | None = None) -> dict:
    """Score how well the generated answer addresses the question.

    Compares the generated answer against both the question and a
    hand-written reference answer from the golden set. The judge sees
    all three and decides how relevant/accurate the generation is.

    Args:
        question: The user's question (from golden set).
        answer: The RAG system's generated answer.
        reference_answer: The hand-written correct answer (from golden set).
        model: Optional model override for the judge. Defaults to settings.eval_judge_model.

    Returns:
        Dict with:
          - 'score': float 0.0-1.0 (normalized)
          - 'raw_score': int 1-5 (from the judge)
          - 'reasoning': str (judge's explanation)
    """
    judge_model = model or settings.eval_judge_model

    prompt = RELEVANCE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        answer=answer,
    )

    result = _call_judge(prompt, judge_model)
    raw_score = result["score"]

    return {
        "score": _normalize_score(raw_score),
        "raw_score": raw_score,
        "reasoning": result["reasoning"],
    }


def score_faithfulness(question: str, answer: str, context_texts: list[str], model: str | None = None) -> dict:
    """Score how well the generated answer is grounded in the retrieved context.

    This catches hallucination — if the LLM made up facts that aren't in
    the context chunks, faithfulness will be low even if the answer sounds good.

    Args:
        question: The user's question.
        answer: The RAG system's generated answer.
        context_texts: List of chunk text strings that were fed to the LLM.
        model: Optional model override for the judge. Defaults to settings.eval_judge_model.

    Returns:
        Dict with:
          - 'score': float 0.0-1.0 (normalized)
          - 'raw_score': int 1-5 (from the judge)
          - 'reasoning': str (judge's explanation)
    """
    judge_model = model or settings.eval_judge_model

    # Format context chunks with numbered labels so the judge can reference them
    context = "\n".join(f"[{i+1}] {text}" for i, text in enumerate(context_texts))

    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )

    result = _call_judge(prompt, judge_model)
    raw_score = result["score"]

    return {
        "score": _normalize_score(raw_score),
        "raw_score": raw_score,
        "reasoning": result["reasoning"],
    }
