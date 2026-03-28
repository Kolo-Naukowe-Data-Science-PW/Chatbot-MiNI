"""
LLM-as-a-judge: score two chatbot answers on 1–5 scales and pick the better variant via OpenRouter.
"""

import json
import logging
import os
import re
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class JudgeScores(BaseModel):
    """Per-variant star ratings (1–5) for usefulness, accuracy, and conciseness."""

    usefulness: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)


class JudgeResult(BaseModel):
    """Full judge output: scores for both variants, preferred variant, and short rationale."""

    variant_a: JudgeScores
    variant_b: JudgeScores
    better_variant: Literal["A", "B", "tie"]
    reason: str


def build_judge_prompt(
    query: str,
    answer_a: str,
    answer_b: str,
    language: str = "pl",
) -> list[dict[str, str]]:
    """
    Build chat messages for the judge model (system + user) with query, both answers, and rubric.

    Returns:
        Messages ready for OpenAI-compatible chat completion API.
    """
    system_prompt = (
        "You are an impartial evaluator for chatbot responses. "
        "Rate each answer on a 1-5 star scale for exactly these criteria: "
        "Usefulness, Accuracy, Conciseness. "
        "Use integers only. "
        "Then pick the better variant: A, B, or tie. "
        "Return strictly valid JSON with keys: "
        "variant_a, variant_b, better_variant, reason. "
        "Each variant object must have keys usefulness, accuracy, conciseness. "
        "Do not include markdown or extra text."
    )
    user_prompt = (
        f"Language: {language}\n"
        f"User query:\n{query}\n\n"
        f"Variant A:\n{answer_a}\n\n"
        f"Variant B:\n{answer_b}\n\n"
        "Scoring rubric:\n"
        "- Usefulness: how helpful and actionable for the user query.\n"
        "  - 1: Not helpful or misses the user need.\n"
        "  - 3: Partially helpful but missing important details.\n"
        "  - 5: Fully helpful and directly actionable.\n"
        "- Accuracy: factual correctness and internal consistency.\n"
        "  - 1: Contains major factual errors or hallucinations.\n"
        "  - 3: Mostly correct but includes notable uncertainty/inaccuracy.\n"
        "  - 5: Factually correct and consistent with no evident errors.\n"
        "- Conciseness: brevity without losing essential information.\n"
        "  - 1: Verbose/rambling or hard to parse.\n"
        "  - 3: Reasonably concise but can be shorter/cleaner.\n"
        "  - 5: Clear, compact, and complete.\n\n"
        "Use full 1-5 scale when appropriate, not only 1/3/5.\n"
        "Output JSON now."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object(text: str) -> str:
    """
    Extract a single JSON object from raw model text (handles extra prose around JSON).

    Raises:
        ValueError: If no `{...}` block is found.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response")
    return match.group(0)


def parse_judge_response(raw_content: str) -> JudgeResult:
    """
    Parse and validate judge JSON into a ``JudgeResult`` (Pydantic).
    """
    json_blob = _extract_json_object(raw_content)
    data = json.loads(json_blob)
    return JudgeResult(**data)


def judge_pair(
    query: str,
    answer_a: str,
    answer_b: str,
    *,
    judge_model: str,
    language: str = "pl",
    temperature: float = 0.0,
    max_tokens: int = 400,
    max_retries: int = 2,
) -> JudgeResult:
    """
    Call OpenRouter with the judge prompt; parse JSON response. Retries on parse/API errors.

    Args:
        query: Original user question.
        answer_a / answer_b: Bot answers to compare.
        judge_model: OpenRouter model id for judging.
        language: Hint for the judge (e.g. ``pl``).
        temperature / max_tokens: Passed to the completion API.
        max_retries: Extra attempts after failures.

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is missing or all retries fail.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    messages = build_judge_prompt(query, answer_a, answer_b, language=language)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw_content = completion.choices[0].message.content or ""
            return parse_judge_response(raw_content)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Judge parse/validation failed on attempt %s/%s: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
        except Exception as exc:  # pragma: no cover - network/provider errors
            last_error = exc
            logger.warning(
                "Judge call failed on attempt %s/%s: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )

    raise RuntimeError(f"Judge evaluation failed after retries: {last_error}")
