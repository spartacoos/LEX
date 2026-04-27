# tests/test_lex.py
"""
LEX integration + evaluation tests.

Two test groups, separated by pytest marker:

  * (unmarked)        — Smoke tests. Fast, no LLM required.
  * @pytest.mark.services — Requires live Qdrant, Redis, and LLM server.
  * @pytest.mark.eval — Full evaluation against the gold standard. Minutes.

Run:
    uv run pytest -m "not eval and not services"   # CI-safe smoke tests
    uv run pytest -m services                      # service health checks
    uv run pytest -m eval                          # full evaluation suite
    uv run pytest -m eval -v                       # verbose, per-question progress
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from lex.commands import AnswerCmd, RetrieveCmd, RetrieveFilter
from lex.config import get_settings
from lex.generation import build_answer_deps, handle_answer
from lex.retrieval import build_retrieve_deps, handle_retrieve

from conftest import record_eval_row


GOLD_PATH = Path(__file__).parent / "gold_standard.json"


# ===========================================================================
# Section 1: Service health checks
#
# Require live Qdrant, Redis, and LLM. Skipped in CI.
# Run locally with: uv run pytest -m services
# ===========================================================================

@pytest.mark.services
@pytest.mark.asyncio
async def test_services_reachable():
    """Qdrant, Redis, and the LLM endpoint all respond."""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{settings.qdrant.url}/")
        assert r.status_code == 200, "Qdrant not responding"

        r = await client.get(f"{settings.llm.base_url}/models")
        assert r.status_code == 200, "LLM server not responding"

    import redis.asyncio as aioredis
    r = aioredis.from_url(settings.redis.url)
    assert await r.ping()
    await r.aclose()


# ===========================================================================
# Section 2: Smoke tests
#
# Fast, no LLM required. Only need Qdrant running + a directive ingested.
# These run in CI.
# ===========================================================================

@pytest.mark.asyncio
async def test_retrieve_end_to_end():
    """A known-good query returns non-empty, sensibly-scored chunks."""
    settings = get_settings()
    deps = build_retrieve_deps(settings)

    try:
        cmd = RetrieveCmd(
            query="very high capacity network",
            top_k=3,
            filters=RetrieveFilter(language="en", celex_id="32018L1972"),
        )
        result = await handle_retrieve(cmd, deps)
    finally:
        await deps.qdrant.close()

    assert len(result.chunks) == 3
    # Rerank scores should be monotonically non-increasing.
    scores = [c.rerank_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    # Top chunk should mention VHCN-related concepts.
    top_text = result.chunks[0].chunk.text.lower()
    assert any(kw in top_text for kw in ("very high capacity", "vhcn", "optical"))


# ===========================================================================
# Section 3: Evaluation
#
# Uses DeepEval to score each answer against the gold standard.
# Requires live LLM + judge model.
# Run with: uv run pytest -m eval
# ===========================================================================

def _load_gold() -> list[dict]:
    """Read gold_standard.json, return the list of question dicts."""
    data = json.loads(GOLD_PATH.read_text())
    return data["questions"]


def _make_deepeval_model(settings):
    """
    Build a DeepEvalBaseLLM pointing at the judge LLM.

    Uses settings.eval_judge.base_url when set, else falls back to
    the RAG LLM config. Override via env vars:

        export LEX_EVAL_JUDGE__BASE_URL=http://localhost:8081/v1
        export LEX_EVAL_JUDGE__MODEL=gemma-4-31b
    """
    from deepeval.models.base_model import DeepEvalBaseLLM
    from openai import OpenAI

    judge_cfg = settings.eval_judge
    base_url = judge_cfg.base_url or settings.llm.base_url
    model = judge_cfg.model or settings.llm.model
    api_key = judge_cfg.api_key if judge_cfg.base_url else settings.llm.api_key
    timeout = judge_cfg.timeout_s
    temperature = judge_cfg.temperature
    max_tokens = judge_cfg.max_tokens

    class Judge(DeepEvalBaseLLM):
        def __init__(self):
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            self._model = model

        def load_model(self):
            return self._client

        def generate(self, prompt: str) -> str:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self._model

    return Judge()


def _citation_correctness(
    expected_refs: list[str],
    actual_refs: set[str],
) -> float:
    """
    Fraction of expected article references that appear in actual citations.

    Empty expected_refs scores 1.0 if the model also cited nothing (correct
    refusal), 0.0 otherwise.
    """
    if not expected_refs:
        return 1.0 if not actual_refs else 0.0
    if not actual_refs:
        return 0.0
    hits = sum(1 for r in expected_refs if r in actual_refs)
    return hits / len(expected_refs)


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("question", _load_gold(), ids=lambda q: q["id"])
async def test_eval_question(question: dict):
    """
    Run one gold-standard question end-to-end and score it.

    Each question becomes one pytest case (visible in -v output),
    and each produces one row in the CSV/Markdown report.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    settings = get_settings()
    deps = await build_answer_deps(settings)
    judge = _make_deepeval_model(settings)

    try:
        cmd = AnswerCmd(
            query=question["question"],
            filters=RetrieveFilter(language="en", celex_id="32018L1972"),
            stream=False,
        )
        result = await handle_answer(cmd, deps)

        expected = set(question.get("expected_article_refs", []))
        actual = {c.article for c in result.citations if c.article}
        citation_score = _citation_correctness(
            question["expected_article_refs"], actual,
        )

        test_case = LLMTestCase(
            input=question["question"],
            actual_output=result.answer,
            expected_output=question.get("expected_answer_summary", ""),
            retrieval_context=[rc.chunk.text for rc in result.chunks],
        )

        metrics_out: dict[str, float] = {}
        for name, metric_cls in [
            ("context_precision", ContextualPrecisionMetric),
            ("context_recall",    ContextualRecallMetric),
            ("faithfulness",      FaithfulnessMetric),
            ("answer_relevancy",  AnswerRelevancyMetric),
        ]:
            metric = metric_cls(
                model=judge,
                threshold=0.0,
                include_reason=False,
            )
            metric.measure(test_case)
            metrics_out[name] = float(metric.score)

        record_eval_row({
            "id": question["id"],
            "category": question["category"],
            "question": question["question"],
            **metrics_out,
            "citation_correctness": citation_score,
            "retrieved_articles": ",".join(sorted(actual)),
            "expected_articles": ",".join(question["expected_article_refs"]),
            "answer_chars": len(result.answer),
            "citations_count": len(result.citations),
        })
    finally:
        await deps.retrieve.qdrant.close()

    assert result.answer, "handler produced empty answer"
