"""Answer-quality scoring: an LLM judge plus an embedding-similarity metric.

Two independent signals, because each fails differently. The judge reads the
answer against the retrieved context and catches unsupported claims, but it is
another language model and can be talked into agreeing. SemScore is a cosine
similarity against the reference answer — mechanical, cheap, and blind to
correctness, but impossible to argue with.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from docs_rag.embeddings import make_embeddings
from docs_rag.settings import Settings

log = logging.getLogger("evaluate")

# How each metric contributes to the headline number. Groundedness is weighted
# highest because a fluent, relevant, ungrounded answer is the failure mode this
# whole system exists to prevent.
OVERALL_WEIGHTS: Dict[str, float] = {
    "relevance": 0.20,
    "groundedness": 0.35,
    "completeness": 0.20,
    "usability": 0.10,
    "semscore": 0.15,
}

JUDGE_PROMPT = """You are grading an answer produced by a documentation question-answering system.

QUESTION
{question}

REFERENCE ANSWER (one correct answer, not necessarily the only one)
{expected}

DOCUMENTATION EXCERPTS THE SYSTEM WAS GIVEN
{context}

THE SYSTEM'S ANSWER
{actual}

Score each criterion from 0.0 to 1.0.

relevance      Does the answer address the question that was asked?
groundedness   Is every factual claim supported by the excerpts above? Penalise
               heavily any invented command, path, flag, port or version. An
               answer that correctly says the documentation does not cover the
               question scores HIGH here, not low.
completeness   Does it cover the substance of the reference answer? A different
               but equally valid technical approach is fine — do not require a
               word-for-word match.
usability      Could a reader act on this? Concrete steps and cited sources
               score well; vague gestures at the docs do not.

Return a brief reason for each score.
"""


class JudgeScores(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    usability: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {
            "relevance": self.relevance,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
            "usability": self.usability,
        }


def make_judge(settings: Settings) -> ChatOpenAI:
    """The grading model. Temperature 0 so a re-run reproduces the scores."""
    kwargs = dict(settings.llm_kwargs())
    kwargs["model"] = settings.eval_judge_model
    kwargs["temperature"] = 0.0
    return ChatOpenAI(timeout=180, max_retries=2, **kwargs)


async def judge_answer(
    judge: ChatOpenAI,
    question: str,
    expected: str,
    context: str,
    actual: str,
    max_context_chars: int = 12000,
) -> JudgeScores:
    """Score one answer. Structured output — never parsed out of free text."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        context=context[:max_context_chars],
        actual=actual,
    )
    structured = judge.with_structured_output(JudgeScores)
    return await structured.ainvoke(prompt)


async def semscore(generated: str, expected: str, settings: Settings) -> float:
    """Cosine similarity between the two answers' embeddings, clamped to [0, 1]."""
    if not generated.strip() or not expected.strip():
        return 0.0
    embeddings = make_embeddings(settings)
    vectors = await embeddings.aembed_documents([generated, expected])
    a, b = np.array(vectors[0]), np.array(vectors[1])
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denominator)))


def compute_overall(metrics: Dict[str, float]) -> float:
    """Weighted mean over whichever metrics are present.

    Renormalises against the weights actually used, so a run without SemScore
    is still scored on a 0-1 scale rather than silently capped at 0.85.
    """
    total_weight = sum(OVERALL_WEIGHTS[name] for name in metrics if name in OVERALL_WEIGHTS)
    if total_weight == 0.0:
        return 0.0
    weighted = sum(
        value * OVERALL_WEIGHTS[name]
        for name, value in metrics.items()
        if name in OVERALL_WEIGHTS
    )
    return weighted / total_weight


def aggregate(per_question: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric across questions, preserving first-seen order."""
    if not per_question:
        return {}
    names: List[str] = []
    for metrics in per_question:
        for name in metrics:
            if name not in names:
                names.append(name)
    return {
        name: sum(m.get(name, 0.0) for m in per_question) / len(per_question)
        for name in names
    }
