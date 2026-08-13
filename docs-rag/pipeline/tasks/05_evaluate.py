#!/usr/bin/env python3
"""Node 05 — score end-to-end answer quality against a fixed question set.

In:   <data-root>/03_indices/ + pipeline/config/eval_samples.json
Out:  <data-root>/99_state/eval_report.json

For each question: retrieve, answer, then score the answer with an LLM judge
(relevance, groundedness, completeness, usability) and SemScore against the
reference answer.

Node 04 asks "does retrieval work at all"; this asks "is the answer any good".
A run that fails here is worth looking at, not worth stopping the DAG for, so
one bad question is recorded and skipped rather than raised.

    python pipeline/tasks/05_evaluate.py --project all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag.evaluate import aggregate, compute_overall, judge_answer, make_judge, semscore
from docs_rag.rag import RAGChat
from docs_rag.retriever import Retriever, format_context
from docs_rag.settings import get_settings
from pipeline.io import EvalQuestionResult, EvalReport, SourcesConfig
from pipeline.paths import EVAL_SAMPLES_JSON, eval_report_path, indices_dir
from pipeline.preflight import require_env, require_stage
from pipeline.runner import run_task, select_sources

log = logging.getLogger("05_evaluate")


def load_samples(path: Path, limit: int, projects: List[str]) -> List[dict]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    # Only score questions whose corpus is actually in this run — a --project
    # run would otherwise be graded on documents it never indexed.
    samples = [s for s in samples if s.get("project") in projects]
    return samples[:limit] if limit > 0 else samples


async def score_one(
    sample: dict, retriever: Retriever, judge, settings, projects: List[str]
) -> EvalQuestionResult:
    question = sample["question"]
    chunks = await retriever.retrieve(question, projects, mode="hybrid")
    context = format_context(chunks, settings.max_chars_per_chunk)

    top_l2 = min(
        (c["similarity_score"] for c in chunks if "similarity_score" in c),
        default=None,
    )

    chat = RAGChat(retriever, settings)
    answer = "".join([token async for token in chat.answer_with_context(question, context)])

    scores = await judge_answer(judge, question, sample["answer"], context, answer)
    metrics: Dict[str, float] = scores.as_dict()
    metrics["semscore"] = await semscore(answer, sample["answer"], settings)

    return EvalQuestionResult(
        question=question,
        project=sample.get("project", ""),
        source=sample.get("source", ""),
        top_l2=top_l2,
        retrieved_chunks=len(chunks),
        response_chars=len(answer),
        metrics=metrics,
        overall=compute_overall(metrics),
    )


async def evaluate_all(cfg: SourcesConfig, args: argparse.Namespace, data_root: Path):
    settings = get_settings()
    projects = [s.name for s in select_sources(cfg, args.project)]

    retriever = Retriever(indices_dir(data_root), settings)
    retriever.load()
    projects = [p for p in projects if p in retriever.projects]
    if not projects:
        raise SystemExit("no indices loaded — nothing to evaluate")

    samples = load_samples(EVAL_SAMPLES_JSON, settings.eval_sample_size, projects)
    if not samples:
        raise SystemExit(
            f"no eval questions for projects {projects} in {EVAL_SAMPLES_JSON}"
        )

    judge = make_judge(settings)
    results: List[EvalQuestionResult] = []

    for i, sample in enumerate(samples, start=1):
        log.info("[%d/%d] %s", i, len(samples), sample["question"][:80])
        try:
            results.append(await score_one(sample, retriever, judge, settings, projects))
        except Exception as exc:  # noqa: BLE001 - one bad question must not lose the run
            log.warning("question %d failed: %s: %s", i, type(exc).__name__, exc)
            results.append(
                EvalQuestionResult(
                    question=sample["question"],
                    project=sample.get("project", ""),
                    source=sample.get("source", ""),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return settings, projects, results


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_env("OPENAI_API_KEY")
    require_stage(data_root, "03_indices", args.project)

    settings, projects, results = asyncio.run(evaluate_all(cfg, args, data_root))

    scored = [r for r in results if r.error is None]
    summary = aggregate([r.metrics for r in scored])
    if scored:
        summary["overall"] = sum(r.overall for r in scored) / len(scored)
        l2s = [r.top_l2 for r in scored if r.top_l2 is not None]
        if l2s:
            summary["mean_top_l2"] = sum(l2s) / len(l2s)

    report = EvalReport(
        generated_at=datetime.now(timezone.utc),
        rag_model=settings.llm_model,
        judge_model=settings.eval_judge_model,
        sample_size=len(results),
        projects=projects,
        aggregate=summary,
        results=results,
    )
    path = eval_report_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    log.info("scored %d/%d questions: %s", len(scored), len(results), summary)
    return {
        "evaluated": len(scored),
        "failed": len(results) - len(scored),
        "overall": round(summary.get("overall", 0.0), 4),
        "report": str(path),
    }


if __name__ == "__main__":
    run_task("05_evaluate", handler, seed_stages=("03_indices", "99_state"))
