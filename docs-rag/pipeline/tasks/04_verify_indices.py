#!/usr/bin/env python3
"""Node 04 — prove each index actually answers a question about its own corpus.

In:   <data-root>/03_indices/  + the verify_query on each source
Out:  <data-root>/99_state/verify_report.json

A dual canary per source: the top semantic hit must be closer than the L2
threshold AND the lexical arm must return at least one hit. Either alone can
pass on a broken index — a FAISS index built from empty files still returns its
nearest neighbour, and BM25 matches a common word in anything.

Exits non-zero if any source fails, so the DAG stops before serving a bad index.

    python pipeline/tasks/04_verify_indices.py --project all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag.retriever import Retriever
from docs_rag.settings import get_settings
from pipeline.io import SourcesConfig, VerifyProjectResult, VerifyReport
from pipeline.paths import indices_dir, verify_report_path
from pipeline.preflight import require_env, require_stage
from pipeline.runner import run_task, select_sources

log = logging.getLogger("04_verify_indices")


async def verify_all(cfg: SourcesConfig, args: argparse.Namespace, data_root: Path):
    settings = get_settings()
    retriever = Retriever(indices_dir(data_root), settings)
    retriever.load()

    results = {}
    for spec in select_sources(cfg, args.project):
        if not spec.verify_query:
            results[spec.name] = VerifyProjectResult(
                project=spec.name, query="", passed=False,
                error="no verify_query declared in sources.yaml",
            )
            continue

        if spec.name not in retriever.projects:
            results[spec.name] = VerifyProjectResult(
                project=spec.name, query=spec.verify_query, passed=False,
                error=f"no index loaded from {indices_dir(data_root, spec.name)}",
            )
            continue

        semantic = await retriever.search_semantic([spec.name], spec.verify_query, k=1)
        lexical = await retriever.search_lexical([spec.name], spec.verify_query, k=5)

        hits = semantic.get(spec.name, [])
        lexical_hits = len(lexical.get(spec.name, []))
        top_l2 = hits[0]["similarity_score"] if hits else None
        top_source = hits[0]["metadata"].get("relative_path") if hits else None

        passed = (
            top_l2 is not None
            and top_l2 < settings.max_l2
            and lexical_hits > 0
        )
        results[spec.name] = VerifyProjectResult(
            project=spec.name,
            query=spec.verify_query,
            passed=passed,
            top_l2=top_l2,
            top_source=top_source,
            lexical_hits=lexical_hits,
        )

    return settings, results


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_env("OPENAI_API_KEY")
    require_stage(data_root, "03_indices", args.project)

    settings, results = asyncio.run(verify_all(cfg, args, data_root))

    report = VerifyReport(
        generated_at=datetime.now(timezone.utc),
        l2_threshold=settings.max_l2,
        results=results,
    )
    path = verify_report_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    failed = [name for name, r in results.items() if not r.passed]
    for name in failed:
        r = results[name]
        log.error(
            "%s FAILED: query=%r top_l2=%s lexical_hits=%s error=%s",
            name, r.query, r.top_l2, r.lexical_hits, r.error,
        )
    for name, r in results.items():
        if r.passed:
            log.info(
                "%s ok: top_l2=%.3f lexical_hits=%d source=%s",
                name, r.top_l2, r.lexical_hits, r.top_source,
            )

    if failed:
        raise RuntimeError(f"index verification failed for: {', '.join(failed)}")

    return {"passed": len(results), "failed": 0, "report": str(path)}


if __name__ == "__main__":
    run_task("04_verify_indices", handler, seed_stages=("03_indices", "99_state"))
