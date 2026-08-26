#!/usr/bin/env python3
"""Node 03 — chunk the markdown and build one FAISS + BM25 index per source.

In:   <data-root>/02_docs_md/<name>/
Out:  <data-root>/03_indices/<name>/{index.faiss, index.pkl, bm25.pkl}

This is the only node that spends money: it calls the embedding API once per
chunk batch. Everything downstream reads the indices it writes.

    python pipeline/tasks/03_build_indices.py --project all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag.indexer import DocumentIndexer
from docs_rag.settings import get_settings
from pipeline.io import SourcesConfig
from pipeline.paths import docs_md_dir, indices_dir
from pipeline.preflight import require_env, require_stage
from pipeline.runner import run_task, select_sources

log = logging.getLogger("03_build_indices")


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_env("OPENAI_API_KEY")
    require_stage(data_root, "02_docs_md", args.project)

    settings = get_settings()
    names = [s.name for s in select_sources(cfg, args.project)]

    indexer = DocumentIndexer(
        docs_root=docs_md_dir(data_root),
        indices_root=indices_dir(data_root),
        settings=settings,
    )

    project_docs = indexer.collect_documents(names)
    missing = [n for n in names if not project_docs.get(n)]
    if missing:
        raise SystemExit(
            f"no chunks collected for: {', '.join(missing)}. The convert-docs node "
            "produced no usable markdown for them."
        )

    # Remove the previous index before writing, so a source that shrank does not
    # keep serving chunks for pages that no longer exist.
    for name in names:
        target = indices_dir(data_root, name)
        if target.exists():
            shutil.rmtree(target)

    built = asyncio.run(indexer.build(project_docs))

    log.info(
        "built %d index/indices with %s (chunk_size=%d overlap=%d)",
        len(built), settings.embedding_model, settings.chunk_size, settings.chunk_overlap,
    )
    return {
        "chunks": built,
        "total_chunks": sum(built.values()),
        "embedding_model": settings.embedding_model,
    }


if __name__ == "__main__":
    run_task("03_build_indices", handler, seed_stages=("02_docs_md",))
