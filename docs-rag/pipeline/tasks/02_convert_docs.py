#!/usr/bin/env python3
"""Node 02 — turn each cloned repo's documentation into flat markdown.

In:   <data-root>/01_repos/<name>/
Out:  <data-root>/02_docs_md/<name>/

Which files are taken is decided entirely by the include/exclude globs in
sources.yaml. `.rst` is converted with pandoc; `.md` is copied through.

    python pipeline/tasks/02_convert_docs.py --project all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docs_rag.convert import convert_files, reset_output
from pipeline.io import SourcesConfig
from pipeline.paths import docs_md_dir, repos_dir
from pipeline.preflight import require_binary, require_stage
from pipeline.runner import run_task, select_sources

log = logging.getLogger("02_convert_docs")


async def convert_all(cfg: SourcesConfig, args: argparse.Namespace, data_root: Path) -> dict:
    summary: dict = {}
    for spec in select_sources(cfg, args.project):
        repo_root = repos_dir(data_root, spec.name)
        if not repo_root.is_dir():
            raise SystemExit(
                f"{spec.name}: no clone at {repo_root}. Did the clone-docs node run?"
            )

        files = spec.select_files(repo_root)
        if not files:
            # A source that matches nothing is a broken glob, not an empty repo.
            # Failing here beats an empty index that only surfaces at query time.
            raise SystemExit(
                f"{spec.name}: include globs {spec.include} matched no files under "
                f"{repo_root}. Check the patterns in sources.yaml against the repo layout."
            )

        out_root = docs_md_dir(data_root, spec.name)
        reset_output(out_root)
        written, failures = await convert_files(files, repo_root, out_root)

        for failure in failures:
            log.warning("%s: %s", spec.name, failure)
        log.info(
            "%s: %d selected -> %d written (%d failed)",
            spec.name, len(files), written, len(failures),
        )
        if written == 0:
            raise SystemExit(f"{spec.name}: every file failed to convert; see the warnings above.")

        summary[spec.name] = f"written={written} failed={len(failures)}"
    return summary


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_binary("pandoc")
    require_stage(data_root, "01_repos", args.project)
    summary = asyncio.run(convert_all(cfg, args, data_root))
    return {"sources": summary, "count": len(summary)}


if __name__ == "__main__":
    run_task("02_convert_docs", handler, seed_stages=("01_repos",))
