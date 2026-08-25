#!/usr/bin/env python3
"""Node 06 — copy the results into the persistent pipeline folder.

In:   <data-root>/{03_indices,99_state}/
Out:  $PIPELINE_VFROOT/{03_indices,99_state}/

/pipeline/vfroot is the only location that outlives the run. On the Backend.AI
side it appears as the auto-created vFolder `<pipeline-name>-<id>/.pipeline/`,
so this is where an operator goes to retrieve the indices and the reports.

    PIPELINE_VFROOT=./pipeline/vfroot python pipeline/tasks/06_publish.py
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.io import SourcesConfig
from pipeline.paths import STAGE_INDICES, STAGE_STATE, resolve_vfroot
from pipeline.preflight import on_fasttrack, require_stage
from pipeline.runner import run_task

log = logging.getLogger("06_publish")

PUBLISH_STAGES = (STAGE_INDICES, STAGE_STATE)


def handler(args: argparse.Namespace, cfg: SourcesConfig, data_root: Path) -> dict:
    require_stage(data_root, STAGE_INDICES, args.project)

    vfroot = resolve_vfroot()
    if vfroot is None:
        if on_fasttrack():
            # Reporting success having published nothing is the worst outcome:
            # the DAG goes green and the artifacts are gone with the container.
            raise SystemExit(
                "PIPELINE_VFROOT is not set, but this is a FastTrack run. "
                "/pipeline/vfroot is auto-mounted — set PIPELINE_VFROOT=/pipeline/vfroot "
                "in your .env so this node knows where to publish."
            )
        log.info("PIPELINE_VFROOT unset — nothing to publish (local run)")
        return {"published": False, "reason": "PIPELINE_VFROOT unset"}

    vfroot.mkdir(parents=True, exist_ok=True)
    if not os.access(vfroot, os.W_OK):
        raise SystemExit(f"{vfroot} is not writable — cannot publish")

    copied = []
    for stage in PUBLISH_STAGES:
        source = data_root / stage
        # ensure_dirs() creates every stage, so "exists" is not evidence of
        # content — check for actual entries.
        if not source.is_dir() or not any(source.iterdir()):
            log.warning("skipping %s: empty", stage)
            continue
        shutil.copytree(source, vfroot / stage, dirs_exist_ok=True)
        copied.append(stage)
        log.info("published %s -> %s", stage, vfroot / stage)

    if STAGE_INDICES not in copied:
        raise RuntimeError(f"published nothing useful: {STAGE_INDICES} was empty")

    return {"published": True, "vfroot": str(vfroot), "stages": copied}


if __name__ == "__main__":
    run_task("06_publish", handler, seed_stages=(STAGE_INDICES, STAGE_STATE))
