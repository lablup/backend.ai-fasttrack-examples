"""Shared task-runner: argparse scaffolding, input seeding, manifest, exit codes.

Tasks call `run_task(name, handler, seed_stages=...)` and stay standalone
scripts — no task imports another task.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from docs_rag.settings import load_layered_dotenv
from pipeline.io import SourceSpec, SourcesConfig, TaskResult
from pipeline.paths import (
    SOURCES_YAML,
    ensure_dirs,
    manifest_path,
    resolve_input_root,
    resolve_output_root,
    resolve_vfroot,
)
from pipeline.preflight import log_environment, on_fasttrack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def make_parser(task_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=task_name)
    p.add_argument(
        "--project",
        default="all",
        help="Source name to process, or 'all' (default).",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Pipeline data root. Defaults to PIPELINE_OUTPUT_ROOT, then ./pipeline/data.",
    )
    p.add_argument(
        "--sources",
        default=str(SOURCES_YAML),
        help="Path to sources.yaml (default: pipeline/config/sources.yaml).",
    )
    return p


def select_sources(cfg: SourcesConfig, project: str) -> List[SourceSpec]:
    if project == "all":
        return cfg.sources
    matches = [s for s in cfg.sources if s.name == project]
    if not matches:
        raise SystemExit(
            f"Unknown project '{project}'. Valid: " + ", ".join(cfg.names())
        )
    return matches


def seed_from_input(
    output_root: Path,
    input_root: Path,
    stages: Optional[Sequence[str]] = None,
) -> List[str]:
    """Copy prior-stage dirs from the FastTrack input mount into the output root.

    FastTrack chains tasks: each writes /pipeline/outputs and the dependent task
    sees that as /pipeline/input1. To keep the output cumulative (node 05 needs
    03_indices, which node 03 wrote two hops back), seed the output root from
    the input root before the handler runs. Merge-copy so the handler's own
    stage write overwrites cleanly.

    `stages` restricts the copy. It matters twice over: without it every node
    drags the cloned repos and the converted markdown through the output mount —
    tens of gigabytes of pure I/O per run for data nothing downstream reads —
    and an empty tuple is how the head node declares that it consumes nothing.

    No-op when input_root is None, missing, or identical to output_root.

    Callers depend on that no-op for head-node detection, so do not "helpfully"
    create the input root here. Replacing the `exists()` check with a mkdir, or
    materialising the output mirror eagerly, would make the head node seed a
    stale tree and silently re-break the chain.
    """
    if input_root is None or input_root == output_root or not input_root.exists():
        return []
    seeded: List[str] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for child in sorted(input_root.iterdir()):
        if not child.is_dir():
            continue
        if stages is not None and child.name not in stages:
            continue
        shutil.copytree(child, output_root / child.name, dirs_exist_ok=True)
        seeded.append(child.name)
    return seeded


def append_manifest(data_root: Path, result: TaskResult) -> None:
    path = manifest_path(data_root)
    rows = []
    if path.exists():
        try:
            rows = json.loads(path.read_text())
        except json.JSONDecodeError:
            rows = []
    rows.append(result.model_dump(mode="json"))
    path.write_text(json.dumps(rows, indent=2))


def run_task(
    task_name: str,
    handler: Callable[[argparse.Namespace, SourcesConfig, Path], dict],
    seed_stages: Optional[Sequence[str]] = None,
) -> None:
    """Standard task entrypoint.

    `handler(args, sources_config, data_root)` returns a dict of details for the
    manifest and raises on failure.
    """
    load_layered_dotenv()
    args = make_parser(task_name).parse_args()
    data_root = resolve_output_root(args.data_root)
    log = logging.getLogger(task_name)

    # Head-ness is structural, never positional. Do NOT gate this on
    # BACKENDAI_PIPELINE_JOB_INDEX == "1" — that value is not a reliable DAG
    # position, and a cluster that reports "1" for a mid-chain task makes every
    # node skip seeding and die on an empty upstream stage.
    #
    # The head declares `seed_stages=()`: it consumes nothing. That is a claim
    # this repo makes about its own DAG, so it holds no matter what the
    # scheduler mounts — where relying on "FastTrack gives the head no
    # /pipeline/input1" would leave a stale or empty mount to be copied forward.
    input_root = resolve_input_root()
    expects_upstream = seed_stages is None or bool(seed_stages)

    log_environment(task_name, data_root, input_root, resolve_vfroot())

    seeded = seed_from_input(data_root, input_root, seed_stages)
    if seeded:
        log.info("seeded from input_root=%s: %s", input_root, ", ".join(seeded))
    elif expects_upstream and on_fasttrack():
        # This node reads an upstream stage and got nothing. On FastTrack that
        # is always wrong — an unset PIPELINE_INPUT_ROOT, a missing dependency
        # edge, or an upstream task that wrote nothing. Say so here rather than
        # letting it surface later as a confusing empty-directory error.
        # Locally there is no input mount by design, so this stays quiet.
        log.warning(
            "nothing seeded from input_root=%s (wanted: %s). Expect the next "
            "require_stage() to fail: check this node's dependency edge and "
            "that the upstream task actually wrote its stage.",
            input_root, ", ".join(seed_stages) if seed_stages else "every stage",
        )

    ensure_dirs(data_root)
    log.info("data_root=%s project=%s", data_root, args.project)

    cfg = SourcesConfig.load(Path(args.sources))
    started = datetime.now(timezone.utc)

    try:
        details = handler(args, cfg, data_root) or {}
    # SystemExit is a BaseException, so it would slip past `Exception` and leave
    # no failure row — the preflight checks all raise it, which would silently
    # hollow out the audit trail.
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - tasks must report all failures
        finished = datetime.now(timezone.utc)
        append_manifest(
            data_root,
            TaskResult(
                task=task_name,
                project=args.project,
                started_at=started,
                finished_at=finished,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        if isinstance(exc, SystemExit):
            # A failed precondition is an actionable message, not a crash — a
            # traceback would just bury it.
            log.error("Task failed: %s", exc)
        else:
            log.exception("Task failed")
        sys.exit(1)

    finished = datetime.now(timezone.utc)
    append_manifest(
        data_root,
        TaskResult(
            task=task_name,
            project=args.project,
            started_at=started,
            finished_at=finished,
            success=True,
            details=details,
        ),
    )
    log.info("Task succeeded in %.1fs: %s", (finished - started).total_seconds(), details)
